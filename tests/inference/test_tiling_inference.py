"""Tests for the Phase-A tiled single-instance inference path.

The parity tests validate the **stitching infrastructure** — they do NOT need a
tiling-trained model. We build a plain ``SingleInstanceLayer`` from the
single-instance checkpoint, then compare ``TiledLayer(inner, ...).predict(frame)``
against the whole-frame ``inner.predict(frame)`` on the *same* model:

* a single tile that covers the whole frame must reproduce the whole-frame
  keypoints exactly, and
* a real multi-tile grid (with overlap) must reproduce them within a few pixels.

The remaining tests cover the sizematcher-bypass gate, the ``tile_batch_size``
forward bound, the baked-backend refusal, and the geometry parity check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.data.tiling import generate_tile_grid
from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
from sleap_nn.inference.layers.tiled import TiledLayer
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import (
    _build_tiled_layer,
    _resolve_tiling_cfg,
    _select_layer,
)
from sleap_nn.training.lightning_modules import SingleInstanceLightningModule

DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"
SMALL_ROBOT_SLP = DATA_ROOT / "small_robot_minimal.slp"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _build_inner(
    ckpt_dir,
    *,
    return_confmaps: bool = False,
    peak_threshold: float = 0.2,
    sizematcher: bool = False,
) -> SingleInstanceLayer:
    """Build a plain ``SingleInstanceLayer`` from the single-instance ckpt.

    Sizematcher is off by default so the whole-frame and tiled paths process
    the frame at the *same* native (input-scaled) resolution — the only
    difference between them is then the tiling itself.
    """
    cfg = OmegaConf.load(f"{ckpt_dir}/training_config.yaml")
    module = SingleInstanceLightningModule.load_from_checkpoint(
        f"{ckpt_dir}/best.ckpt",
        backbone_config=cfg.model_config.backbone_config,
        head_configs=cfg.model_config.head_configs,
        model_type="single_instance",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )
    pp_kwargs = dict(scale=cfg.data_config.preprocessing.scale)
    if sizematcher:
        pp_kwargs.update(
            max_height=cfg.data_config.preprocessing.max_height,
            max_width=cfg.data_config.preprocessing.max_width,
        )
    return SingleInstanceLayer(
        backend=TorchBackend(model=module.to("cpu"), device="cpu"),
        output_stride=cfg.model_config.head_configs.single_instance.confmaps.output_stride,
        max_stride=cfg.model_config.backbone_config.unet.max_stride,
        preprocess_config=PreprocessConfig(**pp_kwargs),
        postprocess_config=PostprocessConfig(
            peak_threshold=peak_threshold,
            refinement="integral",
            return_confmaps=return_confmaps,
        ),
    )


def _load_frame():
    """Load the small-robot frame as a contiguous ``(H, W, C)`` uint8 array."""
    labels = sio.load_slp(str(SMALL_ROBOT_SLP))
    return np.ascontiguousarray(labels[0].image)  # (320, 560, 3) uint8


class _SpyBackend:
    """A ModelBackend proxy that records the batch size of every forward."""

    def __init__(self, real):
        self._real = real
        self.batch_sizes: list[int] = []

    @property
    def device(self):
        return self._real.device

    @property
    def does_baked_postproc(self):
        return self._real.does_baked_postproc

    def __call__(self, x):
        self.batch_sizes.append(int(x.shape[0]))
        return self._real(x)

    def warmup(self, input_shape):
        return self._real.warmup(input_shape)


class _FakeBakedBackend:
    """Minimal stand-in whose ``does_baked_postproc`` is True (ONNX/TRT)."""

    device = "cpu"
    does_baked_postproc = True

    def __call__(self, x):  # pragma: no cover — never invoked
        raise AssertionError("baked backend should be refused before any forward")

    def warmup(self, input_shape):  # pragma: no cover
        pass


# ─────────────────────────────────────────────────────────────────────────
# 1. Single tile == whole frame (exact)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tile_size,overlap", [(320, 64), (288, 64)])
def test_tiled_equals_whole_when_one_tile(
    minimal_instance_single_instance_ckpt, tile_size, overlap
):
    """A single tile covering the whole frame reproduces the exact keypoints."""
    frame = _load_frame()
    inner_w = _build_inner(minimal_instance_single_instance_ckpt, peak_threshold=0.2)
    inner_t = _build_inner(minimal_instance_single_instance_ckpt, peak_threshold=0.2)

    # Scaled frame is 160x280 -> tile_size >= 280 gives a single tile.
    assert generate_tile_grid((160, 280), tile_size, overlap, 4, 4, 0.25) == [(0, 0)]

    out_w = inner_w.predict(frame)
    out_t = TiledLayer(inner_t, tile_size=tile_size, overlap=overlap).predict(frame)

    kw = out_w.pred_keypoints.numpy()
    kt = out_t.pred_keypoints.numpy()
    assert kt.shape == kw.shape == (1, 1, out_w.n_nodes, 2)
    np.testing.assert_allclose(kt, kw, atol=1e-3, rtol=0, equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────
# 2. Multi-tile stitch ≈ whole frame (within a few px)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tile_size,overlap", [(128, 64), (96, 48), (160, 80)])
def test_tiled_vs_whole_parity(
    minimal_instance_single_instance_ckpt, tile_size, overlap
):
    """A real multi-tile grid reproduces whole-frame keypoints within a few px."""
    frame = _load_frame()
    inner_w = _build_inner(
        minimal_instance_single_instance_ckpt, return_confmaps=True, peak_threshold=0.2
    )
    inner_t = _build_inner(
        minimal_instance_single_instance_ckpt, return_confmaps=True, peak_threshold=0.2
    )

    grid = generate_tile_grid((160, 280), tile_size, overlap, 4, 4, 0.25)
    assert len(grid) >= 2, "expected a real multi-tile grid with seams"

    out_w = inner_w.predict(frame)
    tiled = TiledLayer(inner_t, tile_size=tile_size, overlap=overlap, blend="gaussian")
    out_t = tiled.predict(frame)

    # One pose per frame.
    assert out_t.pred_keypoints.shape == (1, 1, out_w.n_nodes, 2)

    kw = out_w.pred_keypoints.numpy()[0, 0]
    kt = out_t.pred_keypoints.numpy()[0, 0]
    both = ~np.isnan(kw).any(-1) & ~np.isnan(kt).any(-1)
    assert both.any(), "no shared valid keypoints to compare"
    assert np.nanmax(np.abs(kw[both] - kt[both])) <= 2.0

    # Stitched confmaps were requested and attached.
    assert out_t.pred_confmaps is not None
    assert out_t.pred_confmaps.shape == out_w.pred_confmaps.shape


def test_tiled_batched_frames(minimal_instance_single_instance_ckpt):
    """A batch of frames produces one pose per frame with the tiled path."""
    frame = _load_frame()
    batch = np.stack([frame, frame], axis=0)  # (2, H, W, C)
    inner = _build_inner(minimal_instance_single_instance_ckpt, peak_threshold=0.2)
    out = TiledLayer(inner, tile_size=128, overlap=64).predict(batch)
    assert isinstance(out, Outputs)
    assert out.batch_size == 2
    assert out.pred_keypoints.shape[:2] == (2, 1)
    # Both frames are identical -> identical keypoints.
    k = out.pred_keypoints.numpy()
    np.testing.assert_allclose(k[0], k[1], atol=1e-4, equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────
# 3. Sizematcher bypass
# ─────────────────────────────────────────────────────────────────────────


def test_sizematcher_bypass_shape(minimal_instance_single_instance_ckpt):
    """``skip_sizematcher=True`` skips the resize-to-max and returns ones."""
    inner = _build_inner(minimal_instance_single_instance_ckpt, sizematcher=True)
    # (H, W) larger than (max_height, max_width) = (320, 560) so the default
    # path WOULD shrink it; the bypass must not.
    x = torch.zeros(1, 3, 480, 800, dtype=torch.float32)

    scaled_bypass, eff_bypass, orig = inner._apply_full_preprocess(
        x,
        max_stride=inner.output_stride,
        unsqueeze_n_samples=False,
        skip_sizematcher=True,
    )
    # eff_scale all ones and no resize-to-max (only input_scale=0.5 applied).
    assert torch.allclose(eff_bypass, torch.ones_like(eff_bypass))
    assert orig == (480, 800)
    assert scaled_bypass.shape[-2:] == (240, 400)  # 0.5 input scale, no sizematcher

    # Default path (skip_sizematcher=False) is unchanged: it DOES shrink to fit
    # max_height/max_width, so eff_scale != 1 and dims differ from the bypass.
    scaled_def, eff_def, _ = inner._apply_full_preprocess(
        x, max_stride=inner.output_stride, unsqueeze_n_samples=False
    )
    assert not torch.allclose(eff_def, torch.ones_like(eff_def))
    assert scaled_def.shape[-2:] != scaled_bypass.shape[-2:]
    assert scaled_def.shape[-1] <= 560  # matched to max_width * input_scale


# ─────────────────────────────────────────────────────────────────────────
# 4. tile_batch_size bounds the forward batch
# ─────────────────────────────────────────────────────────────────────────


def test_tile_batch_size_bounds_forward(minimal_instance_single_instance_ckpt):
    """No backend forward exceeds ``tile_batch_size`` tiles."""
    frame = _load_frame()
    inner = _build_inner(minimal_instance_single_instance_ckpt, peak_threshold=0.2)
    spy = _SpyBackend(inner.backend)
    inner.backend = spy

    tbs = 3
    tiled = TiledLayer(inner, tile_size=128, overlap=64, tile_batch_size=tbs)
    # (160, 280) with tile 128 / overlap 64 -> a 2x4 = 8-tile grid.
    grid = generate_tile_grid((160, 280), 128, 64, 4, 4, 0.25)
    assert len(grid) == 8

    tiled.predict(frame)
    assert spy.batch_sizes, "backend was never called"
    assert max(spy.batch_sizes) <= tbs
    # 8 tiles at batch 3 -> chunks [3, 3, 2].
    assert spy.batch_sizes == [3, 3, 2]


# ─────────────────────────────────────────────────────────────────────────
# 5. Baked backend is refused
# ─────────────────────────────────────────────────────────────────────────


def test_baked_backend_refused():
    """``_build_tiled_layer`` refuses a baked (ONNX/TRT) backend."""

    class _Inner:
        backend = _FakeBakedBackend()
        output_stride = 4
        max_stride = 4

    tiling = OmegaConf.create(
        {
            "enabled": True,
            "tile_size": 128,
            "overlap": 32,
            "blend": "gaussian",
            "sigma_scale": 0.125,
            "min_overlap_fraction": 0.25,
            "tile_batch_size": None,
            "accumulator_device": "auto",
            "cpu_thresh": 0.4,
        }
    )
    with pytest.raises(NotImplementedError, match="Tiled inference is not supported"):
        _build_tiled_layer(_Inner(), tiling, device="cpu")


# ─────────────────────────────────────────────────────────────────────────
# 6. Geometry parity check + routing
# ─────────────────────────────────────────────────────────────────────────


def test_check_tiling_parity_override_mismatch_raises():
    """A geometry override that diverges from the trained value is a hard error."""
    from sleap_nn.config.utils import check_tiling_parity

    cfg = OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "tiling": {"enabled": True, "tile_size": 128, "overlap": 32}
                }
            }
        }
    )
    # Matching values are fine.
    check_tiling_parity(cfg, tile_size_override=128, overlap_override=32)
    # A divergent override raises.
    with pytest.raises(ValueError, match="tile_size override"):
        check_tiling_parity(cfg, tile_size_override=256)
    with pytest.raises(ValueError, match="overlap override"):
        check_tiling_parity(cfg, overlap_override=64)
    # Disabled -> no-op even with a divergent override.
    cfg.data_config.preprocessing.tiling.enabled = False
    check_tiling_parity(cfg, tile_size_override=999)


def test_resolve_tiling_cfg_default_disabled(minimal_instance_single_instance_ckpt):
    """The stock ckpt config carries no tiling subtree -> resolves to disabled/None."""
    from sleap_nn.inference.loaders import load_model_assets

    loaded, model_types = load_model_assets(
        [str(minimal_instance_single_instance_ckpt)],
        device="cpu",
        preprocess_config=OmegaConf.create(
            {
                "ensure_rgb": None,
                "ensure_grayscale": None,
                "crop_size": None,
                "max_width": None,
                "max_height": None,
                "scale": None,
            }
        ),
    )
    assert model_types == ["single_instance"]
    tiling = _resolve_tiling_cfg(loaded)
    # No tiling key in the legacy config -> None (inert, default whole-frame path).
    assert tiling is None or not tiling.enabled
    layer = _select_layer(loaded, model_types, "cpu")
    assert isinstance(layer, SingleInstanceLayer)


def test_select_layer_routes_to_tiled_when_enabled(
    minimal_instance_single_instance_ckpt,
):
    """When tiling is enabled in the config, ``_select_layer`` builds a TiledLayer."""
    from sleap_nn.inference.loaders import load_model_assets

    loaded, model_types = load_model_assets(
        [str(minimal_instance_single_instance_ckpt)],
        device="cpu",
        preprocess_config=OmegaConf.create(
            {
                "ensure_rgb": None,
                "ensure_grayscale": None,
                "crop_size": None,
                "max_width": None,
                "max_height": None,
                "scale": None,
            }
        ),
    )
    # Inject a valid tiling geometry (max_stride == output_stride == 4).
    OmegaConf.update(
        loaded.confmap_config,
        "data_config.preprocessing.tiling",
        {
            "enabled": True,
            "tile_size": 128,
            "overlap": 32,
            "blend": "gaussian",
            "sigma_scale": 0.125,
            "min_overlap_fraction": 0.25,
            "tile_batch_size": None,
            "accumulator_device": "auto",
            "cpu_thresh": 0.4,
        },
        force_add=True,
    )
    layer = _select_layer(loaded, model_types, "cpu")
    assert isinstance(layer, TiledLayer)
    assert layer.tile_size == 128
    assert layer.overlap == 32
    assert layer.tile_batch_size == 8  # None -> conservative default
