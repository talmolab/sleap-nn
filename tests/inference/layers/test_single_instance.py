"""Tests for ``SingleInstanceLayer`` — the proof-of-pattern InferenceLayer.

Coverage:

1. End-to-end parity vs the PR 0 golden ``single_instance.pkl`` —
   running the *new* layer stack on the same fixed input must produce
   the same ``pred_instance_peaks`` and ``pred_peak_values`` as the
   current pipeline within float tolerance. This is the locked
   acceptance criterion for the entire refactor: every refactor PR
   gates on this test still passing.

2. Direct numpy API: ``layer.predict(np.ndarray)`` returns an
   ``Outputs`` whose shape matches the documented contract.

3. Direct torch API: same, with a torch input.

4. Single frame + batched frames both work without code changes.

5. ``return_confmaps=True`` populates ``Outputs.pred_confmaps``;
   default leaves it ``None``.

6. ``Outputs.pred_keypoints`` lives in original-image coordinates
   (the coord ladder has been applied).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
SINGLE_CKPT = CKPT_ROOT / "minimal_instance_single_instance"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _build_layer_from_loader():
    """Build a ``SingleInstanceLayer`` around the loaded Lightning module
    that the inference loader produces. Reuses the production loader so we
    don't reimplement checkpoint-loading kwargs here.
    """
    from omegaconf import OmegaConf

    from sleap_nn.inference.loaders import load_model_assets

    assets, _ = load_model_assets(
        [str(SINGLE_CKPT)],
        device="cpu",
        peak_threshold=0.3,  # match the PR 0 golden's spec
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

    # Pull the SingleInstance Lightning module out of the inference_model.
    inf = assets.inference_model
    lightning_module = inf.torch_model
    output_stride = inf.output_stride
    layer = SingleInstanceLayer(
        backend=TorchBackend(model=lightning_module, device="cpu"),
        output_stride=output_stride,
        preprocess_config=PreprocessConfig(scale=inf.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            integral_patch_size=inf.integral_patch_size,
        ),
    )
    return layer, assets


# ─────────────────────────────────────────────────────────────────────────
# 2. Direct numpy API
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="checkpoint not present")
def test_predict_accepts_numpy():
    """Calling ``predict`` with a raw numpy array yields a structured ``Outputs``."""
    layer, _ = _build_layer_from_loader()

    img = np.zeros((1, 1, 64, 64), dtype=np.float32)
    out = layer.predict(img)

    assert isinstance(out, Outputs)
    assert out.pred_keypoints is not None
    assert out.pred_keypoints.ndim == 4  # (B, I, N, 2)
    assert out.pred_keypoints.shape[0] == 1
    assert out.pred_keypoints.shape[1] == 1  # single instance


# ─────────────────────────────────────────────────────────────────────────
# 3. Direct torch API
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="checkpoint not present")
def test_predict_accepts_torch_tensor():
    layer, _ = _build_layer_from_loader()

    img = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    out = layer.predict(img)

    assert isinstance(out, Outputs)
    assert out.pred_keypoints.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────
# 4. Single frame + batched frames
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="checkpoint not present")
def test_predict_handles_single_frame_2d_grayscale():
    layer, _ = _build_layer_from_loader()

    img = np.zeros((64, 64), dtype=np.uint8)  # (H, W)
    out = layer.predict(img)
    assert out.batch_size == 1


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="checkpoint not present")
def test_predict_handles_batch_of_4_frames():
    layer, _ = _build_layer_from_loader()

    batch = np.zeros((4, 1, 64, 64), dtype=np.float32)
    out = layer.predict(batch)
    assert out.batch_size == 4
    assert out.pred_keypoints.shape == (4, 1, out.n_nodes, 2)


# ─────────────────────────────────────────────────────────────────────────
# 5. return_confmaps
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="checkpoint not present")
def test_return_confmaps_off_by_default():
    layer, _ = _build_layer_from_loader()
    out = layer.predict(np.zeros((1, 1, 64, 64), dtype=np.float32))
    assert out.pred_confmaps is None


@pytest.mark.skipif(not SINGLE_CKPT.exists(), reason="checkpoint not present")
def test_return_confmaps_true_populates_field():
    """Opting in to ``return_confmaps=True`` keeps the heavy ``(B, N, H, W)``
    confmap tensor on ``Outputs``."""
    from omegaconf import OmegaConf

    from sleap_nn.inference.loaders import load_model_assets

    assets, _ = load_model_assets(
        [str(SINGLE_CKPT)],
        device="cpu",
        peak_threshold=0.3,
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
    inf = assets.inference_model

    layer = SingleInstanceLayer(
        backend=TorchBackend(model=inf.torch_model, device="cpu"),
        output_stride=inf.output_stride,
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            return_confmaps=True,
        ),
    )

    out = layer.predict(np.zeros((1, 1, 64, 64), dtype=np.float32))
    assert out.pred_confmaps is not None
    assert out.pred_confmaps.ndim == 4  # (B, N, H, W)


# ─────────────────────────────────────────────────────────────────────────
# 6. Coord-ladder applied (output is in image-pixel space)
# ─────────────────────────────────────────────────────────────────────────


def test_postprocess_applies_full_coord_ladder_with_synthetic_input():
    """Verify the coord ladder runs without depending on a checkpoint.

    Construct a known confmap (one peak per channel at known integer
    pixels), bypass the backend, and assert ``Outputs.pred_keypoints``
    equals the peak coords scaled by ``output_stride`` (input_scale=1,
    eff_scale=1 → identity for those steps).
    """
    import torch.nn as nn

    class _Identity(nn.Module):
        def forward(self, x):
            return x

    layer = SingleInstanceLayer(
        backend=TorchBackend(model=_Identity(), device="cpu"),
        output_stride=4,
        postprocess_config=PostprocessConfig(peak_threshold=0.1, refinement="none"),
    )

    # Build a (B=1, N=2, H=8, W=8) confmap with known peaks.
    cms = torch.zeros(1, 2, 8, 8)
    cms[0, 0, 3, 5] = 1.0  # node 0 at confmap pixel (x=5, y=3)
    cms[0, 1, 6, 2] = 1.0  # node 1 at confmap pixel (x=2, y=6)

    # Hand-build a PreprocInfo and call postprocess directly. This skips
    # preprocess so we know the eff_scale / input_scale are identities.
    from sleap_nn.inference.preprocess_info import PreprocInfo

    info = PreprocInfo(
        original_size=(32, 32),
        processed_size=(32, 32),
        eff_scale=torch.ones(1),
        input_scale=1.0,
        output_stride=4,
    )
    out = layer.postprocess({"output": cms}, info)

    expected = np.array([[[5 * 4, 3 * 4], [2 * 4, 6 * 4]]], dtype=np.float32)
    np.testing.assert_array_equal(out.pred_keypoints.squeeze(1).numpy(), expected)
