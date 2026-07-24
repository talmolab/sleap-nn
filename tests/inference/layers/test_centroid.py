"""Tests for ``CentroidLayer``.

Coverage:

1. **Parity vs legacy ``CentroidCrop``** — for both the model-driven path
   (``use_gt_centroids=False``) and the GT-centroid path
   (``use_gt_centroids=True``), the new layer's output must match the
   legacy Lightning module's output within the design-doc tolerance
   budget (1e-4 abs / 1e-5 rel).

2. **Direct numpy/torch APIs** — ``CentroidLayer.predict(np.ndarray)``
   and ``predict(torch.Tensor)`` both yield a structured ``Outputs``.

3. **NaN-padding to ``max_instances``** — sub-cap detections produce NaN
   slots; super-cap detections keep top-k by confidence value.

4. **GT path requires ``instances``** — calling ``predict(image)`` with
   ``use_gt_centroids=True`` and no ``instances`` raises ``ValueError``.

5. **``return_confmaps``** — opt-in populates ``Outputs.pred_confmaps``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"
DATA_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "datasets"

PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-5


class _StubBackend:
    """Minimal stub satisfying the ``ModelBackend`` runtime protocol."""

    device = "cpu"
    does_baked_postproc = False

    def __call__(self, x):
        return {}

    def warmup(self, input_shape):
        return None

NEUTRAL_PREPROCESS = OmegaConf.create(
    {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
)


# ─────────────────────────────────────────────────────────────────────────
# Build helpers
# ─────────────────────────────────────────────────────────────────────────


def _build_predictor():
    """Build a topdown Predictor and pull its initialized inference_model."""
    from sleap_nn.inference.predictors import Predictor

    predictor = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
    return predictor


def _build_layer(predictor) -> CentroidLayer:
    """Build a ``CentroidLayer`` around the predictor's centroid module."""
    legacy = predictor.inference_model.centroid_crop
    return CentroidLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        output_stride=legacy.output_stride,
        max_instances=legacy.max_instances,
        max_stride=legacy.max_stride,
        anchor_ind=legacy.anchor_ind,
        use_gt_centroids=False,
        preprocess_config=PreprocessConfig(scale=legacy.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            integral_patch_size=legacy.integral_patch_size,
            max_instances=legacy.max_instances,
        ),
    )


# Per-layer "parity vs legacy CentroidCrop.forward" was removed:
# end-to-end byte-for-byte parity is captured at the top of the stack
# via the PR 0 goldens.


# ─────────────────────────────────────────────────────────────────────────
# 1. Direct numpy / torch APIs
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_predict_accepts_numpy():
    layer = _build_layer(_build_predictor())
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_centroids is not None
    assert out.pred_centroids.shape[0] == 1
    assert out.pred_centroids.shape[2] == 2  # (x, y)


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_predict_accepts_torch_tensor():
    layer = _build_layer(_build_predictor())
    img = torch.zeros(1, 1, 384, 384)
    out = layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_centroids.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────
# 3. NaN-padding to max_instances
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_pred_centroids_padded_to_max_instances():
    layer = _build_layer(_build_predictor())
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = layer.predict(img)
    assert out.pred_centroids.shape == (1, 6, 2)  # max_instances=6
    assert out.pred_centroid_values.shape == (1, 6)
    # On a zero image the model returns no peaks; everything is NaN.
    assert torch.isnan(out.pred_centroids).all()


# ─────────────────────────────────────────────────────────────────────────
# 4. GT path requires `instances`
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_use_gt_centroids_requires_instances():
    """Calling without ``instances`` on the GT path raises ``ValueError``."""
    layer = _build_layer(_build_predictor())
    layer.use_gt_centroids = True
    with pytest.raises(ValueError, match="instances"):
        layer.predict(np.zeros((1, 1, 384, 384), dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────
# 5. return_confmaps
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_return_confmaps_off_by_default():
    layer = _build_layer(_build_predictor())
    out = layer.predict(np.zeros((1, 1, 384, 384), dtype=np.float32))
    assert out.pred_confmaps is None


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_return_confmaps_true_populates_field():
    predictor = _build_predictor()
    legacy = predictor.inference_model.centroid_crop
    layer = CentroidLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        output_stride=legacy.output_stride,
        max_instances=legacy.max_instances,
        max_stride=legacy.max_stride,
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            return_confmaps=True,
            max_instances=legacy.max_instances,
        ),
    )
    out = layer.predict(np.zeros((1, 1, 384, 384), dtype=np.float32))
    assert out.pred_confmaps is not None
    assert out.pred_confmaps.ndim == 4  # (B, N, H, W)


# ─────────────────────────────────────────────────────────────────────────
# 6. Predict-time max_instances override (sleap#2831)
# ─────────────────────────────────────────────────────────────────────────


def test_postprocess_config_max_instances_override():
    """A predict-time max_instances override on postprocess_config is honored.

    Regression test for talmolab/sleap#2831: CentroidLayer.postprocess()
    read ``self.max_instances`` (frozen at __init__) instead of
    ``self.postprocess_config.max_instances`` (set by _postprocess_overrides),
    so predict-time --max_instances had no effect.
    """
    layer = CentroidLayer(
        backend=_StubBackend(),
        output_stride=1,
        max_instances=None,
        postprocess_config=PostprocessConfig(max_instances=None),
    )

    # Simulate 5 raw detections across 1 frame by calling postprocess directly.
    B, n_peaks = 1, 5
    device = torch.device("cpu")
    raw_peaks = torch.rand(n_peaks, 2, device=device)
    raw_vals = torch.rand(n_peaks, device=device)
    sample_inds = torch.zeros(n_peaks, dtype=torch.long, device=device)

    # Monkey-patch _extract_confmaps and find_local_peaks to inject detections.
    import sleap_nn.inference.layers.centroid as centroid_mod
    from sleap_nn.inference.preprocess_info import PreprocInfo

    original_find = centroid_mod.find_local_peaks

    def fake_find(confmaps, threshold, refinement, integral_patch_size):
        channel_inds = torch.zeros(n_peaks, dtype=torch.long)
        return raw_peaks, raw_vals, sample_inds, channel_inds

    centroid_mod.find_local_peaks = fake_find
    layer._extract_confmaps = lambda raw_out: torch.zeros(B, 1, 8, 8)

    info = PreprocInfo(
        original_size=(8, 8),
        processed_size=(8, 8),
        eff_scale=torch.ones(B),
        input_scale=1.0,
        output_stride=1,
    )

    try:
        # Without override: all 5 detections survive.
        out_all = layer.postprocess({}, info)
        assert out_all.pred_centroids.shape[1] == n_peaks

        # Apply predict-time override to cap at 2.
        import attrs

        layer.postprocess_config = attrs.evolve(
            layer.postprocess_config, max_instances=2
        )
        out_capped = layer.postprocess({}, info)
        assert out_capped.pred_centroids.shape[1] == 2
    finally:
        centroid_mod.find_local_peaks = original_find


def test_predict_from_gt_respects_postprocess_config_max_instances():
    """_predict_from_gt honors postprocess_config.max_instances override.

    Same root cause as the postprocess path (sleap#2831): the GT path
    read ``self.max_instances`` instead of ``self.postprocess_config``.
    """
    layer = CentroidLayer(
        backend=_StubBackend(),
        output_stride=1,
        max_instances=None,
        use_gt_centroids=True,
        postprocess_config=PostprocessConfig(max_instances=None),
    )

    # 1 frame, 4 GT instances, 2 nodes each.
    instances = torch.rand(1, 4, 2, 2)
    image = torch.zeros(1, 1, 8, 8)

    # Without override: all 4 GT centroids survive.
    out_all = layer.predict(image, instances=instances)
    assert out_all.pred_centroids.shape[1] == 4

    # Apply predict-time override to cap at 2.
    import attrs

    layer.postprocess_config = attrs.evolve(
        layer.postprocess_config, max_instances=2
    )
    out_capped = layer.predict(image, instances=instances)
    assert out_capped.pred_centroids.shape[1] == 2
