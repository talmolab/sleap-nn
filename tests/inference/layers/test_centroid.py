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


# ─────────────────────────────────────────────────────────────────────────
# 1. Parity vs legacy CentroidCrop on a deterministic input
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists(), reason="centroid checkpoint not present"
)
def test_centroid_layer_parity_vs_legacy_model_path():
    """``CentroidLayer.predict(image)`` matches ``CentroidCrop.forward({...})``."""
    predictor = _build_predictor()
    layer = _build_layer(predictor)
    legacy = predictor.inference_model.centroid_crop
    legacy.eval()
    legacy.return_crops = False
    legacy.use_gt_centroids = False

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(2, 1, 1, 384, 384)).astype(np.uint8)
    image_t = torch.from_numpy(image).float()  # (B, n_samples, C, H, W)

    legacy_input = {
        "image": image_t,
        "frame_idx": torch.tensor([0, 1], dtype=torch.float32),
        "video_idx": torch.tensor([0, 0], dtype=torch.float32),
        "orig_size": torch.tensor([[384.0, 384.0], [384.0, 384.0]]),
        "eff_scale": torch.ones(2),
    }
    with torch.no_grad():
        legacy_out = legacy(legacy_input)
    legacy_centroids = legacy_out["centroids"].squeeze(1)  # (B, max_inst, 2)
    legacy_vals = legacy_out["centroid_vals"]  # (B, max_inst)

    new_outputs = layer.predict(image_t.squeeze(1))  # (B, C, H, W)

    np.testing.assert_allclose(
        new_outputs.pred_centroids.numpy(),
        legacy_centroids.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="pred_centroids drifted vs legacy CentroidCrop",
    )
    np.testing.assert_allclose(
        new_outputs.pred_centroid_values.numpy(),
        legacy_vals.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="pred_centroid_values drifted vs legacy CentroidCrop",
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. Direct numpy / torch APIs
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
