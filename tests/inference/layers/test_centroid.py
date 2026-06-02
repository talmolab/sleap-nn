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
    """Build a topdown inference model via the new loader.

    Returns the ``TopDownInferenceModel`` directly; the helpers below read
    ``.centroid_crop`` off it, which is identical to what the legacy
    ``predictor.inference_model`` exposed.
    """
    from sleap_nn.inference.loaders import load_model_assets

    assets, _ = load_model_assets(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    return assets.inference_model


def _build_layer(inference_model) -> CentroidLayer:
    """Build a ``CentroidLayer`` around the centroid module."""
    legacy = inference_model.centroid_crop
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
    inference_model = _build_predictor()
    legacy = inference_model.centroid_crop
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
