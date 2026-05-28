"""Tests for ``BottomUpMultiClassLayer``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.bottomup_multiclass import BottomUpMultiClassLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
MULTICLASS_BU_CKPT = CKPT_ROOT / "minimal_instance_multiclass_bottomup"

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


def _build_predictor():
    """Build a multiclass-bottomup Predictor with the test checkpoint."""
    from sleap_nn.inference.predictors import Predictor

    predictor = Predictor.from_model_paths(
        [str(MULTICLASS_BU_CKPT)],
        device="cpu",
        peak_threshold=0.05,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
    return predictor


def _build_layer(predictor) -> BottomUpMultiClassLayer:
    legacy = predictor.inference_model
    max_stride = predictor.bottomup_config.model_config.backbone_config[
        predictor.backbone_type
    ]["max_stride"]
    return BottomUpMultiClassLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        cms_output_stride=legacy.cms_output_stride,
        class_maps_output_stride=legacy.class_maps_output_stride,
        max_stride=max_stride,
        preprocess_config=PreprocessConfig(scale=legacy.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            integral_patch_size=legacy.integral_patch_size,
        ),
    )


@pytest.mark.skipif(
    not MULTICLASS_BU_CKPT.exists(), reason="multiclass-bottomup checkpoint not present"
)
def test_returns_outputs_with_canonical_shape():
    layer = _build_layer(_build_predictor())
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = layer.predict(img)
    assert isinstance(out, Outputs)
    # (B, n_classes, n_nodes, 2)
    assert out.pred_keypoints.ndim == 4
    assert out.pred_keypoints.shape[0] == 1


# Per-layer "parity vs legacy BottomUpMultiClassInferenceModel.forward"
# was removed: end-to-end parity is captured at the top of the stack
# via the PR 0 goldens. Locking each layer to legacy byte-for-byte
# also locks in float-op-ordering quirks we may want to change later.
