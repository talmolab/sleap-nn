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


@pytest.mark.skipif(
    not MULTICLASS_BU_CKPT.exists(), reason="multiclass-bottomup checkpoint not present"
)
def test_parity_vs_legacy():
    """Match ``BottomUpMultiClassInferenceModel.forward`` within tolerance."""
    predictor = _build_predictor()
    layer = _build_layer(predictor)
    legacy = predictor.inference_model
    legacy.eval()

    # The legacy ``BottomUpMultiClassInferenceModel.forward`` does NOT
    # resize its input — it expects the image to already be at the model's
    # native (pre-input-scale) size, with the data loader having done the
    # resize upstream. The new layer's ``preprocess`` does the resize itself
    # so the user can pass an original-resolution image. To test parity, we
    # feed the legacy a pre-resized image and the new layer the original.
    from sleap_nn.data.resizing import resize_image as _resize

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(2, 1, 1, 384, 384)).astype(np.uint8)
    image_t = torch.from_numpy(image).float()  # (B, 1, C, H, W) original

    image_legacy = _resize(image_t.squeeze(1), legacy.input_scale).unsqueeze(1)
    legacy_input = {
        "image": image_legacy,
        "frame_idx": torch.tensor([0, 1], dtype=torch.float32),
        "video_idx": torch.tensor([0, 0], dtype=torch.float32),
        "orig_size": torch.tensor([[384.0, 384.0], [384.0, 384.0]]),
        "eff_scale": torch.ones(2),
    }
    with torch.no_grad():
        legacy_out_list = legacy(legacy_input)
    legacy_out = legacy_out_list[0]
    # Legacy returns ``pred_instance_peaks`` as a list of per-batch tensors
    # (because of the per-batch eff_scale division loop). Stack to compare.
    legacy_peaks = torch.stack(legacy_out["pred_instance_peaks"], dim=0)
    legacy_vals = legacy_out["pred_peak_values"]
    legacy_scores = legacy_out["instance_scores"]

    new_outputs = layer.predict(image_t.squeeze(1))

    np.testing.assert_allclose(
        new_outputs.pred_keypoints.numpy(),
        legacy_peaks.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="pred_keypoints drifted vs legacy BottomUpMultiClass",
    )
    np.testing.assert_allclose(
        new_outputs.pred_peak_values.numpy(),
        legacy_vals.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="pred_peak_values drifted vs legacy BottomUpMultiClass",
    )
    np.testing.assert_allclose(
        new_outputs.instance_scores.numpy(),
        legacy_scores.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="instance_scores drifted vs legacy BottomUpMultiClass",
    )
