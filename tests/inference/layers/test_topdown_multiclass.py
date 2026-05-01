"""Tests for ``CenteredInstanceMultiClassLayer`` + ``TopDownMultiClassLayer``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.topdown_multiclass import (
    CenteredInstanceMultiClassLayer,
    TopDownMultiClassLayer,
)
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
MULTICLASS_TD_CKPT = CKPT_ROOT / "minimal_instance_multiclass_centered_instance"

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
    from sleap_nn.inference.predictors import Predictor

    predictor = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(MULTICLASS_TD_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
    return predictor


def _build_inst_layer(predictor) -> CenteredInstanceMultiClassLayer:
    legacy = predictor.inference_model.instance_peaks
    return CenteredInstanceMultiClassLayer(
        backend=TorchBackend(model=legacy.torch_model, device="cpu"),
        output_stride=legacy.output_stride,
        max_stride=legacy.max_stride,
        preprocess_config=PreprocessConfig(scale=legacy.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy.peak_threshold,
            refinement=legacy.refinement or "none",
            integral_patch_size=legacy.integral_patch_size,
        ),
    )


@pytest.mark.skipif(
    not MULTICLASS_TD_CKPT.exists(), reason="multiclass-topdown checkpoint not present"
)
def test_centered_instance_multiclass_layer_parity_vs_legacy():
    """Per-crop output matches legacy ``TopDownMultiClassFindInstancePeaks``."""
    predictor = _build_predictor()
    layer = _build_inst_layer(predictor)
    legacy = predictor.inference_model.instance_peaks
    legacy.eval()

    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, size=(2, 1, 1, 96, 96)).astype(np.uint8)
    crops_t = torch.from_numpy(crops).float()

    legacy_input = {
        "instance_image": crops_t,
        "instance_bbox": torch.zeros(2, 1, 4, 2),
        "eff_scale": torch.ones(2),
    }
    with torch.no_grad():
        legacy_out = legacy(legacy_input)
    legacy_peaks = legacy_out["pred_instance_peaks"]
    legacy_vals = legacy_out["pred_peak_values"]
    legacy_scores = legacy_out["instance_scores"]

    new_outputs = layer.predict(crops_t.squeeze(1))
    new_peaks = new_outputs.pred_keypoints.squeeze(1)
    new_vals = new_outputs.pred_peak_values.squeeze(1)
    new_scores = new_outputs.instance_scores.squeeze(1)

    np.testing.assert_allclose(
        new_peaks.numpy(),
        legacy_peaks.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="pred_keypoints drifted vs legacy MultiClass topdown",
    )
    np.testing.assert_allclose(
        new_vals.numpy(),
        legacy_vals.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    np.testing.assert_allclose(
        new_scores.numpy(),
        legacy_scores.numpy(),
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and MULTICLASS_TD_CKPT.exists()),
    reason="multiclass-topdown checkpoints not present",
)
def test_topdown_multiclass_layer_returns_outputs():
    """End-to-end ``TopDownMultiClassLayer`` returns a populated ``Outputs``."""
    predictor = _build_predictor()
    legacy_centroid = predictor.inference_model.centroid_crop
    legacy_inst = predictor.inference_model.instance_peaks

    centroid_layer = CentroidLayer(
        backend=TorchBackend(model=legacy_centroid.torch_model, device="cpu"),
        output_stride=legacy_centroid.output_stride,
        max_instances=legacy_centroid.max_instances,
        max_stride=legacy_centroid.max_stride,
        anchor_ind=legacy_centroid.anchor_ind,
        preprocess_config=PreprocessConfig(scale=legacy_centroid.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy_centroid.peak_threshold,
            refinement=legacy_centroid.refinement or "none",
            max_instances=legacy_centroid.max_instances,
        ),
    )
    inst_layer = _build_inst_layer(predictor)
    crop_h, crop_w = legacy_centroid.crop_hw
    layer = TopDownMultiClassLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(crop_h, crop_w),
    )
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints is not None
    assert out.pred_keypoints.shape[0] == 1


def test_topdown_multiclass_rejects_wrong_inner_layer():
    """Constructor type-checks that the centered-instance layer is the
    multi-class variant, not the plain one."""
    from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer

    centroid = CentroidLayer.__new__(CentroidLayer)
    plain_inst = CenteredInstanceLayer.__new__(CenteredInstanceLayer)
    with pytest.raises(TypeError, match="MultiClass"):
        TopDownMultiClassLayer(
            centroid_layer=centroid,
            centered_instance_layer=plain_inst,  # type: ignore[arg-type]
            crop_size=(8, 8),
        )
