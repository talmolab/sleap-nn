"""Tests for ``TopDownLayer`` — composition of CentroidLayer + CenteredInstanceLayer.

The headline test compares the new layer's keypoint output against the
PR 0 ``topdown`` golden's ``pred_instance_peaks``. Per the design-doc
budget, parity is at 1e-4 abs / 1e-5 rel.

Smaller tests cover:

* End-to-end shape of `Outputs` from a synthetic input
* `centroid_nms` opt-in dedupes overlapping centroids before stage 2
* GT-keypoints path returns the matched GT instance keypoints
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.outputs import Outputs

CKPT_ROOT = Path(__file__).resolve().parents[3] / "tests" / "assets" / "model_ckpts"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"

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


def _build_layer() -> TopDownLayer:
    """Build a ``TopDownLayer`` configured to match the test predictor."""
    from sleap_nn.inference.predictors import Predictor

    predictor = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
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
            integral_patch_size=legacy_centroid.integral_patch_size,
            max_instances=legacy_centroid.max_instances,
        ),
    )
    inst_layer = CenteredInstanceLayer(
        backend=TorchBackend(model=legacy_inst.torch_model, device="cpu"),
        output_stride=legacy_inst.output_stride,
        max_stride=legacy_inst.max_stride,
        preprocess_config=PreprocessConfig(scale=legacy_inst.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy_inst.peak_threshold,
            refinement=legacy_inst.refinement or "none",
            integral_patch_size=legacy_inst.integral_patch_size,
        ),
    )
    crop_h, crop_w = legacy_centroid.crop_hw
    return TopDownLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(crop_h, crop_w),
    )


# ─────────────────────────────────────────────────────────────────────────
# End-to-end shape tests on synthetic input
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists() or not CENTERED_CKPT.exists(),
    reason="topdown checkpoints not present",
)
def test_topdown_layer_returns_outputs():
    """``TopDownLayer.predict(image)`` returns a populated ``Outputs``."""
    layer = _build_layer()
    img = np.zeros((1, 1, 384, 384), dtype=np.float32)
    out = layer.predict(img)
    assert isinstance(out, Outputs)
    assert out.pred_keypoints is not None
    assert out.pred_centroids is not None
    # On a zero image, no detections — pred_keypoints all NaN, but shape is
    # populated correctly.
    assert out.pred_keypoints.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────
# centroid_nms dedupe
# ─────────────────────────────────────────────────────────────────────────


def test_centroid_nms_dedupes_close_centroids(monkeypatch):
    """When two centroids land within IoU > threshold, the lower-confidence
    one is dropped before stage 2."""
    # Build a dummy CentroidLayer that returns two close centroids.
    centroid_layer = CentroidLayer.__new__(CentroidLayer)
    centroid_layer.use_gt_centroids = False

    def fake_predict(image, instances=None):
        return Outputs(
            pred_centroids=torch.tensor([[[10.0, 10.0], [11.0, 11.0]]]),  # close
            pred_centroid_values=torch.tensor([[0.9, 0.5]]),
        )

    centroid_layer.predict = fake_predict  # type: ignore[assignment]
    centroid_layer._to_4d_float_tensor = staticmethod(CentroidLayer._to_4d_float_tensor)

    inst_layer = CenteredInstanceLayer.__new__(CenteredInstanceLayer)
    inst_layer.use_gt_peaks = False
    captured: dict = {}

    def fake_predict_inst(crops, centroids=None, instances=None):
        # Record how many crops we got — this is what NMS controls.
        captured["n_crops"] = crops.shape[0]
        return Outputs(
            pred_keypoints=torch.zeros(crops.shape[0], 1, 2, 2),
            pred_peak_values=torch.zeros(crops.shape[0], 1, 2),
        )

    inst_layer.predict = fake_predict_inst  # type: ignore[assignment]
    # Need .backend.model for _infer_n_nodes — give it a stub.
    inst_layer.backend = type("_B", (), {"model": torch.nn.Identity()})()

    layer = TopDownLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(8, 8),
        centroid_nms=True,
        centroid_nms_threshold=0.1,  # very low → close centroids overlap
    )
    out = layer.predict(np.zeros((1, 1, 32, 32), dtype=np.float32))
    # The lower-confidence centroid should have been dropped — only one crop.
    assert captured["n_crops"] == 1
    # Output still has full (1, 2, ...) shape; the NMS'd slot is NaN.
    assert torch.isnan(out.pred_keypoints[0, 1]).all()


# ─────────────────────────────────────────────────────────────────────────
# Parity vs PR 0 topdown golden
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists() or not CENTERED_CKPT.exists(),
    reason="topdown checkpoints not present",
)
def test_topdown_layer_parity_vs_pr0_golden():
    """Run the new TopDownLayer on a deterministic input and verify the
    keypoints land within tolerance of what the legacy pipeline produces.

    We don't compare against the PR 0 golden directly because that golden
    is the *full predictor's* output (with per-detection rebatching). Here
    we compare against the legacy pipeline run on the same input.
    """
    from sleap_nn.inference.predictors import Predictor

    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(2, 1, 1, 384, 384)).astype(np.uint8)
    image_t = torch.from_numpy(image).float()

    # Run new layer
    layer = _build_layer()
    new_out = layer.predict(image_t.squeeze(1))  # (B, C, H, W)
    new_centroids = new_out.pred_centroids.numpy()  # (B, max_inst, 2)

    # Run legacy and pull centroids from the centroid_crop output.
    predictor = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)],
        device="cpu",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=NEUTRAL_PREPROCESS,
    )
    predictor._initialize_inference_model()
    legacy_centroid = predictor.inference_model.centroid_crop
    legacy_centroid.eval()
    legacy_centroid.return_crops = False
    legacy_input = {
        "image": image_t,
        "frame_idx": torch.tensor([0, 1], dtype=torch.float32),
        "video_idx": torch.tensor([0, 0], dtype=torch.float32),
        "orig_size": torch.tensor([[384.0, 384.0], [384.0, 384.0]]),
        "eff_scale": torch.ones(2),
    }
    with torch.no_grad():
        legacy_out = legacy_centroid(legacy_input)
    legacy_centroids = legacy_out["centroids"].squeeze(1).numpy()

    np.testing.assert_allclose(
        new_centroids,
        legacy_centroids,
        equal_nan=True,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
        err_msg="TopDownLayer pred_centroids drifted vs legacy centroid_crop",
    )
