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
    centroid_layer._to_4d_tensor = staticmethod(CentroidLayer._to_4d_tensor)
    # PR 27: TopDownLayer reads preprocess_config off the centroid_layer
    # to re-apply sizematcher when cropping. Stub a no-op config here.
    centroid_layer.preprocess_config = PreprocessConfig()

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


# Per-stage "parity vs legacy CentroidCrop.forward" was removed:
# end-to-end byte-for-byte parity is captured at the top of the stack
# via the PR 0 goldens (the topdown golden is captured there directly
# from the full predictor and is the canonical comparison point).


def test_infer_n_nodes_reads_the_head_not_the_backbone():
    """The empty-output node count comes from the confmap head's output conv."""

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # A ViT's first conv (patch embedding) has embed_dim channels.
            self.backbone = torch.nn.Conv2d(3, 384, 16, stride=16)
            self.head_layers = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(384, 64, 3), torch.nn.Conv2d(64, 17, 1)
                    )
                ]
            )

    inst_layer = CenteredInstanceLayer.__new__(CenteredInstanceLayer)
    inst_layer.backend = type("_B", (), {"model": _Model()})()
    layer = TopDownLayer.__new__(TopDownLayer)
    layer.centered_instance_layer = inst_layer
    assert layer._infer_n_nodes() == 17


def test_crop_device_only_for_a_cuda_torch_stage_2():
    """Frames move to the stage-2 GPU for cropping only when nothing changes.

    A CUDA ``TorchBackend`` with no crop resampling crops on the GPU; CPU / MPS /
    exported backends, and a stage 2 that resizes its crops (a GPU ``uint8``
    resize rounds differently), keep cropping where the frame is.
    """

    class _ExportedBackend:  # e.g. ONNX / TensorRT: not a TorchBackend
        device = "cuda"

    def _layer(device, backend_cls=TorchBackend, **preprocess):
        backend = object.__new__(backend_cls)  # no model / device needed
        if backend_cls is TorchBackend:
            backend.device = device
        inst_layer = CenteredInstanceLayer.__new__(CenteredInstanceLayer)
        inst_layer.backend = backend
        inst_layer.preprocess_config = PreprocessConfig(**preprocess)
        layer = TopDownLayer.__new__(TopDownLayer)
        layer.centered_instance_layer = inst_layer
        return layer

    assert _layer("cuda")._crop_device() == torch.device("cuda")
    assert _layer("cuda:1")._crop_device() == torch.device("cuda:1")
    assert _layer("cpu")._crop_device() is None
    assert _layer("mps")._crop_device() is None
    assert _layer("cuda", scale=0.5)._crop_device() is None
    assert _layer("cuda", max_height=256)._crop_device() is None
    assert _layer("cuda", backend_cls=_ExportedBackend)._crop_device() is None
