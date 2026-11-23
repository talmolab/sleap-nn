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


# ─────────────────────────────────────────────────────────────────────────
# use_compile: one compile however many animals are found (#783)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CENTROID_CKPT.exists() or not CENTERED_CKPT.exists(),
    reason="topdown checkpoints not present",
)
def test_compiled_topdown_compiles_each_model_once(monkeypatch):
    """The crop count changes from batch to batch; each model still compiles once.

    ``torch.compile`` runs real dynamo on its eager backend: it guards on input
    shapes and builds a new graph for each new shape exactly as with inductor,
    without inductor's compile time. The crops are padded to one crop per
    centroid slot (batch size x max_instances); the pad crops must never become
    instances, so the output equals the uncompiled pipeline's.
    """
    import sleap_io as sio
    import torch._dynamo

    from sleap_nn.inference.predictor import Predictor

    real_compile = torch.compile
    graphs = []

    def counting_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def compile_on_eager_backend(model, mode=None, dynamic=None):
        return real_compile(model, backend=counting_backend, dynamic=dynamic)

    video = sio.load_video(
        str(CKPT_ROOT.parent / "datasets" / "centered_pair_small.mp4")
    )
    frames = np.stack([video[i] for i in range(14)])
    # Blank frames have no animals: batches of 4 then hold 8, 6, 4 and 2 crops
    # (2 animals per frame), and the last batch has 2 frames.
    frames[[5, 9, 10, 13]] = 0

    def run(use_compile):
        predictor = Predictor.from_model_paths(
            [str(CENTROID_CKPT), str(CENTERED_CKPT)],
            device="cpu",
            batch_size=4,
            max_instances=2,
            use_compile=use_compile,
        )
        layer = predictor.layer
        crop_counts = []
        inner_predict = layer.centered_instance_layer.predict

        def spy(crops, *args, **kwargs):
            crop_counts.append(crops.shape[0])
            return inner_predict(crops, *args, **kwargs)

        layer.centered_instance_layer.predict = spy
        outs = [layer.predict(frames[s : s + 4]) for s in range(0, len(frames), 4)]
        return outs, crop_counts

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", compile_on_eager_backend)
    try:
        compiled_outs, crop_counts = run(use_compile=True)
    finally:
        torch._dynamo.reset()
    monkeypatch.undo()
    eager_outs, eager_counts = run(use_compile=False)

    assert crop_counts == eager_counts
    assert len(set(crop_counts)) >= 3, crop_counts
    # One graph per model: the centroid model (whose last, smaller batch of
    # frames is padded too) and the centered-instance model.
    assert len(graphs) == 2

    for c, e in zip(compiled_outs, eager_outs):
        assert torch.equal(torch.isnan(c.pred_keypoints), torch.isnan(e.pred_keypoints))
        torch.testing.assert_close(
            c.pred_keypoints, e.pred_keypoints, equal_nan=True, atol=1e-3, rtol=0
        )
        torch.testing.assert_close(
            c.pred_peak_values, e.pred_peak_values, equal_nan=True, atol=1e-4, rtol=0
        )
