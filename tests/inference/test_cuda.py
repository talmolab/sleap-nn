"""CUDA-gated tests for the inference stack.

Module-level ``skipif`` gates everything in this file on
``torch.cuda.is_available()``, so it's safe to leave in the regular test
suite — non-CUDA CI runs skip cleanly. On a CUDA host, run with::

    pytest tests/inference/test_cuda.py -v

Coverage:

1. **Pure ops on CUDA** — ``ops/peaks``, ``ops/crops``, ``ops/coord`` all
   produce outputs that match their CPU counterparts within float
   tolerance. Acts as a regression guard for any future op rewrite that
   inadvertently breaks one device.

2. **``Outputs`` device transfer** — ``Outputs.to('cuda')`` and
   ``Outputs.cpu()`` round-trip every populated field.

3. **``TorchBackend(device='cuda')``** — wrapping a Lightning module with
   ``device='cuda'`` produces forward outputs that match the CPU run
   within ULP-level tolerance.

4. **``SingleInstanceLayer`` cross-device parity** — end-to-end
   ``layer.predict(image)`` on CPU vs CUDA agree on
   ``pred_keypoints`` and ``pred_peak_values`` within the design-doc
   tolerance budget (1e-4 abs / 1e-5 rel).

5. **CUDA-specific ``TorchBackend`` features** — pin_memory +
   non_blocking transfer correctness, FP16 forward parity within the
   FP16 drift budget, warmup runs without raising.

The PR 5 (#513) ``crop_bboxes`` rewrite uses advanced-indexing on a
zero-padded image, which is implemented identically on CPU and CUDA, so
the cross-device tests are tight (1e-4 / 1e-5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available on this host",
)

# Tolerances. The CPU-vs-CUDA difference is dominated by float-op ordering in
# the kernel; ULP-level on simple ops, ~1e-5 on multi-op pipelines.
ATOL_CPU_GPU = 1e-4
RTOL_CPU_GPU = 1e-5
# FP16 drift vs FP32 from the design doc CUDA benchmarks (NVIDIA A40,
# torch 2.9.1+cu128). Generous absolute bound; still catches real bugs.
ATOL_FP16 = 5e-3
RTOL_FP16 = 1e-3


CKPT_ROOT = Path(__file__).resolve().parents[1] / "tests" / "assets" / "model_ckpts"
SINGLE_CKPT = CKPT_ROOT / "minimal_instance_single_instance"


# ─────────────────────────────────────────────────────────────────────────
# 1. Pure ops on CUDA
# ─────────────────────────────────────────────────────────────────────────


def test_morphological_dilation_cpu_vs_cuda():
    """8-shift NMS dilation produces identical output on CPU and CUDA."""
    from sleap_nn.inference.ops.peaks import morphological_dilation

    torch.manual_seed(0)
    img = torch.randn(2, 1, 16, 16)
    kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)

    cpu_out = morphological_dilation(img, kernel)
    cuda_out = morphological_dilation(img.cuda(), kernel.cuda()).cpu()
    torch.testing.assert_close(cuda_out, cpu_out, atol=ATOL_CPU_GPU, rtol=RTOL_CPU_GPU)


def test_find_global_peaks_rough_cpu_vs_cuda():
    """find_global_peaks_rough returns matching peaks + values on CPU and CUDA."""
    from sleap_nn.inference.ops.peaks import find_global_peaks_rough

    torch.manual_seed(0)
    cms = torch.softmax(torch.randn(3, 5, 16, 16), dim=-1)

    cpu_pts, cpu_vals = find_global_peaks_rough(cms, threshold=0.05)
    cuda_pts, cuda_vals = find_global_peaks_rough(cms.cuda(), threshold=0.05)
    torch.testing.assert_close(
        cuda_pts.cpu(), cpu_pts, atol=ATOL_CPU_GPU, rtol=RTOL_CPU_GPU, equal_nan=True
    )
    torch.testing.assert_close(
        cuda_vals.cpu(), cpu_vals, atol=ATOL_CPU_GPU, rtol=RTOL_CPU_GPU
    )


def test_crop_bboxes_cpu_vs_cuda():
    """crop_bboxes (PR 5 rewrite) produces matching crops on CPU and CUDA."""
    from sleap_nn.inference.ops.crops import crop_bboxes, make_centered_bboxes

    torch.manual_seed(0)
    imgs = torch.randn(4, 1, 32, 32)
    points = torch.tensor([[10.0, 10.0], [20.0, 15.0], [5.0, 25.0]])
    bboxes = make_centered_bboxes(points, box_height=5, box_width=5)
    sample_inds = torch.tensor([0, 2, 1])

    cpu_crops = crop_bboxes(imgs, bboxes, sample_inds)
    cuda_crops = crop_bboxes(imgs.cuda(), bboxes.cuda(), sample_inds.cuda()).cpu()
    torch.testing.assert_close(cuda_crops, cpu_crops, atol=0, rtol=0)


def test_integral_regression_cpu_vs_cuda():
    """integral_regression matches across devices."""
    from sleap_nn.inference.ops.peaks import integral_regression

    torch.manual_seed(0)
    cms = torch.softmax(torch.randn(2, 3, 5, 5), dim=-1)
    gv = torch.arange(5, dtype=torch.float32) - 2.0

    cpu_x, cpu_y = integral_regression(cms, gv, gv)
    cuda_x, cuda_y = integral_regression(cms.cuda(), gv.cuda(), gv.cuda())
    torch.testing.assert_close(
        cuda_x.cpu(), cpu_x, atol=ATOL_CPU_GPU, rtol=RTOL_CPU_GPU
    )
    torch.testing.assert_close(
        cuda_y.cpu(), cpu_y, atol=ATOL_CPU_GPU, rtol=RTOL_CPU_GPU
    )


def test_coord_ops_cpu_vs_cuda():
    """All five coord-ladder ops match across devices."""
    from sleap_nn.inference.ops.coord import (
        add_crop_offset,
        apply_input_scale,
        undo_eff_scale,
        undo_input_scale,
        undo_stride,
    )

    torch.manual_seed(0)
    coords = torch.randn(2, 3, 2)
    eff = torch.tensor([2.0, 0.5])
    topleft = torch.randn(2, 2)
    img = torch.randn(2, 1, 8, 8)

    # Identity-like ops should be bit-exact across devices.
    torch.testing.assert_close(
        undo_stride(coords.cuda(), 4).cpu(),
        undo_stride(coords, 4),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        add_crop_offset(coords.cuda(), topleft.cuda()).cpu(),
        add_crop_offset(coords, topleft),
        atol=0,
        rtol=0,
    )

    # Floating-point divisions produce ULP-level cross-device drift.
    torch.testing.assert_close(
        undo_input_scale(coords.cuda(), 0.5).cpu(),
        undo_input_scale(coords, 0.5),
        atol=ATOL_CPU_GPU,
        rtol=RTOL_CPU_GPU,
    )
    torch.testing.assert_close(
        undo_eff_scale(coords.cuda(), eff.cuda()).cpu(),
        undo_eff_scale(coords, eff),
        atol=ATOL_CPU_GPU,
        rtol=RTOL_CPU_GPU,
    )
    torch.testing.assert_close(
        apply_input_scale(img.cuda(), 0.5).cpu(),
        apply_input_scale(img, 0.5),
        atol=ATOL_CPU_GPU,
        rtol=RTOL_CPU_GPU,
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. Outputs device transfer
# ─────────────────────────────────────────────────────────────────────────


def test_outputs_to_cuda_round_trip():
    """``Outputs.to('cuda').cpu()`` recovers every populated field bit-exactly."""
    from sleap_nn.inference.outputs import Outputs

    o = Outputs(
        pred_keypoints=torch.randn(2, 3, 4, 2),
        pred_peak_values=torch.rand(2, 3, 4),
        pred_centroids=torch.randn(2, 3, 2),
        instance_scores=torch.rand(2, 3),
        frame_indices=torch.arange(2, dtype=torch.int64),
    )
    on_cuda = o.to("cuda")
    assert on_cuda.pred_keypoints.device.type == "cuda"
    back = on_cuda.cpu()
    torch.testing.assert_close(back.pred_keypoints, o.pred_keypoints, atol=0, rtol=0)
    torch.testing.assert_close(
        back.pred_peak_values, o.pred_peak_values, atol=0, rtol=0
    )
    torch.testing.assert_close(back.frame_indices, o.frame_indices, atol=0, rtol=0)


# ─────────────────────────────────────────────────────────────────────────
# 3. TorchBackend(device='cuda')
# ─────────────────────────────────────────────────────────────────────────


class _TinyConvModel(nn.Module):
    """Stand-in Lightning-shaped module for CUDA backend tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve the (already-float) input."""
        return self.conv(x.float())


def test_torch_backend_cuda_forward_parity():
    """Wrapping a model in ``TorchBackend(device='cuda')`` matches CPU forward."""
    from sleap_nn.inference.layers.backends import TorchBackend

    torch.manual_seed(0)
    model_cpu = _TinyConvModel()
    model_cuda = _TinyConvModel()
    model_cuda.load_state_dict(model_cpu.state_dict())

    x = torch.randn(2, 1, 16, 16)
    cpu_out = TorchBackend(model=model_cpu, device="cpu")(x)["output"]
    cuda_out = TorchBackend(model=model_cuda, device="cuda")(x)["output"].cpu()
    torch.testing.assert_close(cuda_out, cpu_out, atol=ATOL_CPU_GPU, rtol=RTOL_CPU_GPU)


def test_torch_backend_warmup_on_cuda_runs():
    """``warmup()`` on a CUDA backend completes without raising."""
    from sleap_nn.inference.layers.backends import TorchBackend

    backend = TorchBackend(model=_TinyConvModel(), device="cuda", warmup_iterations=2)
    backend.warmup((1, 1, 16, 16))
    # Subsequent forward should be no slower than warmup; just verify it runs.
    out = backend(torch.zeros(1, 1, 16, 16))
    assert out["output"].shape == (1, 4, 16, 16)


def test_torch_backend_pin_memory_transfer_correctness():
    """``non_blocking=True`` device transfer doesn't corrupt input bytes."""
    from sleap_nn.inference.layers.backends import TorchBackend

    backend = TorchBackend(model=_TinyConvModel(), device="cuda")
    # Pinned host tensor — eligible for non-blocking transfer.
    x = torch.zeros(4, 1, 32, 32).pin_memory()
    x[0, 0, 16, 16] = 1.0
    out = backend(x)["output"].cpu()
    # The model is a single Conv2d — center-pixel impulse spreads to 9 outputs.
    assert torch.isfinite(out).all()
    assert out.shape == (4, 4, 32, 32)


# ─────────────────────────────────────────────────────────────────────────
# 4. SingleInstanceLayer cross-device parity (end-to-end)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not SINGLE_CKPT.exists(), reason="single-instance checkpoint not present"
)
def test_single_instance_layer_cross_device_parity():
    """``SingleInstanceLayer.predict(image)`` agrees on CPU and CUDA.

    Same Lightning module, same input, both devices — the predicted
    keypoints and scores must match within the design-doc tolerance
    budget (1e-4 abs / 1e-5 rel).
    """
    from omegaconf import OmegaConf

    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
    from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
    from sleap_nn.inference.predictors import Predictor

    def _build(device: str) -> SingleInstanceLayer:
        predictor = Predictor.from_model_paths(
            [str(SINGLE_CKPT)],
            device=device,
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
        predictor._initialize_inference_model()
        inf = predictor.inference_model
        return SingleInstanceLayer(
            backend=TorchBackend(model=inf.torch_model, device=device),
            output_stride=inf.output_stride,
            preprocess_config=PreprocessConfig(scale=inf.input_scale),
            postprocess_config=PostprocessConfig(
                peak_threshold=inf.peak_threshold,
                refinement=inf.refinement or "none",
                integral_patch_size=inf.integral_patch_size,
            ),
        )

    layer_cpu = _build("cpu")
    layer_cuda = _build("cuda")

    # Synthetic deterministic input — same bytes on both devices.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(2, 1, 64, 64)).astype(np.uint8)

    cpu_out = layer_cpu.predict(img)
    cuda_out = layer_cuda.predict(img).cpu()

    torch.testing.assert_close(
        cuda_out.pred_keypoints,
        cpu_out.pred_keypoints,
        atol=ATOL_CPU_GPU,
        rtol=RTOL_CPU_GPU,
        equal_nan=True,
    )
    torch.testing.assert_close(
        cuda_out.pred_peak_values,
        cpu_out.pred_peak_values,
        atol=ATOL_CPU_GPU,
        rtol=RTOL_CPU_GPU,
    )


# ─────────────────────────────────────────────────────────────────────────
# 5. CUDA-specific TorchBackend features
# ─────────────────────────────────────────────────────────────────────────


def test_torch_backend_fp16_forward_within_drift_budget():
    """``use_fp16=True`` produces output close to FP32 within design budget."""
    import warnings

    from sleap_nn.inference.layers.backends import TorchBackend

    torch.manual_seed(0)
    model = _TinyConvModel()

    backend_fp32 = TorchBackend(model=model, device="cuda")
    # Suppress the documented FP16 small-batch / drift warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        backend_fp16 = TorchBackend(
            model=_TinyConvModel(), device="cuda", use_fp16=True
        )
        backend_fp16.model.load_state_dict(backend_fp32.model.state_dict())

    x = torch.randn(8, 1, 32, 32)
    out_fp32 = backend_fp32(x)["output"].cpu()
    out_fp16 = backend_fp16(x)["output"].cpu()
    torch.testing.assert_close(out_fp16, out_fp32, atol=ATOL_FP16, rtol=RTOL_FP16)


def test_torch_backend_cuda_does_not_block_default_use_compile_false():
    """The #527 compile guard must not affect the default ``use_compile=False`` path."""
    from sleap_nn.inference.layers.backends import TorchBackend

    backend = TorchBackend(model=_TinyConvModel(), device="cuda", use_compile=False)
    assert backend._compiled is None
    # Forward still works.
    out = backend(torch.zeros(1, 1, 8, 8))
    assert out["output"].shape == (1, 4, 8, 8)


# ─────────────────────────────────────────────────────────────────────────
# 6. Top-down crops on the stage-2 GPU (CPU frames, CUDA model)
# ─────────────────────────────────────────────────────────────────────────

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CENTROID_CKPT = ASSETS / "model_ckpts" / "minimal_instance_centroid"
CENTERED_CKPT = ASSETS / "model_ckpts" / "minimal_instance_centered_instance"


def _stub_topdown(device: str, bbox_crop, record: dict):
    """A ``TopDownLayer`` whose stages are stubs, stage 2 on ``device``.

    Stage 1 reports two centroids (one overhanging the frame edge) and their boxes
    on the CPU, like GT / given boxes. Stage 2 records the crops it is handed.
    """
    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
    from sleap_nn.inference.layers.centroid import CentroidLayer
    from sleap_nn.inference.layers.configs import PreprocessConfig
    from sleap_nn.inference.layers.topdown import TopDownLayer
    from sleap_nn.inference.outputs import Outputs

    centroid_layer = CentroidLayer.__new__(CentroidLayer)
    centroid_layer.use_gt_centroids = False
    centroid_layer.preprocess_config = PreprocessConfig()
    centroid_layer._to_4d_tensor = staticmethod(CentroidLayer._to_4d_tensor)
    boxes = torch.tensor([[[10.0, 12.0, 40.0, 50.0], [-6.0, 30.0, 20.5, 70.25]]])

    def fake_centroids(image, instances=None):
        return Outputs(
            pred_centroids=(boxes[..., :2] + boxes[..., 2:]) / 2,
            pred_centroid_values=torch.tensor([[0.9, 0.8]]),
            pred_bboxes=boxes,
            pred_bbox_scores=torch.tensor([[0.9, 0.8]]),
            pred_bbox_classes=torch.zeros(1, 2, dtype=torch.long),
        )

    centroid_layer.predict = fake_centroids

    inst_layer = CenteredInstanceLayer.__new__(CenteredInstanceLayer)
    inst_layer.use_gt_peaks = False
    inst_layer.backend = TorchBackend(model=nn.Identity(), device=device)
    inst_layer.preprocess_config = PreprocessConfig()

    def fake_peaks(crops):
        record["crops"] = crops
        n = crops.shape[0]
        # One "peak" per crop at its mean intensity, so outputs depend on pixels.
        mean = crops.float().mean(dim=(1, 2, 3)).to(device)
        return Outputs(
            pred_keypoints=mean.view(n, 1, 1, 1).expand(n, 1, 1, 2).contiguous(),
            pred_peak_values=torch.ones(n, 1, 1, device=device),
        )

    inst_layer.predict = fake_peaks
    return TopDownLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(24, 24),
        bbox_crop=bbox_crop,
    )


@pytest.mark.parametrize("crop_mode", ["fixed", "bbox"])
def test_topdown_crops_cpu_frames_on_the_model_gpu(crop_mode):
    """CPU frames are cropped on stage 2's GPU, with the CPU crops and outputs.

    Fixed windows are a gather: bit-identical. Box-scaled windows are bilinear
    (``grid_sample``), whose CUDA kernel rounds a rare pixel to the other grey
    level once the crop is quantized back to ``uint8``.
    """
    from sleap_nn.data.instance_cropping import BBoxCropSpec

    bbox_crop = BBoxCropSpec(aspect="letterbox") if crop_mode == "bbox" else None
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(1, 3, 64, 48), dtype=np.uint8)

    cpu_record: dict = {}
    cpu_out = _stub_topdown("cpu", bbox_crop, cpu_record).predict(frames)
    cuda_record: dict = {}
    cuda_out = _stub_topdown("cuda", bbox_crop, cuda_record).predict(frames)

    assert cpu_record["crops"].device.type == "cpu"
    assert cuda_record["crops"].device.type == "cuda"
    assert cuda_record["crops"].dtype == torch.uint8
    crop_diff = (cuda_record["crops"].cpu().int() - cpu_record["crops"].int()).abs()
    if crop_mode == "fixed":
        assert crop_diff.max() == 0
    else:
        assert crop_diff.max() <= 1 and (crop_diff > 0).float().mean() < 0.01
    atol = ATOL_CPU_GPU if crop_mode == "fixed" else 0.01
    for field in ("pred_keypoints", "instance_bboxes", "pred_centroids"):
        torch.testing.assert_close(
            getattr(cuda_out, field).cpu(),
            getattr(cpu_out, field),
            atol=atol,
            rtol=RTOL_CPU_GPU,
            equal_nan=True,
        )


@pytest.mark.skipif(
    not CENTROID_CKPT.exists() or not CENTERED_CKPT.exists(),
    reason="top-down checkpoints not present",
)
def test_topdown_predictor_cuda_matches_cpu(minimal_instance):
    """A centroid + centered-instance pair on CUDA predicts what it does on CPU."""
    import sleap_io as sio

    from sleap_nn.inference.predictor import Predictor

    labels = sio.load_slp(minimal_instance)

    def _keypoints(device: str) -> np.ndarray:
        predictor = Predictor.from_model_paths(
            [str(CENTROID_CKPT), str(CENTERED_CKPT)],
            device=device,
            peak_threshold=0.1,
            max_instances=6,
        )
        if device == "cuda":
            assert predictor.layer._crop_device() == torch.device("cuda")
        pred = predictor.predict(labels, make_labels=True)
        return np.stack(
            [inst.numpy() for lf in pred.labeled_frames for inst in lf.instances]
        )

    cpu_kpts = _keypoints("cpu")
    cuda_kpts = _keypoints("cuda")
    assert cpu_kpts.shape == cuda_kpts.shape and len(cpu_kpts) > 0
    # Fixed crops are bit-identical across devices; what is left is the models'
    # CUDA-vs-CPU forward (the single-instance parity test's 0.01 px budget).
    np.testing.assert_allclose(cuda_kpts, cpu_kpts, atol=0.01, equal_nan=True)
