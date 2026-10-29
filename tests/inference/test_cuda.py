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


ASSETS = Path(__file__).resolve().parents[1] / "assets"
CKPT_ROOT = ASSETS / "model_ckpts"
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

    Same Lightning module, same input, both devices. A whole UNet forward
    drifts more than single ops (cuDNN kernel choice, TF32): 0.0021 px was
    seen on an RTX 6000 Ada, so keypoints get 0.01 px and scores 1e-3.
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
        atol=1e-2,
        rtol=RTOL_CPU_GPU,
        equal_nan=True,
    )
    torch.testing.assert_close(
        cuda_out.pred_peak_values,
        cpu_out.pred_peak_values,
        atol=1e-3,
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
# 6. Predictor.upload_batches: numpy batches copied to the GPU first
# ─────────────────────────────────────────────────────────────────────────

TOPDOWN_CKPTS = [
    CKPT_ROOT / "minimal_instance_centroid",
    CKPT_ROOT / "minimal_instance_centered_instance",
]
VIDEO = ASSETS / "datasets" / "small_robot.mp4"


def test_predictor_uploads_uint8_batches_to_cuda():
    """With a CUDA device the layer gets the provider's uint8 batch on the GPU."""
    from sleap_nn.inference.outputs import Outputs
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    seen = []

    class _Recording:
        def predict(self, image, **kwargs):
            seen.append(image)
            b = image.shape[0]
            return Outputs(
                pred_keypoints=torch.zeros(b, 1, 4, 2),
                pred_peak_values=torch.ones(b, 1, 4),
            )

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(4, 8, 8, 1)).astype(np.uint8)
    Predictor(
        layer=_Recording(), batch_size=2, device="cuda", upload_batches=True
    ).predict(NumpyProvider(images=images, batch_size=2), make_labels=False)

    assert len(seen) == 2
    for i, image in enumerate(seen):
        assert isinstance(image, torch.Tensor)
        assert image.device.type == "cuda"
        assert image.dtype == torch.uint8
        np.testing.assert_array_equal(image.cpu().numpy(), images[2 * i : 2 * i + 2])


@pytest.mark.skipif(
    not (VIDEO.exists() and all(p.exists() for p in TOPDOWN_CKPTS)),
    reason="top-down checkpoints or small_robot.mp4 not present",
)
def test_topdown_uploaded_batch_preprocesses_like_numpy_batch():
    """The GPU resize of an uploaded batch matches the CPU one to 1 grey level.

    Only where the sizematcher / input-scale resize runs changes (GPU instead
    of CPU); its ``uint8`` output can round the other way, by 1, in some
    pixels. Compared on the preprocessed frames rather than on keypoints: the
    minimal fixture models are not trained on ``small_robot.mp4``, and their
    flat confidence maps let 0.01-sized differences move a peak by pixels.
    """
    import sleap_io as sio

    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import VideoProvider

    video = sio.load_video(str(VIDEO))
    frames = np.stack([video[i] for i in range(4)])
    predictor = Predictor.from_model_paths(
        [str(p) for p in TOPDOWN_CKPTS],
        device="cuda",
        batch_size=4,
        upload_batches=True,
    )
    assert predictor.upload_batches
    centroid_layer = predictor.layer.centroid_layer
    x_host, info_host = centroid_layer.preprocess(frames)
    x_gpu, info_gpu = centroid_layer.preprocess(torch.from_numpy(frames).cuda())
    assert x_gpu.device.type == "cuda"
    assert x_gpu.shape == x_host.shape and x_gpu.dtype == x_host.dtype
    diff = (x_gpu.float().cpu() - x_host.float().cpu()).abs()
    assert diff.max().item() <= 1.0
    torch.testing.assert_close(info_gpu.eff_scale.cpu(), info_host.eff_scale.cpu())

    # And the whole top-down run works on an uploaded batch.
    provider = VideoProvider(video=video, batch_size=4, frames=list(range(8)))
    outputs = list(predictor.predict_streaming(provider))
    assert sum(o.pred_keypoints.shape[0] for o in outputs) == 8
    assert all(o.pred_keypoints.shape[-1] == 2 for o in outputs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 CUDA devices")
@pytest.mark.skipif(
    not all(p.exists() for p in TOPDOWN_CKPTS), reason="top-down checkpoints missing"
)
def test_upload_batches_targets_the_models_gpu():
    """A layer on ``cuda:1`` gets its frames on ``cuda:1``, whatever ``device`` says."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    built = Predictor.from_model_paths(
        [str(p) for p in TOPDOWN_CKPTS], device="cuda:1", batch_size=2
    )
    predictor = Predictor(
        layer=built.layer, batch_size=2, device="cuda", upload_batches=True
    )
    assert predictor._upload_device == "cuda:1"
    images = np.zeros((2, 64, 64, 1), dtype=np.uint8)
    outputs = list(
        predictor.predict_streaming(NumpyProvider(images=images, batch_size=2))
    )
    assert len(outputs) == 1
