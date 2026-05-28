"""End-to-end integration tests: real fixture ckpt → VideoProvider → Outputs.

These tests run the **full** ``Predictor.from_model_paths(...).predict_streaming(
VideoProvider(small_robot.mp4))`` pipeline on every supported model type, on
both CPU and (when available) MPS.

Why these exist (PR 26): the CUDA benchmark surfaced device-mismatch bugs that
the existing test suite missed entirely. Those tests either (a) used
``_StubLayer`` instead of a real backend, (b) used ``NumpyProvider`` with
synthetic frames, or (c) mocked the factory. None of them exercised the actual
video → preprocess → backend forward → postprocess → Outputs chain on a real
fixture. The fix was to allocate output buffers on the model's device instead
of always-CPU (`torch.full(..., device=...)`); without these tests, that
anti-pattern can creep back in silently and only fail on non-CPU devices.

Run cost: ~10-30s per model type on CPU, similar on MPS (Mac M-series).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import sleap_io as sio

from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.providers import VideoProvider

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
VIDEO = Path(__file__).resolve().parents[1] / "assets" / "datasets" / "small_robot.mp4"


def _ckpts_for(model_type: str) -> list[Path]:
    """Map a logical model_type label to its fixture path(s)."""
    mapping = {
        "single_instance": [CKPT_ROOT / "minimal_instance_single_instance"],
        "centroid_only": [CKPT_ROOT / "minimal_instance_centroid"],
        "topdown": [
            CKPT_ROOT / "minimal_instance_centroid",
            CKPT_ROOT / "minimal_instance_centered_instance",
        ],
        "bottomup": [CKPT_ROOT / "minimal_instance_bottomup"],
        "multiclass_bottomup": [CKPT_ROOT / "minimal_instance_multiclass_bottomup"],
    }
    return mapping[model_type]


def _have_fixtures(model_type: str) -> bool:
    return VIDEO.exists() and all(p.exists() for p in _ckpts_for(model_type))


MODEL_TYPES = [
    "single_instance",
    "centroid_only",
    "topdown",
    "bottomup",
    "multiclass_bottomup",
]


# ──────────────────────────────────────────────────────────────────────
# CPU end-to-end
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_predict_streaming_cpu(model_type):
    """Each fixture model_type runs end-to-end against small_robot.mp4 on CPU."""
    if not _have_fixtures(model_type):
        pytest.skip(f"missing fixtures for {model_type}")

    video = sio.load_video(str(VIDEO))
    n_frames = 8  # keep small — this is a correctness check, not a perf bench
    predictor = Predictor.from_model_paths(
        [str(p) for p in _ckpts_for(model_type)], device="cpu", batch_size=4
    )
    provider = VideoProvider(video=video, batch_size=4, frames=list(range(n_frames)))
    outputs = list(predictor.predict_streaming(provider))
    assert outputs, f"no batches yielded for {model_type}"

    # At least one of pred_keypoints / pred_centroids must be populated, on the
    # right device (cpu in this test).
    first = outputs[0]
    assert (
        first.pred_keypoints is not None or first.pred_centroids is not None
    ), f"{model_type}: neither pred_keypoints nor pred_centroids set"
    for field in ("pred_keypoints", "pred_centroids"):
        t = getattr(first, field)
        if t is not None:
            assert (
                t.device.type == "cpu"
            ), f"{model_type}: {field} ended up on {t.device}, expected cpu"


# ──────────────────────────────────────────────────────────────────────
# MPS end-to-end (gated)
# ──────────────────────────────────────────────────────────────────────


_HAS_MPS = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
    and torch.backends.mps.is_built()
)


@pytest.mark.skipif(not _HAS_MPS, reason="MPS not available")
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_predict_streaming_mps(model_type):
    """Each fixture model_type runs end-to-end against small_robot.mp4 on MPS.

    Regression guard: PR 26 fixed several layers that allocated output buffers
    on CPU regardless of model device. Pre-fix, this test failed for the
    ``topdown`` case (scatter from mps:0 into a cpu buffer raised
    ``RuntimeError: Expected all tensors to be on the same device``).
    """
    if not _have_fixtures(model_type):
        pytest.skip(f"missing fixtures for {model_type}")

    video = sio.load_video(str(VIDEO))
    n_frames = 8
    predictor = Predictor.from_model_paths(
        [str(p) for p in _ckpts_for(model_type)], device="mps", batch_size=4
    )
    provider = VideoProvider(video=video, batch_size=4, frames=list(range(n_frames)))
    outputs = list(predictor.predict_streaming(provider))
    assert outputs, f"no batches yielded for {model_type} on MPS"
