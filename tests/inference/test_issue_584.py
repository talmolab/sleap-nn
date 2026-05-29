"""Regression tests for issue #584 (coverage + minor-correctness follow-ups).

Locks the minor-correctness fixes and fills the higher-value coverage gaps that
don't already have a dedicated file (run.predict -> test_run.py; ONNX
_select_providers -> test_select_providers.py; legacy loader -> test_loaders.py):

* threshold 0.0-as-unset (#40), Outputs.slim() CPU contract (#47),
  PreprocessConfig ensure_rgb/ensure_grayscale validator (#9),
  find_points_mean per-axis NaN (#12), skip_paf pred_paf_graph (#51),
  Predictor.retrack wrapper, crop far-OOB behavior is in test_crops.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import sleap_io as sio

from sleap_nn.data.instance_centroids import find_points_mean
from sleap_nn.inference.layers.configs import PreprocessConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.streaming import GroupingParams, ScoredBatch, group_scored_batch

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"


# ─────────────────────────────────────────────────────────────────────────
# #9 — PreprocessConfig rejects ensure_rgb + ensure_grayscale both True
# ─────────────────────────────────────────────────────────────────────────


def test_preprocess_config_rejects_both_rgb_and_grayscale():
    """ensure_rgb=True + ensure_grayscale=True is a construction-time error (#584)."""
    with pytest.raises(ValueError, match="cannot both be True"):
        PreprocessConfig(ensure_rgb=True, ensure_grayscale=True)
    # Single flags + neither are fine.
    PreprocessConfig(ensure_rgb=True)
    PreprocessConfig(ensure_grayscale=True)
    PreprocessConfig()


# ─────────────────────────────────────────────────────────────────────────
# #12 — find_points_mean counts non-NaN PER AXIS
# ─────────────────────────────────────────────────────────────────────────


def test_find_points_mean_counts_per_axis():
    """A single-axis NaN must not drag that axis's mean toward 0 (#584)."""
    # p0=(0,4), p1=(10, NaN). x: both present -> mean 5; y: only p0 -> mean 4.
    pts = torch.tensor([[0.0, 4.0], [10.0, float("nan")]])
    out = find_points_mean(pts)
    torch.testing.assert_close(out, torch.tensor([5.0, 4.0]))
    # All-NaN slot stays NaN.
    allnan = find_points_mean(
        torch.tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]])
    )
    assert torch.isnan(allnan).all()


# ─────────────────────────────────────────────────────────────────────────
# #47 — Outputs.slim()/cpu() move PreprocInfo's nested tensors to CPU
# ─────────────────────────────────────────────────────────────────────────


def _outputs_with_info(device: str) -> Outputs:
    info = PreprocInfo(
        eff_scale=torch.tensor([1.0], device=device),
        crop_offsets=torch.zeros(1, 2, device=device),
    )
    return Outputs(
        pred_keypoints=torch.zeros(1, 1, 2, 2, device=device),
        pred_peak_values=torch.ones(1, 1, 2, device=device),
        preprocess_info=info,
    )


def test_slim_moves_preprocess_info_to_cpu():
    """slim() returns a PreprocInfo whose nested tensors are CPU (#584)."""
    slim = _outputs_with_info("cpu").slim()
    assert isinstance(slim.preprocess_info, PreprocInfo)
    assert slim.preprocess_info.eff_scale.device.type == "cpu"
    assert slim.preprocess_info.crop_offsets.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_slim_moves_cuda_preprocess_info_to_cpu():
    """The leak case: a CUDA eff_scale/crop_offsets must land on CPU after slim."""
    slim = _outputs_with_info("cuda").slim()
    assert slim.preprocess_info.eff_scale.device.type == "cpu"
    assert slim.preprocess_info.crop_offsets.device.type == "cpu"
    # numpy() keeps a PreprocInfo but moves it to CPU too.
    npd = _outputs_with_info("cuda").numpy()
    assert npd["preprocess_info"].eff_scale.device.type == "cpu"


# ─────────────────────────────────────────────────────────────────────────
# #51 — skip_paf still emits pred_paf_graph when requested
# ─────────────────────────────────────────────────────────────────────────


def test_skip_paf_emits_pred_paf_graph():
    """The skip_paf short-circuit honors return_paf_graph (legacy parity, #584)."""
    info = PreprocInfo(
        original_size=(64, 64),
        processed_size=(64, 64),
        eff_scale=torch.ones(1),
        input_scale=1.0,
        output_stride=1,
    )
    scored = ScoredBatch(
        cms_peaks=[torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
        cms_peak_vals=[torch.tensor([0.9, 0.8])],
        cms_peak_channel_inds=[torch.tensor([0, 1], dtype=torch.int32)],
        edge_inds=[],  # skip path: no scored edges
        edge_peak_inds=[],
        line_scores=[],
        info=info,
        n_samples=1,
        n_nodes=2,
        skip_paf=True,
    )
    out = group_scored_batch(
        scored, GroupingParams(paf_scorer_kwargs={}, return_paf_graph=True)
    )
    assert out.pred_paf_graph is not None
    peaks, edge_inds, edge_peak_inds, line_scores = out.pred_paf_graph
    assert peaks.shape == (2, 2)  # the found peaks survive
    assert edge_inds.numel() == 0 and line_scores.numel() == 0  # no edges on skip
    # Without the flag it stays None.
    out2 = group_scored_batch(scored, GroupingParams(paf_scorer_kwargs={}))
    assert out2.pred_paf_graph is None


# ─────────────────────────────────────────────────────────────────────────
# Predictor.retrack wrapper (coverage gap)
# ─────────────────────────────────────────────────────────────────────────


def _predicted_labels(skeleton, video, frames, empty_last=False):
    pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    lfs = []
    for i in range(frames):
        insts = []
        if not (empty_last and i == frames - 1):
            insts = [
                sio.PredictedInstance.from_numpy(
                    points_data=pts + i, skeleton=skeleton, score=0.9
                )
            ]
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, instances=insts))
    return sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)


def test_retrack_assigns_tracks():
    """Predictor.retrack runs the tracker on existing predictions (#584)."""
    from sleap_nn.inference.tracking import TrackerConfig

    skel = sio.Skeleton(nodes=["a", "b"])
    labels = _predicted_labels(skel, sio.Video(filename="d.mp4"), frames=3)
    out = Predictor.retrack(labels, TrackerConfig(candidates_method="fixed_window"))
    assert len(out.labeled_frames) == 3
    assert all(
        inst.track is not None for lf in out.labeled_frames for inst in lf.instances
    )


def test_retrack_clean_empty_frames_drops_empty():
    """clean_empty_frames removes frames with no instances after tracking (#584)."""
    from sleap_nn.inference.tracking import TrackerConfig

    skel = sio.Skeleton(nodes=["a", "b"])
    labels = _predicted_labels(
        skel, sio.Video(filename="d.mp4"), frames=3, empty_last=True
    )
    out = Predictor.retrack(
        labels,
        TrackerConfig(candidates_method="fixed_window"),
        clean_empty_frames=True,
    )
    assert all(len(lf.instances) > 0 for lf in out.labeled_frames)
    assert len(out.labeled_frames) == 2


# ─────────────────────────────────────────────────────────────────────────
# #40 — per-stage threshold 0.0 override is honored (top-down)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and CENTERED_CKPT.exists()),
    reason="top-down checkpoints not present",
)
def test_topdown_centroid_threshold_zero_override_is_honored():
    """centroid_threshold=0.0 reaches the centroid stage (not swallowed by `or`)."""
    predictor = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)], device="cpu", peak_threshold=0.2
    )
    centroid_layer = predictor.layer.centroid_layer
    with predictor._postprocess_overrides(centroid_threshold=0.0):
        assert centroid_layer.postprocess_config.peak_threshold == 0.0
    # Restored afterwards.
    assert centroid_layer.postprocess_config.peak_threshold == 0.2
