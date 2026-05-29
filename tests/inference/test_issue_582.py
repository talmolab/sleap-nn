"""Regression tests for issue #582 (post-#530 parity/correctness follow-ups).

Covers the five clusters fixed for #582:

* Bottom-up ``max_instances`` (build-time + predict-time override + top-N-by-score
  truncation).
* Multi-video support (``MultiVideoProvider``, writer video memoization,
  ``predict_to_file`` video attribution, out-of-range video-index guard).
* Exported ONNX/TRT parity (centroid anchor resolution, CPU-normalized grouping,
  dtype-preserving helper, ``peak_conf_threshold`` threading).
* Tracking target-count / ``max_tracks`` defaulting + gate tightening.
* GT-centroid / centered-instance minors (sizematcher, user-instance GT filter,
  GT centroid-confidence score).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sleap_io as sio

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor, _select_export_layer
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.streaming import (
    GroupingParams,
    ScoredBatch,
    group_scored_batch,
)

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
BOTTOMUP_CKPT = CKPT_ROOT / "minimal_instance_bottomup"


# ─────────────────────────────────────────────────────────────────────────
# Cluster 1 — bottom-up max_instances (findings 20, 21, 52)
# ─────────────────────────────────────────────────────────────────────────


def _two_instance_scored_batch() -> ScoredBatch:
    """A single-sample ScoredBatch that groups into TWO instances.

    Instance A (peaks near x=0) has LOW peak confidence; instance B (near
    x=100) has HIGH confidence. Their assembly order is independent of score,
    so a correct top-N-by-score truncation keeps B.
    """
    # peaks: [A0, B0, A1, B1]; channels [0, 0, 1, 1].
    cms_peaks = [torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 10.0], [100.0, 10.0]])]
    cms_peak_vals = [torch.tensor([0.3, 0.9, 0.3, 0.9])]
    cms_peak_channel_inds = [torch.tensor([0, 0, 1, 1], dtype=torch.int32)]
    # Candidate connections: A0->A1 (idx 0->2) and B0->B1 (idx 1->3).
    # The PAF instance score is driven by the line score, so give instance B
    # (assembled second) the HIGHER score to prove top-N-by-score reordering.
    edge_inds = [torch.tensor([0, 0], dtype=torch.int32)]
    edge_peak_inds = [torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)]
    line_scores = [torch.tensor([0.40, 0.95])]
    info = PreprocInfo(
        original_size=(128, 128),
        processed_size=(128, 128),
        eff_scale=torch.ones(1),
        input_scale=1.0,
        output_stride=1,
    )
    return ScoredBatch(
        cms_peaks=cms_peaks,
        cms_peak_vals=cms_peak_vals,
        cms_peak_channel_inds=cms_peak_channel_inds,
        edge_inds=edge_inds,
        edge_peak_inds=edge_peak_inds,
        line_scores=line_scores,
        info=info,
        n_samples=1,
        n_nodes=2,
        skip_paf=False,
    )


def _grouping_params(max_instances):
    return GroupingParams(
        paf_scorer_kwargs={
            "part_names": ["n0", "n1"],
            "edges": [("n0", "n1")],
            "pafs_stride": 1,
            "max_edge_length_ratio": 2.0,
            "dist_penalty_weight": 1.0,
            "n_points": 5,
            "min_instance_peaks": 0,
            "min_line_scores": 0.0,
        },
        max_instances=max_instances,
    )


def test_group_scored_batch_uncapped_keeps_both_instances():
    """With no cap, both assembled instances survive."""
    out = group_scored_batch(_two_instance_scored_batch(), _grouping_params(None))
    kpts = out.pred_keypoints[0]  # (max_inst, n_nodes, 2)
    n_real = int((~torch.isnan(kpts).all(dim=(-1, -2))).sum())
    assert n_real == 2


def test_group_scored_batch_truncates_top_n_by_score():
    """max_instances=1 keeps the TOP instance by score, not the first assembled.

    Instance B (peaks near x=100) has higher confidence than A (near x=0), so
    the surviving instance must be B regardless of grouping/assembly order
    (legacy parity — findings 21/52).
    """
    out = group_scored_batch(_two_instance_scored_batch(), _grouping_params(1))
    kpts = out.pred_keypoints[0]  # (1, n_nodes, 2)
    real = kpts[~torch.isnan(kpts).all(dim=(-1, -2))]
    assert real.shape[0] == 1
    # The surviving instance is the high-confidence one (x ~= 100), not x ~= 0.
    assert float(real[0, :, 0].mean()) > 50.0


@pytest.mark.skipif(
    not BOTTOMUP_CKPT.exists(), reason="bottomup checkpoint asset not present"
)
def test_from_model_paths_threads_max_instances_to_bottomup_layer():
    """from_model_paths(max_instances=N) actually reaches the BottomUpLayer.

    Finding 20: the value was dropped because loaders never set it on the
    inference model. It is now carried on LoadedAssets and read by the layer
    builder.
    """
    predictor = Predictor.from_model_paths(
        [str(BOTTOMUP_CKPT)], device="cpu", peak_threshold=0.2, max_instances=1
    )
    assert predictor.layer.max_instances == 1
    # grouping_params snapshots the effective cap.
    assert predictor.layer.grouping_params().max_instances == 1


@pytest.mark.skipif(
    not BOTTOMUP_CKPT.exists(), reason="bottomup checkpoint asset not present"
)
def test_bottomup_predict_time_max_instances_override():
    """A predict-time max_instances override is honored by grouping_params.

    The override is written onto postprocess_config; grouping_params prefers it
    over the build-time value (finding 20, per-call path).
    """
    import attrs

    predictor = Predictor.from_model_paths(
        [str(BOTTOMUP_CKPT)], device="cpu", peak_threshold=0.2
    )
    assert predictor.layer.grouping_params().max_instances is None
    predictor.layer.postprocess_config = attrs.evolve(
        predictor.layer.postprocess_config, max_instances=2
    )
    assert predictor.layer.grouping_params().max_instances == 2


# ─────────────────────────────────────────────────────────────────────────
# Cluster 2 — multi-video (findings 39, 48 + multi-source gap)
# ─────────────────────────────────────────────────────────────────────────


class _StubBackend:
    """Minimal object satisfying the ``ModelBackend`` runtime protocol.

    The GT-peaks centered-instance path never calls the backend, but
    construction validates ``isinstance(backend, ModelBackend)``.
    """

    device = "cpu"
    does_baked_postproc = False

    def __call__(self, x):  # pragma: no cover — never called on the GT path
        return {}

    def warmup(self, input_shape):  # pragma: no cover
        return None


class _StubLayer:
    """Layer-shaped stub returning one constant instance per frame."""

    def predict(self, image, **kwargs) -> Outputs:
        b = int(image.shape[0])
        return Outputs(
            pred_keypoints=torch.zeros(b, 1, 2, 2),
            pred_peak_values=torch.ones(b, 1, 2),
            instance_scores=torch.ones(b, 1) * 0.9,
        )


def test_multivideo_provider_assigns_monotonic_video_indices():
    """MultiVideoProvider re-stamps each source's batches with its ordinal."""
    from sleap_nn.inference.providers import MultiVideoProvider, NumpyProvider

    p0 = NumpyProvider(images=np.zeros((3, 1, 8, 8), dtype=np.float32), batch_size=2)
    p1 = NumpyProvider(images=np.zeros((2, 1, 8, 8), dtype=np.float32), batch_size=2)
    mvp = MultiVideoProvider(providers=[p0, p1])

    seen = []
    for batch in mvp:
        seen.append(np.asarray(batch.video_indices))
    flat = np.concatenate(seen)
    # 3 frames from source 0, then 2 frames from source 1.
    assert flat.tolist() == [0, 0, 0, 1, 1]
    assert len(mvp) == len(p0) + len(p1)


def test_incremental_writer_resolve_videos_is_memoized(tmp_path):
    """_resolve_videos returns the SAME object across calls (finding 48)."""
    from sleap_nn.inference.writer import IncrementalLabelsWriter

    skel = sio.Skeleton(nodes=[sio.Node(name="n0")])
    writer = IncrementalLabelsWriter(path=str(tmp_path / "o.slp"), skeleton=skel)
    assert writer._resolve_videos() is writer._resolve_videos()


def test_incremental_writer_shares_one_video_across_flushes(tmp_path):
    """Per-flush writes share one Video, not N+1 placeholders (finding 48)."""
    from sleap_nn.inference.writer import IncrementalLabelsWriter

    skel = sio.Skeleton(nodes=[sio.Node(name="n0"), sio.Node(name="n1")])
    out_path = tmp_path / "out.slp"
    # write_interval=1 forces a flush per write -> 3 separate to_labels calls.
    writer = IncrementalLabelsWriter(
        path=str(out_path), skeleton=skel, write_interval=1
    )
    with writer:
        for i in range(3):
            writer.write(
                Outputs(
                    pred_keypoints=torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
                    pred_peak_values=torch.tensor([[[0.9, 0.9]]]),
                    frame_indices=torch.tensor([i]),
                    video_indices=torch.tensor([0]),
                )
            )
    labels = sio.load_slp(str(out_path))
    assert len(labels.videos) == 1
    assert len({id(lf.video) for lf in labels.labeled_frames}) == 1
    assert labels.labeled_frames[0].video in labels.videos


def test_predict_to_file_attaches_source_videos(tmp_path, minimal_instance):
    """predict_to_file preserves the real source video (finding 39)."""
    predictor = Predictor(layer=_StubLayer())
    skel = sio.Skeleton(nodes=[sio.Node(name="n0"), sio.Node(name="n1")])
    out_path = tmp_path / "preds.slp"
    predictor.predict_to_file(str(minimal_instance), path=str(out_path), skeleton=skel)
    labels = sio.load_slp(str(out_path))
    assert len(labels.videos) >= 1
    # Not the backend-less 'unknown' placeholder.
    assert all("unknown" not in str(v.filename) for v in labels.videos)


def test_to_labels_out_of_range_video_index_raises():
    """A video index with no matching video raises for genuine multi-video."""
    v0, v1 = sio.Video(filename="a.mp4"), sio.Video(filename="b.mp4")
    out = Outputs(
        pred_keypoints=torch.zeros(1, 1, 2, 2),
        pred_peak_values=torch.ones(1, 1, 2),
        instance_scores=torch.ones(1, 1),
        frame_indices=torch.tensor([0]),
        video_indices=torch.tensor([5]),  # out of range for [v0, v1]
    )
    skel = sio.Skeleton(nodes=[sio.Node(name="n0"), sio.Node(name="n1")])
    with pytest.raises(IndexError, match="out of range"):
        out.to_labels(skeleton=skel, videos=[v0, v1])


# ─────────────────────────────────────────────────────────────────────────
# Cluster 3 — exported ONNX/TRT parity (findings 29, 30, 31, 32)
# ─────────────────────────────────────────────────────────────────────────


def test_export_helper_preserves_dtype():
    """_to_4d_tensor_for_export keeps uint8 as uint8 (finding 32)."""
    from sleap_nn.inference.layers.exported import _to_4d_tensor_for_export

    u = _to_4d_tensor_for_export(np.zeros((16, 16), dtype=np.uint8))
    assert u.dtype == torch.uint8
    assert tuple(u.shape) == (1, 1, 16, 16)
    f = _to_4d_tensor_for_export(np.zeros((16, 16), dtype=np.float32))
    assert f.dtype == torch.float32


def test_raw_to_cpu_moves_tensors():
    """_raw_to_cpu detaches+moves tensors to CPU and passes non-tensors (finding 30)."""
    from sleap_nn.inference.layers.exported import _raw_to_cpu

    out = _raw_to_cpu({"t": torch.zeros(3), "s": "scalar"})
    assert out["t"].device.type == "cpu"
    assert out["s"] == "scalar"


def test_select_export_layer_resolves_centroid_anchor():
    """Exported centroid anchor resolves from metadata.anchor_part (finding 29)."""
    meta = SimpleNamespace(
        model_type="centroid", anchor_part="n1", node_names=["n0", "n1"]
    )
    layer = _select_export_layer(metadata=meta, backend=None, return_confmaps=False)
    assert layer.anchor_ind == 1


def test_select_export_layer_missing_anchor_raises():
    """A configured anchor_part absent from node_names raises (finding 29)."""
    meta = SimpleNamespace(
        model_type="centroid", anchor_part="nope", node_names=["n0", "n1"]
    )
    with pytest.raises(ValueError, match="not found"):
        _select_export_layer(metadata=meta, backend=None, return_confmaps=False)


def test_select_export_layer_no_anchor_part_keeps_node_zero():
    """Old exports without anchor_part keep node-0 behavior (anchor_ind None)."""
    meta = SimpleNamespace(
        model_type="centroid", anchor_part=None, node_names=["n0", "n1"]
    )
    layer = _select_export_layer(metadata=meta, backend=None, return_confmaps=False)
    assert layer.anchor_ind is None


def test_packaging_anchor_ind_reads_exported_centroid():
    """_packaging_anchor_ind honors ExportedCentroidLayer.anchor_ind (finding 29)."""
    from sleap_nn.inference.layers.exported import ExportedCentroidLayer

    p = Predictor(layer=ExportedCentroidLayer(backend=None, anchor_ind=2))
    assert p._packaging_anchor_ind() == 2


def test_select_export_layer_threads_peak_conf_threshold():
    """from_export_dir's resolved peak_conf_threshold reaches the layer (finding 31)."""
    meta = SimpleNamespace(
        model_type="bottomup",
        node_names=["n0", "n1"],
        edge_inds=[(0, 1)],
        max_peaks_per_node=5,
        input_scale=1.0,
        peak_threshold=0.2,
    )
    layer = _select_export_layer(
        metadata=meta, backend=None, return_confmaps=False, peak_conf_threshold=0.5
    )
    assert layer.peak_conf_threshold == 0.5


def _bottomup_export_raw(peak_val: float) -> dict:
    """A minimal exported bottom-up wrapper output: 2 nodes, k=1, 1 edge, 1 cand."""
    return {
        "peaks": torch.tensor([[[[0.0, 0.0]], [[0.0, 10.0]]]]),  # (1,2,1,2)
        "peak_vals": torch.tensor([[[peak_val], [peak_val]]]),  # (1,2,1)
        "peak_mask": torch.ones(1, 2, 1, dtype=torch.bool),
        "line_scores": torch.tensor([[[0.95]]]),  # (1,1,1)
        "candidate_mask": torch.ones(1, 1, 1, dtype=torch.bool),
    }


class _FakeCudaBottomUpBackend:
    """ModelBackend stub that returns the bottom-up wrapper dict on CUDA.

    Reproduces the TensorRTBackend behavior (CUDA-resident outputs) so the
    grouping adapter's CPU/CUDA tensor-mix crash (finding 30) is exercised on
    a GPU host — CPU-only CI cannot see it.
    """

    device = "cuda"
    does_baked_postproc = True

    def __call__(self, x):
        raw = _bottomup_export_raw(0.9)
        return {k: v.to("cuda") for k, v in raw.items()}

    def warmup(self, input_shape):  # pragma: no cover
        return None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA to reproduce the device mix"
)
def test_exported_bottomup_grouping_handles_cuda_backend():
    """Grouping adapter doesn't crash when the backend returns CUDA tensors.

    Finding 30: index/output tensors were built on CPU and indexed by CUDA
    masks. ``_raw_to_cpu`` normalizes to CPU first. Output stays on CPU.
    """
    from sleap_nn.inference.layers.exported import ExportedBottomUpLayer

    layer = ExportedBottomUpLayer(
        backend=_FakeCudaBottomUpBackend(),
        node_names=["n0", "n1"],
        edge_inds=[(0, 1)],
        max_peaks_per_node=1,
    )
    out = layer.predict(np.zeros((1, 1, 32, 32), dtype=np.uint8))
    assert out.pred_keypoints.device.type == "cpu"


def test_exported_bottomup_peak_conf_threshold_gates_candidates():
    """peak_conf_threshold above the peak vals drops the candidate (finding 31)."""
    from sleap_nn.inference.layers.exported import ExportedBottomUpLayer

    info = PreprocInfo(
        original_size=(32, 32),
        processed_size=(32, 32),
        eff_scale=torch.ones(1),
        input_scale=1.0,
        output_stride=1,
    )
    # No threshold -> the candidate edge survives.
    layer_open = ExportedBottomUpLayer(
        backend=None,
        node_names=["n0", "n1"],
        edge_inds=[(0, 1)],
        max_peaks_per_node=1,
    )
    scored_open = layer_open._build_scored_batch(_bottomup_export_raw(0.4), info)
    assert int(scored_open.edge_inds[0].numel()) == 1

    # Threshold above the peak vals -> candidate gated out.
    layer_strict = ExportedBottomUpLayer(
        backend=None,
        node_names=["n0", "n1"],
        edge_inds=[(0, 1)],
        max_peaks_per_node=1,
        peak_conf_threshold=0.5,
    )
    scored_strict = layer_strict._build_scored_batch(_bottomup_export_raw(0.4), info)
    assert int(scored_strict.edge_inds[0].numel()) == 0


# ─────────────────────────────────────────────────────────────────────────
# Cluster 4 — tracking target-count / max_tracks (findings 56, 57, 58 + gap)
# ─────────────────────────────────────────────────────────────────────────


def test_build_tracker_config_defaults_target_from_max_instances():
    """post_connect_single_breaks + max_instances derives the target (finding 56)."""
    from sleap_nn.cli import _build_tracker_config

    cfg = _build_tracker_config(
        {"post_connect_single_breaks": True, "max_instances": 2}
    )
    assert cfg.tracking_target_instance_count == 2
    # post_connect also defaults max_tracks from max_instances (legacy).
    assert cfg.max_tracks == 2


def test_build_tracker_config_pre_cull_defaults_target_from_max_instances():
    """tracking_pre_cull_to_target + max_instances derives the target (finding 57)."""
    from sleap_nn.cli import _build_tracker_config

    cfg = _build_tracker_config({"tracking_pre_cull_to_target": 1, "max_instances": 3})
    assert cfg.tracking_target_instance_count == 3


def test_build_tracker_config_max_tracks_defaults_local_queues():
    """max_tracks with no explicit method defaults to local_queues (gap)."""
    from sleap_nn.cli import _build_tracker_config

    cfg = _build_tracker_config({"max_tracks": 3})
    assert cfg.candidates_method == "local_queues"
    assert cfg.max_tracks == 3


def test_build_tracker_config_explicit_fixed_window_respected():
    """An explicit candidates_method is not overridden by max_tracks (gap)."""
    from sleap_nn.cli import _build_tracker_config

    cfg = _build_tracker_config({"max_tracks": 3, "candidates_method": "fixed_window"})
    assert cfg.candidates_method == "fixed_window"


def test_build_tracker_config_defaults_fixed_window_without_max_tracks():
    """No max_tracks -> default candidates_method stays fixed_window (gap)."""
    from sleap_nn.cli import _build_tracker_config

    cfg = _build_tracker_config({})
    assert cfg.candidates_method == "fixed_window"


def _tracking_labels(skeleton, video, n=3):
    lfs = []
    for i in range(n):
        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
            skeleton=skeleton,
            score=0.9,
        )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, instances=[inst]))
    return sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)


def test_apply_tracking_post_connect_only_max_tracks_raises():
    """max_tracks alone no longer satisfies the post_connect gate (finding 58)."""
    from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

    skel = sio.Skeleton(nodes=["head", "tail"])
    labels = _tracking_labels(skel, sio.Video(filename="d.mp4"))
    cfg = TrackerConfig(post_connect_single_breaks=True, max_tracks=3)
    with pytest.raises(ValueError, match="tracking_target_instance_count"):
        apply_tracking(labels, cfg)


def test_apply_tracking_pre_cull_requires_target():
    """pre_cull without a target raises (legacy parity, finding 57)."""
    from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

    skel = sio.Skeleton(nodes=["head", "tail"])
    labels = _tracking_labels(skel, sio.Video(filename="d.mp4"))
    cfg = TrackerConfig(tracking_pre_cull_to_target=1)
    with pytest.raises(ValueError, match="tracking_target_instance_count"):
        apply_tracking(labels, cfg)


def test_apply_tracking_post_connect_with_target_succeeds():
    """post_connect with an explicit target count runs (positive control)."""
    from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

    skel = sio.Skeleton(nodes=["head", "tail"])
    labels = _tracking_labels(skel, sio.Video(filename="d.mp4"))
    cfg = TrackerConfig(
        post_connect_single_breaks=True, tracking_target_instance_count=2
    )
    out = apply_tracking(labels, cfg)
    assert len(out.labeled_frames) == 3


# ─────────────────────────────────────────────────────────────────────────
# Cluster 5 — GT-centroid / centered-instance minors (findings 14, 16, 43)
# ─────────────────────────────────────────────────────────────────────────


def test_centered_instance_gt_carries_centroid_confidence():
    """use_gt_peaks reports centroid confidence as the instance score (finding 14)."""
    from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer

    layer = CenteredInstanceLayer(
        backend=_StubBackend(), output_stride=1, use_gt_peaks=True
    )
    centroids = torch.tensor([[[0.0, 0.0], [100.0, 0.0]]])  # (1, 2, 2)
    instances = torch.tensor(
        [[[[0.0, 0.0], [1.0, 1.0]], [[100.0, 0.0], [101.0, 1.0]]]]
    )  # (1, 2, 2, 2)
    cvals = torch.tensor([[0.7, 0.4]])
    out = layer.predict(
        crops=None, centroids=centroids, instances=instances, centroid_vals=cvals
    )
    assert torch.allclose(out.instance_scores, cvals)
    assert torch.allclose(out.pred_centroid_values, cvals)


def test_centered_instance_gt_score_falls_back_to_ones():
    """Without centroid_vals the GT score stays all-ones (back-compat, finding 14)."""
    from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer

    layer = CenteredInstanceLayer(
        backend=_StubBackend(), output_stride=1, use_gt_peaks=True
    )
    centroids = torch.tensor([[[0.0, 0.0]]])  # (1, 1, 2)
    instances = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])  # (1, 1, 2, 2)
    out = layer.predict(crops=None, centroids=centroids, instances=instances)
    assert torch.allclose(out.instance_scores, torch.ones(1, 1))


def test_centered_instance_gt_nan_centroid_gets_nan_score():
    """NaN-padded centroid slots get a NaN score, not a spurious value."""
    from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer

    layer = CenteredInstanceLayer(
        backend=_StubBackend(), output_stride=1, use_gt_peaks=True
    )
    centroids = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]]])
    instances = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]], [[5.0, 5.0], [6.0, 6.0]]]])
    cvals = torch.tensor([[0.8, 0.5]])
    out = layer.predict(
        crops=None, centroids=centroids, instances=instances, centroid_vals=cvals
    )
    assert float(out.instance_scores[0, 0]) == pytest.approx(0.8)
    assert torch.isnan(out.instance_scores[0, 1])


def _labels_with_user_and_predicted():
    """Labels with one user-only, one predicted-only, and one mixed frame."""
    skel = sio.Skeleton(nodes=["a", "b"])
    video = sio.Video(filename="d.mp4")
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    user = sio.Instance.from_numpy(points_data=pts, skeleton=skel)
    pred = sio.PredictedInstance.from_numpy(
        points_data=pts + 50, skeleton=skel, score=0.9
    )
    lf_user = sio.LabeledFrame(video=video, frame_idx=0, instances=[user])
    lf_pred = sio.LabeledFrame(video=video, frame_idx=1, instances=[pred])
    lf_mixed = sio.LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[
            sio.Instance.from_numpy(points_data=pts, skeleton=skel),
            sio.PredictedInstance.from_numpy(
                points_data=pts + 50, skeleton=skel, score=0.9
            ),
        ],
    )
    return sio.Labels(
        videos=[video], skeletons=[skel], labeled_frames=[lf_user, lf_pred, lf_mixed]
    )


def test_labels_provider_only_labeled_drops_predicted_only_frames():
    """only_labeled_frames keeps only frames with user instances (finding 43)."""
    from sleap_nn.inference.providers import LabelsProvider

    provider = LabelsProvider(
        labels=_labels_with_user_and_predicted(), only_labeled_frames=True
    )
    assert [lf.frame_idx for lf in provider._labeled_frames] == [0, 2]


def test_labels_provider_frame_instances_excludes_predicted_in_gt_mode():
    """In only_labeled_frames mode, GT excludes PredictedInstances (finding 43)."""
    from sleap_nn.inference.providers import LabelsProvider

    labels = _labels_with_user_and_predicted()
    provider = LabelsProvider(labels=labels, only_labeled_frames=True)
    mixed = [lf for lf in labels.labeled_frames if lf.frame_idx == 2][0]
    # Only the single user instance is exposed as GT (not the predicted one).
    assert len(provider._frame_instances(mixed)) == 1


def test_labels_provider_non_gt_mode_keeps_all_instances():
    """With only_labeled_frames=False every instance is exposed."""
    from sleap_nn.inference.providers import LabelsProvider

    labels = _labels_with_user_and_predicted()
    provider = LabelsProvider(labels=labels, only_labeled_frames=False)
    mixed = [lf for lf in labels.labeled_frames if lf.frame_idx == 2][0]
    assert len(provider._frame_instances(mixed)) == 2
