"""Tests for sleap_nn.export.inference module."""

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sleap_io as sio
import torch

from sleap_nn.export.inference import (
    _bottomup_postprocess_worker,
    _find_training_config_for_predict,
    _predict_bottomup_frames,
    _predict_bottomup_raw,
    _predict_centroid_frames,
    _predict_multiclass_bottomup_frames,
    _predict_multiclass_topdown_combined_frames,
    _predict_single_instance_frames,
    _predict_topdown_frames,
    _prefetch_video_batches,
    _raw_results_to_labeled_frames,
    load_video_batch,
    predict,
)

# Skeleton config format compatible with get_skeleton_from_config / SkeletonYAMLDecoder
_SKELETON_CFG = [
    {
        "name": "test",
        "nodes": [{"name": "head"}, {"name": "thorax"}, {"name": "abdomen"}],
        "edges": [
            {"source": {"name": "head"}, "destination": {"name": "thorax"}},
            {"source": {"name": "thorax"}, "destination": {"name": "abdomen"}},
        ],
    }
]


# =============================================================================
# TestLoadVideoBatch
# =============================================================================


class TestLoadVideoBatch:
    def test_basic_2d_grayscale(self, simple_video):
        """2D grayscale frames become (N, 1, H, W) uint8."""
        # Override __getitem__ to return 2D (H, W) arrays
        simple_video.__getitem__ = MagicMock(
            side_effect=lambda idx: np.zeros((64, 64), dtype=np.uint8)
        )
        result = load_video_batch(simple_video, [0, 1])
        assert result.shape == (2, 1, 64, 64)
        assert result.dtype == np.uint8

    def test_3d_hwc_frames(self, simple_video):
        """3D HWC frames → (N, C, H, W)."""
        simple_video.__getitem__ = MagicMock(
            side_effect=lambda idx: np.zeros((64, 64, 3), dtype=np.uint8)
        )
        result = load_video_batch(simple_video, [0, 1, 2])
        assert result.shape == (3, 3, 64, 64)
        assert result.dtype == np.uint8

    def test_dtype_conversion(self, simple_video):
        """Float input gets cast to uint8."""
        simple_video.__getitem__ = MagicMock(
            side_effect=lambda idx: np.ones((32, 32, 1), dtype=np.float32) * 128.0
        )
        result = load_video_batch(simple_video, [0])
        assert result.dtype == np.uint8
        assert result.shape == (1, 1, 32, 32)
        assert result[0, 0, 0, 0] == 128


# =============================================================================
# TestFindTrainingConfig
# =============================================================================


class TestFindTrainingConfig:
    def test_finds_yaml(self, tmp_path):
        (tmp_path / "training_config.yaml").touch()
        result = _find_training_config_for_predict(tmp_path, "single_instance")
        assert result == tmp_path / "training_config.yaml"

    def test_finds_json(self, tmp_path):
        (tmp_path / "training_config.json").touch()
        result = _find_training_config_for_predict(tmp_path, "single_instance")
        assert result == tmp_path / "training_config.json"

    def test_topdown_priority(self, tmp_path):
        """For topdown, prefers training_config_centered_instance.yaml."""
        (tmp_path / "training_config.yaml").touch()
        (tmp_path / "training_config_centered_instance.yaml").touch()
        result = _find_training_config_for_predict(tmp_path, "topdown")
        assert result.name == "training_config_centered_instance.yaml"

    def test_multiclass_topdown_priority(self, tmp_path):
        """For multi_class_topdown_combined, prefers the specific config."""
        (tmp_path / "training_config.yaml").touch()
        (tmp_path / "training_config_multi_class_topdown.yaml").touch()
        result = _find_training_config_for_predict(
            tmp_path, "multi_class_topdown_combined"
        )
        assert result.name == "training_config_multi_class_topdown.yaml"

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No training_config found"):
            _find_training_config_for_predict(tmp_path, "single_instance")

    def test_falls_back_to_model_type_yaml(self, tmp_path):
        """Falls back to training_config_{model_type}.yaml."""
        (tmp_path / "training_config_bottomup.yaml").touch()
        result = _find_training_config_for_predict(tmp_path, "bottomup")
        assert result.name == "training_config_bottomup.yaml"


# =============================================================================
# TestPredictSingleInstanceFrames
# =============================================================================


class TestPredictSingleInstanceFrames:
    def test_basic(self, simple_skeleton, simple_video):
        """Synthetic peaks produce 1 instance per frame."""
        peaks = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                [[11.0, 21.0], [31.0, 41.0], [51.0, 61.0]],
            ]
        )
        peak_vals = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])
        outputs = {"peaks": peaks, "peak_vals": peak_vals}

        frames = _predict_single_instance_frames(
            outputs, [0, 5], simple_video, simple_skeleton
        )

        assert len(frames) == 2
        assert frames[0].frame_idx == 0
        assert frames[1].frame_idx == 5
        assert len(frames[0].instances) == 1
        inst = frames[0].instances[0]
        assert isinstance(inst, sio.PredictedInstance)
        np.testing.assert_allclose(inst.numpy()[:, :2], peaks[0], atol=1e-5)

    def test_instance_score(self, simple_skeleton, simple_video):
        """Instance score is the mean of valid peak values."""
        peaks = np.array([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])
        peak_vals = np.array([[0.9, 0.6, 0.3]])
        outputs = {"peaks": peaks, "peak_vals": peak_vals}

        frames = _predict_single_instance_frames(
            outputs, [0], simple_video, simple_skeleton
        )
        assert abs(frames[0].instances[0].score - 0.6) < 1e-5

    def test_nan_peaks(self, simple_skeleton, simple_video):
        """All-NaN peaks still produce a frame with score=0.0."""
        peaks = np.full((1, 3, 2), np.nan)
        peak_vals = np.full((1, 3), np.nan)
        outputs = {"peaks": peaks, "peak_vals": peak_vals}

        frames = _predict_single_instance_frames(
            outputs, [0], simple_video, simple_skeleton
        )
        assert len(frames) == 1
        assert frames[0].instances[0].score == 0.0


# =============================================================================
# TestPredictTopdownFrames
# =============================================================================


class TestPredictTopdownFrames:
    def _make_outputs(self, batch=2, max_inst=3, n_nodes=3, n_valid=2):
        """Helper to create valid topdown outputs."""
        peaks = np.random.rand(batch, max_inst, n_nodes, 2).astype(np.float32) * 100
        peak_vals = np.random.rand(batch, max_inst, n_nodes).astype(np.float32)
        centroids = np.random.rand(batch, max_inst, 2).astype(np.float32) * 100
        centroid_vals = np.random.rand(batch, max_inst).astype(np.float32)
        instance_valid = np.zeros((batch, max_inst), dtype=np.float32)
        for b in range(batch):
            instance_valid[b, :n_valid] = 1.0
        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "centroids": centroids,
            "centroid_vals": centroid_vals,
            "instance_valid": instance_valid,
        }

    def test_basic(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(batch=2, n_valid=2)
        frames = _predict_topdown_frames(outputs, [0, 1], simple_video, simple_skeleton)
        assert len(frames) == 2
        for lf in frames:
            assert len(lf.instances) == 2

    def test_invalid_instances_filtered(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(batch=1, max_inst=3, n_valid=1)
        frames = _predict_topdown_frames(outputs, [0], simple_video, simple_skeleton)
        assert len(frames) == 1
        assert len(frames[0].instances) == 1

    def test_max_instances(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(batch=1, max_inst=5, n_valid=5)
        frames = _predict_topdown_frames(
            outputs, [0], simple_video, simple_skeleton, max_instances=2
        )
        assert len(frames[0].instances) == 2

    def test_empty_frame(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(batch=1, max_inst=3, n_valid=0)
        frames = _predict_topdown_frames(outputs, [0], simple_video, simple_skeleton)
        assert len(frames) == 0


# =============================================================================
# TestPredictCentroidFrames
# =============================================================================


class TestPredictCentroidFrames:
    def _make_outputs(self, batch=1, max_inst=3, n_valid=2):
        centroids = np.random.rand(batch, max_inst, 2).astype(np.float32) * 100
        centroid_vals = np.random.rand(batch, max_inst).astype(np.float32)
        instance_valid = np.zeros((batch, max_inst), dtype=np.float32)
        for b in range(batch):
            instance_valid[b, :n_valid] = 1.0
        return {
            "centroids": centroids,
            "centroid_vals": centroid_vals,
            "instance_valid": instance_valid,
        }

    def test_basic(self, simple_skeleton, simple_video):
        """Only anchor node is filled, rest NaN."""
        outputs = self._make_outputs(batch=1, n_valid=2)
        frames = _predict_centroid_frames(
            outputs, [0], simple_video, simple_skeleton, anchor_node_idx=1
        )
        assert len(frames) == 1
        assert len(frames[0].instances) == 2
        inst = frames[0].instances[0]
        pts = inst.numpy()
        # Node 1 (thorax / anchor) should have values, others NaN
        assert not np.isnan(pts[1, 0])
        assert np.isnan(pts[0, 0])  # head
        assert np.isnan(pts[2, 0])  # abdomen

    def test_max_instances(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(batch=1, max_inst=5, n_valid=5)
        frames = _predict_centroid_frames(
            outputs,
            [0],
            simple_video,
            simple_skeleton,
            anchor_node_idx=0,
            max_instances=2,
        )
        assert len(frames[0].instances) == 2

    def test_invalid_filtered(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(batch=1, max_inst=3, n_valid=0)
        frames = _predict_centroid_frames(
            outputs, [0], simple_video, simple_skeleton, anchor_node_idx=0
        )
        assert len(frames) == 0


# =============================================================================
# TestPredictMulticlassTopdownCombinedFrames
# =============================================================================


class TestPredictMulticlassTopdownCombinedFrames:
    def _make_outputs(self, batch=1, max_inst=3, n_nodes=3, n_classes=2, n_valid=2):
        peaks = np.random.rand(batch, max_inst, n_nodes, 2).astype(np.float32) * 100
        peak_vals = np.random.rand(batch, max_inst, n_nodes).astype(np.float32)
        centroids = np.random.rand(batch, max_inst, 2).astype(np.float32) * 100
        centroid_vals = np.random.rand(batch, max_inst).astype(np.float32)
        # Make class logits strongly favor one class per instance
        class_logits = np.zeros((batch, max_inst, n_classes), dtype=np.float32)
        for b in range(batch):
            for i in range(n_valid):
                class_logits[b, i, i % n_classes] = 10.0
        instance_valid = np.zeros((batch, max_inst), dtype=np.float32)
        for b in range(batch):
            instance_valid[b, :n_valid] = 1.0
        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "centroids": centroids,
            "centroid_vals": centroid_vals,
            "class_logits": class_logits,
            "instance_valid": instance_valid,
        }

    def test_basic(self, simple_skeleton, simple_video):
        """Hungarian matching assigns classes, tracks have correct names."""
        class_names = ["female", "male"]
        outputs = self._make_outputs(n_valid=2, n_classes=2)
        frames = _predict_multiclass_topdown_combined_frames(
            outputs, [0], simple_video, simple_skeleton, class_names
        )
        assert len(frames) == 1
        assert len(frames[0].instances) == 2
        track_names = {inst.track.name for inst in frames[0].instances}
        assert track_names == {"female", "male"}

    def test_no_valid_instances(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(n_valid=0)
        frames = _predict_multiclass_topdown_combined_frames(
            outputs, [0], simple_video, simple_skeleton, ["female", "male"]
        )
        assert len(frames) == 0

    def test_max_instances(self, simple_skeleton, simple_video):
        outputs = self._make_outputs(max_inst=4, n_valid=4, n_classes=4)
        frames = _predict_multiclass_topdown_combined_frames(
            outputs,
            [0],
            simple_video,
            simple_skeleton,
            ["a", "b", "c", "d"],
            max_instances=2,
        )
        assert len(frames[0].instances) == 2


# =============================================================================
# TestPredictMulticlassBottomupFrames
# =============================================================================


class TestPredictMulticlassBottomupFrames:
    def _make_outputs(self, batch=1, n_nodes=3, max_peaks=5, n_classes=2, conf=0.9):
        peaks = np.random.rand(batch, n_nodes, max_peaks, 2).astype(np.float32) * 100
        peak_vals = np.full((batch, n_nodes, max_peaks), conf, dtype=np.float32)
        peak_mask = np.zeros((batch, n_nodes, max_peaks), dtype=np.float32)
        # Mark first n_classes peaks as valid for each node
        for n in range(n_nodes):
            peak_mask[0, n, :n_classes] = 1.0
        # Make class probs strongly favour one class per peak
        class_probs = np.zeros((batch, n_nodes, max_peaks, n_classes), dtype=np.float32)
        for n in range(n_nodes):
            for p in range(n_classes):
                class_probs[0, n, p, p] = 0.95
        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "peak_mask": peak_mask,
            "class_probs": class_probs,
        }

    def test_basic(self, simple_skeleton, simple_video):
        class_names = ["female", "male"]
        outputs = self._make_outputs(n_classes=2)
        frames = _predict_multiclass_bottomup_frames(
            outputs, [0], simple_video, simple_skeleton, class_names
        )
        assert len(frames) == 1
        track_names = {inst.track.name for inst in frames[0].instances}
        assert track_names == {"female", "male"}

    def test_peak_conf_threshold(self, simple_skeleton, simple_video):
        """Low-confidence peaks are filtered out."""
        outputs = self._make_outputs(conf=0.05)  # below default threshold=0.2
        frames = _predict_multiclass_bottomup_frames(
            outputs,
            [0],
            simple_video,
            simple_skeleton,
            ["female", "male"],
            peak_conf_threshold=0.2,
        )
        assert len(frames) == 0

    def test_all_nan(self, simple_skeleton, simple_video):
        """No valid peaks → empty result."""
        outputs = self._make_outputs()
        outputs["peak_mask"] = np.zeros_like(outputs["peak_mask"])
        frames = _predict_multiclass_bottomup_frames(
            outputs, [0], simple_video, simple_skeleton, ["female", "male"]
        )
        assert len(frames) == 0


# =============================================================================
# TestPredictBottomupFrames
# =============================================================================


class TestPredictBottomupFrames:
    """Tests using a real PAFScorer with a tiny skeleton."""

    @pytest.fixture
    def paf_setup(self, simple_skeleton):
        """Set up PAFScorer and candidate template for a 3-node, 2-edge skeleton."""
        from sleap_nn.export.utils import build_bottomup_candidate_template
        from sleap_nn.inference.paf_grouping import PAFScorer

        part_names = ["head", "thorax", "abdomen"]
        edges = [("head", "thorax"), ("thorax", "abdomen")]
        paf_scorer = PAFScorer(
            part_names=part_names,
            edges=edges,
            pafs_stride=1,
            max_edge_length_ratio=0.5,
            dist_penalty_weight=1.0,
            n_points=3,
            min_instance_peaks=0,
            min_line_scores=0.0,
        )
        n_nodes = 3
        max_peaks = 2
        edge_inds_tuples = [(0, 1), (1, 2)]
        peak_channel_inds, edge_inds_tensor, edge_peak_inds = (
            build_bottomup_candidate_template(n_nodes, max_peaks, edge_inds_tuples)
        )
        candidate_template = {
            "peak_channel_inds": peak_channel_inds,
            "edge_inds": edge_inds_tensor,
            "edge_peak_inds": edge_peak_inds,
        }
        return paf_scorer, candidate_template, n_nodes, max_peaks

    def _make_outputs(self, n_nodes=3, max_peaks=2, n_edges=2):
        """Create synthetic bottom-up outputs with clear peak structure.

        Places one strong peak per node at distinct locations and constructs
        high line_scores for valid connections between them.
        """
        batch = 1
        peaks = np.full((batch, n_nodes, max_peaks, 2), np.nan, dtype=np.float32)
        peak_vals = np.zeros((batch, n_nodes, max_peaks), dtype=np.float32)

        # Place 1 strong peak per node
        for n in range(n_nodes):
            peaks[0, n, 0] = [10.0 + n * 20.0, 10.0 + n * 20.0]
            peak_vals[0, n, 0] = 0.9

        # line_scores: (batch, n_edges, max_peaks, max_peaks)
        line_scores = np.zeros((batch, n_edges, max_peaks, max_peaks), dtype=np.float32)
        # Set high score for the (0,0) peak pair on each edge
        for e in range(n_edges):
            line_scores[0, e, 0, 0] = 0.9

        # candidate_mask: same shape as line_scores
        candidate_mask = np.zeros_like(line_scores, dtype=np.float32)
        for e in range(n_edges):
            candidate_mask[0, e, 0, 0] = 1.0

        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "line_scores": line_scores,
            "candidate_mask": candidate_mask,
        }

    def test_basic(self, simple_skeleton, simple_video, paf_setup):
        paf_scorer, candidate_template, n_nodes, max_peaks = paf_setup
        outputs = self._make_outputs(n_nodes=n_nodes, max_peaks=max_peaks)
        frames = _predict_bottomup_frames(
            outputs,
            [0],
            simple_video,
            simple_skeleton,
            paf_scorer,
            candidate_template,
            input_scale=1.0,
            peak_conf_threshold=0.1,
        )
        # Should produce at least one frame with instances
        assert len(frames) >= 1
        assert len(frames[0].instances) >= 1

    def test_max_instances(self, simple_skeleton, simple_video, paf_setup):
        paf_scorer, candidate_template, n_nodes, max_peaks = paf_setup
        outputs = self._make_outputs(n_nodes=n_nodes, max_peaks=max_peaks)
        frames = _predict_bottomup_frames(
            outputs,
            [0],
            simple_video,
            simple_skeleton,
            paf_scorer,
            candidate_template,
            input_scale=1.0,
            peak_conf_threshold=0.1,
            max_instances=1,
        )
        if frames:
            assert len(frames[0].instances) <= 1


# =============================================================================
# TestPredictBottomupRaw
# =============================================================================


class TestPredictBottomupRaw:
    @pytest.fixture
    def paf_setup(self):
        from sleap_nn.export.utils import build_bottomup_candidate_template
        from sleap_nn.inference.paf_grouping import PAFScorer

        part_names = ["head", "thorax", "abdomen"]
        edges = [("head", "thorax"), ("thorax", "abdomen")]
        paf_scorer = PAFScorer(
            part_names=part_names,
            edges=edges,
            pafs_stride=1,
            max_edge_length_ratio=0.5,
            dist_penalty_weight=1.0,
            n_points=3,
            min_instance_peaks=0,
            min_line_scores=0.0,
        )
        n_nodes = 3
        max_peaks = 2
        edge_inds_tuples = [(0, 1), (1, 2)]
        peak_channel_inds, edge_inds_tensor, edge_peak_inds = (
            build_bottomup_candidate_template(n_nodes, max_peaks, edge_inds_tuples)
        )
        candidate_template = {
            "peak_channel_inds": peak_channel_inds,
            "edge_inds": edge_inds_tensor,
            "edge_peak_inds": edge_peak_inds,
        }
        return paf_scorer, candidate_template

    def _make_outputs(self, n_nodes=3, max_peaks=2, n_edges=2):
        batch = 1
        peaks = np.full((batch, n_nodes, max_peaks, 2), np.nan, dtype=np.float32)
        peak_vals = np.zeros((batch, n_nodes, max_peaks), dtype=np.float32)
        for n in range(n_nodes):
            peaks[0, n, 0] = [10.0 + n * 20.0, 10.0 + n * 20.0]
            peak_vals[0, n, 0] = 0.9

        line_scores = np.zeros((batch, n_edges, max_peaks, max_peaks), dtype=np.float32)
        for e in range(n_edges):
            line_scores[0, e, 0, 0] = 0.9

        candidate_mask = np.zeros_like(line_scores, dtype=np.float32)
        for e in range(n_edges):
            candidate_mask[0, e, 0, 0] = 1.0

        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "line_scores": line_scores,
            "candidate_mask": candidate_mask,
        }

    def test_returns_raw_dicts(self, paf_setup):
        paf_scorer, candidate_template = paf_setup
        outputs = self._make_outputs()
        results = _predict_bottomup_raw(
            outputs,
            [0],
            paf_scorer,
            candidate_template,
            input_scale=1.0,
            peak_conf_threshold=0.1,
        )
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "frame_idx" in r
            assert "instance_peaks" in r
            assert "instance_peak_scores" in r
            assert "instance_scores" in r
            assert r["instance_peaks"].ndim == 3  # (n_inst, n_nodes, 2)

    def test_max_instances(self, paf_setup):
        paf_scorer, candidate_template = paf_setup
        outputs = self._make_outputs()
        results = _predict_bottomup_raw(
            outputs,
            [0],
            paf_scorer,
            candidate_template,
            input_scale=1.0,
            peak_conf_threshold=0.1,
            max_instances=1,
        )
        if results:
            assert len(results[0]["instance_scores"]) <= 1


# =============================================================================
# TestRawResultsToLabeledFrames
# =============================================================================


class TestRawResultsToLabeledFrames:
    def test_basic(self, simple_skeleton, simple_video):
        raw = [
            {
                "frame_idx": 0,
                "instance_peaks": np.array(
                    [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
                ),
                "instance_peak_scores": np.array([[0.9, 0.8, 0.7]]),
                "instance_scores": np.array([0.85]),
            },
            {
                "frame_idx": 5,
                "instance_peaks": np.array(
                    [[[11.0, 21.0], [31.0, 41.0], [51.0, 61.0]]]
                ),
                "instance_peak_scores": np.array([[0.6, 0.5, 0.4]]),
                "instance_scores": np.array([0.5]),
            },
        ]
        frames = _raw_results_to_labeled_frames(raw, simple_video, simple_skeleton)
        assert len(frames) == 2
        assert frames[0].frame_idx == 0
        assert frames[1].frame_idx == 5
        assert len(frames[0].instances) == 1
        assert isinstance(frames[0].instances[0], sio.PredictedInstance)

    def test_empty_input(self, simple_skeleton, simple_video):
        frames = _raw_results_to_labeled_frames([], simple_video, simple_skeleton)
        assert frames == []


# =============================================================================
# TestPrefetchVideoBatches
# =============================================================================


class TestPrefetchVideoBatches:
    def test_batches_queued(self, simple_video):
        q = queue.Queue(maxsize=10)
        frame_indices = list(range(10))
        _prefetch_video_batches(
            simple_video, frame_indices, batch_size=4, prefetch_queue=q
        )

        batches = []
        while True:
            item = q.get()
            if item is None:
                break
            batches.append(item)

        # 10 frames with batch_size=4 → 3 batches (4, 4, 2)
        assert len(batches) == 3
        assert batches[0][0].shape[0] == 4
        assert batches[2][0].shape[0] == 2


# =============================================================================
# TestBottomupPostprocessWorker
# =============================================================================


class TestBottomupPostprocessWorker:
    def test_processes_items(self):
        """Run the worker in-process with queues to verify it processes items."""
        from sleap_nn.export.utils import build_bottomup_candidate_template

        part_names = ["head", "thorax", "abdomen"]
        edges = [("head", "thorax"), ("thorax", "abdomen")]
        paf_scorer_kwargs = {
            "part_names": part_names,
            "edges": edges,
            "pafs_stride": 1,
            "max_edge_length_ratio": 0.5,
            "dist_penalty_weight": 1.0,
            "n_points": 3,
            "min_instance_peaks": 0,
            "min_line_scores": 0.0,
        }
        n_nodes = 3
        max_peaks = 2
        edge_inds_tuples = [(0, 1), (1, 2)]
        peak_channel_inds, edge_inds_tensor, edge_peak_inds = (
            build_bottomup_candidate_template(n_nodes, max_peaks, edge_inds_tuples)
        )
        candidate_template_data = {
            "peak_channel_inds": peak_channel_inds.numpy(),
            "edge_inds": edge_inds_tensor.numpy(),
            "edge_peak_inds": edge_peak_inds.numpy(),
        }

        # Build a small synthetic batch
        n_edges = 2
        peaks = np.full((1, n_nodes, max_peaks, 2), np.nan, dtype=np.float32)
        peak_vals = np.zeros((1, n_nodes, max_peaks), dtype=np.float32)
        for n in range(n_nodes):
            peaks[0, n, 0] = [10.0 + n * 20.0, 10.0 + n * 20.0]
            peak_vals[0, n, 0] = 0.9
        line_scores = np.zeros((1, n_edges, max_peaks, max_peaks), dtype=np.float32)
        candidate_mask = np.zeros_like(line_scores, dtype=np.float32)
        for e in range(n_edges):
            line_scores[0, e, 0, 0] = 0.9
            candidate_mask[0, e, 0, 0] = 1.0

        outputs = {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "line_scores": line_scores,
            "candidate_mask": candidate_mask,
        }

        gpu_output_queue = queue.Queue()
        result_queue = queue.Queue()

        gpu_output_queue.put((0, outputs, [0]))
        gpu_output_queue.put(None)  # sentinel

        _bottomup_postprocess_worker(
            gpu_output_queue,
            result_queue,
            paf_scorer_kwargs,
            candidate_template_data,
            input_scale=1.0,
            peak_conf_threshold=0.1,
            max_instances=None,
        )

        assert not result_queue.empty()
        seq_id, results = result_queue.get()
        assert seq_id == 0
        assert isinstance(results, list)


# =============================================================================
# TestPredict (integration, heavily mocked)
# =============================================================================


class TestPredict:
    @pytest.fixture
    def mock_export_dir(self, tmp_path):
        """Set up a fake export directory with metadata and model files."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        # Create a fake model.onnx
        (export_dir / "model.onnx").write_bytes(b"fake")

        # Create a training config
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"data_config": {"skeletons": _SKELETON_CFG}})
        OmegaConf.save(cfg, str(export_dir / "training_config.yaml"))

        return export_dir

    def _save_metadata(self, export_dir, model_type="single_instance", **overrides):
        from tests.export.conftest import _make_export_metadata

        meta = _make_export_metadata(model_type=model_type, **overrides)
        meta.save(export_dir / "export_metadata.json")
        return meta

    def _mock_predictor_outputs(self, model_type, n_frames, batch_size):
        """Return a callable that produces correct-shaped outputs for model_type."""
        n_nodes = 3
        max_inst = 10

        def predict_fn(batch):
            b = batch.shape[0]
            if model_type == "single_instance":
                return {
                    "peaks": np.random.rand(b, n_nodes, 2).astype(np.float32),
                    "peak_vals": np.random.rand(b, n_nodes).astype(np.float32),
                }
            elif model_type == "topdown":
                return {
                    "peaks": np.random.rand(b, max_inst, n_nodes, 2).astype(np.float32),
                    "peak_vals": np.random.rand(b, max_inst, n_nodes).astype(
                        np.float32
                    ),
                    "centroids": np.random.rand(b, max_inst, 2).astype(np.float32),
                    "centroid_vals": np.random.rand(b, max_inst).astype(np.float32),
                    "instance_valid": np.ones((b, max_inst), dtype=np.float32),
                }
            elif model_type == "centroid":
                return {
                    "centroids": np.random.rand(b, max_inst, 2).astype(np.float32),
                    "centroid_vals": np.random.rand(b, max_inst).astype(np.float32),
                    "instance_valid": np.ones((b, max_inst), dtype=np.float32),
                }
            return {}

        return predict_fn

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_single_instance(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        self._save_metadata(mock_export_dir, "single_instance")
        mock_video_cls.return_value = simple_video
        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(
            side_effect=self._mock_predictor_outputs("single_instance", 10, 4)
        )
        mock_load_model.return_value = mock_predictor

        labels, stats = predict(
            export_dir=mock_export_dir,
            video_path="/fake/video.mp4",
            runtime="onnx",
            batch_size=4,
            n_frames=4,
        )

        assert isinstance(labels, sio.Labels)
        assert "total_time" in stats
        assert "fps" in stats
        assert len(labels.labeled_frames) == 4

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_topdown(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        self._save_metadata(mock_export_dir, "topdown")
        mock_video_cls.return_value = simple_video
        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(
            side_effect=self._mock_predictor_outputs("topdown", 10, 4)
        )
        mock_load_model.return_value = mock_predictor

        # Need centered_instance config for topdown
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"data_config": {"skeletons": _SKELETON_CFG}})
        OmegaConf.save(
            cfg, str(mock_export_dir / "training_config_centered_instance.yaml")
        )

        labels, stats = predict(
            export_dir=mock_export_dir,
            video_path="/fake/video.mp4",
            runtime="onnx",
            batch_size=4,
            n_frames=4,
        )

        assert isinstance(labels, sio.Labels)
        assert len(labels.labeled_frames) > 0

    def test_missing_metadata(self, tmp_path):
        export_dir = tmp_path / "empty_export"
        export_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            predict(export_dir=export_dir, video_path="/fake/video.mp4")

    def test_missing_model(self, mock_export_dir):
        """No model file at all → FileNotFoundError."""
        self._save_metadata(mock_export_dir, "single_instance")
        # Remove the fake model file
        (mock_export_dir / "model.onnx").unlink()
        with pytest.raises(FileNotFoundError, match="No model found"):
            predict(export_dir=mock_export_dir, video_path="/fake/video.mp4")

    def test_unknown_runtime(self, mock_export_dir):
        self._save_metadata(mock_export_dir, "single_instance")
        with pytest.raises(ValueError, match="Unknown runtime"):
            predict(
                export_dir=mock_export_dir,
                video_path="/fake/video.mp4",
                runtime="xla",
            )

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_unsupported_model_type(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        self._save_metadata(mock_export_dir, "nonexistent_type")
        mock_video_cls.return_value = simple_video
        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(return_value={})
        mock_load_model.return_value = mock_predictor

        with pytest.raises(ValueError, match="Unsupported model_type"):
            predict(
                export_dir=mock_export_dir,
                video_path="/fake/video.mp4",
                runtime="onnx",
                batch_size=4,
                n_frames=1,
            )

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_centroid(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        self._save_metadata(mock_export_dir, "centroid")
        mock_video_cls.return_value = simple_video
        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(
            side_effect=self._mock_predictor_outputs("centroid", 10, 4)
        )
        mock_load_model.return_value = mock_predictor

        # Need config with centroid head for anchor_part
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "data_config": {"skeletons": _SKELETON_CFG},
                "model_config": {
                    "head_configs": {
                        "centroid": {"confmaps": {"anchor_part": "thorax"}}
                    }
                },
            }
        )
        OmegaConf.save(cfg, str(mock_export_dir / "training_config.yaml"))

        labels, stats = predict(
            export_dir=mock_export_dir,
            video_path="/fake/video.mp4",
            runtime="onnx",
            batch_size=4,
            n_frames=4,
        )
        assert isinstance(labels, sio.Labels)
        assert len(labels.labeled_frames) > 0

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_progress_callback(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        """Progress callback is invoked with correct (processed, total) args."""
        self._save_metadata(mock_export_dir, "single_instance")
        mock_video_cls.return_value = simple_video
        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(
            side_effect=self._mock_predictor_outputs("single_instance", 10, 4)
        )
        mock_load_model.return_value = mock_predictor

        progress_calls = []

        def callback(processed, total):
            progress_calls.append((processed, total))

        predict(
            export_dir=mock_export_dir,
            video_path="/fake/video.mp4",
            runtime="onnx",
            batch_size=4,
            n_frames=8,
            progress_callback=callback,
        )
        assert len(progress_calls) > 0
        # Last call should report total frames processed
        assert progress_calls[-1][1] == 8

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_multiclass_bottomup(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        self._save_metadata(
            mock_export_dir,
            "multi_class_bottomup",
            n_classes=2,
            class_names=["female", "male"],
        )
        mock_video_cls.return_value = simple_video
        n_nodes = 3
        max_peaks = 5
        n_classes = 2

        def predict_fn(batch):
            b = batch.shape[0]
            peaks = np.random.rand(b, n_nodes, max_peaks, 2).astype(np.float32)
            peak_vals = np.full((b, n_nodes, max_peaks), 0.9, dtype=np.float32)
            peak_mask = np.zeros((b, n_nodes, max_peaks), dtype=np.float32)
            for bn in range(b):
                for n in range(n_nodes):
                    peak_mask[bn, n, :n_classes] = 1.0
            class_probs = np.zeros((b, n_nodes, max_peaks, n_classes), dtype=np.float32)
            for bn in range(b):
                for n in range(n_nodes):
                    for p in range(n_classes):
                        class_probs[bn, n, p, p] = 0.95
            return {
                "peaks": peaks,
                "peak_vals": peak_vals,
                "peak_mask": peak_mask,
                "class_probs": class_probs,
            }

        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(side_effect=predict_fn)
        mock_load_model.return_value = mock_predictor

        labels, stats = predict(
            export_dir=mock_export_dir,
            video_path="/fake/video.mp4",
            runtime="onnx",
            batch_size=4,
            n_frames=4,
        )
        assert isinstance(labels, sio.Labels)
        assert len(labels.labeled_frames) > 0

    @patch("sleap_nn.export.inference.load_exported_model")
    @patch("sleap_nn.export.inference.sio.Video.from_filename")
    def test_multiclass_topdown_combined(
        self, mock_video_cls, mock_load_model, mock_export_dir, simple_video
    ):
        self._save_metadata(
            mock_export_dir,
            "multi_class_topdown_combined",
            n_classes=2,
            class_names=["female", "male"],
        )
        mock_video_cls.return_value = simple_video
        n_nodes = 3
        max_inst = 10
        n_classes = 2

        def predict_fn(batch):
            b = batch.shape[0]
            return {
                "peaks": np.random.rand(b, max_inst, n_nodes, 2).astype(np.float32),
                "peak_vals": np.random.rand(b, max_inst, n_nodes).astype(np.float32),
                "centroids": np.random.rand(b, max_inst, 2).astype(np.float32),
                "centroid_vals": np.random.rand(b, max_inst).astype(np.float32),
                "class_logits": np.random.rand(b, max_inst, n_classes).astype(
                    np.float32
                ),
                "instance_valid": np.ones((b, max_inst), dtype=np.float32),
            }

        mock_predictor = MagicMock()
        mock_predictor.predict = MagicMock(side_effect=predict_fn)
        mock_load_model.return_value = mock_predictor

        # Provide multiclass topdown config
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"data_config": {"skeletons": _SKELETON_CFG}})
        OmegaConf.save(
            cfg,
            str(mock_export_dir / "training_config_multi_class_topdown.yaml"),
        )

        labels, stats = predict(
            export_dir=mock_export_dir,
            video_path="/fake/video.mp4",
            runtime="onnx",
            batch_size=4,
            n_frames=2,
        )
        assert isinstance(labels, sio.Labels)
        assert len(labels.labeled_frames) > 0
