"""End-to-end accuracy tests comparing PyTorch and ONNX/TensorRT inference paths.

These tests export the ``minimal_instance_bottomup`` checkpoint to ONNX (and
optionally TensorRT), run inference on the same video frames with both the
native PyTorch predictor and the exported model, then compare predictions
using the evaluation utilities in ``sleap_nn.evaluation``.

The goal is to catch systematic bugs like wrong coordinate scaling, broken PAF
scoring, or missing normalization that could silently ship in the export
pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner

from tests.export.conftest import (
    requires_onnx,
    requires_onnxruntime,
    requires_gpu,
    requires_tensorrt,
)

# ---------------------------------------------------------------------------
# Paths (resolved once, reused across session-scoped fixtures)
# ---------------------------------------------------------------------------

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_CKPTS = _ASSETS / "model_ckpts"
_BOTTOMUP_CKPT = _CKPTS / "minimal_instance_bottomup"
_SINGLE_INSTANCE_CKPT = _CKPTS / "minimal_instance_single_instance"
_CENTROID_CKPT = _CKPTS / "minimal_instance_centroid"
_CENTERED_INSTANCE_CKPT = _CKPTS / "minimal_instance_centered_instance"
_MULTICLASS_BOTTOMUP_CKPT = _CKPTS / "minimal_instance_multiclass_bottomup"
_MULTICLASS_CI_CKPT = _CKPTS / "minimal_instance_multiclass_centered_instance"
_VIDEO_1CH = _ASSETS / "datasets" / "centered_pair_small.mp4"  # grayscale
_VIDEO_3CH = _ASSETS / "datasets" / "small_robot.mp4"  # RGB
_VIDEO = _VIDEO_1CH  # legacy alias for bottom-up tests
_SLP = _ASSETS / "datasets" / "minimal_instance.pkg.slp"

# Inference parameters shared between PyTorch and ONNX paths
_N_FRAMES = 10
_PEAK_THRESHOLD = 0.05

# Baseline measured over 3 runs (PyTorch vs ONNX with integral_refinement=None
# are numerically identical on this checkpoint):
#   - All keypoint distances: 0.00 px (p50, p95, p99, max)
#   - Instance counts: 2 per frame, 20 total across 10 frames
#   - Instance count deviation: 0.0
# Thresholds are set at baseline + 3 px to flag regressions.
_BASELINE_DIST_PX = 0.0
_WARN_ABOVE_BASELINE_PX = 3.0
_BASELINE_TOTAL_INSTANCES = 20  # 2 instances * 10 frames


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bottomup_ckpt_path():
    """Path to the minimal bottom-up checkpoint directory."""
    assert _BOTTOMUP_CKPT.exists(), f"Missing checkpoint: {_BOTTOMUP_CKPT}"
    return _BOTTOMUP_CKPT


@pytest.fixture(scope="session")
def video_path():
    """Path to the test video."""
    assert _VIDEO.exists(), f"Missing video: {_VIDEO}"
    return _VIDEO


@pytest.fixture(scope="session")
def exported_bottomup_onnx_dir(bottomup_ckpt_path, tmp_path_factory):
    """Export the bottom-up checkpoint to ONNX and return the export directory."""
    pytest.importorskip("onnx")

    from sleap_nn.export.cli import export_model

    export_dir = tmp_path_factory.mktemp("export_bottomup_onnx")
    runner = CliRunner()
    result = runner.invoke(
        export_model,
        [
            str(bottomup_ckpt_path),
            "-o",
            str(export_dir),
            "--format",
            "onnx",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, f"Export failed:\n{result.output}\n{result.exception}"
    assert (export_dir / "model.onnx").exists()
    assert (export_dir / "export_metadata.json").exists()
    return export_dir


@pytest.fixture(scope="session")
def pytorch_bottomup_labels(bottomup_ckpt_path, video_path):
    """Run PyTorch inference on the test video and return Labels."""
    from sleap_nn.predict import run_inference

    labels = run_inference(
        data_path=str(video_path),
        model_paths=[str(bottomup_ckpt_path)],
        peak_threshold=_PEAK_THRESHOLD,
        integral_refinement=None,  # match ONNX argmax-based detection
        device="cpu",
        frames=list(range(_N_FRAMES)),
        make_labels=True,
    )
    assert isinstance(labels, sio.Labels)
    return labels


@pytest.fixture(scope="session")
def onnx_bottomup_labels(exported_bottomup_onnx_dir, video_path):
    """Run ONNX inference on the test video and return Labels."""
    pytest.importorskip("onnxruntime")

    from omegaconf import OmegaConf

    from sleap_nn.export.cli import _find_training_config_for_predict
    from sleap_nn.export.metadata import ExportMetadata
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import VideoProvider
    from sleap_nn.inference.utils import get_skeleton_from_config

    metadata = ExportMetadata.load(exported_bottomup_onnx_dir / "export_metadata.json")
    cfg_path = _find_training_config_for_predict(
        exported_bottomup_onnx_dir, metadata.model_type
    )
    cfg = OmegaConf.load(cfg_path.as_posix())
    skeleton = get_skeleton_from_config(cfg.data_config.skeletons)[0]

    sio_video = sio.Video.from_filename(str(video_path))
    n_total = min(_N_FRAMES, len(sio_video))
    provider = VideoProvider(video=sio_video, batch_size=4, frames=list(range(n_total)))
    predictor = Predictor.from_export_dir(
        export_dir=exported_bottomup_onnx_dir, runtime="onnx", device="cpu"
    )
    labels = predictor.predict(
        provider, make_labels=True, skeleton=skeleton, videos=[sio_video]
    )
    assert isinstance(labels, sio.Labels)
    return labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames_by_idx(labels: sio.Labels) -> dict[int, sio.LabeledFrame]:
    """Index labeled frames by frame_idx for fast lookup."""
    return {lf.frame_idx: lf for lf in labels.labeled_frames}


def _match_instances_for_frame(
    lf_a: sio.LabeledFrame, lf_b: sio.LabeledFrame
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Match instances between two frames using OKS and return paired point arrays.

    Returns a list of (points_a, points_b) tuples, each of shape (n_nodes, 2).
    """
    from sleap_nn.evaluation import match_instances

    positive_pairs, _ = match_instances(lf_a, lf_b, threshold=0)
    pairs = []
    for inst_a, inst_b, _ in positive_pairs:
        pairs.append((inst_a.instance.numpy(), inst_b.instance.numpy()))
    return pairs


# ---------------------------------------------------------------------------
# ONNX accuracy tests
# ---------------------------------------------------------------------------


@requires_onnx
@requires_onnxruntime
class TestBottomUpONNXAccuracy:
    """Compare bottom-up ONNX export against PyTorch inference."""

    def test_export_produces_valid_files(self, exported_bottomup_onnx_dir):
        """Exported directory contains model.onnx with correct metadata."""
        assert (exported_bottomup_onnx_dir / "model.onnx").exists()

        meta_path = exported_bottomup_onnx_dir / "export_metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["model_type"] == "bottomup"
        assert meta["n_nodes"] == 2
        assert meta["node_names"] == ["A", "B"]

    def test_both_paths_produce_predictions(
        self, pytorch_bottomup_labels, onnx_bottomup_labels
    ):
        """Both PyTorch and ONNX paths produce at least some predictions."""
        assert len(pytorch_bottomup_labels.labeled_frames) > 0
        assert len(onnx_bottomup_labels.labeled_frames) > 0

    def test_instance_count_deviation(
        self, pytorch_bottomup_labels, onnx_bottomup_labels
    ):
        """Mean absolute instance count difference across common frames is bounded."""
        pt_frames = _frames_by_idx(pytorch_bottomup_labels)
        onnx_frames = _frames_by_idx(onnx_bottomup_labels)
        common_idxs = sorted(set(pt_frames.keys()) & set(onnx_frames.keys()))

        if not common_idxs:
            pytest.skip("No common frames between PyTorch and ONNX predictions")

        count_diffs = []
        for idx in common_idxs:
            n_pt = len(pt_frames[idx].instances)
            n_onnx = len(onnx_frames[idx].instances)
            count_diffs.append(abs(n_pt - n_onnx))

        mean_diff = np.mean(count_diffs)
        # Baseline: 0.0 (identical counts across 3 runs).
        # Fail above 1 instance deviation on average.
        assert mean_diff <= 1.0, (
            f"Mean instance count deviation {mean_diff:.2f} is too large "
            f"(baseline=0.0). Per-frame diffs: {count_diffs}"
        )

    def test_matched_instance_distances(
        self, pytorch_bottomup_labels, onnx_bottomup_labels
    ):
        """Median Euclidean distance between matched keypoints is bounded."""
        pt_frames = _frames_by_idx(pytorch_bottomup_labels)
        onnx_frames = _frames_by_idx(onnx_bottomup_labels)
        common_idxs = sorted(set(pt_frames.keys()) & set(onnx_frames.keys()))

        all_dists = []
        for idx in common_idxs:
            pairs = _match_instances_for_frame(pt_frames[idx], onnx_frames[idx])
            for pts_pt, pts_onnx in pairs:
                d = np.linalg.norm(pts_pt - pts_onnx, axis=-1)
                all_dists.extend(d[~np.isnan(d)].tolist())

        if not all_dists:
            pytest.skip("No matched keypoints to compare")

        all_dists = np.array(all_dists)
        median_dist = np.median(all_dists)

        # Baseline: 0.00 px across 3 runs.  Threshold = baseline + 3 px.
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX
        assert median_dist <= threshold, (
            f"Median matched keypoint distance {median_dist:.2f} px exceeds "
            f"threshold {threshold:.1f} px (baseline={_BASELINE_DIST_PX:.1f}). "
            f"p50={np.percentile(all_dists, 50):.2f}, "
            f"p95={np.percentile(all_dists, 95):.2f}, "
            f"p99={np.percentile(all_dists, 99):.2f}, "
            f"max={np.max(all_dists):.2f}"
        )

    def test_p99_keypoint_distance(self, pytorch_bottomup_labels, onnx_bottomup_labels):
        """99th percentile keypoint distance is bounded (baseline + 3 px)."""
        pt_frames = _frames_by_idx(pytorch_bottomup_labels)
        onnx_frames = _frames_by_idx(onnx_bottomup_labels)
        common_idxs = sorted(set(pt_frames.keys()) & set(onnx_frames.keys()))

        all_dists = []
        for idx in common_idxs:
            pairs = _match_instances_for_frame(pt_frames[idx], onnx_frames[idx])
            for pts_pt, pts_onnx in pairs:
                d = np.linalg.norm(pts_pt - pts_onnx, axis=-1)
                all_dists.extend(d[~np.isnan(d)].tolist())

        if not all_dists:
            pytest.skip("No matched keypoints to compare")

        all_dists = np.array(all_dists)
        p99 = float(np.percentile(all_dists, 99))
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX
        assert p99 <= threshold, (
            f"p99 keypoint distance {p99:.2f} px exceeds threshold "
            f"{threshold:.1f} px (baseline p99=0.0)"
        )

    def test_total_instance_count(self, pytorch_bottomup_labels, onnx_bottomup_labels):
        """Total detected instances match between PyTorch and ONNX paths."""
        pt_total = sum(
            len(lf.instances) for lf in pytorch_bottomup_labels.labeled_frames
        )
        onnx_total = sum(
            len(lf.instances) for lf in onnx_bottomup_labels.labeled_frames
        )
        assert pt_total == onnx_total, (
            f"Total instance count mismatch: PyTorch={pt_total}, ONNX={onnx_total} "
            f"(baseline={_BASELINE_TOTAL_INSTANCES})"
        )

    def test_no_catastrophic_coordinate_errors(
        self, pytorch_bottomup_labels, onnx_bottomup_labels
    ):
        """No individual matched keypoint distance exceeds a large threshold.

        This catches coordinate scaling bugs (e.g. forgetting to multiply by
        output_stride) which would produce distances in the hundreds of pixels.
        """
        pt_frames = _frames_by_idx(pytorch_bottomup_labels)
        onnx_frames = _frames_by_idx(onnx_bottomup_labels)
        common_idxs = sorted(set(pt_frames.keys()) & set(onnx_frames.keys()))

        max_dist = 0.0
        for idx in common_idxs:
            pairs = _match_instances_for_frame(pt_frames[idx], onnx_frames[idx])
            for pts_pt, pts_onnx in pairs:
                d = np.linalg.norm(pts_pt - pts_onnx, axis=-1)
                d_valid = d[~np.isnan(d)]
                if len(d_valid) > 0:
                    max_dist = max(max_dist, float(np.max(d_valid)))

        # Baseline max: 0.00 px.  Use a 10 px hard ceiling -- anything above
        # this strongly suggests a coordinate scaling or offset bug (a real
        # scaling error would produce 100+ px).
        assert max_dist <= 10.0, (
            f"Max matched keypoint distance {max_dist:.2f} px exceeds 10 px "
            f"(baseline max=0.0). This suggests a coordinate scaling or offset "
            f"bug in the export pipeline."
        )


# ---------------------------------------------------------------------------
# Reusable helpers for multi-model-type parity tests
# ---------------------------------------------------------------------------


def _export_ckpts_to_onnx(
    ckpt_paths: list[Path],
    export_dir: Path,
    extra_args: list[str] | None = None,
) -> Path:
    """Export checkpoint(s) to ONNX via the CLI and return the export dir."""
    from sleap_nn.export.cli import export_model

    runner = CliRunner()
    args = [str(p) for p in ckpt_paths] + [
        "-o",
        str(export_dir),
        "--format",
        "onnx",
        "--device",
        "cpu",
    ]
    if extra_args:
        args.extend(extra_args)
    result = runner.invoke(export_model, args)
    assert result.exit_code == 0, f"Export failed:\n{result.output}\n{result.exception}"
    assert (export_dir / "model.onnx").exists()
    return export_dir


def _pytorch_labels(
    ckpt_paths: list[Path],
    source: Path,
    n_frames: int = _N_FRAMES,
    peak_threshold: float = _PEAK_THRESHOLD,
) -> sio.Labels:
    """Run PyTorch inference via the new ``predict()`` entry point."""
    from sleap_nn.inference.run import predict

    return predict(
        str(source),
        model_paths=[str(p) for p in ckpt_paths],
        peak_threshold=peak_threshold,
        integral_refinement=None,
        device="cpu",
        frames=list(range(n_frames)),
    )


def _onnx_labels(
    export_dir: Path,
    source: Path,
    n_frames: int = _N_FRAMES,
) -> sio.Labels:
    """Run ONNX inference via ``Predictor.from_export_dir``."""
    from sleap_nn.inference.predictor import Predictor

    predictor = Predictor.from_export_dir(
        export_dir=export_dir, runtime="onnx", device="cpu"
    )
    video = sio.load_video(str(source))
    return predictor.predict(video, frames=list(range(n_frames)))


def _collect_distances(labels_a: sio.Labels, labels_b: sio.Labels) -> np.ndarray:
    """Match instances between two Labels and return all keypoint distances."""
    frames_a = _frames_by_idx(labels_a)
    frames_b = _frames_by_idx(labels_b)
    common = sorted(set(frames_a.keys()) & set(frames_b.keys()))

    all_dists = []
    for idx in common:
        pairs = _match_instances_for_frame(frames_a[idx], frames_b[idx])
        for pts_a, pts_b in pairs:
            d = np.linalg.norm(pts_a - pts_b, axis=-1)
            all_dists.extend(d[~np.isnan(d)].tolist())
    return np.array(all_dists) if all_dists else np.array([])


# ---------------------------------------------------------------------------
# Single-instance ONNX parity
# ---------------------------------------------------------------------------


@requires_onnx
@requires_onnxruntime
class TestSingleInstanceONNXAccuracy:
    """PyTorch vs ONNX parity for single-instance models.

    The single-instance ONNX wrapper bakes ``input_scale`` into the
    TorchScript trace as a constant, which introduces small rounding
    differences vs the PyTorch path's dynamic rescaling. Max per-keypoint
    deviation is ~12 px on the test fixture; the threshold is set at 15 px
    (well below the 100+ px that a real coordinate-scaling bug produces).
    """

    # Wider than the generic 10 px ceiling because of traced input_scale.
    _MAX_DIST_PX = 15.0

    @pytest.fixture(scope="class")
    def exported_dir(self, tmp_path_factory):
        pytest.importorskip("onnx")
        d = tmp_path_factory.mktemp("export_si_onnx")
        return _export_ckpts_to_onnx([_SINGLE_INSTANCE_CKPT], d)

    @pytest.fixture(scope="class")
    def pt_labels(self):
        return _pytorch_labels([_SINGLE_INSTANCE_CKPT], _VIDEO_3CH)

    @pytest.fixture(scope="class")
    def onnx_labels(self, exported_dir):
        return _onnx_labels(exported_dir, _VIDEO_3CH)

    def test_both_produce_predictions(self, pt_labels, onnx_labels):
        assert len(pt_labels.labeled_frames) > 0
        assert len(onnx_labels.labeled_frames) > 0

    def test_instance_counts_match(self, pt_labels, onnx_labels):
        pt_total = sum(len(lf.instances) for lf in pt_labels.labeled_frames)
        onnx_total = sum(len(lf.instances) for lf in onnx_labels.labeled_frames)
        assert pt_total == onnx_total, f"PyTorch={pt_total}, ONNX={onnx_total}"

    def test_keypoint_distances_bounded(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints")
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX
        assert (
            np.median(dists) <= threshold
        ), f"Median distance {np.median(dists):.2f} px > {threshold} px"

    def test_no_catastrophic_errors(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints")
        assert (
            np.max(dists) <= self._MAX_DIST_PX
        ), f"Max distance {np.max(dists):.2f} px > {self._MAX_DIST_PX} px"


# ---------------------------------------------------------------------------
# Top-down (centroid + centered-instance) ONNX parity
# ---------------------------------------------------------------------------


@requires_onnx
@requires_onnxruntime
class TestTopDownONNXAccuracy:
    """PyTorch vs ONNX parity for top-down (combined centroid + CI) models."""

    @pytest.fixture(scope="class")
    def exported_dir(self, tmp_path_factory):
        pytest.importorskip("onnx")
        d = tmp_path_factory.mktemp("export_td_onnx")
        return _export_ckpts_to_onnx([_CENTROID_CKPT, _CENTERED_INSTANCE_CKPT], d)

    @pytest.fixture(scope="class")
    def pt_labels(self):
        return _pytorch_labels([_CENTROID_CKPT, _CENTERED_INSTANCE_CKPT], _VIDEO_1CH)

    @pytest.fixture(scope="class")
    def onnx_labels(self, exported_dir):
        return _onnx_labels(exported_dir, _VIDEO_1CH)

    def test_both_produce_predictions(self, pt_labels, onnx_labels):
        assert len(pt_labels.labeled_frames) > 0
        assert len(onnx_labels.labeled_frames) > 0

    def test_instance_count_deviation(self, pt_labels, onnx_labels):
        pt_frames = _frames_by_idx(pt_labels)
        onnx_frames = _frames_by_idx(onnx_labels)
        common = sorted(set(pt_frames.keys()) & set(onnx_frames.keys()))
        diffs = [
            abs(len(pt_frames[i].instances) - len(onnx_frames[i].instances))
            for i in common
        ]
        assert np.mean(diffs) <= 1.0, f"Mean count diff {np.mean(diffs):.2f}"

    def test_keypoint_distances_bounded(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints")
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX
        assert (
            np.median(dists) <= threshold
        ), f"Median distance {np.median(dists):.2f} px > {threshold} px"

    def test_no_catastrophic_errors(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints")
        assert np.max(dists) <= 10.0, f"Max distance {np.max(dists):.2f} px > 10 px"


# ---------------------------------------------------------------------------
# Multi-class bottom-up ONNX parity
# ---------------------------------------------------------------------------


@requires_onnx
@requires_onnxruntime
class TestMultiClassBottomUpONNXAccuracy:
    """PyTorch vs ONNX parity for multi-class bottom-up models."""

    @pytest.fixture(scope="class")
    def exported_dir(self, tmp_path_factory):
        pytest.importorskip("onnx")
        d = tmp_path_factory.mktemp("export_mcbu_onnx")
        return _export_ckpts_to_onnx([_MULTICLASS_BOTTOMUP_CKPT], d)

    @pytest.fixture(scope="class")
    def pt_labels(self):
        return _pytorch_labels([_MULTICLASS_BOTTOMUP_CKPT], _VIDEO_1CH)

    @pytest.fixture(scope="class")
    def onnx_labels(self, exported_dir):
        return _onnx_labels(exported_dir, _VIDEO_1CH)

    def test_both_produce_predictions(self, pt_labels, onnx_labels):
        assert len(pt_labels.labeled_frames) > 0
        assert len(onnx_labels.labeled_frames) > 0

    def test_instance_count_deviation(self, pt_labels, onnx_labels):
        pt_frames = _frames_by_idx(pt_labels)
        onnx_frames = _frames_by_idx(onnx_labels)
        common = sorted(set(pt_frames.keys()) & set(onnx_frames.keys()))
        diffs = [
            abs(len(pt_frames[i].instances) - len(onnx_frames[i].instances))
            for i in common
        ]
        assert np.mean(diffs) <= 1.0, f"Mean count diff {np.mean(diffs):.2f}"

    def test_keypoint_distances_bounded(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints")
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX
        assert (
            np.median(dists) <= threshold
        ), f"Median distance {np.median(dists):.2f} px > {threshold} px"

    def test_no_catastrophic_errors(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints")
        assert np.max(dists) <= 10.0, f"Max distance {np.max(dists):.2f} px > 10 px"


# ---------------------------------------------------------------------------
# Multi-class top-down ONNX parity
# ---------------------------------------------------------------------------


@requires_onnx
@requires_onnxruntime
class TestMultiClassTopDownONNXAccuracy:
    """PyTorch vs ONNX parity for multi-class top-down (combined) models.

    The minimal test fixtures produce very few (or zero) detections on the
    short test video, so we verify that both paths agree rather than
    requiring a minimum detection count.
    """

    @pytest.fixture(scope="class")
    def exported_dir(self, tmp_path_factory):
        pytest.importorskip("onnx")
        d = tmp_path_factory.mktemp("export_mctd_onnx")
        return _export_ckpts_to_onnx([_CENTROID_CKPT, _MULTICLASS_CI_CKPT], d)

    @pytest.fixture(scope="class")
    def pt_labels(self):
        return _pytorch_labels([_CENTROID_CKPT, _MULTICLASS_CI_CKPT], _VIDEO_1CH)

    @pytest.fixture(scope="class")
    def onnx_labels(self, exported_dir):
        return _onnx_labels(exported_dir, _VIDEO_1CH)

    def test_both_agree_on_detection_count(self, pt_labels, onnx_labels):
        pt_total = sum(len(lf.instances) for lf in pt_labels.labeled_frames)
        onnx_total = sum(len(lf.instances) for lf in onnx_labels.labeled_frames)
        assert abs(pt_total - onnx_total) <= 2, f"PyTorch={pt_total}, ONNX={onnx_total}"

    def test_keypoint_distances_bounded(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints (both flows produced empty)")
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX
        assert (
            np.median(dists) <= threshold
        ), f"Median distance {np.median(dists):.2f} px > {threshold} px"

    def test_no_catastrophic_errors(self, pt_labels, onnx_labels):
        dists = _collect_distances(pt_labels, onnx_labels)
        if len(dists) == 0:
            pytest.skip("No matched keypoints (both flows produced empty)")
        assert np.max(dists) <= 10.0, f"Max distance {np.max(dists):.2f} px > 10 px"


# ---------------------------------------------------------------------------
# TensorRT accuracy tests
# ---------------------------------------------------------------------------


@requires_tensorrt
@requires_gpu
class TestBottomUpTensorRTAccuracy:
    """Compare bottom-up TensorRT export against ONNX inference."""

    @pytest.fixture(scope="class")
    def exported_trt_dir(self, bottomup_ckpt_path, tmp_path_factory):
        """Export to TensorRT format."""
        from sleap_nn.export.cli import export_model

        export_dir = tmp_path_factory.mktemp("export_bottomup_trt")
        runner = CliRunner()
        result = runner.invoke(
            export_model,
            [
                str(bottomup_ckpt_path),
                "-o",
                str(export_dir),
                "--format",
                "both",
                "--device",
                "cuda",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"TRT export failed:\n{result.output}\n{result.exception}"
        assert (export_dir / "model.trt").exists()
        return export_dir

    @pytest.fixture(scope="class")
    def trt_bottomup_labels(self, exported_trt_dir, video_path):
        """Run TensorRT inference on the test video."""
        from sleap_nn.export.inference import predict

        labels, stats = predict(
            export_dir=exported_trt_dir,
            video_path=video_path,
            runtime="tensorrt",
            device="cuda",
            batch_size=4,
            n_frames=_N_FRAMES,
            peak_conf_threshold=_PEAK_THRESHOLD,
        )
        return labels

    def test_tensorrt_export_and_instance_count(
        self, trt_bottomup_labels, onnx_bottomup_labels
    ):
        """TensorRT instance counts roughly match ONNX."""
        assert len(trt_bottomup_labels.labeled_frames) > 0

        trt_frames = _frames_by_idx(trt_bottomup_labels)
        onnx_frames = _frames_by_idx(onnx_bottomup_labels)
        common_idxs = sorted(set(trt_frames.keys()) & set(onnx_frames.keys()))

        if not common_idxs:
            pytest.skip("No common frames between TRT and ONNX predictions")

        count_diffs = [
            abs(len(trt_frames[i].instances) - len(onnx_frames[i].instances))
            for i in common_idxs
        ]
        assert np.mean(count_diffs) <= 1.0, (
            f"Mean TRT vs ONNX instance count diff {np.mean(count_diffs):.2f} "
            f"exceeds threshold"
        )

    def test_tensorrt_matched_distances(
        self, trt_bottomup_labels, onnx_bottomup_labels
    ):
        """TensorRT vs ONNX keypoint distances are bounded."""
        trt_frames = _frames_by_idx(trt_bottomup_labels)
        onnx_frames = _frames_by_idx(onnx_bottomup_labels)
        common_idxs = sorted(set(trt_frames.keys()) & set(onnx_frames.keys()))

        all_dists = []
        for idx in common_idxs:
            pairs = _match_instances_for_frame(onnx_frames[idx], trt_frames[idx])
            for pts_onnx, pts_trt in pairs:
                d = np.linalg.norm(pts_onnx - pts_trt, axis=-1)
                all_dists.extend(d[~np.isnan(d)].tolist())

        if not all_dists:
            pytest.skip("No matched keypoints to compare")

        all_dists = np.array(all_dists)
        # TRT fp16 quantization can introduce some deviation vs ONNX fp32.
        # Use wider tolerance than ONNX-vs-PyTorch since fp16 is lossy.
        threshold = _BASELINE_DIST_PX + _WARN_ABOVE_BASELINE_PX * 2  # 6 px
        assert np.median(all_dists) <= threshold, (
            f"Median TRT vs ONNX distance {np.median(all_dists):.2f} px "
            f"exceeds threshold {threshold:.1f} px"
        )
