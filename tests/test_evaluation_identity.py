"""Tests for the MOT-style identity-persistence metrics in ``sleap_nn.evaluation``.

Every case is built so the right answer is known by construction: two
trajectories far enough apart that detection matching is unambiguous, with
identity swaps and dropped frames injected at known positions. That way a
failure points at the metric, not at the matcher.
"""

import json

import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner

from sleap_nn.evaluation import (
    IdentityMetrics,
    compare_identity_metrics,
    identity_metrics,
    motion_diagnostic,
    run_identity_evaluation,
)

FNAME = "synthetic_identity_video.mp4"


def _synthetic_pose(n_frames, swap_at=(), drop=(), step=1.0, body_length=12.0):
    """Two trajectories; ``swap_at`` swaps predicted identities onward.

    Args:
        n_frames: Number of frames to generate.
        swap_at: Frames at which the predicted identity assignment flips.
        drop: Frames for which the prediction has no detections at all.
        step: Per-frame displacement, in pixels, of both animals.
        body_length: Distance between the two nodes. ``0`` yields a single-node
            (centroid-style) skeleton with no measurable extent.

    Returns:
        ``(gt_labels, pred_labels)``.
    """
    skeleton = sio.Skeleton(["head", "tail"] if body_length else ["head"])
    video = sio.Video.from_filename(FNAME)
    swap_at, drop = set(swap_at), set(drop)

    gt_tracks = [sio.Track("g0"), sio.Track("g1")]
    pred_tracks = [sio.Track("p0"), sio.Track("p1")]
    gt_frames, pred_frames = [], []
    flipped = False
    for frame_idx in range(n_frames):
        if frame_idx in swap_at:
            flipped = not flipped
        points = []
        for x0, y0 in ((10.0, 10.0), (200.0, 200.0)):
            nodes = [[x0 + step * frame_idx, y0]]
            if body_length:
                nodes.append([x0 + step * frame_idx + body_length, y0])
            points.append(np.array(nodes))
        gt_frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                instances=[
                    sio.Instance.from_numpy(
                        points[k], skeleton=skeleton, track=gt_tracks[k]
                    )
                    for k in range(2)
                ],
            )
        )
        if frame_idx in drop:
            pred_frames.append(
                sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=[])
            )
            continue
        pred_frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        points[k],
                        skeleton=skeleton,
                        score=1.0,
                        point_scores=np.ones(len(points[k])),
                        tracking_score=1.0,
                        track=pred_tracks[(k + 1) % 2 if flipped else k],
                    )
                    for k in range(2)
                ],
            )
        )
    return (
        sio.Labels(
            labeled_frames=gt_frames,
            videos=[video],
            skeletons=[skeleton],
            tracks=gt_tracks,
        ),
        sio.Labels(
            labeled_frames=pred_frames,
            videos=[video],
            skeletons=[skeleton],
            tracks=pred_tracks,
        ),
    )


def _square(height, width, cy, cx, half):
    """Return a boolean mask with a filled square centered at ``(cy, cx)``."""
    mask = np.zeros((height, width), dtype=bool)
    mask[cy - half : cy + half, cx - half : cx + half] = True
    return mask


def _synthetic_mask(n_frames, swap_at=()):
    """Mask-carrier twin of :func:`_synthetic_pose`, one square per animal."""
    video = sio.Video.from_filename(FNAME)
    swap_at = set(swap_at)
    gt_tracks = [sio.Track("g0"), sio.Track("g1")]
    pred_tracks = [sio.Track("p0"), sio.Track("p1")]
    gt_frames, pred_frames = [], []
    flipped = False
    for frame_idx in range(n_frames):
        if frame_idx in swap_at:
            flipped = not flipped
        squares = [
            _square(64, 64, 12 + frame_idx, 12, 5),
            _square(64, 64, 48 + frame_idx, 48, 5),
        ]
        gt_masks = []
        for k in range(2):
            mask = sio.UserSegmentationMask.from_numpy(squares[k])
            mask.track = gt_tracks[k]
            gt_masks.append(mask)
        gt_frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, masks=gt_masks)
        )
        pred_frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                masks=[
                    sio.PredictedSegmentationMask.from_numpy(
                        squares[k],
                        score=0.9,
                        track=pred_tracks[(k + 1) % 2 if flipped else k],
                    )
                    for k in range(2)
                ],
            )
        )
    return (
        sio.Labels(labeled_frames=gt_frames, videos=[video], tracks=gt_tracks),
        sio.Labels(labeled_frames=pred_frames, videos=[video], tracks=pred_tracks),
    )


# ---------------------------------------------------------------------------
# identity_metrics: known-by-construction cases
# ---------------------------------------------------------------------------
def test_identity_metrics_perfect_tracking():
    """No swaps, no drops -> zero switches and a perfect IDF1."""
    gt, pred = _synthetic_pose(100)
    m = identity_metrics(gt, pred, "pose")

    assert m.id_switches == 0
    assert m.idf1 == pytest.approx(1.0)
    assert m.idp == pytest.approx(1.0)
    assert m.idr == pytest.approx(1.0)
    assert m.mean_track_purity == pytest.approx(1.0)
    assert m.mean_gt_coverage == pytest.approx(1.0)
    assert (m.mostly_tracked, m.partly_tracked, m.mostly_lost) == (2, 0, 0)
    assert m.fragmentations == 0
    assert m.n_matched == 200
    assert m.n_gt_dets == 200
    assert m.n_pred_dets == 200
    assert m.n_frames_compared == 100
    assert (m.n_gt_tracks, m.n_pred_tracks) == (2, 2)
    assert m.notes == []


def test_identity_metrics_one_swap_counts_two_switches():
    """One swap costs a switch on EACH GT trajectory, and halves purity."""
    gt, pred = _synthetic_pose(100, swap_at=[50])
    m = identity_metrics(gt, pred, "pose")

    assert m.id_switches == 2
    assert m.mean_track_purity == pytest.approx(0.5)
    # Coverage is untouched: every GT detection still matched something.
    assert m.mean_gt_coverage == pytest.approx(1.0)
    assert m.n_matched == 200
    # A swap does not create tracks, so IDF1 caps at the half each identity kept.
    assert m.idf1 == pytest.approx(0.5)


def test_identity_metrics_two_swaps_restore_identity():
    """Swapping back is still two more switches, but purity recovers."""
    gt, pred = _synthetic_pose(100, swap_at=[30, 60])
    m = identity_metrics(gt, pred, "pose")

    assert m.id_switches == 4
    assert m.mean_track_purity == pytest.approx(0.7)


def test_identity_metrics_gap_is_fragmentation_not_switch():
    """A dropped run mid-trajectory fragments coverage without a switch."""
    gt, pred = _synthetic_pose(100, drop=range(40, 50))
    m = identity_metrics(gt, pred, "pose")

    assert m.id_switches == 0
    assert m.fragmentations == 2  # one per GT trajectory
    assert m.mean_gt_coverage == pytest.approx(0.9)
    assert m.n_matched == 180
    # 180 IDTP, 20 IDFN, 0 IDFP.
    assert m.idf1 == pytest.approx(2 * 180 / (2 * 180 + 0 + 20))
    assert m.idp == pytest.approx(1.0)
    assert m.idr == pytest.approx(0.9)


def test_identity_metrics_trailing_gap_is_not_a_fragmentation():
    """A trajectory that simply ends is lost coverage, not a fragmentation."""
    gt, pred = _synthetic_pose(100, drop=range(90, 100))
    m = identity_metrics(gt, pred, "pose")

    assert m.fragmentations == 0
    assert m.mean_gt_coverage == pytest.approx(0.9)


def test_identity_metrics_coverage_buckets_mt_pt_ml():
    """The MT/PT/ML cuts follow per-trajectory coverage, not the mean."""
    gt, pred = _synthetic_pose(100, drop=range(50, 100))
    m = identity_metrics(gt, pred, "pose", mt_threshold=0.8, ml_threshold=0.2)

    # Both trajectories sit at 0.5 coverage: partly tracked, neither MT nor ML.
    assert (m.mostly_tracked, m.partly_tracked, m.mostly_lost) == (0, 2, 0)
    m_strict = identity_metrics(gt, pred, "pose", mt_threshold=0.4, ml_threshold=0.2)
    assert (m_strict.mostly_tracked, m_strict.partly_tracked) == (2, 0)
    m_loose = identity_metrics(gt, pred, "pose", mt_threshold=0.9, ml_threshold=0.6)
    assert (m_loose.mostly_lost, m_loose.partly_tracked) == (2, 0)


def test_identity_metrics_untracked_predictions_are_counted_not_matched():
    """Untracked predicted detections inflate no metric but are reported."""
    gt, pred = _synthetic_pose(10)
    for lf in pred:
        for inst in lf.instances:
            inst.track = None
    m = identity_metrics(gt, pred, "pose")

    assert m.n_pred_dets == 20
    assert m.n_pred_untracked == 20
    assert m.n_matched == 0
    assert m.n_pred_tracks == 0
    assert m.mostly_lost == 2
    assert np.isnan(m.idf1)


def test_identity_metrics_threshold_rejects_far_matches():
    """Below-threshold similarity leaves the GT detection unmatched."""
    gt, pred = _synthetic_pose(20)
    m_loose = identity_metrics(gt, pred, "pose", match_threshold=0.5)
    m_strict = identity_metrics(gt, pred, "pose", match_threshold=1.5)

    assert m_loose.n_matched == 40
    assert m_strict.n_matched == 0
    assert m_strict.mean_gt_coverage == pytest.approx(0.0)


def test_identity_metrics_no_frames_in_common_reports_a_note():
    """Disjoint frame ranges return an empty result with an explanatory note."""
    gt, _ = _synthetic_pose(10)
    _, pred = _synthetic_pose(10)
    for lf in pred:
        lf.frame_idx += 1000
    m = identity_metrics(gt, pred, "pose")

    assert m.n_frames_compared == 0
    assert m.n_matched == 0
    assert any("no frames in common" in note for note in m.notes)


def test_identity_metrics_missing_predicted_frames_reports_a_note():
    """GT frames with no predicted counterpart are flagged, not silently dropped."""
    gt, pred = _synthetic_pose(10)
    pred.labeled_frames = pred.labeled_frames[:6]
    m = identity_metrics(gt, pred, "pose")

    assert m.n_frames_compared == 6
    assert any("no predicted counterpart" in note for note in m.notes)


def test_identity_metrics_aligns_a_distinct_video_object_on_the_same_path():
    """A re-loaded prediction aligns on the video path, with no caveat needed."""
    gt, pred = _synthetic_pose(10)
    same_path_video = sio.Video.from_filename(FNAME)
    for lf in pred:
        lf.video = same_path_video
    pred.videos = [same_path_video]
    m = identity_metrics(gt, pred, "pose")

    assert m.n_frames_compared == 10
    assert m.id_switches == 0
    assert m.notes == []


def test_identity_metrics_falls_back_to_frame_idx_on_a_renamed_clip():
    """One video per side under different paths: the only pairing is taken."""
    gt, pred = _synthetic_pose(10)
    renamed = sio.Video.from_filename("copied_elsewhere.mp4")
    for lf in pred:
        lf.video = renamed
    pred.videos = [renamed]
    m = identity_metrics(gt, pred, "pose")

    assert m.n_frames_compared == 10
    assert m.id_switches == 0
    assert any("frame_idx only" in note for note in m.notes)


def test_identity_metrics_pairs_a_single_video_prediction_with_a_project():
    """A prediction on GT's SECOND video must not be paired with its first.

    Keying by position in ``labels.videos`` would compare the prediction against
    the wrong video's ground truth (or find nothing); keying by path pairs it
    correctly.
    """
    gt, pred = _synthetic_pose(10)
    # Turn the GT into a two-video project whose labeled frames belong to the
    # second video, leaving the prediction a single-video run on that same path.
    decoy = sio.Video.from_filename("some_other_recording.mp4")
    gt.videos = [decoy, gt.videos[0]]
    m = identity_metrics(gt, pred, "pose")

    assert m.n_frames_compared == 10
    assert m.n_matched == 20
    assert m.id_switches == 0
    assert m.notes == []


def test_identity_metrics_rejects_unknown_carrier():
    """An unknown carrier fails loudly instead of scoring the wrong thing."""
    gt, pred = _synthetic_pose(2)
    with pytest.raises(ValueError, match="carrier must be one of"):
        identity_metrics(gt, pred, "keypoints")


# ---------------------------------------------------------------------------
# The mask carrier
# ---------------------------------------------------------------------------
def test_identity_metrics_mask_carrier_perfect_and_swapped():
    """Masks carry identity too, scored by IoU instead of OKS."""
    gt, pred = _synthetic_mask(20)
    m = identity_metrics(gt, pred, "mask")
    assert m.n_matched == 40
    assert m.id_switches == 0
    assert m.idf1 == pytest.approx(1.0)

    gt, pred = _synthetic_mask(20, swap_at=[10])
    m_swapped = identity_metrics(gt, pred, "mask")
    assert m_swapped.id_switches == 2
    assert m_swapped.mean_track_purity == pytest.approx(0.5)


def test_identity_metrics_mask_carrier_ignores_instances():
    """The mask carrier scores `lf.masks`; poses in the same file are irrelevant."""
    gt, pred = _synthetic_mask(5)
    m = identity_metrics(gt, pred, "mask")
    # The same files have no instances at all, so the pose carrier sees nothing.
    m_pose = identity_metrics(gt, pred, "pose")

    assert m.n_gt_dets == 10
    assert m_pose.n_gt_dets == 0


# ---------------------------------------------------------------------------
# motion_diagnostic
# ---------------------------------------------------------------------------
def test_motion_diagnostic_flags_continuous_video():
    """Small per-frame steps relative to body size read as continuous."""
    gt, _ = _synthetic_pose(50, step=0.5)
    diagnostic = motion_diagnostic(gt, "pose")

    assert diagnostic["is_continuous"] is True
    assert diagnostic["step_over_size"] < 0.5


def test_motion_diagnostic_flags_sparse_split():
    """A teleporting animal is reported as not continuous (the .pkg.slp trap)."""
    gt, _ = _synthetic_pose(50, step=400.0)
    diagnostic = motion_diagnostic(gt, "pose")

    assert diagnostic["is_continuous"] is False
    assert diagnostic["step_over_size"] > 0.5


def test_motion_diagnostic_single_node_skeleton_cannot_be_judged():
    """A centroid-style skeleton has no extent, so continuity is inconclusive.

    Reporting it as sparse would fire the "not continuous video" warning on
    every centroid-model tracking evaluation.
    """
    gt, _ = _synthetic_pose(50, step=0.5, body_length=0.0)
    diagnostic = motion_diagnostic(gt, "pose")

    assert np.isnan(diagnostic["step_over_size"])
    assert diagnostic["is_continuous"] is False
    assert "no measurable extent" in diagnostic["note"]


def test_motion_diagnostic_without_tracks_says_so():
    """Untracked labels cannot be judged, and the result says why."""
    gt, _ = _synthetic_pose(5)
    for lf in gt:
        for inst in lf.instances:
            inst.track = None
    diagnostic = motion_diagnostic(gt, "pose")

    assert diagnostic["is_continuous"] is False
    assert np.isnan(diagnostic["step_over_size"])
    assert "not enough tracked detections" in diagnostic["note"]


def test_motion_diagnostic_mask_carrier():
    """The mask carrier measures step against equivalent-circle diameter."""
    gt, _ = _synthetic_mask(20)
    diagnostic = motion_diagnostic(gt, "mask")

    assert diagnostic["is_continuous"] is True
    assert diagnostic["median_step_px"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def test_compare_identity_metrics_renders_arms_and_nan():
    """The table carries one row per arm and prints NaN cells as n/a."""
    gt, pred = _synthetic_pose(20)
    table = compare_identity_metrics(
        {
            "geometry": identity_metrics(gt, pred, "pose"),
            "empty": IdentityMetrics(),
        }
    )

    assert "| geometry |" in table
    assert "| empty |" in table
    assert "n/a" in table
    assert "Lower is better" in table


def test_identity_metrics_as_dict_and_summary_are_serializable():
    """`as_dict` is JSON-safe and `summary` is a single line."""
    gt, pred = _synthetic_pose(10)
    m = identity_metrics(gt, pred, "pose")

    payload = m.as_dict()
    json.dumps(payload)
    assert payload["id_switches"] == 0
    assert "IDSW=0" in m.summary()
    assert "\n" not in m.summary()


# ---------------------------------------------------------------------------
# run_identity_evaluation + the CLI
# ---------------------------------------------------------------------------
def test_run_identity_evaluation_round_trip(tmp_path):
    """The driver loads both files, scores them, and saves JSON."""
    gt, pred = _synthetic_pose(30, swap_at=[15])
    gt_path = tmp_path / "gt.slp"
    pred_path = tmp_path / "pred.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())
    out_path = tmp_path / "identity.json"

    result = run_identity_evaluation(
        gt_path.as_posix(),
        pred_path.as_posix(),
        save_metrics=out_path.as_posix(),
    )

    assert result["carrier"] == "pose"
    assert result["id_switches"] == 2
    assert result["match_threshold"] == 0.5
    assert "step_over_size" in result["motion_diagnostic"]
    saved = json.loads(out_path.read_text())
    assert saved["id_switches"] == 2


def test_run_identity_evaluation_auto_detects_mask_carrier(tmp_path):
    """A mask-only prediction selects the mask carrier without being told."""
    gt, pred = _synthetic_mask(10)
    gt_path = tmp_path / "gt_mask.slp"
    pred_path = tmp_path / "pred_mask.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())

    result = run_identity_evaluation(gt_path.as_posix(), pred_path.as_posix())

    assert result["carrier"] == "mask"
    assert result["n_matched"] == 20


def test_run_identity_evaluation_untracked_prediction_returns_none(tmp_path, caplog):
    """An untracked prediction is skipped with a pointer at `sleap-nn track`."""
    gt, pred = _synthetic_pose(10)
    for lf in pred:
        for inst in lf.instances:
            inst.track = None
    gt_path = tmp_path / "gt.slp"
    pred_path = tmp_path / "pred_untracked.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())

    assert run_identity_evaluation(gt_path.as_posix(), pred_path.as_posix()) is None


def test_run_identity_evaluation_warns_on_sparse_ground_truth(tmp_path, caplog):
    """A sparse GT set is called out rather than silently scored."""
    gt, pred = _synthetic_pose(20, step=400.0)
    gt_path = tmp_path / "gt_sparse.slp"
    pred_path = tmp_path / "pred_sparse.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())

    result = run_identity_evaluation(gt_path.as_posix(), pred_path.as_posix())

    assert result is not None
    assert result["motion_diagnostic"]["is_continuous"] is False


def test_cli_eval_tracking_smoke(tmp_path):
    """`sleap-nn eval-tracking` wires the options through to the driver."""
    from sleap_nn.cli import cli

    gt, pred = _synthetic_pose(20, swap_at=[10])
    gt_path = tmp_path / "gt.slp"
    pred_path = tmp_path / "pred.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())
    out_path = tmp_path / "ids.json"

    result = CliRunner().invoke(
        cli,
        [
            "eval-tracking",
            "-g",
            gt_path.as_posix(),
            "-p",
            pred_path.as_posix(),
            "--carrier",
            "pose",
            "--match_threshold",
            "0.5",
            "-s",
            out_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out_path.read_text())["id_switches"] == 2


# ---------------------------------------------------------------------------
# user_labels_only: predictions living in the ground-truth file
# ---------------------------------------------------------------------------
def test_identity_metrics_drops_predicted_detections_from_ground_truth():
    """Predictions carried in a GT project must not be scored as ground truth.

    A proofread project routinely holds both user labels and predictions from an
    earlier run, and those predictions carry tracks. Counted as ground truth
    they invent trajectories and inflate every total.
    """
    gt, pred = _synthetic_pose(10)
    # Splice a third, predicted "animal" into the GT file, tracked as its own
    # identity -- exactly what an earlier tracking run leaves behind.
    stray_track = sio.Track("stale_prediction")
    skeleton = gt.skeletons[0]
    for lf in gt:
        lf.instances.append(
            sio.PredictedInstance.from_numpy(
                np.array([[500.0, 500.0], [512.0, 500.0]]),
                skeleton=skeleton,
                score=1.0,
                point_scores=np.ones(2),
                tracking_score=1.0,
                track=stray_track,
            )
        )

    kept = identity_metrics(gt, pred, "pose", user_labels_only=True)
    default = identity_metrics(gt, pred, "pose")

    # Filtered: the two real animals only, and a note saying what happened.
    assert kept.n_gt_dets == 20
    assert kept.n_gt_tracks == 2
    assert kept.idf1 == pytest.approx(1.0)
    assert any("were dropped" in note for note in kept.notes)

    # Default (off): the phantom third trajectory is scored, dragging IDF1 down
    # and showing up as mostly-lost -- with a note pointing at the flag.
    assert default.n_gt_dets == 30
    assert default.n_gt_tracks == 3
    assert default.mostly_lost == 1
    assert default.idf1 < kept.idf1
    assert any("are model output" in note for note in default.notes)


def test_identity_metrics_user_labels_only_can_empty_the_ground_truth():
    """Predicted GT + the filter = nothing scored, said out loud.

    Tracked ground truth is usually predicted poses with tracks assigned
    afterwards, which is why the filter is off by default: on the re-ID
    benchmark's own GT sessions, turning it on takes 2465 detections to 0.
    """
    gt, pred = _synthetic_pose(10)
    # Rebuild the GT side entirely out of predicted instances, keeping tracks.
    for lf in gt:
        lf.instances = [
            sio.PredictedInstance.from_numpy(
                inst.numpy(),
                skeleton=gt.skeletons[0],
                score=1.0,
                point_scores=np.ones(len(inst.numpy())),
                tracking_score=1.0,
                track=inst.track,
            )
            for inst in lf.instances
        ]

    assert identity_metrics(gt, pred, "pose").n_gt_dets == 20

    filtered = identity_metrics(gt, pred, "pose", user_labels_only=True)
    assert filtered.n_gt_dets == 0
    assert filtered.n_matched == 0
    assert any("left NO ground truth" in note for note in filtered.notes)


def test_identity_metrics_mask_carrier_drops_predicted_gt_masks():
    """Same rule on the mask carrier, with the decoded arrays kept in step."""
    gt, pred = _synthetic_mask(5)
    stray_track = sio.Track("stale_prediction")
    for lf in gt:
        lf.masks.append(
            sio.PredictedSegmentationMask.from_numpy(
                _square(64, 64, 32, 32, 4), score=0.9, track=stray_track
            )
        )

    kept = identity_metrics(gt, pred, "mask", user_labels_only=True)
    default = identity_metrics(gt, pred, "mask")

    assert kept.n_gt_dets == 10
    assert kept.n_gt_tracks == 2
    assert kept.n_matched == 10
    assert default.n_gt_dets == 15
    assert default.n_gt_tracks == 3


def test_cli_eval_tracking_user_labels_only_flag(tmp_path):
    """The CLI can turn the ground-truth filter on."""
    from sleap_nn.cli import cli

    gt, pred = _synthetic_pose(10)
    gt_path = tmp_path / "gt.slp"
    pred_path = tmp_path / "pred.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())
    out_path = tmp_path / "ids.json"

    result = CliRunner().invoke(
        cli,
        [
            "eval-tracking",
            "-g",
            gt_path.as_posix(),
            "-p",
            pred_path.as_posix(),
            "--user_labels_only",
            "-s",
            out_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    # The synthetic GT is user-labeled, so the filter keeps all of it.
    assert json.loads(out_path.read_text())["n_gt_dets"] == 20
