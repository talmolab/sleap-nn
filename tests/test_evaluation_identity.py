"""Tests for the MOT-style identity-persistence metrics in ``sleap_nn.evaluation``.

Every case is built so the right answer is known by construction: two
trajectories far enough apart that detection matching is unambiguous, with
identity swaps and dropped frames injected at known positions. That way a
failure points at the metric, not at the matcher.
"""

import json
import re

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


def _synthetic_pose(
    n_frames,
    swap_at=(),
    drop=(),
    step=1.0,
    body_length=12.0,
    pred_offset=0.0,
    fname=FNAME,
    x_shift=0.0,
    pred_names=("p0", "p1"),
):
    """Two trajectories; ``swap_at`` swaps predicted identities onward.

    Args:
        n_frames: Number of frames to generate.
        swap_at: Frames at which the predicted identity assignment flips.
        drop: Frames for which the prediction has no detections at all.
        step: Per-frame displacement, in pixels, of both animals.
        body_length: Distance between the two nodes. ``0`` yields a single-node
            (centroid-style) skeleton with no measurable extent.
        pred_offset: Horizontal offset, in pixels, of every predicted point from
            its ground truth.
        fname: Video filename, so two calls can build two different videos.
        x_shift: Horizontal shift, in pixels, of both animals on both sides.
        pred_names: Names of the two predicted tracks.

    Returns:
        ``(gt_labels, pred_labels)``.
    """
    skeleton = sio.Skeleton(["head", "tail"] if body_length else ["head"])
    video = sio.Video.from_filename(fname)
    swap_at, drop = set(swap_at), set(drop)

    gt_tracks = [sio.Track("g0"), sio.Track("g1")]
    pred_tracks = [sio.Track(name) for name in pred_names]
    gt_frames, pred_frames = [], []
    flipped = False
    for frame_idx in range(n_frames):
        if frame_idx in swap_at:
            flipped = not flipped
        points = []
        for x0, y0 in ((10.0 + x_shift, 10.0), (200.0 + x_shift, 200.0)):
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
                        points[k] + np.array([pred_offset, 0.0]),
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
    """GT frames with no predicted counterpart are flagged -- and scored as misses."""
    gt, pred = _synthetic_pose(10)
    pred.labeled_frames = pred.labeled_frames[:6]
    m = identity_metrics(gt, pred, "pose")

    assert m.n_frames_compared == 10
    assert m.n_frames_missing_pred == 4
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
    # The detector-independent identity score is a column of its own.
    assert "IDF1m" in table.splitlines()[0]


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


# ---------------------------------------------------------------------------
# Scoring correctness (emb-review C): each case runs through BOTH real entry
# points -- `identity_metrics` and `sleap-nn eval-tracking` -- so a fix that
# lands in one and not the other fails here.
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
ENTRIES = ("api", "cli")


def _plain(text: str) -> str:
    """CLI output with ANSI codes and panel borders stripped, whitespace collapsed.

    rich-click renders errors into a bordered panel wrapped at the terminal
    width, so a raw substring check is layout-dependent.
    """
    text = re.sub(r"[│┃─━╭╮╰╯]", " ", _ANSI.sub("", text))
    return " ".join(text.split())


def _cli_args(gt, pred, tmp_path, out_path=None, **opts):
    """Save both files and build `sleap-nn eval-tracking` arguments."""
    gt_path, pred_path = tmp_path / "gt.slp", tmp_path / "pred.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())
    args = ["eval-tracking", "-g", gt_path.as_posix(), "-p", pred_path.as_posix()]
    if out_path is not None:
        args += ["-s", out_path.as_posix()]
    if "carrier" in opts:
        args += ["--carrier", opts["carrier"]]
    if opts.get("match_threshold") is not None:
        args += ["--match_threshold", str(opts["match_threshold"])]
    if opts.get("global_identity"):
        args += ["--global_identity"]
    return args


def _score(entry, gt, pred, tmp_path, **opts):
    """Score ``pred`` against ``gt`` through ``identity_metrics`` or the CLI.

    Returns the metrics as a dict (the CLI's saved JSON, where NaN is ``None``).
    """
    if entry == "api":
        carrier = opts.pop("carrier", "pose")
        return identity_metrics(gt, pred, carrier, **opts).as_dict()

    from sleap_nn.cli import cli

    out_path = tmp_path / "ids.json"
    result = CliRunner().invoke(cli, _cli_args(gt, pred, tmp_path, out_path, **opts))
    assert result.exit_code == 0, result.output
    return json.loads(out_path.read_text())


def _retrack_names(labels, names):
    """Give ``labels``' predicted tracks new names, in order of appearance."""
    mapping = {}
    for lf in labels:
        for inst in lf.instances:
            if inst.track is not None:
                old = inst.track.name
                if old not in mapping:
                    mapping[old] = sio.Track(names[len(mapping) % len(names)])
                inst.track = mapping[old]
    labels.tracks = list({id(t): t for t in mapping.values()}.values())
    return labels


def _concat(first, second):
    """One ``Labels`` holding both inputs' frames, videos, skeletons and tracks."""
    return sio.Labels(
        labeled_frames=first.labeled_frames + second.labeled_frames,
        videos=first.videos + second.videos,
        skeletons=first.skeletons + second.skeletons,
        tracks=first.tracks + second.tracks,
    )


# --- C1: GT frames missing from the prediction are misses, not skipped --------
@pytest.mark.parametrize("entry", ENTRIES)
def test_missing_predicted_frames_score_as_misses(entry, tmp_path):
    """A frame the prediction lacks scores exactly like a present, empty frame.

    Frames go missing via `--no_empty_frames` / `clean_empty_frames` or when the
    detector found nothing. Skipping them scored a perfect tracker missing half
    the frames IDF1 1.000 -- above the same tracker with FEWER misses.
    """
    gt, pred = _synthetic_pose(10)
    pred.labeled_frames = pred.labeled_frames[:5]  # frames 5-9 absent
    m = _score(entry, gt, pred, tmp_path)

    # 10 of the 20 tracked GT detections have no prediction.
    assert m["idf1"] == pytest.approx(2 * 10 / (2 * 10 + 0 + 10))
    assert m["idr"] == pytest.approx(0.5)
    assert m["idp"] == pytest.approx(1.0)
    assert m["mean_gt_coverage"] == pytest.approx(0.5)
    assert m["mostly_tracked"] == 0
    assert m["n_frames_compared"] == 10
    assert m["n_frames_missing_pred"] == 5

    # The same misses as present-but-empty frames: identical scores.
    gt_ref, pred_ref = _synthetic_pose(10, drop=range(5, 10))
    ref = identity_metrics(gt_ref, pred_ref, "pose")
    assert m["idf1"] == pytest.approx(ref.idf1)
    assert m["mean_gt_coverage"] == pytest.approx(ref.mean_gt_coverage)


def test_missing_predicted_frames_mid_trajectory_are_fragmentations():
    """An absent run of frames interrupts a trajectory like an empty one does."""
    gt, pred = _synthetic_pose(30)
    pred.labeled_frames = [lf for lf in pred if not 10 <= lf.frame_idx < 15]
    m = identity_metrics(gt, pred, "pose")

    assert m.fragmentations == 2  # one per GT trajectory
    assert m.id_switches == 0
    assert m.mean_gt_coverage == pytest.approx(25 / 30)


def test_gt_video_without_any_prediction_is_not_scored():
    """A project video the prediction never covered is named, not scored as misses.

    Scoring a one-video prediction against a multi-video GT project is the
    normal workflow; the other videos were simply not predicted.
    """
    gt_a, pred_a = _synthetic_pose(10)
    gt_b, _ = _synthetic_pose(10, fname="other_session.mp4")
    gt = _concat(gt_a, gt_b)
    m = identity_metrics(gt, pred_a, "pose")

    assert m.idf1 == pytest.approx(1.0)
    assert m.n_videos_compared == 1
    assert m.n_frames_compared == 10
    assert any("share no frame with the prediction" in n for n in m.notes)


# --- C2: single-node (centroid) skeletons match by distance -------------------
@pytest.mark.parametrize("entry", ENTRIES)
def test_single_node_skeleton_matches_by_distance(entry, tmp_path):
    """A 1 px centroid offset is a match, not an OKS of 0.

    OKS normalizes by the GT pose's area, which is 0 for one node, so any
    non-zero offset scored 0 and a perfect centroid tracker got IDF1 0.000.
    """
    gt, pred = _synthetic_pose(10, body_length=0.0, pred_offset=1.0)
    m = _score(entry, gt, pred, tmp_path)

    assert m["n_matched"] == 20
    assert m["idf1"] == pytest.approx(1.0)
    assert m["id_switches"] == 0
    assert m["match_method"] == "distance"
    assert m["match_threshold"] == pytest.approx(50.0)


@pytest.mark.parametrize("entry", ENTRIES)
def test_single_node_match_threshold_is_in_pixels(entry, tmp_path):
    """`--match_threshold` overrides the 50 px radius, in pixels."""
    gt, pred = _synthetic_pose(10, body_length=0.0, pred_offset=5.0)

    wide = _score(entry, gt, pred, tmp_path, match_threshold=10.0)
    assert wide["n_matched"] == 20
    assert wide["match_threshold"] == pytest.approx(10.0)

    narrow = _score(entry, gt, pred, tmp_path, match_threshold=2.0)
    assert narrow["n_matched"] == 0
    assert any("no tracked prediction matched" in n for n in narrow["notes"])


def test_centroid_predictions_match_full_pose_ground_truth():
    """Centroid-model output is scored against pose GT at the GT's centroid.

    The centroid + embedding workflow emits a single-node skeleton, while the
    ground truth it is scored against is usually full pose.
    """
    gt, pose_pred = _synthetic_pose(10)
    centroid = sio.Skeleton(["centroid"])
    pred_frames = []
    for lf in pose_pred:
        pred_frames.append(
            sio.LabeledFrame(
                video=lf.video,
                frame_idx=lf.frame_idx,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        inst.numpy().mean(axis=0, keepdims=True) + [1.0, 0.0],
                        skeleton=centroid,
                        score=1.0,
                        point_scores=np.ones(1),
                        track=inst.track,
                    )
                    for inst in lf.instances
                ],
            )
        )
    pred = sio.Labels(
        labeled_frames=pred_frames, videos=pose_pred.videos, skeletons=[centroid]
    )
    m = identity_metrics(gt, pred, "pose")

    assert m.match_method == "distance"
    assert m.n_matched == 20
    assert m.idf1 == pytest.approx(1.0)


def test_zero_area_pose_matches_by_distance_under_oks():
    """A multi-node GT pose spanning zero area falls back to distance, as in eval.

    Both synthetic nodes sit on one row, so the GT bounding box has zero area
    and OKS degenerates into an exact-equality test -- the fallback
    `match_instances` already applies for the detection metrics.
    """
    gt, pred = _synthetic_pose(10, pred_offset=1.0)
    m = identity_metrics(gt, pred, "pose")

    assert m.match_method == "oks"
    assert m.n_matched == 20
    assert m.idf1 == pytest.approx(1.0)


# --- C3: trajectories are per video by default --------------------------------
def _two_video_pose():
    """Two videos, GT names reused per video, the tracker minting fresh ids in B."""
    gt_a, pred_a = _synthetic_pose(10)
    gt_b, pred_b = _synthetic_pose(
        10, fname="second_video.mp4", pred_names=("p2", "p3")
    )
    return _concat(gt_a, gt_b), _concat(pred_a, pred_b)


@pytest.mark.parametrize("entry", ENTRIES)
def test_trajectories_are_keyed_per_video(entry, tmp_path):
    """Perfect per-video tracking scores perfect, whatever names each video got.

    Keying by track name across videos read GT 'g0' in video B as the same
    animal as 'g0' in video A, charging the tracker 2 switches and IDF1 0.5.
    """
    gt, pred = _two_video_pose()
    m = _score(entry, gt, pred, tmp_path)

    assert m["id_switches"] == 0
    assert m["idf1"] == pytest.approx(1.0)
    assert m["mean_track_purity"] == pytest.approx(1.0)
    assert m["n_videos_compared"] == 2
    assert m["n_gt_tracks"] == 4
    assert m["global_identity"] is False
    assert any("recur across videos" in n for n in m["notes"])


@pytest.mark.parametrize("entry", ENTRIES)
def test_global_identity_keys_trajectories_by_name(entry, tmp_path):
    """With global identities, keeping an animal's name across videos is scored."""
    gt, pred = _two_video_pose()
    lost = _score(entry, gt, pred, tmp_path, global_identity=True)
    # The tracker gave video B fresh ids, so each global identity switched once.
    assert lost["global_identity"] is True
    assert lost["id_switches"] == 2
    assert lost["idf1"] == pytest.approx(0.5)
    assert lost["n_gt_tracks"] == 2

    # A prediction that kept the global names (e.g. a multi-class ID model).
    _retrack_names(pred, ["p0", "p1"])
    kept = _score(entry, gt, pred, tmp_path, global_identity=True)
    assert kept["id_switches"] == 0
    assert kept["idf1"] == pytest.approx(1.0)


# --- C4: `--carrier auto` + tracks resolved through the pose <-> mask link -----
def _pose_and_mask(n_frames=6, pred_links=False):
    """Pose + mask files: GT tracked on its INSTANCES, prediction on its MASKS.

    Ground truth proofread in the GUI carries tracks on the instances, its
    masks linked to them. The prediction is mask-carrier tracking output: masks
    tracked, poses untracked -- linked to their masks only if ``pred_links``.
    """
    skeleton = sio.Skeleton(["head", "tail"])
    video = sio.Video.from_filename(FNAME)
    gt_tracks = [sio.Track("g0"), sio.Track("g1")]
    pred_tracks = [sio.Track("p0"), sio.Track("p1")]
    gt_frames, pred_frames = [], []
    for frame_idx in range(n_frames):
        gt_insts, gt_masks, pred_insts, pred_masks = [], [], [], []
        for k, (cy, cx) in enumerate(((12 + frame_idx, 12), (48 + frame_idx, 48))):
            pts = np.array([[cx - 3.0, cy - 3.0], [cx + 3.0, cy + 3.0]])
            gt_inst = sio.Instance.from_numpy(
                pts, skeleton=skeleton, track=gt_tracks[k]
            )
            gt_mask = sio.UserSegmentationMask.from_numpy(_square(64, 64, cy, cx, 5))
            gt_mask.instance = gt_inst  # the track lives on the instance only
            gt_insts.append(gt_inst)
            gt_masks.append(gt_mask)

            pred_inst = sio.PredictedInstance.from_numpy(
                pts, skeleton=skeleton, score=1.0, point_scores=np.ones(2)
            )
            pred_mask = sio.PredictedSegmentationMask.from_numpy(
                _square(64, 64, cy, cx, 5), score=0.9, track=pred_tracks[k]
            )
            if pred_links:
                pred_mask.instance = pred_inst
            pred_insts.append(pred_inst)
            pred_masks.append(pred_mask)
        gt_frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=frame_idx, instances=gt_insts, masks=gt_masks
            )
        )
        pred_frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                instances=pred_insts,
                masks=pred_masks,
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


def test_cli_auto_carrier_scores_a_mask_tracked_pose_and_mask_file(tmp_path):
    """`--carrier auto` picks the carrier the tracks are ON.

    It used to pick pose whenever the prediction had instances, then report
    the (untracked) poses as untracked and exit 0 with nothing scored.
    """
    from sleap_nn.cli import cli

    gt, pred = _pose_and_mask()
    out_path = tmp_path / "ids.json"
    result = CliRunner().invoke(cli, _cli_args(gt, pred, tmp_path, out_path))

    assert result.exit_code == 0, result.output
    saved = json.loads(out_path.read_text())
    assert saved["carrier"] == "mask"
    # GT masks carry no track; they resolve through their linked instances.
    assert saved["n_gt_dets"] == 12
    assert saved["n_matched"] == 12
    assert saved["idf1"] == pytest.approx(1.0)


@pytest.mark.parametrize("entry", ENTRIES)
def test_tracks_resolve_through_the_pose_mask_link(entry, tmp_path):
    """Each carrier reads a missing track off its linked detection.

    Mask carrier: the GT masks inherit their instances' tracks. Pose carrier:
    the untracked predicted poses inherit their linked masks' tracks.
    """
    gt, pred = _pose_and_mask(pred_links=True)

    as_mask = _score(entry, gt, pred, tmp_path, carrier="mask")
    assert as_mask["n_gt_dets"] == 12
    assert as_mask["idf1"] == pytest.approx(1.0)

    as_pose = _score(entry, gt, pred, tmp_path, carrier="pose")
    assert as_pose["n_pred_untracked"] == 0
    assert as_pose["n_matched"] == 12
    assert as_pose["idf1"] == pytest.approx(1.0)


def _untracked_prediction():
    gt, pred = _synthetic_pose(10)
    for lf in pred:
        for inst in lf.instances:
            inst.track = None
    return gt, pred


def _untracked_ground_truth():
    gt, pred = _synthetic_pose(10)
    for lf in gt:
        for inst in lf.instances:
            inst.track = None
    return gt, pred


def _disjoint_frames():
    gt, pred = _synthetic_pose(10)
    for lf in pred:
        lf.frame_idx += 1000
    return gt, pred


@pytest.mark.parametrize(
    "make,carrier,expected",
    [
        (_untracked_prediction, "auto", "0 tracked detections on either carrier"),
        (_untracked_prediction, "pose", "0 tracked pose detections"),
        (_untracked_ground_truth, "auto", "no tracked ground-truth pose detections"),
        (_disjoint_frames, "auto", "shares no frame with the ground truth"),
        (_pose_and_mask, "pose", "pass --carrier mask"),
    ],
)
def test_cli_exits_nonzero_when_nothing_is_scoreable(tmp_path, make, carrier, expected):
    """Nothing to score is a failure the shell can see, with the reason and no JSON."""
    from sleap_nn.cli import cli

    gt, pred = make()
    out_path = tmp_path / "ids.json"
    result = CliRunner().invoke(
        cli, _cli_args(gt, pred, tmp_path, out_path, carrier=carrier)
    )

    assert result.exit_code != 0, result.output
    assert expected in _plain(result.output)
    assert not out_path.exists()


# --- C5: untracked ground truth is don't-care ---------------------------------
@pytest.mark.parametrize("entry", ENTRIES)
def test_untracked_ground_truth_is_dont_care(entry, tmp_path):
    """A correct prediction on an animal the GT left untracked is not an FP.

    Filtering untracked GT out BEFORE matching turned every prediction on it
    into a false positive: one of two animals tracked, perfect tracker, IDP 0.5.
    """
    gt, pred = _synthetic_pose(10)
    for lf in gt:
        lf.instances[1].track = None
    m = _score(entry, gt, pred, tmp_path)

    assert m["idp"] == pytest.approx(1.0)
    assert m["idf1"] == pytest.approx(1.0)
    assert m["n_gt_dets"] == 10
    assert m["n_gt_untracked"] == 10
    assert m["n_dont_care"] == 10
    assert m["n_pred_tracks"] == 2


def test_dont_care_does_not_forgive_unmatched_predictions():
    """A prediction on NO ground truth at all is still a false positive."""
    gt, pred = _synthetic_pose(10)
    for lf in gt:
        lf.instances[1].track = None
    stray = sio.Track("p_stray")
    for lf in pred:
        lf.instances.append(
            sio.PredictedInstance.from_numpy(
                np.array([[600.0, 600.0], [612.0, 600.0]]),
                skeleton=pred.skeletons[0],
                score=1.0,
                point_scores=np.ones(2),
                track=stray,
            )
        )
    m = identity_metrics(gt, pred, "pose")

    assert m.n_dont_care == 10
    # 10 IDTP; the 10 stray detections are IDFP.
    assert m.idp == pytest.approx(0.5)
    assert m.idr == pytest.approx(1.0)


# --- C6: IDF1 includes detection errors; idf1_matched factors them out --------
@pytest.mark.parametrize("entry", ENTRIES)
def test_idf1_counts_detector_misses_and_idf1_matched_does_not(entry, tmp_path):
    """The docs said IDF1 never penalizes the detector's misses -- it does.

    IDFN counts every missed GT detection, so a perfect tracker on a detector
    that misses 10 of 40 detections scores IDF1 < 1. `idf1_matched` is the
    identity score over matched detections alone.
    """
    gt, pred = _synthetic_pose(20, drop=range(10, 15))
    m = _score(entry, gt, pred, tmp_path)
    assert m["idf1"] == pytest.approx(2 * 30 / (2 * 30 + 0 + 10))
    assert m["idf1_matched"] == pytest.approx(1.0)

    gt, pred = _synthetic_pose(20, swap_at=[10])
    swapped = _score(entry, gt, pred, tmp_path)
    assert swapped["idf1_matched"] == pytest.approx(0.5)


# --- C7: motion_diagnostic per video, per frame -------------------------------
def test_cli_motion_diagnostic_normalizes_by_frame_gap(tmp_path):
    """GT labeled every 10th frame of continuous video is continuous, not sparse.

    Without dividing by the frame gap, a 1 px/frame animal labeled every 10th
    frame moved 10 px per "step" against a 12 px body and was called sparse.
    """
    from sleap_nn.cli import cli

    gt, pred = _synthetic_pose(100, step=1.0)
    gt.labeled_frames = gt.labeled_frames[::10]
    out_path = tmp_path / "ids.json"
    result = CliRunner().invoke(cli, _cli_args(gt, pred, tmp_path, out_path))

    assert result.exit_code == 0, result.output
    motion = json.loads(out_path.read_text())["motion_diagnostic"]
    assert motion["is_continuous"] is True
    assert motion["median_frame_gap"] == pytest.approx(10.0)
    assert motion["median_step_px"] == pytest.approx(1.0)


def test_cli_motion_diagnostic_follows_each_video_separately(tmp_path):
    """Same track names in two videos are two animals, not one teleporting one.

    Sorting by frame index alone interleaved the videos, so each "step" jumped
    between recordings and genuine video was called sparse.
    """
    from sleap_nn.cli import cli

    gt_a, pred_a = _synthetic_pose(20)
    gt_b, pred_b = _synthetic_pose(20, fname="second_video.mp4", x_shift=300.0)
    gt, pred = _concat(gt_a, gt_b), _concat(pred_a, pred_b)
    assert motion_diagnostic(gt, "pose")["is_continuous"] is True

    out_path = tmp_path / "ids.json"
    result = CliRunner().invoke(cli, _cli_args(gt, pred, tmp_path, out_path))
    assert result.exit_code == 0, result.output
    motion = json.loads(out_path.read_text())["motion_diagnostic"]
    assert motion["is_continuous"] is True
    assert motion["median_step_px"] == pytest.approx(1.0)
