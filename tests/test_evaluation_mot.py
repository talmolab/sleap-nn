"""Tests for the TrackEval-faithful MOT adapter ``sleap_nn.evaluation.compute_mot_metrics``.

Each case is built so the right answer is known by construction -- two animals
far apart, with identity swaps, missing frames and untracked detections
injected at known positions -- and the expected numbers are the TrackEval values
of the matching pure-core toy (``tests/tracking/test_mot_metrics.py``).
"""

import json

import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner

from sleap_nn.evaluation import (
    _mot_similarity,
    compute_mot_metrics,
    identity_metrics,
    run_identity_evaluation,
)

FNAME = "synthetic_mot_video.mp4"


def _pose_pair(
    n_frames=10,
    swap_at=(),
    missing=(),
    pred_offset=0.0,
    fname=FNAME,
    pred_names=("p0", "p1"),
    single_node=False,
):
    """Two animals moving in parallel, 280 px apart; returns ``(gt, pred)``.

    Args:
        n_frames: Number of frames.
        swap_at: Frames at which the predicted identity assignment flips.
        missing: Frames the prediction does not have at all.
        pred_offset: Horizontal offset, in pixels, of every predicted point.
        fname: Video filename, so two calls can build two different videos.
        pred_names: Names of the two predicted tracks.
        single_node: Use a one-node (centroid) skeleton.
    """
    skeleton = sio.Skeleton(["head"] if single_node else ["head", "tail"])
    video = sio.Video.from_filename(fname)
    gt_tracks = [sio.Track("g0"), sio.Track("g1")]
    pred_tracks = [sio.Track(name) for name in pred_names]
    gt_frames, pred_frames = [], []
    flipped = False
    for frame_idx in range(n_frames):
        if frame_idx in swap_at:
            flipped = not flipped
        points = []
        for x0, y0 in ((20.0, 20.0), (220.0, 220.0)):
            x = x0 + 2.0 * frame_idx
            # Two nodes spanning a non-zero box, so OKS has a real scale.
            nodes = [[x, y0]] if single_node else [[x, y0], [x + 12.0, y0 + 8.0]]
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
        if frame_idx in missing:
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


def _concat(first, second):
    """One ``Labels`` holding both inputs' frames, videos, skeletons and tracks."""
    return sio.Labels(
        labeled_frames=first.labeled_frames + second.labeled_frames,
        videos=first.videos + second.videos,
        skeletons=first.skeletons + second.skeletons,
        tracks=first.tracks + second.tracks,
    )


def _square(cy, cx, half=5, size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[cy - half : cy + half, cx - half : cx + half] = True
    return mask


def _mask_pair(n_frames=10, swap_at=()):
    """Mask-carrier twin of :func:`_pose_pair`: one square per animal."""
    video = sio.Video.from_filename(FNAME)
    gt_tracks = [sio.Track("g0"), sio.Track("g1")]
    pred_tracks = [sio.Track("p0"), sio.Track("p1")]
    gt_frames, pred_frames = [], []
    flipped = False
    for frame_idx in range(n_frames):
        if frame_idx in swap_at:
            flipped = not flipped
        squares = [_square(12 + frame_idx, 12), _square(40 + frame_idx, 48)]
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


def _assert_same_metrics(a: dict, b: dict) -> None:
    """Equal up to float summation order (1e-12), NaN equal to NaN; notes ignored."""
    assert set(a) == set(b)
    for key in a:
        if key == "notes":
            continue
        if isinstance(a[key], (str, bool)):
            assert a[key] == b[key], key
        else:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, err_msg=key)


# ---------------------------------------------------------------------------
# Known-by-construction cases
# ---------------------------------------------------------------------------
def test_perfect_tracking_scores_one():
    gt, pred = _pose_pair()
    m = compute_mot_metrics(gt, pred, "pose")

    assert m.mota == pytest.approx(1.0)
    assert m.motp == pytest.approx(1.0)
    assert m.idf1 == pytest.approx(1.0)
    assert m.hota == pytest.approx(1.0)
    assert m.det_a == m.ass_a == m.loc_a == pytest.approx(1.0)
    assert (m.idsw, m.frag, m.mt, m.pt, m.ml) == (0, 0, 2, 0, 0)
    assert (m.clr_tp, m.clr_fn, m.clr_fp) == (20, 0, 0)
    assert (m.n_gt_dets, m.n_pred_dets, m.n_frames, m.n_videos) == (20, 20, 10, 1)
    assert m.match_method == "oks"
    assert m.match_threshold == 0.5
    assert len(m.hota_alpha) == 19
    assert m.notes == []


def test_swap_halfway_matches_trackeval_toy_b():
    """One swap of two animals over 10 frames: the pure-core toy B numbers."""
    gt, pred = _pose_pair(swap_at=[5])
    m = compute_mot_metrics(gt, pred, "pose")

    assert m.idsw == 2
    assert m.mota == pytest.approx(0.9)
    assert m.idf1 == pytest.approx(0.5)
    assert m.idtp == 10
    assert m.hota == pytest.approx(np.sqrt(1 / 3))
    assert m.det_a == pytest.approx(1.0)
    assert m.ass_a == pytest.approx(1 / 3)
    assert m.hota_0 == pytest.approx(np.sqrt(1 / 3))
    np.testing.assert_allclose(m.hota_alpha, np.sqrt(1 / 3))


def test_untracked_ground_truth_is_dont_care():
    """A correct prediction on an animal the GT left untracked is no false positive."""
    gt, pred = _pose_pair()
    for lf in gt:
        lf.instances[1].track = None
    m = compute_mot_metrics(gt, pred, "pose")

    assert (m.clr_tp, m.clr_fn, m.clr_fp) == (10, 0, 0)
    assert m.mota == pytest.approx(1.0)
    assert m.idf1 == pytest.approx(1.0)
    assert m.hota == pytest.approx(1.0)
    assert m.n_gt_untracked == 10
    assert m.n_dont_care == 10
    assert m.n_pred_dets == 10
    assert any("don't-care" in note for note in m.notes)


def test_dont_care_does_not_forgive_predictions_on_nothing():
    gt, pred = _pose_pair()
    for lf in gt:
        lf.instances[1].track = None
    stray = sio.Track("p_stray")
    for lf in pred:
        lf.instances.append(
            sio.PredictedInstance.from_numpy(
                np.array([[600.0, 600.0], [612.0, 608.0]]),
                skeleton=pred.skeletons[0],
                score=1.0,
                point_scores=np.ones(2),
                track=stray,
            )
        )
    m = compute_mot_metrics(gt, pred, "pose")

    assert (m.clr_tp, m.clr_fp, m.n_dont_care) == (10, 10, 10)
    assert m.idp == pytest.approx(0.5)


def test_missing_prediction_frames_are_misses():
    gt, pred = _pose_pair(missing=range(5, 10))
    m = compute_mot_metrics(gt, pred, "pose")

    assert (m.clr_tp, m.clr_fn, m.clr_fp) == (10, 10, 0)
    assert m.mota == pytest.approx(0.5)
    assert m.det_re == pytest.approx(0.5)
    assert m.n_frames == 10
    assert m.n_frames_missing_pred == 5
    assert any("no predicted counterpart" in note for note in m.notes)


def test_untracked_predictions_are_removed_and_counted():
    gt, pred = _pose_pair()
    for lf in pred:
        lf.instances[1].track = None
    m = compute_mot_metrics(gt, pred, "pose")

    assert m.n_pred_untracked == 10
    # Their animal is simply missed; they are not false positives.
    assert (m.clr_tp, m.clr_fn, m.clr_fp) == (10, 10, 0)
    assert m.n_pred_dets == 10


def test_global_identity_resets_at_video_boundaries():
    """Per video, fresh ids in video B are fine; globally they cost IDF1, not IDSW."""
    gt_a, pred_a = _pose_pair()
    gt_b, pred_b = _pose_pair(fname="second_video.mp4", pred_names=("p2", "p3"))
    gt, pred = _concat(gt_a, gt_b), _concat(pred_a, pred_b)

    per_video = compute_mot_metrics(gt, pred, "pose")
    assert (per_video.idsw, per_video.n_videos) == (0, 2)
    assert per_video.idf1 == pytest.approx(1.0)
    assert per_video.hota == pytest.approx(1.0)
    assert (per_video.mt, per_video.pt, per_video.ml) == (4, 0, 0)

    lost = compute_mot_metrics(gt, pred, "pose", global_identity=True)
    assert lost.global_identity is True
    # Continuity resets at the boundary: no switch, no fragment across it ...
    assert (lost.idsw, lost.frag) == (0, 0)
    # ... but each global identity carries two names, half its detections each.
    assert lost.idf1 == pytest.approx(0.5)
    assert lost.ass_a < 1.0
    assert (lost.mt, lost.pt, lost.ml) == (2, 0, 0)

    # A prediction that kept the global names scores perfectly.
    _gt_b, pred_b_kept = _pose_pair(fname="second_video.mp4")
    kept = compute_mot_metrics(
        gt, _concat(pred_a, pred_b_kept), "pose", global_identity=True
    )
    assert (kept.idsw, kept.frag) == (0, 0)
    assert kept.idf1 == pytest.approx(1.0)
    assert kept.hota == pytest.approx(1.0)


def test_renaming_tracks_and_permuting_detections_changes_nothing():
    gt, pred = _pose_pair(swap_at=[3, 7], pred_offset=0.3)
    reference = compute_mot_metrics(gt, pred, "pose").as_dict()

    renamed = {"p0": sio.Track("zebra"), "p1": sio.Track("aardvark")}
    rng = np.random.default_rng(0)
    for labels in (gt, pred):
        for lf in labels:
            if labels is pred:
                for inst in lf.instances:
                    inst.track = renamed[inst.track.name]
            lf.instances = [lf.instances[i] for i in rng.permutation(2)]
    pred.tracks = list(renamed.values())

    _assert_same_metrics(compute_mot_metrics(gt, pred, "pose").as_dict(), reference)


def test_ground_truth_as_prediction_is_perfect_detection():
    """The GT-detection arm invariant: retracked GT has DetA = LocA = 1, FP = FN = 0."""
    gt, pred = _pose_pair(swap_at=[4])
    gt_as_pred, _ = _pose_pair()
    for gt_lf, pred_lf in zip(gt_as_pred, pred):
        for inst, pred_inst in zip(gt_lf.instances, pred_lf.instances):
            inst.track = pred_inst.track
    m = compute_mot_metrics(gt, gt_as_pred, "pose")

    assert (m.clr_fn, m.clr_fp) == (0, 0)
    assert m.det_a == m.det_re == m.det_pr == m.loc_a == m.motp == 1.0
    assert m.mota == pytest.approx(1 - m.idsw / m.n_gt_dets)
    assert m.hota == pytest.approx(np.sqrt(m.ass_a))


def test_duplicate_track_in_a_frame_raises():
    gt, pred = _pose_pair()
    pred.labeled_frames[3].instances[1].track = (
        pred.labeled_frames[3].instances[0].track
    )
    with pytest.raises(ValueError, match=r"Predicted track\(s\) \['p0'\].*frame 3"):
        compute_mot_metrics(gt, pred, "pose")


def test_no_frames_in_common_returns_nan_with_a_note():
    gt, pred = _pose_pair()
    _gt2, pred2 = _pose_pair(fname="other.mp4")
    pred = _concat(pred, pred2)
    for lf in pred:
        lf.frame_idx += 1000
    m = compute_mot_metrics(gt, pred, "pose")

    assert m.n_frames == 0
    assert np.isnan(m.mota) and np.isnan(m.hota)
    assert any("no frames in common" in note for note in m.notes)


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------
def test_mask_carrier_swap():
    gt, pred = _mask_pair(swap_at=[5])
    m = compute_mot_metrics(gt, pred, "mask")

    assert m.match_method == "iou"
    assert m.idsw == 2
    assert m.mota == pytest.approx(0.9)
    assert m.hota == pytest.approx(np.sqrt(1 / 3))


def test_mask_similarity_scores_empty_masks_zero():
    empty, full = np.zeros((8, 8), bool), np.ones((8, 8), bool)
    sim = _mot_similarity([empty, full], [empty, full], "iou", 0.5, 0.025)
    np.testing.assert_array_equal(sim, [[0.0, 0.0], [0.0, 1.0]])


def test_single_node_distance_maps_pixels_to_similarity():
    """sim = 1 - d / (2 * threshold): 0.5 is exactly ``threshold`` pixels."""
    gt, pred = _pose_pair(single_node=True, pred_offset=10.0)

    m = compute_mot_metrics(gt, pred, "pose")
    assert (m.match_method, m.match_threshold) == ("distance", 50.0)
    assert m.motp == pytest.approx(1 - 10 / 100)
    assert m.clr_tp == 20

    at_threshold = compute_mot_metrics(gt, pred, "pose", match_threshold=10.0)
    assert at_threshold.motp == pytest.approx(0.5)
    assert at_threshold.clr_tp == 20

    beyond = compute_mot_metrics(gt, pred, "pose", match_threshold=9.0)
    assert (beyond.clr_tp, beyond.idtp) == (0, 0)
    assert any("no tracked prediction matched" in note for note in beyond.notes)

    with pytest.raises(ValueError, match="positive pixel distance"):
        compute_mot_metrics(gt, pred, "pose", match_threshold=0.0)


def test_oks_stddev_is_threaded_into_both_blocks():
    """A 1 px offset fails OKS at stddev 0.025 and passes at 0.1."""
    gt, pred = _pose_pair(pred_offset=1.0)

    tight = compute_mot_metrics(gt, pred, "pose")
    loose = compute_mot_metrics(gt, pred, "pose", oks_stddev=0.1)
    assert (tight.clr_tp, loose.clr_tp) == (0, 20)
    assert loose.oks_stddev == 0.1

    assert identity_metrics(gt, pred, "pose").n_matched == 0
    assert identity_metrics(gt, pred, "pose", oks_stddev=0.1).n_matched == 20
    # The default is the historical value.
    _assert_same_metrics(
        identity_metrics(gt, pred, "pose", oks_stddev=0.025).as_dict(),
        identity_metrics(gt, pred, "pose").as_dict(),
    )


def test_as_dict_is_json_serializable_and_summary_names_the_headline():
    gt, pred = _pose_pair(swap_at=[5])
    m = compute_mot_metrics(gt, pred, "pose")

    json.dumps(m.as_dict())
    summary = m.summary()
    for token in ("MOTA=0.9000", "IDSW=2", "IDF1=0.5000", "HOTA=0.5774"):
        assert token in summary


# ---------------------------------------------------------------------------
# run_identity_evaluation / sleap-nn eval-tracking
# ---------------------------------------------------------------------------
def _save(gt, pred, tmp_path):
    gt_path, pred_path = tmp_path / "gt.slp", tmp_path / "pred.slp"
    gt.save(gt_path.as_posix())
    pred.save(pred_path.as_posix())
    return gt_path.as_posix(), pred_path.as_posix()


def test_run_identity_evaluation_adds_the_mot_block(tmp_path):
    from loguru import logger

    gt, pred = _pose_pair(swap_at=[5])
    gt_path, pred_path = _save(gt, pred, tmp_path)
    out_path = tmp_path / "ids.json"

    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        result = run_identity_evaluation(
            gt_path, pred_path, save_metrics=out_path.as_posix()
        )
    finally:
        logger.remove(sink)
    log = "".join(messages)

    assert result["id_switches"] == 2  # legacy block, unchanged
    assert result["mot"]["idsw"] == 2
    assert result["mot"]["mota"] == pytest.approx(0.9)
    saved = json.loads(out_path.read_text())
    assert saved["mot"]["hota"] == pytest.approx(np.sqrt(1 / 3))
    assert "Identity (legacy definitions)" in log
    assert "TrackEval-faithful MOT" in log
    assert "HOTA=0.5774" in log


def test_run_identity_evaluation_reports_a_mot_error_without_failing(tmp_path):
    """A track on two detections of one frame: legacy block kept, MOT says why."""
    gt, pred = _pose_pair()
    frame = pred.labeled_frames[2]
    frame.instances[1].track = frame.instances[0].track
    gt_path, pred_path = _save(gt, pred, tmp_path)

    result = run_identity_evaluation(gt_path, pred_path)

    assert "id_switches" in result
    assert "frame 2" in result["mot"]["error"]


def test_cli_eval_tracking_writes_mot_and_takes_oks_stddev(tmp_path):
    from sleap_nn.cli import cli

    gt, pred = _pose_pair(pred_offset=1.0)
    gt_path, pred_path = _save(gt, pred, tmp_path)
    out_path = tmp_path / "ids.json"
    args = ["eval-tracking", "-g", gt_path, "-p", pred_path, "-s", str(out_path)]

    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    tight = json.loads(out_path.read_text())
    assert tight["mot"]["clr_tp"] == 0 and tight["n_matched"] == 0

    result = CliRunner().invoke(cli, args + ["--oks_stddev", "0.1"])
    assert result.exit_code == 0, result.output
    loose = json.loads(out_path.read_text())
    assert loose["mot"]["clr_tp"] == 20 and loose["n_matched"] == 20
    assert loose["mot"]["oks_stddev"] == 0.1
    assert loose["mot"]["mota"] == pytest.approx(1.0)
    assert loose["id_switches"] == 0
