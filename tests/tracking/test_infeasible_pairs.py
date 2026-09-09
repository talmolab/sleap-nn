"""Infeasible (non-finite-cost) pairings must never DELETE a detection.

`Tracker.assign_tracks` drops a pairing whose original cost was non-finite: the
detection had no valid candidate for that track, so persisting the match would
steal an arbitrary identity and write a `-inf` tracking score. The dropped row is
meant to spawn a fresh track instead.

That only holds when a fresh track can actually be minted. Under
`candidates_method="local_queues"` at `max_tracks`, `get_new_track_id()` returns
`None` and `update_tracks` filters the detection out entirely -- so the drop
silently DELETED it from the tracked output. NaN scores arise on the plain
geometry-only path too (a track whose candidates all fail `min_match_points`, or
an all-NaN detection making `compute_oks` 0/0), so this was a regression on the
default path, not just the appearance one.
"""

import numpy as np
import sleap_io as sio

from sleap_nn.tracking.tracker import Tracker

_SKEL = sio.Skeleton(nodes=["a", "b", "c"], name="s")


def _pose(x, y, n_visible=3):
    """A predicted pose at ``(x, y)`` with ``n_visible`` non-NaN nodes."""
    pts = np.full((3, 2), np.nan)
    for i in range(n_visible):
        pts[i] = (x + i, y)
    return sio.PredictedInstance.from_numpy(
        pts, skeleton=_SKEL, score=0.9, point_scores=np.ones(3)
    )


def _tracker(**over):
    kwargs = dict(
        candidates_method="local_queues",
        window_size=5,
        features="keypoints",
        scoring_method="oks",
        min_match_points=2,
        max_tracks=2,
    )
    kwargs.update(over)
    return Tracker.from_config(**kwargs)


def test_geometry_only_detection_is_not_deleted_at_max_tracks():
    """The review's repro: frame 1 must emit BOTH detections, not one.

    Frame 0 seeds two tracks, one of them from a barely-visible pose whose
    candidates then all fail `min_match_points` -> that track's whole score column
    is NaN -> inf cost. At `max_tracks=2` the dropped row cannot spawn a third
    track, so the detection would vanish; the forced match is kept instead.
    """
    tracker = _tracker()
    tracker.track([_pose(10, 10), _pose(100, 100, n_visible=1)], frame_idx=0)
    out = tracker.track([_pose(11, 10), _pose(101, 100)], frame_idx=1)

    assert len(out) == 2, "a detection was deleted from the tracked output"
    assert all(i.track is not None for i in out)
    assert len({i.track.name for i in out}) == 2, "both detections share a track"


def test_infeasible_pair_still_spawns_a_fresh_track_when_it_can():
    """With headroom under `max_tracks`, an infeasible pairing IS dropped.

    That is the behavior the gate exists for -- an embedding-less detection must
    not steal an identity -- so it has to survive the "never delete" fix.
    """
    tracker = _tracker(max_tracks=None)
    tracker.track([_pose(10, 10), _pose(100, 100, n_visible=1)], frame_idx=0)
    out = tracker.track([_pose(11, 10), _pose(101, 100)], frame_idx=1)

    assert len(out) == 2
    names = {i.track.name for i in out}
    assert len(names) == 2
    # The filtered track's column was all-NaN, so its row spawned a NEW track
    # rather than being forced onto it.
    assert any(n not in ("track_0", "track_1") for n in names), names


def test_no_detection_is_deleted_under_greedy_matching():
    """`greedy_matching` returns infeasible pairs too, so it needs the same care."""
    tracker = _tracker(track_matching_method="greedy")
    tracker.track([_pose(10, 10), _pose(100, 100, n_visible=1)], frame_idx=0)
    out = tracker.track([_pose(11, 10), _pose(101, 100)], frame_idx=1)
    assert len(out) == 2


def test_available_new_tracks_reports_headroom():
    """The capacity query must not mutate the queue (it mirrors `get_new_track_id`)."""
    from sleap_nn.tracking.candidates.fixed_window import FixedWindowCandidates
    from sleap_nn.tracking.candidates.local_queues import LocalQueueCandidates

    assert FixedWindowCandidates().available_new_tracks() is None
    assert LocalQueueCandidates(max_tracks=None).available_new_tracks() is None

    cand = LocalQueueCandidates(max_tracks=2)
    assert cand.available_new_tracks() == 2
    assert cand.get_new_track_id() == 0
    cand.current_tracks.append(0)
    assert cand.available_new_tracks() == 1
    assert cand.get_new_track_id() == 1
    cand.current_tracks.append(1)
    assert cand.available_new_tracks() == 0
    assert cand.get_new_track_id() is None
    # Still 0 -- the query is side-effect free.
    assert cand.available_new_tracks() == 0


def test_unpaired_detection_keeps_its_slot():
    """A drop must not take the `max_tracks` slot an UNPAIRED detection needed.

    3 detections, 2 existing tracks, 1 slot of headroom. The matcher pairs two of
    them (one infeasibly) and leaves the third unpaired. Both the dropped row and
    the unpaired row would spawn from that single slot, so a naive drop deleted
    whichever `update_tracks` reached second. Reserving the unpaired rows' slots
    keeps every detection.
    """
    tracker = _tracker(max_tracks=3)
    tracker.track([_pose(10, 10), _pose(100, 100, n_visible=1)], frame_idx=0)
    out = tracker.track([_pose(11, 10), _pose(101, 100), _pose(200, 200)], frame_idx=1)
    assert len(out) == 3, "a detection was deleted competing for the last slot"
    assert all(i.track is not None for i in out)
