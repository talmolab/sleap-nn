"""Tests for mask-IoU tracking of bottom-up segmentation masks (#619).

Covers the tracking helpers added for ``sio.PredictedSegmentationMask`` —
``get_mask`` / ``compute_mask_iou`` / ``count_valid_points`` /
``is_segmentation_mask`` in ``sleap_nn.tracking.utils`` — and end-to-end track
assignment via ``Tracker`` with ``features="masks"``/``scoring_method="mask_iou"``
on both candidate makers. ``apply_tracking`` integration (auto-defaults, mask
preservation, forbidden combos) lives in ``tests/inference/test_tracking.py``.
"""

import numpy as np
import pytest

import sleap_io as sio

from sleap_nn.tracking.tracker import Tracker
from sleap_nn.tracking.utils import (
    compute_mask_iou,
    count_valid_points,
    get_mask,
    is_segmentation_mask,
)


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


def _mask(cy, cx, r=10, h=80, w=80, score=0.9):
    return sio.PredictedSegmentationMask.from_numpy(_disk(h, w, cy, cx, r), score=score)


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------


def test_compute_mask_iou_identical_disjoint_partial():
    """mask-IoU: 1.0 identical, 0.0 disjoint, exact for partial overlap."""
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True  # 16 px
    assert compute_mask_iou(a, a) == 1.0

    b = np.zeros((10, 10), dtype=bool)
    b[6:9, 6:9] = True  # disjoint
    assert compute_mask_iou(a, b) == 0.0

    c = np.zeros((10, 10), dtype=bool)
    c[4:8, 2:6] = True  # overlaps a in 2 of 4 rows -> inter 8, union 24
    assert abs(compute_mask_iou(a, c) - (8 / 24)) < 1e-9


def test_compute_mask_iou_shape_mismatch_and_empty():
    """Shape mismatch is top-left aligned; two empty masks -> 1.0 (degenerate)."""
    a = np.zeros((10, 10), dtype=bool)
    a[1:4, 1:4] = True
    big = np.zeros((20, 20), dtype=bool)
    big[1:4, 1:4] = True
    assert compute_mask_iou(a, big) == 1.0
    assert compute_mask_iou(np.zeros((5, 5), bool), np.zeros((5, 5), bool)) == 1.0


def test_compute_mask_iou_is_similarity_not_negated():
    """mask_iou must be higher-is-better (cost negation happens in the tracker)."""
    a = _disk(40, 40, 20, 20, 10)
    assert compute_mask_iou(a, a) == 1.0  # not -1.0


def test_get_mask_decodes_and_passes_ndarray():
    """get_mask returns the bool mask data; an ndarray passes through unchanged."""
    m = _mask(20, 20)
    data = get_mask(m)
    assert isinstance(data, np.ndarray) and data.dtype == bool
    assert int(data.sum()) == int(m.area)
    arr = _disk(30, 30, 15, 15, 5)
    assert get_mask(arr) is arr


def test_is_segmentation_mask_and_count_valid_points():
    """Type predicate + validity count dispatch on mask vs keypoint instance."""
    m = _mask(20, 20, r=10)
    skel = sio.Skeleton(nodes=["a", "b"])
    inst = sio.PredictedInstance.from_numpy(
        np.array([[1.0, 2.0], [np.nan, np.nan]]), skeleton=skel, score=0.9
    )
    assert is_segmentation_mask(m) is True
    assert is_segmentation_mask(inst) is False
    # Mask: support is foreground area (px); instance: non-NaN keypoint rows.
    assert count_valid_points(m) == int(m.area) > 0
    assert count_valid_points(inst) == 1  # one visible node, one NaN


# ---------------------------------------------------------------------------
# Tracker.track() on masks (both candidate makers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidates_method", ["fixed_window", "local_queues"])
def test_tracker_masks_overlapping_same_track(candidates_method):
    """A mask overlapping its predecessor keeps the same track id."""
    tracker = Tracker.from_config(
        features="masks",
        scoring_method="mask_iou",
        candidates_method=candidates_method,
        window_size=5,
    )
    out0 = tracker.track([_mask(20, 20)], frame_idx=0)
    out1 = tracker.track([_mask(22, 22)], frame_idx=1)  # overlaps frame-0 mask
    assert out0[0].track is not None and out1[0].track is not None
    assert out0[0].track.name == out1[0].track.name
    assert out1[0].tracking_score is not None  # mask-IoU of the matched pair


@pytest.mark.parametrize("candidates_method", ["fixed_window", "local_queues"])
def test_tracker_masks_disjoint_extra_spawns_new_track(candidates_method):
    """An extra non-overlapping detection spawns a NEW track; the overlapping
    one keeps its track (mask-IoU + Hungarian leaves the disjoint mask
    unmatched, so it is assigned a fresh id)."""
    tracker = Tracker.from_config(
        features="masks",
        scoring_method="mask_iou",
        candidates_method=candidates_method,
        window_size=5,
    )
    tracker.track([_mask(20, 20)], frame_idx=0)  # -> track_0
    out1 = tracker.track([_mask(22, 22), _mask(60, 60)], frame_idx=1)
    by_lane = {("near" if m.bbox[0] < 40 else "far"): m.track.name for m in out1}
    assert by_lane["near"] != by_lane["far"]  # disjoint extra -> new track


def test_tracker_masks_empty_area_filtered_with_threshold():
    """min_new_track_points read as a pixel-area floor: a tiny mask below it
    spawns no track (validity uses mask area, not keypoints)."""
    tracker = Tracker.from_config(
        features="masks",
        scoring_method="mask_iou",
        candidates_method="fixed_window",
        min_new_track_points=50,  # require >50 px of foreground
    )
    out = tracker.track([_mask(20, 20, r=2)], frame_idx=0)  # ~13 px < 50
    assert all(m.track is None for m in out) or len(out) == 0
