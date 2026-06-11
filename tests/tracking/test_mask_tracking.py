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

from sleap_nn.evaluation import _mask_iou
from sleap_nn.tracking.tracker import Tracker
from sleap_nn.tracking.utils import (
    MaskFeature,
    _mask_feature_centroid,
    _mask_feature_from_dense,
    compute_mask_iou,
    compute_mask_iou_dist,
    count_valid_points,
    get_centroid,
    get_mask,
    get_mask_centroid,
    is_segmentation_mask,
    translate_mask_feature,
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


def test_get_mask_returns_compact_feature():
    """get_mask returns a MaskFeature (tight bbox crop + offset + area)."""
    m = _mask(20, 20, r=10)
    feat = get_mask(m)
    assert isinstance(feat, MaskFeature)
    assert feat.area == int(m.area) > 0
    # Crop is the foreground bbox, not the full 80x80 canvas.
    assert feat.crop.dtype == bool
    assert feat.crop.shape[0] < 80 and feat.crop.shape[1] < 80
    assert int(feat.crop.sum()) == feat.area
    # Idempotent on a MaskFeature; dense ndarray -> tight-bbox feature.
    assert get_mask(feat) is feat
    arr = _disk(40, 40, 20, 20, 6)
    f2 = get_mask(arr)
    assert isinstance(f2, MaskFeature) and f2.area == int(arr.sum())


def _stride_mask(cy, cx, r=10, h=80, w=80, score=0.9, stride=2):
    """A mask stored at output-stride resolution (scale=1/stride), as the default
    inference path (``full_res_masks=False``) emits."""
    full = _disk(h, w, cy, cx, r)
    small = full[::stride, ::stride]
    s = 1.0 / stride
    return sio.PredictedSegmentationMask.from_numpy(small, scale=(s, s), score=score)


def test_get_mask_decodes_output_stride_to_image_grid():
    """get_mask decodes stride-stored masks (scale!=1) onto the IMAGE grid.

    Regression (the finalize-review blocker): the bbox-crop shortcut sliced the
    stride-res ``.data`` with image-space ``.bbox`` indices, yielding an empty
    feature (area 0) -> ``compute_mask_iou`` 1.0 for every pair -> scrambled
    identity on the DEFAULT inference path (``full_res_masks=False``, scale~0.5).
    """
    from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

    m = _stride_mask(40, 40)
    assert tuple(m.scale) != (1.0, 1.0)  # genuinely stride-stored
    feat = get_mask(m)
    ref = _mask_feature_from_dense(decode_mask_to_image_res(m))
    assert feat.area == ref.area > 0  # NOT the buggy 0

    # Two shifted stride masks: IoU matches the image-grid reference and is a
    # real partial overlap, not the degenerate empty-vs-empty 1.0.
    m2 = _stride_mask(46, 46)
    iou = compute_mask_iou(get_mask(m), get_mask(m2))
    iou_ref = compute_mask_iou(
        _mask_feature_from_dense(decode_mask_to_image_res(m)),
        _mask_feature_from_dense(decode_mask_to_image_res(m2)),
    )
    assert np.isclose(iou, iou_ref)
    assert 0.0 < iou < 1.0


def test_compute_mask_iou_matches_full_canvas_reference():
    """The cropped fast IoU is numerically identical to _mask_iou (fuzz)."""
    rng = np.random.RandomState(1)
    for _ in range(120):
        h, w = rng.randint(20, 60), rng.randint(20, 60)
        a = rng.rand(h, w) > rng.uniform(0.3, 0.9)
        b = rng.rand(h, w) > rng.uniform(0.3, 0.9)
        if rng.rand() < 0.1:  # exercise empty + shape-mismatch
            a = np.zeros((h, w), dtype=bool)
        if rng.rand() < 0.1:
            b = np.zeros((rng.randint(20, 60), rng.randint(20, 60)), dtype=bool)
        assert abs(compute_mask_iou(a, b) - _mask_iou(a, b)) < 1e-9


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


# ---------------------------------------------------------------------------
# Blended scorer: mask-IoU + centroid-distance fallback (Phase A)
# ---------------------------------------------------------------------------


def test_mask_iou_dist_equals_mask_iou_when_overlapping():
    """Whenever the masks overlap, mask_iou_dist == mask_iou (no behavior change)."""
    a = _disk(80, 80, 40, 40, 10)
    b = _disk(80, 80, 44, 44, 10)  # overlaps a
    assert compute_mask_iou(a, b) > 0.0
    assert compute_mask_iou_dist(a, b) == compute_mask_iou(a, b)
    # Identical + empty/empty degenerate contract preserved.
    assert compute_mask_iou_dist(a, a) == 1.0
    assert compute_mask_iou_dist(np.zeros((5, 5), bool), np.zeros((5, 5), bool)) == 1.0


def test_mask_iou_dist_no_overlap_fallback_monotonic_and_below_iou():
    """Non-overlapping masks fall back to centroid proximity in (-1, 0):
    closer wins, and any non-overlap stays strictly below any real IoU."""
    a = MaskFeature(np.ones((4, 4), bool), 0, 0, 16)
    near = MaskFeature(np.ones((4, 4), bool), 0, 8, 16)  # disjoint, close
    far = MaskFeature(np.ones((4, 4), bool), 0, 40, 16)  # disjoint, far
    s_near, s_far = compute_mask_iou_dist(a, near), compute_mask_iou_dist(a, far)
    assert -1.0 < s_far < s_near < 0.0
    # A barely-overlapping pair must still beat the closest non-overlapping one.
    overlap = MaskFeature(np.ones((4, 4), bool), 3, 3, 16)
    assert compute_mask_iou_dist(a, overlap) > 0.0 > s_near
    # Empty centroid (no foreground) -> neutral 0.0, not a crash.
    assert compute_mask_iou_dist(a, MaskFeature(np.zeros((0, 0), bool), 0, 0, 0)) == 0.0


def test_mask_centroid_translate_and_get_centroid_dispatch():
    """Mask centroid (image coords), rigid translate, and get_centroid dispatch."""
    f = MaskFeature(np.ones((4, 4), bool), y0=10, x0=20, area=16)
    assert np.allclose(_mask_feature_centroid(f), [21.5, 11.5])  # x=20+1.5, y=10+1.5
    # Public get_mask_centroid on a real mask matches its disk center.
    m = _mask(30, 25, r=8)
    cx, cy = get_mask_centroid(m)
    assert abs(cx - 25) < 1.0 and abs(cy - 30) < 1.0
    # get_centroid dispatches to the mask centroid for mask objects.
    assert np.allclose(get_centroid(m), get_mask_centroid(m))
    # Rigid translate shifts the offset, preserves crop/area; NaN disp holds.
    t = translate_mask_feature(f, np.array([5.0, -3.0]))
    assert (t.x0, t.y0, t.area) == (25, 7, 16) and t.crop is f.crop
    assert translate_mask_feature(f, np.array([np.nan, 1.0])) is f


@pytest.mark.parametrize("candidates_method", ["fixed_window", "local_queues"])
def test_tracker_masks_no_overlap_keeps_identity_with_dist(candidates_method):
    """Fast motion: consecutive masks of the same animal do NOT overlap, so plain
    mask_iou is 0 everywhere. mask_iou_dist's centroid fallback still binds each
    detection to its same-lane predecessor across two well-separated lanes."""
    tracker = Tracker.from_config(
        features="masks",
        scoring_method="mask_iou_dist",
        candidates_method=candidates_method,
        window_size=5,
    )
    # Two lanes far apart (rows 15 and 65); each steps 25 px/frame in x with r=8 on
    # a wide canvas, so consecutive same-animal masks are disjoint (step > 2r) but
    # the lanes never cross. Identity must follow the lane, not flip.
    names = {"top": set(), "bottom": set()}
    for fi in range(5):
        x = 10 + 25 * fi
        out = tracker.track(
            [_mask(15, x, r=8, w=200), _mask(65, x, r=8, w=200)], frame_idx=fi
        )
        for m in out:
            lane = "top" if m.bbox[1] < 40 else "bottom"
            names[lane].add(m.track.name)
    # Each lane kept a single stable identity across all frames.
    assert len(names["top"]) == 1 and len(names["bottom"]) == 1
    assert names["top"] != names["bottom"]


def test_tracker_masks_kalman_centroid_runs_and_tracks():
    """Opt-in centroid Kalman motion model on a moving 2-mask sequence: it warms
    up, fits per-track centroid filters, and keeps both identities stable (the
    candidate mask is rigidly translated to the predicted location)."""
    tracker = Tracker.from_config(
        features="masks",
        scoring_method="mask_iou_dist",
        candidates_method="local_queues",
        window_size=10,
        max_tracks=2,
        use_kalman=True,
        kf_track_features="centroid",
        kf_init_frame_count=5,
    )
    top_names, bottom_names = set(), set()
    for fi in range(16):
        x = 10 + 4 * fi  # smooth constant-velocity motion (overlapping steps)
        out = tracker.track([_mask(20, x, r=10), _mask(60, x, r=10)], frame_idx=fi)
        assert len(out) == 2
        for m in out:
            (top_names if m.bbox[1] < 40 else bottom_names).add(m.track.name)
    # No more than the 2 true identities ever appear (filter did not scramble them).
    assert len(top_names) == 1 and len(bottom_names) == 1
    assert top_names != bottom_names
    # The motion model actually engaged (filters were fit past warm-up).
    assert len(tracker._kalman_filters) >= 1
