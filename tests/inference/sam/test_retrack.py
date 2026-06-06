"""Tests for mask-based re-tracking (`sleap_nn.inference.sam.retrack`).

The end-to-end test mirrors prototype P3: a (simulated) pose tracker produced
track labels with an identity SWAP over a window of frames; identity-consistent
per-frame masks are used to detect and correct the swap purely from
mask<->pose overlap.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.inference.sam.reconciliation import require_min_keypoints_inside
from sleap_nn.inference.sam.retrack import RetrackResult, retrack


@pytest.fixture
def skeleton():
    """A simple 3-node skeleton."""
    return sio.Skeleton(nodes=[sio.Node("a"), sio.Node("b"), sio.Node("c")])


def _make_instance(cx, cy, skeleton, track=None, predicted=False):
    """Build a 3-node instance centered near (cx, cy)."""
    coords = np.array([[cx, cy], [cx + 5, cy + 5], [cx - 5, cy - 5]], dtype=float)
    if predicted:
        return sio.PredictedInstance.from_numpy(
            coords,
            skeleton=skeleton,
            point_scores=np.ones(3),
            score=1.0,
            track=track,
        )
    return sio.Instance.from_numpy(coords, skeleton=skeleton, track=track)


def _make_mask(cx, cy, size=12, shape=(100, 100)):
    """Build a square binary mask centered at (cx, cy)."""
    m = np.zeros(shape, dtype=bool)
    m[cy - size : cy + size, cx - size : cx + size] = True
    return m


def _build_swap_clip(skeleton, n_frames=6, swap_window=(2, 4), predicted=False):
    """Two-mouse clip with a track-label swap injected over ``swap_window``.

    Returns (labeled_frames, masks, object_ids, tracks). obj_id 0 always covers
    mouse at (20, 20); obj_id 1 always covers mouse at (70, 70) (masks are the
    *correct* identity signal).
    """
    t0 = sio.Track("mouse0")
    t1 = sio.Track("mouse1")
    lo, hi = swap_window
    frames, masks, obj_ids = [], [], []
    for f in range(n_frames):
        i0 = _make_instance(20, 20, skeleton, track=t0, predicted=predicted)
        i1 = _make_instance(70, 70, skeleton, track=t1, predicted=predicted)
        if lo <= f < hi:  # injected swap
            i0.track, i1.track = t1, t0
        frames.append(sio.LabeledFrame(video=None, frame_idx=f, instances=[i0, i1]))
        masks.append(np.stack([_make_mask(20, 20), _make_mask(70, 70)]))
        obj_ids.append(np.array([0, 1]))
    return frames, masks, obj_ids, (t0, t1)


def _names(frames):
    return [
        [inst.track.name if inst.track else None for inst in lf.instances]
        for lf in frames
    ]


class TestRetrackEndToEnd:
    """End-to-end re-tracking on a simulated identity swap (P3 scenario)."""

    def test_corrects_minority_swap(self, skeleton):
        """A minority-window swap is fully corrected by majority-vote naming."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton)

        # Pre-condition: frames 2 and 3 are swapped.
        before = _names(frames)
        assert before[2] == ["mouse1", "mouse0"]
        assert before[3] == ["mouse1", "mouse0"]

        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )

        assert isinstance(result, RetrackResult)
        assert result.canonical_map == {0: "mouse0", 1: "mouse1"}
        assert result.num_matched == 12  # 2 instances x 6 frames
        assert result.num_relabeled == 4  # 2 instances x 2 swapped frames

        after = _names(frames)
        for f in range(6):
            assert after[f] == ["mouse0", "mouse1"], f"frame {f} not corrected"

    def test_accuracy_before_after(self, skeleton):
        """Track-name accuracy goes from <1.0 to 1.0 after re-tracking."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton)
        truth = {0: "mouse0", 1: "mouse1"}  # pose order -> true name

        def accuracy(names):
            correct = total = 0
            for frame_names in names:
                for j, got in enumerate(frame_names):
                    total += 1
                    correct += int(got == truth[j])
            return correct / total

        acc_before = accuracy(_names(frames))
        retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        acc_after = accuracy(_names(frames))

        assert acc_before < 1.0
        assert acc_after == 1.0

    def test_works_on_predicted_instances(self, skeleton):
        """Re-tracking operates on `PredictedInstance`s (the inference case)."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton, predicted=True)
        assert all(
            isinstance(inst, sio.PredictedInstance)
            for lf in frames
            for inst in lf.instances
        )
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        assert result.num_relabeled == 4
        assert _names(frames) == [["mouse0", "mouse1"]] * 6


class TestRetrackBehavior:
    """Targeted behavior / option tests."""

    def test_no_swap_is_noop(self, skeleton):
        """A clean clip is left untouched (zero relabels)."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton, swap_window=(0, 0))
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        assert result.num_relabeled == 0
        assert _names(frames) == [["mouse0", "mouse1"]] * 6

    def test_in_place_false_preserves_input(self, skeleton):
        """``in_place=False`` leaves the caller's frames untouched."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton)
        before = _names(frames)
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
            in_place=False,
        )
        # Original frames unchanged...
        assert _names(frames) == before
        # ...but the correction is returned on the result's frames.
        assert result.num_relabeled == 4
        assert result.canonical_map == {0: "mouse0", 1: "mouse1"}
        assert result.labeled_frames is not frames
        assert _names(result.labeled_frames) == [["mouse0", "mouse1"]] * 6

    def test_in_place_true_returns_same_frames(self, skeleton):
        """``in_place=True`` returns the caller's own frame objects."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton)
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        assert all(a is b for a, b in zip(result.labeled_frames, frames))

    def test_explicit_anchor_frames(self, skeleton):
        """Explicit (clean) anchor frames define identity for all frames."""
        frames, masks, obj_ids, _ = _build_swap_clip(skeleton)
        # Anchor only on clean frames 0 and 5.
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
            anchor_frame_indices=[0, 5],
        )
        assert result.anchor_frames == [0, 5]
        assert result.canonical_map == {0: "mouse0", 1: "mouse1"}
        assert _names(frames) == [["mouse0", "mouse1"]] * 6

    def test_cross_anchor_reassignment_routes_per_frame(self, skeleton):
        """A genuine cross-anchor identity change flips at the nearest anchor.

        Two sparse anchors (frames 0 and 10) disagree on what obj_id 0 is
        (mouse0 vs mouse1) -> obj_id is ambiguous (tie) and must resolve via the
        nearest anchor, not a global canonical name.
        """
        t0 = sio.Track("mouse0")
        t1 = sio.Track("mouse1")
        frames, masks, obj_ids = [], [], []
        for f in range(11):
            # Anchor frames carry GT (user) tracks; others are predicted/untracked.
            if f == 0:
                tracks = (t0, t1)  # obj0 -> mouse0
                pred = False
            elif f == 10:
                tracks = (t1, t0)  # obj0 -> mouse1 (real reassignment)
                pred = False
            else:
                tracks = (None, None)
                pred = True
            i0 = _make_instance(20, 20, skeleton, track=tracks[0], predicted=pred)
            i1 = _make_instance(70, 70, skeleton, track=tracks[1], predicted=pred)
            frames.append(sio.LabeledFrame(video=None, frame_idx=f, instances=[i0, i1]))
            masks.append(np.stack([_make_mask(20, 20), _make_mask(70, 70)]))
            obj_ids.append(np.array([0, 1]))

        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        # Only the two GT frames anchor (predicted frames are not anchors).
        assert result.anchor_frames == [0, 10]
        # obj_id 0 is ambiguous across anchors -> not in the canonical map.
        assert 0 not in result.canonical_map
        names = _names(frames)
        # Frames nearer anchor 0 -> obj0 is mouse0; nearer anchor 10 -> mouse1.
        assert names[1][0] == "mouse0"
        assert names[9][0] == "mouse1"

    def test_exclude_nodes(self, skeleton):
        """Excluded nodes do not contribute to keypoints-inside matching."""
        frames, masks, obj_ids, _ = _build_swap_clip(
            skeleton, n_frames=2, swap_window=(0, 0)
        )
        # Excluding all nodes makes every match fail the >=1 predicate.
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            exclude_nodes={"a", "b", "c"},
        )
        assert result.num_matched == 0
        assert result.num_relabeled == 0


class TestRetrackInputHandling:
    """Input normalization and validation."""

    def test_squeezes_4d_masks(self, skeleton):
        """`(N, 1, H, W)` SAM3 mask format is squeezed transparently."""
        frames, masks2d, obj_ids, _ = _build_swap_clip(
            skeleton, n_frames=2, swap_window=(0, 0)
        )
        masks4d = [m[:, None, :, :] for m in masks2d]  # (N, 1, H, W)
        result = retrack(
            frames,
            masks4d,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        assert result.num_matched == 4  # 2 instances x 2 frames

    def test_drops_padded_object_ids(self, skeleton):
        """Padded entries (object_id < 0) are dropped before matching."""
        frames, masks, obj_ids, _ = _build_swap_clip(
            skeleton, n_frames=2, swap_window=(0, 0)
        )
        # Pad each frame with a third sentinel mask/obj_id.
        padded_masks = [
            np.concatenate([m, np.zeros((1,) + m.shape[1:], dtype=bool)], axis=0)
            for m in masks
        ]
        padded_ids = [np.array([0, 1, -1]) for _ in obj_ids]
        result = retrack(
            frames,
            padded_masks,
            padded_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        # Padding ignored: still a clean 2-mouse match, no spurious obj_id.
        assert -1 not in result.canonical_map
        assert result.canonical_map == {0: "mouse0", 1: "mouse1"}

    def test_aligns_scores_after_padding_drop(self, skeleton):
        """Per-frame scores are realigned when padding is dropped."""
        frames, masks, obj_ids, _ = _build_swap_clip(
            skeleton, n_frames=2, swap_window=(0, 0)
        )
        padded_masks = [
            np.concatenate([m, np.zeros((1,) + m.shape[1:], dtype=bool)], axis=0)
            for m in masks
        ]
        padded_ids = [np.array([0, 1, -1]) for _ in obj_ids]
        scores = [np.array([0.9, 0.8, 0.0]) for _ in obj_ids]
        result = retrack(
            frames,
            padded_masks,
            padded_ids,
            skeleton,
            scores=scores,
            match_predicates=[require_min_keypoints_inside(1)],
        )
        kept_scores = sorted(a.sam3_score for a in result.assignments)
        assert kept_scores == pytest.approx([0.8, 0.8, 0.9, 0.9])

    def test_empty_frame_is_handled(self, skeleton):
        """A frame with no instances/masks contributes nothing."""
        empty = sio.LabeledFrame(video=None, frame_idx=0, instances=[])
        result = retrack(
            [empty],
            [np.zeros((0, 100, 100), dtype=bool)],
            [np.zeros((0,), dtype=int)],
            skeleton,
        )
        assert result.num_matched == 0
        assert result.num_relabeled == 0
        assert result.canonical_map == {}

    def test_length_mismatch_raises(self, skeleton):
        """Mismatched masks/object_ids/frames lengths raise ValueError."""
        frames, masks, obj_ids, _ = _build_swap_clip(
            skeleton, n_frames=2, swap_window=(0, 0)
        )
        with pytest.raises(ValueError, match="same length"):
            retrack(frames, masks[:1], obj_ids, skeleton)

    def test_scores_length_mismatch_raises(self, skeleton):
        """Mismatched scores length raises ValueError."""
        frames, masks, obj_ids, _ = _build_swap_clip(
            skeleton, n_frames=2, swap_window=(0, 0)
        )
        with pytest.raises(ValueError, match="scores"):
            retrack(frames, masks, obj_ids, skeleton, scores=[np.ones(2)])

    def test_fallback_names_for_unmatched_objects(self, skeleton):
        """obj_ids never anchored fall back to provided names."""
        # Single untracked predicted instance -> no anchor name for its obj_id.
        inst = _make_instance(20, 20, skeleton, track=None, predicted=True)
        frames = [sio.LabeledFrame(video=None, frame_idx=0, instances=[inst])]
        masks = [np.stack([_make_mask(20, 20)])]
        obj_ids = [np.array([7])]
        result = retrack(
            frames,
            masks,
            obj_ids,
            skeleton,
            match_predicates=[require_min_keypoints_inside(1)],
            fallback_names={7: "extra"},
        )
        assert result.num_matched == 1
        assert frames[0].instances[0].track.name == "extra"
