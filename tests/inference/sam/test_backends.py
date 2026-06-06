"""Unit tests for the SAM1 backend (CPU; a fake SamPredictor, no real SAM)."""

import numpy as np
import pytest

from sleap_nn.inference.sam.backends import (
    SAM_PRED_IOU_MIN,
    SamBackend,
    _pick,
    _to_3ch_clahe,
    disjointify,
    own_containment,
)
from sleap_nn.inference.sam.prompts import pose_prompt


class FakeSamPredictor:
    """Stand-in for ``segment_anything.SamPredictor``.

    ``set_image`` records the encoded image size; ``predict`` returns a fixed set
    of candidate masks + predicted-IoU scores (3 candidates, ``multimask_output``
    shape) so the backend's selection / packaging can be tested deterministically.
    """

    def __init__(self, candidates):
        # candidates: list of (mask (H,W) bool, score float)
        self._candidates = candidates
        self.encoded_shape = None
        self.last_call = None

    def set_image(self, rgb):
        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        self.encoded_shape = rgb.shape[:2]

    def predict(
        self, point_coords=None, point_labels=None, box=None, multimask_output=True
    ):
        self.last_call = dict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )
        masks = np.stack([c[0] for c in self._candidates]).astype(bool)
        scores = np.array([c[1] for c in self._candidates], np.float32)
        return masks, scores, None


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


def test_to_3ch_clahe_shape_and_dtype():
    gray = (np.random.rand(20, 30) * 255).astype(np.uint8)
    rgb = _to_3ch_clahe(gray, clahe=True)
    assert rgb.shape == (20, 30, 3)
    assert rgb.dtype == np.uint8
    rgb2 = _to_3ch_clahe(gray, clahe=False)
    # No-CLAHE path just replicates channels.
    np.testing.assert_array_equal(rgb2[..., 0], gray)


def test_pick_rejects_whole_arena_candidate():
    h, w = 50, 50
    small = _disk(h, w, 25, 25, 5)
    huge = np.ones((h, w), bool)  # whole-arena over-confident candidate
    masks = np.stack([huge, small])
    scores = np.array([0.99, 0.80], np.float32)  # huge has the higher score
    box = np.array([20, 20, 30, 30], np.float32)
    # Despite the higher score, the huge candidate is area-rejected -> pick small.
    assert _pick(masks, scores, box) == 1


def test_pick_takes_best_score_among_survivors():
    h, w = 50, 50
    a = _disk(h, w, 25, 25, 5)
    b = _disk(h, w, 25, 25, 6)
    masks = np.stack([a, b])
    scores = np.array([0.7, 0.9], np.float32)
    box = np.array([10, 10, 40, 40], np.float32)
    assert _pick(masks, scores, box) == 1


def test_own_containment_score():
    mask = _disk(50, 50, 25, 25, 10)
    inside = np.array([[25.0, 25.0], [20.0, 20.0]])
    assert own_containment(mask, inside, (50, 50)) == 1.0
    half = np.array([[25.0, 25.0], [0.0, 0.0]])
    assert own_containment(mask, half, (50, 50)) == 0.5
    assert own_containment(mask, np.empty((0, 2)), (50, 50)) == 0.0


def test_disjointify_resolves_overlap_by_nearest_keypoint():
    h, w = 40, 40
    # Two overlapping masks; keypoints seed the Voronoi assignment.
    m0 = _disk(h, w, 20, 15, 12)
    m1 = _disk(h, w, 20, 25, 12)
    k0 = [[15.0, 20.0]]
    k1 = [[25.0, 20.0]]
    out = disjointify([m0, m1], [k0, k1])
    # Result is exactly disjoint.
    assert not (out[0] & out[1]).any()
    # Each instance keeps its own keypoint pixel.
    assert out[0][20, 15]
    assert out[1][20, 25]


def test_disjointify_noop_when_uncontested():
    h, w = 30, 30
    m0 = _disk(h, w, 10, 10, 5)
    m1 = _disk(h, w, 20, 20, 5)
    out = disjointify([m0, m1], [[[10.0, 10.0]], [[20.0, 20.0]]])
    np.testing.assert_array_equal(out[0], m0)
    np.testing.assert_array_equal(out[1], m1)


def test_sam_backend_masks_contract():
    h, w = 64, 64
    chosen = _disk(h, w, 32, 32, 10)
    huge = np.ones((h, w), bool)
    pred = FakeSamPredictor(
        candidates=[(huge, 0.99), (chosen, 0.85), (_disk(h, w, 32, 32, 4), 0.5)]
    )
    backend = SamBackend(pred)
    kpts = np.array([[28.0, 28.0], [36.0, 36.0]])
    prompts = [pose_prompt(kpts, (h, w))]
    masks, scores = backend.masks(np.zeros((h, w), np.uint8), prompts)
    assert len(masks) == 1 and len(scores) == 1
    # The whole-arena candidate (idx 0, score 0.99) is rejected; idx 1 wins.
    np.testing.assert_array_equal(masks[0], chosen)
    assert abs(scores[0] - 0.85) < 1e-6
    # The backend encoded the image once at the frame size.
    assert pred.encoded_shape == (h, w)
    # The pose prompt forwarded both points and the box.
    assert pred.last_call["point_coords"].shape == (2, 2)
    assert pred.last_call["box"] is not None


def test_sam_backend_empty_prompts():
    backend = SamBackend(FakeSamPredictor(candidates=[(np.zeros((8, 8), bool), 0.5)]))
    masks, scores = backend.masks(np.zeros((8, 8), np.uint8), [])
    assert masks == [] and scores == []


def test_sam_backend_pred_iou_min_attribute():
    backend = SamBackend(FakeSamPredictor(candidates=[(np.zeros((8, 8), bool), 0.5)]))
    # SAM1's nominal floor; reported (not gated) — carried for SAM3 parity.
    assert backend.pred_iou_min == SAM_PRED_IOU_MIN


def test_sam_backend_raw_score_is_not_gated():
    # A low predicted-IoU still produces a mask + the raw score (no drop-gate).
    h, w = 32, 32
    chosen = _disk(h, w, 16, 16, 6)
    backend = SamBackend(FakeSamPredictor(candidates=[(chosen, 0.10)]))
    masks, scores = backend.masks(
        np.zeros((h, w), np.uint8), [pose_prompt(np.array([[16.0, 16.0]]), (h, w))]
    )
    assert masks[0].any()
    assert abs(scores[0] - 0.10) < 1e-6


def test_load_sam_predictor_missing_dependency(monkeypatch):
    # Simulate `segment-anything` not installed.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "segment_anything":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from sleap_nn.inference.sam.backends import _load_sam_predictor

    with pytest.raises(ImportError, match="sleap-nn\\[sam\\]"):
        _load_sam_predictor("/tmp/nope.pth")
