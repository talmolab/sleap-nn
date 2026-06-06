"""Unit tests for the SAM3 backend (CPU; a fake transformers SAM3, no weights).

SAM3 (`mask_backend="sam3"`) is gated (`facebook/sam3`) and GPU-only end-to-end,
so every test here mocks the transformers `Sam3TrackerModel` / `Sam3TrackerProcessor`
pair. They lock the SAM3-specific recipe (recalibrated 0.5 floor — NEVER SAM1's
0.88 — and the mandatory speckle cleanup), the batched single-call contract, and
the clean, actionable error raised when `transformers`/SAM3 is unavailable.
"""

import numpy as np
import pytest

from sleap_nn.inference.sam import MASK_BACKENDS, get_mask_backend
from sleap_nn.inference.sam.backends import (
    SAM3_CLEANUP_RADIUS,
    SAM3_MODEL_ID,
    SAM3_PRED_IOU_MIN,
    SAM_PRED_IOU_MIN,
    Sam3Backend,
    _cleanup_speckle,
    _cleanup_seed,
)
from sleap_nn.inference.sam.prompts import (
    box_prompt,
    centroid_prompt,
    pose_prompt,
)


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


# ---------------------------------------------------------------------------
# Fake transformers SAM3 surface.
# ---------------------------------------------------------------------------
class _FakeTensor:
    """Minimal stand-in for a torch tensor: carries a numpy array + .cpu/.float."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def float(self):
        return _FakeTensor(self._arr.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeInputs(dict):
    """Processor output: a dict carrying ``original_sizes`` + a no-op ``.to``."""

    def to(self, device):
        return self


class FakeSam3Processor:
    """Stand-in for ``Sam3TrackerProcessor``.

    ``__call__`` records the batched prompt structure and returns the inputs
    dict; ``post_process_masks`` returns the fixed per-object candidate masks the
    test seeded (as ``[ (n_obj, n_cand, H, W) ]``).
    """

    def __init__(self, candidate_masks, hw):
        # candidate_masks: (n_obj, n_cand, H, W) bool
        self._cands = np.asarray(candidate_masks, bool)
        self._hw = hw
        self.last_call = None

    def __call__(
        self,
        images=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors=None,
    ):
        self.last_call = dict(
            images=images,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )
        return _FakeInputs(original_sizes=[self._hw])

    def post_process_masks(self, pred_masks, original_sizes=None, binarize=True):
        # Mirror transformers' API: returns a per-image list of (n_obj,n_cand,H,W).
        return [_FakeTensor(self._cands)]


class _FakeOut:
    def __init__(self, pred_masks, iou_scores):
        self.pred_masks = pred_masks
        self.iou_scores = iou_scores


class FakeSam3Model:
    """Stand-in for ``Sam3TrackerModel``: returns fixed iou scores per object."""

    def __init__(self, iou_scores):
        # iou_scores: (n_obj, n_cand)
        self._scores = np.asarray(iou_scores, np.float32)
        self.called = False

    def __call__(self, multimask_output=True, **inputs):
        self.called = True
        # pred_masks is opaque to the backend (post_process_masks handles it).
        return _FakeOut(
            pred_masks=None, iou_scores=_FakeTensor(self._scores[None, ...])
        )


def _backend(candidate_masks, iou_scores, hw, **kwargs):
    proc = FakeSam3Processor(candidate_masks, hw)
    model = FakeSam3Model(iou_scores)
    return Sam3Backend(model, proc, device="cpu", **kwargs), proc, model


# ---------------------------------------------------------------------------
# Backend selection (explicit, no default — PLAN L2).
# ---------------------------------------------------------------------------
def test_sam3_is_a_registered_backend():
    assert "sam3" in MASK_BACKENDS


def test_get_mask_backend_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown mask_backend"):
        get_mask_backend("nope")


def test_get_mask_backend_none_raises():
    with pytest.raises(ValueError, match="required and has no default"):
        get_mask_backend(None)


# ---------------------------------------------------------------------------
# Recipe constants — the SAM3 floor must NOT be SAM1's (regression guard).
# ---------------------------------------------------------------------------
def test_sam3_floor_is_recalibrated_not_sam1():
    assert SAM3_PRED_IOU_MIN == 0.5
    assert SAM3_PRED_IOU_MIN != SAM_PRED_IOU_MIN  # never share SAM1's 0.88
    assert SAM3_MODEL_ID == "facebook/sam3"
    assert SAM3_CLEANUP_RADIUS == 3


def test_sam3_backend_pred_iou_min_attribute():
    backend, _, _ = _backend(
        np.zeros((1, 1, 8, 8), bool), np.zeros((1, 1), np.float32), (8, 8)
    )
    assert backend.pred_iou_min == SAM3_PRED_IOU_MIN
    # Class-level default, too (so a future caller reading the type sees 0.5).
    assert Sam3Backend.pred_iou_min == SAM3_PRED_IOU_MIN


# ---------------------------------------------------------------------------
# Speckle cleanup (mandatory for SAM3; harvested from #643).
# ---------------------------------------------------------------------------
def test_cleanup_speckle_keeps_keypoint_blob_drops_speckle():
    h, w = 60, 60
    blob = _disk(h, w, 30, 30, 10)
    # Add a detached speck far away.
    speck = _disk(h, w, 5, 5, 1)
    mask = blob | speck
    kpts = np.array([[30.0, 30.0]])
    cleaned = _cleanup_speckle(mask, kpts, radius=SAM3_CLEANUP_RADIUS)
    # The keypoint blob survives; the isolated speck is gone.
    assert cleaned[30, 30]
    assert not cleaned[5, 5]
    # Single connected component after cleanup.
    from scipy import ndimage

    _, n = ndimage.label(cleaned)
    assert n == 1


def test_cleanup_speckle_empty_mask_is_noop():
    empty = np.zeros((10, 10), bool)
    out = _cleanup_speckle(empty, np.array([[5.0, 5.0]]))
    assert not out.any()


def test_cleanup_speckle_falls_back_to_largest_when_no_keypoint_inside():
    h, w = 60, 60
    big = _disk(h, w, 30, 30, 12)
    small = _disk(h, w, 50, 50, 4)
    mask = big | small
    # Keypoint lands in neither component (all-background) -> largest is kept.
    out = _cleanup_speckle(mask, np.array([[0.0, 0.0]]), radius=SAM3_CLEANUP_RADIUS)
    assert out[30, 30]
    assert not out[50, 50]


def test_cleanup_seed_uses_points_else_box_center():
    pts = np.array([[3.0, 4.0], [5.0, 6.0]], np.float32)
    p = pose_prompt(pts, (50, 50))
    seed = _cleanup_seed(p)
    np.testing.assert_allclose(seed, pts)
    # Box-only prompt: seed is the reject-box center.
    b = box_prompt(pts, (50, 50))
    seed_box = _cleanup_seed(b)
    rb = b.reject_box
    np.testing.assert_allclose(
        seed_box[0], [(rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0]
    )


# ---------------------------------------------------------------------------
# The batched .masks() contract (mocked SAM3).
# ---------------------------------------------------------------------------
def test_sam3_masks_batched_contract_and_cleanup():
    h, w = 64, 64
    # Two instances, each with two candidates: the whole-arena over-confident
    # candidate (rejected by _pick despite a higher score) and the real blob.
    huge = np.ones((h, w), bool)
    blob0 = _disk(h, w, 20, 20, 8) | _disk(h, w, 2, 2, 1)  # blob + a speck
    blob1 = _disk(h, w, 44, 44, 8)
    cands = np.stack(
        [
            np.stack([huge, blob0]),  # obj 0
            np.stack([huge, blob1]),  # obj 1
        ]
    )
    scores = np.array([[0.99, 0.62], [0.99, 0.55]], np.float32)
    backend, proc, model = _backend(cands, scores, (h, w))

    p0 = pose_prompt(np.array([[20.0, 20.0]]), (h, w))
    p1 = pose_prompt(np.array([[44.0, 44.0]]), (h, w))
    masks, out_scores = backend.masks(np.zeros((h, w), np.uint8), [p0, p1])

    assert model.called
    assert len(masks) == 2 and len(out_scores) == 2
    # Whole-arena candidate rejected -> the real blob is chosen for both.
    assert masks[0][20, 20] and masks[1][44, 44]
    assert not masks[0].all() and not masks[1].all()
    # The raw chosen SAM3 score is reported (not the rejected 0.99).
    assert abs(out_scores[0] - 0.62) < 1e-5
    assert abs(out_scores[1] - 0.55) < 1e-5
    # Speckle cleanup ran: the detached speck on obj 0 is gone.
    assert not masks[0][2, 2]
    # A single batched call over BOTH objects (the SAM3 batched path).
    assert len(proc.last_call["input_points"][0]) == 2
    # Pose prompts carry a box -> input_boxes is forwarded, one per object, and
    # is the prompt's real box (NOT reject_box).
    assert proc.last_call["input_boxes"] is not None
    assert len(proc.last_call["input_boxes"][0]) == 2
    np.testing.assert_allclose(proc.last_call["input_boxes"][0][0], list(p0.box))
    # Mask shapes honor the (H, W) contract.
    for m in masks:
        assert m.shape == (h, w)


def test_sam3_masks_empty_prompts():
    backend, _, model = _backend(
        np.zeros((1, 1, 8, 8), bool), np.zeros((1, 1), np.float32), (8, 8)
    )
    masks, scores = backend.masks(np.zeros((8, 8), np.uint8), [])
    assert masks == [] and scores == []
    # No forward pass on an empty frame.
    assert not model.called


def test_sam3_masks_raw_score_is_not_gated():
    # A low SAM3 score (below the 0.5 floor) still yields a mask + the raw score.
    h, w = 32, 32
    blob = _disk(h, w, 16, 16, 6)
    cands = np.stack([np.stack([blob])])  # (1 obj, 1 cand, H, W)
    backend, _, _ = _backend(cands, np.array([[0.10]], np.float32), (h, w))
    masks, scores = backend.masks(
        np.zeros((h, w), np.uint8), [centroid_prompt(np.array([16.0, 16.0]), (h, w))]
    )
    assert masks[0].any()
    assert abs(scores[0] - 0.10) < 1e-5


def test_sam3_masks_point_only_prompt_forwards_no_box():
    # Point-only prompts (centroid mode, box is None) must NOT forward a box to
    # SAM3. Feeding reject_box as a real box would hand SAM3 a whole-frame
    # "segment everything" hint and make it diverge from SAM1 — a regression
    # guard for the reject_box-as-box bug.
    h, w = 48, 48
    blob = _disk(h, w, 24, 24, 9)
    cands = np.stack([np.stack([blob])])
    backend, proc, _ = _backend(cands, np.array([[0.7]], np.float32), (h, w))
    prompt = centroid_prompt(np.array([24.0, 24.0]), (h, w))
    assert prompt.box is None  # centroid mode is point-only
    masks, scores = backend.masks(np.zeros((h, w), np.uint8), [prompt])
    assert masks[0][24, 24]
    # The centroid point was forwarded as this object's single positive point...
    assert proc.last_call["input_points"][0][0] == [[24.0, 24.0]]
    # ...and NO box was forwarded for the point-only prompt (the fix).
    assert proc.last_call["input_boxes"] is None


# ---------------------------------------------------------------------------
# Clean error when transformers / SAM3 is unavailable (the env reality).
# ---------------------------------------------------------------------------
def test_load_sam3_missing_transformers(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no module named transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from sleap_nn.inference.sam.backends import _load_sam3

    with pytest.raises(ImportError, match=r"sleap-nn\[sam3\]"):
        _load_sam3()


def test_get_mask_backend_sam3_clean_error_without_transformers(monkeypatch):
    # Selecting the sam3 backend with transformers absent raises the actionable
    # ImportError (not NotImplementedError, not a bare ModuleNotFoundError).
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no module named transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as ei:
        get_mask_backend("sam3", device="cpu")
    msg = str(ei.value)
    assert "transformers>=5" in msg
    assert "facebook/sam3" in msg  # the gated-model auth pointer
    assert "huggingface-cli login" in msg
