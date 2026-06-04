"""Tests for pose -> per-instance segmentation pseudomask generation.

Covers the dependency-free skeleton rasterizer + the ``generate_pseudomasks``
skeleton path (mask count, type, roundtrip), the unknown-source guard, the
SAM-path lazy-import error surface (exercised by faking ``segment-anything``
absent), and the SAM3 speckle cleanup + lazy-import surface (faking
``transformers`` absent). The full SAM / SAM3 GPU paths are gated behind their
deps + a checkpoint / gated model and skipped when unavailable.
"""

import builtins
import os

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.data.pseudomasks import (
    generate_pseudomasks,
    make_pseudomasks_cli,
    rasterize_instance_mask,
    _kpt_box,
    _pick,
    _disjointify,
    _cleanup_speckle,
    _load_sam_predictor,
    _load_sam3,
)


# ---------------------------------------------------------------------------
# rasterize_instance_mask
# ---------------------------------------------------------------------------
def test_rasterize_instance_mask_non_empty_and_covers_nodes():
    """A two-node skeleton rasterizes to a non-empty mask covering both nodes."""
    points = np.array([[20.0, 30.0], [60.0, 80.0]], dtype=np.float32)
    edge_inds = [(0, 1)]
    hw = (128, 128)

    mask = rasterize_instance_mask(points, edge_inds, hw)

    assert mask.shape == hw
    assert mask.dtype == bool
    assert mask.any()
    # The node centers must be foreground (filled disk drawn at each node).
    for x, y in points:
        assert mask[int(round(y)), int(round(x))]
    # The edge midpoint should be covered by the thick line too.
    mx, my = points.mean(axis=0)
    assert mask[int(round(my)), int(round(mx))]


def test_rasterize_instance_mask_skips_nan_nodes():
    """A NaN node is skipped but a single valid node still yields a disk."""
    points = np.array([[40.0, 40.0], [np.nan, np.nan]], dtype=np.float32)
    mask = rasterize_instance_mask(points, [(0, 1)], (96, 96))
    assert mask.any()
    assert mask[40, 40]


def test_rasterize_instance_mask_all_nan_empty():
    """All-NaN points produce an all-False mask."""
    points = np.full((3, 2), np.nan, dtype=np.float32)
    mask = rasterize_instance_mask(points, [(0, 1), (1, 2)], (64, 64))
    assert not mask.any()


def test_rasterize_dilate_frac_grows_mask():
    """A larger dilation fraction grows the silhouette area."""
    points = np.array([[20.0, 20.0], [80.0, 80.0]], dtype=np.float32)
    small = rasterize_instance_mask(
        points, [(0, 1)], (128, 128), dilate_frac=0.0, min_dilate=0
    )
    big = rasterize_instance_mask(
        points, [(0, 1)], (128, 128), dilate_frac=0.5, min_dilate=4
    )
    assert big.sum() > small.sum()


# ---------------------------------------------------------------------------
# generate_pseudomasks (skeleton path)
# ---------------------------------------------------------------------------
def test_generate_pseudomasks_skeleton(minimal_instance):
    """Skeleton source yields LFs whose masks are UserSegmentationMask, 1:1."""
    labels = sio.load_slp(str(minimal_instance))
    n_inst_src = len(labels[0].instances)

    out = generate_pseudomasks(labels, source="skeleton")

    assert len(out.labeled_frames) == 1
    lf = out.labeled_frames[0]
    # One mask per posed instance, poses retained for eval frame-pairing.
    assert len(lf.masks) == n_inst_src
    assert len(lf.instances) == n_inst_src
    for m in lf.masks:
        assert isinstance(m, sio.UserSegmentationMask)
        assert np.asarray(m.data, dtype=bool).any()


def test_generate_pseudomasks_roundtrip(minimal_instance, tmp_path):
    """A skeleton-pseudomask Labels saves embedded and reloads with masks."""
    labels = sio.load_slp(str(minimal_instance))
    out = generate_pseudomasks(labels, source="skeleton")

    out_path = tmp_path / "seg.pkg.slp"
    out.save(out_path.as_posix(), embed=True)

    reloaded = sio.load_slp(out_path.as_posix())
    lf = reloaded.labeled_frames[0]
    assert len(lf.masks) == len(out.labeled_frames[0].masks)
    assert isinstance(lf.masks[0], sio.UserSegmentationMask)
    assert np.asarray(lf.masks[0].data, dtype=bool).any()


def test_make_pseudomasks_cli_skeleton(minimal_instance, tmp_path):
    """The CLI impl fn writes a reloadable seg .pkg.slp with masks."""
    out_path = tmp_path / "cli_seg.pkg.slp"
    overlay_path = tmp_path / "overlay.png"

    result = make_pseudomasks_cli(
        src=str(minimal_instance),
        out=out_path.as_posix(),
        source="skeleton",
        overlay=overlay_path.as_posix(),
    )

    assert out_path.exists()
    assert overlay_path.exists()
    assert len(result.labeled_frames) == 1
    reloaded = sio.load_slp(out_path.as_posix())
    assert len(reloaded.labeled_frames[0].masks) >= 1


def test_generate_pseudomasks_unknown_source(minimal_instance):
    """An unsupported source raises a clear ValueError before any SAM load."""
    labels = sio.load_slp(str(minimal_instance))
    with pytest.raises(ValueError, match="Unknown pseudomask source"):
        generate_pseudomasks(labels, source="bogus")


# ---------------------------------------------------------------------------
# SAM helpers (dependency-free pieces).
# ---------------------------------------------------------------------------
def test_kpt_box_clamped_and_padded():
    """The keypoint box is padded by the margin and clamped to the frame."""
    pos = np.array([[10.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    box = _kpt_box(pos, (100, 120))
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0
    assert x1 <= 119 and y1 <= 99
    # Box strictly contains the keypoint bbox.
    assert x0 <= 10 and y0 <= 10 and x1 >= 20 and y1 >= 30


def test_pick_rejects_oversized_candidate():
    """_pick rejects the whole-frame candidate even with the highest score."""
    h, w = 40, 40
    big = np.ones((h, w), dtype=bool)
    small = np.zeros((h, w), dtype=bool)
    small[5:15, 5:15] = True
    masks = np.stack([big, small])
    scores = np.array([0.99, 0.5])  # big has higher score but is oversized
    box = np.array([4, 4, 16, 16], dtype=np.float32)
    assert _pick(masks, scores, box) == 1


def test_disjointify_resolves_overlap():
    """Overlapping masks become disjoint, each keeping its own keypoint."""
    h, w = 50, 50
    a = np.zeros((h, w), bool)
    a[10:30, 10:30] = True
    b = np.zeros((h, w), bool)
    b[20:40, 20:40] = True  # overlaps a in [20:30, 20:30]
    kpts = [np.array([[15.0, 15.0]]), np.array([[35.0, 35.0]])]
    out = _disjointify([a, b], kpts)
    # No pixel is claimed by both masks.
    assert not (out[0] & out[1]).any()
    # Each instance keeps its own seed keypoint.
    assert out[0][15, 15]
    assert out[1][35, 35]


# ---------------------------------------------------------------------------
# SAM lazy-import / checkpoint error surface.
# ---------------------------------------------------------------------------
def test_load_sam_predictor_missing_dep_raises(monkeypatch):
    """When segment-anything is absent, a friendly ImportError is raised."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "segment_anything" or name.startswith("segment_anything."):
            raise ImportError("No module named 'segment_anything'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"sleap-nn\[sam\]"):
        _load_sam_predictor("/nonexistent/ckpt.pth")


def test_generate_pseudomasks_sam_missing_dep_raises(minimal_instance, monkeypatch):
    """source='sam' surfaces the friendly ImportError when SAM is absent."""
    labels = sio.load_slp(str(minimal_instance))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "segment_anything" or name.startswith("segment_anything."):
            raise ImportError("No module named 'segment_anything'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"sleap-nn\[sam\]"):
        generate_pseudomasks(
            labels, source="sam", sam_checkpoint="/nonexistent/ckpt.pth"
        )


# ---------------------------------------------------------------------------
# Full SAM / hybrid GPU path (gated).
# ---------------------------------------------------------------------------
_SAM_CKPT = os.environ.get("SLEAP_NN_SAM_CHECKPOINT")


@pytest.mark.skipif(
    _SAM_CKPT is None,
    reason="SLEAP_NN_SAM_CHECKPOINT not set; SAM checkpoint unavailable.",
)
def test_generate_pseudomasks_hybrid_smoke(minimal_instance):
    """Hybrid source produces one mask per instance (SAM or skeleton fallback)."""
    pytest.importorskip("segment_anything")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = sio.load_slp(str(minimal_instance))
    n_inst = len(labels[0].instances)
    out = generate_pseudomasks(
        labels, source="hybrid", sam_checkpoint=_SAM_CKPT, device=device
    )
    lf = out.labeled_frames[0]
    # Hybrid guarantees a mask for every kept instance.
    assert len(lf.masks) == n_inst
    assert all(isinstance(m, sio.UserSegmentationMask) for m in lf.masks)


# ---------------------------------------------------------------------------
# SAM3 speckle cleanup (dependency-free).
# ---------------------------------------------------------------------------
def test_cleanup_speckle_keeps_keypoint_blob_drops_specks():
    """Cleanup keeps the keypoint-connected blob and removes detached specks."""
    h, w = 80, 80
    m = np.zeros((h, w), bool)
    m[20:50, 20:50] = True  # main body (contains the keypoint)
    m[5, 5] = True  # isolated single-pixel speck (removed by opening)
    m[70:73, 70:73] = True  # small detached speck far from any keypoint
    kpts = np.array([[35.0, 35.0]], np.float32)

    out = _cleanup_speckle(m, kpts, radius=2)

    # The keypoint stays inside the cleaned mask; the far speck is gone.
    assert out[35, 35]
    assert not out[71, 71]
    assert not out[5, 5]
    # Result is a single connected component.
    from scipy import ndimage

    _, n = ndimage.label(out)
    assert n == 1


def test_cleanup_speckle_empty_mask_is_noop():
    """An all-False mask is returned unchanged (no components to keep)."""
    m = np.zeros((30, 30), bool)
    out = _cleanup_speckle(m, np.array([[10.0, 10.0]], np.float32))
    assert not out.any()


def test_cleanup_speckle_no_keypoint_hit_keeps_largest():
    """If no keypoint lands on a component, the largest one is kept."""
    h, w = 80, 80
    m = np.zeros((h, w), bool)
    m[10:40, 10:40] = True  # large blob
    m[60:65, 60:65] = True  # smaller blob
    # Keypoint in empty space -> falls back to largest component.
    out = _cleanup_speckle(m, np.array([[1.0, 1.0]], np.float32), radius=2)
    assert out[25, 25]  # large blob kept
    assert not out[62, 62]  # smaller blob dropped


# ---------------------------------------------------------------------------
# SAM3 lazy-import error surface (transformers faked absent).
# ---------------------------------------------------------------------------
def test_load_sam3_missing_dep_raises(monkeypatch):
    """When transformers (SAM3) is absent, a friendly ImportError is raised."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"sleap-nn\[sam3\]"):
        _load_sam3()


def test_generate_pseudomasks_sam3_missing_dep_raises(minimal_instance, monkeypatch):
    """source='sam3' surfaces the friendly ImportError when SAM3 is absent."""
    labels = sio.load_slp(str(minimal_instance))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"sleap-nn\[sam3\]"):
        generate_pseudomasks(labels, source="sam3")


# ---------------------------------------------------------------------------
# SAM3 frame-loop wiring (transformers FAKED, no GPU / no weights).
# ---------------------------------------------------------------------------
class _FakeTorchTensor:
    """Minimal stand-in exposing the ``.cpu().numpy()`` / ``.float()`` chain."""

    def __init__(self, arr):
        self._a = arr

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeInputs(dict):
    def to(self, *_a, **_k):
        return self


class _FakeSam3Processor:
    """Builds a fake batched input + returns plausible per-object candidates.

    For each prompted object it emits 3 candidates: a tight box around the
    keypoint bbox plus a couple of detached single-pixel specks (so the real
    ``_cleanup_speckle`` has something to remove), and a deliberately oversized
    whole-frame candidate (so ``_pick`` must reject it).
    """

    def __init__(self, hw, n_obj_ref):
        self._hw = hw
        self._n_obj_ref = n_obj_ref

    def __call__(self, images, input_points, input_boxes, **_k):
        self._boxes = input_boxes[0]  # [obj][xyxy]
        self._points = input_points[0]  # [obj][pt][xy]
        self._n_obj = len(self._boxes)
        self._n_obj_ref[0] = self._n_obj
        return _FakeInputs(original_sizes=[self._hw])

    def post_process_masks(self, pred_masks, original_sizes, binarize):
        h, w = self._hw
        n_cand = 3
        out = np.zeros((self._n_obj, n_cand, h, w), bool)
        for j in range(self._n_obj):
            pts = np.asarray(self._points[j], float)
            x0 = int(np.floor(pts[:, 0].min())) - 6
            y0 = int(np.floor(pts[:, 1].min())) - 6
            x1 = int(np.ceil(pts[:, 0].max())) + 6
            y1 = int(np.ceil(pts[:, 1].max())) + 6
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            # candidate 0: a tight body covering the keypoints (own-containment
            # ok, area comparable to the dilated skeleton so the area-ratio
            # filter accepts it) + 2 detached single-pixel specks for
            # _cleanup_speckle to remove.
            out[j, 0, y0:y1, x0:x1] = True
            out[j, 0, 1, 1] = True
            out[j, 0, h - 2, w - 2] = True
            # candidate 1: slightly smaller body covering the keypoints.
            out[j, 1, y0 + 2 : max(y0 + 3, y1 - 2), x0 + 2 : max(x0 + 3, x1 - 2)] = True
            # candidate 2: oversized whole-frame (must be rejected by _pick).
            out[j, 2, :, :] = True
        return [_FakeTensor4D(out)]


class _FakeTensor4D:
    def __init__(self, arr):
        self._a = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeSam3Model:
    """Returns synthetic ``pred_masks`` + ``iou_scores`` for the frame batch."""

    def __init__(self, n_obj_ref):
        self._n_obj_ref = n_obj_ref

    def __call__(self, original_sizes=None, **_k):
        n_obj = self._n_obj_ref[0]
        # candidate 0 (tight body) gets the highest score among survivors.
        scores = np.tile(np.array([0.8, 0.6, 0.95]), (n_obj, 1))[None]
        return type(
            "Out",
            (),
            {"pred_masks": None, "iou_scores": _FakeTorchTensor(scores)},
        )()


@pytest.mark.parametrize("source", ["sam3", "hybrid_sam3"])
def test_generate_pseudomasks_sam3_frame_loop_faked(
    minimal_instance, monkeypatch, source
):
    """The full sam3/hybrid_sam3 frame loop runs end-to-end with a faked model.

    Exercises prompt building, candidate selection (``_pick`` rejecting the
    oversized whole-frame candidate), speckle cleanup, disjointify, the
    recalibrated quality filter, and assembly into ``UserSegmentationMask`` --
    all without ``transformers`` / GPU / weights.
    """
    import sleap_nn.data.pseudomasks as pm

    labels = sio.load_slp(str(minimal_instance))
    n_inst = len(labels[0].instances)
    h, w = labels.videos[0].shape[1], labels.videos[0].shape[2]

    n_obj_ref = [0]

    def fake_load_sam3(model_id=pm.SAM3_MODEL_ID, device="cuda"):
        proc = _FakeSam3Processor((h, w), n_obj_ref)
        return _FakeSam3Model(n_obj_ref), proc

    monkeypatch.setattr(pm, "_load_sam3", fake_load_sam3)

    out = generate_pseudomasks(labels, source=source, device="cpu")

    assert len(out.labeled_frames) == 1
    lf = out.labeled_frames[0]
    # Both instances keep GT (tight candidate passes the filter / hybrid falls back).
    assert len(lf.masks) == n_inst
    assert len(lf.instances) == n_inst
    for m, inst in zip(lf.masks, lf.instances):
        mm = np.asarray(m.data, dtype=bool)
        assert isinstance(m, sio.UserSegmentationMask)
        assert mm.any()
        # The oversized whole-frame candidate must NOT have been selected.
        assert mm.mean() < 0.9
        # Speckle in the far corner must be gone (cleanup kept the keypoint blob).
        assert not mm[h - 2, w - 2]
    # Disjoint: no pixel claimed by both instance masks.
    a = np.asarray(lf.masks[0].data, bool)
    b = np.asarray(lf.masks[1].data, bool)
    assert not (a & b).any()
