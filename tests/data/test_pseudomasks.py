"""Tests for pose -> per-instance segmentation pseudomask generation.

Covers the dependency-free skeleton rasterizer + the ``generate_pseudomasks``
skeleton path (mask count, type, roundtrip), the unknown-source guard, and the
SAM-path lazy-import error surface (exercised by faking ``segment-anything``
absent). The full SAM / hybrid GPU path is gated behind ``segment_anything`` +
a checkpoint and skipped when unavailable.
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
    _load_sam_predictor,
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
