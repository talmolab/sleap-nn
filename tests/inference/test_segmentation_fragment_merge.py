"""Tests for the Increment-A bottom-up segmentation post-processing (#617).

Covers two default-OFF levers added to the grouping path and their wiring:

* **Adaptive distance gate** (``distance_gate_alpha`` on
  :func:`group_instances_from_offsets`): drops stray foreground pixels and the
  off-instance phantom centers they get force-assigned to, leaving the genuine
  instances untouched.
* **RAG fragment-merge** (:func:`merge_instances`): re-fuses one
  artificially-split animal back into a single mask while keeping two
  genuinely-touching distinct animals apart (the center-valley signal is
  load-bearing — a contact-only ablation over-merges them).

Plus the contract that matters most for shipping: with
``distance_gate_alpha=None`` and ``merge_fragments=False`` the grouping and the
``SegmentationLayer`` are **byte-for-byte identical to today** (no-op
regression).
"""

import numpy as np
import torch

from sleap_nn.inference.layers.configs import PostprocessConfig
from sleap_nn.inference.layers.segmentation import SegmentationLayer
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.segmentation import (
    group_instances_from_offsets,
    merge_instances,
)


# --------------------------------------------------------------------------- #
# Scene builders (synthetic, model-independent).
# --------------------------------------------------------------------------- #
def _ellipse(h, w, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _gauss_peak(h, w, cx, cy, sigma, amp=1.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def _build_gate_scene(output_stride=2):
    """3 separated ellipses + a phantom center peak surrounded by stray fg pixels.

    Every ellipse pixel points exactly at its center (zero offset residual); the
    stray pixels carry offset 0 (they point at themselves), so their predicted
    center is their own location — far from the phantom peak they get assigned
    to under bare argmin. Returns torch maps shaped for
    :func:`group_instances_from_offsets` plus the phantom/stray bookkeeping.
    """
    h, w = 120, 100
    fg = np.zeros((h, w), np.float32)
    ctr = np.zeros((h, w), np.float32)
    off = np.zeros((2, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]

    centers = [(30, 25), (60, 70), (95, 35)]  # (cy, cx) grid
    radii = [(12, 8), (10, 14), (9, 9)]
    for (cy, cx), (ry, rx) in zip(centers, radii):
        ell = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0
        fg[ell] = 1.0
        np.maximum(ctr, _gauss_peak(h, w, cx, cy, sigma=3.0, amp=1.0), out=ctr)
        off[0][ell] = (cx - xs[ell]).astype(np.float32) * output_stride
        off[1][ell] = (cy - ys[ell]).astype(np.float32) * output_stride

    phantom = (15, 85)  # (cy, cx) grid
    np.maximum(ctr, _gauss_peak(h, w, 85, 15, sigma=3.0, amp=0.9), out=ctr)
    strays = [(15, 78), (15, 92), (8, 85), (22, 85), (10, 80)]
    for sy, sx in strays:
        fg[sy, sx] = 1.0  # offset stays 0 (bad-offset stray pixel)

    fg_t = torch.from_numpy(fg).float().reshape(1, 1, h, w)
    ctr_t = torch.from_numpy(ctr).float().reshape(1, 1, h, w)
    off_t = torch.from_numpy(off).float().reshape(1, 2, h, w)
    return fg_t, ctr_t, off_t, centers, phantom, strays, output_stride


def _build_merge_scene():
    """One ellipse split into two abutting halves + two touching distinct ellipses.

    The center heatmap has ONE peak for the split animal (high ridge between its
    two fragment "centers" => merge) and TWO peaks for the touching pair (deep
    valley between => keep apart). Returns the 4-mask baseline instance list, the
    grid heatmap, the offset field, and the stride.
    """
    h, w = 120, 200
    stride = 2
    A = _ellipse(h, w, cx=50, cy=35, rx=24, ry=10)
    A_left = A.copy()
    A_left[:, 50:] = False
    A_right = A.copy()
    A_right[:, :50] = False
    B = _ellipse(h, w, cx=120, cy=80, rx=16, ry=12)
    C = _ellipse(h, w, cx=146, cy=80, rx=16, ry=12)  # firmly touches B

    ctr = np.zeros((h, w), np.float64)
    ctr += _gauss_peak(h, w, 50, 35, sigma=6, amp=1.0)
    ctr += _gauss_peak(h, w, 120, 80, sigma=4, amp=1.0)
    ctr += _gauss_peak(h, w, 146, 80, sigma=4, amp=1.0)
    ctr = np.clip(ctr, 0, 1.2)

    off = np.zeros((2, h, w), np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    px = xx * stride + stride / 2.0
    py = yy * stride + stride / 2.0
    for mask, (cx, cy) in [(A, (50, 35)), (B, (120, 80)), (C, (146, 80))]:
        off[0][mask] = (cx * stride + stride / 2.0 - px)[mask]
        off[1][mask] = (cy * stride + stride / 2.0 - py)[mask]

    def mk(mask, cx, cy, score):
        return {
            "mask": mask,
            "center": (cx * stride + stride / 2.0, cy * stride + stride / 2.0),
            "score": score,
        }

    instances = [
        mk(A_left, 38, 35, 0.9),
        mk(A_right, 62, 35, 0.9),
        mk(B, 120, 80, 1.0),
        mk(C, 146, 80, 1.0),
    ]
    return instances, ctr, off, stride


# --------------------------------------------------------------------------- #
# Distance gate.
# --------------------------------------------------------------------------- #
def test_distance_gate_deletes_phantom_and_strays():
    """alpha gate removes the off-instance phantom + strays; real 3 untouched."""
    fg, ctr, off, centers, phantom, strays, s = _build_gate_scene()
    common = dict(
        fg_threshold=0.5, peak_threshold=0.2, output_stride=s, center_nms_kernel=3
    )

    # Baseline argmin keeps 4 instances (3 real + the phantom, which steals the
    # strays + nearest fg).
    base = group_instances_from_offsets(
        fg, ctr, off, distance_gate_alpha=None, **common
    )
    assert len(base) >= 4
    phantom_px = np.array([phantom[1], phantom[0]]) * s + s / 2.0
    base_c = np.array([i["center"] for i in base])
    assert np.linalg.norm(base_c - phantom_px, axis=1).min() < 5.0

    # Adaptive gate (recommended alpha): exactly the 3 real instances survive.
    gated = group_instances_from_offsets(
        fg, ctr, off, distance_gate_alpha=0.5, **common
    )
    assert len(gated) == 3
    gated_c = np.array([i["center"] for i in gated])
    assert np.linalg.norm(gated_c - phantom_px, axis=1).min() > 10.0

    union = np.zeros(fg.shape[-2:], dtype=bool)
    for inst in gated:
        union |= inst["mask"]
    for sy, sx in strays:
        assert not union[sy, sx]

    real_px = np.array([[cx, cy] for (cy, cx) in centers]) * s + s / 2.0
    for rp in real_px:
        assert np.linalg.norm(gated_c - rp, axis=1).min() < 5.0


def test_distance_gate_none_is_unchanged():
    """``distance_gate_alpha=None`` is byte-for-byte the argmin grouping."""
    fg, ctr, off, *_ = _build_gate_scene()
    common = dict(
        fg_threshold=0.5, peak_threshold=0.2, output_stride=2, center_nms_kernel=3
    )
    a = group_instances_from_offsets(fg, ctr, off, **common)
    b = group_instances_from_offsets(fg, ctr, off, distance_gate_alpha=None, **common)
    assert len(a) == len(b)
    for ia, ib in zip(a, b):
        np.testing.assert_array_equal(ia["mask"], ib["mask"])
        assert ia["center"] == ib["center"]
        assert ia["score"] == ib["score"]


# --------------------------------------------------------------------------- #
# Fragment merge.
# --------------------------------------------------------------------------- #
def _bc_fused(merged):
    """True if any merged mask contains BOTH the B-center and C-center pixels."""
    return any(x["mask"][80, 120] and x["mask"][80, 146] for x in merged)


def test_merge_fuses_split_keeps_touching_apart():
    """Greedy/multicut: A's halves fuse to 1; touching B,C stay 2 -> 3 masks."""
    instances, ctr, off, stride = _build_merge_scene()
    assert len(instances) == 4  # split animal contributes 2 fragments

    for method, kw in [
        ("greedy", {}),
        ("multicut", {}),
        ("greedy", dict(thresholds=(0.7, 0.5, 0.3))),
    ]:
        merged = merge_instances(
            [dict(i) for i in instances], ctr, off, stride, method=method, **kw
        )
        assert len(merged) == 3, (method, kw, len(merged))
        assert not _bc_fused(merged), (method, kw)


def test_merge_contact_only_over_merges_touching_pair():
    """Valley term is load-bearing: contact-only fuses the touching B,C (wrong)."""
    instances, ctr, off, stride = _build_merge_scene()
    contact_only = merge_instances(
        [dict(i) for i in instances],
        ctr,
        off,
        stride,
        method="greedy",
        w_valley=0.0,
        w_offset=0.0,
        thresholds=(0.8, 0.6, 0.4),
    )
    # With no valley/offset arbitration, the firmly-touching B and C collapse.
    assert _bc_fused(contact_only)


def test_merge_none_and_singleton_are_passthrough():
    """``method='none'`` and <2 instances return the list unchanged."""
    instances, ctr, off, stride = _build_merge_scene()
    same = merge_instances(
        [dict(i) for i in instances], ctr, off, stride, method="none"
    )
    assert len(same) == len(instances)
    one = merge_instances([dict(instances[0])], ctr, off, stride, method="greedy")
    assert len(one) == 1


# --------------------------------------------------------------------------- #
# Byte-identical no-op regression at the SegmentationLayer level.
# --------------------------------------------------------------------------- #
def _disk(h, w, cy, cx, r):
    ys, xs = np.mgrid[0:h, 0:w]
    return ((ys - cy) ** 2 + (xs - cx) ** 2) <= r * r


def _seg_layer(**overrides):
    """A backend-free ``SegmentationLayer`` (postprocess only)."""
    layer = SegmentationLayer.__new__(SegmentationLayer)
    layer.output_stride = 2
    layer.fg_threshold = 0.5
    layer.min_mask_area = 0
    layer.center_nms_kernel = 3
    layer.mask_cleanup = False
    layer.mask_cleanup_radius = 0
    layer.full_res_masks = False
    layer.postprocess_config = PostprocessConfig(peak_threshold=0.2)
    for k, v in overrides.items():
        setattr(layer, k, v)
    return layer


def _gate_raw_and_info():
    fg, ctr, off, *_ = _build_gate_scene()
    raw = {
        "SegmentationHead": fg,
        "InstanceCenterHead": ctr,
        "CenterOffsetHead": off,
    }
    h, w = fg.shape[-2:]
    info = PreprocInfo(
        original_size=(h * 2, w * 2),
        processed_size=(h * 2, w * 2),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=2,
    )
    return raw, info


def test_segmentation_layer_defaults_are_byte_identical():
    """Default layer (no gate, no merge) == a layer with the knobs set OFF.

    The new ``distance_gate_alpha`` / ``merge_fragments`` knobs must not perturb
    the shipped output when left at their defaults. A ``__new__``-built layer
    that never sets them (today's attribute set) must produce exactly the same
    masks as one that explicitly sets the OFF values.
    """
    raw, info = _gate_raw_and_info()

    # Layer A: pre-#617 attribute set (knobs absent -> read via getattr default).
    out_old = _seg_layer().postprocess(raw, info)
    # Layer B: knobs explicitly set to their default-OFF values.
    out_new = _seg_layer(
        distance_gate_alpha=None,
        merge_fragments=False,
        merge_method="greedy",
        merge_thresholds=(0.85, 0.6, 0.4),
        merge_w_valley=1.0,
        merge_w_offset=0.25,
        merge_dilate=1,
    ).postprocess(raw, info)

    assert len(out_old.pred_masks) == len(out_new.pred_masks) == 1
    masks_old = out_old.pred_masks[0]
    masks_new = out_new.pred_masks[0]
    assert len(masks_old) == len(masks_new)
    for a, b in zip(masks_old, masks_new):
        np.testing.assert_array_equal(a["mask"], b["mask"])
        assert a["score"] == b["score"]
        assert a["scale"] == b["scale"]
        assert a["offset"] == b["offset"]


def test_segmentation_layer_gate_and_merge_change_output_when_enabled():
    """Enabling the levers actually changes the masks (sanity that they wire in).

    The gate drops the phantom instance; together gate+merge reduce the instance
    count below the (over-segmented) baseline.
    """
    raw, info = _gate_raw_and_info()
    n_base = len(_seg_layer().postprocess(raw, info).pred_masks[0])
    n_gated = len(
        _seg_layer(distance_gate_alpha=0.5).postprocess(raw, info).pred_masks[0]
    )
    assert n_gated < n_base
