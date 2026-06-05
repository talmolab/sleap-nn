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


def test_distance_gate_none_drops_no_foreground_pixel():
    """Independent pin of the no-op claim: ``alpha=None`` keeps EVERY fg pixel.

    Stronger than comparing the default vs explicit ``None`` call (which share a
    code path): with no gate and ``mask_cleanup=False`` every thresholded fg pixel
    is argmin-assigned to some kept center, so the total covered area equals the
    raw foreground count. A gate that erroneously fired on ``None`` would drop
    pixels and break this equality; an active gate (``alpha=0.5``) provably drops
    some (strict ``<``).
    """
    fg, ctr, off, *_ = _build_gate_scene()
    common = dict(
        fg_threshold=0.5, peak_threshold=0.2, output_stride=2, center_nms_kernel=3
    )
    n_fg = int((fg[0, 0] > 0.5).sum())

    none = group_instances_from_offsets(
        fg, ctr, off, distance_gate_alpha=None, **common
    )
    covered_none = int(sum(int(i["mask"].sum()) for i in none))
    assert covered_none == n_fg  # nothing dropped on the default path

    gated = group_instances_from_offsets(
        fg, ctr, off, distance_gate_alpha=0.5, **common
    )
    covered_gated = int(sum(int(i["mask"].sum()) for i in gated))
    assert covered_gated < n_fg  # the gate actually removed stray pixels


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
    """``method='none'`` and <2 instances return the list unchanged (same object)."""
    instances, ctr, off, stride = _build_merge_scene()
    src = [dict(i) for i in instances]
    same = merge_instances(src, ctr, off, stride, method="none")
    assert same is src  # passthrough returns the input object, not a copy
    one_src = [dict(instances[0])]
    one = merge_instances(one_src, ctr, off, stride, method="greedy")
    assert one is one_src and len(one) == 1


def test_merge_edge_cases():
    """Empty input, unknown method, and an empty-then-gated grouping are safe."""
    instances, ctr, off, stride = _build_merge_scene()
    # Empty instance list: no RAG, returns empty (the <=1 short-circuit).
    assert merge_instances([], ctr, off, stride, method="greedy") == []
    # Unknown method raises (only reachable for len>=2 past the short-circuit).
    try:
        merge_instances([dict(i) for i in instances], ctr, off, stride, method="bogus")
        raise AssertionError("expected ValueError for unknown merge method")
    except ValueError:
        pass


def test_group_empty_foreground_returns_empty():
    """No foreground -> ``[]`` for both the default and the gated path (no crash)."""
    h, w = 60, 50
    fg = torch.zeros(1, 1, h, w)
    ctr = torch.zeros(1, 1, h, w)
    off = torch.zeros(1, 2, h, w)
    common = dict(
        fg_threshold=0.5, peak_threshold=0.2, output_stride=2, center_nms_kernel=3
    )
    assert group_instances_from_offsets(fg, ctr, off, **common) == []
    assert (
        group_instances_from_offsets(fg, ctr, off, distance_gate_alpha=0.5, **common)
        == []
    )


def _build_chain_scene(stride=2):
    """Three colinear abutting fragments of ONE animal (a ridge across all three).

    Forces the multicut to sum parallel/transitive edge costs along the chain and
    fuse all three, exercising the cost-summation path beyond the 2-fragment case.
    """
    h, w = 80, 200
    ys, xs = np.mgrid[0:h, 0:w]
    cy = 40
    centers = [(45, cy), (75, cy), (105, cy)]  # (cx, cy) grid, 30 apart
    body = _ellipse(h, w, cx=75, cy=cy, rx=54, ry=10)  # spans all three
    ctr = np.zeros((h, w), np.float64)
    for cx, cyy in centers:
        ctr += _gauss_peak(h, w, cx, cyy, sigma=11, amp=1.0)  # additive -> high ridge
    ctr = np.clip(ctr, 0, 1.0)
    off = np.zeros((2, h, w), np.float64)
    # Split the body into three columns, each pointing at its own peak.
    bounds = [(0, 60), (60, 90), (90, w)]
    insts = []
    for (cx, cyy), (x0, x1) in zip(centers, bounds):
        m = body & (xs >= x0) & (xs < x1)
        off[0][m] = (cx - xs[m]) * stride
        off[1][m] = (cyy - ys[m]) * stride
        insts.append(
            {
                "mask": m,
                "center": (cx * stride + stride / 2.0, cyy * stride + stride / 2.0),
                "score": 1.0,
            }
        )
    return insts, ctr, off, stride


def test_multicut_fuses_three_fragment_chain():
    """Multicut collapses a 3-fragment colinear chain of one animal into one mask."""
    insts, ctr, off, stride = _build_chain_scene()
    assert len(insts) == 3
    merged = merge_instances(
        [dict(i) for i in insts], ctr, off, stride, method="multicut"
    )
    assert len(merged) == 1, [int(m["mask"].sum()) for m in merged]
    # The fused mask is the union of all three fragments.
    union = np.zeros_like(insts[0]["mask"])
    for i in insts:
        union |= i["mask"]
    np.testing.assert_array_equal(merged[0]["mask"], union)


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


def test_segmentation_layer_gate_changes_output_when_enabled():
    """Enabling the distance gate drops the phantom instance at the layer level."""
    raw, info = _gate_raw_and_info()
    n_base = len(_seg_layer().postprocess(raw, info).pred_masks[0])
    n_gated = len(
        _seg_layer(distance_gate_alpha=0.5).postprocess(raw, info).pred_masks[0]
    )
    assert n_gated < n_base


def _build_split_animal_scene(stride=2):
    """Raw head maps for ONE animal over-segmented into two abutting halves.

    Two detectable center peaks sit inside a single elongated ellipse with an
    ADDITIVE (high-ridge) heatmap between them, so
    :func:`group_instances_from_offsets` yields two touching fragments that the
    valley-driven fragment-merge re-fuses into one. Returns the
    ``SegmentationLayer.postprocess`` raw dict + info.
    """
    h, w = 100, 140
    fg = np.zeros((h, w), np.float32)
    ctr = np.zeros((h, w), np.float64)
    off = np.zeros((2, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    cx0, cy0 = 70, 50
    ell = ((ys - cy0) / 14.0) ** 2 + ((xs - cx0) / 34.0) ** 2 <= 1.0
    fg[ell] = 1.0
    for cx, cy in [(60, 50), (80, 50)]:  # 20 grid apart -> two distinct peaks
        ctr += _gauss_peak(h, w, cx, cy, sigma=8.0, amp=1.0)
    ctr = np.clip(ctr, 0, 1.0).astype(np.float32)
    left, right = ell & (xs < cx0), ell & (xs >= cx0)
    off[0][left] = (60 - xs[left]) * stride
    off[1][left] = (50 - ys[left]) * stride
    off[0][right] = (80 - xs[right]) * stride
    off[1][right] = (50 - ys[right]) * stride
    raw = {
        "SegmentationHead": torch.from_numpy(fg).reshape(1, 1, h, w),
        "InstanceCenterHead": torch.from_numpy(ctr).reshape(1, 1, h, w),
        "CenterOffsetHead": torch.from_numpy(off).reshape(1, 2, h, w),
    }
    info = PreprocInfo(
        original_size=(h * stride, w * stride),
        processed_size=(h * stride, w * stride),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=stride,
    )
    return raw, info


def test_segmentation_layer_merge_reduces_masks():
    """``merge_fragments=True`` fuses the two split halves THROUGH the layer.

    Exercises the full postprocess merge branch (the ``center_heatmap[b,0].numpy()``
    / ``offsets[b].numpy()`` conversion + every forwarded ``merge_*`` kwarg) for
    both agglomeration methods, and confirms the OFF default leaves the
    over-segmented pair intact.
    """
    raw, info = _build_split_animal_scene()
    n_off = len(_seg_layer(merge_fragments=False).postprocess(raw, info).pred_masks[0])
    assert n_off == 2  # the animal is over-segmented into two fragments
    for method in ("greedy", "multicut"):
        n_on = len(
            _seg_layer(merge_fragments=True, merge_method=method)
            .postprocess(raw, info)
            .pred_masks[0]
        )
        assert n_on == 1, (method, n_on)


def test_segmentation_layer_merge_dilate_zero_still_merges():
    """``merge_dilate=0`` is clamped to 1 so the merge is not silently disabled.

    The grouped fragments are mutually exclusive; a zero-dilation contact test
    would report no contact and skip every merge. ``_contact_fraction`` clamps to
    at least one dilation, so the fusion still happens.
    """
    raw, info = _build_split_animal_scene()
    n_on = len(
        _seg_layer(merge_fragments=True, merge_method="greedy", merge_dilate=0)
        .postprocess(raw, info)
        .pred_masks[0]
    )
    assert n_on == 1
