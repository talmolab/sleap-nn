"""Tests for #618: output-stride mask encoding + cleanup morphology + polygon output.

Covers the acceptance criteria and the adversarial-review required cases:
  A. encode-at-stride is lossless at model resolution; ~stride-smaller RLE; the
     emitted scale/offset recover the image extent (with the known +/-1 px
     rounding); eval stays correct (scale-aware decode).
  B. morphology cleanup (radius>0) despeckles; radius==0 is byte-identical.
  C. polygon output produces a loadable .slp ROI; mask stays exact.
  + min_mask_area unit conversion, backward-compat, degenerate cases, and the
    training/pseudo-label roundtrip (stride-encoded .slp -> correctly-scaled GT).
"""

import numpy as np
import pytest
import torch

import sleap_io as sio

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.segmentation_convert import (
    build_predicted_roi,
    build_predicted_segmentation_mask,
    decode_mask_to_image_res,
)
from sleap_nn.inference.layers.segmentation import SegmentationLayer
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.layers.configs import PostprocessConfig
from sleap_nn.inference.segmentation import _clean_instance_mask
from sleap_nn.data.segmentation_maps import (
    generate_center_heatmap,
    generate_center_offsets,
    generate_foreground_mask,
)


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _mask_iou(a, b):
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((H, W), bool)
    pb = np.zeros((H, W), bool)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b
    inter = int(np.logical_and(pa, pb).sum())
    union = int(np.logical_or(pa, pb).sum())
    return 1.0 if union == 0 else inter / union


# ── A. encode-at-stride round-trip + scale/offset ────────────────────────────


def test_stride_encode_lossless_at_model_resolution():
    """from_numpy(scale=1/s).data is bit-identical to the stored stride mask."""
    m = np.zeros((50, 60), bool)
    m[10:40, 12:48] = True
    for s in (1, 2, 4):
        sm = build_predicted_segmentation_mask(m, 0.9, scale=(1.0 / s, 1.0 / s))
        assert np.array_equal(np.asarray(sm.data, dtype=bool), m)  # IoU == 1.0
        # image_extent recovers s*size (the original grid the mask maps onto).
        assert sm.image_extent == (50 * s, 60 * s)


def test_stride_encode_rle_smaller_than_full_res_noisy():
    """A noisy mask encodes to fewer RLE bytes at stride than at full res."""
    rng = np.random.default_rng(0)
    full = _disk(256, 256, 128, 128, 60)
    # speckle halo (the RLE-exploding case)
    idx = rng.integers(0, 256 * 256, size=2000)
    flat = full.reshape(-1).copy()
    flat[idx] = True
    full = flat.reshape(256, 256)
    small = full[::2, ::2]  # exact stride-2 downsample stand-in
    full_bytes = np.asarray(
        build_predicted_segmentation_mask(full, 1.0).rle_counts
    ).nbytes
    stride_bytes = np.asarray(
        build_predicted_segmentation_mask(small, 1.0, scale=(0.5, 0.5)).rle_counts
    ).nbytes
    assert stride_bytes < full_bytes


def test_mask_to_stride_scale_offset_and_pad_crop():
    """_mask_to_stride: scale = valid/orig, offset 0, ceil pad crop.

    Worked example (input_scale=0.5, eff=0.8, stride=2, orig=1024): s=0.4,
    scaled=round(1024*0.4)=410, valid=ceil(410/2)=205, and scale=205/1024≈0.2002
    (the exact inverse of the crop, NOT the raw s/stride=0.2), so image_extent
    recovers 1024 EXACTLY. For other configs image_extent can still be off by ±1
    px (round(orig*s) rounding), so it is NOT authoritative for the true frame size.
    """
    layer = SegmentationLayer.__new__(SegmentationLayer)
    layer.output_stride = 2
    head = np.ones((220, 220), bool)  # head map larger than the valid extent
    info = PreprocInfo(
        original_size=(1024, 1024),
        processed_size=(440, 440),
        eff_scale=torch.tensor([0.8]),
        input_scale=0.5,
        output_stride=2,
    )
    mask_stride, scale, offset = layer._mask_to_stride(head, info, 0)
    # scale = valid/orig = 205/1024 ≈ 0.2002 (not the raw s/stride = 0.2).
    assert scale[0] == pytest.approx(0.2, abs=1e-3)
    assert scale[1] == pytest.approx(0.2, abs=1e-3)
    assert offset == (0.0, 0.0)
    assert mask_stride.shape == (205, 205)  # ceil(round(1024*0.4)/2)
    # valid/orig recovers orig exactly for THIS config; ±1 in general (see the
    # parametrized test below). The s/stride bug truncated by up to a stride cell.
    sm = build_predicted_segmentation_mask(mask_stride, 1.0, scale=scale)
    assert sm.image_extent == (1024, 1024)


@pytest.mark.parametrize(
    "orig,eff,iscale,stride",
    [
        (1000, 1.0, 1.0, 16),  # exact
        (200, 1.0, 1.0, 16),  # exact
        (1024, 1.0, 0.2, 4),  # exact
        (777, 1.0, 0.5, 8),  # exact
        (101, 1.0, 1.0, 4),  # exact
        (100, 1.0, 1.0, 16),  # float truncation -> orig-1 (99)
        (50, 1.0, 1.0, 8),  # -> orig-1 (49)
        (25, 1.0, 1.0, 4),  # -> orig-1 (24)
    ],
)
def test_mask_to_stride_image_extent_recovers_orig_at_large_stride(
    orig, eff, iscale, stride
):
    """Stored scale (valid/orig) recovers orig to within ±1 px (it is orig or
    orig-1 — never an overshoot) even at large stride / fractional configs. The
    `int(valid/(valid/orig))` float division can floor-truncate to orig-1 for ~5%
    of sizes; consumers tolerate the ±1. Regression for the s/stride scale bug,
    which truncated orig by up to a full stride cell (e.g. 1000@stride16 -> 992)."""
    layer = SegmentationLayer.__new__(SegmentationLayer)
    layer.output_stride = stride
    head = np.ones((orig, orig), bool)  # >= the valid extent for any of these
    info = PreprocInfo(
        original_size=(orig, orig),
        processed_size=(orig, orig),
        eff_scale=torch.tensor([eff]),
        input_scale=iscale,
        output_stride=stride,
    )
    mask_stride, scale, _ = layer._mask_to_stride(head, info, 0)
    sm = build_predicted_segmentation_mask(mask_stride, 1.0, scale=scale)
    # orig or orig-1, never an overshoot (the s/stride bug gave orig-8 here).
    assert sm.image_extent[0] in (orig - 1, orig)
    assert sm.image_extent[1] in (orig - 1, orig)


def test_postprocess_default_stride_and_full_res_escape():
    """Default emits stride masks (scale 1/s); full_res_masks restores orig res."""
    H, W, s = 128, 128, 2
    masks = [_disk(H, W, 30, 40, 18), _disk(H, W, 90, 95, 18)]
    fg = generate_foreground_mask(masks, (H, W), output_stride=s)
    center = generate_center_heatmap(masks, (H, W), output_stride=s, sigma=6.0)
    offsets, _ = generate_center_offsets(masks, (H, W), output_stride=s)
    raw = {
        "SegmentationHead": fg,
        "InstanceCenterHead": center,
        "CenterOffsetHead": offsets,
    }
    info = PreprocInfo(
        original_size=(H, W),
        processed_size=(H, W),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=s,
    )

    layer = SegmentationLayer.__new__(SegmentationLayer)
    layer.output_stride = s
    layer.fg_threshold = 0.5
    layer.min_mask_area = 0
    layer.postprocess_config = PostprocessConfig(peak_threshold=0.2)

    out = layer.postprocess(raw, info)
    for inst in out.pred_masks[0]:
        assert inst["mask"].shape == (H // s, W // s)
        assert inst["scale"] == (0.5, 0.5)
        assert inst["offset"] == (0.0, 0.0)

    layer.full_res_masks = True
    out_fr = layer.postprocess(raw, info)
    for inst in out_fr.pred_masks[0]:
        assert inst["mask"].shape == (H, W)
        assert inst["scale"] == (1.0, 1.0)


# ── min_mask_area unit conversion ────────────────────────────────────────────


def test_min_mask_area_invariance_integer_stride():
    """For integer 1/(sx*sy) (eff=input_scale=1), stride filter == full_res filter."""
    H, W, s = 128, 128, 2
    big = _disk(H, W, 40, 40, 22)
    tiny = _disk(H, W, 95, 95, 5)
    masks = [big, tiny]
    fg = generate_foreground_mask(masks, (H, W), output_stride=s)
    center = generate_center_heatmap(masks, (H, W), output_stride=s, sigma=6.0)
    offsets, _ = generate_center_offsets(masks, (H, W), output_stride=s)
    raw = {
        "SegmentationHead": fg,
        "InstanceCenterHead": center,
        "CenterOffsetHead": offsets,
    }
    info = PreprocInfo(
        original_size=(H, W),
        processed_size=(H, W),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=s,
    )

    def _layer(min_area, full_res):
        ly = SegmentationLayer.__new__(SegmentationLayer)
        ly.output_stride = s
        ly.fg_threshold = 0.5
        ly.min_mask_area = min_area
        ly.full_res_masks = full_res
        ly.postprocess_config = PostprocessConfig(peak_threshold=0.2)
        return ly

    # Dense sweep so a floor whose /stride^2 has fractional part < 0.5 (where
    # round() would over-keep but ceil() matches full_res) is actually exercised.
    for floor in list(range(0, 130)) + [400, 800]:
        n_stride = len(_layer(floor, False).postprocess(raw, info).pred_masks[0])
        n_full = len(_layer(floor, True).postprocess(raw, info).pred_masks[0])
        assert n_stride == n_full, f"floor={floor}: {n_stride} != {n_full}"


# ── eval scale-awareness ─────────────────────────────────────────────────────


def test_eval_frame_masks_scale_aware_roundtrip():
    """A stride-res pred that is the exact downsample of an orig-res GT -> IoU 1.0."""
    from sleap_nn.evaluation import _frame_masks, _mask_iou as eval_iou

    # GT is block-constant on the stride-2 grid (kron upsample of the small mask),
    # so the stride-encoded pred resamples back to it EXACTLY -> IoU 1.0. This
    # isolates the scale-aware alignment: without the resample, a (60,60) pred vs
    # (120,120) GT would be top-left-aligned and badly misaligned.
    small = _disk(60, 60, 30, 30, 20)
    gt = np.kron(small, np.ones((2, 2), dtype=bool))  # (120, 120)
    gt_mask = sio.UserSegmentationMask.from_numpy(gt)  # scale 1
    pred_mask = sio.PredictedSegmentationMask.from_numpy(
        small, score=1.0, scale=(0.5, 0.5)
    )
    gt_frame = sio.LabeledFrame(video=None, frame_idx=0, masks=[gt_mask])
    pred_frame = sio.LabeledFrame(video=None, frame_idx=0, masks=[pred_mask])
    g = _frame_masks(gt_frame)[0]
    p = _frame_masks(pred_frame)[0]
    assert g.shape == (120, 120)
    assert p.shape == (120, 120)  # resampled up from (60, 60)
    assert eval_iou(g, p) == 1.0
    # Sanity: the naive (no-resample) top-left comparison would be far worse,
    # which is exactly the bug the scale-aware decode fixes.
    assert _mask_iou(small, gt) < 0.5


def test_decode_scale1_fast_path_identity():
    """Scale-1 masks decode unchanged (legacy .slp behavior preserved)."""
    m = _disk(64, 64, 32, 32, 20)
    sm = sio.UserSegmentationMask.from_numpy(m)
    assert np.array_equal(decode_mask_to_image_res(sm), m)


# ── B. cleanup morphology ────────────────────────────────────────────────────


def test_clean_radius0_identical_to_keep_largest_fill():
    """radius=0 reproduces the keep-largest-CC + fill behavior byte-for-byte."""
    from scipy.ndimage import binary_fill_holes
    from scipy.ndimage import label as cc_label

    rng = np.random.default_rng(1)
    blob = _disk(80, 80, 40, 40, 18)
    speckle = rng.random((80, 80)) < 0.02
    noisy = blob | speckle

    labels, n = cc_label(noisy)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    manual = binary_fill_holes(labels == int(counts.argmax()))

    assert np.array_equal(_clean_instance_mask(noisy, radius=0), manual)


def test_clean_radius_removes_speckle():
    """radius>0 morphology drops isolated speckle the CC-keep would otherwise miss
    only because it is disconnected; result RLE is no larger than the clean blob."""
    rng = np.random.default_rng(2)
    blob = _disk(120, 120, 60, 60, 30)
    speckle = rng.random((120, 120)) < 0.03
    noisy = blob | speckle
    cleaned = _clean_instance_mask(noisy, radius=2)
    # speckle gone -> single solid blob close to the original
    assert _mask_iou(cleaned, blob) > 0.9
    clean_runs = len(build_predicted_segmentation_mask(blob, 1.0).rle_counts)
    cleaned_runs = len(build_predicted_segmentation_mask(cleaned, 1.0).rle_counts)
    assert cleaned_runs <= clean_runs * 1.5


# ── C. polygon output ────────────────────────────────────────────────────────


def _seg_outputs(stride=2, H=128, W=128):
    masks = [_disk(H, W, 40, 40, 22)]
    fg = generate_foreground_mask(masks, (H, W), output_stride=stride)
    center = generate_center_heatmap(masks, (H, W), output_stride=stride, sigma=6.0)
    offsets, _ = generate_center_offsets(masks, (H, W), output_stride=stride)
    raw = {
        "SegmentationHead": fg,
        "InstanceCenterHead": center,
        "CenterOffsetHead": offsets,
    }
    info = PreprocInfo(
        original_size=(H, W),
        processed_size=(H, W),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=stride,
    )
    layer = SegmentationLayer.__new__(SegmentationLayer)
    layer.output_stride = stride
    layer.fg_threshold = 0.5
    layer.min_mask_area = 0
    layer.postprocess_config = PostprocessConfig(peak_threshold=0.2)
    out = layer.postprocess(raw, info)
    return Outputs(
        pred_masks=out.pred_masks,
        frame_indices=torch.tensor([0]),
        video_indices=torch.tensor([0]),
    )


def test_mask_output_both_emits_mask_and_roi(tmp_path):
    out = _seg_outputs()
    video = sio.Video.from_filename("dummy.mp4")
    labels = out.to_labels(skeleton=None, videos=[video], mask_output="both")
    assert len(labels[0].masks) == 1
    assert len(labels[0].rois) == 1
    assert isinstance(labels[0].rois[0], sio.PredictedROI)
    # .slp round-trips the ROI.
    p = tmp_path / "both.slp"
    sio.save_slp(labels, p.as_posix())
    reloaded = sio.load_slp(p.as_posix())
    assert len(reloaded[0].rois) == 1
    assert len(reloaded[0].masks) == 1


def test_mask_output_polygon_emits_roi_only():
    out = _seg_outputs()
    video = sio.Video.from_filename("dummy.mp4")
    labels = out.to_labels(skeleton=None, videos=[video], mask_output="polygon")
    assert len(labels[0].masks) == 0
    assert len(labels[0].rois) == 1


def test_polygon_epsilon_reduces_vertices():
    out = _seg_outputs()
    raw_rois = out.to_rois(batch_index=0, epsilon=0.0)
    simp_rois = out.to_rois(batch_index=0, epsilon=0.05)
    n_raw = len(raw_rois[0].geometry.exterior.coords)
    n_simp = len(simp_rois[0].geometry.exterior.coords)
    assert n_simp < n_raw


def test_mask_output_mask_default_no_rois():
    out = _seg_outputs()
    video = sio.Video.from_filename("dummy.mp4")
    labels = out.to_labels(skeleton=None, videos=[video])  # default 'mask'
    assert len(labels[0].masks) == 1
    assert len(labels[0].rois) == 0


def test_incremental_writer_polygon_output(tmp_path):
    """The STREAMING path (IncrementalLabelsWriter) honors mask_output=polygon.

    Covers the writer.mask_output/polygon_epsilon attrs + their to_labels kwargs,
    which the in-memory to_labels tests do not exercise.
    """
    from sleap_nn.inference.writer import IncrementalLabelsWriter

    out = _seg_outputs()
    video = sio.Video.from_filename("dummy.mp4")
    p = tmp_path / "poly_stream.slp"
    writer = IncrementalLabelsWriter(
        path=p.as_posix(),
        skeleton=None,
        videos=[video],
        mask_output="polygon",
        polygon_epsilon=0.02,
    )
    with writer:
        writer.write(out)
    reloaded = sio.load_slp(p.as_posix())
    assert len(reloaded[0].masks) == 0
    assert len(reloaded[0].rois) == 1


def test_incremental_writer_both_output(tmp_path):
    """Streaming path with mask_output=both emits exact mask + simplified ROI."""
    from sleap_nn.inference.writer import IncrementalLabelsWriter

    out = _seg_outputs()
    video = sio.Video.from_filename("dummy.mp4")
    p = tmp_path / "both_stream.slp"
    writer = IncrementalLabelsWriter(
        path=p.as_posix(), skeleton=None, videos=[video], mask_output="both"
    )
    with writer:
        writer.write(out)
    reloaded = sio.load_slp(p.as_posix())
    assert len(reloaded[0].masks) == 1
    assert len(reloaded[0].rois) == 1


# ── backward compatibility ───────────────────────────────────────────────────


def test_build_predicted_segmentation_mask_two_arg_back_compat():
    m = _disk(40, 40, 20, 20, 12)
    sm = build_predicted_segmentation_mask(m, 0.8)  # legacy 2-arg call
    assert sm.scale == (1.0, 1.0)
    assert sm.offset == (0.0, 0.0)


def test_to_masks_without_scale_offset_keys():
    """A pred_masks dict lacking scale/offset still converts (identity default)."""
    m = _disk(40, 40, 20, 20, 12)
    out = Outputs(pred_masks=[[{"mask": m, "score": 0.5}]])
    masks = out.to_masks(batch_index=0)
    assert len(masks) == 1
    assert masks[0].scale == (1.0, 1.0)


# ── degenerate cases ─────────────────────────────────────────────────────────


def test_empty_mask_dropped_and_empty_roi_none():
    out = Outputs(pred_masks=[[{"mask": np.zeros((20, 20), bool), "score": 0.5}]])
    assert out.to_masks(0) == []
    assert out.to_rois(0) == []
    empty = build_predicted_segmentation_mask(np.zeros((10, 10), bool), 0.5)
    assert build_predicted_roi(empty, 0.5) is None


# ── training / pseudo-label roundtrip (critic BLOCKER) ───────────────────────


def test_stride_encoded_slp_trains_with_correct_scale(minimal_instance_seg, tmp_path):
    """A .slp whose masks are stride-encoded yields the SAME GT foreground as the
    original full-res masks (the BottomUpSegmentationDataset decode is scale-aware).

    Regression for the silent-corruption path: __getitem__'s resize branch only
    fires on a preprocessing size change, so a stride-res m.data would otherwise
    feed mis-scaled targets to generate_foreground_mask.
    """
    from omegaconf import OmegaConf

    from sleap_nn.data.custom_datasets import BottomUpSegmentationDataset

    orig = sio.load_slp(minimal_instance_seg.as_posix())

    # Re-encode every GT mask at output-stride 2 (scale 0.5), as the inference
    # layer now does by default, and write a new .slp.
    for lf in orig.labeled_frames:
        new_masks = []
        for m in lf.masks:
            dense = np.asarray(m.data, dtype=bool)
            small = dense[::2, ::2]
            new_masks.append(
                sio.UserSegmentationMask.from_numpy(small, scale=(0.5, 0.5))
            )
        lf.masks = new_masks
    stride_path = tmp_path / "stride_encoded.slp"
    sio.save_slp(orig, stride_path.as_posix())

    def _build(labels):
        return BottomUpSegmentationDataset(
            labels=[labels],
            seg_head_config=OmegaConf.create({"output_stride": 2, "loss_weight": 1.0}),
            center_head_config=OmegaConf.create(
                {"sigma": 5.0, "output_stride": 2, "loss_weight": 1.0}
            ),
            offset_head_config=OmegaConf.create(
                {"output_stride": 2, "loss_weight": 0.1}
            ),
            max_stride=16,
            ensure_grayscale=True,
            cache_img=None,
        )

    ref = _build(sio.load_slp(minimal_instance_seg.as_posix()))[0]
    got = _build(sio.load_slp(stride_path.as_posix()))[0]

    # Decoded GT foreground from the stride-encoded .slp matches the original.
    assert got["foreground_mask"].shape == ref["foreground_mask"].shape
    inter = (got["foreground_mask"] * ref["foreground_mask"]).sum().item()
    union = ((got["foreground_mask"] + ref["foreground_mask"]) > 0).sum().item()
    assert union > 0 and inter / union > 0.9
