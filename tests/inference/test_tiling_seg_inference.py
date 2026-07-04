"""Tests for tiled bottom-up instance-segmentation inference.

Covers :class:`~sleap_nn.inference.layers.tiled.TiledSegmentationLayer`:

  - Unit (synthetic, CPU): a single-tile run (``tile_size >= frame``) reproduces
    the whole-frame ``SegmentationLayer`` masks exactly; a 2-tile stitch
    reconstructs a known foreground/center/offset field within tolerance (the
    4-channel accumulate-normalize merge is validated directly).
  - REAL-CKPT gate (slow, skipped if the checkpoint/fixture is missing): the
    just-trained tiled bottom-up seg model auto-routes to
    ``TiledSegmentationLayer`` via ``Predictor.from_model_paths`` and produces a
    finite (non-NaN) mean mask IoU with a positive instance count on the plant
    primary-root validation set.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.segmentation import SegmentationLayer
from sleap_nn.inference.layers.tiled import TiledSegmentationLayer


# ---------------------------------------------------------------------------
# Synthetic backend: a spatially-local, deterministic stand-in for a trained
# bottom-up seg model. Each head is a fixed function of the tile's own pixels
# (an ``output_stride`` average pool), so overlapping tiles predict IDENTICAL
# values for the same global pixel — which is exactly what makes the stitched
# canvas reconstruct the whole-frame forward within float tolerance.
# ---------------------------------------------------------------------------
class _LocalSegBackend:
    """Minimal ``ModelBackend`` producing the three seg heads from the input."""

    device = "cpu"
    does_baked_postproc = False

    def __init__(self, stride: int):
        self.stride = int(stride)

    def __call__(self, x: torch.Tensor) -> dict:
        # x: (n, 1, C, ts, ts) — the layer feeds tiles with an n_samples axis
        # that the backend contract squeezes.
        img = x.squeeze(1).float()  # (n, C, ts, ts)
        small = F.avg_pool2d(img.mean(1, keepdim=True), self.stride)  # (n, 1, h, w)
        fg = torch.sigmoid((small - 0.4) * 30.0)  # already-sigmoid foreground
        cen = small
        off = torch.cat([small * 2.0, small * -3.0], dim=1)  # (n, 2, h, w)
        return {
            "SegmentationHead": fg,
            "InstanceCenterHead": cen,
            "CenterOffsetHead": off,
        }

    def warmup(self, input_shape):  # pragma: no cover — protocol completeness
        pass


def _make_inner(stride: int, tile_max_stride: int) -> SegmentationLayer:
    """A whole-frame ``SegmentationLayer`` over the synthetic local backend."""
    return SegmentationLayer(
        backend=_LocalSegBackend(stride),
        output_stride=stride,
        max_stride=tile_max_stride,
        fg_threshold=0.5,
        preprocess_config=PreprocessConfig(scale=1.0, ensure_grayscale=True),
        postprocess_config=PostprocessConfig(peak_threshold=0.1),
    )


def _masks_sorted(outputs, b: int = 0):
    """Per-instance boolean masks for batch slot ``b``, sorted by pixel count."""
    return sorted(
        (m["mask"] for m in outputs.pred_masks[b]), key=lambda a: (a.sum(), a.shape)
    )


# ---------------------------------------------------------------------------
# 1. One-tile run equals the whole-frame SegmentationLayer masks
# ---------------------------------------------------------------------------
def test_single_tile_equals_whole_frame():
    """A tile >= frame reduces to the whole-frame masks, byte-for-byte."""
    stride, ts = 2, 16
    inner = _make_inner(stride, tile_max_stride=8)
    tiled = TiledSegmentationLayer(
        inner,
        tile_size=ts,
        overlap=8,
        blend="gaussian",
        tile_batch_size=4,
        accumulator_device="cpu",
    )

    rng = np.random.default_rng(0)
    frame = (rng.random((1, 12, 12)) * 255).astype(np.uint8)  # (C, H, W) < tile

    out_tiled = tiled.predict(frame)
    out_whole = inner.predict(frame)

    tm, wm = _masks_sorted(out_tiled), _masks_sorted(out_whole)
    assert len(tm) == len(wm) and len(wm) > 0
    for a, b in zip(tm, wm):
        assert a.shape == b.shape
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 2. A synthetic 2-tile stitch reconstructs a known fg/center/offset field
# ---------------------------------------------------------------------------
def test_two_tile_stitch_reconstructs_field():
    """The 4-channel accumulate-normalize merge recovers the whole-frame heads."""
    stride, ts = 2, 16
    inner = _make_inner(stride, tile_max_stride=8)
    tiled = TiledSegmentationLayer(
        inner,
        tile_size=ts,
        overlap=8,  # step 8 -> a 24x40 frame needs multiple tiles per axis
        blend="gaussian",
        tile_batch_size=4,
        accumulator_device="cpu",
    )

    # Capture the stitched heads the tiled layer hands to postprocess.
    captured: dict = {}
    orig_postprocess = inner.postprocess

    def _spy(raw_out, info):
        captured["fg"] = raw_out["SegmentationHead"].clone()
        captured["cen"] = raw_out["InstanceCenterHead"].clone()
        captured["off"] = raw_out["CenterOffsetHead"].clone()
        return orig_postprocess(raw_out, info)

    inner.postprocess = _spy

    rng = np.random.default_rng(1)
    frame = (rng.random((1, 24, 40)) * 255).astype(np.uint8)  # multi-tile frame
    tiled.predict(frame)
    inner.postprocess = orig_postprocess

    # Reference: the same local backend run on the WHOLE frame in one shot.
    ref = _LocalSegBackend(stride)(torch.from_numpy(frame).unsqueeze(0).unsqueeze(0))

    for key, ref_key in (
        ("fg", "SegmentationHead"),
        ("cen", "InstanceCenterHead"),
        ("off", "CenterOffsetHead"),
    ):
        got = captured[key][0]  # (C, H, W)
        want = ref[ref_key][0]
        assert got.shape == want.shape
        assert float((got - want).abs().max()) < 1e-3, (
            key,
            float((got - want).abs().max()),
        )


# ---------------------------------------------------------------------------
# 3. REAL-CKPT GATE — auto-routing + a real, finite mask IoU on the checkpoint
# ---------------------------------------------------------------------------
_CKPT = Path("scratch/2026-07-01-plant-seg/runs/primary_bu_seg_tiling_convnext")
_VAL = Path("scratch/2026-07-01-plant-seg/masks/primary_val.pkg.slp")
_N_EVAL_FRAMES = 5


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CKPT / "best.ckpt").exists() or not _VAL.exists(),
    reason=f"real tiled seg checkpoint/fixture not found: {_CKPT} / {_VAL}",
)
def test_real_ckpt_tiled_seg_finite_mask_iou():
    """The trained tiled seg ckpt auto-routes to tiling and gives a real IoU."""
    import sleap_io as sio

    from sleap_nn.evaluation import _frame_masks, match_masks
    from sleap_nn.inference.predictor import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor.from_model_paths([str(_CKPT)], device=device)

    # The trained config has tiling.enabled=true (tile_size=768, overlap=384),
    # so the loader MUST auto-route to the tiled segmentation layer.
    assert isinstance(predictor.layer, TiledSegmentationLayer), type(predictor.layer)
    assert predictor.layer.tile_size == 768
    assert predictor.layer.overlap == 384

    full = sio.load_slp(str(_VAL))
    sub = sio.Labels(
        labeled_frames=full.labeled_frames[:_N_EVAL_FRAMES],
        videos=full.videos,
        skeletons=full.skeletons,
    )
    predicted = predictor.predict(sub, make_labels=True)

    per_frame_pred = [len(getattr(lf, "masks", None) or []) for lf in predicted]
    total_pred = int(sum(per_frame_pred))
    assert total_pred > 0, "tiled seg produced no masks on any frame"
    assert any(n > 0 for n in per_frame_pred)

    gt_by_key = {(lf.video, lf.frame_idx): lf for lf in sub.labeled_frames}
    all_ious: list = []
    for lf in predicted:
        gt = gt_by_key.get((lf.video, lf.frame_idx))
        if gt is None:
            continue
        pred_masks = _frame_masks(lf)
        gt_masks = _frame_masks(gt)
        # min_iou ~ 0 keeps every real overlap (the model trained poorly, so IoUs
        # are low; we assert the number is REAL, not that it clears 0.5).
        _mp, _mg, _up, _ug, ious = match_masks(pred_masks, gt_masks, min_iou=1e-6)
        all_ious.extend(float(x) for x in ious)

    mean_iou = float(np.mean(all_ious)) if all_ious else float("nan")
    print(
        f"\n[real-ckpt tiled seg] frames={len(predicted)} "
        f"total_pred_masks={total_pred} per_frame={per_frame_pred} "
        f"matched_pairs={len(all_ious)} mean_mask_iou={mean_iou:.4f}"
    )
    assert np.isfinite(mean_iou), "mean mask IoU must be a finite (non-nan) number"
    assert len(all_ious) > 0, "expected at least one overlapping pred/GT mask pair"
