# Phase C (partial): `bottomup_segmentation` tiling — build notes + deferred corners

**Branch:** `tiling/05-bottomup-seg` (stacked on `tiling/04-viz-docs`)
**Goal:** a *real* tiled-**training** run of `bottomup_segmentation` on the
plant-seg root data (`scratch/2026-07-01-plant-seg`), producing a wandb link.

`bottomup_segmentation` is CenterNet-style — three per-pixel heads: **foreground
mask** + **center heatmap** + **center offsets** (each pixel points to its
instance's center). **No PAF**, so the hardest Phase-C seam problem (PAF field
stitching) does not arise here. The seam-sensitive target is the **center-offset
field**: a foreground pixel whose instance center lies in a neighbor tile would
regress an off-tile (unlearnable) offset. Fix = **ownership + halo**: a tile owns
only instances whose centroid falls inside it, and generates targets from those
owned instances only; with overlap ≥ instance extent, every instance is wholly
owned by *some* tile.

## In scope for this build (tiled TRAINING)

- `BottomUpSegmentationTiledDataset` — mirrors `BottomUpSegmentationDataset`
  (`custom_datasets.py:2955`) but per-`(frame, tile)`: decode frame + masks once
  (per-worker LRU), draw/pin tile origin (foreground centers = mask centroids),
  slice image + co-slice masks to the tile, **ownership-filter** masks by centroid,
  then `generate_foreground_mask` / `generate_center_heatmap` /
  `generate_center_offsets` on tile-local owned masks at `(tile_size, tile_size)`.
- **±15° rotation augmentation (mask-aware halo).** Per the user's target, geometric
  aug is ON (rotation ±15°). The tile is cut with a √2 **halo**, image **and** masks
  are co-transformed by the SAME affine (`apply_geometric_augmentation(..., masks=)`,
  nearest-neighbor, re-binarized — exactly as `BottomUpSegmentationDataset` does at
  `custom_datasets.py:3170`), then trimmed back to `tile_size`. This is NOT the
  shortcut (the config on disk had `geometric: null`); we deliberately enable
  rotation to exercise the mask co-transform path.
- Factory branch: route `bottomup_segmentation` → tiled dataset when `tiling.enabled`
  (train foreground sampling, val grid).
- `check_tiling` hardening: a **supported-model-types** allowlist
  (`single_instance`, `bottomup_segmentation`) — any other type + `tiling.enabled`
  now raises instead of silently training whole-frame (the footgun found this session).

## Deferred corners — COME BACK TO THESE

1. **Tiled bottom-up-seg INFERENCE + grouping is NOT built.** Stitching the
   foreground / center / offset canvases across tiles and running the global
   center-offset grouping is the harder Phase-C second half. Therefore **eval is
   OFF during this training run** (`SegmentationEvaluationCallback` /
   `trainer_config.eval.enabled=false`) — it would otherwise run *whole-frame*
   inference on a tile-trained model (scale mismatch) or need the missing tiled
   inference. The run proves tiled *training* (loss curves + GT/pred viz on tiles),
   not end-to-end tiled prediction quality. **TODO:** implement the tiled seg
   inference layer + grouping, then re-enable eval.
2. **Ownership vs long instances.** Arabidopsis primary roots can reach ~500 px
   (per the config's offset-loss note). If `overlap < instance extent`, a long root
   may not be wholly owned by any tile → its center-offset field is truncated at
   seams even for the owning tile. Mitigation here: pick `tile_size`/`overlap` large
   enough to contain typical roots; very long roots remain imperfectly owned.
   **TODO:** the SAHI full-frame-mix (DQ2b, built but default-off) is the principled
   fix for objects larger than a tile — wire it in for seg.
3. **`scale != 1` tiled seg.** This run is `scale=1.0`. The tiled mask path slices
   masks at image resolution; a non-unit `scale` (mask resize is not pad-aware, per
   the existing seg dataset's own caveat) is untested under tiling. **TODO.**
4. **No per-tile validity/loss mask** for constant-zero-padded border regions.
   Consistent with the Phase-A design (zero pad → background target), but seg
   foreground/offset losses on padded regions are unaudited. **TODO if border
   artifacts appear.**
5. **Empty/background tiles + offset loss.** Foreground sampling yields many
   background tiles (no owned instance → empty foreground / zero center heatmap /
   zero offsets). Confirm the offset + BCE/Dice losses are well-behaved (finite,
   not dominated) under a high fraction of empty tiles. `tile_fg_fraction` tunes
   this. **TODO:** consider batch-Dice (DQ11) if seg training is unstable.
6. **Auto-sizing for thin/long objects.** The label-derived `tile_size`/`overlap`
   auto-sizer uses max-bbox extent; for long thin roots this may over-size. This run
   sets geometry explicitly. **TODO:** validate auto-sizing on thin-object data.

## Run config (tiling variant)

Derived from `scratch/2026-07-01-plant-seg/configs/bottomup_seg_convnext.yaml`:
`tiling.enabled=true`, explicit `tile_size`/`overlap`, geometric aug = rotation ±15°,
`eval.enabled=false`, wandb `project: plant-seg`, a distinct run name/group.
