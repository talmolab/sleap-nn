# Whole-frame semantic (foreground/background) segmentation

Predict **one binary foreground mask over the whole frame** — animal / plant /
region-of-interest vs background — with **no instance grouping**. A lone
`SegmentationHead` runs on the full frame and inference thresholds the foreground
probability into a single instance-less mask per frame.

Use it when you want a clean figure-ground mask and do **not** need to separate
individuals: region masks, "where is the animal?" gating, or **isolating
foreground quality from the grouping bottleneck** (thin structures, high-res /
small-object frames). It is the group-free sibling of bottom-up segmentation
(`bottomup_segmentation`, which adds center + offset heads to split the foreground
into instances) and the whole-frame analog of top-down segmentation
(`centered_instance_segmentation`, which runs the same foreground head on a
centroid crop).

Model type / head-config key: **`semantic_segmentation`**. Sample config:
[`config_semantic_segmentation_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_semantic_segmentation_unet.yaml).

| | `semantic_segmentation` | `bottomup_segmentation` | `centered_instance_segmentation` |
|---|---|---|---|
| Input | whole frame | whole frame | centroid crop |
| Heads | foreground only | foreground + center + offset | foreground only |
| Output | **one mask / frame** (instance-less) | one mask per grouped instance | one mask per centroid crop |
| Grouping | none | offset → nearest center | none (one crop = one instance) |
| Tiling | ✅ | ✅ | — |

## Training data: foreground masks

Training labels carry `sio.SegmentationMask` objects on `LabeledFrame.masks`. The
dataset takes the **union** of every mask on a frame as that frame's foreground
target (masks do **not** need to be linked to instances — `mask.instance` is
ignored). Any dataset that already works for `bottomup_segmentation` works here
unchanged; a per-instance-linked dataset also works (the links are simply unused).

Only frames that have at least one mask are used as training samples. The
foreground loss (BCE+Dice) tolerates all-background masks, so a frame whose masks
happen to be empty supervises "predict background everywhere."

If you only have poses, synthesize masks by rasterizing each instance and
attaching them (unlinked is fine):

```python
import sleap_io as sio

labels = sio.load_slp("poses.pkg.slp")
for lf in labels:
    lf.masks = [
        sio.UserSegmentationMask.from_numpy(rasterize_pose(inst))  # bool (H, W)
        for inst in lf.instances
    ]
labels.save("poses_masked.pkg.slp", embed=True)
```

### Training notes

- **Head:** a single `SegmentationHead` (BCE+Dice, logits) on the whole frame. No
  `center` / `offsets` heads and no `anchor_part` (there is no crop).
- **Augmentation:** intensity aug applies to the image; geometric aug
  (rotation/scale/translate/flip) co-transforms the GT foreground mask with the
  image (shared affine matrix, nearest-neighbor, re-binarized). Keep rotation
  ranges small — a full-frame rotation can clip foreground at the frame edge.
  Erase/mixup stay image-only.
- **Preprocessing:** mask resize is not pad-aware in v1 — train at `scale: 1.0`
  (or an evenly-dividing scale) with input dims divisible by the backbone
  `max_stride`.
- **Metrics:** `train/fg_loss`, `val/fg_loss`, and `val/fg_iou` (whole-frame
  foreground IoU, averaged per image) are logged every epoch. With
  `trainer_config.eval.enabled: true` a matching-free evaluation additionally logs
  `eval/val/fg_mean_iou`, `eval/val/fg_mean_cldice`, `eval/val/fg_mean_boundary_iou`
  and `eval/val/fg_frame_recall`.
- **Viz:** training visualizations overlay the predicted foreground probability and
  the GT mask on the frame.

## Thin / high-resolution structures (e.g. plant roots)

For thin, elongated foreground (roots, fibers, whiskers) at high resolution, the
default `output_stride: 2` bce-dice recipe systematically **under-segments the thin
parts**: the foreground target is area-downsampled by the stride and re-binarized at
50%, which erodes sub-cell structures out of the *label*, and the symmetric loss is
dominated by the easy background + thick parts, so the head hedges below 0.5 on
faint structure. The symptom is that lowering `--fg_threshold` at inference recovers
a lot of real foreground (topology/clDice climbs) — a patch for a training-time
defect. Three head-config knobs fix it at the source so the model is confident at
the default threshold:

- **`output_stride: 1`** (recommended) — dense, full-resolution prediction. No target
  downsample (no erosion) *and* a full-res decoder block to draw sharp thin ridges.
  Heavier per step, but the single biggest lever for thin structures.
- **`target_maxpool: true`** — cheaper alternative that keeps `output_stride: 2`:
  build the foreground target with max-pool semantics (any foreground pixel in a
  stride cell → foreground) instead of area-average + 0.5, so thin structures survive
  the downsample. Inert at `output_stride: 1`.
- **`bce_pos_weight` + `bce_weight` / `dice_weight`** — recalibrate the loss so 0.5 is
  the right operating point. For foreground that is <1% of pixels, `bce_pos_weight`
  ~5–20 up-weights the positive class and a tilt toward Dice
  (`bce_weight: 0.3, dice_weight: 0.7`) reduces easy-background dominance.

```yaml
model_config:
  head_configs:
    semantic_segmentation:
      segmentation:
        output_stride: 1        # dense; or keep 2 with target_maxpool: true
        loss_weight: 1.0
        bce_weight: 0.3
        dice_weight: 0.7
        bce_pos_weight: 10.0
        target_maxpool: false   # set true only if you keep output_stride: 2
```

These knobs live on the shared `SegmentationHeadConfig`, so `bottomup_segmentation`
gets them too; the defaults (`0.5`/`0.5`, `bce_pos_weight: null`, `target_maxpool:
false`) reproduce the previous behavior exactly. On a thin-root benchmark this recipe
raised whole-frame clDice at the *default* threshold from ~0.60 to ~0.68 and removed
the need for a hand-tuned inference threshold. Evaluate thin structures with **clDice**
(topology-aware), not IoU (area-dominated).

## Tiling (high-res / small-object frames)

Semantic segmentation is **tiling-compatible**: cut each frame into overlapping
square tiles, run per-tile at native resolution, and stitch the per-tile
foreground maps into one Gaussian-weighted canvas before thresholding. Enable it
under `data_config.preprocessing.tiling` (off by default; everything is
byte-identical when off):

```yaml
data_config:
  preprocessing:
    tiling:
      enabled: true
      sampling: foreground   # foreground-aware tile draws for training
```

Geometry (`tile_size` / `overlap`) is auto-sized from the labels at train setup,
written into the model config, and parity-checked at inference — you cannot tile a
model that was trained whole-frame (scale parity is a precondition). See the
[tiling guide](tiling.md) for details. Tiling is the primary use case: it isolates
foreground quality (thin roots, tiny animals) from the grouping bottleneck that
caps bottom-up instance segmentation.

## Inference

`sleap-nn predict` auto-detects a semantic model directory (a lone
`semantic_segmentation` head) and emits **one instance-less
`sio.PredictedSegmentationMask` per frame** on `LabeledFrame.masks`
(`mask.instance is None`; persisted in `.slp` as `instance = -1`).

```bash
sleap-nn predict --data_path video.mp4 --model_paths models/semantic_run \
    --fg_threshold 0.5 --min_mask_area 3000
```

Applicable knobs: `--fg_threshold` (foreground probability binarization),
`--min_mask_area` (drop a near-empty whole-frame mask, in original-image pixels),
`--mask_output` (`mask` / `polygon` / `both`), `--full_res_masks`, and
`--mask_cleanup`. The instance-grouping knobs (`--center_nms_kernel`,
`--distance_gate_alpha`, `--merge_*`, `--peak_threshold`) do not apply — there are
no instances to group.

The predicted foreground is a single **semantic** mask and may be spatially
disconnected (multiple blobs); it is intentionally **not** connected-component
cleaned.

## Evaluation

Semantic evaluation is **matching-free**: each frame's masks are unioned into one
foreground and scored directly against the ground-truth foreground with IoU,
centerline Dice (clDice), and boundary IoU — no Hungarian matching, no IoU
threshold. Post-training eval runs automatically; to run it standalone:

```bash
sleap-nn eval --ground_truth_path gt.slp --predicted_path preds.slp \
    --match_method semantic
```

Frames whose ground-truth foreground is empty are skipped (there is no foreground
to score).

## When to use which segmentation model

- **`semantic_segmentation`** — you want region / figure-ground masks and do not
  need to separate individuals, or you want to measure foreground quality free of
  grouping.
- **`bottomup_segmentation`** — you want per-instance masks in a single detector-free
  pass, and instances are separable by center + offset grouping.
- **`centered_instance_segmentation`** — you want per-instance masks and already use
  (or can train) a centroid model; each crop yields exactly one clean mask.
