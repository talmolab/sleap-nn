# Top-down (crop-centered) instance segmentation

Predict a **binary foreground mask per instance** using the top-down recipe: a
`centroid` model detects each animal, a crop is taken around it, and a
segmentation model predicts the mask of the **centered** instance in that crop.
This is the segmentation analog of top-down pose (`centroid` + `centered_instance`)
and sidesteps the bottom-up grouping / over-segmentation problem entirely — each
crop yields exactly one clean mask.

Use it when you want per-instance masks and already use (or can train) a centroid
model; prefer **bottom-up** segmentation (`bottomup_segmentation`) when you want a
single-pass, detector-free model.

Model type / head-config key: **`centered_instance_segmentation`**. Sample config:
[`config_topdown_centered_instance_segmentation_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_centered_instance_segmentation_unet.yaml).

## Training data: per-instance masks

The training labels must carry one `sio.SegmentationMask` per instance on
`LabeledFrame.masks`, **linked to its instance** via `mask.instance` (the data
loader associates each crop with the centered instance's mask through that link,
falling back to bounding-box IoU if it is absent). Tracks are preserved.

If you only have poses, you can synthesize plausible masks by rasterizing each
instance's skeleton (thick edges + node disks + a dilation) into a boolean array
and attaching it with the link set:

```python
import sleap_io as sio

labels = sio.load_slp("poses.pkg.slp")
for lf in labels:
    masks = []
    for inst in lf.instances:
        blob = rasterize_pose(inst)               # bool (H, W) — your rasterizer
        masks.append(
            sio.UserSegmentationMask.from_numpy(
                blob, instance=inst, track=getattr(inst, "track", None)
            )
        )
    lf.masks = masks
labels.save("poses_masked.pkg.slp", embed=True)   # embed to keep it self-contained
```

### Training notes

- **Head:** a single `SegmentationHead` (BCE+Dice, logits). The crop is centered
  on `anchor_part` (a node name in the `segmentation` head config) — set it to
  match your centroid model's anchor; `null` centers on the mean of each
  instance's visible nodes.
- **Supervision:** only the centered instance is foreground; other instances'
  pixels inside the crop are background (no ignore region in v1).
- **Augmentation:** intensity augmentation is supported, and **geometric
  augmentation (rotation/scale/translate/flip) co-transforms the centered-instance
  mask** with the same affine matrix as the image+keypoints (nearest-neighbor, then
  re-binarized) on the oversized `sqrt(2)` crop, which gives rotation headroom
  before the re-crop. Erase/mixup stay image-only. Train at `scale: 1.0` with crop
  dims divisible by `max_stride` (the mask resize/pad mirror the image but are not
  sub-pixel pad-aware).
- **Metrics:** `val/fg_iou` (per-crop foreground IoU; each crop is one instance, so
  this is a per-instance mask-quality metric) is logged each epoch. With
  `trainer_config.eval.enabled: true`, a `SegmentationEvaluationCallback` also logs
  instance-level mask metrics (`eval/val/mask_mean_iou`, `mask_precision`,
  `mask_recall`, `mask_f1`). Training viz overlays the GT mask on the predicted
  foreground each epoch (`viz/.../gt_mask`).

```bash
sleap-nn train --config-name config_topdown_centered_instance_segmentation_unet \
  data_config.train_labels_path=[train.pkg.slp] data_config.val_labels_path=[val.pkg.slp]
```

## Inference

Top-down seg needs **two model dirs** — a `centroid` model and the
`centered_instance_segmentation` model — composed via repeated `--model_paths`:

```bash
sleap-nn predict video.mp4 \
  --model_paths centroid_model_dir \
  --model_paths centered_instance_segmentation_model_dir \
  --fg_threshold 0.5 -o predictions.slp
```

Each predicted instance's mask is placed back at its full-frame location and
written as a `sio.PredictedSegmentationMask` on `frame.masks` (RLE-backed, with
the crop `scale`/`offset` baked in). Top-down segmentation honors the
`--fg_threshold`, `--mask_output {mask,polygon,both}`, and `--polygon_epsilon`
post-processing flags, and adds no new flags. The remaining segmentation flags
— `--min_mask_area`, `--mask_cleanup`/`--mask_cleanup_radius`, and
`--full_res_masks` — are **bottom-up-only** (each crop already yields one clean
mask, so they are no-ops for top-down). Rendering needs no extra wiring:

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
for m in labels[0].masks:
    assert isinstance(m, sio.PredictedSegmentationMask)
sio.render_image(labels[0], save_path="overlay.png")   # masks auto-overlay
```

### GT-centroid fallback (no centroid model)

Given **only** the segmentation model dir, inference crops the **ground-truth**
instances from the source `.slp` (a GT-centroid fallback) and predicts a mask per
GT instance. This is what post-training evaluation uses, and is handy for scoring
the segmentation model in isolation:

```bash
sleap-nn predict labeled.pkg.slp --model_paths centered_instance_segmentation_model_dir
```

## Evaluation

Post-training evaluation runs automatically (`trainer_config.eval.enabled: true`)
using mask-IoU matching, equivalent to:

```bash
sleap-nn eval --ground_truth_path gt.slp --predicted_path pred.slp --match_method mask
```

`match_method="mask"` matches predicted vs GT masks by IoU (Hungarian) and reports
detection precision/recall/F1, mean mask IoU, panoptic quality, boundary IoU, and
COCO-style mask AP/AR. `match_threshold` is the minimum IoU in `(0, 1]`.

> **Note:** frame pairing matches predicted and GT frames by video identity. If
> your GT `.pkg.slp` was *derived* by re-embedding images from another file, its
> embedded videos point back at the original source, so a prediction's video
> provenance may not match the derived file and pairing is skipped (logged, not
> fatal). Evaluate against a `.slp` whose videos resolve consistently (a normal
> single-source `.pkg.slp` or on-disk videos).

## See also

- [Bottom-up segmentation](inference-guide.md) — single-pass, detector-free masks.
- [Centroid-only inference](centroid-only-inference.md) — the centroid stage standalone.
