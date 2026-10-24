# Fast top-down data path (`crop_pipeline: warp`)

`preprocessing.crop_pipeline: warp` is a lean training data path for box-scaled
centered-instance models (`crop_mode: bbox`). It is built for large multi-dataset
training, such as human pose on COCO + AI Challenger + MPII, where the stock data
path is the bottleneck. It is off by default (`crop_pipeline: legacy`).

Per sample it does one image read and **one** `cv2.warpAffine` from the full frame
straight to the crop, with the box-to-crop mapping and the rotation / scale / shift /
flip augmentation folded into that single affine. Samples are uint8 crops plus
keypoints, and the confidence maps can be rendered on the GPU in the training step.

```yaml
data_config:
  bbox_source: bboxes            # the person boxes linked to each instance
  unlabeled_points: ignore       # joints with no location get zero loss (ViTPose)
  preprocessing:
    ensure_rgb: true
    crop_mode: bbox
    crop_bbox_aspect: square     # one scale for both axes, box fit to the crop aspect
    crop_bbox_padding: 0.125     # x1.25 on the longest side
    crop_size: [256, 192]        # [height, width]
    crop_pipeline: warp
    gpu_confmaps: true
    crop_bbox_categories: [person]
    scale: 1.0
  augmentation_config:
    geometric:
      rotation_min: -40.0
      rotation_max: 40.0
      rotation_p: 0.6
      scale_min: 0.65
      scale_max: 1.35
      scale_p: 1.0
      flip_p: 0.5
trainer_config:
  train_data_loader:
    batch_size: 128
    num_workers: 20
```

---

## What it changes

| | `legacy` | `warp` |
|---|---|---|
| Frame | decoded, size-matched to `max_height`/`max_width` (resize + pad of the whole frame) | read at native size; frames of any size mix freely |
| Crop | oversized pre-crop, augmented, re-cropped | one affine warp from the full frame |
| Sample | float crop + float32 confidence maps | uint8 crop + keypoints (+ maps unless `gpu_confmaps`) |
| Skeletons | every file must have the same skeleton | nodes matched **by name** across files |
| Crop shape | square | square or `[height, width]` |

Without augmentation a warp crop is the inference crop: the affine is exactly the
box-scaled window of `bbox_crop_windows` that the top-down inference layer cuts with
`crop_windows` (they differ by under one grey level on average, from bilinear
rounding), and keypoints round-trip through the crop to ~1e-5 px.

Inference needs nothing new: the saved config carries `crop_size: [h, w]` and no
size matcher, and `sleap-nn predict` cuts the same windows from frames of any size.

## Mixing datasets

Files with different skeletons train one model. The model's nodes are
`part_names` (default: every training file's nodes, first seen first), and its
skeleton gets every edge and left/right symmetry the files define on those nodes.
A node a file does not have, or does not label (`provenance["labeled_nodes"]`, see
[Partially Labeled Skeletons](partial-labels.md)), gets zero loss for that file's
instances.

`trainer_config.balanced_sampler` fixes how many samples of each training file go in
every batch, so a small dataset is not drowned by a large one:

```yaml
trainer_config:
  train_data_loader:
    batch_size: 128
  balanced_sampler:
    counts: [48, 80]        # per training file, in train_labels_path order
    # weights: [0.4, 0.6]   # or relative shares, rounded to batch_size
  train_steps_per_epoch: 2000
```

Each file is drawn by reshuffled passes over its instances, so a small file is
oversampled. An epoch is `train_steps_per_epoch` batches summed over GPUs (default:
one pass over the training instances). Under DDP each rank draws its own stream
(seeded by `trainer_config.seed` and the rank) rather than a shard. The stream is a
pure function of the seed, the rank and the batch index, and the number of batches
delivered is saved in the checkpoint, so `resume_ckpt_path` continues the same
stream where it stopped.

## Crop boxes and filters

- `crop_bbox_categories` keeps only linked user boxes of those categories as crop
  boxes (e.g. `[person]`). Other linked boxes, such as MPII's *head* boxes, fall back
  to the instance's keypoint box.
- `data_config.instance_filter` drops training instances (validation keeps every
  instance):

```yaml
data_config:
  instance_filter:
    min_labeled_nodes: 6         # at least 6 of count_nodes labeled
    count_nodes: [left_shoulder, right_shoulder, left_elbow, right_elbow,
                  left_wrist, right_wrist, left_hip, right_hip, left_knee,
                  right_knee, left_ankle, right_ankle]
    min_bbox_size: 0.0           # sqrt(box area) in frame pixels
```

ViTPose trains on every person with at least one keypoint: `min_labeled_nodes: 1`.

## Augmentation

The warp path reads the usual `augmentation_config` sections:

- Rotation and scale act about the crop center; `scale > 1` zooms in, as in the stock
  path.
- `translate_width` / `translate_height` shift the window by that fraction of the
  **padded box** (mmpose's `RandomBBoxTransform` shift), not of the crop.
- `flip_p` mirrors the crop and swaps left/right nodes.
- Contrast pivots on mid-grey (127.5) and brightness scales, as in the stock path;
  Gaussian noise is applied on the uint8 crop.
- `erase_p`, `mixup_p` and `uniform_noise_p` are not implemented here and are
  ignored with a warning.
- `crop_bbox_aspect: letterbox` is not implemented; the warp path crops as `square`.

## Throughput

Every batch has a fixed per-sample cost of one JPEG decode plus one warp (~2 ms of
CPU). See the PR for measured loader rates; feeding a ViT-S on 4 GPUs takes
50-70 workers in total.

## Limits

- `centered_instance` only (other model types raise), `crop_mode: bbox`, and
  `scale: 1.0` (the crop is cut at `crop_size` directly).
- `crop_size` must be set or auto-sized, and divisible by the backbone's `max_stride`.
- Boxes come from linked user boxes or keypoints; `bbox_source: masks` raises.
- `data_pipeline_fw` caching does not apply: the warp path reads image files itself
  (frames of non-image videos, e.g. `.pkg.slp` or `.mp4`, are read through the labels).
