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

### Detector-box crops

Top-down inference crops each person from a detector box, but training crops come
from GT boxes by default, and the two differ in size, aspect and centering. To train
on the boxes inference sees, store person-detector boxes matched to the GT people in
the labels file and switch the crop box source:

```yaml
data_config:
  preprocessing:
    crop_bbox_source: detector   # default: gt
    crop_bbox_fallback: gt       # people with no matched detection: GT box (or drop)
    crop_bbox_min_score: 0.0     # ignore matched boxes scoring lower
    crop_bbox_detector: null     # only boxes with this `source` (null: any)
    crop_bbox_jitter: 0.0        # optional per-edge Gaussian noise, x box size
```

The labels file carries one `sio.PredictedBoundingBox` per matched detection, linked
to its instance (`box.instance = inst`), next to the user boxes. Unlinked predicted
boxes (unmatched detections) are ignored. With several boxes linked to one instance,
the highest-scoring one is used, and `crop_bbox_categories` applies to them too.
User boxes and the `gt` path never read predicted boxes, so adding them to a file does
not change `crop_bbox_source: gt` training. The log reports how many crops came from
detector boxes and how many fell back.

Matching follows Sárándi et al. (WACV 2023): per image, Hungarian matching on IoU
between the detections and the GT people, keeping pairs with IoU > 0.1. Matching
against every non-crowd GT person box, including people without keypoints, keeps a
detection on an unlabeled person from being assigned to a labeled neighbour. For
files without person boxes (MPII), the keypoint extent is the GT box. A minimal
sketch (keypoint-extent GT boxes only):

```python
import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

def attach_matched_boxes(lf, det_xyxy, det_scores, source, min_iou=0.1):
    insts = lf.user_instances
    gt = np.array([[*i.numpy()[np.isfinite(i.numpy()).all(1)].min(0),
                    *i.numpy()[np.isfinite(i.numpy()).all(1)].max(0)] for i in insts])
    lt = np.maximum(gt[:, None, :2], det_xyxy[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], det_xyxy[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area = lambda b: np.prod(b[:, 2:] - b[:, :2], -1)
    iou = inter / (area(gt)[:, None] + area(det_xyxy)[None] - inter)
    r, c = linear_sum_assignment(-iou)
    match = {j: insts[i] for i, j in zip(r, c) if iou[i, j] > min_iou}
    for j, (box, score) in enumerate(zip(det_xyxy, det_scores)):
        lf.bboxes.append(sio.PredictedBoundingBox(
            *map(float, box), instance=match.get(j), category="person",
            score=float(score), source=source))
```

`crop_bbox_jitter` perturbs each edge of the crop box independently (training only)
to mimic detector noise without detector boxes, or on top of them.

## Augmentation

The warp path reads the usual `augmentation_config` sections, plus keys that only it
implements (the stock path accepts them and warns once that it ignores them). All of
them are off by default.

**Geometry** is folded into the one warp:

- Rotation and scale act about the crop center; with the default
  `affine_sampling: uniform`, draws are uniform and `scale > 1` zooms in, as in the
  stock path.
- `translate_width` / `translate_height` shift the window by that fraction of the
  **padded box** (mmpose's `RandomBBoxTransform` shift), not of the crop.
- `flip_p` mirrors the crop and swaps left/right nodes.
- `affine_sampling: mmpose` follows mmpose's `RandomBBoxTransform`: truncated-normal
  draws (±1 sigma = the range ends) around the middle of each range, `scale_*` is the
  scale of the crop **box** (content scale = 1 / box scale), and the shift moves the
  box before it is rotated. mmpose config numbers can then be used as they are.
- `half_body_p` is mmpose's `RandomHalfBody`: an instance with at least
  `half_body_min_total_keypoints` visible nodes is cropped to its upper or lower body
  (`half_body_upper_prob` prefers the upper half). The box is centered on the chosen
  nodes, `half_body_padding` times their extent. Lower-body nodes are those named
  hip / knee / ankle / heel / toe / foot unless `half_body_lower_nodes` lists them.
  It replaces the crop window, so it costs nothing extra.

**Photometric** ops run on the uint8 crop, in this order:

| Op | Keys | Notes |
|---|---|---|
| Contrast, brightness | `contrast_*`, `brightness_*` | contrast pivots on 127.5, as in the stock path |
| HSV shift | `hsv_p`, `hsv_{hue,saturation,value}_delta` | pixel-identical to mmdet's `YOLOXHSVRandomAug` |
| Box blur | `blur_p`, `blur_kernel_{min,max}` | albumentations `Blur` |
| Median blur | `median_blur_p`, `median_blur_kernel_{min,max}` | albumentations `MedianBlur` |
| Gaussian blur | `gaussian_blur_p`, `gaussian_blur_sigma_{min,max}` | |
| Uniform noise | `uniform_noise_*` | as in the stock path |
| Gaussian noise | `gaussian_noise_*` | as in the stock path |
| Coarse dropout | `coarse_dropout_p`, `coarse_dropout_{min,max}_holes`, `coarse_dropout_{min,max}_size` | black holes, sides as fractions of the crop (albumentations `CoarseDropout`) |
| Random erase | `erase_*` | random-colour box, as in the stock path |

Keypoints inside dropout holes and erased boxes keep their targets, as in mmpose: the
model learns to infer occluded joints. Half-body crops push the other half's joints
out of the crop; with `unlabeled_points: ignore` those get zero loss instead of
training as absent.

`mixup_p > 0` raises in the warp path (the stock path, which never implemented it,
now warns instead of silently ignoring it). `crop_bbox_aspect: letterbox` is not
implemented here; the warp path crops as `square`.

### Recipes

From mmpose's main-branch configs (`td-hm_ViTPose-base_8xb64-210e_coco-256x192.py`
and `rtmpose-m_8xb256-420e_coco-256x192.py`):

=== "ViTPose"
    ```yaml
    augmentation_config:
      geometric:
        affine_sampling: mmpose
        flip_p: 0.5
        half_body_p: 0.3
        translate_width: 0.16
        translate_height: 0.16
        translate_p: 0.3
        scale_min: 0.5            # box scale
        scale_max: 1.5
        scale_p: 1.0
        rotation_min: -80.0
        rotation_max: 80.0
        rotation_p: 0.6
    ```

=== "RTMPose (two stages)"
    ```yaml
    augmentation_config:
      geometric:
        affine_sampling: mmpose
        flip_p: 0.5
        half_body_p: 0.3
        translate_width: 0.16
        translate_height: 0.16
        translate_p: 0.3
        scale_min: 0.6
        scale_max: 1.4
        scale_p: 1.0
        rotation_min: -80.0
        rotation_max: 80.0
        rotation_p: 0.6
        coarse_dropout_p: 1.0
        coarse_dropout_min_size: 0.2
        coarse_dropout_max_size: 0.4
      intensity:
        hsv_p: 1.0
        blur_p: 0.1
        median_blur_p: 0.1
      stage2:                     # the last 30 of 420 epochs
        start_epoch: 390
        geometric:
          translate_p: 0.0
          scale_min: 0.75
          scale_max: 1.25
          rotation_min: -60.0
          rotation_max: 60.0
          coarse_dropout_p: 0.5
    ```

The original ViTPose repository used ±40° rotation and 0.65–1.35 scale (uniform),
narrower than mmpose's current ViTPose config.

### Stage-2 switch

`augmentation_config.stage2: {start_epoch | start_step, geometric, intensity}` trains
the last part of a run with different augmentation (RTMPose's `PipelineSwitchHook`).
Its `geometric` / `intensity` entries are overrides merged over the stage-1 sections.
`AugmentationStageCallback` switches the training dataset through a shared-memory
flag that persistent workers read per sample, so it works with forked and spawned
workers without restarting them, and a resumed run lands in the right stage. Batches
already prefetched at the switch (at most `num_workers x prefetch_factor`) still use
stage 1. `stage2` is refused for any other pipeline.

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
