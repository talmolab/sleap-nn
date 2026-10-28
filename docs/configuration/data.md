# Data Config

Configure data loading, preprocessing, and augmentation.

---

## Data Paths

```yaml
data_config:
  train_labels_path:
    - train.slp
    - more_train.slp      # Multiple files supported
  val_labels_path:
    - val.slp             # Optional
  validation_fraction: 0.1  # Used if val_labels_path not set
  test_file_path: test.slp  # For post-training evaluation
```

| Option | Description | Default |
|--------|-------------|---------|
| `train_labels_path` | Training data files | Required |
| `val_labels_path` | Validation data files | `null` |
| `validation_fraction` | Fraction for auto-split | `0.1` |
| `test_file_path` | Test data for evaluation | `null` |
| `user_instances_only` | Only use user-labeled instances | `true` |
| `use_negative_frames` | Include user-confirmed negative frames (see [guide](../guides/negative-frames.md)) | `false` |
| `negative_loss_weight` | Loss weight multiplier for negative samples | `1.0` |
| `node_loss_mask` | Per-node loss weights: unlabeled points (sleap-io 0.10 `labeled=False`, e.g. COCO `v=0`) and nodes a file does not label (`provenance["labeled_nodes"]`) get zero loss (see [guide](../guides/partial-labels.md)) | `true` |
| `unlabeled_points` | Points with no location (NaN, or outside the image/crop after augmentation): `absent` trains them as absent, `ignore` gives them zero loss | `absent` |
| `supervise_occluded` | Train occluded points that have estimated coordinates (labeled, not visible; COCO `v=1`) at those coordinates; `false` gives them zero loss | `true` |

---

## Data Pipeline

```yaml
data_config:
  data_pipeline_fw: torch_dataset  # or cache options
  cache_img_path: /path/to/cache   # For disk caching
```

| Option | Description | When to Use |
|--------|-------------|-------------|
| `torch_dataset` | Load on demand | Default, works everywhere |
| `torch_dataset_cache_img_memory` | Cache in RAM | Faster, needs enough RAM |
| `torch_dataset_cache_img_disk` | Cache to disk | Large datasets, multi-GPU |

!!! tip "With caching, you can use `num_workers > 0`"

---

## Preprocessing

```yaml
data_config:
  preprocessing:
    ensure_rgb: false          # Force 3 channels
    ensure_grayscale: false    # Force 1 channel
    scale: 1.0                 # Resize factor (0.5 = half size)
    max_height: null           # Pad to this height
    max_width: null            # Pad to this width
    crop_size: null            # For centered-instance models
    min_crop_size: 100         # Minimum auto crop size
    crop_padding: null         # Extra padding for crops
```

### Common Patterns

=== "Downscale for speed"
    ```yaml
    preprocessing:
      scale: 0.5  # Half resolution, 4x faster
    ```

=== "Fixed dimensions"
    ```yaml
    preprocessing:
      max_height: 512
      max_width: 512
    ```

=== "Centered-instance model"
    ```yaml
    preprocessing:
      crop_size: 256  # Fixed crop size
      # or
      crop_size: null  # Auto-compute from data
      min_crop_size: 100
      crop_padding: 20
    ```

---

## Augmentation

```yaml
data_config:
  use_augmentations_train: true
  augmentation_config:
    intensity:
      # ... intensity settings
    geometric:
      # ... geometric settings
```

### Intensity Augmentation

```yaml
augmentation_config:
  intensity:
    # Noise
    uniform_noise_p: 0.0
    gaussian_noise_p: 0.0
    gaussian_noise_mean: 0.0
    gaussian_noise_std: 0.02

    # Contrast
    contrast_min: 0.9
    contrast_max: 1.1
    contrast_p: 0.0

    # Brightness
    brightness_min: 0.9
    brightness_max: 1.1
    brightness_p: 0.0
```

### Geometric Augmentation

```yaml
augmentation_config:
  geometric:
    # Rotation (degrees)
    rotation_min: -15.0
    rotation_max: 15.0

    # Scale
    scale_min: 0.9
    scale_max: 1.1

    # Translation (fraction of image size)
    translate_width: 0.0
    translate_height: 0.0

    # Combined probability (used when individual *_p values are null)
    affine_p: 0.0

    # Independent probabilities (override affine_p when set)
    rotation_p: 1.0   # Always apply rotation by default
    scale_p: 1.0      # Always apply scaling by default
    translate_p: null # Uses affine_p if null

    # Random erase
    erase_p: 0.0
    erase_scale_min: 0.0001
    erase_scale_max: 0.01

    # Mixup
    mixup_p: 0.0
    mixup_lambda_min: 0.01
    mixup_lambda_max: 0.05

    # Flip (mirror image + keypoints left/right, swapping symmetric node pairs)
    flip_p: 0.0  # 0 disables; >0 enables left/right flip (uses skeleton symmetries)
```

> **Flip & skeleton symmetries:** when `flip_p > 0`, the image and keypoints are
> mirrored left/right and left/right symmetric body parts (e.g. `left_paw` ↔
> `right_paw`) are swapped so labels stay correct. Symmetry pairs are read from the
> skeleton. If the skeleton has **no** symmetries, flipping is only correct when the
> animal is truly left/right symmetric in labeling — a warning is logged otherwise.

---

## Examples

### No Augmentation

```yaml
data_config:
  use_augmentations_train: false
```

### Geometric Only (using defaults)

```yaml
data_config:
  use_augmentations_train: true
  # Default augmentation_config applies rotation ±15° and scale 0.9-1.1 with p=1.0
```

### Geometric Only (explicit)

```yaml
data_config:
  use_augmentations_train: true
  augmentation_config:
    intensity: null
    geometric:
      rotation_min: -30.0
      rotation_max: 30.0
      rotation_p: 0.5        # 50% chance of rotation
      scale_min: 0.8
      scale_max: 1.2
      scale_p: 0.5           # 50% chance of scaling
```

### Full Augmentation

```yaml
data_config:
  use_augmentations_train: true
  augmentation_config:
    intensity:
      contrast_p: 0.3
      brightness_p: 0.3
      gaussian_noise_p: 0.2
      gaussian_noise_std: 0.1
    geometric:
      rotation_min: -30.0
      rotation_max: 30.0
      scale_min: 0.8
      scale_max: 1.2
      affine_p: 0.7
```

---

## Full Reference

### DataConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `train_labels_path` | list | `null` | List of paths to training `.slp` files |
| `val_labels_path` | list | `null` | List of paths to validation `.slp` files |
| `validation_fraction` | float | `0.1` | Fraction of training data for auto-split validation |
| `use_same_data_for_val` | bool | `false` | Use same data for both training and validation (useful for intentional overfitting) |
| `test_file_path` | str/list | `null` | Path(s) to test data for post-training evaluation |
| `provider` | str | `LabelsReader` | Data provider class (only `LabelsReader` supported) |
| `user_instances_only` | bool | `true` | Only use user-labeled instances (not predicted) |
| `data_pipeline_fw` | str | `torch_dataset` | Data loading framework: `torch_dataset`, `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk` |
| `cache_img_path` | str | `null` | Path for disk caching (used with `torch_dataset_cache_img_disk`) |
| `use_existing_imgs` | bool | `false` | Use existing cached images instead of regenerating |
| `delete_cache_imgs_after_training` | bool | `true` | Delete cached images after training completes |
| `parallel_caching` | bool | `true` | Use parallel processing for caching (faster for large datasets) |
| `cache_workers` | int | `0` | Number of workers for parallel caching (0 = auto: min(4, cpu_count)) |
| `use_augmentations_train` | bool | `true` | Apply augmentations during training |
| `bbox_source` | str | `auto` | Ground-truth boxes for `bbox_*` models: `auto` (the file's user boxes if it has any, else the tight box of each user instance's visible points, else of each mask), `bboxes`, `instances` or `masks`. Ignored by other model types |
| `bbox_padding` | float | `0.0` | Pixels added on every side of boxes derived from instances or masks (keypoints sit inside the outline; the config generator uses 5% of the median animal size for `bbox_detr`). Use the same value in a box-scaled centered-instance model |
| `instance_filter` | InstanceFilterConfig | `null` | `crop_pipeline: warp` only: train only on instances with at least `min_labeled_nodes` of `count_nodes` (default: every node) labeled and a box of `sqrt(area) >= min_bbox_size` px. Validation keeps every instance |

### PreprocessingConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ensure_rgb` | bool | `false` | Convert images to 3 channels (RGB). Single-channel images are replicated |
| `ensure_grayscale` | bool | `false` | Convert images to 1 channel (grayscale) |
| `max_height` | int | `null` | Pad images to this height |
| `max_width` | int | `null` | Pad images to this width |
| `scale` | float | `1.0` | Resize factor (e.g., 0.5 = half size). Must be `1.0` for `bbox_detr`, whose input size is `max_height` x `max_width` |
| `resize_mode` | str | `letterbox` | How a frame is fitted to `max_height` x `max_width`: `letterbox` (one scale, padded bottom/right) or `squash` (each axis scaled on its own; RF-DETR's resize, needed for its imported weights). `squash` needs both sizes and is `bbox_detr` only |
| `crop_size` | int or `[h, w]` | `null` | Crop size for centered-instance models. If `null`, auto-computed from data. `[height, width]` (e.g. `[256, 192]` for people) needs `crop_mode: bbox` and `crop_pipeline: warp` |
| `crop_pipeline` | str | `legacy` | How `centered_instance` training crops are cut: `legacy`, or `warp` (one affine warp from the full frame, uint8 samples, no size matcher; needs `crop_mode: bbox`). See [Fast top-down data path](../guides/warp-data-path.md) |
| `gpu_confmaps` | bool | `false` | `crop_pipeline: warp`: render the confidence maps on the GPU in the training step instead of in the workers |
| `crop_bbox_categories` | list | `null` | `crop_pipeline: warp`: only linked user boxes of these categories (e.g. `[person]`) are crop boxes; others (e.g. MPII head boxes) fall back to the keypoint box |
| `crop_bbox_source` | str | `gt` | `crop_pipeline: warp`: `gt` crops from the instance's user (or keypoint) box; `detector` crops from the `PredictedBoundingBox` linked to the instance (a detector box matched to the GT), so training crops look like inference crops. See [Detector-box crops](../guides/warp-data-path.md#detector-box-crops) |
| `crop_bbox_detector` | str | `null` | `crop_bbox_source: detector`: only predicted boxes whose `source` is this string count |
| `crop_bbox_min_score` | float | `0.0` | `crop_bbox_source: detector`: only predicted boxes scoring at least this count |
| `crop_bbox_fallback` | str | `gt` | `crop_bbox_source: detector`: an instance without a usable detector box is cropped from its `gt` box, or left out (`drop`) |
| `crop_bbox_jitter` | float | `0.0` | `crop_pipeline: warp`, training only: each box edge moves by Gaussian noise with this std, as a fraction of the box width / height (detector-like box noise). Off at `0` |
| `min_crop_size` | int | `100` | Minimum crop size when auto-computing |
| `crop_padding` | int | `null` | Extra padding around crops. If `null`, auto-computed from augmentation settings |

### IntensityConfig (augmentation_config.intensity)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `uniform_noise_min` | float | `0.0` | Minimum uniform noise value (0-1 scale, multiplied by 255 internally) |
| `uniform_noise_max` | float | `0.04` | Maximum uniform noise value (0-1 scale, multiplied by 255 internally) |
| `uniform_noise_p` | float | `0.0` | Probability of applying uniform noise |
| `gaussian_noise_mean` | float | `0.0` | Mean of Gaussian noise distribution (0-1 scale, multiplied by 255 internally) |
| `gaussian_noise_std` | float | `0.02` | Standard deviation of Gaussian noise (0-1 scale, multiplied by 255 internally) |
| `gaussian_noise_p` | float | `0.0` | Probability of applying Gaussian noise |
| `contrast_min` | float | `0.9` | Minimum contrast factor |
| `contrast_max` | float | `1.1` | Maximum contrast factor |
| `contrast_p` | float | `0.0` | Probability of applying contrast adjustment |
| `brightness_min` | float | `0.9` | Minimum brightness factor |
| `brightness_max` | float | `1.1` | Maximum brightness factor (max 2.0) |
| `brightness_p` | float | `0.0` | Probability of applying brightness adjustment |
| `hsv_p` | float | `0.0` | `crop_pipeline: warp` only: random HSV shift (mmdet `YOLOXHSVRandomAug`); `hsv_hue_delta` / `hsv_saturation_delta` / `hsv_value_delta` (`5` / `30` / `30`) bound the shifts |
| `blur_p` | float | `0.0` | `crop_pipeline: warp` only: box blur, odd kernel in `[blur_kernel_min, blur_kernel_max]` (`3`-`7`) |
| `median_blur_p` | float | `0.0` | `crop_pipeline: warp` only: median blur, kernel in `[median_blur_kernel_min, median_blur_kernel_max]` (`3`-`7`) |
| `gaussian_blur_p` | float | `0.0` | `crop_pipeline: warp` only: Gaussian blur, sigma in `[gaussian_blur_sigma_min, gaussian_blur_sigma_max]` (`0.1`-`2.0`) crop px |

### AugmentationStage2Config (augmentation_config.stage2)

`crop_pipeline: warp` only: a second augmentation stage, see [Stage-2 switch](../guides/warp-data-path.md#stage-2-switch).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `start_epoch` | int | `null` | First (0-based) epoch of stage 2 |
| `start_step` | int | `null` | First global step of stage 2 (used when `start_epoch` is `null`) |
| `geometric` | dict | `null` | Overrides merged over `augmentation_config.geometric` |
| `intensity` | dict | `null` | Overrides merged over `augmentation_config.intensity` |

### DetrAugmentationConfig (augmentation_config.detr)

RF-DETR's training geometry, `bbox_detr` only. Applied to training frames before
`intensity` / `geometric` (set `geometric: null` explicitly for keypoint-derived
boxes: an omitted section means the default rotation). Defaults are RF-DETR's COCO
values; scale `shortest_sides` / `crop_range` to your input size (the config
generator uses the input's short side and 2/3 to 1x of it).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `crop_p` | float | `0.5` | Probability of the resize + crop branch: resize the shortest side to one of `shortest_sides`, crop a random square with a side in `crop_range`, resize it to the input size. Otherwise the whole frame is resized. Boxes are clipped to the crop, dropped when no area is left |
| `shortest_sides` | list[int] | `[400, 500, 600]` | Pre-crop shortest sides |
| `crop_range` | list[int] | `[384, 600]` | Inclusive `[min, max]` crop side |
| `flip_p` | float | `0.5` | Horizontal flip probability |
| `multi_scale` | bool | `true` | Per-batch multi-scale: every training batch is resized to one size drawn per optimizer step (the same on every rank) from `max_height x max_width` +/- `multi_scale_steps` multiples of the backbone's `max_stride` |
| `multi_scale_steps` | int | `5` | Size steps up and down |

### GeometricConfig (augmentation_config.geometric)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rotation_min` | float | `-15.0` | Minimum rotation angle in degrees |
| `rotation_max` | float | `15.0` | Maximum rotation angle in degrees |
| `rotation_p` | float | `1.0` | Probability of rotation (independent). If `null`, uses `affine_p` |
| `scale_min` | float | `0.9` | Minimum scale factor |
| `scale_max` | float | `1.1` | Maximum scale factor |
| `scale_p` | float | `1.0` | Probability of scaling (independent). If `null`, uses `affine_p` |
| `translate_width` | float | `0.0` | Maximum horizontal translation as fraction of width (`crop_pipeline: warp`: of the padded box) |
| `translate_height` | float | `0.0` | Maximum vertical translation as fraction of height (`crop_pipeline: warp`: of the padded box) |
| `translate_p` | float | `null` | Probability of translation (independent). If `null`, uses `affine_p` |
| `affine_p` | float | `0.0` | Probability of bundled affine transform (rotation + scale + translate) |
| `erase_scale_min` | float | `0.0001` | Minimum erased area as proportion of image |
| `erase_scale_max` | float | `0.01` | Maximum erased area as proportion of image |
| `erase_ratio_min` | float | `1.0` | Minimum aspect ratio of erased area |
| `erase_ratio_max` | float | `1.0` | Maximum aspect ratio of erased area |
| `erase_p` | float | `0.0` | Probability of random erasing |
| `mixup_lambda_min` | float | `0.01` | Minimum mixup strength |
| `mixup_lambda_max` | float | `0.05` | Maximum mixup strength |
| `mixup_p` | float | `0.0` | Not implemented: a value > 0 warns (and raises with `crop_pipeline: warp`) |
| `flip_p` | float | `0.0` | Probability of mirroring image + keypoints left/right (swaps symmetric node pairs) |
| `affine_sampling` | str | `uniform` | `crop_pipeline: warp` only: `mmpose` draws rotation / scale / shift like mmpose's `RandomBBoxTransform` (truncated normal, `scale_*` = box scale, shift before rotation) |
| `half_body_p` | float | `0.0` | `crop_pipeline: warp` only: mmpose `RandomHalfBody` (with `half_body_min_total_keypoints` `9`, `half_body_min_upper_keypoints` `2`, `half_body_min_lower_keypoints` `3`, `half_body_padding` `1.5`, `half_body_upper_prob` `0.7`, `half_body_lower_nodes` `null` = names with hip/knee/ankle/heel/toe/foot) |
| `coarse_dropout_p` | float | `0.0` | `crop_pipeline: warp` only: black holes (`coarse_dropout_min_holes`/`_max_holes` `1`, sides `coarse_dropout_min_size`/`_max_size` `0.2`-`0.4` of the crop) |
