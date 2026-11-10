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
    max_height: null           # Resize to fit, then pad to this height (null = largest training frame)
    max_width: null            # Resize to fit, then pad to this width (null = largest training frame)
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

=== "Centered-instance model (fixed crop)"
    ```yaml
    preprocessing:
      crop_size: 256  # Fixed crop size
    ```

=== "Centered-instance model (auto crop)"
    ```yaml
    preprocessing:
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

    # Independent probabilities. If ANY of these is set, each transform runs
    # on its own probability, a null one never runs, and affine_p is ignored.
    rotation_p: 1.0   # Always apply rotation by default
    scale_p: 1.0      # Always apply scaling by default
    translate_p: null # null here = no translation (rotation_p/scale_p are set)

    # Bundled probability: rotation + scale + translation together. Used only
    # when rotation_p, scale_p and translate_p are ALL null.
    affine_p: 0.0

    # Random erase
    erase_p: 0.0
    erase_scale_min: 0.0001
    erase_scale_max: 0.01

    # Mixup: accepted for config compatibility but NOT implemented (ignored)
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
      rotation_p: 0.7
      scale_min: 0.8
      scale_max: 1.2
      scale_p: 0.7
```

!!! note "`affine_p` vs the per-transform probabilities"
    `rotation_p` and `scale_p` default to `1.0`, so `affine_p` is ignored unless
    you set `rotation_p`, `scale_p` **and** `translate_p` all to `null`. To apply
    rotation, scale and translation together with one probability, use
    `affine_p: 0.7` with `rotation_p: null`, `scale_p: null`, `translate_p: null`.

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
| `bbox_padding` | float | `0.0` | Pixels added on every side of boxes derived from instances or masks |
| `use_negative_frames` | bool | `false` | Include user-confirmed negative frames (`labels.negative_frames`) as all-zero targets (see [guide](../guides/negative-frames.md)) |
| `negative_loss_weight` | float | `1.0` | Loss weight of negative samples relative to positives (must be > 0; only used with `use_negative_frames`) |
| `skeletons` | list | `null` | Set automatically: the skeleton(s) read from the training labels and saved into `training_config.yaml` |
| `centroids_from_masks` | str | `null` | Derive `UserCentroid`s from segmentation masks at load time so a mask-only dataset can train a centroid model: `center_of_mass` or `bbox_center`. `null` = off. Frames that already have user centroids are left alone |
| `identity` | IdentityConfig | `null` | `embedding` only: what the track labels mean, so the objective can validate its positives/negatives. See [IdentityConfig](#identityconfig-identity) |
| `split` | SplitConfig | `null` | Group-aware train/val split (by frame, video or identity) used instead of the random `validation_fraction` split when `val_labels_path` is not set. See [SplitConfig](#splitconfig-split) |

### IdentityConfig (`identity`)

Used by the `embedding` model type (see the [embedding tracking guide](../guides/embedding-tracking.md)).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tracks_are_proofread` | bool | `false` | Tracks are swap-free within a video. Gates `positives.scope: tracklet` (warns if `false`) |
| `track_names_are_global` | bool | `false` | The same track name is the same animal across videos, so a track name can stand in for a `sio.Identity` under `positives.scope: global_id` |
| `detections_deduplicated` | bool | `true` | No duplicate/over-segmented detections per frame. Gates `same_frame` negatives (warns if `false`) |

### SplitConfig (`split`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `split_by` | str | `frame` | Group key: `frame` (stratified-random over labeled frames, identity-balanced), `video` (hold out whole videos) or `identity` (hold out whole track names) |
| `n_folds` | int | `5` | Number of folds; the validation partition is `1 / n_folds` of the data |
| `fold` | int | `0` | Which fold (0-based) to hold out as validation |
| `seed` | int | `0` | Random seed of the splitter |

### PreprocessingConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ensure_rgb` | bool | `false` | Convert images to 3 channels (RGB). Single-channel images are replicated |
| `ensure_grayscale` | bool | `false` | Convert images to 1 channel (grayscale) |
| `max_height` | int | `null` | Resize images to fit (one scale for both axes), then pad to this height. `null` = set at training to the largest frame height across the training labels and saved into the config |
| `max_width` | int | `null` | Resize images to fit (one scale for both axes), then pad to this width. `null` = set at training to the largest frame width across the training labels and saved into the config |
| `scale` | float | `1.0` | Resize factor (e.g., 0.5 = half size) |
| `crop_size` | int | `null` | Crop size for centered-instance models. If `null`, auto-computed from data |
| `min_crop_size` | int | `100` | Minimum crop size when auto-computing |
| `crop_padding` | int | `null` | Extra padding around crops. If `null`, auto-computed from augmentation settings |
| `crop_mode` | str | `fixed` | How a centered-instance crop is cut (`centered_instance` / `multi_class_topdown`): `fixed` (a `crop_size` window at the centroid) or `bbox` (the instance's box, resized to `crop_size`; inference needs a `bbox_*` detector or GT boxes). See the [box-seeded top-down guide](../guides/bbox-topdown.md) |
| `crop_bbox_aspect` | str | `square` | `crop_mode: bbox` only: `square` (expand to a square on the longest side), `letterbox` (same, pixels outside the box zeroed) or `stretch` (resize each axis separately) |
| `crop_bbox_padding` | float | `0.1` | `crop_mode: bbox` only: margin on every side of the box, as a fraction of its longest side |
| `crop_bbox_min_size` | float | `null` | `crop_mode: bbox` only: smallest crop window side (frame pixels). `null` = set at training to a quarter of the median training window and saved into the config |
| `tiling` | TilingConfig | disabled | Sliding-window tiling for high-resolution frames; inert unless `tiling.enabled: true`. See the [tiling guide](../guides/tiling.md) for its fields |
| `burn_in` | bool | `false` | `embedding` only: burn the instance mask into the crop (background set to `background_fill`) |
| `background_fill` | str | `black` | `embedding` with `burn_in` only: `black`, `grey`, `mean` or `noise` |
| `crop_centering` | str | `auto` | `embedding` (mask mode) only: crop center: `auto` (mask center of mass), `mask_com` or `bbox` (mask box midpoint) |

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

### GeometricConfig (augmentation_config.geometric)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rotation_min` | float | `-15.0` | Minimum rotation angle in degrees |
| `rotation_max` | float | `15.0` | Maximum rotation angle in degrees |
| `rotation_p` | float | `1.0` | Probability of rotation. If any of `rotation_p` / `scale_p` / `translate_p` is set, a `null` one means the transform never runs |
| `scale_min` | float | `0.9` | Minimum scale factor |
| `scale_max` | float | `1.1` | Maximum scale factor |
| `scale_p` | float | `1.0` | Probability of scaling (same rule as `rotation_p`) |
| `translate_width` | float | `0.0` | Maximum horizontal translation as fraction of width |
| `translate_height` | float | `0.0` | Maximum vertical translation as fraction of height |
| `translate_p` | float | `null` | Probability of translation (same rule; with the defaults, `null` means no translation) |
| `affine_p` | float | `0.0` | Probability of a bundled rotation + scale + translation. Used only when `rotation_p`, `scale_p` and `translate_p` are all `null` |
| `erase_scale_min` | float | `0.0001` | Minimum erased area as proportion of image |
| `erase_scale_max` | float | `0.01` | Maximum erased area as proportion of image |
| `erase_ratio_min` | float | `1.0` | Minimum aspect ratio of erased area |
| `erase_ratio_max` | float | `1.0` | Maximum aspect ratio of erased area |
| `erase_p` | float | `0.0` | Probability of random erasing |
| `mixup_lambda_min` | float | `0.01` | Minimum mixup strength. Mixup is not implemented: accepted and validated, but ignored |
| `mixup_lambda_max` | float | `0.05` | Maximum mixup strength (ignored; see `mixup_lambda_min`) |
| `mixup_p` | float | `0.0` | Probability of mixup. Not implemented: any value is ignored and no mixup is applied |
| `flip_p` | float | `0.0` | Probability of mirroring image + keypoints left/right (swaps symmetric node pairs) |
