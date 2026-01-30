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
    gaussian_noise_std: 1.0

    # Contrast
    contrast_min: 0.9
    contrast_max: 1.1
    contrast_p: 0.0

    # Brightness
    brightness_min: 1.0
    brightness_max: 1.0
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
```

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

### PreprocessingConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ensure_rgb` | bool | `false` | Convert images to 3 channels (RGB). Single-channel images are replicated |
| `ensure_grayscale` | bool | `false` | Convert images to 1 channel (grayscale) |
| `max_height` | int | `null` | Pad images to this height |
| `max_width` | int | `null` | Pad images to this width |
| `scale` | float | `1.0` | Resize factor (e.g., 0.5 = half size) |
| `crop_size` | int | `null` | Crop size for centered-instance models. If `null`, auto-computed from data |
| `min_crop_size` | int | `100` | Minimum crop size when auto-computing |
| `crop_padding` | int | `null` | Extra padding around crops. If `null`, auto-computed from augmentation settings |

### IntensityConfig (augmentation_config.intensity)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `uniform_noise_min` | float | `0.0` | Minimum uniform noise value (0-1 scale) |
| `uniform_noise_max` | float | `1.0` | Maximum uniform noise value (0-1 scale) |
| `uniform_noise_p` | float | `0.0` | Probability of applying uniform noise |
| `gaussian_noise_mean` | float | `0.0` | Mean of Gaussian noise distribution |
| `gaussian_noise_std` | float | `1.0` | Standard deviation of Gaussian noise |
| `gaussian_noise_p` | float | `0.0` | Probability of applying Gaussian noise |
| `contrast_min` | float | `0.9` | Minimum contrast factor |
| `contrast_max` | float | `1.1` | Maximum contrast factor |
| `contrast_p` | float | `0.0` | Probability of applying contrast adjustment |
| `brightness_min` | float | `1.0` | Minimum brightness factor |
| `brightness_max` | float | `1.0` | Maximum brightness factor (max 2.0) |
| `brightness_p` | float | `0.0` | Probability of applying brightness adjustment |

### GeometricConfig (augmentation_config.geometric)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rotation_min` | float | `-15.0` | Minimum rotation angle in degrees |
| `rotation_max` | float | `15.0` | Maximum rotation angle in degrees |
| `rotation_p` | float | `1.0` | Probability of rotation (independent). If `null`, uses `affine_p` |
| `scale_min` | float | `0.9` | Minimum scale factor |
| `scale_max` | float | `1.1` | Maximum scale factor |
| `scale_p` | float | `1.0` | Probability of scaling (independent). If `null`, uses `affine_p` |
| `translate_width` | float | `0.0` | Maximum horizontal translation as fraction of width |
| `translate_height` | float | `0.0` | Maximum vertical translation as fraction of height |
| `translate_p` | float | `null` | Probability of translation (independent). If `null`, uses `affine_p` |
| `affine_p` | float | `0.0` | Probability of bundled affine transform (rotation + scale + translate) |
| `erase_scale_min` | float | `0.0001` | Minimum erased area as proportion of image |
| `erase_scale_max` | float | `0.01` | Maximum erased area as proportion of image |
| `erase_ratio_min` | float | `1.0` | Minimum aspect ratio of erased area |
| `erase_ratio_max` | float | `1.0` | Maximum aspect ratio of erased area |
| `erase_p` | float | `0.0` | Probability of random erasing |
| `mixup_lambda_min` | float | `0.01` | Minimum mixup strength |
| `mixup_lambda_max` | float | `0.05` | Maximum mixup strength |
| `mixup_p` | float | `0.0` | Probability of applying mixup |
