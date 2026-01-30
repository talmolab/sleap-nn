# Model Config

Configure model architecture: backbone and heads.

---

## Structure

```yaml
model_config:
  init_weights: default
  pretrained_backbone_weights: null
  pretrained_head_weights: null

  backbone_config:
    unet: {...}       # Only one backbone
    convnext: null
    swint: null

  head_configs:
    single_instance: {...}  # Only one head type
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
```

---

## Backbones

Choose **one** backbone. Set others to `null`.

### UNet

Most flexible, good for any resolution:

```yaml
backbone_config:
  unet:
    in_channels: 1          # 1=grayscale, 3=RGB
    filters: 32             # Base filters (16, 32, 64)
    filters_rate: 2.0       # Filter multiplier per level
    max_stride: 16          # Receptive field (16 or 32)
    kernel_size: 3
    middle_block: true
    up_interpolate: true
    stacks: 1
    convs_per_block: 2
    output_stride: 2        # Output resolution divisor
```

**Size presets:**

| Preset | filters | max_stride | Speed | Accuracy |
|--------|---------|------------|-------|----------|
| Small | 16 | 16 | Fast | Good |
| Medium | 32 | 16 | Medium | Better |
| Large | 64 | 32 | Slow | Best |

### ConvNeXt

Modern CNN, good with pretrained weights:

```yaml
backbone_config:
  convnext:
    model_type: tiny       # tiny, small, base, large
    in_channels: 1
    max_stride: 32
    output_stride: 2
    up_interpolate: true
    pre_trained_weights: ConvNeXt_Tiny_Weights  # ImageNet
```

### Swin Transformer

Vision transformer, captures global context:

```yaml
backbone_config:
  swint:
    model_type: tiny       # tiny, small, base
    in_channels: 1
    max_stride: 32
    output_stride: 2
    up_interpolate: true
    pre_trained_weights: Swin_T_Weights  # ImageNet
```

---

## Heads

Choose the head type for your task. Set others to `null`.

### Single Instance

One animal per frame:

```yaml
head_configs:
  single_instance:
    confmaps:
      part_names: null    # null = all from skeleton
      sigma: 5.0          # Gaussian spread
      output_stride: 2
```

### Centroid (Top-Down Stage 1)

Detect instance centers:

```yaml
head_configs:
  centroid:
    confmaps:
      anchor_part: null   # null = bbox center
      sigma: 5.0
      output_stride: 2
```

### Centered Instance (Top-Down Stage 2)

Pose on cropped instances:

```yaml
head_configs:
  centered_instance:
    confmaps:
      anchor_part: null
      sigma: 5.0
      output_stride: 2
```

### Bottom-Up

All keypoints + grouping:

```yaml
head_configs:
  bottomup:
    confmaps:
      sigma: 2.5
      output_stride: 4
      loss_weight: 1.0
    pafs:
      sigma: 75.0
      output_stride: 8
      loss_weight: 1.0
```

### Multi-Class Bottom-Up

Bottom-up with identity:

```yaml
head_configs:
  multi_class_bottomup:
    confmaps:
      sigma: 5.0
      output_stride: 2
      loss_weight: 1.0
    class_maps:
      classes: null       # null = from track names
      sigma: 5.0
      output_stride: 2
      loss_weight: 1.0
```

### Multi-Class Top-Down

Top-down with identity:

```yaml
head_configs:
  multi_class_topdown:
    confmaps:
      anchor_part: null
      sigma: 5.0
      output_stride: 2
      loss_weight: 1.0
    class_vectors:
      classes: null
      num_fc_layers: 1
      num_fc_units: 64
      output_stride: 16   # Match backbone max_stride
      loss_weight: 1.0
```

---

## Pretrained Weights

### From Previous Training

```yaml
model_config:
  pretrained_backbone_weights: /path/to/best.ckpt
  pretrained_head_weights: /path/to/best.ckpt
```

### From Legacy SLEAP

```yaml
model_config:
  pretrained_backbone_weights: /path/to/best_model.h5
  pretrained_head_weights: /path/to/best_model.h5
```

!!! note "UNet only"
    Legacy SLEAP weights only work with UNet backbone.

---

## Common Configurations

### Single Instance (Medium)

```yaml
model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 16
      output_stride: 2
  head_configs:
    single_instance:
      confmaps:
        sigma: 5.0
        output_stride: 2
```

### Bottom-Up (Large)

```yaml
model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 32
      output_stride: 4
  head_configs:
    bottomup:
      confmaps:
        sigma: 2.5
        output_stride: 4
      pafs:
        sigma: 75.0
        output_stride: 8
```

### Top-Down with ConvNeXt

```yaml
# Centroid model
model_config:
  backbone_config:
    convnext:
      model_type: tiny
      pre_trained_weights: ConvNeXt_Tiny_Weights
  head_configs:
    centroid:
      confmaps:
        sigma: 5.0

# Instance model (separate training)
model_config:
  backbone_config:
    convnext:
      model_type: tiny
  head_configs:
    centered_instance:
      confmaps:
        sigma: 5.0
```

---

## Tips

!!! tip "Match output_stride"
    Set `backbone_config.*.output_stride` to match the minimum `head_configs.*.output_stride`.

!!! tip "Sigma tuning"
    - Larger sigma (5-10): Easier to learn, less precise
    - Smaller sigma (1-3): More precise, harder to learn

!!! tip "PAF sigma"
    Use larger sigma for PAFs (50-100) than for confmaps.

---

## Full Reference

### ModelConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `init_weights` | str | `default` | Weight initialization: `default` (kaiming) or `xavier` |
| `pretrained_backbone_weights` | str | `null` | Path to `.ckpt` or `.h5` file for backbone weights |
| `pretrained_head_weights` | str | `null` | Path to `.ckpt` or `.h5` file for head weights |

### UNetConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `in_channels` | int | `1` | Input channels (1=grayscale, 3=RGB) |
| `kernel_size` | int | `3` | Convolution kernel size |
| `filters` | int | `32` | Base number of filters |
| `filters_rate` | float | `1.5` | Filter multiplier per level |
| `max_stride` | int | `16` | Maximum stride (16 or 32) |
| `stem_stride` | int | `null` | Additional downsampling in stem |
| `middle_block` | bool | `true` | Add block at encoder end |
| `up_interpolate` | bool | `true` | Use interpolation (vs transposed conv) for upsampling |
| `stacks` | int | `1` | Number of decoder stacks |
| `convs_per_block` | int | `2` | Convolutions per block |
| `output_stride` | int | `1` | Output resolution divisor |

### ConvNextConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_type` | str | `tiny` | Architecture: `tiny`, `small`, `base`, `large` |
| `pre_trained_weights` | str | `null` | ImageNet weights: `ConvNeXt_Tiny_Weights`, etc. |
| `in_channels` | int | `1` | Input channels |
| `kernel_size` | int | `3` | Convolution kernel size |
| `filters_rate` | float | `2` | Filter multiplier |
| `convs_per_block` | int | `2` | Convolutions per block |
| `stem_patch_kernel` | int | `4` | Stem layer kernel size |
| `stem_patch_stride` | int | `2` | Stem layer stride |
| `up_interpolate` | bool | `true` | Use interpolation for upsampling |
| `output_stride` | int | `1` | Output resolution divisor |
| `max_stride` | int | `32` | Fixed at 32 for all ConvNeXt |

### SwinTConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_type` | str | `tiny` | Architecture: `tiny`, `small`, `base` |
| `pre_trained_weights` | str | `null` | ImageNet weights: `Swin_T_Weights`, `Swin_S_Weights`, `Swin_B_Weights` |
| `in_channels` | int | `1` | Input channels |
| `kernel_size` | int | `3` | Convolution kernel size |
| `filters_rate` | float | `2` | Filter multiplier |
| `convs_per_block` | int | `2` | Convolutions per block |
| `patch_size` | int | `4` | Patch size for stem |
| `stem_patch_stride` | int | `2` | Stem stride |
| `window_size` | int | `7` | Attention window size |
| `up_interpolate` | bool | `true` | Use interpolation for upsampling |
| `output_stride` | int | `1` | Output resolution divisor |
| `max_stride` | int | `32` | Fixed at 32 for all SwinT |

### single_instance.confmaps

For single animal per frame.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `part_names` | list | `null` | Body parts to predict (null = all from skeleton) |
| `sigma` | float | `5.0` | Gaussian spread in pixels |
| `output_stride` | int | `1` | Output resolution divisor |

```yaml
head_configs:
  single_instance:
    confmaps:
      part_names: null
      sigma: 5.0
      output_stride: 2
```

### centroid.confmaps

For detecting instance centers (top-down stage 1).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `anchor_part` | str | `null` | Anchor point (null = bbox center) |
| `sigma` | float | `5.0` | Gaussian spread in pixels |
| `output_stride` | int | `1` | Output resolution divisor |

```yaml
head_configs:
  centroid:
    confmaps:
      anchor_part: null
      sigma: 5.0
      output_stride: 2
```

### centered_instance.confmaps

For pose estimation on cropped instances (top-down stage 2).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `part_names` | list | `null` | Body parts to predict (null = all from skeleton) |
| `anchor_part` | str | `null` | Anchor point (null = bbox center) |
| `sigma` | float | `5.0` | Gaussian spread in pixels |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `1.0` | Loss weighting |

```yaml
head_configs:
  centered_instance:
    confmaps:
      part_names: null
      anchor_part: null
      sigma: 5.0
      output_stride: 2
```

### bottomup.confmaps

For detecting all keypoints in bottom-up models.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `part_names` | list | `null` | Body parts to predict (null = all from skeleton) |
| `sigma` | float | `5.0` | Gaussian spread in pixels |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `null` | Loss weighting |

```yaml
head_configs:
  bottomup:
    confmaps:
      part_names: null
      sigma: 2.5
      output_stride: 4
      loss_weight: 1.0
```

### bottomup.pafs

Part Affinity Fields for grouping keypoints into instances.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `edges` | list | `null` | Edge connections (null = from skeleton) |
| `sigma` | float | `15.0` | PAF spread (use larger than confmaps, typically 50-100) |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `null` | Loss weighting |

```yaml
head_configs:
  bottomup:
    confmaps:
      sigma: 2.5
      output_stride: 4
      loss_weight: 1.0
    pafs:
      sigma: 75.0
      output_stride: 8
      loss_weight: 1.0
```

### multi_class_bottomup.confmaps

Confidence maps for bottom-up models with identity.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `part_names` | list | `null` | Body parts to predict |
| `sigma` | float | `5.0` | Gaussian spread |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `null` | Loss weighting |

### multi_class_bottomup.class_maps

Class/identity maps for bottom-up ID models.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `classes` | list | `null` | Class names (null = from track names) |
| `sigma` | float | `5.0` | Gaussian spread |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `null` | Loss weighting |

```yaml
head_configs:
  multi_class_bottomup:
    confmaps:
      sigma: 5.0
      output_stride: 2
      loss_weight: 1.0
    class_maps:
      classes: null  # inferred from track names
      sigma: 5.0
      output_stride: 2
      loss_weight: 1.0
```

### multi_class_topdown.confmaps

Confidence maps for top-down models with identity.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `part_names` | list | `null` | Body parts to predict |
| `anchor_part` | str | `null` | Anchor point |
| `sigma` | float | `5.0` | Gaussian spread |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `1.0` | Loss weighting |

### multi_class_topdown.class_vectors

Classification head for top-down ID models.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `classes` | list | `null` | Class names (null = from track names) |
| `num_fc_layers` | int | `1` | Fully-connected layers before output |
| `num_fc_units` | int | `64` | Units per FC layer |
| `global_pool` | bool | `true` | Use global pooling |
| `output_stride` | int | `1` | Should match backbone max_stride |
| `loss_weight` | float | `1.0` | Loss weighting |

```yaml
head_configs:
  multi_class_topdown:
    confmaps:
      anchor_part: null
      sigma: 5.0
      output_stride: 2
      loss_weight: 1.0
    class_vectors:
      classes: null  # inferred from track names
      num_fc_layers: 1
      num_fc_units: 64
      output_stride: 16  # match backbone max_stride
      loss_weight: 1.0
```
