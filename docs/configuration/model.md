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
    bottomup_segmentation: null
    bbox_centernet: null
    bbox_ssd: null
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

### Pretrained (HuggingFace)

Reuse an external pretrained image encoder (ConvNeXtV2, ResNet, Swinv2,
DINOv2/v3, CLIP, SigLIP, MAE, …) from the HuggingFace Hub as the backbone —
frozen or fine-tuned. Requires the `backbones` extra (`pip install
"sleap-nn[backbones]"` / `uv sync --extra backbones`), which brings
`transformers` and `timm` (`timm` is needed only for `timm/...` checkpoints).

```yaml
backbone_config:
  pretrained:
    source: hf                                   # HuggingFace Hub
    model_name: facebook/convnextv2-nano-22k-224 # HF model id, or a timm/... checkpoint
    weights: true          # download & load pretrained weights (false = random init)
    mode: auto             # auto | decoder (spatial heads) | encoder (pooled heads)
    freeze: false          # true = feature extraction (encoder frozen)
    normalize: true        # apply the model's image_mean/std
    revision: null         # pin an HF commit sha/tag for reproducibility
    in_channels: 3         # HF stems are 3-channel (grayscale is replicated)
    output_stride: 2       # finest decoder output (decoder mode)
    max_stride: 32         # 32 for CNN/Swin; a multiple of 2 x token_stride for a plain ViT
    # Plain-ViT knobs (ignored by hierarchical backbones unless noted):
    token_stride: null     # null = power of two nearest the patch size (16 for patch 14/16)
    layers: null           # null = 4 evenly spaced taps, e.g. [3, 6, 9, 12]
    vit_neck: dpt          # dpt | sfp
    neck_channels: null    # null = [h/8, h/4, h/2, h] for hidden size h
    gradient_checkpointing: false  # any encoder
    drop_path_rate: null           # stochastic depth, ramped 0 -> rate over blocks
    # encoder_lr_scale / layer_decay moved to trainer_config.optimizer
    detail_stem: false     # conv stream adding stride-2/4/8 skips to the ViT pyramid
    detail_channels: null  # null = [32, 64, 128]
    vit_decoder: unet      # unet (neck + UNet decoder) | classic | simple (ViTPose heads)
    vit_decoder_channels: 256  # classic deconv channels
    vit_final_norm: null   # null = on for classic/simple: last tap uses the encoder's final norm
```

Three integration cases (`mode: auto` picks one from the backbone family and the
model's heads; `mode` forces it):

- **Case A — `decoder`, hierarchical backbone** (ConvNeXtV2 / ResNet / Swinv2 /
  DINOv3-ConvNeXt): its stride-4/8/16/32 stage pyramid feeds sleap-nn's decoder,
  so any spatial head (pose, centroid, PAFs, segmentation) works.
- **Case B — `encoder`, encoder-only**: a single bottleneck feature map, no
  decoder, for pooled heads. `auto` resolves to it only for the `embedding`
  (re-ID) model type.
- **Case C — `decoder`, plain ViT** (DINOv2/v3, CLIP, SigLIP, MAE, PE-Spatial,
  EVA02, …): 4 tapped transformer layers go through a feature-pyramid neck
  (strides `token_stride/4` to `2 × token_stride`) into the same decoder, so a
  plain ViT can feed spatial heads. `auto` resolves to it whenever the model has a
  spatial head.

See the [Pretrained Backbones guide](../guides/pretrained-backbones.md) for the
tested-model table, `timm/...` checkpoints, the ViT knobs, gating/`HF_TOKEN` and
offline workflow, and freeze / fine-tune guidance. The full field list is in
[PretrainedConfig](#pretrainedconfig) below.

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

### Bottom-Up Segmentation

Single-stage instance **segmentation** (Panoptic-DeepLab style): a foreground
head (BCE+Dice), an instance-center heatmap, and a per-pixel center-offset field
used to group foreground pixels into per-instance masks. Three heads, all sharing
a fine output stride:

```yaml
head_configs:
  bottomup_segmentation:
    segmentation:        # foreground probability (union of all instance masks)
      output_stride: 2
      loss_weight: 1.0
    center:              # instance-center heatmap
      sigma: 10.0
      output_stride: 2
      loss_weight: 1.0
    offsets:             # per-pixel (dx, dy) to the instance center
      output_stride: 2
      loss_weight: 0.1
```

Notes:

- `fg_threshold` and `min_mask_area` are **inference-time** post-processing args
  (`sleap-nn predict --fg_threshold` / `--min_mask_area`), not training config.
- Train at `scale: 1.0` with input dims divisible by `max_stride` (mask resize is
  not pad-aware in v1). Geometric augmentation (rotation/scale/translate/flip)
  co-transforms the per-instance masks with the image (nearest-neighbor,
  re-binarized); prefer small rotation ranges since a full-frame rotation can clip
  instances at the frame edge.
- See the sample config
  [`config_bottomup_segmentation_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bottomup_segmentation_unet.yaml).

### Bounding-Box Detectors

Axis-aligned box detection, one box per animal (written as
`sio.PredictedBoundingBox` on `LabeledFrame.bboxes`). Ground-truth boxes come from
the labels' own boxes or are derived from poses / masks
(`data_config.bbox_source`). Two styles:

```yaml
head_configs:
  bbox_centernet:        # CenterNet-style: center heatmap + size + sub-cell offset
    center:
      categories: null   # filled from the training boxes
      sigma: null        # null = size-adaptive Gaussian radius
      output_stride: 4
    size:
      output_stride: 4
      loss_weight: 0.1
    offset:
      output_stride: 4
      loss_weight: 1.0
```

```yaml
head_configs:
  bbox_ssd:              # SSD-style: multi-scale anchors over strides 8/16/32
    ssd:
      output_stride: 8   # finest level; set the backbone's output_stride to match
      num_levels: 3      # needs backbone max_stride >= 32
      anchor_sizes: null # null = auto-sized from the training boxes, saved into the config
```

Notes:

- Keep rotation augmentation off: the axis-aligned box of rotated corners is larger
  than the rotated animal.
- With `trainer_config.eval.enabled: true`, the best checkpoint and early stopping
  follow `eval/val/bbox_mAP` (see `eval.select_metric`).
- See the [Bounding-box detection guide](../guides/bbox-detection.md) and the sample
  configs
  [`config_bbox_centernet_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_centernet_unet.yaml) /
  [`config_bbox_ssd_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_ssd_unet.yaml).

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

### PretrainedConfig

External HuggingFace / timm backbone (`backbone_config.pretrained`). See the
[Pretrained Backbones guide](../guides/pretrained-backbones.md).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `source` | str | `hf` | Backbone source; only `hf` (HuggingFace Hub) is supported |
| `model_name` | str | `facebook/convnextv2-nano-22k-224` | HF model id, or a `timm/...` checkpoint (needs `timm`) |
| `weights` | bool | `true` | Download and load pretrained weights; `false` = random init from the model config |
| `mode` | str | `auto` | `auto`, `decoder` (Case A hierarchical / Case C plain ViT), or `encoder` (Case B pooled bottleneck). `auto` picks `decoder` when the model has a spatial head, `encoder` for `embedding` |
| `freeze` | bool | `false` | Freeze the pretrained encoder (feature extraction); the neck, decoder and heads still train |
| `revision` | str | `null` | HF revision (commit sha / tag) to pin |
| `normalize` | bool | `true` | Apply the model's mean/std inside the backbone (forced off for `embedding`) |
| `image_mean` | list[float] | `null` | Explicit per-channel mean; `null` = timm `pretrained_cfg` / `AutoImageProcessor` / ImageNet |
| `image_std` | list[float] | `null` | Explicit per-channel std |
| `out_indices` | list[int] | `null` | Hierarchical stage indices to tap; `null` = all stages, deduplicated by stride. HF backbones index `stage_names` (`0` is the stem: a ConvNeXt's stages are `[1, 2, 3, 4]`); `timm/...` checkpoints index `feature_info` (`0` is the first stage) |
| `in_channels` | int | `3` | Stem input channels (set to 3 by the trainer) |
| `filters_rate` | float | `2.0` | Decoder filter growth factor |
| `convs_per_block` | int | `2` | Convolutions per decoder block |
| `kernel_size` | int | `3` | Decoder conv kernel size |
| `up_interpolate` | bool | `true` | Bilinear upsampling (vs. transposed conv) in the decoder |
| `output_stride` | int | `2` | Finest decoder output stride (plain ViT: at most `token_stride`) |
| `max_stride` | int | `32` | Deepest encoder stride: 32 for hierarchical backbones; for a plain ViT a multiple of `2 × token_stride` |
| `token_stride` | int | `null` | Plain ViT: stride of the token grid (power of two ≥ 4; the input is resized to match). `null` = power of two nearest the patch size (16 for patch 14 or 16). `8` = finer grid, ~4× encoder cost |
| `layers` | list[int] | `null` | Plain ViT: layers to tap, 1-indexed (`-1` = last). `null` = one evenly spaced tap per built pyramid level, ending at the last, for `dpt` (`[3, 6, 9, 12]` for 12 layers); the last layer for `sfp` / encoder mode (which take exactly one) |
| `vit_neck` | str | `dpt` | Plain ViT: `dpt` (one tapped layer per pyramid level) or `sfp` (ViTDet Simple Feature Pyramid from the last tap) |
| `neck_channels` | list[int] | `null` | Plain ViT: channels of the 4 neck levels, finest first. `null` = `[h/8, h/4, h/2, h]` for hidden size `h` |
| `gradient_checkpointing` | bool | `false` | Enable gradient checkpointing in the encoder during training (less memory, slower) |
| `encoder_lr_scale` | float | `1.0` | **Deprecated**: use [`trainer_config.optimizer.encoder_lr_scale`](trainer.md#optimizer). Still read, with a warning, when that is unset |
| `layer_decay` | float | `null` | **Deprecated**: use [`trainer_config.optimizer.layer_decay`](trainer.md#optimizer). Still read, with a warning, when that is unset |
| `drop_path_rate` | float | `null` | Stochastic depth of the encoder, ramped linearly from `0` (first block) to this value (last block). HF DINOv2 / DINOv3-ViT apply the rate uniformly; sleap-nn converts them to the ramp. Ignored, with a warning, for families without stochastic depth (CLIP, SigLIP, ViT, MAE). `null` keeps the checkpoint's setting |
| `detail_stem` | bool | `false` | Plain ViT: add a light conv stream whose stride-2/4/8 features join the pyramid as decoder skips |
| `detail_channels` | list[int] | `null` | Detail-stem channels at strides 2, 4, 8 (exactly three). `null` = `[32, 64, 128]` |
| `vit_decoder` | str | `unet` | Plain ViT: `unet` (`vit_neck` pyramid + UNet decoder), or a ViTPose-style direct decoder on the last tap: `classic` (deconv 4×4/s2 + BN + ReLU blocks down to `output_stride`) or `simple` (ReLU + bilinear upsampling, no parameters; the heads' output conv becomes 3×3, as in ViTPose). The direct decoders take one tap and ignore `vit_neck`, `neck_channels`, `detail_stem`; `max_stride` need only be a multiple of `token_stride` |
| `vit_decoder_channels` | int | `256` | Channels of the `classic` deconv blocks |
| `vit_final_norm` | bool | `null` | Plain ViT: normalize a last-layer tap with the encoder's own pretrained final norm instead of a fresh LayerNorm. `null` = on for `classic`/`simple`, off for `unet` |

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
