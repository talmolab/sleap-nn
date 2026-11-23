# Model Config

Configure model architecture: backbone and heads.

---

## Structure

```yaml
model_config:
  init_weights: default
  pretrained_backbone_weights: null
  pretrained_head_weights: null
  init_from: null     # Initialize from another model (see below)

  backbone_config:
    unet: {...}       # Only one backbone
    convnext: null
    swint: null
    pretrained: null       # HuggingFace / timm encoder (also bbox_detr's DINOv3)
    dinov2_windowed: null  # RF-DETR's windowed DINOv2 (bbox_detr only)

  head_configs:
    single_instance: {...}  # Only one head type
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
    bottomup_segmentation: null
    centered_instance_segmentation: null
    semantic_segmentation: null
    embedding: null
    bbox_centernet: null
    bbox_ssd: null
    bbox_detr: null

  total_params: null  # set automatically when training starts
```

---

## Backbones

Choose **one** backbone. Set others to `null`.

### UNet

Most flexible, good for any resolution:

```yaml
backbone_config:
  unet:
    in_channels: 1          # 1=grayscale, 3=RGB; reset at training to match the images
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

**Receptive-field presets** (the `unet_medium_rf` / `unet_large_rf` backbones of
`sleap-nn config` and the Python config API):

| Preset | filters | filters_rate | max_stride | Receptive field |
|--------|---------|--------------|------------|-----------------|
| `unet_medium_rf` | 32 | 2.0 | 16 | Medium |
| `unet_large_rf` | 24 | 1.5 | 32 | Large |

A larger `max_stride` adds a down/up-sampling level, so the model sees more context
per output pixel at a higher cost.

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

A `bbox_detr` model uses a plain ViT in a fourth way, `mode: tokens` (written for
you when `mode: auto`): the tapped layers' token maps go straight to the DETR
projector, each through the encoder's pretrained final norm, with no neck or
decoder. See the [DETR box detector guide](../guides/bbox-detr.md).

See the [Pretrained Backbones guide](../guides/pretrained-backbones.md) for the
tested-model table, `timm/...` checkpoints, the ViT knobs, gating/`HF_TOKEN` and
offline workflow, and freeze / fine-tune guidance. The full field list is in
[PretrainedConfig](#pretrainedconfig) below.

### Windowed DINOv2 (RF-DETR, `bbox_detr` only)

The backbone of RF-DETR's checkpoints: DINOv2 on a 16 px token grid (its 14 x 14
patch embedding resampled to 16 x 16) with windowed attention in the blocks
between taps. `sleap-nn import-rfdetr` writes it; set `weights` to start it from a
plain DINOv2 checkpoint instead. Defaults are RF-DETR-M's.

```yaml
backbone_config:
  dinov2_windowed:
    hidden_size: 384         # ViT-S (768: ViT-B)
    num_layers: 12
    num_heads: 6
    patch_size: 16
    pos_embed_grid: 36       # training size / 16 (RF-DETR: 24 / 32 / 36 / 44 for N / S / M / L)
    out_indices: [3, 6, 9, 12]
    num_windows: 2           # input sides must be multiples of patch_size * num_windows
    window_block_indexes: null  # null = every block up to the last tap except the taps
    drop_path_rate: 0.0
    weights: null            # e.g. facebook/dinov2-small; null = random (a checkpoint loads after)
    output_stride: 16
    max_stride: 32
```

Full field list: [Dinov2WindowedConfig](#dinov2windowedconfig).

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
      anchor_part: null   # null = mean of the visible nodes (center of mass)
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
      output_stride: 16   # set to the backbone max_stride by the trainer
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
  Field tables: [bottomup_segmentation](#bottomup_segmentation).

### Centered-Instance Segmentation (Top-Down Stage 2)

A single foreground-mask head that predicts the **centered** instance's binary
mask on a centroid crop (the segmentation analog of `centered_instance`). Train it
alongside a `centroid` model; `sleap-nn predict` composes the two. See the
[Top-down segmentation guide](../guides/topdown-segmentation.md).

```yaml
head_configs:
  centered_instance_segmentation:
    segmentation:
      output_stride: 2
      loss_weight: 1.0
      anchor_part: null   # null = mean of the visible nodes
```

Field table: [centered_instance_segmentation.segmentation](#centered_instance_segmentationsegmentation).
Sample: [`config_topdown_centered_instance_segmentation_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_centered_instance_segmentation_unet.yaml).

### Semantic Segmentation (Whole Frame)

One binary foreground/background map over the **whole frame** (the union of the
frame's masks): no crop, no center/offset heads, no instance grouping. Prediction
writes one instance-less mask per frame. See the
[Semantic segmentation guide](../guides/semantic-segmentation.md).

```yaml
head_configs:
  semantic_segmentation:
    segmentation:
      output_stride: 2
      loss_weight: 1.0
      bce_weight: 0.5
      dice_weight: 0.5
```

Field table: [semantic_segmentation.segmentation](#semantic_segmentationsegmentation).
Sample: [`config_semantic_segmentation_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_semantic_segmentation_unet.yaml).

### Embedding (Appearance / Re-ID)

A pooled head that maps each instance crop to an appearance vector, trained with a
contrastive objective. The vectors feed embedding-based tracking. See the
[Embedding tracking guide](../guides/embedding-tracking.md).

```yaml
head_configs:
  embedding:
    embedding:
      embedding_dim: 128
      num_fc_layers: 1
      num_fc_units: 256
      normalize: true
      objective:
        positives:
          scope: global_id      # aug_view | tracklet | global_id
        negatives:
          sources: [same_frame, in_batch]
        loss:
          name: supcon          # supcon | infonce | triplet
        sampler:
          kind: pk              # pk | within_video | random
          groups_per_batch: 8
          samples_per_group: 16
```

Field tables: [embedding.embedding](#embeddingembedding) and
[embedding.embedding.objective](#embeddingembeddingobjective).
Sample: [`config_embedding_convnext.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_embedding_convnext.yaml).

### Bounding-Box Detectors

Axis-aligned box detection, one box per animal (written as
`sio.PredictedBoundingBox` on `LabeledFrame.bboxes`). Ground-truth boxes come from
the labels' own boxes or are derived from poses / masks
(`data_config.bbox_source`). Three styles:

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
      output_stride: 8   # finest level (strides 8, 16, 32)
      num_levels: 3      # the backbone max_stride is raised to the deepest level if needed
      anchor_sizes: null # null = auto-sized from the training boxes, saved into the config
```

```yaml
head_configs:
  bbox_detr:             # RF-DETR-style set prediction (no NMS); needs a ViT backbone
    detr:
      categories: null   # filled from the training boxes
      num_queries: 100   # detections per frame (RF-DETR: 300)
      num_select: 100
      dec_layers: 3
      group_detr: 13     # query groups in training
```

Notes:

- Keep rotation augmentation off: the axis-aligned box of rotated corners is larger
  than the rotated animal.
- `bbox_detr` takes a `pretrained` plain ViT (DINOv3 / DINOv2) or `dinov2_windowed`
  backbone, `preprocessing.scale: 1.0` with `max_height` / `max_width` as the input
  size, and usually RF-DETR's augmentation (`augmentation_config.detr`). See the
  [DETR box detector guide](../guides/bbox-detr.md) and
  [`config_bbox_detr_dinov3_s_mouse.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_detr_dinov3_s_mouse.yaml).
- With `trainer_config.eval.enabled: true`, the best checkpoint and early stopping
  follow `eval/val/bbox_mAP` (see `eval.select_metric`).
- See the [Bounding-box detection guide](../guides/bbox-detection.md) and the sample
  configs
  [`config_bbox_centernet_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_centernet_unet.yaml) /
  [`config_bbox_ssd_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_ssd_unet.yaml).
  Field tables: [bbox_centernet](#bbox_centernet) and [bbox_ssd.ssd](#bbox_ssdssd).

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

!!! note "Shape mismatches raise"
    A `.ckpt` tensor whose shape differs from the new model's raises an error
    with these two keys, and head channels are copied by position, as in
    earlier releases. Set `init_from.remap_nodes: true` to copy head channels by
    node name (a skeleton that grows or is reordered), `init_from.strict: false`
    to skip tensors that do not fit with a warning (the target keeps its init),
    or use `init_from.path`, which does both by default.

### From Another Model, Tolerating Differences

`init_from` loads what fits from another sleap-nn model: head channels are matched
by node name (so the skeleton can grow or be reordered), learned positional
embeddings are resized, and selected parts can be re-initialized. Every tensor is
accounted for in a logged report.

```yaml
model_config:
  init_from:
    path: /runs/coco17          # model directory, .ckpt or state dict
    include: null               # e.g. [backbone]
    exclude: null
    node_map: null              # {target node: source node}
    reinit: null                # e.g. [head] or ["blocks[-2:]", head]
    strict: false
```

See the [Initializing from Another Model](../guides/init-from.md) guide.

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

Two separate configs, one per model.

Centroid model:

```yaml
model_config:
  backbone_config:
    convnext:
      model_type: tiny
      pre_trained_weights: ConvNeXt_Tiny_Weights
  head_configs:
    centroid:
      confmaps:
        sigma: 5.0
```

Instance model (trained separately):

```yaml
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

!!! tip "output_stride is set from the heads"
    At training setup the backbone's `output_stride` is set to the smallest head
    `output_stride`, and its `max_stride` is raised to the largest head stride if it
    is smaller. The values written in `backbone_config` are overridden, so set the
    strides on the heads. Every head counts, including the pooled ones:
    `embedding.output_stride` (default `32`) and
    `multi_class_topdown.class_vectors.output_stride` raise the backbone
    `max_stride` first and are then set equal to it, so an `embedding` model on a
    UNet with `max_stride: 16` trains at `max_stride: 32` unless you lower the
    head's `output_stride` to 16.

!!! tip "in_channels is set from the images"
    At training setup the backbone's `in_channels` is reset to the channel count of
    the training images (after `ensure_rgb` / `ensure_grayscale`). A ConvNeXt / Swin
    backbone with `pre_trained_weights`, or a `pretrained` backbone, is forced to 3
    channels (for non-`embedding` models, grayscale images are also converted to RGB).

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
| `init_from` | InitFromConfig | `null` | Shape-tolerant, node-name-aware init from another model, plus re-init ([guide](../guides/init-from.md)) |
| `total_params` | int | `null` | Set automatically: the model's parameter count, computed when training starts |

### InitFromConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | str | `null` | Model directory (`best.ckpt`), `.ckpt` or raw state dict. `null`: load nothing (`reinit` still applies) |
| `include` | list | `null` | Selectors of weights to load (`null` = all): `all`, `backbone`, `encoder`, `decoder`, `neck`, `head`, `embeddings`, `blocks[a:b]`, globs |
| `exclude` | list | `null` | Selectors of weights to leave at their init |
| `remap_nodes` | bool | `null` | Copy per-node head channels by node name. `null`: yes for `path`, no (by position) for `pretrained_head_weights` |
| `node_map` | dict | `null` | `{target node: source node}` overrides. For `bbox_detr` the class-head rows are matched by class name the same way (the never-positive extra logit is `__extra__`; leave it unmapped so it takes the source's own); classes the source lacks start fresh |
| `source_part_names` | list | `null` | Source node names, for checkpoints that record none |
| `interpolate_pos_embed` | bool | `true` | Resize learned positional embeddings to the new token grid |
| `reinit` | list | `null` | Selectors re-initialized after loading |
| `reinit_layer_scale` | float | `1e-5` | LayerScale for re-initialized ViT blocks |
| `strict` | bool | `null` | Raise on any shape-mismatched tensor. `null`: `init_from.path` skips with a warning, the `.ckpt` forms of `pretrained_*_weights` raise. `true` / `false` apply one rule to both |

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
| `max_stride` | int | `32` | Set by the trainer to `stem_patch_stride * 16` (32 with the default stem stride); a configured value is overridden |
| `arch` | dict | `null` | Custom `{depths: [...], channels: [...]}`, used only when `model_type` is not one of the four built-in types |

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
| `max_stride` | int | `32` | Set by the trainer to `stem_patch_stride * 16` (32 with the default stem stride); a configured value is overridden |
| `arch` | dict | `null` | Custom `{embed, depths, num_heads}`. Effectively unused: `model_type` must be one of `tiny`/`small`/`base`, which always selects a built-in architecture |

### PretrainedConfig

External HuggingFace / timm backbone (`backbone_config.pretrained`). See the
[Pretrained Backbones guide](../guides/pretrained-backbones.md).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `source` | str | `hf` | Backbone source; only `hf` (HuggingFace Hub) is supported |
| `model_name` | str | `facebook/convnextv2-nano-22k-224` | HF model id, or a `timm/...` checkpoint (needs `timm`) |
| `weights` | bool | `true` | Download and load pretrained weights; `false` = random init from the model config |
| `mode` | str | `auto` | `auto`, `decoder` (Case A hierarchical / Case C plain ViT), `encoder` (Case B pooled bottleneck), or `tokens` (the tapped token maps, for `bbox_detr`). `auto` picks `decoder` when the model has a spatial head, `encoder` for `embedding`, `tokens` for `bbox_detr` |
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
| `layers` | list[int] | `null` | Plain ViT: layers to tap, 1-indexed (`-1` = last). `null` = one evenly spaced tap per built pyramid level, ending at the last, for `dpt` (`[3, 6, 9, 12]` for 12 layers); the last layer for `sfp` / encoder mode (which take exactly one). `dpt` takes one tap per built level, or 4 taps (one per level of the full pyramid): when `output_stride` leaves the finer levels out, their (shallowest) taps are dropped, so `[3, 6, 9, 12]` at `output_stride: 8` and token stride 16 uses `[6, 9, 12]`. Any other count is an error that names the counts to use |
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
| `anchor_part` | str | `null` | Anchor node (null = mean of the visible nodes) |
| `centroid_method` | str | `null` | How the centroid is computed: `center_of_mass`, `bbox_center`, `geometric_median` or `anchor`. `null` = `anchor` when `anchor_part` is set, else `center_of_mass`. A non-anchor method together with `anchor_part` is an error |
| `centroid_fallback` | str | `null` | Used when the `anchor_part` node is not visible: `center_of_mass` (what `null` means), `bbox_center` or `geometric_median` |
| `centroid_source` | str | `null` | Training target: `user` (the labels' `UserCentroid`s; frames with poses but no user centroid are dropped) or `computed` (from keypoints, ignoring user centroids). `null` = `user` if the labels have user centroids, else `computed`, with a warning |
| `sigma` | float | `5.0` | Gaussian spread in pixels |
| `output_stride` | int | `1` | Output resolution divisor |
| `use_sigmoid_activation` | bool | `false` | Apply a sigmoid to the head output so it is a `(0, 1)` probability. Required by the focal loss |
| `focal_loss_alpha` | float | `0.0` | Non-zero replaces the MSE train loss with a CenterNet-style penalty-reduced focal loss (needs `use_sigmoid_activation: true`). `0` = plain MSE. See the [Centroid focal loss guide](../guides/centroid-focal-loss.md) |
| `focal_loss_beta` | float | `4.0` | Penalty-reduction exponent for negative pixels near a peak (focal loss only) |
| `focal_loss_pos_threshold` | float | `0.5` | Minimum target value for a pixel to count as positive (focal loss only) |

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
| `anchor_part` | str | `null` | Anchor node (null = mean of the visible nodes) |
| `centroid_method` | str | `null` | How the centroid is computed: `center_of_mass`, `bbox_center`, `geometric_median` or `anchor`. `null` = `anchor` when `anchor_part` is set, else `center_of_mass`. A non-anchor method together with `anchor_part` is an error |
| `centroid_fallback` | str | `null` | Used when the `anchor_part` node is not visible: `center_of_mass` (what `null` means), `bbox_center` or `geometric_median` |
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

For detecting all keypoints in bottom-up models. A `bottomup` head needs **both**
`confmaps` and `pafs`; leaving either one out fails at training setup.

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
    pafs:                # required alongside confmaps (see below)
      sigma: 75.0
      output_stride: 8
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
| `class_output` | str | `track` | How a predicted class is written: `track` (a per-video `sio.Track` only) or `identity` (also a global `sio.Identity` named after the class; use only when each class is one distinct animal) |
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
| `anchor_part` | str | `null` | Anchor node (null = mean of the visible nodes) |
| `centroid_method` | str | `null` | How the centroid is computed: `center_of_mass`, `bbox_center`, `geometric_median` or `anchor`. `null` = `anchor` when `anchor_part` is set, else `center_of_mass`. A non-anchor method together with `anchor_part` is an error |
| `centroid_fallback` | str | `null` | Used when the `anchor_part` node is not visible: `center_of_mass` (what `null` means), `bbox_center` or `geometric_median` |
| `sigma` | float | `5.0` | Gaussian spread |
| `output_stride` | int | `1` | Output resolution divisor |
| `loss_weight` | float | `1.0` | Loss weighting |

### multi_class_topdown.class_vectors

Classification head for top-down ID models.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `classes` | list | `null` | Class names (null = from track names) |
| `class_output` | str | `track` | How a predicted class is written: `track` (a per-video `sio.Track` only) or `identity` (also a global `sio.Identity` named after the class; use only when each class is one distinct animal) |
| `num_fc_layers` | int | `1` | Fully-connected layers before output |
| `num_fc_units` | int | `64` | Units per FC layer |
| `global_pool` | bool | `true` | Use global pooling |
| `output_stride` | int | `1` | Set by the trainer to the backbone `max_stride` (a value above the backbone's `max_stride` raises it first) |
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
      output_stride: 16  # set to the backbone max_stride by the trainer
      loss_weight: 1.0
```

### bottomup_segmentation

`segmentation` (foreground head; the same fields as `semantic_segmentation.segmentation`):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output_stride` | int | `2` | Output resolution divisor. `1` keeps thin structures sharp |
| `loss_weight` | float | `1.0` | Loss weighting |
| `bce_weight` | float | `0.5` | Weight of the BCE term of the BCE+Dice loss |
| `dice_weight` | float | `0.5` | Weight of the Dice term of the BCE+Dice loss |
| `bce_pos_weight` | float | `null` | Positive-class weight for BCE (e.g. 5-20 for thin/rare foreground). `null` = unweighted |
| `target_maxpool` | bool | `false` | Downsample the target with max-pool (any foreground in a stride cell counts) instead of area-average + 0.5 threshold; no effect at `output_stride: 1` |

`center` (instance-center heatmap):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sigma` | float | `4.0` | Gaussian spread of the center heatmap, in pixels. Dataset-dependent: elongated or large animals may want more |
| `output_stride` | int | `2` | Output resolution divisor |
| `loss_weight` | float | `1.0` | Loss weighting |

`offsets` (per-pixel offset to the instance center):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output_stride` | int | `2` | Output resolution divisor |
| `loss_weight` | float | `0.1` | Loss weighting |

### centered_instance_segmentation.segmentation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output_stride` | int | `2` | Output resolution divisor, relative to the crop |
| `loss_weight` | float | `1.0` | Weight of the BCE+Dice loss |
| `anchor_part` | str | `null` | Node used to center crops (null = mean of the visible nodes) |
| `centroid_method` | str | `null` | As for [`centered_instance.confmaps`](#centered_instanceconfmaps) |
| `centroid_fallback` | str | `null` | As for [`centered_instance.confmaps`](#centered_instanceconfmaps) |

### semantic_segmentation.segmentation

Same fields and defaults as the `segmentation` head of
[bottomup_segmentation](#bottomup_segmentation): `output_stride` (`2`),
`loss_weight` (`1.0`), `bce_weight` (`0.5`), `dice_weight` (`0.5`),
`bce_pos_weight` (`null`), `target_maxpool` (`false`).

### embedding.embedding

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_dim` | int | `128` | Length of the output vector |
| `num_fc_layers` | int | `1` | Fully-connected layers before the output |
| `num_fc_units` | int | `256` | Units per FC layer |
| `pool` | str | `null` | Pooling of the encoder features: `gem`, `max` or `avg`. `null` = set at training: `avg` for a frozen `pretrained` encoder, else `gem` |
| `normalize` | bool | `true` | L2-normalize the output vector |
| `output_stride` | int | `32` | Set by the trainer to the backbone `max_stride`. It is first counted among the head strides, so the default `32` raises a smaller backbone `max_stride` (e.g. a UNet's `16`) to `32`; set it to the backbone's `max_stride` to keep that |
| `freeze_backbone` | bool | `false` | Freeze the backbone's pretrained encoder and train only what sits on top (warns if there are no pretrained weights) |
| `anchor_part` | str | `null` | Node used to center the crop in pose mode (null = mean of the visible nodes) |
| `centroid_method` | str | `null` | As for [`centered_instance.confmaps`](#centered_instanceconfmaps); pose mode only |
| `centroid_fallback` | str | `null` | As for [`centered_instance.confmaps`](#centered_instanceconfmaps) |
| `detection_mode` | str | `null` | Detections to crop: `pose` or `mask`. `null` = chosen at training (`mask` under `preprocessing.burn_in`, else the carrier with more samples, ties to `mask`) and written into the saved config |
| `objective` | ObjectiveConfig | `null` | Training objective (below); unset parts take their defaults |

The crop itself is set by `data_config.preprocessing` (`crop_size`, `burn_in`,
`background_fill`, `crop_centering`), and identity semantics by
`data_config.identity` (see [Data Config](data.md)).

### embedding.embedding.objective

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `positives.scope` | str | `global_id` | Positives of an anchor: `aug_view` (its own augmented views; self-supervised), `tracklet` (same video + track) or `global_id` (same `sio.Identity`, else same track name across videos) |
| `positives.aug_views` | int | `2` | Augmented views per anchor; only `2` is supported |
| `negatives.sources` | list | `[same_frame, in_batch]` | Negative sources, any of `same_frame` and `in_batch` |
| `negatives.exclude_same_track` | bool | `true` | Drop same (video, track) pairs from the negatives |
| `negatives.restrict_same_video` | bool | `false` | Use only same-video negatives (required for `scope: tracklet`) |
| `loss.name` | str | `supcon` | `supcon`, `infonce` or `triplet` |
| `loss.temperature` | float | `0.1` | Softmax temperature (`supcon` / `infonce`) |
| `loss.margin` | float | `0.2` | Margin (`triplet`) |
| `sampler.kind` | str | `pk` | Batch sampler: `pk` (P groups x K crops), `within_video` (one video per batch) or `random` (for `aug_view`) |
| `sampler.groups_per_batch` | int | `8` | P: groups per batch |
| `sampler.samples_per_group` | int | `16` | K: crops per group (batch size is P x K, doubled by the two views) |
| `use_projection` | bool | `true` | Add a train-only projection head (`supcon` / `infonce`), dropped at inference |
| `projection_dim` | int | `128` | Width of the projection head |

### bbox_centernet

`center`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `categories` | list | `null` | Class names, one heatmap channel each. `null` = filled at training from the training boxes (uncategorized boxes are `object`) |
| `sigma` | float | `null` | Fixed Gaussian sigma in input pixels. `null` = CenterNet's size-adaptive radius |
| `min_overlap` | float | `0.7` | IoU used by the size-adaptive radius |
| `output_stride` | int | `4` | Heatmap stride; must equal the `size` and `offset` strides |
| `loss_weight` | float | `1.0` | Weight of the heatmap focal loss |
| `focal_alpha` | float | `2.0` | Focal exponent on the prediction error |
| `focal_beta` | float | `4.0` | Exponent of the penalty reduction around GT peaks |

`size` / `offset`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output_stride` | int | `4` | Must equal the `center` stride |
| `loss_weight` | float | `0.1` (`size`) / `1.0` (`offset`) | Weight of the L1 loss |

### bbox_ssd.ssd

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `categories` | list | `null` | Class names (background excluded). `null` = filled at training from the training boxes |
| `output_stride` | int | `8` | Stride of the finest pyramid level |
| `num_levels` | int | `3` | Pyramid levels at strides `output_stride * 2**i`; the backbone `max_stride` is raised to the deepest one if needed |
| `anchor_sizes` | list | `null` | Base anchor size per level (input pixels). `null` = sized at training from the training boxes and saved into the config |
| `anchor_scales` | list | `[1.0, 1.26, 1.587]` | Multipliers on each level's base size |
| `aspect_ratios` | list | `[0.5, 1.0, 2.0]` | Anchor `h / w` ratios |
| `pos_iou_threshold` | float | `0.5` | Anchors with IoU >= this with a box are positives (each box's best anchor always is) |
| `neg_iou_threshold` | float | `0.4` | Anchors whose best IoU is below this are background; in between are ignored |
| `neg_pos_ratio` | float | `3.0` | Hard-negative mining: background anchors kept per positive |
| `box_variances` | list | `[0.1, 0.1, 0.2, 0.2]` | SSD `(cx, cy, w, h)` encoding variances |
| `head_channels` | int | `128` | Channels of each level's conv tower |
| `head_convs` | int | `1` | 3x3 conv + ReLU layers per tower |
| `loss_weight` | float | `1.0` | Weight of the box regression loss relative to classification |

### Dinov2WindowedConfig

RF-DETR's windowed DINOv2 (`backbone_config.dinov2_windowed`, `bbox_detr` only).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `in_channels` | int | `3` | Input channels |
| `hidden_size` | int | `384` | Token width: 384 (ViT-S) or 768 (ViT-B) |
| `num_layers` | int | `12` | Transformer blocks |
| `num_heads` | int | `6` | Attention heads (6 for ViT-S, 12 for ViT-B) |
| `mlp_ratio` | float | `4.0` | MLP width as a multiple of `hidden_size` |
| `patch_size` | int | `16` | Patch size (the token stride) |
| `pos_embed_grid` | int | `36` | Side of the token grid the learned position embedding covers (training size / `patch_size`); interpolated for other inputs |
| `out_indices` | list[int] | `[3, 6, 9, 12]` | Tapped hidden states (`0` = embedding output, `i` = output of block `i`) |
| `num_windows` | int | `2` | Attention windows per side (`1` = global attention). Input sides must be multiples of `patch_size * num_windows` |
| `window_block_indexes` | list[int] | `null` | Blocks (0-based) that attend within windows; `null` = every block up to the last tap except the tapped ones (RF-DETR) |
| `drop_path_rate` | float | `0.0` | Stochastic depth, ramped over the blocks |
| `weights` | str | `null` | Initialize from a plain DINOv2 checkpoint (HF id such as `facebook/dinov2-small`, or a directory): patch embedding resampled to `patch_size`, position embedding resized to `pos_embed_grid`. `null` = random init |
| `revision` | str | `null` | HF Hub revision of `weights` |
| `output_stride` | int | `16` | Stride of the token maps (`= patch_size`) |
| `max_stride` | int | `32` | Inputs are padded to multiples of this (`patch_size * num_windows`); also the multi-scale step |

### bbox_detr.detr

The DETR detector after the backbone (`head_configs.bbox_detr.detr`). Defaults are
RF-DETR's; the [guide](../guides/bbox-detr.md) gives animal defaults.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `categories` | list | `null` | Class names; `null` = the training boxes' categories in first-seen order (`object` for boxes without one). The class head has `len(categories) + 1` sigmoid logits; the last is never a positive |
| `output_stride` | int | `16` | Stride of the backbone's token maps |
| `hidden_dim` | int | `256` | Width of the projected map and the queries |
| `num_queries` | int | `300` | Queries per group: the most boxes per frame. Needs at least this many tokens in every input (training sizes with fewer are dropped from multi-scale) |
| `group_detr` | int | `13` | Query groups in training (Group DETR); inference uses one |
| `dec_layers` | int | `3` | Decoder layers |
| `sa_nheads` | int | `8` | Decoder self-attention heads |
| `ca_nheads` | int | `16` | Deformable cross-attention heads |
| `dec_n_points` | int | `2` | Sampling points per cross-attention head |
| `dim_feedforward` | int | `2048` | Decoder FFN width |
| `dropout` | float | `0.0` | Decoder dropout |
| `c2f_num_blocks` | int | `3` | Bottlenecks in the projector's C2f block |
| `c2f_expansion` | float | `0.5` | C2f hidden width as a fraction of `hidden_dim` |
| `num_select` | int | `300` | Detections kept per frame (top-k over queries x classes) |
| `cls_loss_coef` | float | `1.0` | Weight of the IoU-aware BCE class loss |
| `bbox_loss_coef` | float | `5.0` | Weight of the L1 box loss |
| `giou_loss_coef` | float | `2.0` | Weight of the GIoU loss |
| `set_cost_class` | float | `2.0` | Matcher weight of the focal class cost |
| `set_cost_bbox` | float | `5.0` | Matcher weight of the L1 box cost |
| `set_cost_giou` | float | `2.0` | Matcher weight of the GIoU cost |
| `focal_alpha` | float | `0.25` | Alpha of the matcher's focal cost and of the IA-BCE soft target |
| `focal_gamma` | float | `2.0` | Gamma of the matcher's focal cost and of the IA-BCE negative weight |
| `lr_component_decay` | float | `0.7` | The decoder trains at `lr * lr_component_decay`, the backbone at `lr * encoder_lr_scale * lr_component_decay^2` (times its layer decay), everything else at `lr` |
