# Model Types

Understand the different model architectures in SLEAP-NN.

---

## Overview

| Model Type | Animals | Occlusion | Training | Use Case |
|------------|---------|-----------|----------|----------|
| Single Instance | 1 | N/A | 1 model | Isolated animals |
| Top-Down | Many | Some | 2 models | Multiple non-overlapping and animal sizes are smaller compared to the whole image |
| Bottom-Up | Many | Heavy | 1 model | Crowded scenes |
| Multi-Class | Many | Varies | 1-2 models | Known identities |

---

## Single Instance

**One animal per frame.**

```
Image → Backbone → Confidence Maps → Peaks
```

### When to Use

- Single animal videos
- No need for tracking
- Fastest training and inference

### Configuration

```yaml
head_configs:
  single_instance:
    confmaps:
      sigma: 5.0
      output_stride: 2
```

### Inference

```bash
sleap-nn track -i video.mp4 -m models/single_instance/
```

---

## Top-Down {#top-down}

**Two-stage: detect centers, then estimate pose.**

```
Stage 1: Image → Backbone → Centroid Map → Instance Centers
Stage 2: Crop → Backbone → Confidence Maps → Keypoints
```

### When to Use

- Multiple animals
- Animals not heavily overlapping
- Need precise localization

### Configuration

**Centroid model:**
```yaml
head_configs:
  centroid:
    confmaps:
      anchor_part: null  # Use bbox center
      sigma: 5.0
```

**Instance model:**
```yaml
head_configs:
  centered_instance:
    confmaps:
      anchor_part: null
      sigma: 5.0

data_config:
  preprocessing:
    crop_size: 256
```

### Training

Train two models separately:

```bash
sleap-nn train --config centroid_config.yaml
sleap-nn train --config instance_config.yaml
```

### Inference

```bash
sleap-nn track -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/
```

---

## Bottom-Up {#bottom-up}

**Detect all keypoints, then group into instances.**

```
Image → Backbone → [Confidence Maps + Part Affinity Fields] → Grouping → Instances
```

### When to Use

- Multiple animals
- Heavy occlusion or overlap
- Animals touching/interacting

### How It Works

1. **Confidence maps**: Locate all keypoints of all animals
2. **Part Affinity Fields (PAFs)**: Encode connections between keypoints
3. **Grouping**: Hungarian matching to assemble instances

### Configuration

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

### Inference

```bash
sleap-nn track -i video.mp4 -m models/bottomup/
```

---

## Multi-Class (Identity Models)

**Pose estimation + supervised identity prediction.**

Use when you have labeled identity/track information in training data.

### Multi-Class Bottom-Up

```yaml
head_configs:
  multi_class_bottomup:
    confmaps:
      sigma: 5.0
      loss_weight: 1.0
    class_maps:
      classes: null  # Infer from track names
      sigma: 5.0
      loss_weight: 1.0
```

### Multi-Class Top-Down

```yaml
head_configs:
  multi_class_topdown:
    confmaps:
      sigma: 5.0
      loss_weight: 1.0
    class_vectors:
      classes: null
      num_fc_layers: 1
      num_fc_units: 64
      loss_weight: 1.0
```

---

## Backbones

### UNet

- Most flexible
- Works at any resolution
- Configurable depth/width

```yaml
backbone_config:
  unet:
    filters: 32
    max_stride: 16
```

### ConvNeXt

- Modern CNN architecture
- ImageNet pretrained weights
- Good for transfer learning

```yaml
backbone_config:
  convnext:
    model_type: tiny
    pre_trained_weights: ConvNeXt_Tiny_Weights
```

### Swin Transformer

- Vision transformer
- Best for global context
- Highest memory usage

```yaml
backbone_config:
  swint:
    model_type: tiny
    pre_trained_weights: Swin_T_Weights
```

---

## Choosing a Model

```mermaid
graph TD
    A[How many animals?] -->|One| B[Single Instance]
    A -->|Multiple| C[Do they overlap?]
    C -->|No/Rarely| D[Top-Down]
    C -->|Yes/Often| E[Bottom-Up]
    D --> F[Need identity?]
    E --> F
    F -->|Yes| G[Multi-Class variant]
    F -->|No| H[Standard variant]
```

### Quick Guidelines

| Scenario | Recommendation |
|----------|----------------|
| Single fly in chamber | Single Instance |
| 2-3 mice, separate areas | Top-Down |
| Social behavior, touching | Bottom-Up |
| Same individuals across sessions | Multi-Class |

---

## Performance Comparison

Approximate training times on RTX 3090 (1000 labeled frames):

| Model | Training Time | Inference Speed |
|-------|--------------|-----------------|
| Single Instance | ~30 min | ~500 FPS |
| Top-Down | ~1 hr (2 models) | ~100 FPS |
| Bottom-Up | ~1 hr | ~80 FPS |

---

## Tips

!!! tip "Start simple"
    Try Single Instance or Top-Down first. Only use Bottom-Up if needed.

!!! tip "More data helps Bottom-Up"
    PAF learning benefits from diverse poses and interactions.

!!! tip "Anchor part for Top-Down"
    Set `anchor_part` to a reliable body part (e.g., "thorax") for better cropping.
