# Config Generator

Generate training configurations interactively or programmatically.

!!! warning "Experimental"
    This feature is experimental and may change in future releases.

!!! tip "Web-based Config Picker"
    Prefer a browser-based interface? Try the [Config Picker](../configuration/config-picker/index.md) to generate configs interactively without installing anything.

---

## Overview

The `sleap-nn config` command helps you create training configuration files for your SLEAP label files. It can:

- Analyze your data and recommend optimal settings
- Generate configs via an interactive TUI (Terminal User Interface)
- Auto-generate configs with sensible defaults

---

## Quick Start

### Interactive Mode (TUI)

Launch the interactive configuration wizard:

```bash
sleap-nn config labels.slp
```

The TUI guides you through:

1. Loading and analyzing your data
2. Selecting a model pipeline
3. Configuring training parameters
4. Exporting the configuration file

### Auto Mode

Generate a config with smart defaults based on your data:

```bash
sleap-nn config labels.slp --auto -o config.yaml
```

For top-down pipelines, this creates two config files:

- `config_centroid.yaml`
- `config_centered_instance.yaml`

The box-detector top-down pipelines (`--pipeline topdown_bbox_centernet` /
`topdown_bbox_ssd` / `topdown_bbox_detr`) write `config_bbox_centernet.yaml` (or
`config_bbox_ssd.yaml`, `config_bbox_detr.yaml`) and a `config_centered_instance.yaml`
with box-scaled crops (`crop_mode: bbox`); see
[Top-down pose with a box detector](bbox-topdown.md).

---

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output path for config file(s) | `<slp_name>_config.yaml` |
| `--auto` | | Auto-generate without interactive TUI | `false` |
| `--pipeline` | | Pipeline: `single_instance`, `bottomup`, `topdown`, `multi_class_bottomup`, `multi_class_topdown`, `centroid`, `bbox_centernet`, `bbox_ssd`, `bbox_detr`, `topdown_bbox_centernet`, `topdown_bbox_ssd` or `topdown_bbox_detr` | Auto-detected |
| `--gpu_memory_gb` | | GPU memory (GB) of the training machine, or `auto`; lets the recommender pick `topdown_bbox_detr` when it fits | Not considered |
| `--show-yaml` | | Print YAML to stdout instead of saving | `false` |

---

## Examples

### Basic Auto-Generation

```bash
# Generate with all defaults
sleap-nn config labels.slp --auto

# Specify output path
sleap-nn config labels.slp --auto -o my_config.yaml

# Preview without saving
sleap-nn config labels.slp --auto --show-yaml
```

### Override Settings

```bash
# Force bottom-up pipeline
sleap-nn config labels.slp --auto --pipeline bottomup
```

### Train with Generated Config

```bash
# Generate config
sleap-nn config labels.slp --auto -o config.yaml

# Train model
sleap-nn train config.yaml
```

---

## Python API

The `ConfigGenerator` class provides a fluent API for programmatic configuration:

### Basic Usage

```python
from sleap_nn.config_generator import ConfigGenerator

# Auto-configure based on data
config = ConfigGenerator.from_slp("labels.slp").auto().build()

# Save to file
ConfigGenerator.from_slp("labels.slp").auto().save("config.yaml")
```

### Customization

```python
from sleap_nn.config_generator import ConfigGenerator

# Chain customizations
config = (
    ConfigGenerator.from_slp("labels.slp")
    .auto()
    .pipeline("bottomup")
    .batch_size(8)
    .max_epochs(100)
    .sigma(3.0)
    .build()
)
```

### Available Methods

| Method | Description |
|--------|-------------|
| `.auto()` | Auto-configure all parameters |
| `.pipeline(type)` | Set pipeline type |
| `.backbone(type)` | Set backbone architecture |
| `.batch_size(n)` | Set batch size |
| `.max_epochs(n)` | Set max training epochs |
| `.learning_rate(lr)` | Set learning rate |
| `.input_scale(scale)` | Set input scaling (0.0-1.0) |
| `.sigma(sigma)` | Set confidence map sigma |
| `.rotation(min, max)` | Set rotation augmentation range |
| `.early_stopping(enabled, patience)` | Configure early stopping |
| `.crop_size(size)` | Set crop size (centered_instance) |
| `.anchor_part(name)` | Set anchor part (top-down) |
| `.centroid_method(method, fallback=None)` | How the centroid / crop center is derived: `center_of_mass`, `bbox_center`, `geometric_median`, `anchor` (#586) |
| `.detr(preset, num_queries, init_from, max_side)` | `bbox_detr` backbone / decoder preset (`dinov3_s`, `dinov3_b`, `rfdetr_n\|s\|m\|l`), queries, a checkpoint to start from, and a fixed input side |
| `.detr_resolution()` | The `bbox_detr` input size the token-width rule picks for the data |

### Get Recommendations

```python
from sleap_nn.config_generator import ConfigGenerator

gen = ConfigGenerator.from_slp("labels.slp")

# View dataset statistics
print(gen.stats)

# Get recommendations without building
rec = gen.recommend()
print(f"Recommended pipeline: {rec.pipeline.recommended}")
print(f"Reason: {rec.pipeline.reason}")

# Get memory estimate
mem = gen.memory_estimate()
print(f"Estimated GPU memory: {mem.total_gpu_gb:.1f} GB")

# Full summary
print(gen.auto().summary())
```

---

## Pipeline Selection

The config generator analyzes your data to recommend a pipeline:

| Pipeline | Use Case |
|----------|----------|
| `single_instance` | One animal per frame |
| `bottomup` | Multiple animals, same skeleton |
| `topdown` | Multiple animals, need centroids + instances |
| `multi_class_bottomup` | Multiple animals with identity classes |
| `multi_class_topdown` | Multiple animals with classes, top-down approach |
| `bbox_centernet` / `bbox_ssd` / `bbox_detr` | A bounding-box detector (boxes derived from the poses) |
| `topdown_bbox_centernet` / `topdown_bbox_ssd` / `topdown_bbox_detr` | Top-down with a box detector as stage 1 and box-scaled crops |

**`bbox_detr`** is recommended (as `topdown_bbox_detr`) for multi-animal top-down
data when the recommender knows the GPU and the default DETR run fits in 80% of
it: always in the TUI (it reads the GPU), with `--gpu_memory_gb` on the CLI,
`gpu_memory_gb=` in `auto()` / `recommend()`. Otherwise the centroid pair stays the
recommendation, `topdown_bbox_detr` is listed as an alternative, and a warning
suggests it when animals touch in at least 5% of frames (GT boxes intersect):
DETR has no NMS step that could merge touching animals. Its defaults (DINOv3
ViT-S, 100 queries, the input size that keeps small animals 3 tokens wide, 200
epochs) and the presets' parameter / memory estimates are in the
[DETR box detector guide](bbox-detr.md).

The recommendation is based on:

- Number of instances per frame
- Skeleton complexity
- Image size and resolution

---

## Output Files

### Single-Animal or Bottom-Up Pipelines

A single config file is generated:

```
config.yaml
```

### Top-Down Pipelines

Two config files are generated:

```
config_centroid.yaml
config_centered_instance.yaml
```

Train both models:

```bash
sleap-nn train config_centroid.yaml
sleap-nn train config_centered_instance.yaml
```

---

## Tips

- Start with `--auto` mode and adjust from there
- Use `--show-yaml` to preview before committing
- Check memory estimates before training on large datasets
- For top-down models, train the centroid model first
