# Config Generator

Generate training configurations interactively or programmatically.

!!! warning "Experimental"
    This feature is experimental and may change in future releases.

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

---

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output path for config file(s) | `<slp_name>_config.yaml` |
| `--auto` | | Auto-generate without interactive TUI | `false` |
| `--view` | | View type for skeleton: `side` or `front` | `side` |
| `--pipeline` | | Pipeline: `single_instance`, `bottomup`, or `topdown` | Auto-detected |
| `--batch-size` | | Override batch size | Auto-detected |
| `--max-epochs` | | Override max epochs | `200` |
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

# Custom batch size and epochs
sleap-nn config labels.slp --auto --batch-size 8 --max-epochs 100

# Side-view animals (affects rotation augmentation)
sleap-nn config labels.slp --auto --view side
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
| `.auto(view=None)` | Auto-configure all parameters |
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

The recommendation is based on:

- Number of instances per frame
- Skeleton complexity
- Image size and resolution

---

## View Types

The `--view` option affects rotation augmentation:

| View | Rotation Range | Best For |
|------|----------------|----------|
| `side` | ±15° | Side-view cameras |
| `front` | ±180° | Top-down/overhead cameras |

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
