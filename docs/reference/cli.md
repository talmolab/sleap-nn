# CLI Reference

Complete command-line interface documentation.

!!! tip "Reading the tables"
    In the **Values** column:

    - **Flag** = No argument needed; presence enables the option (e.g., `--tracking`)
    - **`INT`** / **`FLOAT`** = Numeric value (e.g., `--batch_size 8`)
    - **`PATH`** = File or directory path
    - **Comma-separated values** = Choose one (e.g., `auto`, `cuda`, `cpu`)

---

## Commands

| Command | Description |
|---------|-------------|
| [`sleap-nn train`](#sleap-nn-train) | Train models |
| [`sleap-nn track`](#sleap-nn-track) | Run inference/tracking |
| [`sleap-nn eval`](#sleap-nn-eval) | Evaluate predictions |
| [`sleap-nn export`](#sleap-nn-export) | Export to ONNX/TensorRT |
| [`sleap-nn predict`](#sleap-nn-predict) | Inference on exported models |
| [`sleap-nn config`](#sleap-nn-config) | Generate training configs (experimental) |
| [`sleap-nn system`](#sleap-nn-system) | System diagnostics |

---

## `sleap-nn train`

Train pose estimation models.

```bash
sleap-nn train --config CONFIG_PATH [OPTIONS] [OVERRIDES]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | | Path to config YAML file |
| `--config-dir` | `-d` | Directory containing config file |
| `--config-name` | `-c` | Config file name (without .yaml) |
| `--video-paths` | `-v` | Replace video paths (multiple allowed) |
| `--video-path-map` | | Map old path to new (OLD NEW) |
| `--prefix-map` | | Map path prefix (OLD_PREFIX NEW_PREFIX) |

### Examples

```bash
# Simple
sleap-nn train --config config.yaml

# With directory/name
sleap-nn train -d /configs -c my_config

# Override values
sleap-nn train --config config.yaml trainer_config.max_epochs=200

# Remap video paths
sleap-nn train --config config.yaml --prefix-map /old/data /new/data
```

---

## `sleap-nn track`

Run inference and/or tracking.

```bash
sleap-nn track --data_path INPUT --model_paths MODEL [OPTIONS]
```

### Essential Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--data_path` | `-i` | Video or labels file | `PATH` | Required |
| `--model_paths` | `-m` | Model directory (multiple for top-down) | `PATH` | Required* |
| `--output_path` | `-o` | Output file path | `PATH` | `<input>.predictions.slp` |
| `--device` | `-d` | Compute device | `auto`, `cuda`, `cuda:0`, `cuda:1`, `cpu`, `mps` | `auto` |
| `--batch_size` | `-b` | Batch size | `INT` | `4` |
| `--tracking` | `-t` | Enable tracking | Flag | `false` |

*Not required for track-only mode.

### Output Options

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--gui` | Output JSON progress for GUI integration | Flag | `false` |

### Data Selection

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--frames` | Frame indices to process | `0-100`, `0-100,200-300` | All frames |
| `--video_index` | Video index in multi-video .slp | `INT` | `0` |
| `--only_labeled_frames` | Only process labeled frames | Flag | `false` |
| `--only_suggested_frames` | Only process suggested frames | Flag | `false` |
| `--exclude_user_labeled` | Skip user-labeled frames | Flag | `false` |
| `--only_predicted_frames` | Only process frames with predictions | Flag | `false` |

### Filtering

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--filter_overlapping` | Remove duplicate instances | Flag | `false` |
| `--filter_overlapping_method` | Overlap calculation method | `iou`, `oks` | `iou` |
| `--filter_overlapping_threshold` | Similarity threshold for filtering | `FLOAT` (0.0-1.0) | `0.8` |
| `--max_instances` | Max instances per frame (forward pass only) | `INT` | None |

### Tracking

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--tracking` | Enable tracking | Flag | `false` |
| `--tracking_window_size` | Frames to look back | `INT` | `5` |
| `--candidates_method` | Candidate selection method | `fixed_window`, `local_queues` | `fixed_window` |
| `--features` | Features for matching | `keypoints`, `centroids`, `bboxes`, `image` | `keypoints` |
| `--scoring_method` | Similarity scoring method | `oks`, `cosine_sim`, `iou`, `euclidean_dist` | `oks` |
| `--use_flow` | Enable optical flow | Flag | `false` |

### Examples

```bash
# Basic inference
sleap-nn track -i video.mp4 -m models/bottomup/

# Top-down (two models)
sleap-nn track -i video.mp4 -m models/centroid/ -m models/instance/

# With tracking
sleap-nn track -i video.mp4 -m models/bottomup/ -t

# Track-only (no inference)
sleap-nn track -i labels.slp -t

# Filter overlapping + tracking
sleap-nn track -i video.mp4 -m models/ --filter_overlapping -t
```

---

## `sleap-nn eval`

Evaluate predictions against ground truth.

```bash
sleap-nn eval --ground_truth_path GT --predicted_path PRED [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--ground_truth_path` | `-g` | Ground truth labels | `PATH` | Required |
| `--predicted_path` | `-p` | Predicted labels | `PATH` | Required |
| `--save_metrics` | `-s` | Save metrics to .npz | `PATH` | None |
| `--oks_stddev` | | OKS standard deviation | `FLOAT` | `0.05` |
| `--match_threshold` | | Instance matching threshold | `FLOAT` | `0.0` |
| `--user_labels_only` | | Only evaluate user-labeled | Flag | `false` |

### Example

```bash
sleap-nn eval -g ground_truth.slp -p predictions.slp -s metrics.npz
```

---

## `sleap-nn export`

Export models to ONNX/TensorRT.

```bash
sleap-nn export MODEL_PATH [MODEL_PATH_2] -o OUTPUT_DIR [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--output-dir` | `-o` | Output directory | `PATH` | Required |
| `--format` | `-f` | Export format | `onnx`, `tensorrt`, `both` | `onnx` |
| `--precision` | | TensorRT precision | `fp32`, `fp16` | `fp16` |
| `--max-instances` | `-n` | Max instances per frame | `INT` | `20` |
| `--max-batch-size` | `-b` | Max batch size | `INT` | `8` |

### Examples

```bash
# ONNX only
sleap-nn export models/bottomup -o exports/ --format onnx

# Both formats
sleap-nn export models/bottomup -o exports/ --format both

# Top-down (two models)
sleap-nn export models/centroid models/instance -o exports/
```

---

## `sleap-nn predict`

Run inference on exported models.

```bash
sleap-nn predict EXPORT_DIR INPUT_PATH [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--output` | `-o` | Output .slp path | `PATH` | `<input>.predictions.slp` |
| `--runtime` | `-r` | Inference runtime | `auto`, `onnx`, `tensorrt` | `auto` |
| `--batch-size` | `-b` | Batch size | `INT` | `4` |
| `--n-frames` | `-n` | Frames to process (0=all) | `INT` | `0` |
| `--device` | | Compute device | `auto`, `cuda`, `cpu` | `auto` |

### Example

```bash
sleap-nn predict exports/model video.mp4 -o predictions.slp --runtime tensorrt
```

---

## `sleap-nn config`

!!! warning "Experimental"
    This feature is experimental and may change in future releases.

Generate training configuration files interactively or automatically.

```bash
sleap-nn config SLP_PATH [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--output` | `-o` | Output path for config file(s) | `PATH` | `<slp_name>_config.yaml` |
| `--auto` | | Auto-generate without interactive TUI | Flag | `false` |
| `--pipeline` | | Model pipeline type | `single_instance`, `bottomup`, `topdown` | Auto-detected |
| `--batch-size` | | Override batch size | `INT` | Auto-detected |
| `--max-epochs` | | Override max epochs | `INT` | `200` |
| `--show-yaml` | | Print YAML to stdout instead of saving | Flag | `false` |

### Modes

#### Interactive TUI

Launch an interactive terminal UI to configure training:

```bash
sleap-nn config labels.slp
```

#### Auto Mode

Generate a config with smart defaults based on your data:

```bash
sleap-nn config labels.slp --auto -o config.yaml
```

### Examples

```bash
# Interactive configuration
sleap-nn config labels.slp

# Auto-generate with defaults
sleap-nn config labels.slp --auto

# Auto-generate with custom output
sleap-nn config labels.slp --auto -o my_config.yaml

# Auto-generate with overrides
sleap-nn config labels.slp --auto --pipeline bottomup --batch-size 8

# Preview YAML without saving
sleap-nn config labels.slp --auto --show-yaml
```

### Output

For **top-down** pipelines, two config files are generated:

- `<name>_centroid.yaml` - Centroid model config
- `<name>_centered_instance.yaml` - Centered instance model config

For other pipelines, a single config file is generated.

See the [Config Generator Guide](../guides/config-generator.md) for detailed usage.

---

## `sleap-nn system`

Display system information and GPU diagnostics.

```bash
sleap-nn system
```

Shows:
- Python version
- PyTorch version and build
- CUDA/cuDNN versions
- GPU details (name, memory, compute capability)
- Driver compatibility
- Installed package versions

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control visible GPUs |
| `WANDB_API_KEY` | WandB API key |

---

## Global Options

```bash
sleap-nn --version  # Show version
sleap-nn --help     # Show help
```
