# CLI Reference

Complete command-line interface documentation.

---

## Commands

| Command | Description |
|---------|-------------|
| [`sleap-nn train`](#sleap-nn-train) | Train models |
| [`sleap-nn track`](#sleap-nn-track) | Run inference/tracking |
| [`sleap-nn eval`](#sleap-nn-eval) | Evaluate predictions |
| [`sleap-nn export`](#sleap-nn-export) | Export to ONNX/TensorRT |
| [`sleap-nn predict`](#sleap-nn-predict) | Inference on exported models |
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

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--data_path` | `-i` | Video or labels file | Required |
| `--model_paths` | `-m` | Model directory (multiple for top-down) | Required* |
| `--output_path` | `-o` | Output file path | `<input>.predictions.slp` |
| `--device` | `-d` | Device (auto/cuda/cpu/mps) | `auto` |
| `--batch_size` | `-b` | Batch size | `4` |
| `--tracking` | `-t` | Enable tracking | `false` |

*Not required for track-only mode.

### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gui` | Output JSON progress for GUI integration (instead of Rich progress bar) | `false` |

### Data Selection

| Option | Description |
|--------|-------------|
| `--frames` | Frame indices (e.g., `0-100,200-300`) |
| `--video_index` | Video index in multi-video .slp |
| `--only_labeled_frames` | Only labeled frames |
| `--only_suggested_frames` | Only suggested frames |
| `--exclude_user_labeled` | Skip user-labeled frames |
| `--only_predicted_frames` | Only frames with predictions |

### Filtering

| Option | Description | Default |
|--------|-------------|---------|
| `--filter_overlapping` | Remove duplicate instances | `false` |
| `--filter_overlapping_method` | `iou` or `oks` | `iou` |
| `--filter_overlapping_threshold` | Similarity threshold | `0.8` |
| `--max_instances` | Maximum instances per frame | None |

### Tracking

| Option | Description | Default |
|--------|-------------|---------|
| `--tracking` | Enable tracking | `false` |
| `--tracking_window_size` | Frames to look back | `5` |
| `--candidates_method` | `fixed_window` or `local_queues` | `fixed_window` |
| `--features` | `keypoints`/`centroids`/`bboxes`/`image` | `keypoints` |
| `--scoring_method` | `oks`/`cosine_sim`/`iou`/`euclidean_dist` | `oks` |
| `--use_flow` | Enable optical flow | `false` |

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

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--ground_truth_path` | `-g` | Ground truth labels | Required |
| `--predicted_path` | `-p` | Predicted labels | Required |
| `--save_metrics` | `-s` | Save metrics to .npz | None |
| `--oks_stddev` | | OKS standard deviation | `0.05` |
| `--match_threshold` | | Instance matching threshold | `0.0` |
| `--user_labels_only` | | Only evaluate user-labeled | `false` |

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

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-dir` | `-o` | Output directory | Required |
| `--format` | `-f` | `onnx`/`tensorrt`/`both` | `onnx` |
| `--precision` | | TensorRT: `fp32`/`fp16` | `fp16` |
| `--max-instances` | `-n` | Max instances per frame | `20` |
| `--max-batch-size` | `-b` | Max batch size | `8` |

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

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output .slp path | `<input>.predictions.slp` |
| `--runtime` | `-r` | `auto`/`onnx`/`tensorrt` | `auto` |
| `--batch-size` | `-b` | Batch size | `4` |
| `--n-frames` | `-n` | Frames to process (0=all) | `0` |
| `--device` | | `auto`/`cuda`/`cpu` | `auto` |

### Example

```bash
sleap-nn predict exports/model video.mp4 -o predictions.slp --runtime tensorrt
```

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
