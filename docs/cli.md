# CLI Reference

This page provides a quick reference for all `sleap-nn` command-line interface (CLI) commands. For detailed usage and examples, see the linked guides.

## Commands Overview

| Command | Description | Guide |
|---------|-------------|-------|
| [`sleap-nn train`](#sleap-nn-train) | Train pose estimation models | [Training Guide](training.md) |
| [`sleap-nn track`](#sleap-nn-track) | Run inference and/or tracking | [Inference Guide](inference.md) |
| [`sleap-nn eval`](#sleap-nn-eval) | Evaluate predictions against ground truth | [Inference Guide](inference.md#evaluation-metrics) |
| [`sleap-nn export`](#sleap-nn-export) | Export models to ONNX/TensorRT | [Export Guide](export.md) |
| [`sleap-nn predict`](#sleap-nn-predict) | Run inference on exported models | [Export Guide](export.md#sleap-nn-predict) |
| [`sleap-nn system`](#sleap-nn-system) | Display system info and GPU diagnostics | - |

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show sleap-nn version and exit |

```bash
sleap-nn --version
# Output: sleap-nn 0.1.0a0
```

---

## `sleap-nn train`

Train SLEAP models using a configuration file.

```bash
sleap-nn train --config-dir <dir> --config-name <name> [options] [overrides]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config-name` | `-c` | Name of the config file (without `.yaml` extension) |
| `--config-dir` | `-d` | Path to the directory containing the config file |
| `--video-paths` | `-v` | Video paths to replace existing paths in labels (can be specified multiple times) |
| `--video-path-map` | | Map old video path to new path (takes two arguments: OLD NEW) |
| `--prefix-map` | | Map old path prefix to new prefix (takes two arguments: OLD NEW) |

### Config Overrides

Override any configuration value directly from the command line:

```bash
# Change max epochs
sleap-nn train -c config -d /path/to/dir trainer_config.max_epochs=100

# Change batch size
sleap-nn train -c config -d /path/to/dir trainer_config.train_data_loader.batch_size=8

# Set training data path
sleap-nn train -c config -d /path/to/dir "data_config.train_labels_path=[train.slp]"

# Resume from checkpoint
sleap-nn train -c config -d /path/to/dir trainer_config.resume_ckpt_path=/path/to/best.ckpt
```

### Video Path Remapping

When training on a different machine than where labels were created:

```bash
# Replace paths by order
sleap-nn train -c config -d /path/to/dir \
    --video-paths /new/path/video1.mp4 \
    --video-paths /new/path/video2.mp4

# Map specific paths
sleap-nn train -c config -d /path/to/dir \
    --video-path-map /old/video.mp4 /new/video.mp4

# Replace path prefixes (updates all matching videos)
sleap-nn train -c config -d /path/to/dir \
    --prefix-map /old/server/data /new/local/data
```

See [Remapping Video Paths](training.md#remapping-video-paths) for more details.

---

## `sleap-nn track`

Run inference on videos or labels files, with optional tracking.

```bash
sleap-nn track --data_path <input> --model_paths <model_dir> [options]
```

### Essential Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--data_path` | `-i` | Path to video (`.mp4`) or labels (`.slp`) file | Required |
| `--model_paths` | `-m` | Path to model checkpoint directory (can be specified multiple times for top-down) | Required* |
| `--output_path` | `-o` | Output path for predictions | `<input>.predictions.slp` |
| `--device` | `-d` | Device to use (`cpu`, `cuda`, `cuda:0`, `mps`, `auto`) | `auto` |
| `--batch_size` | `-b` | Frames per batch (higher = faster but more memory). Note: `--queue_maxsize` should be at least 2x batch size. | `4` |
| `--queue_maxsize` | | Maximum size of the frame buffer queue (should be at least 2x batch size) | `8` |
| `--max_instances` | `-n` | Maximum instances per frame | `None` |
| `--tracking` | `-t` | Enable tracking | `False` |

*Not required for track-only workflow on labeled data.

### Common Examples

```bash
# Basic inference
sleap-nn track -i video.mp4 -m models/bottomup/

# Top-down inference (two models)
sleap-nn track -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/

# Inference with tracking
sleap-nn track -i video.mp4 -m models/bottomup/ -t

# Specific frames only
sleap-nn track -i video.mp4 -m models/bottomup/ --frames 0-100,200-300

# Track-only (no inference, just assign tracks to existing labels)
sleap-nn track -i labels.slp -t
```

### All Options by Category

#### Model Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--backbone_ckpt_path` | Path to alternative backbone weights | `None` |
| `--head_ckpt_path` | Path to alternative head weights | `None` |

#### Image Pre-processing

| Option | Description | Default |
|--------|-------------|---------|
| `--max_height` | Pad image to this height | From training config |
| `--max_width` | Pad image to this width | From training config |
| `--input_scale` | Scale factor for input image | From training config |
| `--crop_size` | Crop size for centered-instance models | From training config |
| `--ensure_rgb` | Convert to RGB (3 channels) | `False` |
| `--ensure_grayscale` | Convert to grayscale (1 channel) | `False` |
| `--anchor_part` | Node name for centroid anchor | From training config |

#### Data Selection

| Option | Description | Default |
|--------|-------------|---------|
| `--frames` | Frame indices (e.g., `0-100,200-300`) | All frames |
| `--video_index` | Video index in `.slp` file | `None` |
| `--video_dataset` | Dataset name for HDF5 videos | `None` |
| `--video_input_format` | Input format for HDF5 videos | `channels_last` |
| `--only_labeled_frames` | Only run on labeled frames | `False` |
| `--only_suggested_frames` | Only run on suggested frames | `False` |
| `--exclude_user_labeled` | Skip frames with user-labeled instances | `False` |
| `--only_predicted_frames` | Only run on frames with existing predictions | `False` |
| `--no_empty_frames` | Remove empty frames from output | `False` |

#### Peak Detection

| Option | Description | Default |
|--------|-------------|---------|
| `--peak_threshold` | Minimum confidence for valid peaks | `0.2` |
| `--integral_refinement` | Refinement method (`integral` or `None`) | `integral` |
| `--integral_patch_size` | Patch size for integral refinement | `5` |

#### Filtering Overlapping Instances

| Option | Description | Default |
|--------|-------------|---------|
| `--filter_overlapping` | Enable filtering of overlapping instances | `False` |
| `--filter_overlapping_method` | Similarity method: `iou` (bbox) or `oks` (keypoints) | `iou` |
| `--filter_overlapping_threshold` | Similarity threshold (higher = less filtering) | `0.8` |

#### Tracking Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tracking` / `-t` | Enable tracking | `False` |
| `--tracking_window_size` | Frames to look back for candidates | `5` |
| `--candidates_method` | `fixed_window` or `local_queues` | `fixed_window` |
| `--features` | Feature type: `keypoints`, `centroids`, `bboxes`, `image` | `keypoints` |
| `--scoring_method` | Matching score: `oks`, `cosine_sim`, `iou`, `euclidean_dist` | `oks` |
| `--scoring_reduction` | Score reduction: `mean`, `max`, `robust_quantile` | `mean` |
| `--robust_best_instance` | Quantile for robust similarity (0-1, or 1 for max) | `1.0` |
| `--track_matching_method` | Matching algorithm: `hungarian`, `greedy` | `hungarian` |
| `--max_tracks` | Maximum number of tracks (local queues only) | `None` |
| `--min_new_track_points` | Minimum points to spawn new track | `0` |
| `--min_match_points` | Minimum points for match candidates | `0` |
| `--post_connect_single_breaks` | Connect single track breaks (local queues only) | `False` |

#### Optical Flow Tracking

| Option | Description | Default |
|--------|-------------|---------|
| `--use_flow` | Use optical flow for tracking | `False` |
| `--of_img_scale` | Image scale for optical flow (lower = faster) | `1.0` |
| `--of_window_size` | Window size at each pyramid level | `21` |
| `--of_max_levels` | Number of pyramid levels | `3` |

#### Bottom-Up Specific

| Option | Description | Default |
|--------|-------------|---------|
| `--max_edge_length_ratio` | Max edge length as fraction of image size | `0.25` |
| `--dist_penalty_weight` | Distance penalty weight | `1.0` |
| `--n_points` | Points to sample along line integral | `10` |
| `--min_instance_peaks` | Minimum peaks for valid instance | `0` |
| `--min_line_scores` | Minimum line score for PAF matching | `0.25` |

#### Legacy Tracking Options

!!! warning
    These options are provided for backwards compatibility and may be removed in future releases. Use `--max_instances` instead.

| Option | Description | Default |
|--------|-------------|---------|
| `--tracking_target_instance_count` | Target instances per frame | `0` |
| `--tracking_pre_cull_to_target` | Cull instances before tracking | `0` |
| `--tracking_pre_cull_iou_threshold` | IOU threshold for pre-cull | `0` |
| `--tracking_clean_instance_count` | Target instances after tracking | `0` |
| `--tracking_clean_iou_threshold` | IOU threshold for post-tracking cull | `0` |

See [Inference Guide](inference.md) for detailed examples and explanations.

---

## `sleap-nn eval`

Evaluate predictions against ground truth labels.

```bash
sleap-nn eval --ground_truth_path <gt.slp> --predicted_path <pred.slp> [options]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--ground_truth_path` | `-g` | Path to ground truth labels (`.slp`) | Required |
| `--predicted_path` | `-p` | Path to predicted labels (`.slp`) | Required |
| `--save_metrics` | `-s` | Path to save metrics (`.npz`) | `None` |
| `--oks_stddev` | | Standard deviation for OKS calculation | `0.05` |
| `--oks_scale` | | Scale factor for OKS calculation | `None` |
| `--match_threshold` | | Threshold for instance matching | `0.0` |
| `--user_labels_only` | | Only evaluate user-labeled frames | `False` |

### Example

```bash
sleap-nn eval \
    -g ground_truth.slp \
    -p predictions.slp \
    -s metrics.npz
```

See [Evaluation Metrics](inference.md#evaluation-metrics) for more details.

---

## `sleap-nn export`

!!! warning "Experimental"
    This command is experimental. See the [Export Guide](export.md) for details.

Export trained models to ONNX and/or TensorRT format for optimized inference.

```bash
sleap-nn export MODEL_PATH [MODEL_PATH_2] -o OUTPUT_DIR [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `MODEL_PATH` | Path to trained model checkpoint directory |
| `MODEL_PATH_2` | Second model path for top-down (centroid + instance) |

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-dir` | `-o` | Output directory for exported models | Required |
| `--format` | `-f` | Export format: `onnx`, `tensorrt`, or `both` | `onnx` |
| `--precision` | | TensorRT precision: `fp32` or `fp16` | `fp16` |
| `--max-instances` | `-n` | Maximum instances per frame | `20` |
| `--max-batch-size` | `-b` | Maximum batch size for dynamic shapes | `8` |
| `--input-scale` | | Input resolution scale factor | `1.0` |
| `--device` | | Device for export: `cuda` or `cpu` | `cuda` |

### Examples

```bash
# Export single model to ONNX
sleap-nn export models/single_instance -o exports/model --format onnx

# Export to both ONNX and TensorRT
sleap-nn export models/bottomup -o exports/model --format both

# Export top-down (combined centroid + instance)
sleap-nn export models/centroid models/centered_instance -o exports/topdown
```

See [Export Guide](export.md) for detailed documentation.

---

## `sleap-nn predict`

!!! warning "Experimental"
    This command is experimental. See the [Export Guide](export.md) for details.

Run inference on exported ONNX/TensorRT models.

```bash
sleap-nn predict EXPORT_DIR INPUT_PATH [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `EXPORT_DIR` | Path to exported model directory |
| `INPUT_PATH` | Path to video file |

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output path for predictions (`.slp`) | `<input>.predictions.slp` |
| `--runtime` | `-r` | Runtime: `auto`, `onnx`, or `tensorrt` | `auto` |
| `--batch-size` | `-b` | Inference batch size | `4` |
| `--n-frames` | `-n` | Number of frames to process (0 = all) | `0` |
| `--device` | | Device: `auto`, `cuda`, or `cpu` | `auto` |

### Examples

```bash
# Basic inference with auto-detected runtime
sleap-nn predict exports/model video.mp4 -o predictions.slp

# Use TensorRT for maximum speed
sleap-nn predict exports/model video.mp4 -o predictions.slp --runtime tensorrt

# Process first 1000 frames with batch size 8
sleap-nn predict exports/model video.mp4 -o predictions.slp -n 1000 -b 8
```

See [Export Guide](export.md#sleap-nn-predict) for detailed documentation.

---

## `sleap-nn system`

Display system information and GPU diagnostics. Useful for troubleshooting GPU issues and verifying your installation.

```bash
sleap-nn system
```

### Output Includes

- Python version and platform
- PyTorch version and build info
- CUDA/cuDNN versions with driver compatibility check
- GPU details (name, compute capability, memory)
- Functional GPU tests (tensor operations, convolution)
- Installed package versions (sleap-nn, sleap-io, torch, etc.)

### Example Output

```
System Information
==================
Python: 3.12.8
Platform: Linux-6.14.0-27-generic-x86_64-with-glibc2.39
PyTorch: 2.7.0+cu130

CUDA Available: Yes
CUDA Version: 13.0
cuDNN Version: 90800
Driver Version: 570.133.20
...
```

This command is particularly helpful when:

- Debugging GPU detection issues
- Verifying CUDA/cuDNN compatibility
- Reporting issues or asking for help

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control which GPUs are visible (e.g., `0,1`) |
| `WANDB_API_KEY` | Weights & Biases API key for logging |

## See Also

- [Installation Guide](installation.md) - Setup instructions
- [Configuration Guide](config.md) - Full config reference
- [Training Guide](training.md) - Detailed training documentation
- [Inference Guide](inference.md) - Detailed inference documentation
- [Export Guide](export.md) - Model export and fast inference
