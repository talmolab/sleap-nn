# Running Inference

Run predictions on videos and label files.

!!! info "Using uv workflow"
    - If using `uvx`, no installation needed
    - If using `uv sync`, prefix commands with `uv run`:
      ```bash
      uv run sleap-nn track ...
      ```

---

## Basic Inference

```bash
sleap-nn track --data_path video.mp4 --model_paths models/my_model/
```

Output: `video.mp4.predictions.slp`

---

## CLI Arguments

### Essential Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` / `-i` | Video or labels file | Required |
| `--model_paths` / `-m` | Model directory (repeat for top-down) | Required* |
| `--output_path` / `-o` | Output file path | `<input>.predictions.slp` |
| `--device` / `-d` | Device (auto/cuda/cpu/mps) | `auto` |
| `--batch_size` / `-b` | Frames per batch | `4` |
| `--max_instances` / `-n` | Max instances per frame | `None` |
| `--tracking` / `-t` | Enable tracking | `False` |
| `--peak_threshold` | Min confidence for peaks | `0.2` |

*Not required for track-only mode.

### Image Pre-processing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_height` | Pad to this height | From training |
| `--max_width` | Pad to this width | From training |
| `--input_scale` | Scale factor | From training |
| `--ensure_rgb` | Force 3 channels | `False` |
| `--ensure_grayscale` | Force 1 channel | `False` |
| `--crop_size` | Crop size (original coords) | From training |
| `--anchor_part` | Centroid anchor node | From training |

!!! warning "Breaking Change in v0.1.0: crop_size"
    `crop_size` now represents size in **original image coordinates** (crop-first-then-resize).

    Migration: If using `crop_size=128` with `input_scale=0.5`, use `crop_size=256`.

### Data Selection

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--frames` | Frame indices (e.g., `0-100,200-300`) | All |
| `--video_index` | Video index in multi-video .slp | `None` |
| `--only_labeled_frames` | Only labeled frames | `False` |
| `--only_suggested_frames` | Only suggested frames | `False` |
| `--exclude_user_labeled` | Skip user-labeled | `False` |
| `--no_empty_frames` | Remove empty frames | `False` |

---

## Model Types

### Single Instance / Bottom-Up

```bash
sleap-nn track -i video.mp4 -m models/bottomup/
```

### Top-Down

Provide both models:

```bash
sleap-nn track -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/
```

---

## Filtering Overlapping Instances

Remove duplicate detections with greedy NMS:

```bash
# Enable with default IOU
sleap-nn track -i video.mp4 -m models/ --filter_overlapping

# Use OKS with custom threshold
sleap-nn track -i video.mp4 -m models/ \
    --filter_overlapping \
    --filter_overlapping_method oks \
    --filter_overlapping_threshold 0.5
```

| Method | Description |
|--------|-------------|
| `iou` | Bounding box intersection-over-union |
| `oks` | Object Keypoint Similarity (pose-aware) |

| Threshold | Effect |
|-----------|--------|
| 0.3 | Aggressive filtering |
| 0.5 | Moderate |
| 0.8 | Permissive (default) |

---

## Python API

### Basic Usage

```python
from sleap_nn.predict import run_inference

labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/my_model/"],
    output_path="predictions.slp",
    make_labels=True,
)
```

### Get Raw Outputs

```python
results = run_inference(
    data_path="video.mp4",
    model_paths=["models/my_model/"],
    make_labels=False,
    return_confmaps=True,
)
```

---

## Provenance Metadata

Output files include metadata about how predictions were generated:

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
provenance = labels.provenance

print(f"sleap-nn version: {provenance.get('sleap_nn_version')}")
print(f"Model type: {provenance.get('model_type')}")
print(f"Runtime: {provenance.get('runtime_sec')}s")
```

### Recorded Information

| Category | Fields |
|----------|--------|
| **Timestamps** | `timestamp_start`, `timestamp_end`, `runtime_sec` |
| **Versions** | `sleap_nn_version`, `sleap_io_version`, `torch_version` |
| **Model** | `model_paths`, `model_type`, `head_type` |
| **Input** | `source_path`, `source_video_paths` |
| **Config** | `peak_threshold`, `batch_size`, `max_instances` |
| **System** | `device`, `python_version`, `cuda_version`, `gpu_names` |

---

## Legacy SLEAP Model Support

Run inference with SLEAP <=v1.4 models (UNet only):

```bash
sleap-nn track -i video.mp4 -m /path/to/sleap_model/
```

The directory should contain:
- `best_model.h5`
- `training_config.json`

!!! warning "UNet only"
    Only UNet backbone models from SLEAP ≤1.4 are supported.

---

## Troubleshooting

??? question "Out of memory"
    Reduce batch size: `--batch_size 2`

??? question "Slow inference"
    - Increase batch size (if memory allows)
    - Use GPU: `--device cuda`
    - Consider [ONNX/TensorRT export](export.md)

??? question "Poor predictions"
    - Lower `--peak_threshold`
    - Verify preprocessing matches training
    - Check model was trained on similar data

---

## Next Steps

- [:octicons-arrow-right-24: Evaluation](evaluation.md) - Assess model performance
- [:octicons-arrow-right-24: Tracking](tracking.md) - Assign IDs across frames
- [:octicons-arrow-right-24: Export](export.md) - Deploy models
