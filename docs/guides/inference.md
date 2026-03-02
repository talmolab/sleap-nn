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

See the [CLI Reference](../reference/cli.md) for all available parameters.

---

## Essential Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--data_path` / `-i` | Video or labels file | `PATH` | Required |
| `--model_paths` / `-m` | Model directory (repeat for top-down) | `PATH` | Required* |
| `--output_path` / `-o` | Output file path | `PATH` | `<input>.predictions.slp` |
| `--device` / `-d` | Compute device | `auto`, `cuda`, `cuda:0`, `cpu`, `mps` | `auto` |
| `--batch_size` / `-b` | Frames per batch | `INT` | `4` |
| `--max_instances` / `-n` | Max instances per frame (forward pass only) | `INT` | `None` |
| `--tracking` / `-t` | Enable tracking | Flag | `False` |
| `--peak_threshold` | Min confidence for peaks | `FLOAT` | `0.2` |

*Not required for track-only mode.

!!! tip "Device selection"
    The `--device` parameter accepts:

    - `auto` - Automatically detect best available device
    - `cuda` - Use default CUDA GPU
    - `cuda:0`, `cuda:1`, etc. - Use a specific GPU by index
    - `cpu` - Use CPU
    - `mps` - Use Apple Metal (macOS)

!!! note "Max instances"
    The `--max_instances` / `-n` parameter is only applied during the model forward pass. When running in **track-only mode** (without model inference), this parameter has no effect since no peak detection is performed.

For all parameters including image pre-processing and data selection options, see the [CLI Reference](../reference/cli.md).

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

!!! note "Inference only"
    `--filter_overlapping` is only applied during inference. When running in **track-only mode** (without model paths), this parameter has no effect.

---

## Processing Order

When running inference with tracking enabled, operations are applied in this order:

```
1. Model Forward Pass
   └── --max_instances applied (limits detections during peak finding)

2. Filtering (before tracking)
   └── --filter_overlapping applied (removes duplicate instances)

3. Tracking
   └── Instance identity assignment across frames
```

!!! note "Why filtering happens before tracking"
    Filtering is applied **before** tracking to prevent spurious track creation. If duplicates were removed after tracking, the tracker would assign IDs to instances that are later filtered out, causing track switches in subsequent frames.

In **track-only mode** (no model paths), filtering is still applied before tracking on existing predictions.

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
