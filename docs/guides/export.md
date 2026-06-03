# ONNX & TensorRT Export

Export models for high-performance production inference.

!!! warning "Experimental"
    Export functionality is experimental. Report issues at [github.com/talmolab/sleap-nn/issues](https://github.com/talmolab/sleap-nn/issues).

---

## Why Export?

| Format | Use Case | Speedup |
|--------|----------|---------|
| **ONNX** | Cross-platform, CPU inference | ~1x |
| **TensorRT FP16** | Maximum GPU throughput | 3-6x |

---

## Installation

Export requires additional dependencies. Install them with your preferred method:

=== "uv (recommended)"

    ```bash
    # ONNX export (CPU)
    uv tool install "sleap-nn[export]" --torch-backend auto

    # ONNX export (GPU runtime)
    uv tool install "sleap-nn[export-gpu]" --torch-backend auto

    # TensorRT export (Linux/Windows with NVIDIA GPU)
    uv tool install "sleap-nn[export-gpu,tensorrt]" --torch-backend auto
    ```

=== "pip"

    ```bash
    # ONNX export (CPU)
    pip install sleap-nn[export]

    # TensorRT export (NVIDIA GPUs)
    pip install sleap-nn[export-gpu,tensorrt]
    ```

!!! note "TensorRT availability"
    TensorRT is only available on Linux and Windows with NVIDIA GPUs.

---

## Quick Start

### Export

```bash
# ONNX only
sleap-nn export models/my_model -o exports/ --format onnx

# Both ONNX and TensorRT
sleap-nn export models/my_model -o exports/ --format both
```

### Run Inference

Exported-model inference goes through the unified `sleap-nn predict` command —
point `-m` at the export directory and it is auto-detected as an exported model:

```bash
sleap-nn predict -m exports/my_model -i video.mp4 -o predictions.slp
```

---

## Export Options

```bash
sleap-nn export MODEL_PATH [options]
```

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `-o`, `--output-dir` | Output directory | `PATH` | Required |
| `-f`, `--format` | Export format | `onnx`, `tensorrt`, `both` | `onnx` |
| `--precision` | TensorRT precision | `fp32`, `fp16` | `fp16` |
| `-n`, `--max-instances` | Max instances per frame | `INT` | `20` |
| `-b`, `--max-batch-size` | Max batch size | `INT` | `8` |

---

## Model Types

### Single Instance / Bottom-Up

```bash
sleap-nn export models/bottomup -o exports/bottomup --format both
```

### Top-Down

Export both models together:

```bash
sleap-nn export models/centroid models/centered_instance \
    -o exports/topdown --format both
```

### Multi-Class

```bash
sleap-nn export models/multi_class_bottomup \
    -o exports/multiclass --format onnx
```

### Standalone Centroid

A standalone centroid model (a first-class single-stage "animals-as-points"
pipeline, *not* a top-down stage-1) exports from a **single** directory. The
exported runtime reads the full training skeleton from `training_config.yaml`
and collapses the output to a single-node `'centroid'` skeleton, bit-for-bit
identical to the checkpoint flow.

```bash
# Export one centroid directory (NOT a centroid + centered_instance pair).
sleap-nn export models/centroid -o exports/centroid --format onnx

# Run the exported model via the unified predict command. Choose the output
# representation with --centroid-output:
#   instance (default) -> single-node PredictedInstance (frontend-compatible)
#   centroid           -> sio.PredictedCentroid
#   both               -> both
sleap-nn predict -m exports/centroid -i video.mp4 -o centroids.slp \
    --centroid-output instance
```

!!! note "One directory only"
    Passing two centroid directories raises an error — that pattern is almost
    always a mistake (you likely meant a centroid + centered-instance top-down
    bundle). Pass a single centroid directory for a standalone export.

See the [Centroid-Only Inference guide](centroid-only-inference.md) for the
full train → infer → eval → export workflow and the output representation
contract.

---

## Inference Options

Exported-model inference is run through the unified `sleap-nn predict` command.
When `-m`/`--model_paths` points at a directory containing `model.onnx` or
`model.trt`, `predict` auto-detects it as an exported model and runs the
exported runtime; otherwise it loads a trained checkpoint. Use `--runtime` to
choose between ONNX and TensorRT (it is ignored for checkpoints).

```bash
sleap-nn predict -m EXPORT_DIR -i VIDEO [options]
```

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `-m`, `--model_paths` | Export dir (auto-detected) or checkpoint dir | `PATH` | Required |
| `-i`, `--data_path` | Video or labels file | `PATH` | Required |
| `-o`, `--output_path` | Output path | `PATH` | `<input>.predictions.slp` |
| `--runtime` | Exported-model runtime (ignored for checkpoints) | `auto`, `onnx`, `tensorrt` | `auto` |
| `-b`, `--batch_size` | Batch size | `INT` | `4` |
| `--centroid-output` | Standalone-centroid output representation | `instance`, `centroid`, `both` | `instance` |

The default `--runtime auto` prefers TensorRT and falls back to ONNX.

!!! tip "One command for checkpoints and exported models"
    `sleap-nn predict` handles **both** trained checkpoint directories and
    exported ONNX/TensorRT model directories — it auto-detects which one you
    passed via `-m`. See the [Inference guide](inference.md) for the full set of
    `predict` options (data selection, filtering, device, tracking, etc.).

### Examples

```bash
# Maximum speed with TensorRT
sleap-nn predict -m exports/model -i video.mp4 --runtime tensorrt --batch_size 8

# CPU inference with ONNX
sleap-nn predict -m exports/model -i video.mp4 --runtime onnx --device cpu
```

---

## Performance

Benchmarks on NVIDIA RTX A6000:

| Model | Resolution | PyTorch | TensorRT FP16 | Speedup |
|-------|------------|---------|---------------|---------|
| Single Instance | 192x192 | 1.8 ms | 0.31 ms | 5.9x |
| Top-Down | 1024x1024 | 11.4 ms | 2.31 ms | 4.9x |
| Bottom-Up | 1024x1280 | 12.3 ms | 2.52 ms | 4.9x |

---

## Python API

### Export

```python
from sleap_nn.export import export_to_onnx, export_to_tensorrt

export_to_onnx(model, "model.onnx", input_shape=(1, 1, 192, 192))
export_to_tensorrt(model, "model.trt", input_shape=(1, 1, 192, 192))
```

### Inference

```python
from sleap_nn.export.predictors import ONNXPredictor
import numpy as np

predictor = ONNXPredictor("model.onnx")
frames = np.random.randint(0, 256, (4, 1, 192, 192), dtype=np.uint8)
outputs = predictor.predict(frames)
```

---

## Output Files

```
exports/my_model/
├── model.onnx
├── model.onnx.metadata.json
├── model.trt                   # If TensorRT
└── model.trt.metadata.json     # If TensorRT
```

---

## Limitations

- **TensorRT engines are GPU-specific** - Regenerate for different hardware
- **Fixed image size** - Height/width set at export time
- **Bottom-up grouping on CPU** - PAF matching can't be GPU-accelerated
- **No standalone centered-instance** - Export a combined top-down bundle (a standalone *centroid* model exports fine — see [Standalone Centroid](#standalone-centroid))

---

## Troubleshooting

??? question "ONNX Runtime falls back to CPU"
    ```bash
    # Check available providers
    python -c "import onnxruntime as ort; print(ort.get_available_providers())"

    # Install GPU support
    pip install onnxruntime-gpu
    ```

??? question "TensorRT export fails"
    - Check TensorRT version: `python -c "import tensorrt; print(tensorrt.__version__)"`
    - Ensure CUDA version matches
    - Check GPU memory during export

??? question "Bottom-up inference is slow"
    Bottom-up requires CPU-side Hungarian matching. Use larger batch sizes for better throughput.
