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
    uv tool install "sleap-nn[torch,export]" --torch-backend auto

    # ONNX export (GPU runtime)
    uv tool install "sleap-nn[torch,export-gpu]" --torch-backend auto

    # TensorRT export (Linux/Windows with NVIDIA GPU)
    uv tool install "sleap-nn[torch,export-gpu,tensorrt]" --torch-backend auto
    ```

=== "pip"

    ```bash
    # ONNX export (CPU)
    pip install sleap-nn[torch,export]

    # TensorRT export (NVIDIA GPUs)
    pip install sleap-nn[torch,export-gpu,tensorrt]
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

```bash
sleap-nn predict exports/my_model video.mp4 -o predictions.slp
```

---

## Export Options

```bash
sleap-nn export MODEL_PATH [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output-dir` | Output directory | Required |
| `-f`, `--format` | `onnx`, `tensorrt`, `both` | `onnx` |
| `--precision` | TensorRT: `fp32`, `fp16` | `fp16` |
| `-n`, `--max-instances` | Max instances per frame | `20` |
| `-b`, `--max-batch-size` | Max batch size | `8` |

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

---

## Inference Options

```bash
sleap-nn predict EXPORT_DIR VIDEO [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output` | Output path | `<video>.predictions.slp` |
| `-r`, `--runtime` | `auto`, `onnx`, `tensorrt` | `auto` |
| `-b`, `--batch-size` | Batch size | `4` |
| `-n`, `--n-frames` | Frames to process (0=all) | `0` |

### Examples

```bash
# Maximum speed with TensorRT
sleap-nn predict exports/model video.mp4 --runtime tensorrt --batch-size 8

# CPU inference
sleap-nn predict exports/model video.mp4 --runtime onnx --device cpu

# First 1000 frames
sleap-nn predict exports/model video.mp4 --n-frames 1000
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
- **No standalone centroid/instance** - Use combined top-down or Python API

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
