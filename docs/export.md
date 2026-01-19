# Model Export and Fast Inference

!!! warning "Experimental Feature"
    The export module is **experimental** and under active development. APIs and behavior may change in future releases. Please report issues at [github.com/talmolab/sleap-nn/issues](https://github.com/talmolab/sleap-nn/issues).

This guide covers exporting trained SLEAP models to optimized formats (ONNX and TensorRT) for high-performance inference without the full training environment.

## Overview

The export module provides:

- **ONNX export** - Portable models that run on any platform with ONNX Runtime
- **TensorRT export** - Maximum performance on NVIDIA GPUs with FP16 optimization
- **Unified prediction CLI** - Run inference on exported models with SLP output

### When to Use Exported Models

| Use Case | Recommended Format |
|----------|-------------------|
| Cross-platform deployment | ONNX |
| Maximum GPU throughput | TensorRT FP16 |
| CPU-only inference | ONNX |
| Embedding in applications | ONNX or TensorRT |
| Production pipelines | TensorRT FP16 |

### Performance Comparison

Benchmarks on **NVIDIA RTX A6000** (48 GB).

**Batch size 1** (latency-optimized):

| Model Type | Resolution | PyTorch | ONNX-GPU | TensorRT FP16 | Speedup |
|------------|------------|---------|----------|---------------|---------|
| Single Instance | 192×192 | 1.8 ms | 1.3 ms | 0.31 ms | 5.9x |
| Centroid | 1024×1024 | 2.5 ms | 2.7 ms | 0.77 ms | 3.2x |
| Top-Down | 1024×1024 | 11.4 ms | 9.7 ms | 2.31 ms | 4.9x |
| Bottom-Up | 1024×1280 | 12.3 ms | 9.6 ms | 2.52 ms | 4.9x |
| Multiclass Top-Down | 1024×1024 | 8.3 ms | 9.1 ms | 1.84 ms | 4.5x |
| Multiclass Bottom-Up | 1024×1024 | 9.4 ms | 9.4 ms | 2.64 ms | 3.6x |

**Batch size 8** (throughput-optimized):

| Model Type | Resolution | PyTorch | ONNX-GPU | TensorRT FP16 | Speedup |
|------------|------------|---------|----------|---------------|---------|
| Single Instance | 192×192 | 3,111 FPS | 3,165 FPS | 11,039 FPS | 3.5x |
| Centroid | 1024×1024 | 453 FPS | 474 FPS | 1,829 FPS | 4.0x |
| Top-Down | 1024×1024 | 94 FPS | 122 FPS | 525 FPS | 5.6x |
| Bottom-Up | 1024×1280 | 113 FPS | 121 FPS | 524 FPS | 4.6x |
| Multiclass Top-Down | 1024×1024 | 127 FPS | 145 FPS | 735 FPS | 5.8x |
| Multiclass Bottom-Up | 1024×1024 | 116 FPS | 120 FPS | 470 FPS | 4.1x |

*Speedup is relative to PyTorch baseline.*

---

## Installation

Export functionality requires additional dependencies:

```bash
# For ONNX export (CPU inference)
pip install sleap-nn[export]

# For ONNX with GPU inference
pip install sleap-nn[export-gpu]

# For TensorRT export (NVIDIA GPUs only)
pip install sleap-nn[tensorrt]
```

Or install all export dependencies:

```bash
pip install sleap-nn[export-gpu,tensorrt]
```

---

## Quick Start

### Export a Model

```bash
# Export to ONNX only
sleap-nn export /path/to/model -o exports/my_model --format onnx

# Export to TensorRT FP16 (includes ONNX)
sleap-nn export /path/to/model -o exports/my_model --format both

# Export top-down model (centroid + instance)
sleap-nn export /path/to/centroid_model /path/to/instance_model \
    -o exports/topdown --format both
```

### Run Inference

```bash
# Run inference on exported model
sleap-nn predict exports/my_model video.mp4 -o predictions.slp

# Use TensorRT for maximum speed
sleap-nn predict exports/my_model video.mp4 -o predictions.slp --runtime tensorrt

# Specify batch size
sleap-nn predict exports/my_model video.mp4 -o predictions.slp --batch-size 8
```

---

## `sleap-nn export`

Export trained models to ONNX and/or TensorRT format.

```bash
sleap-nn export MODEL_PATH [MODEL_PATH_2] [options]
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
| `--opset-version` | | ONNX opset version | `17` |

### Examples

**Single Instance Model**

```bash
sleap-nn export models/single_instance.n=1000 \
    -o exports/fly_single \
    --format both
```

**Top-Down Model (Combined)**

For top-down inference, export both centroid and instance models together:

```bash
sleap-nn export models/centroid.n=1000 models/centered_instance.n=1000 \
    -o exports/fly_topdown \
    --format both \
    --max-instances 20
```

**Bottom-Up Model**

```bash
sleap-nn export models/bottomup.n=2000 \
    -o exports/mouse_bottomup \
    --format both \
    --max-instances 10
```

**Multiclass Models**

Export models with supervised identity (class) labels:

```bash
# Bottom-up multiclass
sleap-nn export models/multi_class_bottomup.n=1000 \
    -o exports/flies_multiclass \
    --format onnx

# Top-down multiclass (centroid + multiclass instance)
sleap-nn export models/centroid.n=1000 models/multi_class_topdown.n=1000 \
    -o exports/flies_multiclass_topdown \
    --format both
```

### Output Files

After export, the output directory contains:

```
exports/my_model/
├── model.onnx              # ONNX model
├── model.onnx.metadata.json  # Model metadata
├── model.trt               # TensorRT engine (if --format both/tensorrt)
└── model.trt.metadata.json   # TensorRT metadata
```

---

## `sleap-nn predict`

Run inference on exported models.

```bash
sleap-nn predict EXPORT_DIR INPUT_PATH [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `EXPORT_DIR` | Path to exported model directory |
| `INPUT_PATH` | Path to video file (`.mp4`, `.avi`, etc.) |

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output path for predictions (`.slp`) | `<input>.predictions.slp` |
| `--runtime` | `-r` | Runtime: `auto`, `onnx`, or `tensorrt` | `auto` |
| `--batch-size` | `-b` | Inference batch size | `4` |
| `--n-frames` | `-n` | Number of frames to process (0 = all) | `0` |
| `--device` | | Device: `auto`, `cuda`, or `cpu` | `auto` |

### Examples

**Basic Inference**

```bash
sleap-nn predict exports/my_model video.mp4 -o predictions.slp
```

**High-Throughput TensorRT**

```bash
sleap-nn predict exports/my_model video.mp4 \
    -o predictions.slp \
    --runtime tensorrt \
    --batch-size 8
```

**Process Specific Frames**

```bash
sleap-nn predict exports/my_model video.mp4 \
    -o predictions.slp \
    --n-frames 1000
```

**CPU Inference**

```bash
sleap-nn predict exports/my_model video.mp4 \
    -o predictions.slp \
    --runtime onnx \
    --device cpu
```

---

## Supported Model Types

| Model Type | CLI Name | Description |
|------------|----------|-------------|
| Single Instance | `single_instance` | One animal per frame |
| Centroid | `centroid` | Centroid detection only |
| Centered Instance | `centered_instance` | Instance on cropped images |
| Top-Down | `topdown` | Centroid + instance (combined) |
| Bottom-Up | `bottomup` | Multi-instance with PAF grouping |
| Multiclass Bottom-Up | `multi_class_bottomup` | Bottom-up with identity classes |
| Multiclass Top-Down | `multi_class_topdown` | Top-down with identity classes |

---

## Python API

### Export Models Programmatically

```python
from sleap_nn.export import export_to_onnx, export_to_tensorrt
from sleap_nn.export.wrappers import SingleInstanceONNXWrapper
from sleap_nn.export.metadata import build_base_metadata

# Load your trained model
model = ...  # Your trained PyTorch model

# Create wrapper
wrapper = SingleInstanceONNXWrapper(
    backbone=model.backbone,
    head=model.head,
    max_instances=1,
)

# Export to ONNX
export_to_onnx(
    wrapper,
    output_path="model.onnx",
    input_shape=(1, 1, 192, 192),  # (B, C, H, W)
    input_dtype="uint8",
)

# Export to TensorRT
export_to_tensorrt(
    wrapper,
    output_path="model.trt",
    input_shape=(1, 1, 192, 192),
    precision="fp16",
)
```

### Run Inference with Predictors

```python
from sleap_nn.export.predictors import ONNXPredictor, TensorRTPredictor
import numpy as np

# Load ONNX model
predictor = ONNXPredictor("model.onnx")

# Prepare input (uint8 images)
frames = np.random.randint(0, 256, (4, 1, 192, 192), dtype=np.uint8)

# Run inference
outputs = predictor.predict(frames)
peaks = outputs["peaks"]  # (B, N_nodes, 2)
peak_vals = outputs["peak_vals"]  # (B, N_nodes)
```

### Load Exported Model Metadata

```python
from sleap_nn.export.metadata import load_metadata

metadata = load_metadata("exports/my_model")
print(f"Model type: {metadata.model_type}")
print(f"Nodes: {metadata.node_names}")
print(f"Skeleton edges: {metadata.edge_inds}")
```

---

## Technical Details

### Input Format

Exported models expect **uint8** images in NCHW format:
- Shape: `(batch, channels, height, width)`
- Dtype: `uint8` (0-255)
- Channels: 1 (grayscale) or 3 (RGB)

The wrapper automatically normalizes to `[0, 1]` float32 internally.

### Output Format

All models output fixed-size tensors for ONNX compatibility:

| Output | Shape | Description |
|--------|-------|-------------|
| `peaks` | `(B, N, 2)` | Keypoint coordinates (x, y) |
| `peak_vals` | `(B, N)` | Confidence scores |
| `peak_mask` | `(B, N)` | Valid peak mask (for variable counts) |

For bottom-up models, additional outputs enable instance grouping:
- `line_scores`: PAF scores for peak pairs
- `candidate_mask`: Valid candidate mask

### TensorRT Precision

- **FP32**: Full precision, highest accuracy
- **FP16**: Half precision, ~2x faster, minimal accuracy loss

FP16 is recommended for production use on modern NVIDIA GPUs (Volta+).

### Dynamic Shapes

Exported models support dynamic batch sizes up to `--max-batch-size`. Height and width are fixed at export time based on the training configuration.

---

## Troubleshooting

### ONNX Runtime CUDA Not Available

If ONNX falls back to CPU:

```bash
# Check available providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Install GPU support:
```bash
pip install onnxruntime-gpu
```

### TensorRT Export Fails

Ensure TensorRT is properly installed:

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```

Common issues:
- CUDA version mismatch with TensorRT
- Insufficient GPU memory for engine building
- Missing cuDNN libraries

### Slow Bottom-Up Inference

Bottom-up models require CPU-side Hungarian matching for instance grouping. This is expected and cannot be accelerated with GPU export. For maximum throughput, use larger batch sizes.

### Model Metadata Missing

If metadata files are missing, the model can still be loaded but node names and skeleton information won't be available. Re-export the model to generate metadata.

---

## Known Limitations

!!! info "Current Limitations"
    - **Standalone centroid/centered_instance prediction**: The `sleap-nn predict` command only supports combined models (top-down, bottom-up, single-instance). Standalone centroid or centered-instance models must be used via the Python API.
    - **Bottom-up instance grouping**: PAF-based grouping runs on CPU and may be slower than GPU inference for models with many keypoints.
    - **TensorRT engine portability**: TensorRT engines are GPU-specific and must be regenerated when moving to different GPU hardware.
    - **Dynamic image sizes**: Height and width are fixed at export time. To support different resolutions, re-export with the desired input shape.

---

## See Also

- [Training Guide](training.md) - Train models before export
- [Inference Guide](inference.md) - PyTorch inference with `sleap-nn track`
- [CLI Reference](cli.md) - Complete CLI documentation
