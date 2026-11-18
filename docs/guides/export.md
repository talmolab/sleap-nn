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
| `-o`, `--output` | Output directory | `PATH` | Required |
| `-f`, `--format` | Export format | `onnx`, `tensorrt`, `both` | `onnx` |
| `--precision` | TensorRT precision (see [Precision](#tensorrt-precision)) | `fp32`, `fp16`, `tf32`, `bf16` | `fp16` |
| `--max-instances` | Max instances per frame | `INT` | `20` |
| `--max-batch-size` | Max batch size (TensorRT profile; larger batches run in chunks of it) | `INT` | `8` |
| `--opt-batch-size` | TensorRT: batch the engine's kernels are tuned for (profile `opt`; for throughput, the batch you run); ignored for `--format onnx` (noted in the metadata) | `INT` | `1` |
| `--strongly-typed` | TensorRT 10.x: build fp16 / bf16 strongly typed, with the precision in the graph, as TensorRT 11 always does (see [Precision](#tensorrt-precision)) | Flag | `false` |
| `--force-precision` | Build fp16 / bf16 even for a family whose reduced-precision engines fail its fp32 graph (multi-class bottom-up; see [Precision](#tensorrt-precision)) | Flag | `false` |
| `--input-dtype` | Box detectors and `--network-only` exports: dtype of the graph's image input (`float32` receives the checkpoint path's own input; `auto` = `float32` for `bbox_detr`, `uint8` otherwise) | `auto`, `uint8`, `float32` | `auto` |
| `--nodes` | Export only these confmap nodes (`single_instance` / `centered_instance`, alone or in a top-down bundle); repeat or comma-separate | `TEXT` | all nodes |
| `--node-prefix` | Export only the nodes whose name starts with the prefix | `TEXT` | `None` |
| `--strip-prefix` | Name the `--node-prefix` nodes without the prefix in the exported metadata and `training_config` | Flag | `false` |

### TensorRT precision

| `--precision` | What runs | Needs |
|---------------|-----------|-------|
| `fp32` | strict IEEE fp32 (TF32 off) | TensorRT >= 10.13 (older 10.x: a warning, built with its builder flags as before) |
| `tf32` | fp32 with TensorRT's TF32 kernels (10-bit mantissa in matrix products) | TensorRT >= 10.13 |
| `fp16` | fp16 (fp16's range: large activations can overflow) | TensorRT >= 10.13 |
| `bf16` | bf16 (fp32's range, 8 significant bits); in graphs that compute coordinates (box detectors, peaks found in the graph) every layer outside the backbone stays fp32; transposed convolutions always | TensorRT >= 10.13 and a GPU with compute capability >= 8.0 (judged on the `--device` GPU, where the engine is built) |

How the reduced precisions are built depends on the TensorRT version:

- **TensorRT 10.x** (default): builder flags (weak typing): TensorRT picks fp16 / bf16
  kernels where they are faster, and sleap-nn holds the fp32 layers above in fp32.
- **TensorRT 11** removed those flags: every network is strongly typed, and each
  tensor runs in the type the graph gives it. sleap-nn then writes the precision into
  the graph it hands TensorRT ("strong typing"): it converts the chosen nodes'
  weights and inserts `Cast` nodes where a tensor crosses between fp32 and the reduced
  type, with the same scopes as on 10.x. Always fp32 there: normalization ops,
  reductions, `Pow` / `Sqrt` / `Reciprocal` / `Exp` / `Log` / `CumSum`, nodes that
  shape inference cannot type (counted in the metadata, with a warning), tensors
  that control-flow bodies (`If` / `Loop`) read; under fp16, every constant beyond
  fp16's range (|x| > 65504: initializers of any size, Constant nodes, fill values, e.g.
  the 1e9 sentinels for absent peaks), constants computed from constants (up to a
  million elements), and what the graph computes from those; under bf16, a transposed
  convolution and the node feeding it.
- `--strongly-typed` builds TensorRT 10.x engines the TensorRT 11 way (all of the
  above), to check an upgrade's numbers before making it.

`model.onnx` itself stays fp32. On the strongly typed path the graph TensorRT parsed is
saved as `tensorrt_graph/model.<precision>.onnx` at the export's opset: the fp16 graph
passes ONNX's full check (types included); the **bf16 graph is for TensorRT only**
(ONNX allows bfloat16 in `Conv`, pooling, `TopK`, `GridSample` ... only from opset 22),
as the metadata (`graph_validity`) says.

Multi-class bottom-up models do not reproduce their fp32 graph in fp16 or bf16 (on a
test model, 40-77% of peaks within 0.5 px, on TensorRT 10 and 11): on the strongly
typed path they are built in fp32 with a warning (`precision_requested` /
`precision_fallback` in the metadata); `--force-precision` builds what was asked. On
TensorRT 10's builder flags they are built as asked, with a warning.

The same `--precision` therefore gives slightly different numbers on TensorRT 10 and
11 (TensorRT 10 also picks fp32 for some layers on its own). Engines, and results from
them, are not interchangeable between TensorRT major versions: compare a new
version's engines against `fp32` on your data before switching.

The precision, typing (`weak` / `strong`), the precision per part of the graph
(`precision_by_scope`) and the TensorRT version actually built are written to
`model.trt.metadata.json` (an `fp16` request on a TensorRT 10 GPU without fast fp16
builds, and records, `fp32`). An engine runs only on the TensorRT version (and GPU
model) that built it: after upgrading TensorRT, export the engines again (`sleap-nn
export ... --format tensorrt`); `model.onnx` does not change. Reduced precision moves
outputs: compare against an `fp32` export on your own data. Example: a DINOv3
`bbox_detr` detector loses boxes in `fp16`; in `bf16` with fp32 outside the backbone
its box corners move by about 0.1 px (median) against `fp32`, and by about 1 px when
the whole graph runs in bf16.

`--nodes` / `--node-prefix` slice the confmap head to the kept channels before
tracing, so the exported graph computes only those confidence maps (see
[Predicting a subset of nodes](inference-guide.md#predicting-a-subset-of-nodes)).
The export's metadata and copied `training_config.yaml` describe the kept nodes.

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

### Detector + box-scaled top-down (+ part stage): network-only exports

A box detector with a centered-instance model -- fixed or box-scaled crops
(`crop_mode: bbox`) -- and an optional part model (`--part_model`) runs exported
**stage by stage**: export each directory on its own, the crop models with
`--network-only`, and give `sleap-nn predict` the export directories where it takes
the checkpoint directories:

```bash
sleap-nn export models/detector -o exports/detector --format both --precision bf16
sleap-nn export models/pose -o exports/pose --format both --network-only --input-dtype float32 \
    --precision bf16 --max-batch-size 64 --node-prefix my_set/ --strip-prefix
sleap-nn export models/tail -o exports/tail --format both --network-only --input-dtype float32 \
    --precision bf16 --max-batch-size 64

sleap-nn predict -i video.mp4 -o out.slp -m exports/detector -m exports/pose \
    --part_model exports/tail --part_crop_rule rule.json --runtime tensorrt
```

Pass a `--precision` that your models keep their accuracy in (`bf16` or `fp32` for a
DINOv3 `bbox_detr` detector: the default `fp16` can lose its boxes); `sleap-nn predict`
runs the TensorRT engine of a directory that holds one.

A network-only graph is the network alone (preprocessed crop in, raw confidence maps
out; a detector's graph is network-only by construction). Each export directory stands
in for its checkpoint: the pipeline is the checkpoint pipeline -- size matching and
letterboxing, box decoding, fixed or box-scaled crops, peak finding and refinement,
the part stage's crops from the body keypoints -- with only the forward passes on ONNX
Runtime or TensorRT, so exported and checkpoint predictions differ only by the runtime's
numerics (with `--input-dtype float32` the graphs receive exactly the checkpoint path's
inputs). With several `--model_paths`, and for `--part_model`, a directory that holds
both `best.ckpt` and an export (exported into the checkpoint's own directory) loads
the checkpoint, as before exports could stand in for checkpoints. A single
`--model_paths` directory that holds an export still takes the exported route (and
`--stream-to-file` refuses it), whatever else it holds. `--runtime` picks the runtime for
every export directory (`auto` prefers `model.trt`), on every route (several
`--model_paths`, `--part_model`, `--stream-to-file`); the predictions' provenance
(`exported_models`) records the runtime, and for an engine its precision, of each
exported graph that ran (a part model's tagged `stage: part`). A `--network-only` crop export passed alone (or with `--boxes`)
runs exactly as its checkpoint would there. Node filters are baked in at export (`--nodes` /
`--node-prefix` / `--strip-prefix`); `--node_prefix` at prediction time is refused for
an exported network. `--max-batch-size` is a TensorRT engine's batch limit: larger
crop batches run in chunks of it. A TensorRT engine runs on the current CUDA device:
pick the card with `CUDA_VISIBLE_DEVICES` and pass `--device cuda`. A box detector
exported at another size (`--input-height` / `--input-width`, default the training
size) has that size written to the export's metadata and copied training config; a
`bbox_detr` graph is exported at that fixed height and width (CenterNet and SSD graphs
keep a dynamic size). A `--network-only` graph is traced and profiled at the size the
pipeline feeds it (the trained crop size times the trained scale in whole pixels,
padded to the backbone's stride), so `--input-scale`, `--crop-size`, `--input-height` and
`--input-width` are refused with `--network-only`.

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
    passed via `-m`. See the [Inference guide](inference-guide.md) for the full set of
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

Run an exported model through the unified pipeline by pointing a `Predictor` at
the export directory written by `sleap-nn export` (it contains `export_metadata.json`):

```python
from sleap_nn.inference import Predictor

predictor = Predictor.from_export_dir("exports/my_model", runtime="onnx")
labels = predictor.predict("video.mp4")
```

The CLI equivalent is `sleap-nn predict -m exports/my_model --runtime onnx -i video.mp4`.

---

## Output Files

```
exports/my_model/
├── model.onnx
├── export_metadata.json
├── model.trt                   # If TensorRT
└── model.trt.metadata.json     # If TensorRT
```

---

## Limitations

- **TensorRT engines are GPU-specific** - Regenerate for different hardware
- **Fixed image size** - Height/width set at export time
- **Bottom-up grouping on CPU** - PAF matching can't be GPU-accelerated
- **No standalone centered-instance** - Export a combined top-down bundle (a standalone *centroid* model exports fine — see [Standalone Centroid](#standalone-centroid)), or export it `--network-only` and run it inside `sleap-nn predict` (see [Detector + box-scaled top-down](#detector-box-scaled-top-down-part-stage-network-only-exports))

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
