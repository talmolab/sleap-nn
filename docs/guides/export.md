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
| `-o`, `--output` | Output directory | `PATH` | `<model dir>/exported` (top-down pair: `<centroid dir>/exported_topdown`; multi-class top-down pair: `<centroid dir>/exported_multi_class_topdown`) |
| `-f`, `--format` | Export format | `onnx`, `tensorrt`, `both` | `onnx` |
| `--precision` | TensorRT precision (see [Precision](#tensorrt-precision)) | `fp32`, `fp16`, `tf32`, `bf16` | `fp16` |
| `--max-instances` | Max instances per frame | `INT` | `20` |
| `--max-batch-size` | Max batch size (TensorRT profile; larger batches run in chunks of it) | `INT` | `8` |
| `--opt-batch-size` | TensorRT: batch the engine's kernels are tuned for (profile `opt`; for throughput, the batch you run); ignored for `--format onnx` (noted in the metadata). Unset, the engine is tuned for batch 1 | `INT` | unset |
| `--strongly-typed` | **Experimental.** TensorRT 10.x: build fp16 / bf16 strongly typed, with the precision in the graph, as TensorRT 11 always does (see [Precision](#tensorrt-precision)) | Flag | `false` |
| `--force-precision` | Build fp16 / bf16 even for a family whose reduced-precision engines fail its fp32 graph (multi-class bottom-up; see [Precision](#tensorrt-precision)) | Flag | `false` |
| `--fp32-attention-logits` / `--no-fp32-attention-logits` | **Experimental.** With `--precision fp16`, strongly typed: keep each attention block's logits product (Q.K^T) in fp32 (see [fp16 for ViT backbones](#fp16-for-vit-backbones-fp32-attention-logits)) | Flag | on for plain-ViT backbones, off otherwise |
| `--input-dtype` | Box detectors and `--network-only` exports: dtype of the graph's image input (`float32` receives the checkpoint path's own input; `auto` = `float32` for `bbox_detr`, `uint8` otherwise) | `auto`, `uint8`, `float32` | `auto` |
| `--nodes` | Export only these confmap nodes (`single_instance` / `centered_instance`, alone or in a top-down bundle); repeat or comma-separate | `TEXT` | all nodes |
| `--node-prefix` | Export only the nodes whose name starts with the prefix | `TEXT` | `None` |
| `--strip-prefix` | Name the `--node-prefix` nodes without the prefix in the exported metadata and `training_config` | Flag | `false` |
| `--peak-threshold` | Peak confidence threshold: baked into the graph for pose and centroid models; for box detectors the score threshold, stored in `export_metadata.json` and applied on the host | `FLOAT` | `0.2` |
| `--crop-size` | Crop size for crop-input models (required for `embedding` when the config has none) | `INT` | from config |
| `--device` | Device used for export | `TEXT` | `cpu` |

For every option (opset, input size/scale, bottom-up grouping knobs, TensorRT
workspace, `--verify/--no-verify`), see the
[CLI reference](../reference/cli.md#sleap-nn-export).

### TensorRT precision

| `--precision` | What runs | Needs |
|---------------|-----------|-------|
| `fp32` | strict IEEE fp32 (TF32 off) | TensorRT >= 10.13 (older 10.x: a warning, built with its builder flags as before) |
| `tf32` | fp32 with TensorRT's TF32 kernels (10-bit mantissa in matrix products) | TensorRT >= 10.13 |
| `fp16` | fp16 (fp16's range: large activations can overflow); plain-ViT backbones keep their attention logits in fp32 by default ([below](#fp16-for-vit-backbones-fp32-attention-logits)) | TensorRT >= 10.13 |
| `bf16` | bf16 (fp32's range, 8 significant bits); in graphs that compute coordinates (box detectors, peaks found in the graph) every layer outside the backbone stays fp32; transposed convolutions always | TensorRT >= 10.13 and a GPU with compute capability >= 8.0 (judged on the `--device` GPU, where the engine is built) |

How the reduced precisions are built depends on the TensorRT version:

!!! warning "Experimental: TensorRT 11 and strongly typed engines"
    The strongly typed path (every build on TensorRT 11, `--strongly-typed` on
    TensorRT 10.x, and the default fp16 build of plain-ViT backbones on TensorRT 10.13
    and later) may change in a later release without a deprecation period. The
    test suite stubs TensorRT, so these engines are checked by hand on a GPU, on few
    model types. Compare its engines against an `fp32` export on your own data before
    relying on them.

- **TensorRT 10.x** (default): builder flags (weak typing): TensorRT picks fp16 / bf16
  kernels where they are faster, and sleap-nn holds the fp32 layers above in fp32.
  Under fp16 it also holds in fp32 the few layers that read or compute a value beyond
  fp16's range (the -1e9 sentinels for absent peaks: the same rule as the strongly
  typed path below), so value outputs such as `centroid_vals` / `peak_vals` keep -1e9
  where no peak was found instead of -65504 or -inf (`fp16_fp32_layers` in the
  metadata).
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
test model, 40-77% of peaks within 0.5 px, on TensorRT 10 and 11): they are built in
fp32 with one warning (`precision_requested` / `precision_fallback` in the metadata),
on TensorRT 11, with `--strongly-typed` and on TensorRT 10's builder flags alike;
`--force-precision` builds what was asked. (Before this, TensorRT 10's builder flags
built them as asked, with a warning.)

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
`bbox_detr` detector loses boxes in plain `fp16`; in `bf16` with fp32 outside the backbone
its box corners move by about 0.1 px (median) against `fp32`, and by about 1 px when
the whole graph runs in bf16.

#### fp16 for ViT backbones: `--fp32-attention-logits`

!!! warning "Experimental"
    The switches `--fp32-attention-logits` and `--no-fp32-attention-logits` may change
    or be removed in a later release without a deprecation period. The default for
    plain-ViT backbones (below) is a fix for broken fp16 engines; it was measured on
    DINOv3 ViT-B models on one GPU only (below).

Plain `fp16` engines of DINOv3 ViT-B models fail: two keypoint models kept 17-44% of
their keypoints within 0.5 px of fp32, and a `bbox_detr` detector loses boxes, on
TensorRT 10 and 11. The cause is the attention logits product (Q.K^T): TensorRT's fp16
kernels accumulate it in fp16. `--fp32-attention-logits` (with `--precision fp16`)
keeps that one product per attention block in fp32; everything else is the usual fp16
rewrite above, and in graphs that compute coordinates fp16 runs only inside the
backbone (as `bf16` does). The build is strongly typed: on TensorRT 10 the option turns
`--strongly-typed` on (the builder flags cannot keep one product in fp32).

**This is the default for plain-ViT backbones** (a `pretrained` backbone whose encoder
is a plain, single-stride ViT -- DINOv2 / v3, CLIP, SigLIP, MAE ... -- and the
windowed DINOv2 backbone of `bbox_detr`; an encoder counts when it is single-stride
and has attention modules): an fp16 export of such a model keeps its attention logits
in fp32 without the flag. A bundle (a top-down pair) gets the setting as a whole when
any of its models has such a backbone. On TensorRT 10 the build is then strongly typed,
so the export folder also holds the graph TensorRT parsed
(`tensorrt_graph/model.fp16.onnx`, plus `model.fp16.onnx.data` above 1.8 GB) and the
metadata gains its keys (`typing: strong`, `tensorrt_graph`, `graph_casts`, a
`precision_by_scope` per scope). The export prints a note, and the metadata records
`fp32_islands` and `fp32_islands_mode` (`auto (plain-ViT backbone)`, or `requested`
with the flag). Two exceptions keep the old build: below TensorRT 10.13 (builder
flags, plain fp16, with a warning that the engine may be inaccurate: pass
`--precision bf16` or `fp32`, or use TensorRT >= 10.13), and a GPU without fast fp16
(the builder-flag build is fp32 there). `--no-fp32-attention-logits` builds plain fp16
as before. Other backbones (UNet, ConvNeXt, Swin ...), other precisions and the JIT
method are unchanged. Before this default, `--precision fp16` (the export default)
built plain fp16 engines for ViT backbones too.

The products are found by structure, not by name: every `MatMul` / `Einsum`
with two computed inputs (not weights) that reaches a `Softmax` through scale, mask,
bias, soft-capping and layout ops. When a softmax has more than one such candidate
(relative-position terms, scores carried over from an earlier block), all of them stay
fp32 and the softmax is reported. Softmaxes inside control-flow bodies (`If` / `Loop`)
are counted and reported, not handled. One limit is not detected: a further product
applied to the scores after Q.K^T on the same path (the scores times a data-dependent
matrix) hides Q.K^T from the search, and the record then looks clean; no sleap-nn
backbone is known to do this. The counts are written to the metadata
(`fp32_islands`: softmaxes in scope, products, unmatched, ambiguous, shared). A graph
with no products (UNets, CenterNet / SSD heads) gets a warning and the engine it would
get without products: plain fp16 for crop (`--network-only`) graphs, and fp16 inside the
backbone with fp32 elsewhere for graphs that compute coordinates. The `bbox_detr` and
ViT fp16 warnings are held back (unless `--no-fp32-attention-logits` is given) and
printed after the build unless it kept products and left no softmax unmatched. On an RTX PRO 6000 Blackwell with TensorRT
10.16, DINOv3 ViT-B detector and two keypoint-model engines built so kept TensorRT's fused
attention kernel, ran at plain fp16 speed (detector 3.1 ms per 576 x 768 frame) and
matched fp32 (same boxes on 200 frames, corners within 0.13 px at p99; 99.6% / 98.6%
of their keypoints within 0.5 px, against 96% / 89% for bf16).

Caveats: TensorRT chooses kernels per GPU by speed, so the same export built on another
card (a GeForce card in particular) may accumulate other fp16 products differently and
score differently: this is evidence for DINOv3 ViT-B on that card and TensorRT version
only, and other ViTs (DINOv2 ...) are not tested. Build crop-model engines with
`--opt-batch-size` at your usual number of animals per frame: a pose engine tuned for
one crop took 0.66 ms for one crop and 1.33 ms for three; tuned for three, 0.91 and
1.08 ms.

The multi-class bottom-up fallback above is unchanged by the option (such a family is
still built in fp32); `--force-precision` is still the only override there.

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

A model with an input scale other than 1 (single-instance, centroid,
centered-instance, bottom-up, multi-class and top-down models) resizes each frame,
or each top-down crop, inside the graph, antialiased like checkpoint inference. The
resize is built from gathers and sums (TensorRT's parser takes no antialiased
`Resize`), and its output size follows the input's size, as `int(size * input_scale)`
does on the checkpoint path. For the few scales whose double rounding no fraction
reproduces (0.7, 0.35, ...) the export warns that the resized size can differ by one
pixel for some frame sizes. The resize costs more than a plain one: on the CPU, from
a few per cent of the graph's time for a typical UNet on 1024 x 1024 frames to about
20% on 1080p RGB frames, more for very small models; a few per cent with TensorRT.
Box detectors apply their input scale outside the graph, and network-only exports
take inputs that are already resized.

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

(The part stage and its crop rule are described in [Part stage](part-stage.md).)

Pass a `--precision` that your models keep their accuracy in (a plain `fp16` engine
of a DINOv3 `bbox_detr` detector can lose its boxes: keep the default fp32 attention
logits, or use `bf16` or `fp32`); `sleap-nn predict` runs the TensorRT engine of a
directory that holds one.

A network-only graph is the network alone (preprocessed crop in, raw confidence maps
out; a detector's graph is network-only by construction). Each export directory stands
in for its checkpoint: the pipeline is the checkpoint pipeline -- size matching and
letterboxing, box decoding, fixed or box-scaled crops, peak finding and refinement,
the part stage's crops from the body keypoints -- with only the forward passes on ONNX
Runtime or TensorRT, so exported and checkpoint predictions differ only by the runtime's
numerics (with `--input-dtype float32` the graphs receive exactly the checkpoint path's
inputs). A directory that holds both `best.ckpt` and an export (exported into the
checkpoint's own directory) loads the checkpoint wherever model directories go
through the checkpoint loader: several `--model_paths` (with or without
`--stream-to-file`), a `--part_model` directory, and the Python API with one or more
paths (`predict(model_paths=[d])`, `open_stream`, `Predictor.from_model_paths([d])`);
a `--network-only` export follows the same rule. A lone `--model_paths` on the command
line takes the exported route for such a directory (and `--stream-to-file` with one
directory refuses it). Such a directory listed among several `--model_paths` together
with `--part_model` runs its checkpoint. `--runtime` picks the runtime for
every export directory (`auto` prefers `model.trt`), on every route (several
`--model_paths`, `--part_model`, `--stream-to-file`); the predictions' provenance
(`exported_models`) records the runtime, and for an engine its precision, of each
exported graph that ran (a part model's tagged `stage: part`). A `--network-only` crop export passed alone (or with `--boxes`)
runs exactly as its checkpoint would there. An exported box detector passed alone
decodes its boxes with `--peak_threshold` and `--bbox_nms_iou` when they are typed on
the command line, as it does next to a pose model. Without them it keeps the score
threshold stored at export (`sleap-nn export --peak-threshold`, 0.2 by default) and its
style's default suppression (0.45 for SSD, none for CenterNet and DETR). Node filters are baked in at export (`--nodes` /
`--node-prefix` / `--strip-prefix`); `--node_prefix` at prediction time is refused for
an exported network. `--max-batch-size` is a TensorRT engine's batch limit: larger
crop batches run in chunks of it. A TensorRT engine runs on the CUDA device `--device`
names (`cuda`: the current one; `cuda:N`: card N). A box detector
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

A multi-class top-down model exports as a pair with its centroid model. Without
`-o` it is written to `<centroid dir>/exported_multi_class_topdown`, with the
configs copied as `training_config_centroid.yaml` and
`training_config_multi_class_topdown.yaml`:

```bash
sleap-nn export models/centroid models/multi_class_topdown --format onnx
```

### Standalone Centered-Instance

A single `centered_instance` directory exports on its own. The network takes
crops (it is traced at the model's `crop_size`), not full frames. Box-scaled
centered-instance models (`crop_mode: bbox`) can only be exported this way; see
[Top-down pose with a box detector](bbox-topdown.md).

### Bounding-Box Detectors

`bbox_centernet` and `bbox_ssd` export the network only; box decoding runs on
the host. See [Bounding-box detection](bbox-detection.md#export-onnx-tensorrt).
A detector + top-down pair cannot be exported as one bundle yet.

### Re-ID Embedding

An `embedding` model exports from a single directory and takes fixed-size crops,
so it needs a crop size (`data_config.preprocessing.crop_size` or `--crop-size`).

### Not Exportable Yet

Segmentation models (`bottomup_segmentation`, `centered_instance_segmentation`,
`semantic_segmentation`) cannot be exported yet; `sleap-nn export` raises an
error for them.

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
When `-m`/`--model_paths` points at an export directory (`export_metadata.json`
with `model.onnx` or `model.trt`, as `sleap-nn export` writes it), `predict`
auto-detects it and runs the exported runtime; otherwise it loads a trained
checkpoint. A network-only crop export (`--network-only`) stands in for its
checkpoint in the checkpoint pipeline, and a directory that also holds
`best.ckpt` runs that checkpoint, except a single `-m` directory whose export has
its post-processing in the graph or is a box detector. Use `--runtime` to choose
between ONNX and TensorRT (it is ignored for checkpoints); `auto` prefers TensorRT
on a CUDA device and takes ONNX on any other device.

```bash
sleap-nn predict -m EXPORT_DIR -i VIDEO [options]
```

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `-m`, `--model_paths` | Export dir (auto-detected) or checkpoint dir | `PATH` | Required |
| `-i`, `--data_path` | Video or labels file | `PATH` | Required |
| `-o`, `--output_path` | Output path | `PATH` | input's extension replaced by `.predictions.slp` (`video.mp4` → `video.predictions.slp`) |
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
├── training_config.yaml        # Copied from the model dir (.json for legacy models)
├── model.trt                   # If TensorRT
└── model.trt.metadata.json     # If TensorRT
```

A top-down export copies both configs instead, as `training_config_centroid.yaml`
and `training_config_centered_instance.yaml` (`training_config_multi_class_topdown.yaml`
for a multi-class top-down pair).

---

## Limitations

- **TensorRT engines are GPU-specific** - Regenerate for different hardware
- **Image size** - ONNX graphs keep batch, height and width dynamic, except for RoPE (rotary-embedding ViT) backbones, which export with a static height/width. TensorRT shape profiles allow inputs up to 2x the export height/width (clamped to the static size for RoPE backbones). Graphs of UNet, ConvNeXt, SwinT and `pretrained` backbones zero-pad the network input (the frame or crop after the input scale) at the bottom and right to a multiple of the backbone's `max_stride`, as checkpoint inference does, so the network runs at any input size within the engine's profile (embedding graphs, which take fixed-size crops, are not padded). The pad uses the configured `max_stride`: a ConvNeXt or SwinT backbone built by hand with a `max_stride` below the stride its layers reach (training sets it to 16 x `stem_patch_stride`) can still fail at some sizes
- **Some values are fixed at the traced frame size** - A combined top-down export samples each crop with a grid sized for the frame size it was traced at (`--input-height` / `--input-width`, default the training `max_height` x `max_width`): at other frame sizes its centroids are right, but each crop covers `crop_size` times (frame size / traced size) pixels, so its points are off; export at the frame size you will predict on. A bottom-up graph scores connections with the maximum edge length computed from its PAF size at trace time (`max_edge_length_ratio`), not at each frame's size
- **Bottom-up grouping on CPU** - PAF matching can't be GPU-accelerated
- **No segmentation export** - Segmentation model types are not exportable yet (see [Not Exportable Yet](#not-exportable-yet))
- **Standalone centered-instance takes crops** - For full-frame inference export a combined top-down bundle (a standalone *centroid* model exports fine — see [Standalone Centroid](#standalone-centroid)), or export it `--network-only` and run it inside `sleap-nn predict` (see [Detector + box-scaled top-down](#detector-box-scaled-top-down-part-stage-network-only-exports))

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
