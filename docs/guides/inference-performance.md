# Inference Performance

Tune `sleap-nn` inference for throughput on GPUs. Numbers in this doc come
from a benchmark on an **NVIDIA A40 (sm_86, CUDA 12.8, torch 2.9.1)** using
the project's standard test fixtures. Treat them as **relative speedups**,
not absolute ceilings — your numbers will scale with backbone size, video
resolution, and GPU class.

The bench script that produced these numbers lives at
`scratch/2026-04-30-inference-refactor-implementation/cuda_bench/run_cuda_bench.py`
and can be re-run to validate your own setup.

---

## TL;DR

!!! success "Recommended defaults"
    ```bash
    sleap-nn predict -i video.mp4 -m models/my_model/ \
        --device cuda \
        --batch_size 4
    ```

    - **`--device cuda`** — the only flag that matters for large videos.
    - **`--batch_size 4`** is a good starting point; raise to 8 / 16 if
      VRAM allows.
    - **FP16** delivers the biggest single-flag win (~1.5× on UNets) once
      it's exposed via the CLI. For now, opt in by constructing the
      predictor in Python with `use_fp16=True` (see [FP16](#fp16)).
    - **`torch.compile`** adds another ~1.2-1.3×, but pays a 0.5-3 s
      compile cost — only worth it on long videos. Opt in with
      `use_compile=True`.
    - **`paf_workers=0`** is the right default. Workers are a net loss
      for typical bottom-up workloads at fixture-checkpoint scale.
    - **`--upload_batches`** copies each batch to the GPU before
      preprocessing, so resizing and top-down cropping don't hold up the
      GPU: about 2× on a top-down pair (see
      [Batches on the GPU](#batches-on-the-gpu-upload_batches)).
    - **On NVIDIA, channels-last** (`--memory_format channels_last`)
      roughly doubled a top-down pair's speed on an A100 (see
      [Channels-last](#channels-last-memory_format)).

---

## Backbone-level throughput

Forward-pass latency measured on the A40 at `batch_size=4`,
`(B, 1, C, H, W)` input shape. Numbers are per-call (ms / batch).

| Model type | Eager | `torch.compile` | FP16 autocast | `fuse_layers` |
|---|---:|---:|---:|---:|
| `single_instance` | 1.20 ms | 0.93 ms (**1.29×**) | 0.84 ms (**1.43×**) | 1.20 ms (1.00×) |
| `centroid` | 2.48 ms | 1.96 ms (**1.27×**) | 1.61 ms (**1.54×**) | 2.48 ms (1.00×) |
| `bottomup` | 3.59 ms | 2.94 ms (**1.22×**) | 2.32 ms (**1.55×**) | 3.59 ms (1.00×) |
| `multi_class_bottomup` | 1.86 ms | 1.70 ms (1.10×) | 1.62 ms (1.15×) | 1.85 ms (1.01×) |

Headline observations:

- **FP16 wins across the board on UNet-based heads** — 1.43× to 1.55× on
  the three core model types. Zero-cost speedup; turn it on by default
  on CUDA.
- **`torch.compile` is consistently positive but adds a one-time
  compilation cost** (0.5–3 s per model). Long videos amortize that
  easily; short clips don't.
- **`fuse_layers` is a no-op** (1.00×) on these UNets. The shipped
  default (`use_fp16=False, use_compile=False, fuse_layers=False`) is
  fine for cold-start; revisit `fuse_layers` only if you've profiled
  Conv-BN fusion specifically helping your backbone.

---

## End-to-end throughput

Full `Predictor.predict_streaming(VideoProvider(small_robot.mp4))` on the
same A40, eager only, no opt-ins. Includes preprocessing, model forward,
postprocessing, and (for top-down) the second-stage centered-instance
inference.

| Model type | Frames | Wall time | fps | ms / frame |
|---|---:|---:|---:|---:|
| `single_instance` | 100 | 0.44 s | **228** | 4.4 |
| `centroid_only` | 100 | 0.43 s | **231** | 4.3 |
| `topdown` | 100 | 1.05 s | **95** | 10.5 |
| `bottomup` | 100 | 0.73 s | **137** | 7.3 |

!!! note "About these numbers"
    `small_robot.mp4` is a small 320×560 video and the fixture
    checkpoints are minimal UNets (~1-3 MB ckpts). On a production-sized
    backbone (deeper UNet / ConvNext / SwinT), absolute fps drops but
    relative speedups from FP16 + `torch.compile` are larger.

---

## Per-flag deep dive

### FP16

Enable **CUDA-only**; on MPS the kernels exist but tensor cores don't,
so there's no speedup (and we warn).

```python
from sleap_nn.inference import Predictor

predictor = Predictor.from_model_paths(
    ["models/my_model/"],
    device="cuda",
)
predictor.layer.backend.use_fp16 = True   # opt-in
```

| Reason to enable | Reason to skip |
|---|---|
| Long videos on CUDA where ~1.5× matters | Mac (MPS): no speedup |
| VRAM-constrained — FP16 also halves activation memory | Tasks with very tight accuracy budgets — FP16 introduces ~4e-3 numerical drift vs FP32 (autocast policy) |

The drift typically doesn't affect keypoint coordinates beyond
sub-pixel noise. We've seen `max |Δ| ≤ 0.001 px` on every fixture in
the parity bench, but always benchmark your own checkpoint before
shipping FP16 to production.

### `torch.compile`

Mode: `reduce-overhead` (CUDA-graph capture), `dynamic=False`.

```python
predictor.layer.backend.use_compile = True
```

| Reason to enable | Reason to skip |
|---|---|
| Long videos (>1000 frames) where compile cost amortizes | Notebook / interactive use — the 0.5–3 s compile cost dominates short runs |
| Multiple inference passes on the same model in the same process | One-shot inference where you'll never reuse the compiled module |

!!! warning "Static shapes"
    `dynamic=False` means the compiled graph is locked to one input
    shape. If your batches have varying spatial dims, set
    `dynamic=True` (slower but tolerant), or pre-pad to a uniform
    shape — which the new flow's sizematcher does automatically when
    `max_height` / `max_width` are set in training config.

### Batches on the GPU (`upload_batches`)

`sleap-nn predict --upload_batches` (or `upload_batches=True` on
`Predictor.from_model_paths` / `sleap_nn.inference.predict`) copies each
`uint8` batch from the provider to the GPU the (first) model runs on, before
`layer.predict`. Without it (the default) the layer gets a numpy batch, and
the sizematcher, input scaling and top-down crop extraction run on the CPU;
only the networks run on the GPU. It applies to trained checkpoints in
`sleap-nn predict`; `sleap-nn track` (the legacy pipeline) and exported
ONNX/TensorRT models are unchanged. Off a CUDA device it does nothing (and
logs a warning saying so).

```bash
sleap-nn predict -i video.mp4 -m models/centroid/ -m models/centered_instance/ \
    --device cuda --batch_size 16 --upload_batches
```

Measured on an **NVIDIA A100 40 GB** (torch 2.6, sleap-nn 0.3.3) with a
top-down pair (centroid at input scale 0.25, output stride 2;
centered-instance on 320 px crops at scale 0.5, max stride 32, 24
filters), a 1280×1024 5-animal video, batch 16, fp32:

| | resize | crops | networks + peaks | total ms / frame | fps | GPU busy |
|---|---:|---:|---:|---:|---:|---:|
| numpy batch (`upload_batches=False`) | 3.79 | 1.42 | 4.78 | 10.0 | 100 | 38% |
| uint8 batch on the GPU | 0.04 | 0.07 | 4.50 | 4.6 | 226 | 89% |

Same animals found; 90% of keypoints bit-identical. The rest differ
because the GPU's `uint8` resize rounds differently from the CPU's: on an
RTX 6000 Ada, 5–13% of resized pixels came out 1 grey level apart (never
more). That's why it is opt-in: without it, results are exactly as before.

### Channels-last (`memory_format`)

`sleap-nn predict --memory_format channels_last` (or
`memory_format="channels_last"` on `Predictor.from_model_paths` /
`sleap_nn.inference.predict`) stores the weights of every model in the
composition channels-last (both models of a top-down pair, a tiled layer's
inner model, ...). cuDNN's fastest convolution kernels on NVIDIA GPUs use
that layout; with channels-first weights the activations are converted
around every convolution. Trained checkpoints only: exported ONNX/TensorRT
models ignore it, with a warning.

```bash
sleap-nn predict -i video.mp4 -m models/centroid/ -m models/centered_instance/ \
    --device cuda --batch_size 16 --upload_batches --memory_format channels_last
```

It is also a `TorchBackend` field (`TorchBackend(memory_format=...)`), and
setting it on a built backend converts the model right away.

Same A100 and top-down pair as above, batches on the GPU, fps:

| | default layout | channels-last |
|---|---:|---:|
| fp32, batch 16 | 226 | 493 |
| `use_fp16`, batch 16 | 244 | 640 |
| `use_fp16`, batch 64 | – | 801 |

On this GPU fp16 only pays off once the model is channels-last. The fp16
rows used a peak-finding fix for fp16 confidence maps whose top is two
equal pixels (#782); without it fp16 loses about 0.8% of the animals.

| Reason to enable | Reason to skip |
|---|---|
| NVIDIA GPUs (tensor-core convolutions) | AMD (ROCm): no difference measured |
| | CPU / MPS: not measured |

The channels-last kernels can differ from the default ones in the last
bits (fp32, TF32 when allowed), about as much as running the same model
on a different GPU.

### `fuse_layers` (Conv-BN fusion)

Disabled by default. The post-PR-26 measurement on every fixture in this
repo shows a 1.00× speedup — i.e., **no measurable benefit on these
UNets**. Conv-BN fusion is valuable when:

- The backbone has many `Conv2d → BatchNorm2d` pairs in series (CARE-style
  decoders, large ConvNexts).
- Inference is so cheap that eager-mode Python overhead dominates.

Neither applies to the shipped sleap-nn UNets. Leave `fuse_layers=False`.

### `paf_workers` (bottom-up CPU grouping pool)

Enables a multi-process pool for the CPU-bound part of bottom-up
inference (PAF grouping after the GPU finishes peak finding and PAF
scoring).

| `paf_workers` | fps on the bench | Notes |
|---:|---:|---|
| **0** | **153** | Inline, no pool. The right default. |
| 2 | 28 | 5× slower — spawn + IPC overhead dominates |
| 4 | 25 | Worse, more workers, more overhead |

!!! warning "Workers help only when CPU grouping is the bottleneck"
    Workers help if:

    - The video is long enough to amortize spawn cost (`>=1000` frames)
    - The GPU stage produces many peaks per frame (dense scenes,
      crowded multi-animal videos)
    - The grouping stage measurably dominates wall time when serialized

    For the typical small-multi-animal pipeline benchmarked here, the
    GPU stage is the bottleneck and CPU grouping is well under 1 ms /
    frame — workers can't parallelize what isn't there to parallelize.

### Backbone fusion / `Conv-BN`

See `fuse_layers` above. Not the same as `torch.compile`'s graph fusion.

---

## Workflow recipes

### "I want the fastest correct predictions on a long video"

```python
from sleap_nn.inference import Predictor

predictor = Predictor.from_model_paths(
    ["models/centroid/", "models/centered_instance/"],
    device="cuda",
    batch_size=8,
    upload_batches=True,
    memory_format="channels_last",  # NVIDIA
)
backend = predictor.layer.backend  # or .centroid_layer.backend for top-down
backend.use_fp16 = True
backend.use_compile = True

labels = predictor.predict("long_video.mp4")
labels.save("predictions.slp")
```

Expected speedup vs eager-CPU: 20–50× on CUDA for a 5-minute video.

### "I want a quick sanity check on a 10-frame clip"

```bash
sleap-nn predict -i clip.mp4 -m models/my_model/ --device cuda --batch_size 4
```

Skip FP16 + compile — both add overhead that dominates short runs.

### "I'm on a Mac, MPS"

```bash
sleap-nn predict -i video.mp4 -m models/my_model/ --device mps --batch_size 4
```

FP16 silently has no effect on MPS (warning logged). `torch.compile`
on MPS is unreliable — the new flow disables it for you with a clear
warning. Expect ~2-3× speedup over CPU; not 20× like CUDA.

### "Multi-animal bottom-up on a crowded video"

```python
predictor = Predictor.from_model_paths(
    ["models/bottomup/"],
    device="cuda",
    batch_size=4,
    paf_workers=4,   # try 2 / 4 / 8; measure on your data
)
```

`paf_workers > 0` is the one place you may need to experiment per
dataset. Start at 0; raise only if profiling shows CPU grouping
dominates GPU work.

---

## When to re-benchmark

The numbers above are from one A40 + fixture checkpoints. If you're
deploying to:

- **A different GPU class** (H100 / A100 / 3090 / 4090): FP16 speedup
  can be 1.7-2× on data-center cards with stronger tensor cores.
  Compile speedup roughly stays in the 1.2–1.4× band.
- **A production-sized backbone** (deeper UNet, ConvNext, SwinT): FP16
  gains grow; compile cost grows linearly with graph size; `fuse_layers`
  may finally start mattering.
- **Variable-resolution input** (different videos with different
  shapes): turn `dynamic=True` on compile or skip it.

Re-run the bench:

```bash
SLEAP_NN_REPO=/path/to/sleap-nn \
  python scratch/2026-04-30-inference-refactor-implementation/cuda_bench/run_cuda_bench.py
```

Output lands in the same folder as a timestamped log file.

---

## Parity guarantee

The new inference flow has been validated bit-exactly against the legacy
`Predictor.from_model_paths` flow on CUDA, MPS, and CPU across every
fixture model type × multiple sources:

| Fixture | Source | Max keypoint Δ vs legacy |
|---|---|---:|
| `single_instance` | `small_robot.mp4` | 0.000000 px |
| `single_instance` | `minimal_instance.pkg.slp` | 0.000000 px |
| `topdown` | `small_robot.mp4` | 0.000916 px |
| `topdown` | `minimal_instance.pkg.slp` | 0.000000 px |
| `bottomup` | `small_robot.mp4` | 0.000000 px |
| `bottomup` | `minimal_instance.pkg.slp` | 0.000000 px |

This parity is locked in by `tests/inference/test_parity_vs_legacy.py`,
which runs on every CI build.
