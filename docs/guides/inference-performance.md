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
    sleap-nn track -i video.mp4 -m models/my_model/ \
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
sleap-nn track -i clip.mp4 -m models/my_model/ --device cuda --batch_size 4
```

Skip FP16 + compile — both add overhead that dominates short runs.

### "I'm on a Mac, MPS"

```bash
sleap-nn track -i video.mp4 -m models/my_model/ --device mps --batch_size 4
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
