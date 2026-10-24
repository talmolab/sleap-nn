# Trainer Config

Configure training hyperparameters, optimization, and logging.

---

## Essential Settings

```yaml
trainer_config:
  max_epochs: 100
  save_ckpt: true
  ckpt_dir: models
  run_name: my_experiment
```

| Option | Description | Default |
|--------|-------------|---------|
| `max_epochs` | Training epochs | `10` |
| `save_ckpt` | Save checkpoints | `false` |
| `ckpt_dir` | Checkpoint directory | `null` |
| `run_name` | Run folder name | auto-generated |

---

## Data Loading

```yaml
trainer_config:
  train_data_loader:
    batch_size: 4
    shuffle: true
    num_workers: 0    # int, or "auto"
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0
```

!!! warning "Workers without caching"
    Only use `num_workers > 0` with data caching enabled. The `torch_dataset`
    (streaming) pipeline reads frames through the video backend inside the
    dataset, and those backends cannot be pickled to worker processes. The
    exception is labels whose videos are all image sequences (one file per
    frame, e.g. COCO converted to `.slp`): there is no open handle to pickle, and
    streaming with workers is both the fastest and the leanest pipeline.

### Automatic worker count

Set `num_workers: "auto"` to size the pool for the machine the run lands on:

```yaml
trainer_config:
  train_data_loader:
    num_workers: auto
  val_data_loader:
    num_workers: auto
```

`auto` resolves once, at the start of training, taking the smaller of a **CPU
bound** and a **memory bound**.

**CPU bound** — `min(16, usable_cpus // local_processes - 1)`: the usable cores are
split across the training processes on the node (one per GPU under DDP), leaving
one core per process for the training loop. E.g. 128 cores and 4 GPUs give 16
workers per GPU; 8 cores and 1 GPU give 7.
"Usable" CPUs come from the process's CPU affinity mask where the platform
provides one (Linux cgroups/cpusets, SLURM, `docker --cpuset-cpus`), so a job
confined to part of a node sizes itself to its allocation rather than to the
host. The cap of 4 is there because dataloading here is dominated by decode and
cache lookup rather than heavy CPU transforms, so returns flatten quickly.

**Memory bound** — applies only to `torch_dataset_cache_img_memory`, and scales
with **dataset size**. The cache holds every labeled frame decoded, so its size is
`height x width x channels x labeled_frames` summed over your videos — a
1024x1024 grayscale frame is 1 MB, so ~8,000 labeled frames is ~8 GB. Each worker
then adds a share of that cache on top of the parent process: about **25%** per
worker on Linux (fork + copy-on-write, dirtied by refcounting) and a **full copy**
(100%) on macOS and Windows (spawn, so the cache dict is pickled into each worker).

Workers are therefore a memory *multiplier* on exactly the pipeline that is
already the most memory-hungry, and the multiplier is proportional to your
dataset. `auto` inverts the same estimate the memory check uses and picks the
largest worker count that still fits in available RAM — without this, `auto`
could itself push a run past the memory check and silently downgrade it to disk
caching.

On one fixed machine (16 cores, 16 GB free), varying only the dataset:

| Labeled-image bytes | macOS / Windows | Linux |
|---|---|---|
| up to ~2.5 GB | 4 | 4 |
| ~3 GB | 3 | 4 |
| ~4 GB | 2 | 4 |
| ~6 GB | 1 | 4 |
| ~8 GB | 0 | 2 |
| ~11-13 GB | 0 | 0 |
| ~14 GB and up | 4 | 4 (falls back to disk caching) |

The jump back to 4 at the bottom is intentional: past that size the cache does not
fit at *any* worker count, so the run falls back to disk caching, where workers
stop being a memory multiplier and the CPU bound applies again.

When it is memory rather than CPU that binds, the log says so:

```
`num_workers: auto` limited to 1 worker(s) by available memory (CPU alone would
allow 4): in-memory cache is 6.00 GB and each worker adds ~100% of it on darwin.
```

Two cases skip the memory bound:

- **`torch_dataset` (streaming) → `0`**, per the warning above, so `auto`
  is safe to leave set regardless of pipeline. When every train and val video is
  an image sequence, the CPU bound applies instead.
- **`torch_dataset_cache_img_disk`** keeps no large in-process cache, so it is
  CPU-bound only.

If the cache does not fit even at zero workers, the run falls back to disk
caching anyway — where workers are cheap again — so the CPU bound applies rather
than a pointless `0` (the last row of the table above).

The resolved integer is written into the saved `training_config.yaml`, so the
run records the worker count it actually used; `initial_config.yaml` keeps the
literal `auto`, so reusing that config on another machine re-resolves there.

### Low-worker hint

Caching is what makes `num_workers > 0` usable at all, so a cached run left at the
default of 0 is decoding batches serially in the training process for no reason.
When either caching pipeline is configured with fewer workers than the machine can
support, training logs a one-line hint:

```
Data caching is enabled (`data_pipeline_fw=torch_dataset_cache_img_memory`), which
supports parallel data loading, but `trainer_config.train_data_loader.num_workers`
(0) is below what this machine can support. Consider setting it to 4 — or to
`auto`, which picks this for you — to speed up training.
```

The number it suggests is whatever `auto` would resolve to on this machine and
dataset, not a fixed "use 2-4" — so on a memory-constrained run it suggests 1
rather than 4, and it never suggests a count that would push the run past the
memory check. It stays quiet when the pipeline is streaming, when the loaders are
already at or above the suggestion, or when there is nothing to suggest.

---

## Optimization

### Optimizer

```yaml
trainer_config:
  optimizer_name: Adam    # Adam or AdamW
  optimizer:
    lr: 0.0001
    amsgrad: false
    weight_decay: null      # e.g. 0.1 with AdamW (ViTPose); null = torch default
    encoder_lr_scale: null  # `pretrained` backbones: encoder LR multiplier (null = 1.0)
    layer_decay: null       # plain-ViT `pretrained` encoders: layer-wise LR decay
```

**Weight decay.** With `weight_decay: null` the optimizer keeps PyTorch's default
on every parameter: `0` for `Adam`, but `0.01` for `AdamW`, including biases and
normalization weights. Setting `weight_decay` applies it to weight matrices and
conv kernels only. Biases, normalization weights, any other 1-D parameter,
position embeddings and CLS / register / mask tokens (plus any names a timm
model lists in `no_weight_decay()`) train without decay, as in the ViTPose, MAE
and timm recipes. Each optimizer param group is split into a `<group>/decay` and a
`<group>/no_decay` half that keep the group's learning rate, so this composes with
`layer_decay`.

**Encoder learning rates.** `encoder_lr_scale` and `layer_decay` set the learning
rate of a [`pretrained`](model.md) backbone's encoder relative to `lr` (the neck,
decoder and heads always train at `lr`). Block `i` of `L` trains at
`encoder_lr_scale · layer_decay^(L+1−i)` × `lr`. They used to live on
`model_config.backbone_config.pretrained`; those keys still work, with a
deprecation warning, and training writes them here in the saved
`training_config.yaml`. Setting both places to different values is an error.

Param groups are named (`base` for the neck/decoder/heads, `encoder` or
`encoder/depth<k>` for the encoder, with `/decay` / `/no_decay` suffixes under
`weight_decay`), and `ThroughputMonitor` logs each as `perf/lr/<name>`.

### Learning Rate Schedulers

Choose **one** scheduler. If several are set, the priority order is
`cosine_annealing_warmup` > `linear_warmup_linear_decay` > `multi_step_lr` >
`step_lr` > `reduce_lr_on_plateau`; they never stack.

=== "Reduce on Plateau"
    ```yaml
    lr_scheduler:
      reduce_lr_on_plateau:
        patience: 5
        factor: 0.5
        min_lr: 1e-8
    ```

=== "Step LR"
    ```yaml
    lr_scheduler:
      step_lr:
        step_size: 20    # Every N epochs
        gamma: 0.5       # Multiply by this
    ```

=== "Cosine Annealing + Warmup"
    ```yaml
    lr_scheduler:
      cosine_annealing_warmup:
        warmup_epochs: 5
        warmup_start_lr: 0.0
        eta_min: 1e-6
    ```

=== "Linear Warmup + Decay"
    ```yaml
    lr_scheduler:
      linear_warmup_linear_decay:
        warmup_epochs: 5
        warmup_start_lr: 0.0
        end_lr: 1e-6
    ```

=== "Multi-step (ViTPose)"
    ```yaml
    lr_scheduler:
      multi_step_lr:
        milestones: [170, 200]   # epochs (or steps with milestone_unit: step)
        gamma: 0.1
        milestone_unit: epoch
        warmup_steps: 500        # linear warmup, in optimizer steps
        warmup_start_factor: 0.001
    ```

**Per-step schedules.** `cosine_annealing_warmup` and `linear_warmup_linear_decay`
update the learning rate once per epoch by default, and the warmup counts whole
epochs: with the default `warmup_start_lr: 0.0`, the first epoch trains at LR 0.
Setting `warmup_steps` (any int, `0` included) switches either schedule to
per-iteration updates: the warmup lasts `warmup_steps` optimizer steps and the
decay runs to `max_steps`. When `max_steps` is unset it is the schedule's own
`max_epochs` × optimizer steps per epoch, or else the run's total optimizer steps
(`trainer_config.max_epochs`, capped by `trainer_config.max_steps`).

```yaml
lr_scheduler:
  cosine_annealing_warmup:
    warmup_steps: 1000   # per-iteration schedule
    max_steps: null      # = the whole run
    warmup_start_lr: 0.0
    eta_min: 1e-6
```

`multi_step_lr` always updates per optimizer step. Epoch milestones are converted
with the number of optimizer steps per epoch, so milestone `170` takes effect on
the first step of epoch 170 (0-indexed), as in mmcv's `step` policy. Every
schedule scales its absolute values (`warmup_start_lr`, `eta_min`, `end_lr`,
`min_lr`) by each param group's LR ratio, so `encoder_lr_scale` / `layer_decay`
ratios hold for the whole run.

**Fixed-length schedules and early stopping.** `cosine_annealing_warmup`,
`linear_warmup_linear_decay` and `multi_step_lr` are planned around the run
length. Choosing one turns early stopping off unless
`early_stopping.stop_training_on_plateau` is set explicitly, and the saved config
shows `reduce_lr_on_plateau: null` (it never ran alongside another schedule). An
explicit `stop_training_on_plateau: true` is kept, with a warning that it can cut
the decay short.

---

## Early Stopping

```yaml
trainer_config:
  early_stopping:
    stop_training_on_plateau: null  # null = on, off with a fixed-length LR schedule
    patience: 10        # Epochs without improvement (eval/* metrics: evaluations)
    min_delta: 1e-8     # Minimum improvement
```

Early stopping watches the same metric (and mode) as the best checkpoint:
`model_ckpt.monitor` / `model_ckpt.mode` when set, otherwise the model type's default
(`val/loss`; the retrieval metric for `embedding`; the box metric for `bbox_*` with
eval on). An `eval/*` metric is only measured every `eval.frequency` epochs, so
patience then counts evaluations.


Training writes the resolved `true`/`false` into the saved `training_config.yaml`.

---

## Weight EMA

```yaml
trainer_config:
  ema:
    enabled: false
    decay: 0.9998
    warmup_steps: 2000
```

Keeps an exponential moving average of the weights, updated after every optimizer
step as `ema = d · ema + (1 − d) · weights` with `d = decay · (1 − exp(−n /
warmup_steps))` after `n` updates (RTMPose's `ExpMomentumEMA`; `warmup_steps: 0`
uses `decay` from the first step). The average is kept in float32 under any
`precision`, and covers the trainable parameters and floating-point buffers
(BatchNorm running statistics).

* Validation runs on the EMA weights, so `val/loss`, epoch-end evaluation,
  checkpoint selection, early stopping and the plateau scheduler all see them.
* Checkpoints store the EMA weights as `state_dict`, so inference, export and
  `pretrained_backbone_weights` use them with no changes. The online weights are
  saved under `checkpoint["ema"]`, and resuming restores both. Resuming an EMA
  checkpoint with `ema.enabled: false` continues from the online weights.
* At the end of training the EMA weights are left in the model.

Each DDP rank keeps its own identical copy, so no communication is needed.
Sharded strategies (FSDP, DeepSpeed) are not supported.

---

## Hardware

```yaml
trainer_config:
  trainer_accelerator: auto     # auto, gpu, cpu, mps
  trainer_devices: auto         # Number of devices
  trainer_device_indices: null  # Specific GPUs [0, 2]
  trainer_strategy: auto        # auto, ddp, fsdp
```

---

## Speed

```yaml
trainer_config:
  precision: bf16-mixed            # "32" (default), "bf16-mixed", "16-mixed"
  float32_matmul_precision: high   # TF32 for fp32 matmuls (default null = leave unchanged)
  compile: false                   # true / "default" / "max-autotune" ... (torch.compile)
  gradient_clip_val: null          # e.g. 1.0 (global-norm clipping)
  gradient_clip_algorithm: norm    # "norm" or "value"
  accumulate_grad_batches: 1
  log_every_n_steps: 50            # step-metric logging cadence (default 1; each log syncs the GPU)
  max_steps: null                  # optimizer-step budget (stops at max_steps or max_epochs)
  val_check_interval: null         # validate every N batches (int) / epoch fraction (float)
  throughput_log_every_n_steps: 50 # perf/* metrics window; 0 disables
  confmap_fg_bg_diagnostics: null  # confmap fg/bg MSE split (null = centroid models only)
  train_data_loader:
    prefetch_factor: 2             # batches buffered per worker
    pin_memory: null               # null = true
    drop_last: null                # null = only under compile
```

The training-side defaults are unchanged, so existing configs train as before.
Two logging defaults are new: `perf/*` throughput metrics every 50 batches, and
the confmap fg/bg diagnostics now on for centroid models only (they were logged
for single-instance and centered-instance models too). For ViT backbones on Ampere or newer GPUs, `precision: bf16-mixed`,
`float32_matmul_precision: high`, `compile: true` and `log_every_n_steps: 50`
together gave about a 5x single-GPU speedup in our benchmarks (ViT-S top-down,
batch 128: ~370 to ~1,830 crops/s).

**Precision.** `bf16-mixed` keeps fp32 weights and optimizer state and runs
matmuls/convs in bfloat16 under autocast; the confmap MSE is computed in fp32.
`16-mixed` (fp16 with a gradient scaler) is for GPUs without bf16 support:
confmap gradients (~1e-8) are below fp16's range, so it only trains because of
the scaler.

**TF32.** `float32_matmul_precision: high` calls
`torch.set_float32_matmul_precision("high")` for the duration of training on
CUDA and restores the previous value afterwards, so inference and export are
unaffected. It speeds up fp32 matmuls (all of them under `precision: "32"`, the
ops autocast leaves in fp32 otherwise). cuDNN convolutions already use TF32 by
default in PyTorch.

**Compile.** `compile` runs `torch.compile` on the network in place
(`nn.Module.compile`), so checkpoint keys do not change and checkpoints load into
uncompiled models. The first steps are slow while compiling, and every new input
shape recompiles, so the training loader drops each pass's partial batch under
compile (`train_data_loader.drop_last: null`), unless a rank's share of the data
is smaller than one batch. Validation and in-training evaluation also run the
compiled network and recompile once per new shape.

**Cadence.** `log_every_n_steps` controls how often step metrics such as `loss`
reach the loggers; epoch metrics are unaffected. `max_steps` stops training at an
optimizer-step budget (set `max_epochs: -1` to train by steps only; epoch-based LR
schedulers such as `cosine_annealing_warmup` still count epochs). An int
`val_check_interval` validates every N training batches counted across epochs, so
validation no longer runs at every epoch end; early-stopping `patience` and
`ReduceLROnPlateau` then count validation runs.

**Dataloader.** `prefetch_factor` is the number of batches each worker keeps
ready (at most `prefetch_factor x num_workers` per loader). Before 0.4 it was
tied to `batch_size`, which with 16 workers and batch 128 buffered 2,048 batches
and used 126-231 GB of host RAM.

**Throughput metrics.** The `ThroughputMonitor` callback logs, every
`throughput_log_every_n_steps` training batches and excluding validation:
`perf/samples_per_sec`, `perf/steps_per_sec`, `perf/iter_time_ms`,
`perf/data_wait_ms` / `perf/data_wait_frac` (host time spent waiting for the next
batch), `perf/compute_ms`, `perf/gpu_idle_frac` (GPU idle time between steps,
from CUDA events), `perf/max_mem_gb` / `perf/max_mem_reserved_gb` and
`perf/lr/<group>` for every optimizer param group (`train/lr` is group 0 only). A
summary line is also printed to the console. It costs one GPU sync per window. A
high `gpu_idle_frac` means the dataloader is the bottleneck: raise `num_workers`
or enable caching. (`data_wait_frac` is host-side: once per-step syncs are gone,
e.g. with `log_every_n_steps: 50`, the host runs ahead of the GPU and spends most
of its time waiting for data even when the GPU is saturated.)

The epoch-level `train/time`, `train/steps_per_sec`, `train/samples_per_sec` and
`train/frames_per_sec` exclude validation time. The epoch-averaged `train/loss`
and per-node `train/confmaps/<node>` are summed on the GPU and read back once per
epoch instead of every step.

---

## Visualization

```yaml
trainer_config:
  visualize_preds_during_training: true
  keep_viz: false    # Keep viz folder after training
```

---

## WandB Logging

```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: your-username
    project: your-project
    name: run-name
    api_key: null             # Or set WANDB_API_KEY env
    wandb_mode: online        # online, offline
    save_viz_imgs_wandb: true
    delete_local_logs: null   # Auto-delete online logs
```

---

## Checkpointing

```yaml
trainer_config:
  model_ckpt:
    save_top_k: 1     # Keep N best models
    save_last: false  # Also save last.ckpt (every epoch)
  resume_ckpt_path: null  # Resume from this path
```

---

## Training Control

```yaml
trainer_config:
  min_train_steps_per_epoch: 200  # Minimum steps
  train_steps_per_epoch: null     # Exact steps (null=auto); summed over all GPUs
  enable_progress_bar: true
  seed: 42                        # Random seed (ensures deterministic train/val splits)
```

---

## Online Hard Keypoint Mining

Focus on difficult keypoints:

```yaml
trainer_config:
  online_hard_keypoint_mining:
    online_mining: false
    hard_to_easy_ratio: 2.0
    min_hard_keypoints: 2
    max_hard_keypoints: null
    loss_scale: 5.0
    per_sample: true   # rank hard keypoints per sample (false: pre-0.4 batch ranking)
```

With `per_sample: true` (the default) each sample's nodes are ranked separately, only
over nodes with nonzero loss weight (see
[partially labeled skeletons](../guides/partial-labels.md)), and the OHKM term is the
mean loss of the selected (sample, node) pairs times `loss_scale`. For batch size > 1
this differs from the pre-0.4 behavior (`per_sample: false`), which ranks nodes on
losses summed over the batch.

---

## ZMQ (GUI Integration)

ZMQ monitoring is **off by default** and is intended only for training launched
from the SLEAP GUI, which sets these ports itself. For CLI training leave both
ports unset (`null`) — that is the default:

```yaml
trainer_config:
  zmq:
    publish_port: null
    controller_port: null
    controller_polling_timeout: 10
```

Setting a port opts that channel in: `publish_port` publishes training progress
to `tcp://127.0.0.1:{publish_port}`, and `controller_port` subscribes to stop
commands on `tcp://127.0.0.1:{controller_port}`. Each is enabled independently.

---

## Complete Example

```yaml
trainer_config:
  # Training
  max_epochs: 200
  save_ckpt: true
  ckpt_dir: models
  run_name: fly_bottomup_v1

  # Data loading
  train_data_loader:
    batch_size: 4
    shuffle: true
    num_workers: 0
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0

  # Optimization
  optimizer_name: Adam
  optimizer:
    lr: 0.0001
    amsgrad: false

  lr_scheduler:
    reduce_lr_on_plateau:
      patience: 5
      factor: 0.5
      min_lr: 1e-8

  early_stopping:
    stop_training_on_plateau: true
    patience: 10

  # Hardware
  trainer_accelerator: auto
  trainer_devices: 1

  # Logging
  use_wandb: true
  wandb:
    project: sleap-experiments
    save_viz_imgs_wandb: true

  # Visualization
  visualize_preds_during_training: true
  keep_viz: false
```

---

## Full Reference

### TrainerConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_epochs` | int | `100` | Maximum training epochs |
| `save_ckpt` | bool | `false` | Save model checkpoints |
| `ckpt_dir` | str | `.` | Directory for checkpoints |
| `run_name` | str | `null` | Run folder name (auto-generated if null) |
| `seed` | int | `42` | Random seed for reproducibility and deterministic train/val splits |
| `trainer_accelerator` | str | `auto` | Hardware: `auto`, `gpu`, `cpu`, `mps` |
| `trainer_devices` | int/str | `null` | Number of devices or `auto` |
| `trainer_device_indices` | list | `null` | Specific device indices (e.g., `[0, 2]`) |
| `trainer_strategy` | str | `auto` | Strategy: `auto`, `ddp`, `fsdp` |
| `profiler` | str | `null` | PyTorch profiler: `simple`, `advanced`, `pytorch` |
| `enable_progress_bar` | bool | `true` | Show training progress |
| `min_train_steps_per_epoch` | int | `200` | Minimum batches per epoch (for `embedding`, P×K batches) |
| `train_steps_per_epoch` | int | `null` | Exact steps per epoch (null = auto: one pass over the data; for `embedding`, in P×K batches — `min(P, groups in the video)`×K for `sampler.kind: within_video`). With multiple GPUs this counts batches summed over all GPUs: each GPU runs `train_steps_per_epoch / num_gpus` optimizer steps, so an epoch is still one pass over the data (see [Multi-GPU](../guides/multi-gpu.md#steps-per-epoch)) |
| `visualize_preds_during_training` | bool | `false` | Save prediction visualizations |
| `keep_viz` | bool | `false` | Keep viz folder after training |
| `use_wandb` | bool | `false` | Enable WandB logging |
| `resume_ckpt_path` | str | `null` | Path to checkpoint to resume from |
| `optimizer_name` | str | `Adam` | Optimizer: `Adam` or `AdamW` |
| `precision` | str | `"32"` | Lightning precision: `"32"`, `"bf16-mixed"`, `"16-mixed"` |
| `float32_matmul_precision` | str | `null` | `torch.set_float32_matmul_precision` during training on CUDA, restored afterwards (`high` = TF32; `highest` = strict fp32; `null` = leave unchanged) |
| `compile` | bool/str | `false` | `torch.compile` the model in place (`true` or a mode such as `max-autotune`) |
| `gradient_clip_val` | float | `null` | Gradient clipping threshold |
| `gradient_clip_algorithm` | str | `norm` | `norm` or `value` |
| `accumulate_grad_batches` | int | `1` | Batches per optimizer step |
| `log_every_n_steps` | int | `1` | Step-metric logging cadence |
| `max_steps` | int | `null` | Optimizer-step budget |
| `val_check_interval` | int/float | `null` | Validate every N training batches (int, across epochs) or epoch fraction (float) |
| `throughput_log_every_n_steps` | int | `50` | `perf/*` throughput window (0 disables) |
| `confmap_fg_bg_diagnostics` | bool | `null` | Log confmap fg/bg MSE split (`null` = centroid only) |
| `ema` | EMAConfig | off | Weight EMA; see [EMAConfig](#emaconfig) |

### DataLoaderConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | int | `4` | Samples per batch (per-GPU; global batch = `batch_size × num_GPUs` with multi-GPU) |
| `shuffle` | bool | `true` (train) / `false` (val) | Shuffle data each epoch |
| `num_workers` | int or `"auto"` | `0` | Parallel data loading workers (use with caching only). `"auto"` sizes from usable CPU count; see [Automatic worker count](#automatic-worker-count) |
| `prefetch_factor` | int | `2` | Batches loaded ahead per worker (ignored with `num_workers: 0`) |
| `pin_memory` | bool | `null` | Pinned host memory for batches (`null` = `true`; the val loader falls back to the train loader's setting) |
| `drop_last` | bool | `null` | Train loader only: drop each pass's partial batch (`null` = only with `compile`, to avoid recompiles) |

### OptimizerConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `lr` | float | `1e-4` | Learning rate |
| `amsgrad` | bool | `false` | Enable AMSGrad variant |
| `weight_decay` | float | `null` | Decay for weight matrices / kernels only; biases, norms, 1-D params, position embeddings and special tokens get none (`null` = torch default on every parameter: `0` for Adam, `0.01` for AdamW) |
| `encoder_lr_scale` | float | `null` | `pretrained` encoder LR multiplier (`null` = `1.0`; replaces the deprecated backbone key) |
| `layer_decay` | float | `null` | Plain-ViT `pretrained` encoder: block `i` of `L` trains at `encoder_lr_scale · layer_decay^(L+1−i)` × `lr` (replaces the deprecated backbone key) |

### LRSchedulerConfig

Only one scheduler is used. Priority: `cosine_annealing_warmup` >
`linear_warmup_linear_decay` > `multi_step_lr` > `step_lr` > `reduce_lr_on_plateau`.

#### ReduceLROnPlateauConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | float | `1e-6` | Minimum improvement threshold |
| `threshold_mode` | str | `abs` | Mode: `rel` or `abs` |
| `cooldown` | int | `3` | Epochs to wait after reduction |
| `patience` | int | `5` | Epochs without improvement before reducing |
| `factor` | float | `0.5` | LR multiplication factor |
| `min_lr` | float | `1e-8` | Minimum learning rate |

#### StepLRConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `step_size` | int | `10` | Epochs between LR reductions |
| `gamma` | float | `0.1` | LR multiplication factor |

#### CosineAnnealingWarmupConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `warmup_epochs` | int | `5` | Linear warmup epochs |
| `warmup_start_lr` | float | `0.0` | Starting LR for warmup |
| `eta_min` | float | `0.0` | Minimum LR at end of cosine decay |
| `max_epochs` | int | `null` | Total epochs (auto from trainer) |
| `warmup_steps` | int | `null` | Warmup in optimizer steps; any int switches to per-step updates (`warmup_epochs` ignored) |
| `max_steps` | int | `null` | Per-step schedule length (`null` = `max_epochs` × steps/epoch, else the whole run) |

#### LinearWarmupLinearDecayConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `warmup_epochs` | int | `5` | Linear warmup epochs |
| `warmup_start_lr` | float | `0.0` | Starting LR for warmup |
| `end_lr` | float | `0.0` | Final LR at end of training |
| `max_epochs` | int | `null` | Total epochs (auto from trainer) |
| `warmup_steps` | int | `null` | Warmup in optimizer steps; any int switches to per-step updates (`warmup_epochs` ignored) |
| `max_steps` | int | `null` | Per-step schedule length (`null` = `max_epochs` × steps/epoch, else the whole run) |

#### MultiStepLRConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `milestones` | list[int] | `null` (required) | Strictly increasing epochs (or steps) at which the LR is multiplied by `gamma` |
| `gamma` | float | `0.1` | Multiplicative decay at each milestone |
| `milestone_unit` | str | `epoch` | `epoch` or `step` |
| `warmup_steps` | int | `0` | Linear warmup in optimizer steps (`0` = none) |
| `warmup_start_factor` | float | `0.001` | Fraction of the LR the warmup starts from |

### EarlyStoppingConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stop_training_on_plateau` | bool | `null` | Enable early stopping (`null` = on, except with a fixed-length LR schedule: `cosine_annealing_warmup`, `linear_warmup_linear_decay`, `multi_step_lr`) |
| `patience` | int | `10` | Epochs without improvement. For `embedding`, and for `bbox_*` models with `eval.enabled` (which select on `eval/val/bbox_mAP`), evaluations without improvement (one check every `eval.frequency` epochs) |
| `min_delta` | float | `1e-8` | Minimum improvement |

### EMAConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Track an EMA of the weights; validate and checkpoint with it |
| `decay` | float | `0.9998` | Decay after warmup, in `[0, 1)` |
| `warmup_steps` | int | `2000` | Time constant (optimizer steps) of the decay warmup; `0` disables it |

### ModelCkptConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `save_top_k` | int | `1` | Keep N best models |
| `save_last` | bool | `null` | Also write last.ckpt at the end of every epoch |

### WandBConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `entity` | str | `null` | WandB entity/username |
| `project` | str | `null` | WandB project name |
| `name` | str | `null` | Run name |
| `api_key` | str | `null` | API key (or use WANDB_API_KEY env) |
| `wandb_mode` | str | `null` | Mode: `online` or `offline` |
| `prv_runid` | str | `null` | Previous run ID (for resuming) |
| `group` | str | `null` | Run group |
| `save_viz_imgs_wandb` | bool | `false` | Upload viz images to WandB |
| `viz_enabled` | bool | `true` | Log pre-rendered matplotlib images |
| `viz_boxes` | bool | `false` | Log interactive keypoint boxes |
| `viz_masks` | bool | `false` | Log confidence map overlay masks |
| `viz_box_size` | float | `5.0` | Keypoint box size in pixels |
| `viz_confmap_threshold` | float | `0.1` | Confidence map mask threshold |
| `log_viz_table` | bool | `false` | Log images to wandb.Table |
| `delete_local_logs` | bool | `null` | Delete local logs (auto if online) |

### HardKeypointMiningConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `online_mining` | bool | `false` | Enable online hard keypoint mining |
| `hard_to_easy_ratio` | float | `2.0` | Ratio threshold for "hard" keypoints |
| `min_hard_keypoints` | int | `2` | Minimum hard keypoints |
| `max_hard_keypoints` | int | `null` | Maximum hard keypoints |
| `loss_scale` | float | `5.0` | Scale factor for hard keypoint losses |
| `per_sample` | bool | `true` | Rank hard keypoints within each sample over nodes with nonzero loss weight; `false` restores batch-level ranking |

### EvalConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable epoch-end evaluation metrics |
| `frequency` | int | `1` | Evaluate every N epochs |
| `oks_stddev` | float | `0.025` | OKS standard deviation (pose models only) |
| `oks_scale` | float | `null` | OKS scale override (pose models only) |
| `oks_sigmas` | str / float / list | `null` | Per-node OKS sigmas for post-training eval (`coco`, `crowdpose`, `ochuman`, `provenance`, `auto`, or one per node); overrides `oks_stddev` there |
| `coco_metrics` | bool | `false` | Add COCOeval-faithful `coco.*` metrics to post-training eval; top-down models then predict with `peak_threshold` 0 and mmpose rescoring. See [COCO-Style Evaluation](../guides/coco-evaluation.md) |
| `match_threshold` | float | `50.0` | Max distance (px) for centroid matching (centroid models only) |

!!! info "Model-Type-Dependent Evaluation"
    SLEAP-NN automatically selects the appropriate evaluation callback based on model type:

    - **Pose models** (single instance, bottom-up, centered instance): Uses OKS/PCK metrics with `oks_stddev` and `oks_scale`
    - **Centroid models**: Uses distance-based metrics with `match_threshold` for prediction-to-GT matching

    See the [Monitoring Guide](../guides/monitoring.md#epoch-end-evaluation) for details.

### ZMQConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `publish_port` | int | `null` | Port for publishing updates |
| `controller_port` | int | `null` | Port for receiving commands |
| `controller_polling_timeout` | int | `10` | Polling timeout in microseconds |
