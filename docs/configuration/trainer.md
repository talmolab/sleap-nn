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
    dataset, and those backends cannot be pickled to worker processes.

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

**CPU bound** — `min(4, usable_cpus - 1)`, leaving one core for the main process.
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
worker on Linux (fork + copy-on-write, dirtied by refcounting) and about **50%**
on macOS and Windows (spawn, so the cache dict is pickled into each worker).

Workers are therefore a memory *multiplier* on exactly the pipeline that is
already the most memory-hungry, and the multiplier is proportional to your
dataset. `auto` inverts the same estimate the memory check uses and picks the
largest worker count that still fits in available RAM — without this, `auto`
could itself push a run past the memory check and silently downgrade it to disk
caching.

On one fixed machine (16 cores, 16 GB free), varying only the dataset:

| Labeled-image bytes | macOS / Windows | Linux |
|---|---|---|
| up to ~4 GB | 4 | 4 |
| ~6 GB | 2 | 4 |
| ~8 GB | 1 | 2 |
| ~11-13 GB | 0 | 0 |
| ~14 GB and up | 4 | 4 (falls back to disk caching) |

The jump back to 4 at the bottom is intentional: past that size the cache does not
fit at *any* worker count, so the run falls back to disk caching, where workers
stop being a memory multiplier and the CPU bound applies again.

When it is memory rather than CPU that binds, the log says so:

```
`num_workers: auto` limited to 1 worker(s) by available memory (CPU alone would
allow 4): in-memory cache is 8.00 GB and each worker adds ~50% of it on darwin.
```

Two cases skip the memory bound:

- **`torch_dataset` (streaming) → always `0`**, per the warning above, so `auto`
  is safe to leave set regardless of pipeline.
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
```

### Learning Rate Schedulers

Choose **one** scheduler:

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

---

## Early Stopping

```yaml
trainer_config:
  early_stopping:
    stop_training_on_plateau: true
    patience: 10        # Epochs without improvement (embedding: evaluations)
    min_delta: 1e-8     # Minimum improvement
```

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
  train_steps_per_epoch: null     # Exact steps (null=auto)
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
```

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
| `train_steps_per_epoch` | int | `null` | Exact steps per epoch (null = auto: one pass over the data; for `embedding`, in P×K batches — `min(P, groups in the video)`×K for `sampler.kind: within_video`) |
| `visualize_preds_during_training` | bool | `false` | Save prediction visualizations |
| `keep_viz` | bool | `false` | Keep viz folder after training |
| `use_wandb` | bool | `false` | Enable WandB logging |
| `resume_ckpt_path` | str | `null` | Path to checkpoint to resume from |
| `optimizer_name` | str | `Adam` | Optimizer: `Adam` or `AdamW` |

### DataLoaderConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | int | `4` | Samples per batch (per-GPU; global batch = `batch_size × num_GPUs` with multi-GPU) |
| `shuffle` | bool | `true` (train) / `false` (val) | Shuffle data each epoch |
| `num_workers` | int or `"auto"` | `0` | Parallel data loading workers (use with caching only). `"auto"` sizes from usable CPU count; see [Automatic worker count](#automatic-worker-count) |

### OptimizerConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `lr` | float | `1e-4` | Learning rate |
| `amsgrad` | bool | `false` | Enable AMSGrad variant |

### LRSchedulerConfig

Only one scheduler should be set at a time.

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

#### LinearWarmupLinearDecayConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `warmup_epochs` | int | `5` | Linear warmup epochs |
| `warmup_start_lr` | float | `0.0` | Starting LR for warmup |
| `end_lr` | float | `0.0` | Final LR at end of training |
| `max_epochs` | int | `null` | Total epochs (auto from trainer) |

### EarlyStoppingConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stop_training_on_plateau` | bool | `true` | Enable early stopping |
| `patience` | int | `10` | Epochs without improvement. For `embedding`, evaluations without improvement (one check every `eval.frequency` epochs) |
| `min_delta` | float | `1e-8` | Minimum improvement |

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

### EvalConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable epoch-end evaluation metrics |
| `frequency` | int | `1` | Evaluate every N epochs |
| `oks_stddev` | float | `0.025` | OKS standard deviation (pose models only) |
| `oks_scale` | float | `null` | OKS scale override (pose models only) |
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
