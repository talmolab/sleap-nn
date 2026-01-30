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
    num_workers: 0    # >0 only with caching
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0
```

!!! warning "Workers without caching"
    Only use `num_workers > 0` with data caching enabled.

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
    patience: 10        # Epochs without improvement
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
    save_last: false  # Also save last.ckpt
  resume_ckpt_path: null  # Resume from this path
```

---

## Training Control

```yaml
trainer_config:
  min_train_steps_per_epoch: 200  # Minimum steps
  train_steps_per_epoch: null     # Exact steps (null=auto)
  enable_progress_bar: true
  seed: null                      # Random seed
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

For SLEAP GUI communication:

```yaml
trainer_config:
  zmq:
    publish_port: 9001
    controller_port: 9000
    controller_polling_timeout: 10
```

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
| `seed` | int | `null` | Random seed for reproducibility |
| `trainer_accelerator` | str | `auto` | Hardware: `auto`, `gpu`, `cpu`, `mps` |
| `trainer_devices` | int/str | `null` | Number of devices or `auto` |
| `trainer_device_indices` | list | `null` | Specific device indices (e.g., `[0, 2]`) |
| `trainer_strategy` | str | `auto` | Strategy: `auto`, `ddp`, `fsdp` |
| `profiler` | str | `null` | PyTorch profiler: `simple`, `advanced`, `pytorch` |
| `enable_progress_bar` | bool | `true` | Show training progress |
| `min_train_steps_per_epoch` | int | `200` | Minimum batches per epoch |
| `train_steps_per_epoch` | int | `null` | Exact steps per epoch (null = auto) |
| `visualize_preds_during_training` | bool | `false` | Save prediction visualizations |
| `keep_viz` | bool | `false` | Keep viz folder after training |
| `use_wandb` | bool | `false` | Enable WandB logging |
| `resume_ckpt_path` | str | `null` | Path to checkpoint to resume from |
| `optimizer_name` | str | `Adam` | Optimizer: `Adam` or `AdamW` |

### DataLoaderConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | int | `4` | Samples per batch |
| `shuffle` | bool | `true` (train) / `false` (val) | Shuffle data each epoch |
| `num_workers` | int | `0` | Parallel data loading workers (use with caching only) |

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
| `patience` | int | `10` | Epochs without improvement |
| `min_delta` | float | `1e-8` | Minimum improvement |

### ModelCkptConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `save_top_k` | int | `1` | Keep N best models |
| `save_last` | bool | `null` | Also save last.ckpt |

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
| `oks_stddev` | float | `0.025` | OKS standard deviation |
| `oks_scale` | float | `null` | OKS scale override |

### ZMQConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `publish_port` | int | `null` | Port for publishing updates |
| `controller_port` | int | `null` | Port for receiving commands |
| `controller_polling_timeout` | int | `10` | Polling timeout in microseconds |
