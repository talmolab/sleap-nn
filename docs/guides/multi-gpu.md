# Multi-GPU Training

Scale training across multiple GPUs.

---

## Auto-Detection

Let SLEAP-NN detect available GPUs:

```yaml
trainer_config:
  trainer_accelerator: auto
  trainer_devices: auto
  trainer_strategy: auto
  run_name: multi_gpu_training
```

---

## Specify GPU Count

```yaml
trainer_config:
  trainer_accelerator: gpu
  trainer_devices: 4          # Use 4 GPUs
  trainer_strategy: ddp       # Distributed Data Parallel
  run_name: multi_gpu_training
```

---

!!! note "Freeze schedules under DDP"
    With `trainer_config.freeze_schedule`, `find_unused_parameters` is turned on
    automatically (frozen parameters get no gradient), and
    `ddp_find_unused_parameters_false` is refused.

---

## Specific GPUs

Use specific GPU indices:

```yaml
trainer_config:
  trainer_accelerator: gpu
  trainer_devices: 2
  trainer_device_indices:
    - 0
    - 2  # Use first and third GPU
  trainer_strategy: ddp
  run_name: multi_gpu_training
```

---

## Validation Splitting

Multi-GPU training seeds validation splits for reproducibility:

```yaml
trainer_config:
  seed: 42  # Fixed seed for reproducible train/val split
```

Without a seed, the split is seeded with 42 by default to ensure all GPU workers get the same split.

---

## Best Practices

### Batch Size

The configured `batch_size` is **per-GPU**, not global. With DDP, each GPU runs its own
data loader with the full `batch_size`, so the effective (global) batch size is
`batch_size × num_GPUs`:

```yaml
# Single GPU -> global batch size = 4
trainer_config:
  train_data_loader:
    batch_size: 4

# 4 GPUs -> global batch size = 4 × 4 = 16
trainer_config:
  train_data_loader:
    batch_size: 4  # still 4 per GPU
```

Keep `batch_size` at the largest value that fits in a single GPU's memory; the effective
batch already scales with GPU count. If you want to match a single-GPU global batch size,
divide `batch_size` by the number of GPUs.

### Steps per Epoch

An epoch is one pass over the training data, whatever the GPU count. With DDP every
GPU draws its own batches from its shard of the data, so `train_steps_per_epoch`
counts batches **summed over all GPUs**, and each GPU (and so the optimizer) runs
`train_steps_per_epoch / num_GPUs` steps per epoch:

```yaml
trainer_config:
  train_data_loader:
    batch_size: 128            # per GPU
  train_steps_per_epoch: 1170  # 1170 x 128 crops per epoch in total
  trainer_devices: 4           # -> 292 optimizer steps per GPU per epoch
```

The log prints both numbers at startup. `min_train_steps_per_epoch` is counted the
same way. Epoch-based settings (`max_epochs`, warmup epochs, `eval.frequency`) mean
the same thing on any number of GPUs.

### Augmentation

Each GPU's dataloader workers are seeded from their own PyTorch seed **and** the GPU's
global rank, so the GPUs draw independent augmentation streams that stay reproducible
for a fixed `seed`. Keep `num_workers > 0`: with `num_workers: 0` augmentation runs in
each GPU's main process, which Lightning seeds identically on every GPU, so all GPUs
would draw the same random augmentations (a warning is logged).

### Evaluation and Checkpoint Selection

Validation is sharded across GPUs. With `trainer_config.eval.enabled`, the epoch-end
evaluation (OKS/PCK for pose models, centroid distances, mask IoU, box AP) gathers
every GPU's predictions, de-duplicates the samples the distributed sampler repeats to
even out the shards, scores the whole validation set once on rank 0, and shares the
resulting `eval/val/*` metrics with every GPU. All GPUs therefore agree on the
monitored value, so a checkpoint or early stopping on an eval metric works as on one
GPU:

```yaml
trainer_config:
  eval:
    enabled: true
    frequency: 5
  model_ckpt:
    monitor: eval/val/mOKS   # also used by early stopping
    mode: max
```

Early stopping always watches the same metric as `best.ckpt`; on an `eval/*` metric
it checks (and counts patience) on evaluation epochs only.

### Launchers

`sleap-nn train` with `trainer_devices > 1` uses Lightning's own launcher: rank 0 runs
the setup (data loading, caching), and the other ranks are started when training
begins and repeat the setup after rank 0 has finished it. Under an external launcher
(`torchrun`, SLURM `srun`) all ranks run the setup at once; sleap-nn then starts the
process group before the setup, so the rank-0-writes-the-disk-cache barrier really
waits.

### Measuring Loader Throughput

To check whether the CPUs can keep N GPUs fed, `scripts/ddp_loader_throughput.py`
builds each rank's train loader exactly as training does (shard, workers,
augmentation) without a model or GPU, and reports the aggregate crops/s and CPU cores
used:

```bash
uv run python scripts/ddp_loader_throughput.py config.yaml --ranks 4 --workers 4 8 16
```

Compare the aggregate crops/s against `GPUs x batch_size / step_time` of a GPU-bound
run.

### Caching

Use disk caching for multi-GPU (memory caching doesn't share across processes):

```yaml
data_config:
  data_pipeline_fw: torch_dataset_cache_img_disk
  cache_img_path: /path/to/shared/cache
```

---

## Troubleshooting

??? question "GPUs not detected"
    ```bash
    # Check visible GPUs
    sleap-nn system
    ```

??? question "Training freezes at epoch start"
    This is usually a DDP synchronization issue. Try:

    - Use disk caching instead of memory caching
    - Reduce `num_workers`
    - Check that all GPUs are accessible

??? question "Out of memory on some GPUs"
    - Reduce batch size
    - Use `trainer_device_indices` to skip problematic GPUs
    - Check for other processes using GPU memory

??? question "\"Grad strides do not match bucket view strides\" warning"
    DDP compares each gradient's strides with its communication buffer exactly. A 1x1
    convolution fed channels-last activations (any ViT backbone) gets a weight
    gradient whose strides differ only in its size-1 dimensions: the same memory
    layout, so nothing is copied differently or slowed down. sleap-nn silences that
    case; the warning still shows for any real layout mismatch.

??? question "Slow multi-GPU training"
    - Ensure NVLink is available for fast GPU communication
    - Use local SSD for disk caching
    - Reduce logging frequency

---

## Multi-Node Training

!!! warning "Experimental"
    Multi-node training is not fully validated. Use at your own risk.

For multi-node, configure your cluster's job scheduler to set the appropriate environment variables (`MASTER_ADDR`, `MASTER_PORT`, etc.).
