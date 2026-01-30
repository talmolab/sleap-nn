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

Scale batch size with GPU count:

```yaml
# Single GPU
trainer_config:
  train_data_loader:
    batch_size: 4

# 4 GPUs - increase batch size
trainer_config:
  train_data_loader:
    batch_size: 16  # 4 per GPU
```

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

??? question "Slow multi-GPU training"
    - Ensure NVLink is available for fast GPU communication
    - Use local SSD for disk caching
    - Reduce logging frequency

---

## Multi-Node Training

!!! warning "Experimental"
    Multi-node training is not fully validated. Use at your own risk.

For multi-node, configure your cluster's job scheduler to set the appropriate environment variables (`MASTER_ADDR`, `MASTER_PORT`, etc.).
