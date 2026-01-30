# Training Models

Train pose estimation models with SLEAP-NN.

!!! info "Using uv workflow"
    - If using `uvx`, no installation needed
    - If using `uv sync`, prefix commands with `uv run`:
      ```bash
      uv run sleap-nn train ...
      ```

---

## Basic Training

### Using CLI

```bash
sleap-nn train --config config.yaml
```

Or with separate config directory and name:

```bash
sleap-nn train --config-dir /path/to/configs --config-name my_config
```

### Using Python API

```python
from omegaconf import OmegaConf
from sleap_nn.train import run_training

config = OmegaConf.load("config.yaml")
run_training(config=config)
```

### With Custom Labels

```python
import sleap_io as sio
from sleap_nn.train import run_training

config = OmegaConf.load("config.yaml")
train_labels = sio.load_slp("train.slp")
val_labels = sio.load_slp("val.slp")

run_training(config=config,
            train_labels=[train_labels],
            val_labels=[val_labels])
```

---

## Config Overrides

Override any config value from the command line:

```bash
# Change epochs
sleap-nn train --config config.yaml trainer_config.max_epochs=200

# Change learning rate
sleap-nn train --config config.yaml trainer_config.optimizer.lr=0.0005

# Change batch size
sleap-nn train --config config.yaml trainer_config.train_data_loader.batch_size=8

# Set training data
sleap-nn train --config config.yaml "data_config.train_labels_path=[train.slp]"

# Set number of GPUs
sleap-nn train --config config.yaml trainer_config.trainer_devices=1
```

---

## Video Path Remapping

When training on a different machine than where labels were created:

=== "Replace by order"
    ```bash
    sleap-nn train --config config.yaml \
        --video-paths /new/path/video1.mp4 \
        --video-paths /new/path/video2.mp4
    ```

=== "Map specific paths"
    ```bash
    sleap-nn train --config config.yaml \
        --video-path-map /old/video.mp4 /new/video.mp4
    ```

=== "Replace prefix"
    ```bash
    sleap-nn train --config config.yaml \
        --prefix-map /old/server/data /new/local/data
    ```

!!! warning "Choose one option"
    You can only use one of `--video-paths`, `--video-path-map`, or `--prefix-map` at a time.

---

## Training Without Config

Quick training with minimal setup using presets:

```python
from sleap_nn.train import train

train(
    train_labels_path=["labels.slp"],
    backbone_config="unet_medium_rf",  # or "unet_large_rf"
    head_configs="bottomup",           # or "single_instance", etc.
    save_ckpt=True,
)
```

With augmentation:

```python
train(
    train_labels_path=["labels.slp"],
    backbone_config="unet_medium_rf",
    head_configs="bottomup",
    use_augmentations_train=True,
    intensity_aug="uniform_noise",
    geometric_aug=["rotation", "scale"],
)
```

---

## Top-Down Training

Top-down models need two separate training runs:

```bash
# Train centroid model
sleap-nn train -d /path/to/configs -c centroid_unet \
    "data_config.train_labels_path=[labels.pkg.slp]"

# Train centered instance model
sleap-nn train -d /path/to/configs -c centered_instance_unet \
    "data_config.train_labels_path=[labels.pkg.slp]"
```

---

## Monitoring Training

### Weights & Biases Integration

Enable WandB logging:

```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: your-username
    project: your-project
```

#### Visualization Options

```yaml
trainer_config:
  wandb:
    viz_enabled: true         # Pre-rendered matplotlib images
    viz_boxes: false          # Interactive keypoint boxes with epoch slider
    viz_masks: false          # Confidence map overlay masks
    viz_box_size: 5.0         # Size of keypoint boxes in pixels
    viz_confmap_threshold: 0.1  # Threshold for confmap masks
```

| Option | Description | Default |
|--------|-------------|---------|
| `viz_enabled` | Log pre-rendered images | `True` |
| `viz_boxes` | Interactive keypoint boxes (epoch slider) | `False` |
| `viz_masks` | Confidence map overlay masks | `False` |

!!! tip "Interactive Epoch Slider"
    Enable `viz_boxes: true` to scrub through epochs and see predictions improve.

### Per-Head Loss Monitoring

Multi-head models log individual losses:

- `train_confmap_loss` / `val_confmap_loss`
- `train_paf_loss` / `val_paf_loss` (bottom-up only)

This helps diagnose when individual heads aren't learning effectively.

### Training Visualizations

Enable prediction visualizations during training:

```yaml
trainer_config:
  visualize_preds_during_training: true
  keep_viz: false  # Set true to keep viz folder after training
```

---

## Checkpointing & Artifacts

Each training run creates a checkpoint directory with:

| File | Description |
|------|-------------|
| `best.ckpt` | Best model weights |
| `initial_config.yaml` | Original user config |
| `training_config.yaml` | Full config with computed values |
| `labels_gt.train.0.slp` | Training data split (ground truth) |
| `labels_gt.val.0.slp` | Validation data split (ground truth) |
| `labels_pr.train.slp` | Predictions on training data |
| `labels_pr.val.slp` | Predictions on validation data |
| `metrics.train.0.npz` | Training metrics |
| `metrics.val.0.npz` | Validation metrics |
| `training_log.csv` | Loss/metrics per epoch |

---

## Fine-tuning / Transfer Learning

Initialize with pre-trained weights:

```yaml
model_config:
  pretrained_backbone_weights: /path/to/best.ckpt
  pretrained_head_weights: /path/to/best.ckpt
```

Works with:

- Previous SLEAP-NN checkpoints

- Legacy SLEAP `.h5` files (UNet only)

---

## Resume Training

Resume from a previous checkpoint:

```bash
sleap-nn train --config config.yaml \
    trainer_config.resume_ckpt_path=/path/to/checkpoint.ckpt
```

This restores both model weights and optimizer state.

---

## Multi-GPU Training

For multi-GPU and distributed training, see the dedicated guide:

[:octicons-arrow-right-24: Multi-GPU Training Guide](multi-gpu.md)

---

## Performance Tips

### Enable Caching

```yaml
data_config:
  data_pipeline_fw: torch_dataset_cache_img_memory  # RAM caching
  # or
  data_pipeline_fw: torch_dataset_cache_img_disk    # Disk caching
```

With caching, you can use `num_workers > 0`:

```yaml
trainer_config:
  train_data_loader:
    num_workers: 4
```

!!! warning "Workers without caching"
    Keep `num_workers: 0` when not using caching.

---

## Best Practices

1. **Start Simple**: Begin with default configurations
2. **Cache Data and increase `num_workers`**: Use caching for faster training
3. **Use Augmentation**: Always enable for better generalization
4. **Early Stopping**: Prevents overfitting

---

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce model size (fewer filters)
- Reduce image size with `preprocessing.scale`

### Slow Training

- Enable caching
- Increase `num_workers` (with caching)
- Check GPU utilization

### Poor Performance

- Increase training data
- Adjust augmentation
- Try different architectures
- Tune hyperparameters

---

## Next Steps

- [:octicons-arrow-right-24: Running Inference](inference.md)
- [:octicons-arrow-right-24: Configuration Reference](../configuration/index.md)
- [:octicons-arrow-right-24: Model Architectures](../reference/models.md)
