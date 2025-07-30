# Training Models

## Overview

sleap-nn uses a configuration-based training system powered by Hydra and PyTorch Lightning. This guide covers the basics of training different model types.

## Basic Training Command

```bash
python -m sleap_nn.train --config-name config
```

### Command Line Overrides

Override any configuration from command line:

```bash
# Change batch size
python -m sleap_nn.train --config-name config data_config.batch_size=32

# Use different GPU
python -m sleap_nn.train --config-name config trainer_config.devices=1

# Change learning rate
python -m sleap_nn.train --config-name config trainer_config.learning_rate=5e-4
```

## Model Types

### Single Instance

For videos with exactly one animal:

```bash
python -m sleap_nn.train \
    --config-path configs \
    --config-name single_instance_unet
```

### Top-Down

Two-stage approach (centroid → instance):

```bash
# Train centroid model
python -m sleap_nn.train \
    --config-path configs \
    --config-name centroid_unet

# Train centered instance model
python -m sleap_nn.train \
    --config-path configs \
    --config-name centered_instance_unet
```

### Bottom-Up

Direct multi-instance with PAFs:

```bash
python -m sleap_nn.train \
    --config-path configs \
    --config-name bottomup_unet
```

## Configuration Options

### Data Configuration

```yaml
data_config:
  data_path: /path/to/labels.slp
  video_backend: "pyav"
  batch_size: 16
  num_workers: 4
  augmentation:
    rotate: true
    scale: true
    translate: true
```

### Model Configuration

```yaml
model_config:
  architecture: "unet"
  backbone_config:
    filters: 32
    depth: 5
  head_config:
    output_stride: 2
    loss_weight: 1.0
```

### Training Configuration

```yaml
trainer_config:
  max_epochs: 100
  learning_rate: 1e-3
  accelerator: "gpu"
  devices: 1
  precision: 16
```

## Monitoring Training

### TensorBoard

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir lightning_logs/
```

### Checkpoints

Models are saved automatically:
- Best checkpoint: `lightning_logs/version_X/checkpoints/best.ckpt`
- Last checkpoint: `lightning_logs/version_X/checkpoints/last.ckpt`

## Advanced Options

### Resume Training

```bash
python -m sleap_nn.train \
    --config-name config \
    trainer_config.ckpt_path=/path/to/checkpoint.ckpt
```

### Multi-GPU Training

```yaml
trainer_config:
  accelerator: "gpu"
  devices: 4
  strategy: "ddp"
```

### Mixed Precision

```yaml
trainer_config:
  precision: 16  # or "bf16" for newer GPUs
```

## Best Practices

1. **Start Simple**: Begin with default configurations
2. **Monitor Overfitting**: Watch validation metrics
3. **Adjust Learning Rate**: Use learning rate scheduling
4. **Data Augmentation**: Enable for better generalization
5. **Early Stopping**: Prevent overfitting with callbacks

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Enable gradient accumulation
- Use mixed precision training
- Reduce model size (fewer filters/layers)

### Slow Training

- Increase `num_workers` for data loading
- Enable mixed precision
- Use SSD for data storage
- Check GPU utilization

### Poor Performance

- Increase training data
- Adjust augmentation parameters
- Try different architectures
- Tune hyperparameters

## Next Steps

- [Running Inference](inference.md)
- [Configuration Guide](config.md)
- [Model Architecture Guide](architectures.md)