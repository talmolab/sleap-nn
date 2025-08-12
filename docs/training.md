# Training Models

## Overview

  SLEAP-NN leverages a flexible, configuration-driven training workflow built on Hydra and OmegaConf. This guide will walk you through the essential steps for training pose estimation models using SLEAP-NN, whether you prefer the command-line interface or Python APIs.

!!! note
    Training is only supported with SLEAP label files with ground truth annotations in `.slp` or `.pkg.slp` format.

## Training with Config

This section explains how to train a model using an existing configuration file. If you need help creating or editing a config, see the [configuration guide](config.md). 

### Using CLI

To train a model using CLI, 
```bash
sleap-nn-train --config-name config --config-path path/to/config_dir
```

Override any configuration from command line:

```bash
# Train on list of .slp files
sleap-nn-train --config-name config --config-path path/to/config_dir "data_config.train_labels_path=[labels.pkg.slp,labels.pkg.slp]"

# Change batch size
sleap-nn-train --config-name config --config-path path/to/config_dir data_config.batch_size=32 "data_config.train_labels_path=[labels.pkg.slp]"

# Use different GPU
sleap-nn-train --config-name config --config-path path/to/config_dir trainer_config.devices=1 "data_config.train_labels_path=[labels.pkg.slp]"

# Change learning rate
sleap-nn-train --config-name config --config-path path/to/config_dir trainer_config.learning_rate=5e-4 "data_config.train_labels_path=[labels.pkg.slp]"
```

!!! note
    For topdown, we need to train two models (centroid → instance):

```bash
# Train centroid model
sleap-nn-train \
    --config-path configs \
    --config-name centroid_unet \
    "data_config.train_labels_path=[labels.pkg.slp]"

# Train centered instance model
sleap-nn-train \
    --config-path configs \
    --config-name centered_instance_unet \ 
    "data_config.train_labels_path=[labels.pkg.slp]"
```  

### Using `ModelTrainer` API

```python
from omegaconf import OmegaConf
from sleap_nn.training.model_trainer import ModelTrainer

config = OmegaConf.load("config.yaml")
trainer = ModelTrainer.get_model_trainer_from_config(config=config)
trainer.train()
```

If you have a cutsom labels object which is not in a slp file:
```python
from omegaconf import OmegaConf
from sleap_nn.training.model_trainer import ModelTrainer

config = OmegaConf.load("config.yaml")
trainer = ModelTrainer.get_model_trainer_from_config(config=config, train_labels=[train_labels], val_labels=[val_labels])
trainer.train()
```

## Training without Config

If you prefer not to create a large custom config file, you can quickly train a model by calling the `train()` function directly and passing your desired parameters as arguments.

This approach is much simpler than manually specifying every parameter for each model component. For example, instead of defining all the details for a UNet backbone, you can just set `backbone_config="unet_medium_rf"` or `"unet_large_rf"`, and the appropriate preset values will be used automatically. The same applies to head configurations—just specify the desired preset (e.g., `"bottomup"`), and the defaults are handled for you. To look into the preset values for each of the backbones and heads, refer the [model configs](../api/config/model_config/#sleap_nn.config.model_config).

For a full list of available arguments and their descriptions, see the [`train()` API reference](../api/train/#sleap_nn.train.train) in the documentation.

```python
from sleap_nn.train import train

train(
    train_labels_path=["labels.slp"], # or list of labels
    backbone_config="unet_medium_rf",
    head_configs="bottomup",
    save_ckpt=True,
)

```

Applying data augmentation is also much simpler—you can just specify the augmentation names directly (as a string or list), instead of writing out a full configuration.

```python
from sleap_nn.train import train

train(
    train_labels_path=["labels.slp"], # or list of labels
    backbone_config="unet_medium_rf",
    head_configs="bottomup",
    save_ckpt=True,
    use_augmentations_train=True,
    intensity_aug="uniform_noise",
    geometric_aug=["rotation", "scale"]
)

```

## Monitoring Training

### Weights & Biases (WandB) Integration
  If you set `trainer_config.use_wandb = True` and provide a valid `trainer_config.wandb_config`, all key training metrics—including losses, training/validation times, and visualizations—are automatically logged to your WandB project. This makes it easy to monitor progress and compare runs.

### Checkpointing & Artifacts
  For every training run, a dedicated checkpoint directory is created. This directory contains:

  - The original user-provided config (`initial_config.yaml`)
  - The full training config with computed values (`training_config.yaml`)
  - The best model weights (`best.ckpt`) when `trainer_config.save_ckpt` is set to `True`
  - The training and validation SLP files used
  - A CSV log tracking train/validation loss, times, and learning rate across epochs


### Training Visualization
  To help understand model performance, SLEAP-NN can generate visualizations of model predictions (e.g., confidence maps) after each epoch when `visualize_preds_during_training` is set to `True`. By default, these images are saved temporarily (deleted after training is completed), but you can configure the system to keep them by setting `trainer_config.keep_viz` to `True`. If WandB logging is enabled, these visualizations are also uploaded to your WandB dashboard.

## Advanced Options

### Resume Training

To resume training from a previous checkpoint, 

```bash
sleap-nn-train \
    --config-name config \
    --config-path path/to/config_dir \ 
    trainer_config.ckpt_path=/path/to/checkpoint.ckpt \
    trainer_config.resume_ckpt_path=/path/to/prv_trained/checkpoint.ckpt \
    "data_config.train_labels_path=[labels.pkg.slp]"
```

### Multi-GPU Training

To automatically configure the accelerator and number of devices, set:
```yaml
trainer_config:
  trainer_accelerator: "auto"
  trainer_devices: "auto"
  trainer_strategy: "auto"
```

To set the number of gpus to be used and the accelerator:
```yaml
trainer_config:
  trainer_accelerator: "gpu"
  trainer_devices: 4
  trainer_strategy: "ddp"
```

!!! note
    In a multi-gpu training setup, the effective steps during training would be the given `config.trainer_config.trainer_steps_per_epoch` / `config.trainer_config.trainer_devices`. 
!!! note
    Multi-node trainings have not been validated and should be considered experimental.

## Best Practices

1. **Start Simple**: Begin with default configurations
2. **Cache data**: If you want to get faster training time, consider caching the images on memory (or disk) by setting the relevant `data_config.data_pipeline_fw`.
3. **Monitor Overfitting**: Watch validation metrics
4. **Adjust Learning Rate**: Use learning rate scheduling
5. **Data Augmentation**: Enable augmentations for better generalization
6. **Early Stopping**: Prevent overfitting with early stopping callback.

## Troubleshooting

### Out of Memory

For large models or datasets:

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
- [Model Architecture Guide](models.md)