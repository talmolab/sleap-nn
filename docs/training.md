# Training Models

## Overview

  SLEAP-NN leverages a flexible, configuration-driven training workflow built on Hydra and OmegaConf. This guide will walk you through the essential steps for training pose estimation models using SLEAP-NN, whether you prefer the command-line interface or Python APIs. 

!!! info "Using uv workflow"
    This section assumes you have `sleap-nn` installed. If not, refer to the [installation guide](installation.md).
    
    - If you're using the `uvx` workflow, you do **not** need to install anything. (See [installation using uvx](installation.md#installation-using-uvx) for more details.)
    
    - If you are using `uv sync` or `uv add` installation methods, add `uv run` as a prefix to all CLI commands shown below, for example:

          `uv run sleap-nn train ...`

This section explains how to train a model using an existing configuration file. If you need help creating or editing a config, see the [configuration guide](config.md). 

### Using CLI

For a complete list of all CLI options, see the [CLI Reference](cli.md#sleap-nn-train).

To train a model using CLI,
```bash
sleap-nn train --config-name config --config-dir /path/to/config_dir
```

- `config-name` or `-c`: Name of the config file
- `config-dir` or `-d`: Path to the config file

If your config file is in the path: `/path/to/config_dir/config.yaml`, then `config-name` would be `config.yaml` and `config-dir` would be `/path/to/config_dir`.


Override any configuration from command line:

```bash
# Train on list of .slp files
sleap-nn train -c config -d /path/to/config_dir/ "data_config.train_labels_path=[labels.pkg.slp,labels.pkg.slp]"

# Change batch size
sleap-nn train -c config -d /path/to/config_dir/ trainer_config.train_data_loader.batch_size=8 trainer_config.val_data_loader.batch_size=8 "data_config.train_labels_path=[labels.pkg.slp]"

# Set number of GPUs to be used
sleap-nn train -c config -d /path/to/config_dir/ trainer_config.trainer_devices=1 "data_config.train_labels_path=[labels.pkg.slp]"

# Change learning rate
sleap-nn train -c config -d /path/to/config_dir/ trainer_config.optimizer.lr=5e-4 "data_config.train_labels_path=[labels.pkg.slp]"
```

!!! note "Training TopDown Model"
    For topdown, we need to train two models (centroid → instance). To know more about topdown models, refer [Model types](models.md/#-top-down)

    ```bash
    # Train centroid model
    sleap-nn train \
        -d /path/to/config_dir/ \
        -c centroid_unet \
        "data_config.train_labels_path=[labels.pkg.slp]"

    # Train centered instance model
    sleap-nn train \
        -d /path/to/config_dir/ \
        -c centered_instance_unet \
        "data_config.train_labels_path=[labels.pkg.slp]"
    ```

#### Remapping Video Paths

When training on a different machine than where your labels were created, the video file paths in your `.slp` file may no longer be valid. SLEAP-NN provides three CLI options to remap video paths at training time without modifying your labels file.

**`--video-paths` / `-v`:**

Replace video paths by specifying new paths in order. The order must match the order of videos in your labels file.

```bash
# Single video
sleap-nn train -c config -d /path/to/config_dir \
    --video-paths /new/path/to/video.mp4

# Multiple videos (specify multiple times)
sleap-nn train -c config -d /path/to/config_dir \
    --video-paths /new/path/to/video1.mp4 \
    --video-paths /new/path/to/video2.mp4
```

**`--video-path-map`:**

Map specific old video paths to new paths. Takes two arguments: the old path and the new path.

```bash
# Single mapping
sleap-nn train -c config -d /path/to/config_dir \
    --video-path-map /old/path/video.mp4 /new/path/video.mp4

# Multiple mappings (specify multiple times)
sleap-nn train -c config -d /path/to/config_dir \
    --video-path-map /old/path/video1.mp4 /new/path/video1.mp4 \
    --video-path-map /old/path/video2.mp4 /new/path/video2.mp4
```

**`--prefix-map`:**

Replace path prefixes for all videos that share the same prefix. This is useful when moving data between machines where only the base directory differs.

```bash
# Replace prefix for all matching videos
sleap-nn train -c config -d /path/to/config_dir \
    --prefix-map /old/server/data /new/local/data
```

For example, if your labels file references:

- `/old/server/data/experiment1/video.mp4`
- `/old/server/data/experiment2/video.mp4`

Using `--prefix-map /old/server/data /new/local/data` will remap both to:

- `/new/local/data/experiment1/video.mp4`
- `/new/local/data/experiment2/video.mp4`

!!! warning "Choose one option"
    You can only use one of `--video-paths`, `--video-path-map`, or `--prefix-map` at a time.

### Using `ModelTrainer` API

To train a model using the sleap-nn APIs:

  ```python linenums="1"
  from omegaconf import OmegaConf
  from sleap_nn.training.model_trainer import ModelTrainer

  # load config
  config = OmegaConf.load("config.yaml")

  # create trainer instance
  trainer = ModelTrainer.get_model_trainer_from_config(config=config)

  # start training
  trainer.train()
  ```

If you have a cutsom labels object which is not in a slp file:
```python linenums="1"
from omegaconf import OmegaConf
import sleap_io as sio
from sleap_nn.training.model_trainer import ModelTrainer

# create `sio.Labels` objects
train_labels = sio.load_slp("train.slp")
val_labels = sio.load_slp("val.slp")

# load config
config = OmegaConf.load("config.yaml")

# create trainer instance
trainer = ModelTrainer.get_model_trainer_from_config(config=config, train_labels=[train_labels], val_labels=[val_labels])

# start training
trainer.train()
```

## Training without Config

If you prefer not to create a large custom config file, you can quickly train a model by calling the `train()` function directly and passing your desired parameters as arguments.

This approach is much simpler than manually specifying every parameter for each model component. For example, instead of defining all the details for a UNet backbone, you can just set `backbone_config="unet_medium_rf"` or `"unet_large_rf"`, and the appropriate preset values will be used automatically. The same applies to head configurations—just specify the desired preset (e.g., `"bottomup"`), and the defaults are handled for you. To look into the preset values for each of the backbones and heads, refer the [model configs](../api/config/model_config/#sleap_nn.config.model_config).

For a full list of available arguments and their descriptions, see the [`train()` API reference](../api/train/#sleap_nn.train.train) in the documentation.

```python linenums="1"
from sleap_nn.train import train

train(
    train_labels_path=["labels.slp"],
    backbone_config="unet_medium_rf",
    head_configs="bottomup",
    save_ckpt=True,
)

```

Applying data augmentation is also much simpler—you can just specify the augmentation names directly (as a string or list), instead of writing out a full configuration.

```python linenums="1"
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
  If you set `trainer_config.use_wandb = True` and provide a valid `trainer_config.wandb_config`, all key training metrics—including losses, training/validation times, and visualizations (if `wandb_config.save_viz_imgs_wandb` is set to True)—are automatically logged to your WandB project. This makes it easy to monitor progress and compare runs.

#### WandB Visualization Options

Control how training visualizations appear in WandB with these options:

```yaml
trainer_config:
  use_wandb: true
  wandb_config:
    # ... other wandb settings ...
    viz_enabled: true         # Pre-rendered matplotlib images (default)
    viz_boxes: false          # Interactive keypoint boxes with sliders
    viz_masks: false          # Confidence map overlay masks
    viz_box_size: 5.0         # Size of keypoint boxes in pixels
    viz_confmap_threshold: 0.1  # Threshold for confidence map masks
```

| Option | Description | Default |
|--------|-------------|---------|
| `viz_enabled` | Log pre-rendered matplotlib images to WandB | `True` |
| `viz_boxes` | Log interactive keypoint boxes (enables epoch slider) | `False` |
| `viz_masks` | Log confidence map overlay masks | `False` |
| `viz_box_size` | Size of keypoint boxes in pixels | `5.0` |
| `viz_confmap_threshold` | Minimum value to display in confmap masks | `0.1` |

!!! tip "Interactive Epoch Slider"
    Enable `viz_boxes: true` to get an interactive slider in WandB that lets you scrub through epochs and see how predictions improve over training.

#### Per-Head Loss Monitoring

Multi-head models (BottomUp, MultiClassBottomUp, MultiClassTopDown) now log individual head losses in addition to the total loss:

- `train_confmap_loss` / `val_confmap_loss` - Confidence map head loss
- `train_paf_loss` / `val_paf_loss` - Part affinity field head loss (bottom-up only)

This helps diagnose when individual heads aren't learning effectively—for example, if confidence maps converge but PAFs plateau.

### Checkpointing & Artifacts
  For every training run, a dedicated checkpoint directory is created. This directory contains:

  - The original user-provided config (`initial_config.yaml`)
  - The full training config with computed values (`training_config.yaml`)
  - The best model weights (`best.ckpt`) when `trainer_config.save_ckpt` is set to `True`
  - The training and validation SLP files used
  - A CSV log tracking train/validation loss, times, and learning rate across epochs


### Visualizing training performance
  To help understand model performance, SLEAP-NN can generate visualizations of model predictions (e.g., confidence maps) after each epoch when `trainer_config.visualize_preds_during_training` is set to `True`. By default, these images are saved temporarily (deleted after training is completed), but you can configure the system to keep them by setting `trainer_config.keep_viz` to `True`.

## Advanced Options

### Fine-tuning / Transfer Learning

SLEAP-NN makes it easy to fine-tune or transfer-learn from existing models. To initialize your model with pre-trained weights, simply set the following options in your configuration:

- `model_config.pretrained_backbone_weights`: Path to a checkpoint file (or `.h5` file path from SLEAP <=1.4 - only UNet backbone is supported) containing the backbone weights you want to load. This will initialize the backbone (e.g., UNet, Swin Transformer) with the specified weights.
- `model_config.pretrained_head_weights`: Path to a checkpoint file (or `.h5` file from SLEAP ≤1.4 - only UNet backbone is supported) to initialize the model's head weights (e.g., for bottomup or topdown heads). The head and backbone weights are usually the same checkpoint, but you can specify a different file here if you want to use separate weights for the head (for example, when adapting a model to a new output head or architecture).

By specifying these options, your model will be initialized with the provided weights, allowing you to fine-tune on new data or adapt to a new task. You can use this for transfer learning from a model trained on a different dataset, or to continue training with a modified head or backbone.

### Resume Training

To resume training from a previous checkpoint (restoring both model weights and optimizer state), simply provide the path to your previous checkpoint file using the `trainer_config.resume_ckpt_path` option. This allows you to continue training seamlessly from where you left off.

```bash
sleap-nn train \
    -c config \
    -d /path/to/config_dir/ \ 
    trainer_config.resume_ckpt_path=/path/to/prv_trained/checkpoint.ckpt \
    "data_config.train_labels_path=[labels.pkg.slp]"
```

### Multi-GPU Training

To automatically configure the accelerator and number of devices, set:
```yaml
trainer_config:
  ckpt_dir: models
  run_name: multi_gpu_training_1
  trainer_accelerator: "auto"
  trainer_devices:
  trainer_device_indices:
  trainer_strategy: "auto"
```

To set the number of gpus to be used and the accelerator:
```yaml
trainer_config:
ckpt_dir: models
  run_name: multi_gpu_training_1
  trainer_accelerator: "gpu"
  trainer_devices: 4
  trainer_device_indices:
  trainer_strategy: "ddp"
```

To set the devices to use (use first and third gpu):
```yaml
trainer_config:
ckpt_dir: models
  run_name: multi_gpu_training_1
  trainer_accelerator: "gpu"
  trainer_devices: 2
  trainer_device_indices:
    - 0
    - 2
  trainer_strategy: "ddp"
```

!!! note "Training steps in multi-gpu setting"
    - In a multi-gpu training setup, the effective steps during training would be `config.trainer_config.trainer_steps_per_epoch` / `config.trainer_config.trainer_devices`. 
    - If validation labels are not provided in a multi-GPU training setup, we now ensure deterministic splitting of labels into train/val sets by seeding with 42 (when no seed is given). This prevents each GPU worker from producing a different split. To generate a different train-val split, set a custom seed via `config.trainer_config.seed`.
!!! note "Multi-node training"
    Multi-node trainings have not been validated and should be considered experimental.

## Best Practices

1. **Start Simple**: Begin with default configurations
2. **Cache data**: If you want to get faster training time, consider caching the images on memory (or disk) by setting the relevant `data_config.data_pipeline_fw`. (num_workers could be set >0 if caching frameworks are used!)
3. **Monitor Overfitting**: Watch validation metrics
4. **Adjust Learning Rate**: Use learning rate scheduling
5. **Data Augmentation**: Enable augmentations for better generalization
6. **Early Stopping**: Prevent overfitting with early stopping callback.

## Troubleshooting

### Out of Memory

For large models or datasets:

- Reduce `batch_size`
- Reduce model size (fewer filters/layers)
- Reduce number of workers

### Slow Training

- Use caching methods (`data_config.data_pipeline_fw`)
- Increase `num_workers` for data loading
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