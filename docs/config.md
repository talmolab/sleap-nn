# Configuration Guide

## Overview

This document provides a detailed guide to all configuration options available for training and running inference with sleap-nn models.

The config file has three main sections:

1. **[`data_config`](#data-configuration-data_config)**: Creating a data pipeline  
2. **[`model_config`](#model-configuration-model_config)**: Initialize the sleap-nn backbone and head models  
3. **[`trainer_config`](#trainer-configuration-trainer_config)**: Hyperparameters required to train the model with Lightning

## Basic Configuration Structure

### Sample configuration format

```yaml
data_config:
  train_labels_path: 
    - path/to/your/training_data.slp
  val_labels_path:
    - path/to/your/validation_data.slp
  validation_fraction: 0.1
  provider: LabelsReader
  user_instances_only: true
  data_pipeline_fw: torch_dataset
  preprocessing:
    ensure_rgb: false
    ensure_grayscale: false
    scale: 1.0
    crop_hw: null
    min_crop_size: 100
  use_augmentations_train: true
  augmentation_config:
    intensity:
      uniform_noise_p: 0.0
      gaussian_noise_p: 0.0
      contrast_p: 0.0
      brightness_p: 0.0
    geometric:
      rotation_min: -15.0
      rotation_max: 15.0
      scale_min: 0.9
      scale_max: 1.1
      affine_p: 1.0

model_config:
  init_weights: default
  pretrained_backbone_weights: null
  pretrained_head_weights: null
  backbone_config:
    unet:
      in_channels: 1
      kernel_size: 3
      filters: 16
      filters_rate: 2.0
      max_stride: 16
      middle_block: true
      up_interpolate: true
      stacks: 1
      convs_per_block: 2
      output_stride: 2
    convnext: null
    swint: null
  head_configs:
    single_instance:
      confmaps:
        part_names: null
        sigma: 2.5
        output_stride: 2
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null

trainer_config:
  train_data_loader:
    batch_size: 4
    shuffle: true
    num_workers: 0
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0
  model_ckpt:
    save_top_k: 1
    save_last: false
  trainer_devices: auto
  trainer_accelerator: auto
  enable_progress_bar: true
  min_train_steps_per_epoch: 200
  visualize_preds_during_training: true
  keep_viz: false
  max_epochs: 200
  seed: 0
  use_wandb: false
  save_ckpt: true
  save_ckpt_path: your_model_name
  optimizer_name: Adam
  optimizer:
    lr: 0.0001
    amsgrad: false
  lr_scheduler:
    step_lr: null
    reduce_lr_on_plateau:
      threshold: 1.0e-06
      threshold_mode: rel
      cooldown: 3
      patience: 5
      factor: 0.5
      min_lr: 1.0e-08
  early_stopping:
    min_delta: 1.0e-08
    patience: 10
    stop_training_on_plateau: true
  online_hard_keypoint_mining:
    online_mining: false
    hard_to_easy_ratio: 2.0
    min_hard_keypoints: 2
    max_hard_keypoints: null
    loss_scale: 5.0
  zmq:
    publish_address:
    controller_address:
    controller_polling_timeout: 10

name: 'your_experiment_name'
description: 'Brief description of your experiment'
sleap_nn_version: '0.0.1'
filename: ''
```

### Available Sample Configurations

Refer to the sample configuration files [here](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs).


- `config_single_instance_unet.yaml` - Basic single instance detection with UNet backbone
- `config_centroid_unet.yaml` - Centroid-based detection with UNet backbone
- `config_centroid_swint.yaml` - Centroid-based detection with Swin Transformer backbone

- `config_topdown_centered_instance_unet.yaml` - Top-down centered instance detection with UNet backbone
- `config_topdown_multi_class_centered_instance_unet.yaml` - Top-down multi-class centered instance detection with UNet backbone

- `config_bottomup_unet.yaml` - Bottom-up detection with UNet backbone
- `config_bottomup_convnext.yaml` - Bottom-up detection with ConvNeXt backbone
- `config_multi_class_bottomup_unet.yaml` - Multi-class bottom-up detection with UNet backbone


---

## Converting legacy SLEAP (≤1.4) `config.json` to SLEAP-NN YAML

If you have a SLEAP (v1.4 or earlier) `config.json` file from a previous project, you can easily convert it to a SLEAP-NN-compatible YAML configuration using the following code snippet:

```python
from sleap_nn.config.training_job_config import TrainingJobConfig
from omegaconf import OmegaConf

config = TrainingJobConfig.load_sleap_config("/path/to/config/json")
OmegaConf.save(config, "config.yaml")
```


---

## Data Configuration (`data_config`)

The data configuration section controls how training and validation data is loaded, preprocessed, and augmented.

### Core Data Settings
- `provider`: (str) Provider class to read the input sleap files. Only "LabelsReader" is currently supported for the training pipeline. **Default**: `"LabelsReader"`
- `train_labels_path`: (list) List of paths to training data (`.slp` file(s)). **Default**: `[]`
- `val_labels_path`: (list) List of paths to validation data (`.slp` file(s)). **Default**: `None`
- `validation_fraction`: (float) Float between 0 and 1 specifying the fraction of the training set to sample for generating the validation set. The remaining labeled frames will be left in the training set. If the `validation_labels` are already specified, this has no effect. **Default**: `0.1`
- `test_file_path`: (str) Path to test dataset (`.slp` file or `.mp4` file). **Note**: This is used only with CLI to get evaluation on test set once training is completed. **Default**: `None`
- `user_instances_only`: (bool) `True` if only user labeled instances should be used for training. If `False`, both user labeled and predicted instances would be used. **Default**: `True`

#### Example:

**Single training file:**
```yaml
data_config:
  train_labels_path:
    - path/to/your/single_training_data.slp
  val_labels_path:
    - path/to/your/validation_data.slp
```

**Multiple training files:**
```yaml
data_config:
  train_labels_path:
    - path/to/your/training_data_1.slp
    - path/to/your/training_data_2.slp
    - path/to/your/training_data_3.slp
  val_labels_path:
    - path/to/your/validation_data_1.slp
    - path/to/your/validation_data_2.slp
```

### Data Pipeline Framework
- `data_pipeline_fw`: (str) Framework to create the data loaders. One of [`torch_dataset`, `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`]. **Default**: `"torch_dataset"`. (Note: When using `torch_dataset`, `num_workers` in `trainer_config` should be set to 0 as multiprocessing doesn't work with pickling video backends.)
- `cache_img_path`: (str) Path to save `.jpg` images created with `torch_dataset_cache_img_disk` data pipeline framework. If `None`, the path provided in `trainer_config.save_ckpt` is used. The `train_imgs` and `val_imgs` dirs are created inside this path. **Default**: `None`
- `use_existing_imgs`: (bool) Use existing train and val images/ chunks in the `cache_img_path` for `torch_dataset_cache_img_disk` frameworks. If `True`, the `cache_img_path` should have `train_imgs` and `val_imgs` dirs. **Default**: `False`
- `delete_cache_imgs_after_training`: (bool) If `False`, the images (torch_dataset_cache_img_disk) are retained after training. Else, the files are deleted. **Default**: `True`

### Image Preprocessing
- `preprocessing`:
    - `ensure_rgb`: (bool) True if the input image should have 3 channels (RGB image). If input has only one channel when this is set to `True`, then the images from single-channel is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. **Default**: `False`
    - `ensure_grayscale`: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this is set to True, then we convert the image to grayscale (single-channel) image. If the source image has only one channel and this is set to False, then we retain the single channel input. **Default**: `False`
    - `max_height`: (int) Maximum height the image should be padded to. If not provided, the original image size will be retained. **Default**: `None`
    - `max_width`: (int) Maximum width the image should be padded to. If not provided, the original image size will be retained. **Default**: `None`
    - `scale`: (float) Factor to resize the image dimensions by, specified as a float. **Default**: `1.0`
    - `crop_hw`: (Tuple[int]) Crop height and width of each instance (h, w) for centered-instance model. If `None`, this would be automatically computed based on the largest instance in the `sio.Labels` file. **Default**: `None`
    - `min_crop_size`: (int) Minimum crop size to be used if `crop_hw` is `None`. **Default**: `100`

#### Example: Common Preprocessing Configurations

**Default img channels from source:**
```yaml
data_config:
  preprocessing:
    ensure_rgb: false
    ensure_grayscale: false
    scale: 1.0
    crop_hw: null
    min_crop_size: 100
```

**RGB images:**
```yaml
data_config:
  preprocessing:
    ensure_rgb: true
    ensure_grayscale: false
    scale: 1.0
    crop_hw: null
    min_crop_size: 100
```

**Fixed crop size for centered instance models:**
```yaml
data_config:
  preprocessing:
    ensure_rgb: false
    ensure_grayscale: false
    scale: 1.0
    crop_hw: [256, 256]  # Fixed 256x256 crop
    min_crop_size: 100
```

**Resize/ Pad images with maximum dimensions:**
```yaml
data_config:
  preprocessing:
    ensure_rgb: false
    ensure_grayscale: false
    scale: 1.0
    max_height: 512
    max_width: 512
    crop_hw: null
    min_crop_size: 100
```

### Data Augmentation
- `use_augmentations_train`: (bool) True if the data augmentation should be applied to the training data, else False. **Default**: `False`
- `augmentation_config`: (only if `use_augmentations` is `True`)
  - `intensity`: (Optional)
    - `uniform_noise_min`: (float) Minimum value for uniform noise (uniform_noise_min >=0). **Default**: `0.0`
    - `uniform_noise_max`: (float) Maximum value for uniform noise (uniform_noise_max <>=1). **Default**: `1.0`
    - `uniform_noise_p`: (float) Probability of applying random uniform noise. **Default**: `0.0`
    - `gaussian_noise_mean`: (float) The mean of the gaussian noise distribution. **Default**: `0.0`
    - `gaussian_noise_std`: (float) The standard deviation of the gaussian noise distribution. **Default**: `1.0`
    - `gaussian_noise_p`: (float) Probability of applying random gaussian noise. **Default**: `0.0`
    - `contrast_min`: (float) Minimum contrast factor to apply. **Default**: `0.9`
    - `contrast_max`: (float) Maximum contrast factor to apply. **Default**: `1.1`
    - `contrast_p`: (float) Probability of applying random contrast. **Default**: `0.0`
    - `brightness_min`: (float) Minimum brightness factor to apply. **Default**: `1.0`
    - `brightness_max`: (float) Maximum brightness factor to apply. **Default**: `1.0`
    - `brightness_p`: (float) Probability of applying random brightness. **Default**: `0.0`
  - `geometric`: (Optional)
    - `rotation_min`: (float) Minimum rotation angle in degrees. A random angle in (rotation_min, rotation_max) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation. **Default**: `-15.0`.
    - `rotation_max`: (float) Maximum rotation angle in degrees. A random angle in (rotation_min, rotation_max) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation. **Default**: `15.0`.
    - `scale_min`: (float) Minimum scaling factor. If scale_min and scale_max are provided, the scale is randomly sampled from the range scale_min <= scale <= scale_max for isotropic scaling. **Default**: `0.9`.
    - `scale_max`: (float) Maximum scaling factor. If scale_min and scale_max are provided, the scale is randomly sampled from the range scale_min <= scale <= scale_max for isotropic scaling. **Default**: `1.1`.
    - `translate_width`: (float) Maximum absolute fraction for horizontal translation. For example, if translate_width=a, then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a. Will not translate by default. **Default**: `0.0`
    - `translate_height`: (float) Maximum absolute fraction for vertical translation. For example, if translate_height=a, then vertical shift is randomly sampled in the range -img_height * a < dy < img_height * a. Will not translate by default. **Default**: `0.0`
    - `affine_p`: (float) Probability of applying random affine transformations. **Default**: `0.0`
    - `erase_scale_min`: (float) Minimum value of range of proportion of erased area against input image. **Default**: `0.0001`
    - `erase_scale_max`: (float) Maximum value of range of proportion of erased area against input image. **Default**: `0.01`
    - `erase_ratio_min`: (float) Minimum value of range of aspect ratio of erased area. **Default**: `1.0`
    - `erase_ratio_max`: (float) Maximum value of range of aspect ratio of erased area. **Default**: `1.0`
    - `erase_p`: (float) Probability of applying random erase. **Default**: `0.0`
    - `mixup_lambda_min`: (float) Minimum mixup strength value. **Default**: `0.01`
    - `mixup_lambda_max`: (float) Maximum mixup strength value. **Default**: `0.05`
    - `mixup_p`: (float) Probability of applying random mixup v2. **Default**: `0.0`

#### Example: Common Augmentation Configurations

**No augmentation:**
```yaml
data_config:
  use_augmentations_train: false
  # augmentation_config is not needed when use_augmentations_train is false
```

**Only intensity augmentations:**
```yaml
data_config:
  use_augmentations_train: true
  augmentation_config:
    intensity:
      contrast_p: 0.3
      brightness_p: 0.3
      gaussian_noise_p: 0.1
      gaussian_noise_std: 0.1
    geometric: null  # No geometric augmentations
```

**Only geometric augmentations:**
```yaml
data_config:
  use_augmentations_train: true
  augmentation_config:
    intensity: null  # No intensity augmentations
    geometric:
      rotation_min: -15.0
      roration_max: 15.0
      scale_min: 0.9
      scale_max: 1.1
      translate_width: 0.1
      translate_height: 0.1
      affine_p: 0.5
```

**Both intensity and geometric augmentations:**
```yaml
data_config:
  use_augmentations_train: true
  augmentation_config:
    intensity:
      contrast_p: 0.3
      brightness_p: 0.3
      gaussian_noise_p: 0.1
      gaussian_noise_std: 0.1
    geometric:
      rotation_min: -15.0
      rotation_max: 15.0
      scale_min: 0.9
      scale_max: 1.1
      translate_width: 0.1
      translate_height: 0.1
      affine_p: 0.5
      erase_p: 0.1
```

---

## Model Configuration (`model_config`)

The model configuration section defines the neural network architecture, including backbone and head configurations.

### Model Initialization
- `init_weights`: (str) Model weights initialization method. "default" uses kaiming uniform initialization and "xavier" uses Xavier initialization method. **Default**: `"default"`
- `pretrained_backbone_weights`: (str) Path of the `ckpt` (or `.h5` file from SLEAP) file with which the backbone is initialized. If `None`, random init is used. **Default**: `None`
- `pretrained_head_weights`: (str) Path of the `ckpt` (or `.h5` file from SLEAP) file with which the head layers are initialized. If `None`, random init is used. **Default**: `None`

### Backbone Configuration
**Note**: Configs should be provided only for the model to train and others should be `None`.

#### UNet Backbone
- `backbone_config.unet`:
    - `in_channels`: (int) Number of input channels. **Default**: `1`
    - `kernel_size`: (int) Size of the convolutional kernels. **Default**: `3`
    - `filters`: (int) Base number of filters in the network. **Default**: `32`
    - `filters_rate`: (float) Factor to adjust the number of filters per block. **Default**: `1.5`
    - `max_stride`: (int) Scalar integer specifying the maximum stride which is used to compute the number of down blocks. **Default**: `16`
    - `stem_stride`: (int) If not None, will create additional "down" blocks for initial downsampling based on the stride. These will be configured identically to the down blocks below. **Default**: `None`
    - `middle_block`: (bool) If True, add an additional block at the end of the encoder. **Default**: `True`
    - `up_interpolate`: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. **Default**: `True`
    - `stacks`: (int) Number of upsampling blocks in the decoder. **Default**: `1`
    - `convs_per_block`: (int) Number of convolutional layers per block. **Default**: `2`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. Ideally, this should be minimum of the output strides of all head layers. **Default**: `1`

**Example UNet configuration:**
```yaml
model_config:
  backbone_config:
    unet:
      in_channels: 1
      kernel_size: 3
      filters: 32
      filters_rate: 1.5
      max_stride: 16
      stem_stride: null
      middle_block: true
      up_interpolate: true
      stacks: 1
      convs_per_block: 2
      output_stride: 1
    convnext: null
    swint: null
```

#### ConvNeXt Backbone
- `backbone_config.convnext`:
  - `pre_trained_weights`: (str) Pretrained weights file name supported only for ConvNext backbones. For ConvNext, one of ["ConvNeXt_Base_Weights","ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]. **Default**: `None`
  - `arch`: (Default is `Tiny` architecture config. No need to provide if `model_type` is provided)
    - `depths`: (List[int]) Number of layers in each block. **Default**: `[3, 3, 9, 3]`
    - `channels`: (List[int]) Number of channels in each block. **Default**: `[96, 192, 384, 768]`
  - `model_type`: (str) One of the ConvNext architecture types: ["tiny", "small", "base", "large"]. **Default**: `"tiny"`
  - `max_stride`: (int) Factor by which input image size is reduced through the layers. This is always `32` for all convnext architectures provided stem_stride is 2. **Default**: `32`
  - `stem_patch_kernel`: (int) Size of the convolutional kernels in the stem layer. **Default**: `4`
  - `stem_patch_stride`: (int) Convolutional stride in the stem layer. **Default**: `2`
  - `in_channels`: (int) Number of input channels. **Default**: `1`
  - `kernel_size`: (int) Size of the convolutional kernels. **Default**: `3`
  - `filters_rate`: (float) Factor to adjust the number of filters per block. **Default**: `2`
  - `convs_per_block`: (int) Number of convolutional layers per block. **Default**: `2`
  - `up_interpolate`: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. **Default**: `True`
  - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. Ideally, this should be minimum of the output strides of all head layers. **Default**: `1`

**Example ConvNeXt configuration:**
```yaml
model_config:
  backbone_config:
    unet: null
    convnext:
      pre_trained_weights: "ConvNeXt_Tiny_Weights"
      model_type: "tiny"
      max_stride: 32
      stem_patch_kernel: 4
      stem_patch_stride: 2
      in_channels: 1
      kernel_size: 3
      filters_rate: 2
      convs_per_block: 2
      up_interpolate: true
      output_stride: 1
    swint: null
```

#### Swin Transformer Backbone
- `backbone_config.swint`:
  - `pre_trained_weights`: (str) Pretrained weights file name supported only for SwinT backbones. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"]. **Default**: `None`
  - `model_type`: (str) One of the SwinT architecture types: ["tiny", "small", "base"]. **Default**: `"tiny"`
  - `arch`: Dictionary of embed dimension, depths and number of heads in each layer. Default is "Tiny architecture". {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}. **Default**: `None`
  - `max_stride`: (int) Factor by which input image size is reduced through the layers. This is always `32` for all swint architectures provided stem_stride is 2. **Default**: `32`
  - `patch_size`: (int) Patch size for the stem layer of SwinT. **Default**: `4`
  - `stem_patch_stride`: (int) Stride for the patch. **Default**: `2`
  - `window_size`: (int) Window size. **Default**: `7`
  - `in_channels`: (int) Number of input channels. **Default**: `1`
  - `kernel_size`: (int) Size of the convolutional kernels. **Default**: `3`
  - `filters_rate`: (float) Factor to adjust the number of filters per block. **Default**: `2`
  - `convs_per_block`: (int) Number of convolutional layers per block. **Default**: `2`
  - `up_interpolate`: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. **Default**: `True`
  - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. Ideally, this should be minimum of the output strides of all head layers. **Default**: `1`

**Example SwinT configuration:**
```yaml
model_config:
  backbone_config:
    unet: null
    convnext: null
    swint:
      pre_trained_weights: "Swin_T_Weights"
      model_type: "tiny"
      max_stride: 32
      patch_size: 4
      stem_patch_stride: 2
      window_size: 7
      in_channels: 1
      kernel_size: 3
      filters_rate: 2
      convs_per_block: 2
      up_interpolate: true
      output_stride: 1
```

### Head Configuration
**Note**: Configs should be provided only for the model to train and others should be `None`.

#### Single Instance Head
- `head_configs.single_instance.confmaps`:
    - `part_names`: (List[str]) `None` if nodes from `sio.Labels` file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`

**Example Single Instance configuration:**
```yaml
model_config:
  head_configs:
    single_instance:
      confmaps:
        part_names: null  # Uses all body parts from skeleton
        sigma: 5.0
        output_stride: 1
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
```

#### Centroid Head
- `head_configs.centroid.confmaps`:
    - `anchor_part`: (str) Node name (as per skeleton) to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image. **Default**: `None`
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`

**Example Centroid configuration:**
```yaml
model_config:
  head_configs:
    single_instance: null
    centroid:
      confmaps:
        anchor_part: null  # Uses bounding box midpoint
        sigma: 5.0
        output_stride: 1
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
```

#### Centered Instance Head
- `head_configs.centered_instance.confmaps`:
    - `part_names`: (List[str]) `None` if nodes from `sio.Labels` file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'. **Default**: `None`
    - `anchor_part`: (str) Node name (as per skeleton) to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image. **Default**: `None`
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`

**Example Centered Instance configuration:**
```yaml
model_config:
  head_configs:
    single_instance: null
    centroid: null
    centered_instance:
      confmaps:
        part_names: null  # Uses all body parts from skeleton
        anchor_part: null  # Uses bounding box midpoint
        sigma: 5.0
        output_stride: 1
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
```

#### Bottom-Up Head
- `head_configs.bottomup.confmaps`:
    - `part_names`: (List[str]) `None` if nodes from `sio.Labels` file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'. **Default**: `None`
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`
    - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models. **Default**: `None`

- `head_configs.bottomup.pafs`: (same structure as that of `confmaps`. **Note**: This section is only for BottomUp model.)
    - `edges`: (List[str]) `None` if edges from `sio.Labels` file can be used directly. **Note**: Only for 'PartAffinityFieldsHead'. List of indices `(src, dest)` that form an edge. **Default**: `None`
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `15.0`  
      *Note: For PAFs, a higher sigma is usually used compared to confidence maps, as the vector fields benefit from a broader spatial spread.*
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`
    - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models. **Default**: `None`

**Example Bottom-Up configuration:**
```yaml
model_config:
  head_configs:
    single_instance: null
    centroid: null
    centered_instance: null
    bottomup:
      confmaps:
        part_names: null  # Uses all body parts from skeleton
        sigma: 5.0
        output_stride: 1
        loss_weight: 1.0
      pafs:
        edges: null  # Uses all edges from skeleton
        sigma: 15.0
        output_stride: 1
        loss_weight: 1.0
    multi_class_bottomup: null
    multi_class_topdown: null
```

#### Multi-Class Bottom-Up Head
- `head_configs.multi_class_bottomup.confmaps`:
    - `part_names`: (List[str]) `None` if nodes from `sio.Labels` file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'. **Default**: `None`
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`
    - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models. **Default**: `None`

- `head_configs.multi_class_bottomup.class_maps`:
    - `classes`: (List[str]) List of class (track) names. **Default**: `None`. When `None`, these are inferred from the track names in the labels file.
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`
    - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models. **Default**: `None`

**Example Multi-Class Bottom-Up configuration:**
```yaml
model_config:
  head_configs:
    single_instance: null
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup:
      confmaps:
        part_names: null  # Uses all body parts from skeleton
        sigma: 5.0
        output_stride: 1
        loss_weight: 1.0
      class_maps:
        classes: null  # Inferred from track names in labels file
        sigma: 5.0
        output_stride: 1
        loss_weight: 1.0
    multi_class_topdown: null
```

#### Multi-Class Top-Down Head
- `head_configs.multi_class_topdown.confmaps`:
    - `part_names`: (List[str]) `None` if nodes from `sio.Labels` file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'. **Default**: `None`
    - `anchor_part`: (str) Node name to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image. **Default**: `None`
    - `sigma`: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied. **Default**: `5.0`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. **Default**: `1`. (Ideally this should be same as the backbone's maxstride).
    - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models. **Default**: `None`

- `head_configs.multi_class_topdown.class_vectors`:
    - `classes`: (List[str]) List of class (track) names. **Default**: `None`. When `None`, these are inferred from the track names in the labels file.
    - `num_fc_layers`: (int) Number of fully connected layers after flattening input features. **Default**: `1`
    - `num_fc_units`: (int) Number of units (dimensions) in fully connected layers prior to classification output. **Default**: `64`
    - `global_pool`: (bool) Enable global pooling. **Default**: `True`
    - `output_stride`: (int) The stride of the output confidence maps relative to the input image. These should be the same as max stride set in the backbone config as we take the feature maps from the last layer of the encoder. **Default**: `16`
    - `loss_weight`: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models. **Default**: `None`

**Example Multi-Class Top-Down configuration:**
```yaml
model_config:
  head_configs:
    single_instance: null
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown:
      confmaps:
        part_names: null  # Uses all body parts from skeleton
        anchor_part: null  # Uses bounding box midpoint
        sigma: 5.0
        output_stride: 1
        loss_weight: 1.0
      class_vectors:
        classes: null  # Inferred from track names in labels file
        num_fc_layers: 1
        num_fc_units: 64
        global_pool: true
        output_stride: 16 # should be same as `max_stride` in `backbone_config`
        loss_weight: 1.0
```

---

## Trainer Configuration (`trainer_config`)

The trainer configuration section controls the training process, including data loading, optimization, and monitoring.

### Data Loader Settings
- `train_data_loader`:
    - `batch_size`: (int) Number of samples per batch or batch size for training data. **Default**: `1`
    - `shuffle`: (bool) True to have the data reshuffled at every epoch. **Default**: `False`
    - `num_workers`: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. **Default**: `0`
- `val_data_loader`: (Similar to `train_data_loader`)

**Example Data Loader configurations:**

**Basic configuration (single GPU):**
```yaml
trainer_config:
  train_data_loader:
    batch_size: 4
    shuffle: true
    num_workers: 0
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0
```

### Model Checkpointing
- `model_ckpt`:
    - `save_top_k`: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False. **Default**: `1`
    - `save_last`: (bool) When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. **Default**: `None`

### Hardware Configuration
- `trainer_devices`: (int) Number of devices to train on (int), which devices to train on (list or str), or "auto" to select automatically. **Default**: `"auto"`
- `trainer_accelerator`: (str) One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises the machine the model is running on and chooses the appropriate accelerator for the `Trainer` to be connected to. **Default**: `"auto"`
- `profiler`: (str) Profiler for pytorch Trainer. One of ["advanced", "passthrough", "pytorch", "simple"]. **Default**: `None`
- `trainer_strategy`: (str) Training strategy, one of ["auto", "ddp", "fsdp", "ddp_find_unused_parameters_false", "ddp_find_unused_parameters_true", ...]. This supports any training strategy that is supported by `lightning.Trainer`. **Default**: `"auto"`

### Training Control
- `enable_progress_bar`: (bool) When True, enables printing the logs during training. **Default**: `True`
- `min_train_steps_per_epoch`: (int) Minimum number of iterations in a single epoch. (Useful if model is trained with very few data points). Refer `limit_train_batches` parameter of Torch `Trainer`. **Default**: `200`
- `train_steps_per_epoch`: (int) Number of minibatches (steps) to train for in an epoch. If set to `None`, this is set to the number of batches in the training data or `min_train_steps_per_epoch`, whichever is largest. **Default**: `None`. **Note**: In a multi-gpu training setup, the effective steps during training would be the `trainer_steps_per_epoch` / `trainer_devices`. 
- `visualize_preds_during_training`: (bool) If set to `True`, sample predictions (keypoints + confidence maps) are saved to `viz` folder in the ckpt dir and in wandb table. **Default**: `False`
- `keep_viz`: (bool) If set to `True`, the `viz` folder containing training visualizations will be kept after training completes. If `False`, the folder will be deleted. This parameter only has an effect when `visualize_preds_during_training` is `True`. **Default**: `False`
- `max_epochs`: (int) Maximum number of epochs to run. **Default**: `10`
- `seed`: (int) Seed value for the current experiment. **Default**: `0`
- `use_wandb`: (bool) True to enable wandb logging. **Default**: `False`
- `save_ckpt`: (bool) True to enable checkpointing. **Default**: `False`
- `save_ckpt_path`: (str) Directory path to save the training config and checkpoint files. **Default**: `None`
- `resume_ckpt_path`: (str) Path to `.ckpt` file from which training is resumed. **Default**: `None`

### Optimizer Configuration
- `optimizer_name`: (str) Optimizer to be used. One of ["Adam", "AdamW"]. **Default**: `"Adam"`
- `optimizer`:
    - `lr`: (float) Learning rate of type float. **Default**: `1e-3`
    - `amsgrad`: (bool) Enable AMSGrad with the optimizer. **Default**: `False`

### Learning Rate Schedulers
**Note**: Configs should only be provided for one scheduler. Others should be `None`.

#### Step LR Scheduler
- `lr_scheduler.step_lr`:
    - `step_size`: (int) Period of learning rate decay. If `step_size`=10, then every 10 epochs, learning rate will be reduced by a factor of `gamma`. **Default**: `10`
    - `gamma`: (float) Multiplicative factor of learning rate decay. **Default**: `0.1`

#### Reduce LR on Plateau
- `lr_scheduler.reduce_lr_on_plateau`:
    - `threshold`: (float) Threshold for measuring the new optimum, to only focus on significant changes. **Default**: `1e-4`
    - `threshold_mode`: (str) One of "rel", "abs". In rel mode, dynamic_threshold = best * ( 1 + threshold ) in max mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. **Default**: `"rel"`
    - `cooldown`: (int) Number of epochs to wait before resuming normal operation after lr has been reduced. **Default**: `0`
    - `patience`: (int) Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the third epoch if the loss still hasn't improved then. **Default**: `10`
    - `factor`: (float) Factor by which the learning rate will be reduced. new_lr = lr * factor. **Default**: `0.1`
    - `min_lr`: (float or List[float]) A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. **Default**: `0.0`

**Example Learning Rate Scheduler configurations:**

**No scheduler (constant learning rate):**
```yaml
trainer_config:
  lr_scheduler:
    step_lr: null
    reduce_lr_on_plateau: null
```

**Step LR Scheduler:**
```yaml
trainer_config:
  lr_scheduler:
    step_lr:
      step_size: 20
      gamma: 0.5
    reduce_lr_on_plateau: null
```

**Reduce LR on Plateau:**
```yaml
trainer_config:
  lr_scheduler:
    step_lr: null
    reduce_lr_on_plateau:
      threshold: 1e-6
      threshold_mode: "rel"
      cooldown: 3
      patience: 5
      factor: 0.5
      min_lr: 1e-8
```

### Early Stopping
- `early_stopping`:
    - `stop_training_on_plateau`: (bool) True if early stopping should be enabled. **Default**: `False`
    - `min_delta`: (float) Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement. **Default**: `0.0`
    - `patience`: (int) Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch. **Default**: `1`

### Online Hard Keypoint Mining (OHKM)
- `online_hard_keypoint_mining`:
    - `online_mining`: (bool) If True, online hard keypoint mining (OHKM) will be enabled. When this is enabled, the loss is computed per keypoint (or edge for PAFs) and sorted from lowest (easy) to highest (hard). The hard keypoint loss will be scaled to have a higher weight in the total loss, encouraging the training to focus on tricky body parts that are more difficult to learn. If False, no mining will be performed and all keypoints will be weighted equally in the loss. **Default**: `False`
    - `hard_to_easy_ratio`: (float) The minimum ratio of the individual keypoint loss with respect to the lowest keypoint loss in order to be considered as "hard". This helps to switch focus on across groups of keypoints during training. **Default**: `2.0`
    - `min_hard_keypoints`: (int) The minimum number of keypoints that will be considered as "hard", even if they are not below the `hard_to_easy_ratio`. **Default**: `2`
    - `max_hard_keypoints`: (int) The maximum number of hard keypoints to apply scaling to. This can help when there are few very easy keypoints which may skew the ratio and result in loss scaling being applied to most keypoints, which can reduce the impact of hard mining altogether. **Default**: `None`
    - `loss_scale`: (float) Factor to scale the hard keypoint losses by. **Default**: `5.0`

### WandB Configuration
**Note**: Only required if `use_wandb` is `True`.

- `wandb`:
    - `entity`: (str) Entity of wandb project. **Default**: `None`
    - `project`: (str) Project name for the wandb project. **Default**: `None`
    - `name`: (str) Name of the current run. **Default**: `None`
    - `api_key`: (str) API key. The API key is masked when saved to config files. **Default**: `None`
    - `wandb_mode`: (str) "offline" if only local logging is required. **Default**: `"None"`
    - `prv_runid`: (str) Previous run ID if training should be resumed from a previous ckpt. **Default**: `None`
    - `group`: (str) Group name for the run.

**Example WandB configurations:**

**Basic WandB logging:**
```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: "your_username"
    project: "sleap_nn_experiments"
    name: "single_instance_unet_training"
    api_key: null  # Set via environment variable WANDB_API_KEY
    wandb_mode: "online"
    prv_runid: null
    group: null
```

**Offline WandB logging:**
```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: "your_username"
    project: "sleap_nn_experiments"
    name: "offline_experiment"
    api_key: null
    wandb_mode: "offline"
    prv_runid: null
    group: "experiment_group"
```

**Resume from previous run:**
```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: "your_username"
    project: "sleap_nn_experiments"
    name: "continued_training"
    api_key: null
    wandb_mode: "online"
    prv_runid: "abc123def456"
    group: "continued_experiments"
```

**No WandB logging:**
```yaml
trainer_config:
  use_wandb: false
  # wandb section not needed when use_wandb is false
```

### ZMQ Configuration
- `zmq`:
    - `publish_address`: (str) Specifies the address and port to which the training logs (loss values) should be sent to. **Default**: `None`
    - `controller_address`: (str) Specifies the address and port to listen to to stop the training (specific to SLEAP GUI). **Default**: `None`
    - `controller_polling_timeout`: (int) Polling timeout in microseconds specified as an integer. This controls how long the poller should wait to receive a response and should be set to a small value to minimize the impact on training speed. **Default**: `10`


---

## Next Steps

- [Model Architectures](models.md)
- [Training Guide](training.md)
- [API Reference](api/index.md)