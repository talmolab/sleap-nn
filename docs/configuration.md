# Configuration Reference

sleap-nn uses a hierarchical configuration system based on Hydra and OmegaConf. This guide covers all configuration options.

## Configuration Structure

Configurations are divided into three main sections:

```yaml
data_config: {...}      # Data pipeline settings
model_config: {...}     # Model architecture settings
trainer_config: {...}   # Training process settings
```

## Data Configuration

### Basic Options

```yaml
data_config:
  data_path: "/path/to/labels.slp"  # Path to SLEAP labels file
  video_backend: "pyav"             # Video backend: "pyav" or "opencv"
  batch_size: 16                    # Batch size for training
  num_workers: 4                    # Number of data loading workers
  
  train_split: 0.8                  # Training data fraction
  val_split: 0.1                    # Validation data fraction
  test_split: 0.1                   # Test data fraction
```

### Preprocessing

```yaml
data_config:
  preprocessing:
    input_scale: 1.0              # Input image scaling
    pad_to_stride: 32             # Pad images to multiple of stride
    
  normalization:
    type: "imagenet"              # Normalization: "imagenet", "01", "custom"
    mean: [0.485, 0.456, 0.406]  # Custom mean values
    std: [0.229, 0.224, 0.225]   # Custom std values
```

### Augmentation

```yaml
data_config:
  augmentation:
    # Geometric augmentations
    rotate: true
    rotation_range: 15.0          # Degrees
    
    scale: true
    scale_range: [0.8, 1.2]       # Min/max scale factors
    
    translate: true
    translate_range: [-0.1, 0.1]  # Fraction of image size
    
    # Photometric augmentations
    brightness: 0.2               # Brightness jitter
    contrast: 0.2                 # Contrast jitter
    saturation: 0.2               # Saturation jitter
    hue: 0.1                      # Hue jitter
```

### Instance Processing

```yaml
data_config:
  instance_cropping:
    enabled: true
    crop_size: 128                # Crop size in pixels
    padding: 16                   # Padding around instances
```

## Model Configuration

### Architecture Selection

```yaml
model_config:
  architecture: "unet"            # Options: "unet", "convnext", "swint"
  
  # Model-specific settings
  model_type: "single_instance"   # Type of model
```

### Backbone Configuration

#### UNet
```yaml
model_config:
  backbone_config:
    filters: 32                   # Base number of filters
    depth: 5                      # Number of down/up blocks
    kernel_size: 3                # Convolution kernel size
    activation: "relu"            # Activation function
    batch_norm: true              # Use batch normalization
```

#### ConvNext
```yaml
model_config:
  backbone_config:
    model_variant: "tiny"         # tiny, small, base, large
    pretrained: true              # Use ImageNet weights
    in_channels: 3                # Input channels
```

#### Swin Transformer
```yaml
model_config:
  backbone_config:
    model_variant: "tiny"         # tiny, small, base
    pretrained: true              # Use ImageNet weights
    window_size: 7                # Attention window size
```

### Head Configuration

```yaml
model_config:
  head_config:
    # Common settings
    output_stride: 2              # Output feature stride
    channels: 256                 # Feature channels
    
    # Task-specific heads
    confidence_maps:
      enabled: true
      num_classes: 5              # Number of keypoints
      sigma: 3.0                  # Gaussian sigma for CMs
      loss_weight: 1.0            # Loss weight
    
    part_affinity_fields:
      enabled: true
      num_edges: 4                # Number of connections
      line_width: 5               # PAF line width
      loss_weight: 1.0            # Loss weight
```

## Trainer Configuration

### Basic Training

```yaml
trainer_config:
  max_epochs: 100                 # Maximum training epochs
  learning_rate: 1e-3             # Initial learning rate
  
  # Hardware settings
  accelerator: "gpu"              # "gpu", "cpu", "tpu"
  devices: 1                      # Number of devices
  precision: 32                   # 16, 32, or "bf16"
```

### Optimization

```yaml
trainer_config:
  optimizer:
    type: "adam"                  # adam, sgd, adamw
    weight_decay: 1e-4            # L2 regularization
    
  scheduler:
    type: "cosine"                # cosine, step, exponential
    warmup_epochs: 5              # Warmup period
    min_lr: 1e-6                  # Minimum learning rate
```

### Callbacks

```yaml
trainer_config:
  callbacks:
    early_stopping:
      enabled: true
      monitor: "val_loss"         # Metric to monitor
      patience: 10                # Epochs without improvement
      mode: "min"                 # min or max
    
    model_checkpoint:
      monitor: "val_loss"
      save_top_k: 3               # Number of best models
      save_last: true             # Save last checkpoint
```

### Distributed Training

```yaml
trainer_config:
  strategy: "ddp"                 # ddp, ddp_sharded, fsdp
  sync_batchnorm: true            # Sync BN across GPUs
  
  # DDP settings
  ddp:
    find_unused_parameters: false
    gradient_as_bucket_view: true
```

## Command Line Overrides

Override any configuration from command line:

```bash
# Change batch size
python -m sleap_nn.train data_config.batch_size=32

# Use different GPU
python -m sleap_nn.train trainer_config.devices=1

# Change learning rate
python -m sleap_nn.train trainer_config.learning_rate=5e-4
```

## Configuration Files

Save complete configurations as YAML files:

```yaml
# config/my_config.yaml
data_config:
  data_path: /data/my_dataset.slp
  batch_size: 16
  
model_config:
  architecture: unet
  backbone_config:
    filters: 64
    
trainer_config:
  max_epochs: 200
  learning_rate: 1e-3
```

Use custom config:
```bash
python -m sleap_nn.train --config-path config --config-name my_config
```

## Next Steps

- [Model Architectures](architectures.md)
- [Training Guide](training.md)
- [API Reference](api/index.md)