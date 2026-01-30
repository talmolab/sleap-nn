# Sample Configs

Ready-to-use configuration templates for common scenarios.

---

## By Model Type

### Single Instance

One animal per frame.

| Config | Backbone | Receptive Field |
|--------|----------|-----------------|
| [single_instance_unet_medium](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_single_instance_unet_medium_rf.yaml) | UNet | Medium |
| [single_instance_unet_large](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_single_instance_unet_large_rf.yaml) | UNet | Large |

### Top-Down

Two-stage: centroid detection + pose estimation.

| Config | Backbone | Notes |
|--------|----------|-------|
| [centroid_unet](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_centroid_unet.yaml) | UNet | Stage 1 |
| [centroid_swint](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_centroid_swint.yaml) | Swin-T | Stage 1 |
| [centered_instance_unet_medium](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_centered_instance_unet_medium_rf.yaml) | UNet | Stage 2, Medium |
| [centered_instance_unet_large](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_centered_instance_unet_large_rf.yaml) | UNet | Stage 2, Large |

### Bottom-Up

Single-stage multi-instance.

| Config | Backbone | Receptive Field |
|--------|----------|-----------------|
| [bottomup_unet_medium](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bottomup_unet_medium_rf.yaml) | UNet | Medium |
| [bottomup_unet_large](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bottomup_unet_large_rf.yaml) | UNet | Large |
| [bottomup_convnext](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bottomup_convnext.yaml) | ConvNeXt | - |

### Multi-Class (Identity)

With supervised identity tracking.

| Config | Model Type | Backbone |
|--------|-----------|----------|
| [multi_class_bottomup](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_multi_class_bottomup_unet.yaml) | Bottom-Up | UNet |
| [multi_class_topdown](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_multi_class_centered_instance_unet.yaml) | Top-Down | UNet |

---

## Quick Start Templates

### Minimal Single Instance

```yaml
data_config:
  train_labels_path:
    - train.slp

model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 16
  head_configs:
    single_instance:
      confmaps:
        sigma: 5.0

trainer_config:
  max_epochs: 100
  save_ckpt: true
  ckpt_dir: models
  run_name: single_instance
```

### Minimal Bottom-Up

```yaml
data_config:
  train_labels_path:
    - train.slp

model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 32
      output_stride: 4
  head_configs:
    bottomup:
      confmaps:
        sigma: 2.5
        output_stride: 4
      pafs:
        sigma: 75.0
        output_stride: 8

trainer_config:
  max_epochs: 200
  save_ckpt: true
  ckpt_dir: models
  run_name: bottomup
```

### Minimal Top-Down (Centroid)

```yaml
data_config:
  train_labels_path:
    - train.slp

model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 16
  head_configs:
    centroid:
      confmaps:
        sigma: 5.0

trainer_config:
  max_epochs: 100
  save_ckpt: true
  ckpt_dir: models
  run_name: centroid
```

### Minimal Top-Down (Instance)

```yaml
data_config:
  train_labels_path:
    - train.slp
  preprocessing:
    crop_size: 256

model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 16
  head_configs:
    centered_instance:
      confmaps:
        sigma: 5.0

trainer_config:
  max_epochs: 100
  save_ckpt: true
  ckpt_dir: models
  run_name: centered_instance
```

---

## Download All Samples

```bash
# Clone the repo
git clone https://github.com/talmolab/sleap-nn.git

# Configs are in docs/sample_configs/
ls sleap-nn/docs/sample_configs/
```

---

## Customize a Sample

1. Download a sample config
2. Update `train_labels_path` and `val_labels_path`
3. Adjust `ckpt_dir` and `run_name`
4. Run training:

```bash
sleap-nn train --config my_config.yaml
```

---

## Tips

!!! tip "Start with medium receptive field"
    Medium RF configs are a good balance of speed and accuracy.

!!! tip "Use augmentation"
    All sample configs have augmentation enabled. Disable with `use_augmentations_train: false` for debugging.

!!! tip "Check your data first"
    Open your `.slp` file in SLEAP to verify labels before training.
