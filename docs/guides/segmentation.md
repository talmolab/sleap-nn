# Bottom-Up Segmentation

Train and run the `bottomup_segmentation` model type when each instance is
represented by a segmentation mask instead of a keypoint skeleton.

This pipeline predicts three outputs from a single model:

1. A foreground segmentation map
2. An instance-center heatmap
3. Per-pixel offsets back to each instance center

At inference time, `sleap-nn track` auto-detects a segmentation checkpoint from
the model directory. You do not need a special mode flag.

## When to use it

Use bottom-up segmentation when your labels contain instance masks and you want
per-animal mask predictions instead of pose keypoints.

## Training

### Example config

Start from a normal single-model training config and set
`head_configs.bottomup_segmentation`.

```yaml
data_config:
  train_labels_path:
    - train.pkg.slp
  val_labels_path:
    - val.pkg.slp
  preprocessing:
    scale: 1.0

model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 16
      output_stride: 2
  head_configs:
    single_instance: null
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
    bottomup_segmentation:
      segmentation:
        output_stride: 2
        loss_weight: 1.0
      center:
        sigma: 5.0
        output_stride: 2
        loss_weight: 1.0
      offsets:
        output_stride: 2
        loss_weight: 0.1

trainer_config:
  save_ckpt: true
  ckpt_dir: models
  run_name: segmentation
```

Train it with:

```bash
sleap-nn train --config segmentation.yaml
```

### Training caveats

- Use labels that include segmentation masks (`Labels.masks` in `sleap-io`).
- For the current v1 dataset path, keep `scale: 1.0`.
- Keep image dimensions divisible by the backbone `max_stride`.
- Geometric augmentation is skipped for segmentation models because masks cannot
  be spatially transformed in sync with the image. Intensity augmentation still
  works.

## Inference

Run inference the same way as other single-model pipelines:

```bash
sleap-nn track \
    -i video.mp4 \
    -m models/segmentation/ \
    -o preds.slp
```

### Segmentation-specific flags

Two inference flags only affect `bottomup_segmentation` models:

| Flag | Default | Purpose |
|------|---------|---------|
| `--fg_threshold` | `0.5` | Binarize the predicted foreground map before grouping masks. |
| `--min_mask_area` | `0` | Drop tiny predicted masks in original-image pixels. Useful for suppressing over-segmentation. |

Example:

```bash
sleap-nn track \
    -i video.mp4 \
    -m models/segmentation/ \
    -o preds.slp \
    --fg_threshold 0.5 \
    --min_mask_area 3000
```

`--min_mask_area` is dataset-dependent. A value around `3000` can be a useful
starting point when you need to suppress small spurious blobs, but it should be
tuned on your own validation data.

## Viewing mask predictions

The output `.slp` contains predicted segmentation masks, so you can render them
directly:

```bash
sio render preds.slp
```

You can also load the output in Python:

```python
import sleap_io as sio

labels = sio.load_slp("preds.slp")
print(len(labels.masks))
```
