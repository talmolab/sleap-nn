# Your First Model

A complete walkthrough from labeled data to predictions.

---

## Overview

In this tutorial, you'll:

1. Choose the right model type for your data
2. Create a configuration file
3. Train the model
4. Run inference on new videos
5. Evaluate your results

**Time**: ~20 minutes (plus training time)

---

## Step 1: Choose Your Model Type

The right model depends on your data:

| Scenario | Model Type | Config |
|----------|-----------|--------|
| **One animal per frame** | Single Instance | `single_instance` |
| **Multiple animals, not touching** | Top-Down | `centroid` + `centered_instance` |
| **Multiple animals, overlapping** | Bottom-Up | `bottomup` |
| **Known identities (tracking by ID)** | Multi-Class | `multi_class_bottomup` or `multi_class_topdown` |

For this tutorial, we'll use **Single Instance** (simplest case).

[:octicons-arrow-right-24: Learn more about model types](../reference/models.md)

---

## Step 2: Prepare Your Data

You need a SLEAP labels file (`.slp` or `.pkg.slp`) with:

- Labeled frames showing your animal's pose
- At least 50-100 labeled frames (more is better)

!!! tip "Don't have data?"
    Use our sample dataset:
    ```bash
    # Download sample fly data
    curl -O https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp
    curl -O https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp
    ```

---

## Step 3: Create Your Config

Create a file called `config.yaml`:

```yaml title="config.yaml"
# === DATA ===
data_config:
  train_labels_path:
    - train.pkg.slp       # Your training data
  val_labels_path:
    - val.pkg.slp         # Your validation data (optional)
  validation_fraction: 0.1  # Use 10% for validation if no val file

  # Preprocessing
  preprocessing:
    scale: 1.0            # Image scale (0.5 = half size, faster)

  # Augmentation (helps generalization)
  use_augmentations_train: true
  augmentation_config:
    geometric:
      rotation_min: -15.0
      rotation_max: 15.0
      scale_min: 0.9
      scale_max: 1.1
      affine_p: 0.5       # 50% chance of augmentation

# === MODEL ===
model_config:
  backbone_config:
    unet:
      filters: 32         # Model size (16=small, 32=medium, 64=large)
      max_stride: 16      # Receptive field size
      output_stride: 2    # Output resolution (lower = more precise)

  head_configs:
    single_instance:      # Change this for different model types
      confmaps:
        sigma: 5.0        # Gaussian spread for keypoints

# === TRAINING ===
trainer_config:
  max_epochs: 100         # Training epochs

  # Saving
  save_ckpt: true
  ckpt_dir: models
  run_name: fly_single_instance

  # Data loading
  train_data_loader:
    batch_size: 4         # Increase if you have more GPU memory

  # Optimization
  optimizer:
    lr: 0.0001            # Learning rate

  # Early stopping
  early_stopping:
    stop_training_on_plateau: true
    patience: 10          # Stop if no improvement for 10 epochs
```

---

## Step 4: Train

Run training:

```bash
sleap-nn train --config config.yaml
```

### Monitor Training

Watch the terminal output for:

- **Loss decreasing** over epochs (good!)
- **Validation loss** not increasing (no overfitting)

### Optional: Enable WandB

Add to your config for beautiful training dashboards:

```yaml
trainer_config:
  use_wandb: true
  wandb:
    project: sleap-nn-tutorial
    entity: your-username  # Your WandB username
```

### Training Output

After training, you'll find in `models/fly_single_instance/`:

```
models/fly_single_instance/
├── best.ckpt                  # Best model weights
├── initial_config.yaml        # Your original config
├── training_config.yaml       # Full config with auto-computed values
├── labels_gt.train.0.slp      # Training data split (ground truth)
├── labels_gt.val.0.slp        # Validation data split (ground truth)
├── labels_pr.train.slp        # Predictions on training data after training
├── labels_pr.val.slp          # Predictions on validation data after training
├── metrics.train.0.npz        # Training metrics
├── metrics.val.0.npz          # Validation metrics
└── training_log.csv           # Loss per epoch
```

---

## Step 5: Run Inference

Once training completes, run on `val.pkg.slp` (or a video):

```bash
sleap-nn track \
    --data_path val.pkg.slp \
    --model_paths models/fly_single_instance/ \
    -o predictions.slp
```

### Useful Options

```bash
# Faster inference with larger batches
sleap-nn track -i val.pkg.slp -m models/fly_single_instance/ --batch_size 8

# Process specific frames
sleap-nn track -i val.pkg.slp -m models/fly_single_instance/ --frames 0-1000

# Save to custom path
sleap-nn track -i val.pkg.slp -m models/fly_single_instance/ -o predictions.slp
```

---

## Step 6: Evaluate

Compare predictions to ground truth:

```bash
sleap-nn eval \
    --ground_truth_path val.pkg.slp \
    --predicted_path predictions.slp \
    --save_metrics metrics.npz
```

Or in Python:

```python
import sleap_io as sio
from sleap_nn.evaluation import Evaluator

gt = sio.load_slp("val.pkg.slp")
pred = sio.load_slp("predictions.slp")

evaluator = Evaluator(gt, pred)
metrics = evaluator.evaluate()

print(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']:.3f}")
print(f"Distance error (90th %ile): {metrics['distance_metrics']['p90']:.2f} px")
```

### Understanding Metrics

| Metric | What it measures | Good values |
|--------|-----------------|-------------|
| **OKS mAP** | Overall keypoint accuracy (0-1) | > 0.7 |
| **PCK** | % of keypoints within threshold | > 90% |
| **Distance p50** | Median error in pixels | < 5 px |
| **Distance p90** | 90th percentile error | < 10 px |

### Visualize Predictions

```python
import sleap_io as sio
import matplotlib.pyplot as plt

gt = sio.load_slp("val.pkg.slp")
pred = sio.load_slp("predictions.slp")

# Plot a frame
frame_idx = 0
fig, ax = plt.subplots()
ax.imshow(gt[frame_idx].image, cmap="gray")

# Ground truth (green circles)
for inst in gt[frame_idx].instances:
    pts = inst.numpy()
    ax.plot(pts[:, 0], pts[:, 1], "go", markersize=6, label="GT")

# Predictions (red crosses)
for inst in pred[frame_idx].instances:
    pts = inst.numpy()
    ax.plot(pts[:, 0], pts[:, 1], "rx", markersize=6, label="Pred")

ax.legend()
plt.show()
```

---

## What's Next?

### Improve Your Model

- **More data**: Label more frames, especially failure cases
- **Augmentation**: Enable more augmentations for robustness
- **Fine-tuning**: Start from pre-trained weights

### Multi-Animal Models

Ready for multiple animals?

- [:octicons-arrow-right-24: Top-Down models](../reference/models.md#top-down)
- [:octicons-arrow-right-24: Bottom-Up models](../reference/models.md#bottom-up)

### Production Deployment

Need faster inference?

- [:octicons-arrow-right-24: Export to ONNX/TensorRT](../guides/export.md)

---

## Troubleshooting

??? question "Training is slow"
    - Enable caching: `data_config.data_pipeline_fw: torch_dataset_cache_img_memory`
    - Reduce image size: `data_config.preprocessing.scale: 0.5`
    - Use a smaller model: `model_config.backbone_config.unet.filters: 16`

??? question "Poor predictions"
    - Train longer: increase `max_epochs`
    - Check your labels for errors
    - Add more training data

??? question "Out of memory"
    - Reduce batch size: `train_data_loader.batch_size: 2`
    - Reduce image size: `data_config.preprocessing.scale: 0.5`
    - Use a smaller model
