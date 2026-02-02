# Monitoring & Visualization

Monitor training progress with logging, visualization, and evaluation callbacks.

---

## Weights & Biases Integration

Enable WandB for experiment tracking:

```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: your-username
    project: your-project
```

### Visualization Options

```yaml
trainer_config:
  wandb:
    viz_enabled: true           # Pre-rendered matplotlib images
    viz_boxes: false            # Interactive keypoint boxes with epoch slider
    viz_masks: false            # Confidence map overlay masks
    viz_box_size: 5.0           # Size of keypoint boxes in pixels
    viz_confmap_threshold: 0.1  # Threshold for confmap masks
```

| Option | Description | Default |
|--------|-------------|---------|
| `viz_enabled` | Log pre-rendered prediction images | `true` |
| `viz_boxes` | Interactive keypoint boxes (epoch slider) | `false` |
| `viz_masks` | Confidence map overlay masks | `false` |
| `viz_box_size` | Size of keypoint boxes in pixels | `5.0` |
| `viz_confmap_threshold` | Threshold for confmap mask generation | `0.1` |

!!! tip "Interactive Epoch Slider"
    Enable `viz_boxes: true` to scrub through epochs and see predictions improve over time.

---

## Local Visualizations

Save prediction visualizations to disk during training:

```yaml
trainer_config:
  visualize_preds_during_training: true
  keep_viz: false  # Set true to keep viz folder after training
```

Visualizations are saved to a `viz/` folder in the checkpoint directory.

---

## Per-Head Loss Monitoring

Multi-head models (e.g., bottom-up) log individual losses:

| Metric | Description |
|--------|-------------|
| `train_confmap_loss` / `val_confmap_loss` | Confidence map head loss |
| `train_paf_loss` / `val_paf_loss` | Part affinity field loss (bottom-up only) |

This helps diagnose when individual heads aren't learning effectively.

---

## Epoch-End Evaluation

Compute evaluation metrics during training to track model quality beyond loss values.

### Enable Evaluation

```yaml
trainer_config:
  eval:
    enabled: true
    frequency: 1  # Evaluate every N epochs
```

### How It Works

SLEAP-NN automatically selects the appropriate evaluation callback based on model type:

| Model Type | Callback | Metrics |
|------------|----------|---------|
| Single Instance | `EpochEndEvaluationCallback` | OKS, PCK, distance metrics |
| Bottom-Up | `EpochEndEvaluationCallback` | OKS, PCK, distance metrics |
| Top-Down (Centered Instance) | `EpochEndEvaluationCallback` | OKS, PCK, distance metrics |
| **Centroid** | `CentroidEvaluationCallback` | Distance, precision/recall |

### Pose Model Evaluation (OKS-based)

For pose models (single instance, bottom-up, centered instance), evaluation uses Object Keypoint Similarity (OKS) metrics:

```yaml
trainer_config:
  eval:
    enabled: true
    frequency: 1
    oks_stddev: 0.025   # OKS standard deviation
    oks_scale: null     # OKS scale override (null = auto)
```

**Metrics logged to WandB:**

| Metric | Description |
|--------|-------------|
| `eval/val/oks_voc` | OKS-based VOC score |
| `eval/val/pck` | Percentage of Correct Keypoints |
| `eval/val/dist_p50` | Median distance (pixels) |
| `eval/val/dist_p90` | 90th percentile distance |
| `eval/val/dist_p95` | 95th percentile distance |

### Centroid Model Evaluation (Distance-based)

Centroid models use distance-based metrics appropriate for point detection:

```yaml
trainer_config:
  eval:
    enabled: true
    frequency: 1
    match_threshold: 50.0  # Max distance (px) for matching pred to GT
```

!!! info "Hungarian Matching"
    Predictions are matched to ground truth using the Hungarian algorithm for optimal assignment. Only matches within `match_threshold` pixels are considered true positives.

**Metrics logged to WandB:**

| Metric | Description |
|--------|-------------|
| `eval/val/centroid_dist_avg` | Mean Euclidean distance (pixels) |
| `eval/val/centroid_dist_median` | Median distance |
| `eval/val/centroid_dist_p90` | 90th percentile distance |
| `eval/val/centroid_dist_p95` | 95th percentile distance |
| `eval/val/centroid_dist_max` | Maximum distance |
| `eval/val/centroid_precision` | TP / (TP + FP) |
| `eval/val/centroid_recall` | TP / (TP + FN) |
| `eval/val/centroid_f1` | F1 score |
| `eval/val/centroid_n_tp` | True positive count |
| `eval/val/centroid_n_fp` | False positive count |
| `eval/val/centroid_n_fn` | False negative count |

---

## Configuration Reference

### Full Eval Config

```yaml
trainer_config:
  eval:
    enabled: bool       # Enable epoch-end evaluation (default: false)
    frequency: int      # Evaluate every N epochs (default: 1)
    oks_stddev: float   # OKS standard deviation (default: 0.025)
    oks_scale: float    # OKS scale override, null for auto (default: null)
    match_threshold: float  # Centroid matching threshold in pixels (default: 50.0)
```

See [Trainer Configuration](../configuration/trainer.md#evalconfig) for complete reference.

---

## Example Configurations

### Bottom-Up with Full Monitoring

```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: my-team
    project: pose-estimation
    viz_enabled: true
    viz_boxes: true

  eval:
    enabled: true
    frequency: 5  # Evaluate every 5 epochs
    oks_stddev: 0.025

  visualize_preds_during_training: true
```

### Centroid Model with Evaluation

```yaml
trainer_config:
  use_wandb: true
  wandb:
    entity: my-team
    project: pose-estimation
    viz_enabled: true

  eval:
    enabled: true
    frequency: 1
    match_threshold: 30.0  # Stricter matching for small animals
```

---

## Next Steps

- [:octicons-arrow-right-24: Evaluation Guide](evaluation.md) - Post-training evaluation
- [:octicons-arrow-right-24: Evaluation Metrics Reference](../reference/evaluation_metrics.md) - Detailed metric explanations
- [:octicons-arrow-right-24: Configuration Reference](../configuration/trainer.md) - Full trainer config options
