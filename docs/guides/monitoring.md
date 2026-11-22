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

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `viz_enabled` | Log pre-rendered prediction images | `bool` | `true` |
| `viz_boxes` | Interactive keypoint boxes (epoch slider) | `bool` | `false` |
| `viz_masks` | Confidence map overlay masks | `bool` | `false` |
| `viz_box_size` | Size of keypoint boxes in pixels | `float` | `5.0` |
| `viz_confmap_threshold` | Threshold for confmap mask generation | `float` | `0.1` |

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
| `train/confmaps_loss` / `val/confmaps_loss` | Confidence map head loss |
| `train/paf_loss` / `val/paf_loss` | Part affinity field loss (bottom-up only) |

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
| Multi-Class (`multi_class_bottomup`, `multi_class_topdown`) | `EpochEndEvaluationCallback` | OKS, PCK, distance metrics |
| **Centroid** | `CentroidEvaluationCallback` | Distance, precision/recall |
| Segmentation (bottom-up, centered-instance, semantic) | `SegmentationEvaluationCallback` | Instance: matched mask IoU, clDice, precision / recall / F1; semantic: foreground IoU, clDice, boundary IoU (see [below](#segmentation-model-evaluation)) |
| Box detectors (`bbox_centernet`, `bbox_ssd`) | `BBoxEvaluationCallback` | `eval/val/bbox_*` (COCO-style mAP, AP50/75, AR, F1) — see [Bounding boxes](bbox-detection.md) |
| Embedding | `EmbeddingEvaluationCallback` | Retrieval metrics (rank-1, mAP, ...); runs even with `eval.enabled: false` |

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
| `eval/val/mOKS` | Mean OKS |
| `eval/val/oks_voc_mAP` / `eval/val/oks_voc_mAR` | OKS-based VOC mean average precision / recall |
| `eval/val/mPCK` | Mean Percentage of Correct Keypoints |
| `eval/val/PCK_5` / `eval/val/PCK_10` | PCK at 5 / 10 px |
| `eval/val/distance/avg` | Mean keypoint distance (pixels) |
| `eval/val/distance/p50` / `p95` / `p99` | Distance percentiles (pixels) |
| `eval/val/visibility_precision` / `eval/val/visibility_recall` | Node visibility precision / recall |

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

### Segmentation Model Evaluation

`bottomup_segmentation` and `centered_instance_segmentation` match predicted to ground-truth masks by IoU (pairs below the IoU threshold don't count; the threshold is `match_threshold` when it is in `(0, 1]`, else `0.5`). `semantic_segmentation` has no instances, so it scores each frame's whole foreground without matching.

**Instance mode (metrics logged to WandB):**

| Metric | Description |
|--------|-------------|
| `eval/val/mask_mean_iou` | Mean IoU over matched mask pairs |
| `eval/val/mask_mean_iou_all_gt` | Mean IoU over all GT masks (missed GT counts as 0) |
| `eval/val/mask_mean_cldice` | Mean clDice over matched pairs |
| `eval/val/mask_precision` / `mask_recall` / `mask_f1` | Detection precision / recall / F1 |
| `eval/val/mask_n_tp` / `mask_n_fp` / `mask_n_fn` | Match counts |

**Semantic mode (metrics logged to WandB):**

| Metric | Description |
|--------|-------------|
| `eval/val/fg_mean_iou` | Mean foreground IoU per frame |
| `eval/val/fg_mean_cldice` | Mean foreground clDice per frame |
| `eval/val/fg_mean_boundary_iou` | Mean foreground boundary IoU per frame |
| `eval/val/fg_frame_recall` | Fraction of frames where the model predicted any foreground |

In semantic mode, frames with an empty ground-truth foreground are skipped. Mask AP is not computed during training; it comes from post-training evaluation (the run's `metrics.*.npz`, or `sleap-nn eval --match_method mask`).

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
    select_metric: str  # Checkpoint metric for embedding / bbox models (default: rank1)
```

`select_metric` is ignored by every other model type, which selects checkpoints on `val/loss`. For `embedding` it is one of `rank1`, `mAP`, `auc`, `knn_acc`, `eer`. For the `bbox_*` detectors it is one of `bbox_mAP`, `bbox_AP50`, `bbox_AP75`, `bbox_AR`, `bbox_f1`, `val_loss`; left at the default `rank1`, a detector selects on `bbox_mAP` when `enabled: true` and on `val/loss` otherwise.

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
