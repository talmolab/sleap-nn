# Evaluation

Compare predictions to ground truth and assess model performance.

---

## CLI Usage

```bash
sleap-nn eval \
    -g ground_truth.slp \
    -p predictions.slp \
    -s metrics.npz
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-g` / `--ground_truth_path` | Ground truth labels file | Required |
| `-p` / `--predicted_path` | Predicted labels file | Required |
| `-s` / `--save_metrics` | Save metrics to .npz file | None |
| `--oks_stddev` | OKS standard deviation | `0.025` |
| `--oks_scale` | Scale factor for OKS calculation | None |
| `--match_method` | Instance matcher: `oks`, `centroid`, `mask`, or `auto` (centroid when the prediction skeleton is single-node) | `auto` |
| `--anchor_part` | GT node for centroid-mode ground-truth centroids (defaults to mean of visible nodes) | None |
| `--user_labels_only` / `--no-user_labels_only` | Only evaluate user-labeled frames | `True` |

---

## Python API

### Basic Usage

```python
import sleap_io as sio
from sleap_nn.evaluation import Evaluator

gt = sio.load_slp("ground_truth.slp")
pred = sio.load_slp("predictions.slp")

evaluator = Evaluator(gt, pred)
metrics = evaluator.evaluate()
```

### Accessing Metrics

```python
# Overall metrics
print(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']:.3f}")
print(f"mOKS: {metrics['mOKS']['mOKS']:.3f}")

# Distance errors
print(f"Mean error: {metrics['distance_metrics']['avg']:.2f} px")
print(f"Median error: {metrics['distance_metrics']['p50']:.2f} px")
print(f"90th %ile error: {metrics['distance_metrics']['p90']:.2f} px")
```

---

## Metrics Reference

For a detailed explanation of all evaluation metrics, see the [Evaluation Metrics Reference](../reference/evaluation_metrics.md).

### OKS (Object Keypoint Similarity)

Measures pose similarity accounting for keypoint visibility and scale:

| Metric | Description | Range |
|--------|-------------|-------|
| `mOKS` | Mean OKS across all instances (access via `metrics['mOKS']['mOKS']`) | 0-1 |
| `oks_voc.mAP` | COCO-style mean Average Precision (mean over OKS thresholds) | 0-1 |
| `oks_voc.mAR` | COCO-style mean Average Recall | 0-1 |
| `oks_voc.AP` / `oks_voc.AR` | Per-OKS-threshold AP / AR arrays | 0-1 |

Higher is better. mAP > 0.7 is generally good.

### Distance Metrics

Euclidean distance between predicted and ground truth keypoints (in pixels):

| Metric | Description |
|--------|-------------|
| `avg` | Mean error |
| `p50` | Median (50th percentile) |
| `p75` | 75th percentile |
| `p90` | 90th percentile |
| `p95` | 95th percentile |
| `p99` | 99th percentile |

Lower is better. Values depend on image resolution and animal size. The raw
per-pair distances are also available under `metrics['distance_metrics']['dists']`.

---

## Loading Saved Metrics

```python
import numpy as np

data = np.load("metrics.npz", allow_pickle=True)
metrics = data['metrics'].item()

print(metrics.keys())
```

---

## Next Steps

- [:octicons-arrow-right-24: Evaluation Metrics Reference](../reference/evaluation_metrics.md) - Deep dive into OKS, PCK, and other metrics
- [:octicons-arrow-right-24: Tracking](tracking.md) - Assign IDs across frames
- [:octicons-arrow-right-24: Export](export.md) - Deploy models
