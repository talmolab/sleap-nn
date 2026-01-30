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
| `--user_labels_only` | Only evaluate user-labeled frames | `False` |

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
print(f"mOKS: {metrics['mOKS']:.3f}")

# Distance errors
print(f"Mean error: {metrics['distance_metrics']['mean']:.2f} px")
print(f"Median error: {metrics['distance_metrics']['p50']:.2f} px")
print(f"90th %ile error: {metrics['distance_metrics']['p90']:.2f} px")

# Per-node metrics
for node, oks in metrics['per_node_oks'].items():
    print(f"  {node}: {oks:.3f}")
```

---

## Metrics Reference

### OKS (Object Keypoint Similarity)

Measures pose similarity accounting for keypoint visibility and scale:

| Metric | Description | Range |
|--------|-------------|-------|
| `mOKS` | Mean OKS across all instances | 0-1 |
| `oks_voc.mAP` | COCO-style mean Average Precision | 0-1 |
| `oks_voc.AP@0.5` | AP at OKS threshold 0.5 | 0-1 |
| `oks_voc.AP@0.75` | AP at OKS threshold 0.75 | 0-1 |

Higher is better. mAP > 0.7 is generally good.

### Distance Metrics

Euclidean distance between predicted and ground truth keypoints (in pixels):

| Metric | Description |
|--------|-------------|
| `mean` | Mean error |
| `std` | Standard deviation |
| `p50` | Median (50th percentile) |
| `p90` | 90th percentile |
| `p95` | 95th percentile |
| `p99` | 99th percentile |

Lower is better. Values depend on image resolution and animal size.

### Per-Node Metrics

OKS computed separately for each body part:

```python
for node, oks in metrics['per_node_oks'].items():
    print(f"{node}: {oks:.3f}")
```

Useful for identifying which keypoints are harder to predict.

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

- [:octicons-arrow-right-24: Tracking](tracking.md) - Assign IDs across frames
- [:octicons-arrow-right-24: Export](export.md) - Deploy models
