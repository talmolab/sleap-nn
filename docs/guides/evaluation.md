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

## Plotting Metrics

`sleap_nn.evaluation_plots.plot_metrics` draws the saved metrics. Point it at a
model directory and it reads the metrics `.npz`, the node names from
`training_config.yaml`, and the loss curve from `training_log.csv`:

```python
from sleap_nn.evaluation_plots import plot_metrics

plot_metrics("models/my_model", kind="dashboard", save_path="eval.png")
```

The `dashboard` default draws every panel the metrics support. Ask for a single
one with `kind`:

| `kind` | Shows |
|---|---|
| `error_distribution` | Histogram of localization error with p50/p90/p99 marks, clipped at p99 so one outlier does not flatten the bulk |
| `error_by_node` | Per-node error box plot, worst node first |
| `pck_curve` | PCK against distance threshold |
| `pck_by_node` | Mean PCK per node |
| `precision_recall` | Precision-recall curves at each OKS threshold |
| `visibility` | Node-visibility confusion matrix |
| `training_curve` | Train and validation loss per epoch |
| `dashboard` | All of the above in one figure |

Every panel is also a standalone function taking an `ax`, so they compose into
your own figures:

```python
import matplotlib.pyplot as plt
from sleap_nn.evaluation import load_metrics
from sleap_nn.evaluation_plots import plot_error_by_node, load_node_names

metrics = load_metrics("models/my_model", split="val")
names = load_node_names("models/my_model")

fig, ax = plt.subplots(figsize=(7, 4))
plot_error_by_node(metrics, ax=ax, node_names=names)
```

!!! note "Keypoint plots need keypoint metrics"

    Only `match_method="oks"` produces `pck_metrics` and `visibility_metrics`.
    Centroid, mask, semantic, and bbox evaluations report a different set, and
    asking for a plot they cannot support raises `MetricsNotAvailable` naming
    what the file actually contains.

### Finding the Worst Frames

The metrics file records a video path and frame index alongside every distance,
so the largest errors can be traced back to specific frames and looked at:

```python
from sleap_nn.evaluation_plots import worst_instances

for row in worst_instances(metrics, n=5):
    print(f"frame {row['frame_idx']}: {row['error']:.1f} px mean, "
          f"{row['max_error']:.1f} px worst node")
```

Feed those frame indices to
[`sleap_io.render_image`](https://io.sleap.ai/latest/rendering/) to see the
prediction overlaid on the frame. A mean error of 1 px with a p99 of 11 px means
the model is fine and a handful of frames are not — and this is how you find
which ones.

---

## Next Steps

- [:octicons-arrow-right-24: Evaluation Metrics Reference](../reference/evaluation_metrics.md) - Deep dive into OKS, PCK, and other metrics
- [:octicons-arrow-right-24: Tracking](tracking.md) - Assign IDs across frames
- [:octicons-arrow-right-24: Export](export.md) - Deploy models
