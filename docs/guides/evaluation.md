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
| `-s` / `--save_metrics` | Save metrics to a .npz file; a `.json` copy with the same stem is written next to it | None |
| `--oks_stddev` | OKS standard deviation | `0.025` |
| `--oks_scale` | Scale factor for OKS calculation | None |
| `--match_threshold` | Matching threshold: minimum OKS (`oks`), maximum pixel distance (`centroid`; `0` becomes `50` px), or minimum mask IoU (`mask`; `0` becomes `0.5`) | `0.0` |
| `--match_method` | `oks` (full skeleton), `centroid` (single-point distance), `mask` (instance-mask IoU), `semantic` (whole-frame foreground, no matching), `bbox` (COCO box AP), or `auto`: `bbox` when the predictions are boxes only, `centroid` when the prediction skeleton is single-node, else `oks` (never `mask`/`semantic`) | `auto` |
| `--bbox_source` | `bbox` mode: where ground-truth boxes come from (`auto`, `bboxes`, `instances`, `masks`); pass the model's `data_config.bbox_source` | `auto` |
| `--bbox_padding` | `bbox` mode: pixels of padding on ground-truth boxes derived from poses or masks; pass the model's `data_config.bbox_padding` | `0.0` |
| `--anchor_part` | GT node for centroid-mode ground-truth centroids (defaults to mean of visible nodes) | None |
| `--centroid_method` | How centroid-mode GT centroids are derived: `center_of_mass`, `bbox_center`, `geometric_median`, `anchor`. Pass the value the model was **trained** with (`head_configs.centroid.confmaps.centroid_method`) so the metric compares like with like | infer from `--anchor_part` |
| `--centroid_fallback` | Reduce method used when the `--anchor_part` node is not visible | `center_of_mass` |
| `--user_labels_only` / `--no-user_labels_only` | Only evaluate user-labeled frames | `True` |
| `--coco` | Also report COCOeval-faithful keypoint AP/AR (`coco.*`); see [COCO-Style Evaluation](coco-evaluation.md) | off |
| `--oks_sigmas` | Per-node OKS sigmas (`coco`, `crowdpose`, `ochuman`, `provenance`, `auto`, a float or a comma-separated list); replaces `--oks_stddev` | None |
| `--coco_protocol` / `--coco_area_source` | Dataset protocol preset and GT area source for `--coco` | `auto` / the protocol's |
| `--count_unmatched_as_fp` | Count unmatched predictions as false positives in `oks_voc.*` | off |

!!! warning "Comparing with papers"
    `oks_voc.mAP` / `oks_voc.mAR` are sleap-nn's original (legacy) OKS metrics and do
    not follow COCOeval. To compare with published human pose numbers, use `--coco`
    and see [COCO-Style Evaluation](coco-evaluation.md).

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
| `oks_voc.mAP` | Legacy COCO-style mean Average Precision (mean over OKS thresholds) | 0-1 |
| `oks_voc.mAR` | Legacy COCO-style mean Average Recall | 0-1 |
| `coco.AP`, `coco.AR`, ... | COCOeval-faithful AP/AR, with `--coco` (under `metrics['coco_metrics']`) | 0-1 |
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

### Point error relative to animal size (`norm_scale`)

Pixel errors depend on the image resolution and the animal's size, and the distance
metrics above score only the points a model predicted: a model that leaves hard points
out looks better. The size-relative error fixes both. For every labelled point of a
matched ground-truth instance it takes the distance to the predicted point divided by
the instance's size, and counts a point the model left unpredicted as an error of
`norm_missing_error` sizes (default `1.0`).

The size is `sqrt(width x height)` of a box:

- `norm_scale: bbox`: the user bounding box linked to the ground-truth instance, else
  the tight box of its labelled points;
- `norm_scale: points`: the tight box of its labelled points.

Any box built from points (under either setting) is padded on every side by
`norm_pad_frac` of its longest side (default `0.0`).

Instances with no box and fewer than two labelled points are skipped. It is opt-in
and off by default, for `match_method="oks"` only (not on the `sleap-nn eval` command
line):

```python
evaluator = Evaluator(gt, pred, norm_scale="bbox")
metrics = evaluator.evaluate()
nm = metrics["normalized_distance_metrics"]
print(nm["p50"], nm["p95"], nm["miss_rate"])
```

`normalized_distance_metrics` holds `p50`, `p75`, `p90`, `p95`, `p99` over all
labelled points, `found_p50` / `found_p95` over the predicted points only, `miss_rate`
(the share of labelled points left unpredicted) and the counts. With
`norm_bootstrap: N` it also gives 95% percentile-bootstrap intervals of `p50`, `p95`
and `miss_rate`, resampling frames or videos (`norm_bootstrap_by`).

In training, set the same keys under `trainer_config.eval` (see
[EvalConfig](../configuration/trainer.md#evalconfig)): the epoch-end evaluation then
logs `eval/<file>/norm_distance/p50`, `p95` and `miss_rate`, and the post-training
metrics include `normalized_distance_metrics`.

### Box cover measures (`bbox_cover_metrics`)

For box detectors, `trainer_config.eval.bbox_cover_metrics: true` (with
`eval.enabled`) adds measures that score whether each animal was found, apart from how
tight its box is. A predicted box **covers** a ground-truth box when it holds at least
half of the ground-truth box's area. Per frame, boxes are taken by descending score
and each goes to the free ground-truth box it covers most; a box that covers none (or
a second box on an animal already covered) is a false box. Logged under
`eval/val/`:

| Key | Meaning |
|---|---|
| `bbox_cover_recall_at_0.1fppf`, `_0.25fppf`, `_0.5fppf` | Share of animals covered at the lowest score threshold whose false boxes per frame stay within 0.1 / 0.25 / 0.5 |
| `bbox_cover_AP` | 101-point interpolated AP with the cover rule in place of IoU |
| `bbox_cover_recall`, `bbox_false_per_frame` | Covered share and false boxes per frame at score 0.1 |

Unlike IoU-based recall, the cover rule does not penalise a box for being larger than
a ground-truth box that is thin or partial (for example a box built from a few
keypoints), and reading the curve at a fixed number of false boxes per frame does not
depend on how the detector's scores are calibrated. In Python the same numbers come
from `sleap_nn.evaluation.compute_bbox_cover_metrics(frames)`, where each frame is a
dict with `gt_bboxes`, `pred_bboxes`, `pred_scores` and, optionally,
`gt_ignore_bboxes`: ignore regions (COCO "crowd"-like) in which a prediction that
covers the region and matches no kept box is neither a hit nor a false box.

---

## Loading Saved Metrics

```python
import numpy as np

data = np.load("metrics.npz", allow_pickle=True)
metrics = data['metrics'].item()

print(metrics.keys())
```

The metrics are also written as JSON next to the `.npz` (`metrics.json` here),
readable without NumPy; a few large raw arrays are left out of the JSON.

---

## Next Steps

- [:octicons-arrow-right-24: COCO-Style Evaluation](coco-evaluation.md) - Paper-comparable keypoint AP and box-seeded top-down inference
- [:octicons-arrow-right-24: Evaluation Metrics Reference](../reference/evaluation_metrics.md) - Deep dive into OKS, PCK, and other metrics
- [:octicons-arrow-right-24: Tracking](tracking.md) - Assign IDs across frames
- [:octicons-arrow-right-24: Export](export.md) - Deploy models
