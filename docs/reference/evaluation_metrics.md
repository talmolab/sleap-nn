# Evaluation Metrics

Here, we explain the various evaluation metrics we output at the end of running inference with a trained SLEAP model. Pose models (`--match_method oks`) report:

- [Distance metrics](#distance-metrics)
- [Object Keypoint similarity (OKS)](#object-keypoint-similarity-oks)
- [Percentage of Correct Keypoints (PCK) metrics](#percentage-of-correct-keypoints-pck-metrics)
- [Visibility metrics](#visibility-metrics)
- [VOC metrics](#voc-metrics)

Other model types use their own metrics:

- [Centroid metrics](#centroid-metrics) (centroid models, `--match_method centroid`)
- [Mask metrics](#mask-metrics) (instance segmentation, `--match_method mask`)
- [Semantic metrics](#semantic-metrics) (whole-frame segmentation, `--match_method semantic`)
- [Bounding-box metrics](#bounding-box-metrics) (detectors, `--match_method bbox`)

## Distance Metrics

This metric computes the Euclidean distance between pairs of predicted and ground-truth (gt) instances. For each instance pair, we calculate the L2 norm of the difference between the predicted and corresponding ground-truth keypoints. The following statistics are reported:

- *`Avg_dist`*: Mean Euclidean distance across all (pred, gt) pairs.

- *`Dist@k`*: Percentile-based distance metrics which includes the distance at the 50th, 75th, 90th, 95th, and 99th percentiles (denoted as p50, p75, p90, p95, p99).

These metrics provide insight into the distribution of how far off the predictions are from the ground-truth key-points.


## Object-keypoint similarity (OKS)

This returns the mean OKS score between every pair of ground truth and
predicted instance, ranging from 0 to 1.0 and 1.0 indicating a perfect match. OKS provides a measure of similarity between ground-truth and predicted pose by taking into account the instance size (scale) and node visibility.

OKS is computed by measuring the Euclidean distance between each predicted keypoint and its corresponding ground-truth keypoint. This distance is then normalized based on the scale of the object (bounding box area for the instance) and standard-deviation that defines the spread in the localization accuracy of each node. For each node, keypoint similarity is computed by taking the negative exponent of the normalized distance. Mean OKS is the average of keypoint similarities across all visible nodes.

The implementation is based off of the descriptions in: [Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation." ICCV (2017)](https://arxiv.org/abs/1707.05388).

## Percentage of Correct Keypoints (PCK) metrics

This metric measures the fraction of keypoints that fall within a certain pixel distance (threshold) from the ground-truth location. This is useful to evaluate how precise the predicted points are. The following are generated using PCK metric:

- *`PCKs`*: PCK on each predicted instances for each node at different thresholds. (Thresholds: [1, 2, 3, ..., 10])
- *`mPCK part`*: Mean PCK per node averaged over all predicted instances across thresholds.
- *`mPCK`*: Mean PCK across all predicted nodes and thresholds.

## Visibility metrics

This metric evaluates the visibility accuracy of the predicted nodes. It measures how well the model identifies whether the keypoint is present or missing - independent of its spatial accuracy (i.e. distance from ground-truth). This is useful for evaluating models on datasets with occlusions (NaN nodes).

The following statistics are computed across all matched instance pairs:

- *`True positives (TP)`*: Node is visible in both ground-truth and prediction.
- *`False positives (FP)`*: Node is missing in ground-truth but visible in prediction.
- *`True negatives (TN)`*: Node is missing in both ground-truth and prediction.
- *`False negatives (FN)`*: Node is visible in ground-truth but missing in prediction.
- *`Precision (TP / (TP + FP))`*: Proportion of predicted visible nodes that are correct.
- *`Recall (TP / (TP + FN))`*: Proportion of actual visible nodes that were correctly predicted.

## VOC metrics

The following VOC-style metrics are generated using either OKS or PCK as the matching scores and a set of thresholds where a predicted instance is considered as a True Positive if its match score is greater than the threshold else it is counted as a False Positive.

- *`Average Precision (AP)`*: Average of best precisions over fixed set of recall thresholds, at each match score threshold.
- *`Average Recall (AR)`*: Maximum recall achieved at each match score threshold.
- *`Mean Average Precision (mAP)`*: Mean of average precisions across match thresholds.
- *`Mean Average Recall (mAR)`*: Mean of average recalls across match thresholds.

To learn how to generate these metrics using sleap-nn, see the [Evaluation guide](../guides/evaluation.md).

## Centroid Metrics

Centroid models predict a single point (centroid) per instance rather than full pose skeletons. These models use specialized distance-based metrics that are more appropriate for point detection tasks than OKS/PCK metrics.

### Prediction Matching

Predictions are matched to ground truth using the **Hungarian algorithm** (optimal bipartite matching) to minimize total distance. A `match_threshold` parameter (default: 50 pixels) determines the maximum distance for a valid match:

- **True Positive (TP)**: Prediction matched to GT within threshold
- **False Positive (FP)**: Prediction with no GT match within threshold
- **False Negative (FN)**: GT with no prediction match within threshold

### Metrics

The same quantities are reported under different keys depending on where they come from.

**Post-training / `sleap-nn eval`** (`run_evaluation(match_method="centroid")`) returns two dicts:

| Key | Description |
|-----|-------------|
| `detection_metrics["precision"]` | TP / (TP + FP): proportion of predictions that are correct |
| `detection_metrics["recall"]` | TP / (TP + FN): proportion of GT instances that were detected |
| `detection_metrics["f1"]` | 2 * P * R / (P + R) |
| `detection_metrics["n_tp"]`, `["n_fp"]`, `["n_fn"]` | Counts of true positives, false positives, false negatives |
| `distance_metrics["avg"]` | Mean Euclidean distance (pixels) over matched pairs |
| `distance_metrics["p50"]` … `["p99"]` | 50th, 75th, 90th, 95th and 99th percentile distances (also copied into `detection_metrics`) |

**Epoch-end evaluation during training** (`trainer_config.eval.enabled: true`) logs these keys:

| Key | Description |
|-----|-------------|
| `eval/val/centroid_dist_avg` | Mean distance (pixels) |
| `eval/val/centroid_dist_median`, `_p90`, `_p95`, `_max` | Median, 90th/95th percentile and maximum distance |
| `eval/val/centroid_precision`, `_recall`, `_f1` | Detection precision, recall, F1 |
| `eval/val/centroid_n_tp`, `_n_fp`, `_n_fn` | Counts of true positives, false positives, false negatives |

### When to Use

Centroid metrics are used during epoch-end evaluation when training centroid models with `trainer_config.eval.enabled: true`, and for post-training evaluation of centroid predictions (`sleap-nn eval --match_method centroid`, or `auto` with a single-node prediction skeleton).

!!! tip "Adjusting match_threshold"
    The `match_threshold` parameter should be set based on your expected centroid accuracy:

    - **Smaller threshold** (e.g., 20px): Stricter matching, may undercount true positives
    - **Larger threshold** (e.g., 100px): More lenient, may incorrectly match distant predictions

    For most applications, the default of 50 pixels works well.

## Mask Metrics

Instance-segmentation models (`bottomup_segmentation`, `centered_instance_segmentation`) are evaluated with `--match_method mask`. Predicted and GT masks are matched one-to-one by mask IoU with the Hungarian algorithm; `match_threshold` is the minimum IoU (default `0.5`). Results come in three dicts:

- **`detection_metrics`**: `precision`, `recall`, `f1`, `n_tp`, `n_fp`, `n_fn` over the IoU-matched masks.
- **`mask_metrics`**:
    - `mean_iou` (plus `min`, `max`, `p25`, `p50`, `p75`) over matched pairs only.
    - `mean_iou_all_gt`: IoU averaged over every GT mask, with a miss counted as `0`.
    - Panoptic quality `pq = sq * rq`, where `sq` is the matched mean IoU and `rq` = TP / (TP + 0.5 FP + 0.5 FN).
    - `mean_boundary_iou` and `mean_cldice` (centerline Dice, for thin structures) over matched pairs.
    - `oversegmentation` / `undersegmentation`: GT masks split across ≥ 2 predictions, and predictions spanning ≥ 2 GT masks (each covering ≥ 10% of the GT mask's area).
    - `per_size` (small/medium/large breakdown), `n_matched`, `n_fp`, `n_fn` and the raw `ious`.
- **`mask_voc_metrics`**: COCO-style score-ranked mask AP/AR over IoU 0.50:0.05:0.95. Keys are `mask_voc.mAP`, `mask_voc.AP50`, `mask_voc.AP75`, `mask_voc.AR`, the per-threshold `mask_voc.AP`, and per-size `mask_voc.AP_small` / `_medium` / `_large`. The same per-size APs computed with COCO area edges are under `mask_voc.coco.*`.

## Semantic Metrics

Whole-frame `semantic_segmentation` models are evaluated with `--match_method semantic`. There is no matching: each frame's masks are unioned into one foreground and compared with the GT foreground, and frames with an empty GT foreground are skipped. `semantic_metrics` holds:

- `mean_iou`, `mean_cldice` and `mean_boundary_iou`.
- The per-frame `ious` / `cldices` / `boundary_ious`.
- `n_frames` (frames scored).

## Bounding-box Metrics

Detectors (`bbox_centernet`, `bbox_ssd`) are evaluated with `--match_method bbox`, following the COCO protocol. Per class and IoU threshold, each frame's top-100 detections are greedily matched in descending score order to the highest-IoU unmatched GT box. AP is the 101-point interpolated area under the precision-recall curve. The result is a flat dict:

| Key | Description |
|-----|-------------|
| `bbox_mAP` | AP averaged over IoU 0.50:0.05:0.95 and classes |
| `bbox_AP50`, `bbox_AP75` | AP at IoU 0.5 / 0.75 |
| `bbox_AR` | Maximum recall, averaged over the same IoU thresholds and classes |
| `bbox_precision`, `bbox_recall`, `bbox_f1` | Operating point at IoU 0.5 and the score threshold that maximizes F1 |
| `bbox_f1_threshold` | That score threshold |
| `bbox_mean_iou` | Mean IoU of the matches at the operating point |
| `bbox_n_tp`, `bbox_n_fp`, `bbox_n_fn` | Counts at the operating point |
| `n_frames`, `n_gt`, `n_pred` | Frames, GT boxes and predicted boxes evaluated |
| `ap_per_class`, `ar_per_class`, `pr_curve`, `categories` | Per-class AP/AR arrays, the IoU-0.5 precision-recall curve, and the class names |

See the [Bounding-box detection guide](../guides/bbox-detection.md#evaluation) for how GT boxes are derived and how classes are matched.
