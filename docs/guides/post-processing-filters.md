# Post-Processing Filters

Post-processing filters remove low-quality or duplicate predictions before tracking.

## Node Count Filter

Remove instances with too few detected keypoints:

```bash
# Require at least 3 visible nodes
sleap-nn predict -i video.mp4 -m models/ --filter_min_visible_nodes 3

# Require at least 50% of skeleton nodes to be visible
sleap-nn predict -i video.mp4 -m models/ --filter_min_visible_node_fraction 0.5
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter_min_visible_nodes` | Minimum number of visible keypoints | `0` (disabled) |
| `--filter_min_visible_node_fraction` | Minimum fraction of skeleton nodes | `0.0` (disabled) |

## Confidence Score Filter

Remove instances with low confidence scores:

```bash
# Require mean node confidence >= 0.4
sleap-nn predict -i video.mp4 -m models/ --filter_min_mean_node_score 0.4

# Require instance score >= 0.3
sleap-nn predict -i video.mp4 -m models/ --filter_min_instance_score 0.3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter_min_mean_node_score` | Minimum mean confidence across visible nodes | `0.0` (disabled) |
| `--filter_min_instance_score` | Minimum overall instance score | `0.0` (disabled) |

!!! note "Instance score differs by model type"
    The instance score comes from different sources depending on model type:

    - **Top-down**: Instance score is the **centroid confidence** (how confident the model was that an animal exists at that location)
    - **Bottom-up**: Instance score is derived from **PAF grouping** (how well the keypoints connected together)

## Overlap Filter

Remove duplicate detections with greedy NMS:

```bash
# Enable with default IOU
sleap-nn predict -i video.mp4 -m models/ --filter_overlapping

# Use OKS with custom threshold
sleap-nn predict -i video.mp4 -m models/ \
    --filter_overlapping \
    --filter_overlapping_method oks \
    --filter_overlapping_threshold 0.5
```

| Method | Description |
|--------|-------------|
| `iou` | Bounding box intersection-over-union |
| `oks` | Object Keypoint Similarity (pose-aware) |

| Threshold | Effect |
|-----------|--------|
| 0.3 | Aggressive filtering |
| 0.5 | Moderate |
| 0.8 | Permissive (default) |

## COCO Rescoring and OKS-NMS

Top-down human pose papers (mmpose / ViTPose protocol) rescore each instance and
deduplicate by OKS before evaluating:

```bash
sleap-nn predict -i coco_val2017.slp -m models/centered_instance/ \
    --boxes dets.json --peak_threshold 0 --rescore --oks_nms 0.9
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rescore` / `--no-rescore` | Instance score = stage-1 score (box score / centroid value) × mean of keypoint scores above `--rescore_threshold` | on with `--boxes`, else off |
| `--rescore_threshold` | Keypoint-score threshold for `--rescore` | `0.2` |
| `--oks_nms` | Drop an instance whose OKS with a higher-scoring instance in the frame exceeds this (mmpose `oks_nms`, per-node sigmas, mean crop-window area) | `0` (off) |
| `--oks_sigmas` | Per-node sigmas for `--oks_nms` | by node name |

Unlike `--filter_overlapping_method oks` (a legacy OKS with one falloff constant and
the instance's own keypoint box as its scale), `--oks_nms` uses COCO's per-node OKS.
See [COCO-Style Evaluation](coco-evaluation.md).

## Combining Filters

All filters can be combined. They are applied in order: rescoring → node count →
confidence → OKS-NMS → overlap.

**Example: Strict filtering for clean output**

```bash
sleap-nn predict -i video.mp4 -m models/ \
    --filter_min_visible_nodes 2 \
    --filter_min_visible_node_fraction 0.25 \
    --filter_min_mean_node_score 0.3 \
    --filter_overlapping \
    --filter_overlapping_threshold 0.5
```

## Common Use Cases

**Use case 1: Known number of animals (top-down)**

You know there are exactly 3 mice in the video:

```bash
sleap-nn predict -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/ \
    --max_instances 3
```

**Use case 2: Remove false positives (bottom-up)**

Your bottom-up model produces spurious partial detections:

```bash
sleap-nn predict -i video.mp4 -m models/bottomup/ \
    --filter_min_visible_node_fraction 0.5 \
    --filter_min_instance_score 0.3
```

**Use case 3: Crowded scenes with overlapping animals**

Animals frequently overlap, causing duplicate detections:

```bash
sleap-nn predict -i video.mp4 -m models/ \
    --filter_overlapping \
    --filter_overlapping_method oks \
    --filter_overlapping_threshold 0.4
```

**Use case 4: High-quality predictions only (top-down)**

Keep all keypoints but remove low-confidence instances:

```bash
sleap-nn predict -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/ \
    --peak_threshold 0.1 \
    --filter_min_instance_score 0.4 \
    --filter_min_visible_node_fraction 0.75
```

!!! note "Inference only"
    Filters are only applied during inference. When running in **track-only mode** (without model paths), these parameters have no effect.
