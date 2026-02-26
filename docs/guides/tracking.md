# Tracking

Assign consistent IDs to instances across frames.

---

## Enable Tracking

Add `--tracking` to your inference command:

```bash
sleap-nn track -i video.mp4 -m models/bottomup/ --tracking
```

---

## Tracking Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--tracking` / `-t` | Enable tracking | Flag | `False` |
| `--tracking_window_size` | Frames to look back | `INT` | `5` |
| `--min_new_track_points` | Min points for new track | `INT` | `0` |
| `--candidates_method` | Candidate selection method | `fixed_window`, `local_queues` | `fixed_window` |
| `--min_match_points` | Min non-NaN points for matching | `INT` | `0` |
| `--features` | Features for matching | `keypoints`, `centroids`, `bboxes`, `image` | `keypoints` |
| `--scoring_method` | Similarity scoring method | `oks`, `cosine_sim`, `iou`, `euclidean_dist` | `oks` |
| `--scoring_reduction` | Score reduction method | `mean`, `max`, `robust_quantile` | `mean` |
| `--track_matching_method` | Assignment algorithm | `hungarian`, `greedy` | `hungarian` |
| `--max_tracks` | Maximum track count | `INT` | `None` |
| `--use_flow` | Enable optical flow | Flag | `False` |

---

## Tracking Methods

### Fixed Window (Default)

Uses instances from the last N frames as matching candidates:

```bash
sleap-nn track -i video.mp4 -m models/ \
    -t \
    --candidates_method fixed_window \
    --tracking_window_size 10
```

**Best for**: Most scenarios, good balance of speed and accuracy.

### Local Queues

Maintains separate history for each track ID:

```bash
sleap-nn track -i video.mp4 -m models/ \
    -t \
    --candidates_method local_queues \
    --tracking_window_size 5
```

**Best for**: Robust to track breaks, handles occlusions better.

### Optical Flow

Uses optical flow to predict instance positions:

```bash
sleap-nn track -i video.mp4 -m models/ \
    -t \
    --use_flow
```

**Best for**: Fast-moving animals.

#### Optical Flow Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--of_img_scale` | Image scale (lower = faster) | `FLOAT` | `1.0` |
| `--of_window_size` | Window size per pyramid level | `INT` | `21` |
| `--of_max_levels` | Pyramid levels | `INT` | `3` |

---

## Track-Only Mode

Assign tracks to existing predictions (no inference):

```bash
sleap-nn track -i labels.slp --tracking
```

Note: Omit `--model_paths` for track-only mode.

With specific frames:

```bash
sleap-nn track -i labels.slp -t --frames 0-100 --video_index 0
```

---

## Limit Instances

```bash
# Maximum 5 instances per frame
sleap-nn track -i video.mp4 -m models/ --max_instances 5
```

---

## Example Configurations

### Fast Animals

```bash
sleap-nn track -i video.mp4 -m models/ \
    -t \
    --use_flow \
    --of_img_scale 0.5
```

### Crowded Scenes

```bash
sleap-nn track -i video.mp4 -m models/ \
    -t \
    --candidates_method local_queues \
    --tracking_window_size 10 \
    --max_tracks 10
```

### High Accuracy

```bash
sleap-nn track -i video.mp4 -m models/ \
    -t \
    --scoring_method oks \
    --scoring_reduction mean \
    --track_matching_method hungarian
```

---

## Troubleshooting

??? question "Tracks switch identities"
    - Increase `--tracking_window_size`
    - Try `--candidates_method local_queues`
    - Use `--use_flow` for fast motion

??? question "Too many tracks"
    - Set `--max_tracks` to limit track count
    - Increase `--min_new_track_points`

??? question "Tracking is slow"
    - Reduce `--tracking_window_size`
    - Use `--of_img_scale 0.5` with optical flow
