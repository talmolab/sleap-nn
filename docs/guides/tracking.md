# Tracking

Assign consistent IDs to instances across frames.

---

## Enable Tracking

Add `--tracking` to your inference command:

```bash
sleap-nn predict -i video.mp4 -m models/bottomup/ --tracking
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
| `--max_tracks` | Maximum track count (auto-selects `local_queues`) | `INT` | `None` |
| `--use_flow` | Enable optical flow | Flag | `False` |
| `--use_kalman` | Enable Kalman-filter tracking | Flag | `False` |
| `--kf_init_frame_count` | Warm-up frames before EM init | `INT` | `10` |
| `--kf_node_indices` | Node indices to filter (comma-sep; empty = all) | e.g. `0,1,2` | `None` |
| `--kf_reset_gap_size` | Missed frames before a stale track resets | `INT` | `5` |

---

## Tracking Methods

### Fixed Window (Default)

Uses instances from the last N frames as matching candidates:

```bash
sleap-nn predict -i video.mp4 -m models/ \
    -t \
    --candidates_method fixed_window \
    --tracking_window_size 10
```

**Best for**: Most scenarios, good balance of speed and accuracy.

### Local Queues

Maintains separate history for each track ID:

```bash
sleap-nn predict -i video.mp4 -m models/ \
    -t \
    --candidates_method local_queues \
    --tracking_window_size 5
```

**Best for**: Robust to track breaks, handles occlusions better.

!!! note "`--max_tracks` requires `local_queues`"
    `--max_tracks` (the cap on how many track IDs may be created) is honored
    **only** by `local_queues`; `fixed_window` ignores it. Setting `--max_tracks`
    therefore auto-selects `candidates_method local_queues` (logged at INFO),
    overriding `fixed_window` even if you pass it explicitly. You do **not** need
    to set `--candidates_method` yourself when capping the track count.

### Optical Flow

Uses optical flow to predict instance positions:

```bash
sleap-nn predict -i video.mp4 -m models/ \
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

### Kalman Filter

Tracks each identity with a per-track constant-velocity Kalman filter on the
instance **centroid**. The tracker runs a normal-tracker warm-up for
`--kf_init_frame_count` frames, fits one centroid filter per track via EM, and then
predicts each track's centroid forward and scores the current detections against the
last observed pose **rigidly translated** by that predicted displacement (analogous to
optical flow, but using a motion model instead of image displacements):

```bash
sleap-nn predict -i video.mp4 -m models/centroid models/centered_instance \
    -t \
    --use_kalman \
    --tracking_target_instance_count 2 \
    --kf_init_frame_count 10
```

**Best for**: A known, fixed number of animals whose motion is informative for
association — identities that **cross or pass close** to each other, **converge**, or
move **fast and smoothly** — after the base detector has been culled to the top-N
instances per frame. In these regimes the motion prediction substantially reduces ID
switches over plain similarity matching.

!!! note "When it helps vs. when it doesn't"
    The motion model is a net win where association is ambiguous (crossings, converging
    or fast tracks). Under **heavy detection noise with frequent missed detections on
    long sequences** it can slightly *reduce* IDF1 versus the memoryless similarity
    tracker (any motion prediction occasionally causes a swap a memoryless tracker
    avoids) — though it usually still produces fewer ID switches there. If your
    detections are very noisy, prefer the plain tracker or lower `kf_prediction_blend`
    (e.g. `Tracker.from_config(..., kf_prediction_blend=0.25)`).

Notes:

- Requires a known target identity count: pass `--tracking_target_instance_count`
  (or let it be derived from `--max_instances` / `--max_tracks`).
- Mutually exclusive with `--use_flow`.
- Use `--kf_node_indices` to filter on a stable subset of nodes (e.g. spine nodes):
  `--kf_node_indices 0,1,2`. Leave it unset to use all nodes.
- Depends on the `pykalman` package (a core dependency).

!!! note "Centroid vs keypoints (`--kf_track_features`)"
    By default (`--kf_track_features centroid`) the motion model tracks each instance's
    **centroid** and rigidly translates the last pose — stable and the recommended
    choice. `--kf_track_features keypoints` instead runs one filter **per node** and uses
    the predicted pose directly: it can help when subjects are small and move
    distinctively (e.g. it cut ID switches markedly on a 2-fly clip), but the per-node
    prediction is noisier, so it needs a tolerant similarity score. Pair it with the
    auto-default `--oks_stddev 0.1` (set automatically for keypoints mode; **do not** use
    the strict 0.025), or with `--features bboxes --scoring_method iou` for noisy but
    well-separated, non-rotating subjects. On clean or occlusion-heavy data the centroid
    mode (or the plain tracker) is at least as good, so keypoints mode is an opt-in
    alternative, not a replacement.

The motion model is robustified so it does not degrade tracking outside its sweet
spot: each correction is gated by distance (rejecting false-positive / mismatched
detections), the learned velocity is capped, the filter coasts across occlusion gaps,
stale tracks are reset, and the scoring candidate blends the prediction with the last
observation. These robustness parameters (`kf_prediction_blend`, the gate and
velocity-cap multipliers) have tuned defaults and can be overridden when constructing a
tracker via `Tracker.from_config(...)`.

#### Kalman Filter Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--kf_track_features` | What the motion model tracks | `centroid`, `keypoints` | `centroid` |
| `--oks_stddev` | OKS keypoint-spread tolerance (larger = more forgiving) | `FLOAT` | `0.025`; `0.1` for `keypoints` |
| `--kf_init_frame_count` | Warm-up frames before EM init | `INT` | `10` |
| `--kf_node_indices` | Node indices to filter (comma-sep; empty = all) | e.g. `0,1,2` | `None` |
| `--kf_reset_gap_size` | Missed frames before a stale track resets | `INT` | `5` |

---

## Track-Only Mode

Assign tracks to existing predictions (no inference):

```bash
sleap-nn predict -i labels.slp --tracking
```

Note: Omit `--model_paths` for track-only mode.

With specific frames:

```bash
sleap-nn predict -i labels.slp -t --frames 0-100 --video_index 0
```

---

## Limit Instances

```bash
# Maximum 5 instances per frame
sleap-nn predict -i video.mp4 -m models/ --max_instances 5
```

---

## Example Configurations

### Fast Animals

```bash
sleap-nn predict -i video.mp4 -m models/ \
    -t \
    --use_flow \
    --of_img_scale 0.5
```

### Crowded Scenes

```bash
sleap-nn predict -i video.mp4 -m models/ \
    -t \
    --candidates_method local_queues \
    --tracking_window_size 10 \
    --max_tracks 10
```

### High Accuracy

```bash
sleap-nn predict -i video.mp4 -m models/ \
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
