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
| `--features` | Features for matching | `keypoints`, `centroids`, `bboxes`, `masks`, `embeddings` | `keypoints` |
| `--scoring_method` | Similarity scoring method | `oks`, `cosine_sim`, `iou`, `mask_iou`, `euclidean_dist` | `oks` |
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

### Appearance (embedding / re-ID)

Track by **appearance**: each detection carries a learned appearance vector (from an
[`embedding`](embedding-tracking.md) re-ID model) and candidates are matched by
**cosine similarity** between vectors — the track follows *what the animal looks
like*, not where it is.

Appearance is a **complementary** cue, not a better one. On dense continuous video,
geometry is highly informative and appearance-*only* association measurably loses to
it (227 ID switches against 32 on a held-out session); appearance wins where geometry
has no signal — temporally sparse frames, long occlusions, cross-session identity.
Blended into a geometric score with
[`--appearance_weight`](embedding-tracking.md#blending-appearance-with-geometry) it
beats either cue alone. Read the [embedding tracking
guide](embedding-tracking.md#which-regime-you-want) before choosing.

```bash
# Detections already carry embeddings (saved by a prior embedding-model run):
sleap-nn predict -i embedded.slp -t --features embeddings
```

`--features embeddings` auto-pairs with `--scoring_method cosine_sim`. It works on
both pose (`PredictedInstance`) and segmentation-mask (`PredictedSegmentationMask`)
detections, and is image-free (no `--use_flow` / `--use_kalman`). `local_queues`
(a per-track gallery of recent vectors) is a good pairing.

**Best for**: re-identification across long occlusions, temporally sparse frames, and
multi-session identity — the regimes where geometry has nothing to go on. On dense
continuous video prefer a geometric `--features` with `--appearance_weight`. See the
[embedding (re-ID) tracking guide](embedding-tracking.md) for the full workflow,
including running the embedding model and tracking in one command.

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

## Evaluating Identity Persistence

`sleap-nn eval` scores detection and localization -- whether the animal was found,
and where. It says nothing about whether a track kept the **right** animal.
`sleap-nn eval-tracking` scores that:

```bash
sleap-nn eval-tracking -g ground_truth.slp -p tracked_predictions.slp
```

Both files must be tracked: the ground truth needs `track` set on the detections
you want scored, and the prediction needs tracks from `sleap-nn track` or
`sleap-nn predict -t`. An untracked prediction is skipped with a message rather
than scored as a failure.

Detections are matched to ground truth within each frame first (OKS for poses,
mask IoU for segmentation masks), and identity is scored over those matches --
so a tracker is never penalized for the detector's misses.

| metric | what it measures | better |
|---|---|---|
| `id_switches` | Times a ground-truth trajectory changed which predicted track it matched (CLEAR-MOT). A change across a gap counts; the gap itself does not. | lower |
| `idf1` / `idp` / `idr` | Identity F1 after a global best assignment of predicted identities to ground-truth ones (Ristani et al.). | higher |
| `mostly_tracked` / `partly_tracked` / `mostly_lost` | Ground-truth trajectories bucketed by how much of them was matched at all. | MT higher |
| `fragmentations` | Interruptions of a trajectory that later resumes. A trajectory that simply ends is not counted. | lower |
| `mean_track_purity` | Per predicted track, the share of its matched detections belonging to its dominant ground-truth identity, length-weighted. | higher |
| `mean_gt_coverage` | Mean share of each ground-truth trajectory's frames that matched. | higher |

The coverage and purity columns are there on purpose: without them a tracker can
"win" on ID switches by emitting fewer, shorter, more timid tracks. Read them
together.

Detection counts (`n_gt_dets`, `n_pred_dets`, `n_matched`, `n_pred_untracked`)
are reported alongside but never folded into the identity scores -- which is why
MOTA is deliberately absent. MOTA mixes detection false positives and misses
into one number, so when two trackers are compared over the same detections a
MOTA delta mostly reports detector noise.

### What carries identity

`--carrier` selects what is matched and scored:

- `pose` -- instances, matched by OKS. The default for pose models.
- `mask` -- `PredictedSegmentationMask`es, matched by mask IoU, for segmentation
  models. Scale-aware, so stride-resolution predicted masks and full-resolution
  ground-truth masks are compared on the same pixel grid.
- `auto` (default) -- picks `mask` when the prediction carries masks but no
  instances, else `pose`.

`--match_threshold` is the minimum OKS or IoU for a pair to count as matched
(default `0.5`), and `--mt_threshold` / `--ml_threshold` are the coverage cuts
for MT and ML.

!!! note "`--user_labels_only` is OFF here, unlike `sleap-nn eval`"
    Tracked ground truth usually *is* predicted: the normal workflow predicts
    poses and then assigns or corrects tracks over them in the GUI. Filtering
    the ground-truth side by detection type would therefore discard it -- on the
    re-ID benchmark's own ground-truth sessions, turning the filter on takes
    2465 scored detections to 0.

    So everything carrying a track on the ground-truth side is scored, and the
    count of predicted ground-truth detections is reported in `notes`. Pass
    `--user_labels_only` in the one case it helps: user-labeled ground truth in
    a file that *also* holds stale predictions from an earlier run, which would
    otherwise be scored as extra trajectories.

### Score a video clip, not a training split

!!! warning "Identity metrics on a sparse label file are meaningless"
    An embedded `.pkg.slp` training split renumbers its frames `0..N-1`, so
    `frame_idx` and `frame_numbers` both read as contiguous while the animal has
    actually moved across the arena between two "consecutive" frames. Every
    index-based check passes and the metrics come out looking authoritative.

    `eval-tracking` measures this and warns. The check is exposed directly:

    ```python
    from sleap_nn.evaluation import motion_diagnostic
    import sleap_io as sio

    motion_diagnostic(sio.load_slp("labels.slp"), "pose")
    # continuous video:     {'step_over_size': 0.06, 'is_continuous': True, ...}
    # sparse training split: {'step_over_size': 8.74, 'is_continuous': False, ...}
    ```

    `step_over_size` is how far the same animal moves between consecutive frames
    relative to its own body size. At the high end, consecutive detections of one
    animal do not even overlap, so geometric association has no signal to work
    with and any IoU tracker must fail -- for reasons that have nothing to do
    with the tracker.

### Python API

```python
import sleap_io as sio
from sleap_nn.evaluation import identity_metrics

gt = sio.load_slp("ground_truth.slp")
pred = sio.load_slp("tracked_predictions.slp")

metrics = identity_metrics(gt, pred, "pose")
print(metrics.summary())
# IDSW=32  IDF1=0.8491 (P=0.8491 R=0.8491)  MT/PT/ML=5/0/0  Frag=0 ...

metrics.as_dict()["id_switches"]  # 32
```

Comparing tracker settings is the common case, so there is a table renderer for it:

```python
from sleap_nn.evaluation import compare_identity_metrics

print(compare_identity_metrics({
    "fixed_window": identity_metrics(gt, pred_fixed, "pose"),
    "flow": identity_metrics(gt, pred_flow, "pose"),
    "kalman": identity_metrics(gt, pred_kalman, "pose"),
}))
```

Always compare arms over the **same** detections -- retrack one prediction file
with different tracker settings rather than re-running inference -- so the
difference you read is the tracker's and not the detector's.

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
