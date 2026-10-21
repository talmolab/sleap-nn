# Tracking

Assign consistent IDs to instances across frames.

---

## Enable Tracking

Add `--tracking` to your inference command:

```bash
sleap-nn predict -i video.mp4 -m models/bottomup/ --tracking
```

`--tracker` chooses the tracking algorithm. The default, `classic`, is the
candidate-window tracker described in the next sections. The modern
multi-object trackers -- `sort`, `deepsort`, `strongsort`, `bytetrack`, `botsort`,
`ocsort` and `deepocsort` -- are covered in [Modern MOT
trackers](#modern-mot-trackers). Each family has its own options. Passing an
option that belongs to the other family is an error, so it is never silently
ignored.

---

## Tracking Parameters

These options belong to the `classic` tracker (`--tracker classic`, the default).

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--tracking` / `-t` | Enable tracking | Flag | `False` |
| `--tracker` | Tracking algorithm ([MOT trackers](#modern-mot-trackers)) | `classic`, `sort`, `deepsort`, `strongsort`, `bytetrack`, `botsort`, `ocsort`, `deepocsort` | `classic` |
| `--tracking_window_size` | Frames to look back | `INT` | `5` |
| `--min_new_track_points` | Min points for new track | `INT` | `0` |
| `--candidates_method` | Candidate selection method | `fixed_window`, `local_queues` | `fixed_window` (`local_queues` for `--features embeddings` and mask tracking) |
| `--min_match_points` | Min non-NaN points for matching | `INT` | `0` |
| `--features` | Features for matching | `keypoints`, `centroids`, `bboxes`, `masks`, `embeddings` | `keypoints` |
| `--scoring_method` | Similarity scoring method | `oks`, `cosine_sim`, `iou`, `mask_iou`, `euclidean_dist` | `oks` |
| `--appearance_weight` | Blend appearance (re-ID vector cosine) into the geometric score: `(1 - w) * geometry + w * appearance`. `0.15-0.5` is the measured range; needs embeddings in the input ([details](embedding-tracking.md#blending-appearance-with-geometry)) | `FLOAT` in `[0, 1]` | `0.0` (geometry only) |
| `--euclidean_scale` | Pixel length scale mapping an `euclidean_dist` score to a bounded similarity, `exp(-distance / scale)`. Required with `--appearance_weight` when the score is `euclidean_dist` (centroid-only detections auto-select it); ignored otherwise | `FLOAT` (px) | `None` |
| `--scoring_reduction` | Score reduction method | `mean`, `max`, `robust_quantile` | `mean` |
| `--track_matching_method` | Assignment algorithm | `hungarian`, `greedy` | `hungarian` |
| `--max_tracks` | Maximum track count (auto-selects `local_queues`). With a MOT tracker: the known number of animals ([closed set](#identity-linking-long-term-identity-memory)) | `INT` | `None` |
| `--use_flow` | Enable optical flow | Flag | `False` |
| `--use_kalman` | Enable Kalman-filter tracking | Flag | `False` |
| `--kf_init_frame_count` | Warm-up frames before EM init | `INT` | `10` |
| `--kf_node_indices` | Node indices to filter (comma-sep; empty = all) | e.g. `0,1,2` | `None` |
| `--kf_reset_gap_size` | Missed frames before a stale track resets | `INT` | `5` |
| `--link_tracklets` | After tracking each video, run [identity linking](#identity-linking-long-term-identity-memory). Works with every tracker | Flag | `False` |

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
has no signal — temporally sparse frames and long occlusions.
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
detections, and is image-free (no `--use_flow` / `--use_kalman`). It defaults to
`--candidates_method local_queues` (a per-track gallery of recent vectors, which
re-binds an animal that was absent for longer than the window); pass
`--candidates_method fixed_window` to opt out.

**Best for**: re-identification across long occlusions and temporally sparse frames —
the regimes where geometry has nothing to go on. (It does not link animals across
videos or sessions: each video is tracked on its own, see
[Track-Only Mode](#track-only-mode).) On dense
continuous video prefer a geometric `--features` with `--appearance_weight`. See the
[embedding (re-ID) tracking guide](embedding-tracking.md) for the full workflow,
including running the embedding model and tracking in one command.

---

## Modern MOT trackers

Besides `classic`, `--tracker` accepts seven standard tracking-by-detection
("MOT") algorithms. Each is a native numpy/scipy implementation, adapted to pose
instances and segmentation masks:

```bash
sleap-nn predict -i video.mp4 -m models/centroid/ -m models/centered_instance/ \
    -t --tracker bytetrack
```

All seven share these properties, and differ from `classic` in the same ways:

- **A Kalman motion model per track.** Each track predicts where its animal will
  be on the next frame, and detections are matched against that prediction rather
  than against the last pose. Each algorithm uses the filter it was published with
  (see the table below).
- **A track life cycle.** A new track stays *tentative* until it has been matched
  `--min_hits` times. Once confirmed, a track that misses a frame becomes *lost*. A
  lost track remains available for re-association for `--track_buffer` steps and
  is then removed.
- **No detection is dropped.** Every input detection comes back, in the same
  order. If no confirmed track claims a detection, its `track` is `None`. That
  covers low-score detections and detections of a track that never reached
  `--min_hits`. When a track is confirmed, the detections from its tentative
  frames are labeled retroactively. Unlike `classic`, these trackers never cull
  instances.
- **No images.** Association uses only the detections, which assumes a fixed
  camera (there is no camera-motion compensation). `--use_flow` and `--use_kalman`
  therefore do not apply.
- **Per-video calibration** runs before tracking. It measures body size, motion
  and, when re-ID vectors are present, the appearance thresholds ([see
  below](#per-video-calibration-of-appearance-thresholds)).
- **Optional identity linking** runs after each video (`--link_tracklets`,
  `--max_tracks`). It provides long-term identity memory across gaps and
  crossings.
- **The same output contract as `classic`.** Tracks are named `track_0`,
  `track_1`, … within each video, and the numbering continues across videos. A
  track never spans two videos. `tracking_score` is the geometric similarity of
  the matched pair, and 1.0 on the frame where a track starts.

### Choosing a tracker

| `--tracker` | Motion model (Kalman state) | Appearance | Key idea | Reference |
|---|---|---|---|---|
| `sort` | box center, area, aspect ratio | no | One Hungarian assignment per frame on the similarity between detections and each track's prediction. The baseline. | [Bewley et al., 2016](https://arxiv.org/abs/1602.00763) |
| `deepsort` | box center, aspect ratio, height | **required** | A matching cascade that tries recently seen tracks first. It matches on the best cosine against each track's gallery of re-ID vectors, inside a Mahalanobis motion gate. A geometric stage then handles new and just-lost tracks. | [Wojke et al., 2017](https://arxiv.org/abs/1703.07402) |
| `strongsort` | box center, aspect ratio, height; measurement noise scaled by the detection score (NSA) | **required** | DeepSORT with an EMA appearance feature, a single global assignment, and a motion term blended into the cost. | [Du et al., 2023](https://arxiv.org/abs/2202.13514) |
| `bytetrack` | box center, aspect ratio, height | no | Associates every detection. High-score detections are matched first; low-score detections are then matched to the tracks left over. An occluded animal's score drops, and the second stage keeps it on its track. | [Zhang et al., 2022](https://arxiv.org/abs/2110.06864) |
| `botsort` | box center, width, height | yes | ByteTrack plus re-ID. Appearance can only support a match that is already inside the geometric gate, and only where it tells the candidates apart. | [Aharon et al., 2022](https://arxiv.org/abs/2206.14651) |
| `ocsort` | box center, area, aspect ratio | no | Observation-centric SORT, built for non-linear motion and gaps. It re-updates a re-found track along a virtual path across the gap (ORU), rewards detections that continue the track's direction (OCM), and recovers tracks from their last observation (OCR). | [Cao et al., 2023](https://arxiv.org/abs/2203.14360) |
| `deepocsort` | box center, width, height | yes | OC-SORT with a score-adaptive appearance EMA and an appearance term weighted by how discriminative it is. It can also accept a match with weak geometry when appearance is decisive (rescue). | [Maggiolino et al., 2023](https://arxiv.org/abs/2302.11813) |

The four appearance trackers read the re-ID vector that an
[`embedding` model](embedding-tracking.md#appearance-aware-mot-trackers) attaches to
each detection. They refuse to run when no detection carries one. `deepsort` and
`strongsort` use appearance as their primary cue. `botsort` and `deepocsort` use
it only to support geometry. When the video's calibration finds that appearance
does not tell the animals apart, all four fall back to geometry alone and log a
warning.

Rules of thumb (score the result with
[`sleap-nn eval-tracking`](#evaluating-identity-persistence)):

- **Without an embedding model**, use `bytetrack` or `ocsort`. `bytetrack` is
  designed to keep partly occluded, low-score animals on their tracks. `ocsort` is
  designed for abrupt turns, stops and short gaps. `sort` is the reference
  baseline.
- **With an embedding model**, start with `botsort` or `deepocsort`. Appearance
  stays inside the geometric gate there, so a weak embedder cannot cause jumps
  across the arena. Choose `deepsort` or `strongsort` when appearance is the more
  reliable cue.
- **When the number of animals is known**, add `--max_tracks N`. **When animals
  leave the frame or are occluded for a long time**, add `--link_tracklets`.
  Either option works with any tracker.

### Options

Every MOT option defaults to *unset*, which means the tracker's
[preset](#presets) value. The defaults shown are those of the `animal` preset.

| Flag | What it sets | Default |
|---|---|---|
| `--tracker` | `classic`, `sort`, `deepsort`, `strongsort`, `bytetrack`, `botsort`, `ocsort`, `deepocsort` | `classic` |
| `--track_buffer INT` | How many steps a lost track is kept for re-association before removal (`max_age`). Identity linking also uses it as the gap that splits a track into tracklets. | `30` |
| `--min_hits INT` | Matches needed to confirm a new track; the track's tentative detections are labeled on confirmation. For `sort`, `ocsort` and `deepocsort` the matches must be consecutive and the detection that starts the track does not count, so `3` means four detections in a row. | `3`; `2` for `bytetrack` / `botsort` |
| `--track_high_threshold FLOAT` | Detection score above which a detection is high-confidence (the first stage of ByteTrack, BoT-SORT and OC-SORT). Also the minimum score for a re-ID vector to update a track's appearance memory or join a linking gallery. | per tracker |
| `--track_low_threshold FLOAT` | Detections scoring at or below this are ignored. | `0.1` |
| `--new_track_threshold FLOAT` | Minimum score for an unmatched detection to start a track. | per tracker |
| `--min_match_similarity FLOAT` | Minimum geometric similarity for a first-stage match (the geometric stage for `deepsort` / `strongsort`). | `0.3`; `0.2` (score-fused) for `bytetrack` / `botsort` |
| `--appearance_threshold {auto,FLOAT}` | Appearance trackers only. The minimum cosine similarity for appearance to support a match, or `auto` to [calibrate it per video](#per-video-calibration-of-appearance-thresholds). | `auto` |
| `--link_tracklets` | Run [identity linking](#identity-linking-long-term-identity-memory) after tracking each video. Also works with `--tracker classic`. | off |
| `--max_tracks INT` | The known number of animals. Enables closed-set identity linking (implies `--link_tracklets`), which leaves at most N tracks per video. | unset |
| `--mot_param KEY=VALUE` | Override any [MOT parameter](#mot-parameter-overrides). Repeatable, and accepts `preset=paper`. | — |
| `--scoring_method` | Geometric [similarity](#geometric-similarity): `iou`, `oks`, or `euclidean_dist` (centroid). | `auto` |
| `--oks_stddev FLOAT` | Per-keypoint spread used by `--scoring_method oks`. | `0.1` |
| `--features` | Carrier to track: `keypoints` (pose instances) or `masks`. Left unset, the masks are tracked when the file has masks but no predicted poses. Appearance trackers track whichever carrier holds more re-ID vectors. | auto |

The following combinations are refused with an error rather than silently
ignored. Checks that do not need the labels run before any inference:

- MOT-only flags (`--track_buffer`, `--min_hits`, the three score thresholds,
  `--min_match_similarity`, `--appearance_threshold`, `--mot_param`) with
  `--tracker classic`.
- Classic-only flags given with a MOT tracker. These are
  `--tracking_window_size`, `--candidates_method`, `--min_new_track_points`,
  `--min_match_points`, `--scoring_reduction`, `--robust_best_instance`,
  `--track_matching_method`, `--appearance_weight`, `--euclidean_scale`,
  `--use_flow` / `--of_*`, `--use_kalman` / `--kf_*`,
  `--tracking_target_instance_count`, `--tracking_pre_cull_*`,
  `--tracking_clean_*` and `--post_connect_single_breaks`. Identity linking takes
  over the job of the cull / clean / connect cleanups.
- `--tracker` other than `classic`, or `--link_tracklets`, without `-t`.
- `--appearance_threshold` with a geometry-only tracker (`sort`, `bytetrack`,
  `ocsort`).
- `--features embeddings` / `centroids` / `bboxes`, or `--scoring_method
  mask_iou` / `cosine_sim`, with a MOT tracker.
- An appearance tracker when the detections cannot carry re-ID vectors: the input
  is a video, or detection models are given, and no `embedding` model is among
  `-m`. The same applies to labels on which no detection carries a vector.
  `deepsort` and `strongsort` also refuse vectors that ride on the other carrier;
  `botsort` and `deepocsort` warn and run on geometry alone.
- The same parameter given both as a flag and in `--mot_param`.

### Presets

Unset options take values from a per-tracker **preset**. `animal`, the default,
holds sleap-nn's settings for animal pose and mask data. `paper` holds the
constants published with each algorithm, adapted only where pose or mask geometry
requires it. Use `paper` to compare against the literature:

```bash
sleap-nn predict -i predictions.slp -t --tracker deepsort --mot_param preset=paper
```

The `animal` preset:

| tracker | Kalman state | `min_hits` | high / low / new score thresholds | first-stage min similarity | appearance | other |
|---|---|---|---|---|---|---|
| `sort` | `xysr` | 3 (consecutive) | 0.1 / 0.1 / 0.1 | 0.3 | — | |
| `deepsort` | `xyah` | 3 | 0.5 / 0.1 / 0.1 | 0.3 (geometric stage) | `auto` (gallery gate) | gallery budget 30 |
| `strongsort` | `xyah` + NSA | 3 | 0.5 / 0.1 / 0.1 | 0.3 (geometric stage) | `auto` (EMA gate) | EMA α 0.8, motion weight λ 0.98 |
| `bytetrack` | `xyah` | 2 | 0.5 / 0.1 / 0.6 | 0.2, score-fused | — | low-score stage 0.5, unconfirmed 0.3, duplicates 0.85 |
| `botsort` | `xywh` | 2 | 0.6 / 0.1 / 0.7 | 0.2, score-fused | `auto` (EMA gate) | proximity 0.5, EMA α 0.8 |
| `ocsort` | `xysr` | 3 (consecutive) | 0.6 / 0.1 / 0.6 | 0.3 | — | `delta_t` 3, inertia 0.2, BYTE stage off |
| `deepocsort` | `xywh` | 3 (consecutive) | 0.6 / 0.1 / 0.6 | 0.3 | `auto` (EMA gate) | adaptive EMA α 0.95, appearance weight 0.75, adaptive weighting 0.5, rescue on (min similarity 0.15) |

Every tracker also starts from `track_buffer` 30, similarity `auto`,
`oks_stddev` 0.1, box padding 0.1 and score mode `auto`.

`paper` differs from `animal` in the following ways. `bytetrack` and `ocsort` are
the same in both.

- `sort`: all score thresholds are 0. SORT has no score logic of its own, so every
  detection with a positive score is kept.
- `deepsort`: every score threshold is 0.8 (the reference `min_confidence`), the
  appearance threshold is a fixed 0.8 (cosine distance 0.2), and the gallery
  budget is 100.
- `strongsort`: every score threshold is 0.6, the appearance threshold is 0.7,
  and EMA α is 0.9.
- `botsort`: the appearance threshold is 0.5 and EMA α is 0.9. No appearance
  margin is required.
- `deepocsort`: rescue is off.

!!! note "Paper thresholds were tuned for pedestrian detectors"
    SLEAP detection scores (see [below](#detection-scores)) usually sit well below
    a pedestrian detector's, and the published re-ID cosines assume a person re-ID
    embedder. For example, `deepsort` under `paper` ignores every detection that
    scores 0.8 or less. Use `paper` for comparisons, and `animal` (or your own
    overrides) for real data.

### MOT parameter overrides

Every field of `sleap_nn.tracking.mot.params.MOTParams` can be overridden with
`--mot_param KEY=VALUE`, repeated as needed. Booleans take `true`/`false`, and a
misspelled key is an error that lists the valid ones. Prefer the dedicated flag
where one exists. These are the most useful parameters that have no flag of their
own:

| Key | Default (`animal`) | Used by | Meaning |
|---|---|---|---|
| `preset` | `animal` | all | `animal` or `paper`. |
| `similarity` | `auto` | all | `auto`, `iou`, `oks` or `centroid`: the same choice as `--scoring_method`, under the MOT names. |
| `score_mode` | `auto` | all | Detection score: `auto` (mean point score), `instance` or `none` (all 1.0). See [Detection scores](#detection-scores). |
| `box_padding` | `0.1` | all | Padding added to each side of a keypoint or mask box, as a fraction of `max(w, h)`. |
| `kalman` | `auto` | all | Replace the tracker's own filter with `xywh`, `xyah` or `xysr`. |
| `time_base` | `auto` | all | `frames` or `steps`. See [Time base](#time-base). |
| `fuse_score` | `true` for `bytetrack` / `botsort` | ByteTrack, BoT-SORT | Multiply the similarity by the detection score in the first and unconfirmed stages. |
| `low_stage_min_similarity` | `0.5` | ByteTrack, BoT-SORT | Minimum similarity in the low-score stage. |
| `unconfirmed_min_similarity` | `0.3` | ByteTrack, BoT-SORT | Minimum similarity in the tentative-track stage. |
| `duplicate_similarity` | `0.85` | ByteTrack, BoT-SORT | Similarity above which a tracked track and a lost track count as duplicates. |
| `proximity_threshold` | `0.5` | BoT-SORT | Minimum geometric similarity before appearance is considered at all. |
| `ema_alpha` | `0.8` (`deepocsort` 0.95) | appearance trackers | Weight on the previous appearance feature in the per-track EMA. |
| `gallery_size` | `30` | appearance trackers, calibration | Recent quality-gated vectors kept per track. |
| `gallery_budget` | `30` | DeepSORT | Gallery size for DeepSORT's cascade (`nn_budget`). |
| `mc_lambda` | `0.98` | StrongSORT | Weight of appearance in the cost; motion gets `1 - λ`. |
| `delta_t`, `inertia` | `3`, `0.2` | OC-SORT, Deep OC-SORT | Look-back (in steps) and weight of the direction-consistency term. |
| `use_byte` | `false` | OC-SORT | Add ByteTrack's low-score stage. |
| `w_assoc_emb`, `aw_param`, `aw_off` | `0.75`, `0.5`, `false` | Deep OC-SORT | Appearance weight, adaptive-weighting cap, and switch to turn adaptive weighting off. |
| `rescue`, `rescue_min_similarity` | `true`, `0.15` | Deep OC-SORT | Accept a weak-geometry match (similarity at least 0.15) when appearance is decisive. |
| `link_swap_correction` | `true` | identity linking | Correct identity swaps at crossings. |
| `link_spatial_gate_frames` | `300` | identity linking | Gaps longer than this many steps are not gated by distance. |

```bash
sleap-nn predict -i predictions.slp -t --tracker ocsort \
    --mot_param use_byte=true --mot_param delta_t=5 --mot_param score_mode=none
```

### Detection scores

The score thresholds compare against one confidence per detection, chosen by
`score_mode`:

- `auto` (default). A predicted pose scores the **mean confidence of its visible
  keypoints**; if none of them has a finite score, the instance score is used. A
  predicted mask scores its `score`. A user-labeled `Instance` scores 1.0.
- `instance`. The detection's own `score` (1.0 if it has none).
- `none`. Every detection scores 1.0, which turns the score logic off. Use it for
  curated detections, such as ground-truth poses saved as predictions.

Every score is clipped to `[0, 1]`, and non-finite scores become 1.0. `auto` uses
the mean point score because SLEAP instance scores are not on a common scale.
Bottom-up and single-instance instance scores are sums over edges or nodes, so
they are not limited to `[0, 1]`. The mean keypoint confidence lies in `[0, 1]` for every pose
model.

### Geometric similarity

Every MOT threshold is a **similarity in `[0, 1]`**, where higher is stricter.
Each tracker converts it to the cost its algorithm minimizes. `--scoring_method`
selects the similarity:

| `--scoring_method` | similarity | between |
|---|---|---|
| unset (`auto`) | `iou` for masks; `centroid` for single-node skeletons; `iou` otherwise | |
| `iou` | IoU of padded boxes | the track's predicted box and the detection's box (the extent of the visible keypoints, or the mask's bounding box, padded by `box_padding`). On masks this is **bounding-box** IoU, not pixel IoU. |
| `oks` | OKS with spread `--oks_stddev` (default 0.1) | the detection's keypoints and the track's predicted pose: its last pose shifted by the Kalman-predicted displacement. Requires poses with at least 2 nodes; on masks it falls back to `iou`, and on single-node skeletons to `centroid`, with a warning. |
| `euclidean_dist` | centroid: `exp(-d² / (2 s²))` with `s = body_size / 2` | the predicted box center and the detection's box center. `body_size` is calibrated per video. |

The MOT default spread of 0.1 is 4× the evaluation default. Tracking compares two
noisy observations one frame apart, and at 0.025 an OKS threshold is several
times stricter than the same IoU threshold. The `oks` and `centroid` scales are
chosen so that a threshold means roughly the same displacement under all three
similarities.

!!! note "Choosing the similarity from Python"
    For a MOT tracker, `TrackerConfig.scoring_method` sets the similarity only when
    it differs from its default value or `scoring_method_explicit=True` is passed.
    `TrackerConfig(tracker="bytetrack", scoring_method="oks")` therefore still runs
    `auto`. In Python, set the similarity with `mot_params={"similarity": "oks"}`
    instead. The CLI does not have this problem: an explicit `--scoring_method oks`
    is honored.

### Time base

Kalman prediction, `--track_buffer` and OC-SORT's `delta_t` all count
**steps**. Before tracking, each video's frame indices are examined:

- **Regularly spaced frames**, meaning at least half of the gaps equal the most
  common frame step `k`, use the `frames` time base. One step is `k` frames: a
  video predicted every 5th frame advances one step per processed frame, and a
  genuine gap of 50 frames counts as 10 steps.
- **Irregularly spaced frames**, such as a `--frames` list or a retrack of
  scattered labeled frames, use the `steps` time base: one step per processed
  frame. A warning is logged.

Use `--mot_param time_base=frames` or `time_base=steps` to force either.

### Identity linking (long-term identity memory)

Online trackers forget an animal after `--track_buffer` steps, so an animal that
leaves the frame, hides, or crosses another can come back under a new name.
Identity linking runs **once per video after tracking**, with the whole video in
view. Every detection's vector already exists at that point, so it can compare
complete tracklets on both sides of each gap.

```bash
# Any tracker, classic included:
sleap-nn predict -i predictions.slp -t --tracker ocsort --link_tracklets
# Known number of animals: closed set, at most 4 tracks per video:
sleap-nn predict -i predictions.slp -t --tracker bytetrack --max_tracks 4
```

Linking treats each track as an **identity** made of **tracklets**, which are runs
of the identity's detections separated by gaps of at most `--track_buffer` steps.
It then runs three passes:

1. **Swap correction at crossings.** This pass needs appearance that is
   calibrated for linking. Around each crossing, where the padded boxes of up to
   four identities overlap, it compares each identity's gallery from before the
   crossing with the vectors right after it. It applies a relabeling only when
   appearance prefers the relabeling by the calibrated margin.
2. **Gap linking (re-identification).** An identity that ends is joined to one
   that starts later. The two must never appear in the same frame, and the second
   must start within a distance gate that grows with the gap:
   `body + 3 · step_p95 · (gap if gap ≤ 10 else √(10 · gap))`, with no gate beyond
   `link_spatial_gate_frames` steps. With calibrated appearance, the pair's
   tracklet-to-gallery similarity must clear the calibrated threshold and beat
   every competing candidate by the calibrated margin. Without calibrated
   appearance, a link is made only when the gap is at most 4 × `--track_buffer`
   steps and the two identities are each other's only feasible candidate. The
   best links are applied first. A merged identity keeps the earlier track's
   name.
3. **Closed set** (`--max_tracks N`). While more than N identities remain, the
   best feasible merge is forced. Candidates include an identity that fits inside
   another's gap, ranked by appearance agreement and then by distance. Identities
   that appear alongside every other one, so that they cannot be merged, are
   *untracked* (`track=None`). Duplicate identities go first: those with at least
   half of their frames overlapping another identity at IoU ≥ 0.5. After them,
   the identity with the fewest detections goes. Detections are never deleted.

Only quality-gated vectors enter a gallery. A gated vector comes from a detection
that scores at least `--track_high_threshold`, has at least half of its nodes
visible, and is isolated: its padded-box IoU with every other detection in the
frame is below 0.1. Crops of overlapping animals are contaminated. Linking logs a
one-line summary whenever it changes something, and the counts go to the
[provenance record](#provenance-record).

With `--tracker classic`, `--link_tracklets` links each video after it is tracked,
using the MOT defaults. A `--max_tracks` given there both caps the classic tracker
and closes the set. With a MOT tracker, `--max_tracks` always turns linking on. A
MOT tracker never enforces the count by refusing to start new tracks.

Linking works **within a video**. It does not link one animal across the videos
or sessions of a multi-video file.

### Per-video calibration of appearance thresholds

The re-ID cosines that separate two animals depend on the embedding model, the
species and the recording. A constant tuned for pedestrians does not transfer. So
before tracking each video, `calibrate` measures the thresholds from that video's
own detections, with no ground truth:

1. **Geometry.** Calibration measures body size (the median keypoint-box
   diagonal), nearest-neighbor spacing, the frame step, score quantiles, and
   per-step motion (`step_p95` / `step_p99`). Motion is measured along *anchor
   tracklets*: detections chained frame to frame only where the match is
   unambiguous. A link needs mutual nearest neighbors closer than half the
   typical spacing, with the runner-up at least twice as far and no duplicate
   detections nearby. Anchors are kept when at least 10 detections long.
2. **Appearance trials.** Two anchors that overlap in time are certainly
   different animals, so they supply negatives. Later vectors of the same anchor
   supply positives. Over up to 2000 trials per gap, calibration measures the
   **exact statistics the trackers threshold**: the EMA cosine (BoT-SORT,
   StrongSORT, Deep OC-SORT), the gallery-max cosine (DeepSORT), and the
   tracklet-to-gallery top-5 mean (linking). These are measured one step after
   the gallery and `--track_buffer` steps after it.

The calibration reports these values. Under `--appearance_threshold auto`, they
become the trackers' and the linker's thresholds:

| field | meaning |
|---|---|
| `assoc_rank1` | Fraction of one-step trials in which the true track beats every co-occurring animal. |
| `ema_cos_gate` / `gallery_cos_gate` | The 95th percentile of competitor cosines: a match below this cosine gets no appearance support. |
| `m_assoc` | The minimum lead over the next-best candidate before appearance may decide a pair (at least 0.02). |
| `link_thresh` / `link_margin` / `link_recall` | The linking threshold (99th percentile of competitors), the smallest margin that reaches 99% precision on the trials, and the recall at that margin. |
| `neg_collision_rate` | Fraction of same-frame pairs of different animals that look as similar as the same animal one step apart. |

**When appearance is turned off.** Appearance is used for association only when
the calibration `status` is `ok` and `assoc_rank1 ≥ 0.95`. It is used for linking
only when a `link_margin` reaches 99% precision with recall ≥ 0.2. The `status` is
`no_vectors` when there are no vectors. It is `insufficient` with fewer than 100
one-step trials that have a co-occurring competitor, for example a single animal,
or animals that are never well separated at the same time. In each of these cases
`auto` **turns appearance off** rather than falling back to a published constant.
The appearance trackers then associate on geometry alone and log a warning.
Linking uses only its spatial rules, as the `MOT calibration` log line reports.
An explicit number, such as `--appearance_threshold 0.6`, skips the
`assoc_rank1` check and is used as the minimum association cosine for every
statistic. Linking always uses its calibrated thresholds.

!!! warning "\"The embedding may encode class/sex rather than identity\""
    Calibration logs this warning when more than 5% of same-frame pairs of
    *different* animals are at least as similar as the same animal one step apart.
    The embedder then separates groups of animals, such as sexes, strains or coat
    colors, rather than individuals. Appearance cannot tell those animals apart,
    whatever threshold you choose. Retrain or fine-tune the embedder on identity
    labels, or rely on geometry and `--max_tracks`.

Each video's calibration is logged as one `MOT calibration: ...` line and saved in
the provenance record.

### Provenance record

A MOT run, or `--link_tracklets` with `classic`, records what each video resolved
under `labels.provenance["tracking"]` in the output `.slp`. The record holds plain
JSON:

```python
{
    "tracker": "botsort",
    "carrier": "pose",
    "params": {...},            # every resolved MOTParams value (preset + overrides)
    "videos": [
        {
            "video": 0,
            "kalman": "xywh",
            "time_base": "frames",
            "frame_step": 1,
            "similarity": "iou",    # after `auto` resolution
            "calibration": {...},   # VideoCalibration.as_dict(): status, body_size, gates, ...
            "appearance": {"enabled": False, "reason": "calibration status='insufficient', ...", ...},
            "linking": {"n_identities_before": 5, "n_identities": 4, "n_merges": 1,
                        "n_forced_merges": 0, "n_swaps": 0, "n_untracked": 0, "notes": [...]},
        }
    ],
}
```

`linking` appears only when linking ran. A classic run with `--link_tracklets`
records `{"tracker": "classic", "carrier", "link_tracklets": true, "videos":
[{"video", "calibration", "linking"}]}`. The flags themselves are saved under
`provenance["tracking_config"]`, as for every tracking run. A plain classic run's
provenance is unchanged.

### Worked examples

Detect, embed and track a video in one command. Four animals; appearance-aware
BoT-SORT; identity linking with a closed set:

```bash
sleap-nn predict -i video.mp4 \
    -m models/centroid/ -m models/centered_instance/ -m models/embedding/ \
    -t --tracker botsort --link_tracklets --max_tracks 4 \
    -o tracked.slp
```

This is the fused route of the [embedding tracking
guide](embedding-tracking.md#workflow-3-detect-embed-and-track-a-video-fused). The
detection stack runs, every detection is embedded, and BoT-SORT tracks with the
vectors. Linking then joins tracks across gaps and keeps at most 4 per video.
`--link_tracklets` is implied by `--max_tracks` here and is spelled out only for
clarity. Add `--save_embeddings slp` to keep the vectors for later retracking.

Retrack an existing prediction file with ByteTrack. No model and no GPU are
needed:

```bash
sleap-nn predict -i predictions.slp -t --tracker bytetrack -o bytetrack.slp
```

Score the result against tracked ground truth. The legacy identity block and the
TrackEval-faithful `mot` block (CLEAR-MOT, IDF1, HOTA) are both printed and
saved:

```bash
sleap-nn eval-tracking -g ground_truth.slp -p bytetrack.slp -s bytetrack_metrics.json
```

To compare trackers, retrack the **same** prediction file once per tracker and
score each output. The difference is then the tracker's and not the detector's.

### Benchmark results

<!-- PLACEHOLDER: filled in from the benchmark run. -->

Results on GT-tracked animal clips: see PR description (to be filled in).

---

## Track-Only Mode

Assign tracks to existing predictions (no inference):

```bash
sleap-nn predict -i labels.slp --tracking
```

Note: Omit `--model_paths` for track-only mode.

Each video is tracked independently: a track never spans two videos, and track
names continue across videos (`track_0`, `track_1` in the first; `track_2`, … in the
next) so no two videos share one. Everything tracking does not assign -- masks
on a pose-tracked file, ROIs, centroids, suggestions -- is carried through to the
output unchanged.

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
`sleap-nn predict -t`. When there is nothing to score -- an untracked
prediction, no frame in common, or no tracked ground truth -- the command says
why and exits non-zero rather than printing an empty result.

Detections are matched to ground truth within each frame first (OKS for poses,
mask IoU for segmentation masks, pixel distance for single-node skeletons), and
identity is scored over those matches. What is scored:

- **Per video.** A trajectory is a `(video, track name)` pair: trackers name
  tracks per video, so the same name in two videos is two animals. Pass
  `--global_identity` (`global_identity=True` in Python) when the names are
  global identities (a multi-class ID model, or a project proofread with one
  name per animal) and keeping an animal's name from one video to the next is
  part of the score.
- **Every ground-truth frame** of each video the prediction covers. A frame the
  prediction does not have (dropped by `--no_empty_frames`, or never predicted)
  counts as misses, exactly like a present frame with nothing detected on it. A
  ground-truth video that shares no frame with the prediction is not scored,
  and is named in `notes`.
- **Untracked ground truth is don't-care.** It still takes part in matching,
  and a prediction matched to it is neither a hit nor a false positive, so
  ground truth that tracks only some of the animals does not turn correct
  predictions on the others into false positives.

| metric | what it measures | better |
|---|---|---|
| `id_switches` | Times a ground-truth trajectory changed which predicted track it matched (CLEAR-MOT). A change across a gap counts; the gap itself does not. | lower |
| `idf1` / `idp` / `idr` | Identity F1 after a global best assignment of predicted identities to ground-truth ones (Ristani et al.). **Includes detection errors**: every missed ground-truth detection is an IDFN, every unmatched prediction an IDFP. | higher |
| `idf1_matched` | IDF1 over matched detections only (IDTP / `n_matched`): the share of matched detections carrying their trajectory's identity, with the detector's misses and false positives factored out. | higher |
| `mostly_tracked` / `partly_tracked` / `mostly_lost` | Ground-truth trajectories bucketed by how much of them was matched at all. | MT higher |
| `fragmentations` | Interruptions of a trajectory that later resumes. A trajectory that simply ends is not counted. | lower |
| `mean_track_purity` | Per predicted track, the share of its matched detections belonging to its dominant ground-truth identity, length-weighted. | higher |
| `mean_gt_coverage` | Mean share of each ground-truth trajectory's frames that matched. | higher |

The coverage and purity columns are there on purpose: without them a tracker can
"win" on ID switches by emitting fewer, shorter, more timid tracks. Read them
together.

Detection counts (`n_gt_dets`, `n_pred_dets`, `n_matched`, `n_pred_untracked`,
`n_gt_untracked`, `n_dont_care`, `n_frames_missing_pred`) are reported alongside.
IDF1 is the standard definition, so a detector that misses animals lowers it
even under a perfect tracker (a detector missing 5 of 20 detections caps IDF1 at
0.857). The switch count, purity and `idf1_matched` are computed over matched
detections only and do not move with the detector: over **frozen** detections
-- the usual tracker A/B -- compare arms by those. MOTA is not in this table: it
mixes every detection false positive and miss into one number, so a MOTA delta
between two trackers mostly reports detector noise. It is reported, with the
other standard MOT metrics, in the `mot` block below.

### Standard MOT metrics (CLEAR-MOT, IDF1, HOTA)

The table above uses sleap-nn's own definitions. To put a number next to
published multi-object-tracking results, `eval-tracking` also computes the
standard metrics exactly as **TrackEval** (the reference HOTA / MOTChallenge
evaluation code) does: they are logged as "TrackEval-faithful MOT" and saved
under the `"mot"` key of the `-s` JSON. Same detections, carrier, matcher and
`--match_threshold`; no extra flags.

| `mot` field | what it measures | better |
|---|---|---|
| `mota` / `moda` | `(TP − FP − IDSW) / n_gt` / `(TP − FP) / n_gt`: detection errors and ID switches in one number. | higher |
| `motp` | Mean similarity (OKS / IoU) of the matched pairs. | higher |
| `idsw` / `frag` / `mt`, `pt`, `ml` | CLEAR-MOT ID switches, fragmentations, and trajectories tracked >80% / 20–80% / <20% of their frames. | lower / lower / MT higher |
| `idf1` / `idp` / `idr` | Identity F1 (Ristani et al.). | higher |
| `hota` | Higher-order tracking accuracy, `sqrt(det_a × ass_a)`, averaged over localization thresholds α = 0.05 … 0.95 (`hota_alpha` holds each one). Balances detection and association. | higher |
| `det_a` / `ass_a` / `loc_a` | HOTA's detection, association and localization accuracy. `ass_a` is the identity-keeping part. | higher |
| `clr_tp` / `clr_fn` / `clr_fp` | Matched pairs, missed ground truth, unmatched predictions. | — |

Where the two blocks name the same thing they can disagree, by design of the
legacy block: CLEAR-MOT keeps last frame's match when it still passes the
threshold (fewer switches), TrackEval's IDF1 counts every frame on which two
identities *could* match, "mostly tracked" is strictly above 80%, and a
fragment is a tracked segment minus one. Quote the `mot` numbers when
comparing with the literature, and say which block a number comes from.

How sleap-nn data maps onto TrackEval:

- **Each video is a sequence**, combined as TrackEval combines sequences
  (counts summed, ratios recomputed). With `--global_identity` all videos form
  one sequence keyed by track name; switch and fragment counting restart at
  each video boundary, so a name lost between videos costs IDF1 and `ass_a`,
  not `idsw`.
- **Untracked ground truth is don't-care**, as MOTChallenge treats
  distractors: a prediction matched to it is removed (`n_dont_care`), not a
  false positive. Untracked predictions are removed (`n_pred_untracked`). The
  [MOT trackers](#modern-mot-trackers) leave unconfirmed detections untracked,
  so those detections cost misses (`clr_fn`), not false positives.
- **Single-node skeletons** match by distance `d` with similarity
  `max(0, 1 − d / (2 · match_threshold))`, so the 0.5 cut is exactly
  `--match_threshold` pixels, and `motp` / `loc_a` are in that unit.
- **`--oks_stddev`** (default `0.025`) is the per-keypoint OKS spread used to
  match poses, in both blocks.
- A track on two detections of one frame is ambiguous; TrackEval refuses it, so
  the `mot` block then holds `{"error": ...}` naming the frame while the legacy
  block is still computed.

In Python, `compute_mot_metrics(gt, pred, "pose")` returns a `MOTMetrics`
(`summary()`, `as_dict()`), and `sleap_nn.tracking.mot_metrics` scores raw
per-frame id / similarity sequences directly.

### What carries identity

`--carrier` selects what is matched and scored:

- `pose` -- instances, matched by OKS. Single-node (centroid) skeletons, on
  either side, match by pixel distance between centroids instead: OKS
  normalizes by the pose's area, which is zero for one node, so it would never
  match a centroid that is off by even a pixel. A full-pose ground truth is
  reduced to the mean of its visible nodes for that comparison.
- `mask` -- `PredictedSegmentationMask`es, matched by mask IoU, for segmentation
  models. Scale-aware, so stride-resolution predicted masks and full-resolution
  ground-truth masks are compared on the same pixel grid.
- `auto` (default) -- the carrier the prediction's tracks are on: whichever holds
  more tracked detections (ties go to `pose`). A pose + mask file tracked on its
  masks is scored on its masks.

A detection with no track of its own takes the track of its linked detection on
the other carrier (`mask.instance`): ground truth proofread in the GUI carries its
tracks on the instances, while mask tracking writes them onto the masks, and
either can be scored against the other.

`--match_threshold` is the minimum OKS or IoU for a pair to count as matched
(default `0.5`), or for single-node skeletons the maximum centroid distance in
**pixels** (default `50`, as in `sleap-nn eval --match_method centroid`); a run
where nothing matched says so in `notes`. `--mt_threshold` / `--ml_threshold`
are the coverage cuts for MT and ML.

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

    `step_over_size` is how far the same animal moves per frame relative to its
    own body size. At the high end, consecutive detections of one animal do not
    even overlap, so geometric association has no signal to work with and any IoU
    tracker must fail -- for reasons that have nothing to do with the tracker.

    It follows each animal by its **track**, so it needs tracked labels, and it
    follows each video separately. A step between two detections `k` frames
    apart is divided by `k`, so ground truth labeled every 10th frame of a real
    video is judged by its per-frame motion rather than flagged as sparse, while
    a renumbered `.pkg.slp` split (every gap reads as 1) still is.
    `median_step_px` is therefore the typical per-frame displacement of one
    animal -- the value `--euclidean_scale` asks for.

### Python API

```python
import sleap_io as sio
from sleap_nn.evaluation import identity_metrics

gt = sio.load_slp("ground_truth.slp")
pred = sio.load_slp("tracked_predictions.slp")

metrics = identity_metrics(gt, pred, "pose")
print(metrics.summary())
# IDSW=32  IDF1=0.8491 (P=0.8491 R=0.8491; matched-only 0.8491)  MT/PT/ML=5/0/0 ...

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
difference you read is the tracker's and not the detector's. In Python,
`apply_tracking` / `Predictor.retrack` track the `Labels` you pass **in place**
(and return it), so give each arm its own copy:

```python
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

untracked = sio.load_slp("predictions.slp")
arms = {
    w: apply_tracking(untracked.copy(), TrackerConfig(appearance_weight=w))
    for w in (0.0, 0.3)
}
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

??? question "A MOT tracker leaves many detections untracked"
    A MOT tracker leaves a detection untracked (`track=None`) when no confirmed
    track claims it. These are low-score detections, and detections of tracks
    that never reached `--min_hits`.

    - Compare the score quantiles in the `MOT calibration: ...` log line with the
      thresholds (`--track_low_threshold`, `--track_high_threshold`,
      `--new_track_threshold`), and lower the thresholds if needed.
    - For curated detections, turn the score logic off with
      `--mot_param score_mode=none`.
    - Lower `--min_hits`.

??? question "A MOT tracker fragments tracks"
    - Raise `--track_buffer`, add `--link_tracklets`, or give the known number of
      animals with `--max_tracks N`.
    - Thin, nearly straight animals have a narrow keypoint box, so box IoU and OKS
      fall off quickly with sideways motion. Try `--scoring_method
      euclidean_dist`, which uses centroid distance scaled by body size.

??? question "An appearance tracker is not using appearance"
    Look for the `appearance is disabled ...` warning, or check `"appearance"` →
    `"reason"` in the [provenance record](#provenance-record). Under
    `--appearance_threshold auto`, calibration turned appearance off: either no
    vectors, too few trials (for example a single animal), or a rank-1 below
    0.95 because the embedder does not separate the animals. Pass a number, such as
    `--appearance_threshold 0.6`, to force it on, then check the result with
    `sleap-nn eval-tracking`.
