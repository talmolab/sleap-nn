# Embedding (re-ID) tracking

Track animals by **appearance** in addition to — or instead of — pose and position. An
[`embedding`](../reference/models.md) (re-ID) model turns each instance crop into a
learned appearance vector; the tracker can then score detection-to-track association by
the **cosine similarity** of those vectors, rather than only by where the animal was.

This is the tracker-side counterpart of the embedding model: the model produces the
vectors (`--save_embeddings`), and the tracker consumes them in one of these ways:

- With the default `classic` tracker, as a blend weight on top of a geometric score
  (`--appearance_weight`), or as the sole feature (`--features embeddings`,
  auto-paired with `--scoring_method cosine_sim`).
- Through an [appearance-aware MOT tracker](#appearance-aware-mot-trackers)
  (`--tracker botsort` / `deepocsort` / `deepsort` / `strongsort`). These use
  appearance inside a Kalman-predicted geometric gate, with thresholds calibrated
  per video.
- Through [identity linking](#identity-linking-vs-the-appearance_weight-blend)
  (`--link_tracklets`, `--max_tracks N`), which joins whole tracklets after
  tracking, with any tracker.

It works on **both** carriers — pose `PredictedInstance`s and segmentation
`PredictedSegmentationMask`s. Appearance matching is image-free, so the classic
tracker's motion models (`--use_flow` / `--use_kalman`) do not apply. The MOT
trackers have their own Kalman motion model.

---

## Which regime you want

Appearance is a **complementary** cue, not a strictly better one. Whether it helps, and
how much, depends on how informative geometry already is — which is a property of your
*data*, not of the model:

| regime | how | when it wins |
|---|---|---|
| **Blended** (recommended) | a geometric `--features` + `--appearance_weight 0.15-0.5` | dense continuous video, where geometry is informative and appearance breaks its ties |
| **Appearance-only** | `--features embeddings` | sparse frames and post-occlusion recovery, where geometry has *no* signal |
| **Geometry-only** | the default; no embeddings needed | when identities never need recovering — and as the baseline you should beat |
| **Appearance-aware MOT tracker** | `--tracker botsort` / `deepocsort` (or `deepsort` / `strongsort`) | continuous video with a motion model; appearance is used only where this video's calibration shows it separates the animals ([below](#appearance-aware-mot-trackers)) |
| **+ identity linking** | `--link_tracklets` or `--max_tracks N`, with any of the above | animals that leave, hide for long stretches or cross; links whole tracklets after tracking |

The measurements below compare the first three regimes. They predate the MOT
trackers.

### What was measured

On two held-out gerbil sessions of dense continuous video (per-frame displacement ~4-6%
of body length), geometry alone clearly beats appearance alone:

| 237 center session | ID switches | IDF1 |
|---|---|---|
| keypoints / OKS | **32** | 0.849 |
| embeddings / cosine (appearance-only) | 227 | 0.800 |

Blending the two is better than either: **IDF1 0.849 → 0.878** and, on the second
session, **0.855 → 0.982**, with fewer ID switches on one of them.

The picture inverts when geometry stops being informative. On a temporally sparse set
(consecutive "frames" ~8.7 body lengths apart, so same-animal mask IoU is 0.000 for
most pairs), appearance-only is transformative:

| sparse gerbil set | ID switches | IDF1 |
|---|---|---|
| masks / mask IoU | 169 | 0.245 |
| embeddings / cosine (appearance-only) | **12** | **0.967** |

!!! warning "How far this generalizes"
    Two sessions, one species, one model. The direction of the effect is well founded —
    and mechanical, since it turns on whether consecutive detections of one animal
    overlap at all — but the specific weights are not tuned for your data. Check where
    your own recording sits with
    `sleap_nn.evaluation.motion_diagnostic(tracked_labels, "pose")` (it follows
    each animal by its track, so run it on tracked ground truth or on a
    geometry-only tracking run): a `step_over_size` well
    below 0.5 is the dense regime (blend), well above it is the sparse regime
    (appearance-only), and `sleap-nn eval-tracking` will score any choice you make
    against tracked ground truth.

---

## The core idea

| Pose tracking | Embedding tracking |
|---|---|
| feature = keypoints (`--features keypoints`) | feature = appearance vector (`--features embeddings`) |
| score = OKS (`--scoring_method oks`) | score = cosine similarity (`--scoring_method cosine_sim`) |
| identity follows **position/pose** | identity follows **appearance** |

Blending is the two together: the geometric score for the pair, plus
`appearance_weight` times their appearance similarity. At the default `0.0` the
appearance term is never computed, so a geometry-only run is unchanged.

Everything else in the [tracker](tracking.md) is unchanged: the same candidate makers
(`fixed_window` / `local_queues`), windowing, score reduction, and Hungarian/greedy
assignment. Only the per-detection feature and the pairwise score differ.

!!! tip "`local_queues` is the default here"
    With `--features embeddings` the candidate maker defaults to `local_queues`, which
    keeps a per-track deque of recent vectors — a lightweight appearance *gallery*.
    Matching a new detection against that window (reduced by `--scoring_reduction`,
    e.g. `mean` = soft prototype) is a robust, no-extra-infrastructure re-ID step,
    and unlike `fixed_window` it re-binds an animal that was absent for longer than
    `--tracking_window_size` frames. Pass `--candidates_method fixed_window` to opt
    out.

---

## Blending appearance with geometry

```bash
# Recommended default on continuous video: keep OKS as the primary cue and let
# appearance break its ties.
sleap-nn predict -i video.mp4 -m centroid_dir -m centered_instance_dir -m embedding_dir \
    -t --appearance_weight 0.3
```

`--appearance_weight w` scores each candidate pair as `(1 - w) * geometry + w * appearance`:

- `0.0` (default) — geometry only. Inert: the appearance term is not computed.
- `0.15-0.5` — the measured sweet spot on dense video.
- `1.0` — appearance only, but *still gated by geometry's candidate set*; prefer
  `--features embeddings` if that is what you want.

Two properties worth knowing:

- **A detection with no embedding keeps its geometric score.** It is not blended toward
  a missing value, so it still matches on geometry instead of being dropped and spawning
  a spurious track.
- **The blend cannot invent matches geometry rejected.** Where geometry has no valid
  candidate (nothing passed `--min_match_points`), the pair stays unmatched.

Pair it with a *geometric* `--features` and `--scoring_method`; combining it with
`--features embeddings` is rejected, since that would blend appearance with itself,
and so is `--scoring_method cosine_sim` as the geometric score (it would score the
cosine of raw coordinates, near 1 for any two poses).

!!! note "The blend reads the vectors of the detections it tracks"
    On a pose + mask file (top-down segmentation, SAM output) the embedding model
    attaches the vectors to **the carrier it was trained on**: training records it in
    the model's `head_configs.embedding.embedding.detection_mode` (`pose` or `mask`);
    a model saved before that field existed counts as mask-trained when it uses
    `burn_in`, and otherwise takes the carrier holding more detections (ties to
    masks). A file without that carrier gets its other carrier embedded. When the
    vectors land on the masks, pose tracking reads the poses — so
    `--appearance_weight` with a pose `--features` is rejected there rather than
    running as geometry only. Track the masks instead with `--features masks`
    (appearance blended into mask IoU): every pose linked to exactly one tracked
    mask inherits that mask's track. `--features embeddings` follows the vectors to
    whichever carrier holds more of them.

    On the fused route a **segmentation** detection stack (`centroid` +
    `centered_instance_segmentation`, `bottomup_segmentation`) emits masks only, so
    tracking follows the masks: leave `--features` unset (or pass `--features
    masks`). A pose `--features` or `--scoring_method` is refused before the
    detection stack runs.

!!! warning "Distance scores need `--euclidean_scale`"
    Blending only means something when both terms live on the same scale. `oks`,
    `iou` and `mask_iou` are already bounded, so they blend as-is.
    `euclidean_dist` is not — it returns negative *pixels*, so at any realistic
    image scale the geometric term dominates by two orders of magnitude and the
    weight would be numerically inert.

    It is therefore mapped through a bounded kernel first:

    ```
    geometric similarity = exp(-distance / euclidean_scale)
    ```

    a `(0, 1]` value comparable to the appearance cosine. The kernel is strictly
    monotone in distance, so the candidate *ordering* geometry alone would have
    produced is unchanged — only the scale is.

    `--euclidean_scale` is a length in **pixels** with no universal default, so it
    is **required** for this combination rather than guessed. Pass the typical
    inter-frame displacement of one animal — the distance at which geometric
    similarity falls to ~0.37. `motion_diagnostic` reports it as
    `median_step_px` (per frame) when run on **tracked** labels of your data —
    tracked ground truth, or a geometry-only tracking run.

    This is the path **centroid-only detections** take: a single-node skeleton
    auto-selects `euclidean_dist`, so the guide's centroid + embedding fused
    command needs the scale.

    ```bash
    sleap-nn predict -i video.mp4 -m centroid_dir -m embedding_dir \
        -t --appearance_weight 0.3 --euclidean_scale 25
    ```

    Two consequences worth knowing: the mapping happens **only when blending**, so
    a geometry-only distance run is byte-identical to before; and a blended
    distance run's `tracking_score` is a `(0, 1]` similarity rather than negative
    pixels.

Every incoherent combination is rejected **before** inference starts, not after the
detection stack and embedding pass have run.

---

## Appearance-aware MOT trackers

Four of the [modern MOT trackers](tracking.md#modern-mot-trackers) read the same
`identity_embedding` vectors as the classic tracker, and combine them with a
per-track Kalman motion model:

| `--tracker` | how appearance is used |
|---|---|
| `botsort` | Appearance can support a match only when the pair is already geometrically close (`proximity_threshold`). The cosine to the track's appearance EMA must also clear the gate, and the pair must lead the next-best candidate in its row and column by a calibrated margin. |
| `deepocsort` | A standardized appearance term is added to OC-SORT's objective, weighted more where it is discriminative. A match with weak geometry can be rescued when appearance is decisive. |
| `deepsort` | Appearance is the primary cue: a matching cascade on the best cosine to each track's gallery, inside a Mahalanobis motion gate. |
| `strongsort` | Appearance is the primary cue: one assignment on the EMA cosine blended with motion (`mc_lambda` 0.98). |

They plug into the same three workflows:

```bash
# Workflow 1: retrack a .slp that already carries vectors (no model, no GPU)
sleap-nn predict -i embedded.slp -t --tracker botsort

# Workflow 2: embed the detections of a .slp, then track them
sleap-nn predict -m models/embedding/ -i detections.slp -t --tracker deepocsort

# Workflow 3: detect + embed + track a video, 4 animals, with identity linking
sleap-nn predict -i video.mp4 \
    -m models/centroid/ -m models/centered_instance/ -m models/embedding/ \
    -t --tracker botsort --link_tracklets --max_tracks 4 -o tracked.slp
```

How they differ from the classic appearance options:

- **Calibrated, not hand-set.** The default `--appearance_threshold auto`
  measures the cosine gates, margins and linking thresholds on each video's own
  detections before tracking. When the measurements show that appearance does not
  separate the animals, appearance is turned off and a warning says so; the run
  does not fall back to a constant tuned on person re-ID. See [Per-video
  calibration](tracking.md#per-video-calibration-of-appearance-thresholds),
  including the *"embedding may encode class/sex rather than identity"* warning.
  A number, such as `--appearance_threshold 0.6`, sets a fixed minimum cosine
  for association instead. Linking keeps its calibrated thresholds.
- **Only quality-gated vectors update a track's appearance.** The detection must
  be confident (`--track_high_threshold`), at least half visible, and not
  overlapping another detection or another track's predicted box. Once the
  track's gallery holds 3 vectors, the detection must also favor this track over
  the other nearby tracks by the calibrated margin. Crops taken during a crossing
  would otherwise pull the track's appearance toward the wrong animal.
- **`--appearance_weight` and `--features embeddings` belong to the classic
  tracker.** A MOT tracker refuses both and suggests an appearance tracker
  instead. Pick the carrier with `--features keypoints` or `masks`. Left unset,
  an appearance tracker follows the carrier that holds more vectors.
  `deepsort` / `strongsort` refuse vectors that are on the other carrier;
  `botsort` / `deepocsort` warn and track on geometry alone.
- **Vectors are required.** With no vector on any detection, all four refuse to
  run. When the run produces its own detections (a video input, or detection
  models in `-m`), it also needs an `embedding` model among `-m`. A run in which
  at least 10% of the tracked detections lack a vector logs a warning. Those
  detections are scored on geometry alone.

A geometry-only MOT tracker (`sort`, `bytetrack`, `ocsort`) can still use the
vectors for identity linking:

```bash
sleap-nn predict -m models/embedding/ -i detections.slp -t --tracker bytetrack --link_tracklets
```

Without `--link_tracklets`, `--max_tracks` or `--save_embeddings slp`, that
combination is refused, because the vectors would be computed and then thrown
away.

### Identity linking vs the `appearance_weight` blend

The two options answer different questions. Use either one, or both
(`--appearance_weight` with `--link_tracklets` on the classic tracker).

| | `--appearance_weight w` (classic) | identity linking (`--link_tracklets`, `--max_tracks N`) |
|---|---|---|
| decides | every detection-to-candidate match, frame by frame | whether two whole tracks are the same animal, after the video is tracked |
| evidence | one detection's vector against the candidate window | up to 30 gated vectors at each end of a track (mean of the top-5 cosines per query) |
| threshold | your fixed weight `w` | calibrated per video: a linking threshold and the smallest margin that reaches 99% precision on the video's own trials |
| when appearance does not discriminate | the blend still applies | spatial rules only: a link is made only when it is the unique candidate within 4 × `--track_buffer` steps |
| also does | — | corrects identity swaps at crossings; with `--max_tracks N`, merges or untracks identities down to N |
| works with | `classic` only | every tracker |

The blend breaks ties between nearby candidates while tracking. Linking repairs
what the tracker could not: an animal that returns after a long absence under a
new name, two identities swapped at a crossing, or more tracks than animals. A
link must beat every competing candidate that provably is another animal (they
appear in the same frame), so a link the margin does not support is not made.
See [Identity linking](tracking.md#identity-linking-long-term-identity-memory) for
the full rules.

Linking is per video, like tracking. It does not carry an identity from one
video or session to the next (see [Output](#output-tracks-not-global-identities)).

---

## Workflow 1 — track an existing `.slp` of embeddings

You already ran an embedding model with `--save_embeddings slp`, so each
detection in the `.slp` carries its appearance vector. Track it (no model, no GPU):

```bash
sleap-nn predict -i embedded.slp -t --features embeddings
```

To produce `embedded.slp`, run the embedding model on any `.slp` of detections —
tracked or untracked; every detection gets a vector, and any vectors the input
already carried (e.g. from an earlier model) are replaced:

```bash
sleap-nn predict -m models/embedding/ -i detections.slp --save_embeddings slp -o embedded.slp
```

The default output is `<input>.embeddings.slp` (for a URL input, the URL's file name
in the current directory). `-o` may not name the input when the input stores its own
frames (a `.pkg.slp`): the frames would be destroyed, so the run is refused before
anything is computed (on every `predict` route, including the fused one below).

`--features embeddings` auto-selects `--scoring_method cosine_sim`. This is the
[track-only / retrack path](tracking.md#track-only-mode) — omit `--model_paths`. The
prior tracks (if any) are reassigned from scratch by appearance.

If the labels carry no appearance embedding, you get a clear error pointing you to run
the embedding model first.

---

## Workflow 2 — embed + track in one command

Run the embedding model on a `.slp` of **detections** (tracked *or untracked*) and
track them by appearance in a single command. Every detection is embedded (the
embedding model runs in "include-untracked" mode), the vectors are attached, and the
tracker assigns `sio.Track`s by cosine similarity:

```bash
sleap-nn predict -m models/embedding/ -i detections.slp -t
```

- The output is a **tracked `.slp`** (default `<input>.tracked.slp`, or `-o out.slp`).
- `--features` / `--scoring_method` default to `embeddings` / `cosine_sim` for an
  embedding model; you can still override them. With `--appearance_weight` they
  are not defaulted: the usual geometric resolution applies and appearance is
  blended into it.
- `--tracking` lifts the usual requirement to pass `--save_embeddings` for an embedding
  model (the tracked `.slp` is the output).

### Persisting the vectors

`--save_embeddings` controls whether the appearance vectors are stored in the tracked
`.slp` (independent of the tracks themselves):

| `--save_embeddings` | Tracked `.slp` contents |
|---|---|
| `none` (default) | tracks only — vectors are stripped after tracking |
| `slp` | tracks **and** the appearance vectors (for later re-tracking, retrieval, clustering) |

```bash
# Track AND keep the appearance vectors in the output for later reuse:
sleap-nn predict -m models/embedding/ -i detections.slp -t --save_embeddings slp
```

---

## Workflow 3 — detect, embed, and track a video (fused)

Pass a detection stack **and** the embedding model together to run detection,
embedding, and appearance tracking on a raw video in one command. The detection models
(a centroid, optionally plus a centered-instance model, i.e. top-down) run first to
produce the per-frame detections, those detections are embedded, and the tracker
assigns `sio.Track`s by cosine similarity:

```bash
# top-down detect (centroid + centered_instance) -> embed -> track, in one command
sleap-nn predict -m models/centroid/ -m models/centered_instance/ \
  -m models/embedding/ -i video.mp4 -t --max_tracks 6 -o tracked.slp
```

This is exactly Workflow 2 with the detection step folded in — equivalent to running
`sleap-nn predict -m centroid -m centered_instance -i video.mp4 -o poses.slp` and then
`sleap-nn predict -m embedding -i poses.slp -t -o tracked.slp`. `--save_embeddings slp`
keeps the vectors in the output as above; a centroid-only detector (no
centered-instance model) yields single-node detections to embed + track.

The output is always a `.slp` that references the source video: `--output_format
analysis_h5` and `--embed true|auto` are rejected on this route rather than ignored.
With `--video_index` on a multi-video `.slp`, the default output name includes the
video's name (`<input>.<video>.tracked.slp`), as with `predict`, so per-video runs do
not overwrite each other.

---

## Output: tracks, not global identities

This path emits per-video `sio.Track`s only (classification-as-tracking). It does
**not** fabricate a global `sio.Identity` from track/class names — a track name is not
a global animal identity. (Persisting predicted global identities from appearance is a
separate, future step; multi_class models can emit `sio.Identity` via
`class_output="identity"`.)

Each video is tracked on its own, by a fresh tracker: a track never spans two videos,
and track names continue across videos (`track_0`, `track_1` in the first;
`track_2`, … in the next). So appearance tracking does not link one animal across
the videos or sessions of a multi-video `.slp`. Identity linking
(`--link_tracklets`) also works within one video.

---

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `--appearance_weight w` | Blend appearance into a geometric score: `(1 - w) * geometry + w * appearance`. `0.15-0.5` is the measured range; `0.0` is inert | `0.0` |
| `--features embeddings` | Track by appearance ALONE (the sparse / post-occlusion regime) | — |
| `--scoring_method cosine_sim` | Cosine similarity (auto-selected for embeddings; `euclidean_dist` also allowed) | auto |
| `--euclidean_scale px` | Length scale for the distance→similarity kernel. Required with `--appearance_weight` when the geometric score is `euclidean_dist` (i.e. centroid-only detections); ignored otherwise | — |
| `--candidates_method` | `local_queues` = per-track appearance gallery | `local_queues` with `--features embeddings`, when tracking masks, or with `--max_tracks`; else `fixed_window` |
| `--save_embeddings {none,slp}` | Persist vectors in the tracked `.slp` (WF2/WF3) | `none` |
| `--tracker {botsort,deepocsort,deepsort,strongsort}` | An [appearance-aware MOT tracker](#appearance-aware-mot-trackers) in place of the classic tracker. The classic options above (`--appearance_weight`, `--features embeddings`, `--candidates_method`, …) do not apply to it | `classic` |
| `--appearance_threshold {auto,FLOAT}` | MOT appearance trackers: the minimum cosine for appearance to support a match, or `auto` to calibrate it per video | `auto` |
| `--link_tracklets` | [Identity linking](#identity-linking-vs-the-appearance_weight-blend) after tracking each video, with any tracker | off |
| `--max_tracks N` | Classic: caps the number of tracks. MOT trackers: the known animal count, giving closed-set identity linking | — |

All other [tracking parameters](tracking.md#tracking-parameters)
(`--tracking_window_size`, `--scoring_reduction`, `--track_matching_method`, …)
apply unchanged to the classic tracker. Its motion models (`--use_flow`,
`--use_kalman`) are not supported with appearance features. The MOT trackers take
[their own options](tracking.md#options).

---

## Troubleshooting

??? question "`features='embeddings' but no detection ... carries an appearance embedding`"
    The input `.slp` has no appearance vectors. Run the embedding model first
    (`--save_embeddings slp`), or use Workflow 2 to embed + track in one command.
    `--appearance_weight` on vector-less labels is rejected the same way, rather
    than silently degrading to a geometry-only run.

??? question "`appearance_weight ... requires euclidean_scale`"
    Your `--scoring_method` resolved to `euclidean_dist` — usually because the
    detections are centroid-only (a single-node skeleton auto-selects it). Pass
    `--euclidean_scale <px>`, the typical inter-frame displacement of one animal.
    See the warning under [Blending appearance with
    geometry](#blending-appearance-with-geometry) for how the kernel works, or
    switch to a bounded metric (`oks` / `iou` / `mask_iou`).

??? question "Identity still switches"
    - **If your video is continuous, try the blend rather than appearance alone**:
      a geometric `--features` with `--appearance_weight 0.3`. Appearance-only was
      measurably worse than geometry on dense video (227 ID switches against 32).
    - Keep the default `local_queues` candidate maker and raise `--tracking_window_size`.
    - Check the embedding model actually separates your animals (validate retrieval
      metrics) — appearance tracking is only as good as the embeddings.
    - Cap identities with `--max_tracks N` when the animal count is known.
    - Try an [appearance-aware MOT tracker](#appearance-aware-mot-trackers) with
      identity linking, e.g. `--tracker botsort --link_tracklets` (add
      `--max_tracks N` for a known count). Read the `MOT calibration: ...` log
      line first. If appearance comes out "off", or the *"embedding may encode
      class/sex rather than identity"* warning appears, the embedder does not
      separate these animals, and no tracker setting will fix that.

??? question "How do I know whether the blend helped?"
    Score it against tracked ground truth:

    ```bash
    sleap-nn eval-tracking -g ground_truth.slp -p tracked.slp
    ```

    That reports ID switches, IDF1, MT/PT/ML, fragmentation and track purity, so a
    weight can be chosen on evidence rather than by feel. Compare arms over the SAME
    detections (retrack one prediction file at several weights) so the difference you
    read is the tracker's and not the detector's.

    Identity is scored per video (a trajectory is a `(video, track name)` pair, as
    the tracker names tracks per video); pass `--global_identity` only when the
    ground truth's names are global identities. Centroid-only detections are matched
    to the ground truth by pixel distance (50 px by default; `--match_threshold` is
    in pixels there). See [Evaluating Identity
    Persistence](tracking.md#evaluating-identity-persistence) for what is scored.

    Retracking chains: a `.slp` that carries appearance vectors keeps them through
    a track-only run, so each arm can retrack the same embedded file. Pass
    `--save_embeddings none` if you explicitly want a tracks-only output.

??? question "`appearance_weight` was rejected with 'already appearance-only'"
    `--features embeddings` scores by appearance alone, so blending appearance into it
    is not meaningful. Either drop `--appearance_weight` (appearance-only regime) or
    switch to a geometric `--features` and keep the weight (blended regime).
