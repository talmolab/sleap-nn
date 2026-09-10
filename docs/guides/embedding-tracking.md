# Embedding (re-ID) tracking

Track animals by **appearance** in addition to — or instead of — pose and position. An
[`embedding`](../reference/models.md) (re-ID) model turns each instance crop into a
learned appearance vector; the tracker can then score detection-to-track association by
the **cosine similarity** of those vectors, rather than only by where the animal was.

This is the tracker-side counterpart of the embedding model: the model produces the
vectors (`--save_embeddings`), and the tracker consumes them either as a blend weight
on top of a geometric score (`--appearance_weight`) or as the sole feature
(`--features embeddings`, auto-paired with `--scoring_method cosine_sim`).

It works on **both** carriers — pose `PredictedInstance`s and segmentation
`PredictedSegmentationMask`s. Appearance matching is image-free, so the motion models
(`--use_flow` / `--use_kalman`) do not apply.

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
    `sleap_nn.evaluation.motion_diagnostic(labels, "pose")`: a `step_over_size` well
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

!!! tip "Pair with `local_queues`"
    `--candidates_method local_queues` keeps a per-track deque of recent vectors — a
    lightweight appearance *gallery*. Matching a new detection against that window
    (reduced by `--scoring_reduction`, e.g. `mean` = soft prototype) is a robust,
    no-extra-infrastructure re-ID step.

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

Pair it with a *geometric* `--features`; combining it with `--features embeddings` is
rejected, since that would blend appearance with itself.

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
    similarity falls to ~0.37. `motion_diagnostic` reports it for your data.

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

## Workflow 1 — track an existing `.slp` of embeddings

You already ran an embedding model with `--save_embeddings slp`, so each
detection in the `.slp` carries its appearance vector. Track it (no model, no GPU):

```bash
sleap-nn predict -i embedded.slp -t --features embeddings
```

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
  embedding model; you can still override them.
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

---

## Output: tracks, not global identities

This path emits per-video `sio.Track`s only (classification-as-tracking). It does
**not** fabricate a global `sio.Identity` from track/class names — a track name is not
a global animal identity. (Persisting predicted global identities from appearance is a
separate, future step; multi_class models can emit `sio.Identity` via
`class_output="identity"`.)

---

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `--appearance_weight w` | Blend appearance into a geometric score: `(1 - w) * geometry + w * appearance`. `0.15-0.5` is the measured range; `0.0` is inert | `0.0` |
| `--features embeddings` | Track by appearance ALONE (the sparse / post-occlusion regime) | — |
| `--scoring_method cosine_sim` | Cosine similarity (auto-selected for embeddings; `euclidean_dist` also allowed) | auto |
| `--euclidean_scale px` | Length scale for the distance→similarity kernel. Required with `--appearance_weight` when the geometric score is `euclidean_dist` (i.e. centroid-only detections); ignored otherwise | — |
| `--candidates_method local_queues` | Per-track appearance gallery (recommended) | `fixed_window` |
| `--save_embeddings {none,slp}` | Persist vectors in the tracked `.slp` (WF2/WF3) | `none` |

All other [tracking parameters](tracking.md#tracking-parameters)
(`--tracking_window_size`, `--scoring_reduction`, `--max_tracks`,
`--track_matching_method`, …) apply unchanged. Motion models (`--use_flow`,
`--use_kalman`) are not supported with appearance features.

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
    - Use `--candidates_method local_queues` with a larger `--tracking_window_size`.
    - Check the embedding model actually separates your animals (validate retrieval
      metrics) — appearance tracking is only as good as the embeddings.
    - Cap identities with `--max_tracks N` when the animal count is known.

??? question "How do I know whether the blend helped?"
    Score it against tracked ground truth:

    ```bash
    sleap-nn eval-tracking -g ground_truth.slp -p tracked.slp
    ```

    That reports ID switches, IDF1, MT/PT/ML, fragmentation and track purity, so a
    weight can be chosen on evidence rather than by feel. Compare arms over the SAME
    detections (retrack one prediction file at several weights) so the difference you
    read is the tracker's and not the detector's.

    Retracking chains: a `.slp` that carries appearance vectors keeps them through
    a track-only run, so each arm can retrack the same embedded file. Pass
    `--save_embeddings none` if you explicitly want a tracks-only output.

??? question "`appearance_weight` was rejected with 'already appearance-only'"
    `--features embeddings` scores by appearance alone, so blending appearance into it
    is not meaningful. Either drop `--appearance_weight` (appearance-only regime) or
    switch to a geometric `--features` and keep the weight (blended regime).
