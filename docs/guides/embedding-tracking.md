# Embedding (re-ID) tracking

Track animals by **appearance** instead of pose or position. An
[`embedding`](../reference/models.md) (re-ID) model turns each instance crop into a
learned appearance vector; the tracker then associates detections across frames by the
**cosine similarity** of those vectors. Because identity comes from *what the animal
looks like* rather than *where it is*, appearance tracking holds identity through
crossings, occlusions, and fast motion that confuse pose/OKS or centroid tracking.

This is the tracker-side counterpart of the embedding model: the embedding model
produces the vectors (see the model docs / `--save_embeddings`), and the tracker
consumes them via `--features embeddings` (auto-paired with
`--scoring_method cosine_sim`).

It works on **both** carriers — pose `PredictedInstance`s and segmentation
`PredictedSegmentationMask`s (both store embeddings since sleap-io's mask-modality
support). Appearance matching is image-free, so the motion models (`--use_flow` /
`--use_kalman`) do not apply.

---

## The core idea

| Pose tracking | Embedding tracking |
|---|---|
| feature = keypoints (`--features keypoints`) | feature = `"reid"` vector (`--features embeddings`) |
| score = OKS (`--scoring_method oks`) | score = cosine similarity (`--scoring_method cosine_sim`) |
| identity follows **position/pose** | identity follows **appearance** |

Everything else in the [tracker](tracking.md) is unchanged: the same candidate makers
(`fixed_window` / `local_queues`), windowing, score reduction, and Hungarian/greedy
assignment. Only the per-detection feature and the pairwise score differ.

!!! tip "Pair with `local_queues`"
    `--candidates_method local_queues` keeps a per-track deque of recent vectors — a
    lightweight appearance *gallery*. Matching a new detection against that window
    (reduced by `--scoring_reduction`, e.g. `mean` = soft prototype) is a robust,
    no-extra-infrastructure re-ID step.

---

## Workflow 1 — track an existing `.slp` of embeddings

You already ran an embedding model with `--save_embeddings slp` (or `both`), so each
detection in the `.slp` carries its `"reid"` vector. Track it (no model, no GPU):

```bash
sleap-nn predict -i embedded.slp -t --features embeddings
```

`--features embeddings` auto-selects `--scoring_method cosine_sim`. This is the
[track-only / retrack path](tracking.md#track-only-mode) — omit `--model_paths`. The
prior tracks (if any) are reassigned from scratch by appearance.

If the labels carry no `"reid"` embedding, you get a clear error pointing you to run
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
- `--tracking` lifts the usual requirement to pass `--embeddings_path` /
  `--save_embeddings` for an embedding model (the tracked `.slp` is the output).
- The offline `.h5` sidecar is **not** written in this path.

### Persisting the vectors

`--save_embeddings` controls whether the appearance vectors are stored in the tracked
`.slp` (independent of the tracks themselves):

| `--save_embeddings` | Tracked `.slp` contents |
|---|---|
| `none` (default) | tracks only — vectors are stripped after tracking |
| `slp` / `both` | tracks **and** the `"reid"` vectors (for later re-tracking, retrieval, clustering) |

```bash
# Track AND keep the appearance vectors in the output for later reuse:
sleap-nn predict -m models/embedding/ -i detections.slp -t --save_embeddings both
```

!!! note "Centroid-driven embedding streams"
    Embedding tracking operates on a `.slp` that already has detections (the
    mask-driven / pose path). The composed **centroid + embedding** stream (which crops
    predicted centroids from raw frames) has no source detections to attach to and
    re-track, so `--tracking` is rejected there — embed a `.slp` with detections
    instead (pass only the embedding model directory).

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
| `--features embeddings` | Track by the `"reid"` appearance vector | — |
| `--scoring_method cosine_sim` | Cosine similarity (auto-selected for embeddings; `euclidean_dist` also allowed) | auto |
| `--candidates_method local_queues` | Per-track appearance gallery (recommended) | `fixed_window` |
| `--save_embeddings {none,slp,both}` | Persist vectors in the tracked `.slp` (WF2) | `none` |

All other [tracking parameters](tracking.md#tracking-parameters)
(`--tracking_window_size`, `--scoring_reduction`, `--max_tracks`,
`--track_matching_method`, …) apply unchanged. Motion models (`--use_flow`,
`--use_kalman`) are not supported with appearance features.

---

## Troubleshooting

??? question "`features='embeddings' but no detection ... carries a 'reid' embedding`"
    The input `.slp` has no appearance vectors. Run the embedding model first
    (`--save_embeddings slp`), or use Workflow 2 to embed + track in one command.

??? question "Identity still switches"
    - Use `--candidates_method local_queues` with a larger `--tracking_window_size`.
    - Check the embedding model actually separates your animals (validate retrieval
      metrics) — appearance tracking is only as good as the embeddings.
    - Cap identities with `--max_tracks N` when the animal count is known.
