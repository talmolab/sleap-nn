# Part stage: extra keypoints from keypoint-placed crops

Some body parts fall outside the crop that a top-down pose model sees. A long tail,
for example, often reaches beyond the body box that the first stage and the pose model
use. The **part stage** adds a second centered-instance model, the *part model*,
trained on crops around such a part. At prediction time its crops are placed by a
**crop rule** on each predicted instance's keypoints, and its points are appended to
that instance:

```
first stage (centroid model or box detector) -> pose model -> keypoints
    -> crop rule -> part boxes -> part model -> part keypoints
    -> one instance per animal: pose nodes, then part nodes
```

The part stage is off unless `--part_model` is given; without it nothing in the
prediction path changes.

## Quick start

```bash
sleap-nn predict -i video.mp4 -o video.predictions.slp \
    -m models/detector -m models/pose \
    --part_model models/tail --part_crop_rule tail_rule.json
```

`tail_rule.json`, a square centered on the tail base, sized by the distance from the
tail base to the middle of the two ears:

```json
{
  "center": ["tail_base"],
  "reference": ["left_ear", "right_ear"],
  "reference_fallback": ["neck"],
  "scale": 2.0
}
```

In Python, `predict(..., part_model_paths=["models/tail"],
part_crop_rule="tail_rule.json")` does the same; the rule can also be a `dict` or a
`sleap_nn.inference.keypoint_crops.KeypointCropRule`.

## What you need

- **A pose model that ends in keypoints**, given with `-m` as usual: a single-instance
  or bottom-up model, or a top-down pair (a centroid model or a box detector with a
  centered-instance model). Checkpoint directories or network-only exports (see
  [Export](export.md#detector-box-scaled-top-down-part-stage-network-only-exports)).
- **A part model**: a `centered_instance` model whose skeleton holds the part's nodes
  (e.g. `tail_base`, `tail_mid`, `tail_tip`). It runs on the rule's boxes the way a
  centered-instance model runs on detector boxes ([Top-down with a box
  detector](bbox-topdown.md)): with `crop_mode: bbox` each square box, padded by the
  model's `crop_bbox_padding`, is resized to the crop size, so crops scale with the
  animal; with `crop_mode: fixed` a `crop_size` window is cut at the box center.
- **A crop rule** (`--part_crop_rule`, a `.json` or `.yaml` file).

Train the part model on crops that look like the ones the rule makes. With
`crop_mode: bbox`, give each training instance a user bounding box linked to it
(`box.instance`) built from the labelled keypoints with the same center, reference
nodes and `scale` as the rule. sleap-nn does not write those training boxes for you.

## The crop rule

Each predicted instance gets at most one square box:

| Field | Default | Meaning |
|---|---|---|
| `center` | required | Node names, tried in order: the first one present in the skeleton and located is the box center |
| `reference` | `[]` | Nodes whose mean point sets the size; all of them must be located |
| `reference_fallback` | `[]` | Used when a `reference` node is missing |
| `scale` | `2.0` | Half side of the box = `scale` x the distance from the center to the reference point |
| `fallback` | `"extent"` | With no center or no usable reference: `"extent"` centers the box on the extent of the instance's located points; `"skip"` makes no box (the instance gets no part points) |
| `fallback_scale` | `1.0` | Half side of an extent box = `fallback_scale` x half the extent's longer side |
| `min_half_side` | `1.0` | Smallest half side, in pixels |
| `session_floor` | `null` (off) | Also floor the half side at `session_floor` x the median half side of the rule's boxes in the same video (see below) |
| `session_min_boxes` | `10` | A video with fewer boxes than this uses the median over all videos instead of its own |

Node names match exactly, or by the part after the last `/`: `"tail_base"` matches a
node named `"lab_a/tail_base"`, but only when exactly one node ends that way; an
ambiguous name matches nothing. Unknown keys in the rule file raise an error.

### Per-video floor (`session_floor`)

A box sized from a short reference distance can be too small to hold the part: when
the reference nodes nearly line up with the center (an animal seen end-on, or
rearing), the distance shrinks and so does the crop. `session_floor` keeps the half
side at least `session_floor` times the median half side of the rule's boxes in that
video, about the same center. The median is taken over the whole video's first-stage
predictions before any part crop is made, so the floor needs a second pass over the
video (the default, two-pass part stage). The floors used are written to the output's
provenance (`provenance["part_stage"]["session_floors"]`, per video index).

```json
{
  "center": ["tail_base"],
  "reference": ["left_ear", "right_ear"],
  "scale": 2.0,
  "session_floor": 1.0,
  "session_min_boxes": 10
}
```

## Options

| Option | Python (`predict`) | Description | Default |
|---|---|---|---|
| `--part_model` | `part_model_paths` | The part model directory (checkpoint, or network-only export) | none (off) |
| `--part_crop_rule` | `part_crop_rule` | The rule file (`.json` / `.yaml`); required with `--part_model` | none |
| `--part_node_prefix` | `part_node_filter` | Predict only the part model's nodes whose names start with this prefix (its confidence-map head is sliced to them, as with `--node_prefix`). In Python, `part_node_filter` also takes a list of names, and `part_strip_prefix` drops the prefix | all nodes |
| `--part_peak_threshold` | `part_peak_threshold` | Peak threshold of the part model; points below it are NaN | the model's own |
| `--part_prefix` | `part_prefix` | Prefix added to the part nodes in the merged skeleton. Needed when a part node has the name of a pose node, which otherwise raises | none |
| `--part_streaming` | `part_streaming` | Experimental: run the part model batch by batch, see below | off |

`--runtime` (and, in Python, `precision` and `device`) apply to the part model as to
the pose model.

## Output

Each predicted instance keeps its pose nodes, score, track and boxes, and gains the
part nodes after them (renamed with `--part_prefix`). The merged skeleton keeps the
edges and left/right symmetries of both models. An instance without a part box (rule
`fallback: skip`), or whose part prediction could not be matched back to it, gets NaN
part points. The rule, the prefix and the part model's provenance are recorded under
`provenance["part_stage"]`.

Tracking (`--tracking`) uses the pose nodes only, with or without the part stage.

## Streaming part stage (`--part_streaming`)

!!! warning "Experimental"
    `--part_streaming` may change or be removed in a later release without a
    deprecation period. Its results depend on `--batch_size` when a per-video floor
    is used.

By default the part stage runs after the whole video: it reads every frame a second
time, and the floor is the median of the whole video. `--part_streaming` instead runs
the part model on each batch right after the pose model, on the frames already in
memory, with a **running** floor: `session_floor` x the median half side of the
boxes seen so far in that video, including the rest of the current batch, once at
least `session_min_boxes` boxes were seen (before that, the median over all videos so
far, or no floor). How this differs from the two-pass floor:

- each frame's floor includes the later frames of its batch, so results depend on
  `--batch_size`;
- the first `session_min_boxes` boxes of a run get no floor at all;
- every run (each `--frames` chunk) starts a new floor.

Without `session_floor`, streaming and two-pass runs give the same points.
[`open_stream`](inference-api.md#frame-in-pose-out-open_stream) always runs the part
stage this way.

## Limits

- `--part_model` is refused with `--stream-to-file` and with fused exported models;
  network-only exports work (`sleap-nn export --network-only`).
- One part model per run.
- The crop is a square; the part model sees what the rule's box holds, so a rule that
  misses the part on some poses gives NaN or wrong part points there. Look at a few
  crops before relying on it.
