# Centroid-only inference

Run a trained centroid model standalone — without a paired centered-instance
model — and save the predicted centroids to a `.slp` file. This is a
first-class single-stage pipeline ("animals as points"): use it when you only
need instance localization, tracking, or counting (no per-keypoint pose), or
as a quick sanity check on a centroid model in isolation.

A standalone centroid model is trained exactly like any other head — see the
[`config_centroid_unet_standalone.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_centroid_unet_standalone.yaml)
sample. Training is node-count-agnostic: a single-node skeleton works directly,
and a multi-node skeleton works too (its centroids collapse to a single point
at inference).

## Output representation contract

The predicted `.slp` uses a **single-node `'centroid'` skeleton**
(`sio.get_centroid_skeleton()`) — *not* the full training skeleton NaN-padded
at every non-anchor node. Each detection is one point.

- When the model was trained on a **multi-node** skeleton, inference
  automatically collapses the output to the 1-node `'centroid'` skeleton
  (`Predictor._resolve_centroid_packaging` engages when the head is a
  centroid layer and the training skeleton has more than one node).
- By **default** each detection is a single-node `PredictedInstance` on the
  `'centroid'` skeleton, with the centroid confidence as both the per-node and
  per-instance score. This is loadable by the current SLEAP frontend with no
  changes.

```python
import sleap_io as sio

labels = sio.load_slp("centroids.slp")
assert [n.name for n in labels.skeletons[0].nodes] == ["centroid"]
for frame in labels:
    for inst in frame.instances:
        (x, y) = inst.numpy()[0]   # the centroid point
```

### Opt-in `sio.PredictedCentroid` emission

If you prefer the dedicated centroid object over a single-node instance, opt in
with `--centroid-output` (CLI) / `emit_centroid` (Python). The choices are:

| Value | Output |
|-------|--------|
| `instance` (default) | Single-node `PredictedInstance` on the `'centroid'` skeleton. Frontend-compatible. |
| `centroid` | `sio.PredictedCentroid` in `LabeledFrame.centroids` (carries an instance-level score and a `source` tag). |
| `both` | Both representations. |

The `source` tag on a `PredictedCentroid` mirrors the trained target's meaning
(see the [anchor convention](#anchor-node-convention-586) below): an explicit
anchor records `"anchor:<node>"`; no anchor records `"center_of_mass"`.

## End-to-end workflow

### 1. Train

Train a standalone centroid head (full-resolution, no cropping). Start from the
[standalone sample config](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_centroid_unet_standalone.yaml)
or generate one (`sleap-nn config --pipeline centroid ...`):

```bash
sleap-nn train --config-name config_centroid_unet_standalone.yaml
```

### 2. Infer

`sleap-nn infer` auto-detects a centroid-only model when `--model_paths` points
to a single centroid directory. `--centroid_only` is only needed when you also
pass a centered-instance model but want centroid-only output.

```bash
# Auto-detected: a lone centroid model directory → centroid-only output.
sleap-nn infer \
    -i video.mp4 \
    -m models/centroid/ \
    -o centroids.slp

# Emit sio.PredictedCentroid objects instead of single-node instances.
sleap-nn infer \
    -i video.mp4 \
    -m models/centroid/ \
    -o centroids.slp \
    --centroid-output centroid

# Explicit override: both models configured, but only want centroids.
sleap-nn infer \
    -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/ \
    --centroid_only \
    -o centroids.slp
```

```python
from sleap_nn.inference.run import predict

# Auto-detect a lone centroid directory.
labels = predict(
    data_path="video.mp4",
    model_paths=["models/centroid/"],
    output_path="centroids.slp",
)

# Explicit override on a two-model setup, emitting PredictedCentroid objects.
labels = predict(
    data_path="video.mp4",
    model_paths=["models/centroid/", "models/centered_instance/"],
    centroid_only=True,
    emit_centroid="centroid",
)
```

### 3. Evaluate

Use distance-based matching for centroids — OKS is degenerate for a single
point (it needs the full keypoint set and per-node scales). `--match_method
auto` already selects centroid matching when the prediction skeleton is
single-node, but you can request it explicitly:

```bash
sleap-nn eval \
    -g labels.gt.slp \
    -p centroids.slp \
    --match_method centroid
```

Ground-truth centroids are computed with `generate_centroids` — the configured
`--anchor_part` if given, otherwise the NaN-ignoring mean of visible nodes
(#586). This is the same definition used to build training targets, so GT and
predictions agree.

### 4. Export and run exported inference

Standalone centroid export works end-to-end (ONNX and TensorRT). Export a
single centroid directory, then run the exported model with the same output
representation choices:

```bash
# Export a standalone centroid model.
sleap-nn export models/centroid -o exports/centroid --format onnx

# Run the exported model. --centroid-output mirrors the checkpoint flow.
sleap-nn predict exports/centroid video.mp4 -o centroids.slp \
    --centroid-output instance
```

```python
from sleap_nn.inference.predictor import Predictor

predictor = Predictor.from_export_dir(
    "exports/centroid",
    runtime="onnx",
    device="cpu",
    emit_centroid="centroid",
)
labels = predictor.predict("video.mp4")
```

The exported runtime reads the full training skeleton from
`training_config.yaml` and applies the same collapse, so the output is
bit-for-bit identical to the checkpoint path. See the
[Export guide](export.md#standalone-centroid) for details.

## Anchor-node convention (#586)

The centroid's *meaning* is defined by
[`generate_centroids`](../reference/sleap_nn/data/instance_centroids.md) — the
same function used for training-target generation and GT-centroid evaluation:

1. **`anchor_part`** in `training_config.yaml` (the centroid head config): the
   centroid is that node when visible.
2. **`anchor_part` unset** (recommended for a 1-node skeleton, where the sole
   node *is* the centroid): the centroid is the **NaN-ignoring mean of all
   visible nodes** in each instance.

This is project-wide convention as of v0.3 — earlier versions used the
**bounding-box midpoint**, which differs on asymmetric instances (long tails,
sprawled limbs). If you trained a centroid model on the old bbox-midpoint
convention with `anchor_part` unset, the GT centroid targets for partial
instances shift slightly; re-training is recommended.

## Interaction with filtering, tracking, and metrics

### Filtering

`FilterConfig` knobs apply to centroid-only outputs:

- **`min_instance_score`**: filters on the centroid confidence value.
- **`min_visible_nodes` / `min_visible_node_fraction`**: a single-node
  detection has exactly one visible node, so keep any threshold `<= 1`.
- **`overlapping` with `overlapping_method="oks"`**: emits a `UserWarning` and
  falls back to IoU. OKS needs the full keypoint set, which a centroid lacks.

### Tracking

Use `features="centroids"`. For a single-node skeleton the scoring method
auto-resolves to `euclidean_dist` (pixel distance between centroids); OKS /
keypoint scoring is degenerate on a single point
(`sleap_nn/inference/tracking.py`).

```python
from sleap_nn.inference.tracking import TrackerConfig

tracker_config = TrackerConfig(
    features="centroids",
    # scoring_method auto-resolves to "euclidean_dist" for single-node;
    # set it explicitly if you want to be sure.
    scoring_method="euclidean_dist",
    window_size=5,
    track_matching_method="hungarian",
)
```

### Metrics

Distance-based metrics (centroid localization error, instance count) work as
expected. OKS/PCK expect the full keypoint set and are degenerate for points —
use `--match_method centroid` (above) for evaluation.
