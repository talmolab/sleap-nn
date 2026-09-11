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

`sleap-nn predict` auto-detects a centroid-only model when `--model_paths` points
to a single centroid directory. `--centroid_only` is only needed when you also
pass a centered-instance model but want centroid-only output.

```bash
# Auto-detected: a lone centroid model directory → centroid-only output.
sleap-nn predict \
    -i video.mp4 \
    -m models/centroid/ \
    -o centroids.slp

# Emit sio.PredictedCentroid objects instead of single-node instances.
sleap-nn predict \
    -i video.mp4 \
    -m models/centroid/ \
    -o centroids.slp \
    --centroid-output centroid

# Explicit override: both models configured, but only want centroids.
sleap-nn predict \
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
    source="video.mp4",
    model_paths=["models/centroid/"],
    output_path="centroids.slp",
)

# Explicit override on a two-model setup, emitting PredictedCentroid objects.
labels = predict(
    source="video.mp4",
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

# Run the exported model via the unified predict command (the export dir is
# auto-detected). --centroid-output mirrors the checkpoint flow, and --runtime
# picks ONNX vs TensorRT.
sleap-nn predict -m exports/centroid -i video.mp4 -o centroids.slp \
    --centroid-output instance --runtime onnx
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

## Choosing what "centroid" means (#586)

The centroid's *meaning* is defined by
[`generate_centroids`](../reference/sleap_nn/data/instance_centroids.md) — the
same function used for training-target generation, top-down crop centers and
GT-centroid evaluation, so all three can never disagree. Four methods are
available, spelled exactly as in `sio.Instance.to_centroid`:

| `centroid_method` | centroid is | good for |
|---|---|---|
| `center_of_mass` *(default)* | mean of the visible nodes | most datasets |
| `bbox_center` | midpoint of the visible nodes' bounding box | the pre-v0.3 convention |
| `geometric_median` | Weiszfeld median of the visible nodes | skeletons where a node is sometimes badly localized — measured on flies13 and gerbil pose, one node off by a body length moves this centroid ~1.7× less than `center_of_mass` and ~5× less than `bbox_center`. It is *not* more stable than the mean when a node goes **missing**, which is a different perturbation. |
| `anchor` | the `anchor_part` node, falling back to `centroid_fallback` when it is occluded | a reliable, consistently-visible landmark |

Set it on the head config:

```yaml
model_config:
  head_configs:
    centroid:
      confmaps:
        centroid_method: geometric_median   # or center_of_mass / bbox_center
        anchor_part: null
```

and for an anchor with a non-default fallback:

```yaml
        anchor_part: thorax
        centroid_fallback: bbox_center      # used only when `thorax` is occluded
```

The same two knobs live on **every head that defines a crop or centroid center** —
`centroid.confmaps`, `centered_instance.confmaps`, `multi_class_topdown.confmaps`,
`centered_instance_segmentation.segmentation` and `embedding.embedding` — and mean
the same thing on each. On `embedding.embedding` they set the re-ID crop center for
**pose** data; a mask-driven embedding dataset crops on the mask's own center of
mass instead (see `data_config.preprocessing.crop_centering`).

**Defaults are unchanged.** `centroid_method: null` (the default) means "the
anchor node when `anchor_part` is set, else `center_of_mass`" — exactly the
behavior of every config written before this knob existed. Setting `anchor_part`
*and* a non-anchor `centroid_method` is rejected at setup: they name two
different centroids, and `centroid_fallback` is what you want instead.

**Match it at inference and evaluation.** The knob rides the checkpoint, so
`sleap-nn predict` reproduces the training geometry automatically, and the
recorded `sio.Centroid.source` tag names the method used. When evaluating a
prediction file against ground truth by hand, pass the same method — otherwise
the distance metric scores the model against a different definition of centroid
than the one it was trained on:

```bash
sleap-nn eval --match_method centroid --centroid_method geometric_median ...
```

`center_of_mass` became the project-wide default in v0.3 — earlier versions used
the bounding-box midpoint, which differs on asymmetric instances (long tails,
sprawled limbs). A model trained on the old convention is now expressible
directly as `centroid_method: bbox_center`, so it no longer needs re-training to
be described accurately.

### Training a centroid model on mask-only labels

Labels that carry segmentation masks but no poses have nothing for the centroid
target to be derived from. `data_config.centroids_from_masks` names a method and
derives a `UserCentroid` per mask at load time (via
`sio.SegmentationMask.to_centroid`), after which the ordinary
`centroid_source: user` path takes over unchanged:

```yaml
data_config:
  centroids_from_masks: center_of_mass   # or bbox_center / geometric_median
```

Frames that already carry user centroids are left alone — a real annotation
always outranks a derived one. `anchor` does not apply here: a mask has no nodes.

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
