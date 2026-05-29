# Centroid-only inference

Run a trained centroid model standalone — without a paired centered-instance
model — and save the predicted centroids to a `.slp` file. Useful when you
only need instance localization (no pose), or as a quick sanity check on a
centroid model in isolation.

## Quick start

### CLI

```bash
# Auto-detected: single centroid model_path → centroid-only output.
sleap-nn infer video.mp4 \
    --model_paths models/centroid/ \
    --output_path centroids.slp

# Explicit override: both models configured, but only want centroids.
sleap-nn infer video.mp4 \
    --model_paths models/centroid/ \
    --model_paths models/centered_instance/ \
    --centroid-only \
    --output_path centroids.slp
```

### Python

```python
from sleap_nn.inference import Predictor

# Auto-detect.
predictor = Predictor.from_model_paths(["models/centroid/"])
labels = predictor.predict("video.mp4")
labels.save("centroids.slp")

# Explicit override on a two-model setup.
predictor = Predictor.from_model_paths(
    ["models/centroid/", "models/centered_instance/"],
    centroid_only=True,
)
```

## Output structure: anchor node + NaN padding

The output `.slp` uses the **full skeleton** from training (every node
position is present in every `PredictedInstance`), but only the anchor node
slot is populated:

- **Anchor node** (configured `anchor_part`, or node 0 if unset): receives
  the predicted centroid coordinate. The per-keypoint score equals the
  centroid confidence value.
- **Every other node**: NaN for both coordinates and scores.
- **Per-instance score** (`PredictedInstance.score`): centroid confidence.

This packaging plugs into every standard `.slp` consumer (sleap-io readers,
metrics, viewers) — NaN is the natural "not predicted" marker and downstream
tools degrade gracefully.

```python
import sleap_io as sio
import numpy as np

labels = sio.load_slp("centroids.slp")
for frame in labels:
    for inst in frame.instances:
        pts = inst.numpy()                  # (n_nodes, 2)
        anchor_xy = pts[0]                  # anchor node — populated
        rest = pts[1:]                      # all other nodes — NaN
        assert np.all(np.isnan(rest))
```

## Anchor-node convention

The anchor node is resolved in this priority order, falling back gracefully:

1. **`anchor_part`** in `training_config.yaml` (model head config).
2. **Node 0** if `anchor_part` is unset (documented default).

Note that the *centroid value itself* — the (x, y) that the model is trained
to predict and that comes out at inference — uses a separate fallback when
`anchor_part` is unset: the **NaN-ignoring mean of all visible nodes** in
each instance (see [`generate_centroids`](../reference/sleap_nn/data/instance_centroids.md)).
This is project-wide convention as of v0.3 — earlier versions used the
**bounding-box midpoint**, which differs on asymmetric instances (long
tails, sprawled limbs).

If you trained a centroid model on the old bbox-midpoint convention, the GT
centroid targets for partial instances shift slightly. Re-training is
recommended if `anchor_part` was unset in your training config; otherwise
behavior is unchanged.

## Interaction with filtering, tracking, and metrics

### Filtering

All `FilterConfig` knobs apply to centroid-only outputs:

- **`min_instance_score`**: filters on the centroid confidence value.
- **`min_visible_nodes` / `min_visible_node_fraction`**: use with caution —
  centroid-only outputs always have `n_nodes - 1` NaN nodes per instance,
  so a threshold > 1 will drop every instance.
- **`overlapping` with `overlapping_method="iou"`**: works; bbox-IoU on
  point centroids is effectively an exact-duplicate filter.
- **`overlapping` with `overlapping_method="oks"`**: emits a
  `UserWarning` and falls back to IoU. OKS needs keypoints, which are
  NaN in centroid-only outputs.

### Tracking

Use `features="centroids"` and a centroid-compatible scoring method
(`scoring_method="euclidean_dist"` is the natural choice):

```python
from sleap_nn.inference.tracking import TrackerConfig

tracker_config = TrackerConfig(
    features="centroids",
    scoring_method="euclidean_dist",
    window_size=5,
    track_matching_method="hungarian",
)

predictor = Predictor.from_model_paths(
    ["models/centroid/"],
    tracker_config=tracker_config,
)
```

`features="keypoints"` will technically run but only sees the anchor node;
this is rarely what you want.

### Metrics

Standard sleap-io metrics that expect full skeletons (OKS, PCK) will compute
NaN for every non-anchor node. Centroid-centric metrics (instance count,
centroid localization error) work as expected.
