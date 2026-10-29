# Inference API (Python)

The `sleap_nn.inference` package is the programmatic entry point for running
predictions. It exposes two public symbols:

```python
from sleap_nn.inference import Predictor, predict
```

- [`predict(...)`](#one-liner-predict) — the "I just want predictions" function.
  Builds a `Predictor`, runs it, returns `sio.Labels`.
- [`Predictor`](#the-predictor-class) — the reusable orchestrator. Build it once,
  then call `.predict(...)`, `.predict_streaming(...)`, or `.predict_to_file(...)`
  as many times as you like with per-call overrides.

!!! tip "Just want a `.slp` from the command line?"
    Use the CLI instead — see [Running Inference](inference-guide.md). This guide
    is for embedding inference in Python code.

---

## One-liner: `predict`

```python
from sleap_nn.inference import predict

# Single model (single-instance or bottom-up) → sio.Labels
labels = predict("video.mp4", model_paths=["models/my_model/"])

# Top-down: pass two model paths (centroid + centered-instance, any order)
labels = predict(
    "video.mp4",
    model_paths=["models/centroid/", "models/centered_instance/"],
    device="cuda",
    output_path="predictions.slp",   # also writes the .slp to disk
)
```

`predict` returns a `sio.Labels`. When `output_path` is set it additionally
saves the labels there.

### Key parameters

`predict` accepts construction-time and prediction-time options in one call.
The most common:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `source` | Video path, `sio.Video`, `sio.Labels`, or a Provider | required |
| `model_paths` | List of model dirs — or a model's `best.ckpt` / `training_config.yaml` path, which resolves to its dir (1 for single/bottom-up, 2 for top-down) | `None` |
| `export_dir` | Exported ONNX/TRT dir (alternative to `model_paths`) | `None` |
| `runtime` | Exported-model runtime: `"auto"` (prefer TRT), `"onnx"`, `"tensorrt"` (ignored for checkpoints) | `"auto"` |
| `device` | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"` | `"auto"` |
| `batch_size` | Frames per batch | `4` |
| `frames` | Frame indices to predict (`None` = all) | `None` |
| `peak_threshold` | Override peak threshold for all stages | `None` |
| `centroid_threshold` | Override centroid-stage threshold (top-down) | `None` |
| `keypoint_threshold` | Override centered-instance threshold (top-down) | `None` |
| `max_instances` | Cap instances per frame | `None` |
| `centroid_only` | Force centroid-only output (see below) | `False` |
| `filter_config` | A [`FilterConfig`](#post-processing-filterconfig) | `None` |
| `tracker_config` | A [`TrackerConfig`](#tracking-trackerconfig) | `None` |
| `output_path` | Save the result to this `.slp` | `None` |
| `clean_empty_frames` | Drop frames with no instances | `False` |
| `progress_callback` | `(processed, total)` callback per batch | `None` |

!!! note "`model_paths` or `export_dir`, not both"
    Exactly one of `model_paths` or `export_dir` must be provided;
    passing both (or neither) raises `ValueError`.

---

## The `Predictor` class

Build a `Predictor` once and reuse it. It keeps no state across calls, so the
same instance is safe to run on multiple sources.

### `Predictor.from_model_paths`

```python
from sleap_nn.inference import Predictor

predictor = Predictor.from_model_paths(
    ["models/centroid/", "models/centered_instance/"],
    device="cuda",
    batch_size=8,
    peak_threshold=0.2,
)
```

Selected keyword arguments (all keyword-only):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | `"cpu"`, `"cuda"`, `"mps"`, `"cuda:N"` | `"cpu"` |
| `batch_size` | Default batch size for auto-built providers | `4` |
| `peak_threshold` | `float`, or `[centroid_thresh, keypoint_thresh]` for top-down | `0.2` |
| `integral_refinement` | `"integral"`, `"local"`, or `"none"` / `None` | `"integral"` |
| `integral_patch_size` | Refinement patch size | `5` |
| `max_instances` | Cap on instances per frame | `None` |
| `return_confmaps` | Keep confidence maps on `Outputs` | `False` |
| `anchor_part` | Override centroid anchor node name | `None` |
| `filter_config` | A `FilterConfig` | `None` |
| `tracker_config` | A `TrackerConfig` | `None` |
| `paf_workers` | CPU worker processes for bottom-up PAF grouping | `0` |
| `upload_batches` | Copy each `uint8` batch to the model's GPU before preprocessing (CUDA) | `False` |
| `memory_format` | Weight layout of every model: `"contiguous"` or `"channels_last"` (NVIDIA) | `"contiguous"` |
| `centroid_only` | Force centroid-only output | `False` |

The skeleton is resolved automatically from the training config, so
`make_labels=True` works without passing `skeleton=...`.

### `.predict`

Synchronous; loads everything into memory. Returns `sio.Labels` by default,
or a `List[Outputs]` when `make_labels=False`.

```python
# Default → sio.Labels
labels = predictor.predict("video.mp4")

# Per-call overrides (do not mutate the predictor)
labels = predictor.predict(
    "video.mp4",
    frames=[0, 1, 2, 3],
    peak_threshold=0.3,
    max_instances=3,
)

# Raw Outputs instead of Labels
outputs_list = predictor.predict("video.mp4", make_labels=False)
```

Common `predict` keyword arguments: `make_labels`, `frames`, `skeleton`,
`videos`, `clean_empty_frames`, `progress_callback`, `peak_threshold`,
`centroid_threshold`, `keypoint_threshold`, `max_instances`,
`integral_refinement`, `integral_patch_size`, `return_confmaps`,
`return_crops`. Each override defaults to keeping the predictor's setting;
`integral_refinement=None` turns refinement off for that call.

!!! note "Per-stage thresholds (top-down)"
    For top-down models, `peak_threshold` sets *both* stages; use
    `centroid_threshold` / `keypoint_threshold` to control each stage
    independently.

### `.predict_streaming`

Generator that yields one [`Outputs`](#the-outputs-dataclass) per batch.
Memory stays bounded — use this for long videos when you process results
incrementally.

```python
for outputs in predictor.predict_streaming("long_video.mp4"):
    arrays = outputs.numpy()
    # ... consume one batch at a time ...
```

!!! warning "Streaming + tracking"
    `tracker_config` is not supported on `predict_streaming` /
    `predict_to_file` — end-of-stream tracker cleanup needs the full frame
    list. Use `.predict()` (which buffers in memory) for tracked output, or
    [`Predictor.retrack`](#retracking-without-inference) afterwards.

### `.predict_to_file`

Disk-streaming write of a `.slp`. Memory stays O(`write_interval`).

```python
out_path = predictor.predict_to_file(
    "long_video.mp4",
    "predictions.slp",
    write_interval=500,   # LabeledFrames buffered before each flush
)
```

Returns the resolved destination path string.

### `.from_export_dir`

Build a `Predictor` from an exported ONNX/TensorRT directory (written by
`sleap_nn export`). The directory must contain `export_metadata.json` plus
`model.onnx` and/or `model.trt`.

```python
predictor = Predictor.from_export_dir(
    "exported_model/",
    runtime="auto",       # prefer TRT, fall back to ONNX
    device="cuda",
)
labels = predictor.predict("video.mp4")
```

### Retracking without inference

`Predictor.retrack` is a static method — it applies a tracker to existing
predictions without running any model.

```python
import sleap_io as sio
from sleap_nn.inference import Predictor
from sleap_nn.inference.tracking import TrackerConfig

labels = sio.load_slp("predictions.slp")
tracked = Predictor.retrack(
    labels,
    TrackerConfig(window_size=5),
    clean_empty_frames=False,
)
tracked.save("tracked.slp")
```

---

## The `Outputs` dataclass

Every layer/predictor produces `Outputs` — a structured container of tensors.
When `make_labels=False`, `.predict()` returns a `List[Outputs]` (one per
batch). Import it from `sleap_nn.inference.outputs`:

```python
from sleap_nn.inference.outputs import Outputs
```

Key tensor fields (shape convention: `B`=batch, `I`=max instances,
`N`=nodes, `C`=channels, `H`/`W`=spatial; `NaN` = missing):

| Field | Shape | Meaning |
|-------|-------|---------|
| `pred_keypoints` | `(B, I, N, 2)` | Keypoints in image `(x, y)` |
| `pred_peak_values` | `(B, I, N)` | Per-keypoint confidence |
| `pred_centroids` | `(B, I, 2)` | Centroids (top-down / centroid-only) |
| `pred_centroid_values` | `(B, I)` | Centroid confidence |
| `instance_scores` | `(B, I)` | Per-instance score |
| `pred_confmaps` | `(B, N, H, W)` | Confidence maps (opt-in; heavy) |
| `frame_indices` / `video_indices` | `(B,)` | Source frame / video index |

### Methods

```python
outputs = predictor.predict("video.mp4", make_labels=False)[0]

# Tensor management — each returns a NEW Outputs
outputs.to("cpu")        # move tensors to a device
outputs.cpu()            # shorthand for .to("cpu")
outputs.detach()         # detach autograd

# Inspect shapes
outputs.batch_size       # B
outputs.n_instances      # I
outputs.n_nodes          # N

# Convert all populated tensor fields to a dict of numpy arrays
arrays = outputs.numpy()
peaks = arrays["pred_keypoints"]      # np.ndarray (B, I, N, 2)
scores = arrays["pred_peak_values"]   # np.ndarray (B, I, N)

# Drop heavy intermediates + force CPU/detach → pickle-safe (for IPC)
light = outputs.slim()

# Convert directly to sio.Labels (single Outputs)
import sleap_io as sio
labels = outputs.to_labels(skeleton=predictor.skeleton)
```

`slim()` drops the heavy fields (`original_image`, `processed_image`, `crops`,
`pred_confmaps`, `pred_pafs`, `pred_class_maps`, `pred_paf_graph`) and is
guaranteed pickleable — use it before sending `Outputs` across a process
boundary.

---

## Post-processing: `FilterConfig`

Filters run between the raw model `Outputs` and the final `Labels`, in a fixed
cheap-to-expensive order: per-keypoint threshold → node-count → score →
overlap-NMS.

```python
from sleap_nn.inference import Predictor
from sleap_nn.inference.filters import FilterConfig

filter_config = FilterConfig(
    min_peak_value=0.1,            # NaN-out keypoints below this
    min_instance_score=0.3,        # drop low-score instances
    min_visible_nodes=2,           # drop instances with too few nodes
    overlapping=True,              # enable overlap-NMS (runs last)
    overlapping_method="iou",      # "iou" or "oks"
    overlapping_threshold=0.8,
)

predictor = Predictor.from_model_paths(
    ["models/bottomup/"],
    filter_config=filter_config,
)
labels = predictor.predict("video.mp4")
```

A default `FilterConfig()` is the no-op identity — every knob defaults to
`0` / `False`. Full field list:

| Field | Default | Effect |
|-------|---------|--------|
| `min_peak_value` | `0.0` | NaN-out per-keypoint scores below this |
| `min_instance_score` | `0.0` | Drop instances below this `instance_scores` |
| `min_mean_node_score` | `0.0` | Drop instances below this mean visible-node score |
| `min_visible_nodes` | `0` | Drop instances with fewer than N visible nodes |
| `min_visible_node_fraction` | `0.0` | Drop below this visible-node fraction |
| `overlapping` | `False` | Run greedy overlap-NMS |
| `overlapping_threshold` | `0.8` | Similarity above which the lower-scoring overlap is dropped |
| `overlapping_method` | `"iou"` | `"iou"` (bbox) or `"oks"` (keypoints) |

### Applying filters manually

`FilterPipeline` applies a `FilterConfig` to an `Outputs` directly:

```python
from sleap_nn.inference.filters import FilterConfig, FilterPipeline

pipeline = FilterPipeline(FilterConfig(min_instance_score=0.3))
filtered = pipeline(outputs)            # __call__ is sugar for .apply()

# One-off convenience
filtered = FilterPipeline.run(outputs, FilterConfig(min_instance_score=0.3))
```

---

## Tracking: `TrackerConfig`

`TrackerConfig` is a frozen value type. Attach it to a `Predictor` (tracking
runs after `to_labels`, so `make_labels=True` is required), or pass it to
`predict(...)`.

```python
from sleap_nn.inference import Predictor
from sleap_nn.inference.tracking import TrackerConfig

tracker_config = TrackerConfig(
    window_size=5,
    candidates_method="fixed_window",
    features="keypoints",            # or "centroids"
    scoring_method="oks",            # "oks", "euclidean_dist", ...
    track_matching_method="hungarian",
    max_tracks=None,
)
# Or a modern MOT tracker (see below): TrackerConfig(tracker="bytetrack")

predictor = Predictor.from_model_paths(
    ["models/centroid/", "models/centered_instance/"],
    tracker_config=tracker_config,
)
labels = predictor.predict("video.mp4")   # tracked
```

!!! note "`max_tracks` auto-selects `local_queues`"
    `max_tracks` is honored only by the `local_queues` candidate maker;
    `fixed_window` ignores it. Setting `max_tracks` (here or anywhere a
    `TrackerConfig`/`Tracker.from_config` is built) auto-switches the method to
    `local_queues` (logged at INFO), overriding `candidates_method="fixed_window"`.

### Tracking existing labels with `apply_tracking`

`apply_tracking` is the labels-in / labels-out tracking function. It builds a
fresh tracker for each video, tracks the frames in `(video, frame_idx)` order, and
works **in place**: it writes tracks onto the detections of the `Labels` you pass
and returns that same object. Pass `labels.copy()` to keep the input, for example
when comparing settings.

```python
import sleap_io as sio
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

labels = sio.load_slp("predictions.slp")
tracked = apply_tracking(labels, TrackerConfig(window_size=5))   # tracked is labels
tracked.save("tracked.slp")
```

`Predictor.retrack` wraps `apply_tracking` and adds optional
`clean_empty_frames` handling.

!!! note "post_connect_single_breaks"
    Setting `post_connect_single_breaks=True` requires
    `tracking_target_instance_count` to be set; otherwise `apply_tracking` raises
    `ValueError`. `max_tracks` is not a substitute.

### Modern MOT trackers

`TrackerConfig.tracker` selects the tracker. `"classic"` is the default: the
candidate-window tracker configured by the fields above. The other values are
`"sort"`, `"deepsort"`, `"strongsort"`, `"bytetrack"`, `"botsort"`, `"ocsort"`
and `"deepocsort"`, the [modern MOT trackers](tracking.md#modern-mot-trackers).
They read these fields:

| Field | Meaning | Default |
|---|---|---|
| `tracker` | `"classic"` or a MOT tracker name | `"classic"` |
| `track_buffer` | Steps a lost track is kept for re-association | `None` (preset: 30) |
| `min_hits` | Matches before a track is confirmed | `None` (preset) |
| `track_high_threshold` / `track_low_threshold` / `new_track_threshold` | Detection-score thresholds, in `[0, 1]` | `None` (preset) |
| `min_match_similarity` | Minimum geometric similarity for a first-stage match | `None` (preset) |
| `appearance_threshold` | Appearance trackers: `"auto"` (calibrated per video) or a minimum cosine in `[-1, 1]` | `None` (preset: `"auto"`) |
| `link_tracklets` | Identity linking after each video; also works with `"classic"` | `False` |
| `mot_params` | Overrides for any `sleap_nn.tracking.mot.params.MOTParams` field, including `"preset"` (`"animal"` / `"paper"`), `"similarity"`, `"normalize_scores"` and `"body_size"`. With `"classic"` and `link_tracklets=True`, only `"time_base"` and `"body_size"` (`CLASSIC_LINKING_MOT_PARAMS`), which its identity linking reads | `None` |
| `max_tracks` | Shared field. For a MOT tracker it is the known animal count: closed-set identity linking | `None` |
| `oks_stddev` | Shared field. OKS spread for the `oks` similarity | `None` (MOT: 0.1) |
| `features` | Shared field. Carrier to track: `"keypoints"` or `"masks"` | `"keypoints"` |

`None` means the tracker's preset value. `apply_tracking` validates the config
with `validate_mot_config` before tracking, and you can call it yourself. It
raises `ValueError` for a MOT tracker with any classic-only field away from its
default (`window_size`, `candidates_method`, `use_flow`, `use_kalman`,
`appearance_weight`, the cull / clean / connect options, ...). It also raises for
the classic tracker with any MOT-only field set (except `mot_params` holding only
`time_base` / `body_size` with `link_tracklets=True`). The properties `is_mot`,
`runs_linking`, `uses_appearance`, `requires_vectors` and
`follows_vector_carrier` describe what a config will do.

```python
import sleap_io as sio
from sleap_nn.inference.tracking import (
    TrackerConfig,
    apply_tracking,
    resolve_mot_params,
)

labels = sio.load_slp("predictions.slp")
config = TrackerConfig(
    tracker="ocsort",
    track_buffer=60,           # keep lost tracks for 60 steps
    max_tracks=4,              # 4 animals: closed-set identity linking
    mot_params={"similarity": "oks", "delta_t": 3},
)
print(resolve_mot_params(config))  # the resolved MOTParams (preset + overrides)

tracked = apply_tracking(labels, config)  # in place; returns `labels`
record = tracked.provenance["tracking"]   # resolved params, calibration, linking
entry = record["videos"][0]
print(entry["video"], entry["video_filename"])  # index into labels.videos, its file
print(entry["linking"])
tracked.save("tracked_ocsort.slp")
```

Each `record["videos"]` entry describes one tracked video: `"video"` is its index
in `labels.videos` (a video with no frames has no entry) and `"video_filename"` its
file name, next to the resolved `kalman`, `time_base`, `similarity`, `body_size`
(and `body_size_source`), `score_normalizer`, `calibration`, `appearance` and,
when linking ran, `linking`. The record describes the latest tracking run;
retracking replaces it.

!!! note "Set the MOT similarity with `mot_params`"
    `scoring_method="oks"` is `TrackerConfig`'s default value, so for a MOT
    tracker it counts as *unset*, and the similarity stays `"auto"`. Pass
    `mot_params={"similarity": "oks"}`, which also accepts `"iou"`,
    `"centroid"` and `"auto"`. An explicit non-default `scoring_method`
    (`"iou"`, `"euclidean_dist"`) also works.

The trackers can also be driven frame by frame without `apply_tracking`. A
tracker tracks **one video, in increasing frame order**, so build one per video
(this is what `apply_tracking` does):

```python
from sleap_nn.tracking.mot import MOTParams, build_mot_tracker

params = MOTParams.for_tracker("bytetrack")
offset = 0
for video in labels.videos:
    lfs = sorted(labels.find(video), key=lambda lf: lf.frame_idx)
    frames = [(lf.frame_idx, lf.predicted_instances) for lf in lfs]
    tracker = build_mot_tracker(params, carrier="pose")
    tracker.track_name_offset = offset   # names continue across videos
    tracker.prepare(frames)       # optional: calibration, time base, score scale
    for frame_idx, instances in frames:
        tracker.track(instances, frame_idx)   # sets .track / .tracking_score
    tracker.finalize()            # optional: identity linking, if enabled
    offset = tracker.next_track_name_offset

labels.tracks = []   # rebuild the track list from the new assignments
labels.update()
labels.reindex()
labels.save("tracked.slp")
```

`prepare()` and `finalize()` are optional, but `prepare()` needs every frame up
front, so a live stream skips it. Without `prepare()`:

- an `"auto"` appearance threshold disables appearance (there is no
  calibration);
- scores are not normalized per video (`normalize_scores` needs the
  calibration's score quantiles), so the thresholds apply to raw scores;
- the frame stride and time base are inferred from the frame indices seen so
  far, so tracking every k-th frame works;
- the body size of single-node (centroid) detections is estimated from the
  motion seen so far; pass `body_size=<px>` to `MOTParams.for_tracker` if you
  know it.

A frame index that repeats or goes backwards logs a warning: it means two videos
(or unsorted frames) are going through one tracker. `finalize()` also warns when
no track was ever confirmed.

---

## Real-time / numpy inference

Each layer exposes `.predict(image)` that accepts a raw `np.ndarray` (or
`torch.Tensor`) and returns an `Outputs` directly — no provider, no file I/O.
Build a `Predictor` and reach through to its `.layer` for the hot loop:

```python
import numpy as np
from sleap_nn.inference import Predictor

predictor = Predictor.from_model_paths(["models/single_instance/"], device="cuda")
layer = predictor.layer

# Accepted shapes: (H, W), (H, W, C), (C, H, W), (B, H, W, C), (B, C, H, W).
frame = np.zeros((512, 512, 1), dtype=np.uint8)   # one grayscale frame

outputs = layer.predict(frame)        # Outputs
keypoints = outputs.numpy()["pred_keypoints"]     # (B, I, N, 2)
```

This is the lowest-latency path for streaming a camera feed frame-by-frame:
the layer runs `preprocess → backend → postprocess` on whatever array you
hand it. Note this bypasses the predictor's `FilterPipeline` and tracking;
apply [`FilterPipeline`](#applying-filters-manually) / `apply_tracking`
yourself if needed.

---

## See also

- [Running Inference (CLI)](inference-guide.md) — command-line usage
- [Centroid-only inference](centroid-only-inference.md) — centroid models standalone
- [Inference Performance](inference-performance.md) — FP16, `torch.compile`, `paf_workers`
- [Tracking](tracking.md) — tracking concepts and parameters
- [Export](export.md) — ONNX / TensorRT for production
