# CLI Reference

Complete command-line interface documentation.

!!! tip "Reading the tables"
    In the **Values** column:

    - **Flag** = No argument needed; presence enables the option (e.g., `--tracking`)
    - **`INT`** / **`FLOAT`** = Numeric value (e.g., `--batch_size 8`)
    - **`PATH`** = File or directory path
    - **Comma-separated values** = Choose one (e.g., `auto`, `cuda`, `cpu`)

---

## Commands

| Command | Description |
|---------|-------------|
| [`sleap-nn train`](#sleap-nn-train) | Train models |
| [`sleap-nn predict`](#sleap-nn-predict) | Run inference and tracking on videos/labels (checkpoints or exported models) — **recommended** |
| [`sleap-nn track`](#sleap-nn-track) | Run inference/tracking (legacy pipeline) |
| [`sleap-nn eval`](#sleap-nn-eval) | Evaluate predictions |
| [`sleap-nn eval-tracking`](#sleap-nn-eval-tracking) | Evaluate identity persistence of tracked predictions |
| [`sleap-nn export`](#sleap-nn-export) | Export to ONNX/TensorRT |
| [`sleap-nn config`](#sleap-nn-config) | Generate training configs (experimental) |
| [`sleap-nn info`](#sleap-nn-info) | Inspect trained models |
| [`sleap-nn system`](#sleap-nn-system) | System diagnostics |

---

## `sleap-nn train`

Train pose estimation models.

```bash
sleap-nn train --config CONFIG_PATH [OPTIONS] [OVERRIDES]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | | Path to config YAML file |
| `--config-dir` | `-d` | Directory containing config file |
| `--config-name` | `-c` | Config file name (without .yaml) |
| `--video-paths` | `-v` | Replace video paths (multiple allowed) |
| `--video-path-map` | | Map old path to new (OLD NEW) |
| `--prefix-map` | | Map path prefix (OLD_PREFIX NEW_PREFIX) |
| `--labels` | `-l` | Training labels `.slp`, replacing `data_config.train_labels_path` (multiple allowed) |
| `--val-labels` | | Validation labels `.slp`, replacing `data_config.val_labels_path` (multiple allowed) |
| `--resume` | | Resume a run from its folder or a `.ckpt` file (writes to a new `<run_name>-1` folder) |
| `--dry-run` | | Check the config and data (setup + one train/val batch on one batch of frames, with the config's settings), check the run can write to `ckpt_dir` (leaving nothing behind), print a summary |

### Examples

```bash
# Simple
sleap-nn train --config config.yaml

# With directory/name
sleap-nn train -d /configs -c my_config

# Override values
sleap-nn train --config config.yaml trainer_config.max_epochs=200

# Train a config on a different labels file
sleap-nn train config.yaml --labels labels.slp
sleap-nn train config.yaml -l train.slp --val-labels val.slp

# Remap video paths
sleap-nn train --config config.yaml --prefix-map /old/data /new/data

# Check a config before a long run
sleap-nn train --config config.yaml --dry-run

# Resume a run (and train it longer)
sleap-nn train --resume models/my_run trainer_config.max_epochs=200
```

---

## `sleap-nn predict`

Run inference on videos or labels files, and optionally track. This is the
recommended inference command: it runs trained checkpoints **and** exported
ONNX/TensorRT models (auto-detected), every model type, and all trackers.

```bash
sleap-nn predict --data_path INPUT --model_paths MODEL [OPTIONS]
```

!!! note "`sleap-nn infer` is deprecated"
    `sleap-nn infer` still works as a hidden alias for `predict` but emits a
    `DeprecationWarning`. Use `sleap-nn predict` instead.

Options with two spellings (e.g. `--fg_threshold` / `--fg-threshold`) accept
either. Boolean options shown as `--x` / `--no-x` take either form.

### Essential Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--data_path` | `-i` | Video or labels (`.slp`) file, or a remote URL (`http(s)://`, `s3://`, `gs://`, `gcs://`, `az://`, `abfs://`) that sleap-io can load | `PATH` / `URL` | Required |
| `--model_paths` | `-m` | Model dir, or its `best.ckpt` / `training_config.yaml`; an exported-model dir. Repeat for top-down | `PATH` | Required* |
| `--output_path` | `-o` | Output file path | `PATH` | `<input>.predictions.slp` |
| `--device` | `-d` | Compute device | `auto`, `cuda`, `cuda:0`, `cpu`, `mps` | `auto` |
| `--batch_size` | `-b` | Frames per batch | `INT` | `4` |
| `--max_instances` | `-n` | Max instances per frame (forward pass only) | `INT` | None |
| `--peak_threshold` | | Minimum confidence-map value for a peak (alias `--peak-conf-threshold`). For box detectors, the score threshold | `FLOAT` | `0.2` |
| `--tracking` | `-t` | Track the predictions | Flag | `false` |

*Not needed for track-only mode (a `.slp` input with `-t`) or for SAM masks
(`--mask_backend`).

Each `--model_paths` entry may be a model **directory**, or a path to that
model's `best.ckpt` or `training_config.yaml`/`.json` file — all three resolve
to the model directory and load `best.ckpt`. (A different checkpoint such as
`last.ckpt` still loads `best.ckpt` and warns; use `--backbone_ckpt_path` /
`--head_ckpt_path` for a specific checkpoint.)

The default output replaces the input's extension with `.predictions.slp`, next
to the input (`video.mp4` → `video.predictions.slp`, `labels.pkg.slp` →
`labels.pkg.predictions.slp`), as SLEAP's `sleap-track` names it. With
`--video_index`, the video's name is inserted (`project.slp` →
`project.<video>.predictions.slp`). A remote input writes to the current directory under the
URL's file name.

### Input and Frame Selection

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--frames` | Frames to predict: a comma list or **one** range (`0-100`). Ranges cannot be mixed with commas | `1,2,3` or `0-100` | All frames |
| `--video_index` | With a `.slp` input: predict on **every** frame of this video (or `--frames`, as source frame indices). `--exclude_user_labeled` / `--only_*` then filter those frames by the `.slp`'s annotations. A video embedded in a `.pkg.slp` is limited to its labeled frames | `INT` | Unset (predict on the `.slp`'s frames) |
| `--only_labeled_frames` | Only frames with user labels | Flag | `false` |
| `--only_suggested_frames` | Only unlabeled suggested frames | Flag | `false` |
| `--exclude_user_labeled` | Skip frames with user-labeled instances | Flag | `false` |
| `--only_predicted_frames` | Only frames that already have predictions | Flag | `false` |
| `--video_dataset` | Dataset name inside an HDF5 video | `STR` | None |
| `--video_input_format` | Axis order of an HDF5 video | `channels_last`, `channels_first` | `channels_last` |
| `--headers` | JSON object of HTTP headers for a remote `--data_path`, e.g. `'{"Authorization": "Bearer <token>"}'` | `JSON` | None |
| `--stream-mode` | How a remote `--data_path` is read (passed to sleap-io), e.g. `auto`, `stream`, `download` | `STR` | sleap-io default |

### Output Options

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--output_format` | Output file format; repeat to write several. `analysis_h5` writes one `.analysis.h5` per video | `slp`, `analysis_h5` | `slp` |
| `--no_empty_frames` | Drop frames with no predictions before saving | Flag | `false` |
| `--embed` | Embed images in the `.slp`: `true` (self-contained `.pkg.slp`-style file), `auto` (only if the input was an embedded `.pkg.slp`), `false` (reference the source media) | `auto`, `true`, `false` | `false` |
| `--restore_source_videos` | When a `.pkg.slp` input is not re-embedded, reference the original source videos instead of the `.pkg.slp` | Flag | `false` |
| `--save_embeddings` | Write each detection's re-ID vector into the `.slp` (see [Embedding tracking](../guides/embedding-tracking.md)) | `none`, `slp` | `none` |
| `--stream-to-file` | Stream predictions to this `.slp` path, dropping heavy tensors per batch (lower memory). Not combinable with `--tracking` or exported models | `PATH` | None |
| `--write-interval` | With `--stream-to-file`: frames buffered between conversions | `INT` | `500` |
| `--gui` | Print JSON progress lines (for the SLEAP GUI) instead of a progress bar | Flag | `false` |

### Preprocessing Overrides

Unset values come from the model's training config.

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--input_scale` | Scale applied to each frame | `FLOAT` | Training config |
| `--max_height` / `--max_width` | Size frames are padded to | `INT` | Training config |
| `--crop_size` | Top-down crop size | `INT` | Training config |
| `--ensure_rgb` / `--no-ensure_rgb` | Force 3-channel input | Flag | Training config |
| `--ensure_grayscale` / `--no-ensure_grayscale` | Force 1-channel input | Flag | Training config |
| `--anchor_part` | Node used as the centroid anchor | `STR` | Training config |
| `--backbone_ckpt_path` / `--head_ckpt_path` | Load the backbone / head weights from these `.ckpt` files | `PATH` | None |

### Peak Finding and Model-Specific Options

| Option | Applies to | Description | Values | Default |
|--------|-----------|-------------|--------|---------|
| `--integral_refinement` | Keypoint models | Sub-pixel peak refinement | `integral`, `local`, `none` | `integral` |
| `--integral_patch_size` | Keypoint models | Patch size for integral refinement | `INT` | `5` |
| `--centroid_peak_threshold` | Top-down | Peak threshold for the centroid stage only | `FLOAT` | `--peak_threshold` |
| `--centroid_only` | Top-down | Output centroids only, from a centroid + centered-instance pair (a lone centroid model is detected automatically). See [Centroid-only](../guides/centroid-only-inference.md) | Flag | `false` |
| `--centroid_output` | Centroid-only | `instance` (single-node instance), `centroid` (`sio.PredictedCentroid`), or `both` | `instance`, `centroid`, `both` | `instance` |
| `--max_edge_length_ratio` | Bottom-up | Longest expected edge, as a fraction of image size; longer connections are penalized | `FLOAT` | `0.25` |
| `--dist_penalty_weight` | Bottom-up | Weight of that distance penalty | `FLOAT` | `1.0` |
| `--n_points` | Bottom-up | Points sampled along each PAF line integral | `INT` | `10` |
| `--min_instance_peaks` | Bottom-up | Drop instances with fewer peaks than this | `FLOAT` | `0` |
| `--min_line_scores` | Bottom-up | Minimum line score (−1 to 1) to connect two peaks | `FLOAT` | `0.25` |
| `--bbox_nms_iou` | Box detectors | Class-wise IoU-NMS threshold; `1.0` disables. See [Bounding boxes](../guides/bbox-detection.md) | `FLOAT` | `0.45` (SSD), off (CenterNet) |

Segmentation models (see [Running Inference](../guides/inference-guide.md#bottom-up-segmentation),
[Semantic segmentation](../guides/semantic-segmentation.md)):

| Option | Applies to | Description | Values | Default |
|--------|-----------|-------------|--------|---------|
| `--fg_threshold` | Bottom-up / semantic | Foreground probability threshold | `FLOAT` | `0.5` |
| `--min_mask_area` | Bottom-up / semantic | Drop masks smaller than this many original-image pixels | `INT` | `0` |
| `--mask_output` | Bottom-up / semantic | `mask` (RLE masks), `polygon` (simplified ROIs only), or `both` | `mask`, `polygon`, `both` | `mask` |
| `--polygon_epsilon` | Bottom-up | Polygon simplification tolerance, as a fraction of contour perimeter | `FLOAT` | `0.01` |
| `--center_nms_kernel` | Bottom-up | Odd window for instance-center NMS | `INT` | `3` |
| `--mask_cleanup` / `--no-mask_cleanup` | Bottom-up | Keep each mask's largest component and fill holes | Flag | `false` |
| `--mask_cleanup_radius` | Bottom-up | With `--mask_cleanup`: open→close radius (output-stride px) | `INT` | `0` |
| `--full_res_masks` / `--no-full_res_masks` | Bottom-up | Store masks at full image resolution instead of output stride | Flag | `false` |
| `--distance_gate_alpha` | Bottom-up | Drop pixels whose predicted center is farther than `alpha·sqrt(area/π)` from their instance's center | `FLOAT` | Unset (off) |
| `--merge_fragments` / `--no-merge_fragments` | Bottom-up | Re-join over-segmented pieces of one animal | Flag | `false` |
| `--merge_method` | Bottom-up | Merge algorithm (with `--merge_fragments`) | `greedy`, `multicut` | `greedy` |
| `--merge_thresholds` | Bottom-up | Decreasing affinity thresholds, comma-separated | `FLOATS` | `0.85,0.6,0.4` |
| `--merge_w_valley` / `--merge_w_offset` | Bottom-up | Weights of the center-valley / offset-agreement affinity terms | `FLOAT` | `1.0` / `0.25` |
| `--merge_dilate` | Bottom-up | Dilation iterations for the contact test (min 1) | `INT` | `1` |

SAM-prompted masks from existing poses (no trained segmentation model; see
[SAM-Prompted Segmentation](../guides/sam-inference-segmentation.md)):

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--mask_backend` | Make instance masks from the input `.slp`'s poses. Do not pass `--model_paths` | `sam`, `sam3` | None (off) |
| `--sam_checkpoint` | SAM1 checkpoint (required for `sam`) | `PATH` | None |
| `--sam_model_type` | SAM1 model type | `vit_h`, `vit_l`, `vit_b` | `vit_h` |
| `--sam3_model_id` | Hugging Face id of the gated SAM3 model | `STR` | `facebook/sam3` |
| `--sam_prompt_mode` | Prompt from all keypoints, one anchor point, or the instance box | `pose`, `centroid`, `box` | `pose` |
| `--sam_anchor_ind` | Anchor node index for `--sam_prompt_mode centroid` | `INT` | None |
| `--sam_disjointify_masks` | Make a frame's masks non-overlapping | Flag | `false` |
| `--overlay_path` | Write a review overlay PNG | `PATH` | None |

### Filtering

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--filter_min_visible_nodes` | Min visible keypoints required | `INT` | `0` |
| `--filter_min_visible_node_fraction` | Min fraction of skeleton nodes visible | `FLOAT` (0.0-1.0) | `0.0` |
| `--filter_min_mean_node_score` | Min mean confidence across visible nodes | `FLOAT` (0.0-1.0) | `0.0` |
| `--filter_min_instance_score` | Min overall instance score | `FLOAT` (0.0-1.0) | `0.0` |
| `--filter_overlapping` | Remove duplicate instances (greedy NMS) | Flag | `false` |
| `--filter_overlapping_method` | Overlap measure | `iou`, `oks` | `iou` |
| `--filter_overlapping_threshold` | Similarity above which the lower-scoring instance is dropped | `FLOAT` (0.0-1.0) | `0.8` |
| `--filter_min_centroid_distance` | Centroid-only de-duplication radius in pixels (use instead of `--filter_overlapping` for centroids) | `FLOAT` | `0` (off) |

!!! note "Processing order"
    `--max_instances` (forward pass) → node count filter → confidence filter → overlap filter → tracking. In track-only mode, filtering is applied before tracking on existing predictions. See [Post-Processing Filters](../guides/post-processing-filters.md).

### Tracking

These are the options of the default `classic` tracker. See the [tracking
guide](../guides/tracking.md#tracking-parameters) for details.

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--tracking_window_size` | Frames (or, with `local_queues`, instances per track) to match against | `INT` | `5` |
| `--candidates_method` | Candidate selection method | `fixed_window`, `local_queues` | `fixed_window` (`local_queues` for `--features embeddings`, mask tracking, or with `--max_tracks`) |
| `--max_tracks` | Maximum track count (auto-selects `local_queues`) | `INT` | None |
| `--features` | Features for matching. Unset: `centroids` for single-node skeletons, `masks` for segmentation-only output | `keypoints`, `centroids`, `bboxes`, `masks`, `embeddings` | `keypoints` |
| `--scoring_method` | Similarity scoring method. Unset: `euclidean_dist` for single-node skeletons, `mask_iou` for masks, `cosine_sim` for `--features embeddings` | `oks`, `cosine_sim`, `iou`, `mask_iou`, `euclidean_dist` | `oks` |
| `--scoring_reduction` | How several scores for one track are combined | `mean`, `max`, `robust_quantile` | `mean` |
| `--robust_best_instance` | Quantile for a robust track score; `1` uses the max | `FLOAT` (0-1] | `1.0` |
| `--track_matching_method` | Assignment algorithm | `hungarian`, `greedy` | `hungarian` |
| `--min_new_track_points` | Don't start a track for an instance with fewer points | `INT` | `0` |
| `--min_match_points` | Minimum non-NaN points for a match candidate | `INT` | `0` |
| `--oks_stddev` | OKS keypoint spread | `FLOAT` | `0.025` |
| `--appearance_weight` | Blend re-ID appearance into the geometric score ([details](../guides/embedding-tracking.md#blending-appearance-with-geometry)) | `FLOAT` in `[0, 1]` | `0.0` |
| `--euclidean_scale` | Pixel scale for blending appearance into `euclidean_dist` | `FLOAT` | None |
| `--link_tracklets` | Identity linking after tracking each video ([details](../guides/tracking.md#identity-linking-long-term-identity-memory)) | Flag | `false` |
| `--use_flow` | Optical-flow motion model | Flag | `false` |
| `--of_img_scale` / `--of_window_size` / `--of_max_levels` | Optical-flow image scale, window size and pyramid levels | `FLOAT` / `INT` / `INT` | `1.0` / `21` / `3` |
| `--use_kalman` | Kalman-filter motion model (needs a target instance count) | Flag | `false` |
| `--kf_track_features` | What the Kalman filter tracks | `centroid`, `keypoints` | `centroid` |
| `--kf_init_frame_count` | Warm-up frames before the filters are fit | `INT` | `10` |
| `--kf_node_indices` | Node indices to track (comma-separated) | `STR` | All nodes |
| `--kf_reset_gap_size` | Missed frames after which a track's filter is reset | `INT` | `5` |
| `--tracking_target_instance_count` | Target instances per frame | `INT` | None |
| `--tracking_pre_cull_to_target` | Cull instances above the target count before tracking | `INT` | `0` |
| `--tracking_pre_cull_iou_threshold` | IoU for that pre-tracking cull | `FLOAT` | `0` |
| `--tracking_clean_instance_count` | Target instance count for cleaning after tracking | `INT` | `0` |
| `--tracking_clean_iou_threshold` | IoU for that post-tracking cull | `FLOAT` | `0` |
| `--post_connect_single_breaks` | Join a break where exactly one track is lost and one new track starts (needs `--tracking_target_instance_count`) | Flag | `false` |

!!! note "`--max_tracks` requires `local_queues`"
    With the classic tracker, `--max_tracks` is honored only by `local_queues`;
    `fixed_window` ignores it. Setting it auto-selects `candidates_method
    local_queues` (logged at INFO), so you do not need to set
    `--candidates_method` yourself.

#### Modern MOT trackers

`--tracker` selects one of the [modern MOT
trackers](../guides/tracking.md#modern-mot-trackers). These options apply only to
those trackers. Unset options take the tracker's preset value (`animal` preset
shown). Giving a classic-only option to a MOT tracker, or a MOT-only option to
`classic`, is an error; the one exception is `--mot_param time_base=...` /
`body_size=...` with `classic --link_tracklets`, which its identity linking reads.

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--tracker` | Tracking algorithm | `classic`, `sort`, `deepsort`, `strongsort`, `bytetrack`, `botsort`, `ocsort`, `deepocsort` | `classic` |
| `--track_buffer` | Steps a lost track is kept for re-association | `INT` | `30` (`paper` preset: `1` for `sort`) |
| `--min_hits` | Matches before a track is confirmed (earlier detections then labeled) | `INT` | `3` (`2` for `bytetrack`, `botsort`) |
| `--track_high_threshold` | Score above which a detection is high-confidence | `FLOAT` in `[0, 1]` | per tracker |
| `--track_low_threshold` | Detections scoring at or below this are ignored | `FLOAT` in `[0, 1]` | `0.1` |
| `--new_track_threshold` | Minimum score to start a track | `FLOAT` in `[0, 1]` | per tracker |
| `--min_match_similarity` | Minimum geometric similarity for a first-stage match | `FLOAT` in `[0, 1]` | `0.3` (`0.2` for `bytetrack`, `botsort`) |
| `--appearance_threshold` | Appearance trackers (`deepsort`, `strongsort`, `botsort`, `deepocsort`): minimum cosine, or `auto` (calibrated per video) | `auto`, `FLOAT` in `[-1, 1]` | `auto` |
| `--link_tracklets` | Identity linking after tracking each video | Flag | `false` |
| `--max_tracks` | Known number of animals: closed-set identity linking, at most N tracks per video | `INT` | `None` |
| `--mot_param` | Override any MOT parameter, e.g. `preset=paper`, `score_mode=none`, `use_byte=true`, `normalize_scores=false` (thresholds on raw scores instead of per-video normalized ones), `body_size=40` (animal size in px, e.g. for centroid-only models) (repeatable). With `classic`: only `time_base` / `body_size`, with `--link_tracklets` | `KEY=VALUE` | — |
| `--scoring_method` | Geometric similarity | `iou`, `oks`, `euclidean_dist` (centroid) | auto |
| `--oks_stddev` | OKS spread for `--scoring_method oks` (refused with any other similarity) | `FLOAT` | `0.1` |
| `--features` | Carrier to track (one the labels do not hold is an error) | `keypoints`, `masks` | auto |

### Exported Models and Performance

`predict` auto-detects exported models: if `--model_paths`/`-m` points to a
directory containing `model.onnx` or `model.trt`, it runs the exported-model
runtime instead of loading a trained checkpoint.

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--runtime` | Exported-model runtime; `auto` prefers TensorRT and falls back to ONNX (ignored for checkpoints) | `auto`, `onnx`, `tensorrt` | `auto` |
| `--upload_batches` | CUDA: copy each batch to the GPU before resizing/cropping (checkpoints only). See [Performance](../guides/inference-performance.md#batches-on-the-gpu-upload_batches) | Flag | `false` |
| `--memory_format` | Model weight layout; `channels_last` is faster on NVIDIA GPUs (checkpoints only) | `contiguous`, `channels_last` | `contiguous` |
| `--paf-workers` | CPU worker processes for bottom-up PAF grouping (`0` = in-process). `--cpu-workers` is a deprecated alias | `INT` | `0` |

### Examples

```bash
# Single model (single-instance or bottom-up)
sleap-nn predict -i video.mp4 -m models/bottomup/

# Top-down (two models)
sleap-nn predict -i video.mp4 -m models/centroid/ -m models/centered_instance/ -o predictions.slp

# With tracking
sleap-nn predict -i video.mp4 -m models/bottomup/ -t

# Track-only (no inference)
sleap-nn predict -i labels.slp -t

# Filter overlapping + tracking
sleap-nn predict -i video.mp4 -m models/bottomup/ --filter_overlapping -t

# Modern MOT tracker: retrack existing predictions with ByteTrack
sleap-nn predict -i labels.slp -t --tracker bytetrack

# Exported model directory (auto-detected) with TensorRT
sleap-nn predict -m exports/model -i video.mp4 --runtime tensorrt -o predictions.slp

# Also write a SLEAP Analysis HDF5 file
sleap-nn predict -i video.mp4 -m models/bottomup/ --output_format slp --output_format analysis_h5
```

---

## `sleap-nn track`

!!! warning "Legacy pipeline"
    `sleap-nn track` runs the older legacy inference pipeline (`sleap_nn.legacy_predict.run_inference`).
    It is still supported, but [`sleap-nn predict`](#sleap-nn-predict) is the
    recommended command.

```bash
sleap-nn track --data_path INPUT --model_paths MODEL [OPTIONS]
```

`track` accepts a subset of `predict`'s options, with the same meanings: the
essential, input/frame-selection, preprocessing, peak-finding, bottom-up PAF
grouping, filtering and classic-tracker options above, plus `--gui`. It does
**not** support:

- exported ONNX/TensorRT models (`--runtime`), `--upload_batches`, `--memory_format`, `--paf-workers`, `--stream-to-file`
- centroid-only output, segmentation, box detectors and SAM masks
- the modern MOT trackers (`--tracker` and its options), `--link_tracklets`, `--appearance_weight` / `--euclidean_scale`
- `--output_format`, `--embed`, `--restore_source_videos`, `--save_embeddings`, remote inputs (`--headers`, `--stream-mode`)

Differences in behavior:

- The default output drops the input's extension: `<input stem>.predictions.slp` (`video.mp4` → `video.predictions.slp`).
- The classic-tracker defaults are fixed (`--candidates_method fixed_window`, `--features keypoints`, `--scoring_method oks`) instead of being chosen from the model type.

---

## `sleap-nn eval`

Evaluate predictions against ground truth.

```bash
sleap-nn eval --ground_truth_path GT --predicted_path PRED [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--ground_truth_path` | `-g` | Ground truth labels | `PATH` | Required |
| `--predicted_path` | `-p` | Predicted labels | `PATH` | Required |
| `--save_metrics` | `-s` | Save metrics to .npz | `PATH` | None |
| `--oks_stddev` | | OKS standard deviation | `FLOAT` | `0.025` |
| `--oks_scale` | | Scale factor for OKS calculation | `FLOAT` | None |
| `--match_threshold` | | Instance matching threshold | `FLOAT` | `0.0` |
| `--match_method` | | Matcher: `oks` (full skeleton), `centroid` (point distance), `mask` (mask IoU), `semantic` (whole-frame foreground), `bbox` (COCO-style box AP), `auto` (from the prediction type) | `CHOICE` | `auto` |
| `--anchor_part` | | Centroid mode: GT node used as the ground-truth centroid | `STR` | None |
| `--centroid_method` | | Centroid mode: how GT centroids are computed (pass the model's own setting) | `center_of_mass`, `bbox_center`, `geometric_median`, `anchor` | anchor node if `--anchor_part`, else mean of visible nodes |
| `--centroid_fallback` | | Centroid mode: method used when the `--anchor_part` node is not visible | `center_of_mass`, `bbox_center`, `geometric_median` | `center_of_mass` |
| `--bbox_source` | | Box mode: where GT boxes come from (pass the model's `data_config.bbox_source`) | `auto`, `bboxes`, `instances`, `masks` | `auto` |
| `--bbox_padding` | | Box mode: padding (px) on GT boxes derived from poses or masks | `FLOAT` | `0.0` |
| `--user_labels_only` / `--no-user_labels_only` | | Only evaluate user-labeled | Flag | `True` |
| `--count_unmatched_as_fp` / `--no-count_unmatched_as_fp` | | Count unmatched predictions as false positives in `oks_voc.*` | Flag | off |

### Example

```bash
sleap-nn eval -g ground_truth.slp -p predictions.slp -s metrics.npz
```

---

## `sleap-nn eval-tracking`

Score whether tracks keep the right identity, against tracked ground truth. It
reports sleap-nn's identity metrics (ID switches, IDF1, MT/PT/ML, fragmentation,
purity, coverage). It also reports TrackEval-faithful CLEAR-MOT, IDF1 and HOTA,
saved under the `"mot"` key. See [Evaluating Identity
Persistence](../guides/tracking.md#evaluating-identity-persistence).

```bash
sleap-nn eval-tracking --ground_truth_path GT --predicted_path PRED [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--ground_truth_path` | `-g` | Tracked ground-truth labels | `PATH` | Required |
| `--predicted_path` | `-p` | Tracked predicted labels | `PATH` | Required |
| `--save_metrics` | `-s` | Save metrics to .json | `PATH` | None |
| `--carrier` | | What carries identity | `auto`, `pose`, `mask` | `auto` |
| `--match_threshold` | | Minimum OKS / mask IoU to match; maximum centroid distance (px) for single-node skeletons | `FLOAT` | `0.5` (`50` px) |
| `--mt_threshold` / `--ml_threshold` | | Coverage cuts for mostly-tracked / mostly-lost (legacy identity block; the `mot` block keeps TrackEval's fixed 0.8 / 0.2) | `FLOAT` | `0.8` / `0.2` |
| `--oks_stddev` | | OKS keypoint spread used to match poses | `FLOAT` > 0, finite | `0.025` |
| `--user_labels_only` / `--no-user_labels_only` | | Drop predicted detections from the ground-truth side | Flag | `False` |
| `--global_identity` / `--per_video_identity` | | Score track names as identities shared across videos | Flag | per video |

### Example

```bash
sleap-nn eval-tracking -g ground_truth.slp -p tracked.slp -s tracking_metrics.json
```

---

## `sleap-nn export`

Export models to ONNX/TensorRT.

```bash
sleap-nn export MODEL_PATH [MODEL_PATH_2] [-o OUTPUT_DIR] [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--output` | `-o` | Output directory | `PATH` | `<model dir>/exported` (top-down: `<centroid dir>/exported_topdown`) |
| `--format` | `-f` | Export format | `onnx`, `tensorrt`, `both` | `onnx` |
| `--precision` | | TensorRT precision | `fp32`, `fp16`, `tf32` | `fp16` |
| `--max-instances` | | Max instances per frame | `INT` | `20` |
| `--max-batch-size` | | Max batch size | `INT` | `8` |
| `--peak-threshold` | | Peak threshold baked into the ONNX graph | `FLOAT` | `0.2` |
| `--device` | | Device used for export | `STR` | `cpu` |
| `--verify` / `--no-verify` | | Validate the exported ONNX graph (`onnx.checker`) | Flag | `--verify` |
| `--opset-version` | | ONNX opset | `INT` | `17` |
| `--input-scale` | | Input scale (unset: training config) | `FLOAT` | None |
| `--input-height` / `--input-width` | | Fixed input size | `INT` | None |
| `--crop-size` | | Top-down crop size (unset: training config) | `INT` | None |
| `--max-peaks-per-node` | | Bottom-up: peak candidates kept per node | `INT` | `20` |
| `--n-line-points` | | Bottom-up: points sampled along each PAF line | `INT` | `10` |
| `--max-edge-length-ratio` | | Bottom-up: longest expected edge, as a fraction of image size | `FLOAT` | `0.25` |
| `--dist-penalty-weight` | | Bottom-up: weight of the edge-length penalty | `FLOAT` | `1.0` |
| `--workspace-size-gb` | | TensorRT builder workspace (GB) | `FLOAT` | exporter default |

### Examples

```bash
# ONNX only
sleap-nn export models/bottomup -o exports/ --format onnx

# Both formats
sleap-nn export models/bottomup -o exports/ --format both

# Top-down (two models)
sleap-nn export models/centroid models/instance -o exports/
```

!!! note "Running exported models"
    Inference on exported ONNX/TensorRT models is handled by `sleap-nn predict` — pass the export directory via `-m` and pick the runtime with `--runtime`. See [`sleap-nn predict`](#sleap-nn-predict) above.

---

## `sleap-nn config`

!!! warning "Experimental"
    This feature is experimental and may change in future releases.

Generate training configuration files interactively or automatically.

```bash
sleap-nn config SLP_PATH [OPTIONS]
```

### Options

| Option | Short | Description | Values | Default |
|--------|-------|-------------|--------|---------|
| `--output` | `-o` | Output path for config file(s) | `PATH` | `<slp_name>_config.yaml` |
| `--auto` | | Auto-generate without interactive TUI | Flag | `false` |
| `--pipeline` | | Model pipeline type | `single_instance`, `bottomup`, `topdown`, `multi_class_bottomup`, `multi_class_topdown`, `centroid`, `bbox_centernet`, `bbox_ssd`, `topdown_bbox_centernet`, `topdown_bbox_ssd` | Auto-detected |
| `--show-yaml` | | Print YAML to stdout instead of saving | Flag | `false` |

### Modes

#### Interactive TUI

Launch an interactive terminal UI to configure training:

```bash
sleap-nn config labels.slp
```

#### Auto Mode

Generate a config with smart defaults based on your data:

```bash
sleap-nn config labels.slp --auto -o config.yaml
```

### Examples

```bash
# Interactive configuration
sleap-nn config labels.slp

# Auto-generate with defaults
sleap-nn config labels.slp --auto

# Auto-generate with custom output
sleap-nn config labels.slp --auto -o my_config.yaml

# Auto-generate with overrides
sleap-nn config labels.slp --auto --pipeline bottomup

# Preview YAML without saving
sleap-nn config labels.slp --auto --show-yaml
```

### Output

For **top-down** pipelines, two config files are generated:

- `<name>_centroid.yaml` - Centroid model config
- `<name>_centered_instance.yaml` - Centered instance model config

The box-detector top-down pipelines (`topdown_bbox_centernet` / `topdown_bbox_ssd`)
write `<name>_bbox_centernet.yaml` (or `<name>_bbox_ssd.yaml`) and a
`<name>_centered_instance.yaml` with box-scaled crops.

For other pipelines, a single config file is generated.

See the [Config Generator Guide](../guides/config-generator.md) for detailed usage.

---

## `sleap-nn info`

Inspect a trained model directory or training config file. Shows model architecture, data settings, training configuration, training results, evaluation metrics, and files.

```bash
sleap-nn info PATH
```

`PATH` can be either:

- A **trained model directory** — shows full summary including training results, evaluation metrics, and files
- A **training config YAML file** — shows config summary only

### Output Sections

| Section | Shown when |
|---------|------------|
| **Model Info** | Always — model type, backbone, head, parameters, skeleton |
| **Data** | Always — training/val data paths, preprocessing, augmentations |
| **Training** | Always — optimizer, epochs, batch size, LR scheduler, early stopping |
| **Training Results** | Model directory with `training_log.csv` — epochs trained, losses |
| **Evaluation Metrics** | Model directory with metrics `.npz` files — mOKS, mAP, distances, PCK |
| **Files** | Model directory — file listing with sizes |

### Examples

```bash
# Inspect a trained model
sleap-nn info models/my_training_run/

# View config only
sleap-nn info training_config.yaml
```

---

## `sleap-nn system`

Display system information and GPU diagnostics.

```bash
sleap-nn system
```

Shows:
- Python version
- PyTorch version and build
- CUDA/cuDNN versions
- GPU details (name, memory, compute capability)
- Driver compatibility
- Installed package versions

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control visible GPUs |
| `WANDB_API_KEY` | WandB API key |

---

## Global Options

```bash
sleap-nn --version  # Show version
sleap-nn --help     # Show help
```
