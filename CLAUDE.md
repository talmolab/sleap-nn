# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

sleap-nn is a PyTorch-based neural network backend for animal pose estimation.

## Notes
- We use `uv` for environment management and packaging.
- We use `sleap-io` (https://github.com/talmolab/sleap-io) as the I/O backend for `.slp` files, which are HDF5-based containers for labels and sometimes embedded image data (typically ending with `.pkg.slp`). Refer to https://raw.githubusercontent.com/talmolab/sleap-io/refs/heads/main/docs/examples.md for usage examples when you need to do I/O with SLP files.
- This package (`sleap-nn`) is used as the neural network training backend to SLEAP, which is primarily used through its frontend package `sleap` (https://github.com/talmolab/sleap).

## Common Development Commands

### Testing
- Run all tests: `pytest tests`
- Run specific test file: `pytest tests/path/to/test_file.py`
- Run with coverage: `pytest --cov=sleap_nn --cov-report=xml tests/`
- Run tests with duration info: `pytest --durations=-1 tests/`

### Code Quality
- Format code: `black sleap_nn tests`
- Check formatting: `black --check sleap_nn tests`
- Run linter: `ruff check sleap_nn/`

## Architecture Overview

The codebase follows a modular architecture:

### Core Components

1. **Model Architecture** (`sleap_nn/architectures/`)
   - Backbone networks: UNet, ConvNext, SwinT (via `model.py:get_backbone`)
   - Head modules for different tasks: confidence maps, centroids, PAFs, etc.
   - Model configuration via Hydra/OmegaConf

2. **Data Pipeline** (`sleap_nn/data/`)
   - Providers for reading SLEAP files (`providers.py`)
   - Data augmentation, normalization, resizing
   - Confidence map and edge map generation
   - Instance cropping and centroid computation

3. **Training System** (`sleap_nn/training/`)
   - Lightning-based training modules (`lightning_modules.py`)
   - ModelTrainer class for orchestrating training (`model_trainer.py`)
   - Custom losses and callbacks
   - Configuration via `TrainingJobConfig`

4. **Inference Pipeline** (`sleap_nn/inference/`)
   - Different predictors for each model type (single instance, top-down, bottom-up)
   - Peak finding and PAF grouping for multi-instance
   - Unified prediction interface via `predictors.py`

5. **Tracking** (`sleap_nn/tracking/`)
   - Instance tracking across frames
   - Candidate generation with fixed windows and local queues
   - Candidate-update motion models: optical-flow (`FlowShiftTracker`, `--use_flow`) and
     Kalman filter (`KalmanShiftTracker`, `--use_kalman`; requires a target instance count
     and `pykalman`). Both subclass `Tracker` and override only `update_candidates`.

### Configuration System

The project uses a hierarchical configuration system with three main sections:
- `data_config`: Data pipeline configuration
- `model_config`: Model architecture configuration  
- `trainer_config`: Training hyperparameters and Lightning configuration

Configurations are managed via Hydra and can be specified in YAML files (see `docs/config_*.yaml` examples).

### Key Entry Points

- Training: `sleap_nn/train.py` - Hydra-based training entry point
- Inference: `sleap_nn/legacy_predict.py` - Run inference on trained models
- CLI: `sleap_nn/cli.py` - Command-line interface (currently minimal)
- Evaluation: `sleap_nn/evaluation.py` - Model evaluation utilities
- Centroid-only models: first-class support — train a lone centroid head, `sleap-nn predict` auto-detects a single centroid directory (output collapses to a single-node `'centroid'` skeleton), distance-based eval (`--match_method centroid`), and standalone ONNX/TensorRT export. See `docs/guides/centroid-only-inference.md`.
- Top-down (crop-centered) segmentation: model type `centered_instance_segmentation` — a lone `SegmentationHead` on a centroid crop predicting the centered instance's binary mask. `sleap-nn predict` composes a `centroid` dir + the seg dir (or the seg dir alone via a GT-centroid fallback) and emits `sio.PredictedSegmentationMask` on `frame.masks` at full-frame offsets; mask-IoU eval (`--match_method mask`). Training reads per-instance masks from `lf.masks` (linked via `mask.instance`); geometric aug (rotation/scale/translate/flip) co-transforms the GT masks with the image (shared affine matrix, nearest-neighbor; erase/mixup stay image-only), training viz overlays the GT mask, and `trainer_config.eval.enabled` adds instance-level mask-IoU metrics via `SegmentationEvaluationCallback`. Same augmentation/viz/eval parity holds for `bottomup_segmentation`. See `docs/guides/topdown-segmentation.md`.
- Whole-frame semantic segmentation: model type `semantic_segmentation` — a lone `SegmentationHead` on the **whole frame** (no crop, no center/offset, no grouping), the group-free sibling of `bottomup_segmentation` and whole-frame analog of `centered_instance_segmentation`. The head predicts one binary foreground/background map (union of `lf.masks`, `mask.instance` ignored); `sleap-nn predict` auto-detects a lone semantic dir and emits **one instance-less** `sio.PredictedSegmentationMask` (`instance=None`) per frame. Decoding thresholds the foreground at `fg_threshold` → one mask/frame (no `group_instances_from_offsets`). Tiling-compatible (`TiledSemanticSegmentationLayer` stitches a 1-channel fg canvas). **Load-bearing:** the module `forward` returns `{"SegmentationHead": sigmoid(logits)}` (BottomUp-style probabilities, NOT the TopDown template's bare logits) so the whole-frame layer + tiled Gaussian averaging see probabilities; train/val steps use raw logits via `self.model()`. Matching-free eval: `--match_method semantic` (whole-frame fg IoU/clDice/boundary-IoU, no Hungarian) + a `foreground=True` `SegmentationEvaluationCallback` mode. See `docs/guides/semantic-segmentation.md`.
- Tiling (high-res / small-object frames): sliding-window tiling for 4K++ frames where objects are small — cut each frame into overlapping square tiles, run per-tile at native resolution, stitch. **Explicit opt-in** via `data_config.preprocessing.tiling` (`TilingConfig`); everything is inert/byte-identical when off. Geometry (`tile_size`/`overlap`) is auto-sized from labels at train setup, **written into the model config, and parity-checked at inference** (scale parity is a precondition — you cannot tile an existing whole-frame-trained model). Guards (`check_tiling`): pretrained-encoder backbones and `ClassVectorsHead`/`multi_class_topdown` are unsupported; only implemented model types are allowed (else a clear error — no silent no-op). Training uses foreground-aware tile sampling + a frame-grouped block sampler + per-worker frame LRU; inference (`TiledLayer`) stitches per-tile confmaps into one Gaussian-weighted ACC/CNT canvas then decodes once. **Status: Phase A (`single_instance`) is complete** end-to-end (local branch stack `tiling/01-foundations`..`tiling/04-viz-docs`, ~180 tests). `bottomup_segmentation` tiling is Phase C (in progress on `tiling/05-bottomup-seg`). See `docs/guides/tiling.md`, the design/decisions in `scratch/2026-07-03-tiling-highres-smallobjects/` (`03-design`, `04-decisions`, `05-phase0-A-spec`), and the deferred-corners list in `06-phase-c-bottomup-seg.md`.
- Re-ID identity/embedding persistence: predicted animal identity and appearance embeddings persist into the `.slp` via the **simplified** sleap-io `Identity`/`Embedding` data model (sleap-io #535/#536; sleap-io is git-pinned to `main` until a release ships it). Each detection modality carries a single `identity_embedding: sio.Embedding | None` slot (bare value object wrapping just a `vector` — no space-name/normalized/source metadata) plus `identity`/`identity_score`. (1) **Embeddings:** `sleap-nn predict` on an `embedding` model takes `--save_embeddings {none,slp}` (default `none`); `slp` sets `det.identity_embedding = sio.Embedding(vector)` on each source detection (object-exact via `EmbeddingDataset.mask_idx_list[item_id]["mask_obj"]`) and writes the `.slp` **with `save_embedding_vectors=True`** (the #536 default flipped to `False`, so every persist call passes the flag explicitly; identity/Track links persist regardless). The shared embed kernel is `inference/embedding.py::embed_labels`; `predict_embeddings_to_slp` wraps it. Both pose `Instance` and mask `SegmentationMask` embeddings persist (mask-modality `owner_type=3`). The embedding pathway stores **vectors only** — it does NOT fabricate `sio.Identity` from track/class names (existing identities pass through). (2) **Identity:** a multi_class model's class output is interpreted per its head config `class_output` ∈ `{"track"` (default), `"identity"`, `"category"}`. `"track"` (default) emits only the per-video `sio.Track`. `"identity"` ADDITIVELY assigns a canonical global `sio.Identity(name=<class name>)` + `identity_score` (the class probability) alongside the Track — for classes that enumerate **unique individuals**. The simplified `Identity` matches by **name** (#535 dropped `uuid`), so the class NAME is the canonical cross-file/cross-retrain key: no per-class uuid is minted (the old train-time freeze is deleted; `_multiclass_identities` builds `sio.Identity(name=...)`, dedup by name; the vestigial `class_uuids` config field is accepted-but-ignored for old configs). `"category"` raises `NotImplementedError` (categories removed in #535). `Identity` (global) + `identity_score` mirror `Track`/`tracking_score`; GT-vs-predicted is the host class (`PredictedInstance`) + `identity_score is not None`.
- Embedding (appearance / re-ID) tracking: the core `Tracker` gains a `features="embeddings"` mode — an alternative to pose→`oks`, it reads each detection's appearance vector from the single `identity_embedding` slot (`get_embedding` → `inst.identity_embedding.vector`, works on pose `PredictedInstance` *and* mask `PredictedSegmentationMask`) and scores candidates by `cosine_sim` (hardened: `None`/zero-norm/shape-mismatch → `NaN` → no match). Registered in `Tracker._feature_methods`; `apply_tracking` (`inference/tracking.py`) auto-pairs `embeddings`→`cosine_sim`, skips the single-node/mask auto-resolution that would clobber it, validates the labels actually carry embeddings (`_embedding_carriers` scans `identity_embedding`), and rejects motion models (`--use_flow`/`--use_kalman`) + (mask-carrier) pose cull/clean/connect. Three CLI workflows: **(WF1)** retrack a saved-embeddings `.slp` — `sleap-nn predict -i embedded.slp -t --features embeddings` (track-only path, torch-free). **(WF2)** embed + track from a detections `.slp` — `sleap-nn predict -m <embedding_model> -i detections.slp -t`: `predict_embeddings_to_slp(tracker_config=…)` embeds EVERY detection (`EmbeddingDataset(include_untracked=True)` enumerates untracked pose/mask, `group_id=0` placeholder), attaches vectors, runs `apply_tracking`, writes a tracked `.slp`; `--save_embeddings {none,slp}` controls whether vectors persist in it (`none`=tracks only, stripped via `_strip_embeddings` + `save_embedding_vectors=False`). **(WF3)** fused detect+embed+track from a video — `sleap-nn predict -m <centroid> [-m <centered_instance>] -m <embedding> -i video.mp4 -t`: `cli._run_embeddings` splits detection vs embedding dirs (`_is_embedding_model`), runs the detection stack to a temp `.slp`, then runs WF2 on it. Emits `sio.Track` only (no global `Identity`). Guide: `docs/guides/embedding-tracking.md`.
