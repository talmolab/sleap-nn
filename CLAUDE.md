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
- Re-ID identity/embedding persistence: predicted animal identity and appearance embeddings persist into the `.slp` via the sleap-io `Identity`/`Embedding` data model (requires the sleap-io Identity+Embedding stack; sleap-io is git-pinned to `main` until a release ships it). (1) **Embeddings:** `sleap-nn predict` on an `embedding` model takes `--save_embeddings {none,slp,both}` (default `none` = today's sidecar `.h5` only); `slp`/`both` attach an `sio.Embedding` (`"reid"` space, L2-normalized, `source=<model dir>`) to each source detection (object-exact via `EmbeddingDataset.mask_idx_list[item_id]["mask_obj"]`) and write a `.slp`. Both pose `Instance` and mask `SegmentationMask` embeddings persist (mask-modality `owner_type=3` + `SegmentationMask.identity` landed in sleap-io#527). The embedding pathway stores **vectors only** — it does NOT fabricate `sio.Identity` from track/class names (a track name is not a global animal identity; existing identities pass through). (2) **Identity:** a multi_class model's class output is interpreted per its head config `class_output` ∈ `{"track"` (default), `"identity"`, `"category"}`. `"track"` (default) emits only the per-video `sio.Track` (classification-as-tracking; restores pre-existing behavior). `"identity"` ADDITIVELY assigns a canonical global `sio.Identity` + `identity_score` (the class probability) alongside the Track — for classes that enumerate **unique individuals** (named animals for re-ID); the per-class `uuid` is frozen at train time (`model_trainer.py`, into `training_config.yaml`; reused from name-matched GT `Identity`s else minted) and re-emitted at inference via `_multiclass_identities`. `"category"` (for shared types/roles, e.g. male/female) is NOT yet implemented (raises). Only set `"identity"` when each class is a distinct animal; a shared frozen uuid otherwise falsely claims all instances of a class are the same animal. The migration applies on the in-memory `predict()` path; the streaming `predict_to_disk` writer does not yet thread multi_class packaging (pre-existing). `Identity` (global, uuid-keyed) + `identity_score` mirror `Track`/`tracking_score`; GT-vs-predicted is the host class (`PredictedInstance`) + `identity_score is not None`.
