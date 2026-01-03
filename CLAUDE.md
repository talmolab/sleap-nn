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

### Configuration System

The project uses a hierarchical configuration system with three main sections:
- `data_config`: Data pipeline configuration
- `model_config`: Model architecture configuration  
- `trainer_config`: Training hyperparameters and Lightning configuration

Configurations are managed via Hydra and can be specified in YAML files (see `docs/config_*.yaml` examples).

### Key Entry Points

- Training: `sleap_nn/train.py` - Hydra-based training entry point
- Inference: `sleap_nn/predict.py` - Run inference on trained models
- CLI: `sleap_nn/cli.py` - Command-line interface (currently minimal)
- Evaluation: `sleap_nn/evaluation.py` - Model evaluation utilities
