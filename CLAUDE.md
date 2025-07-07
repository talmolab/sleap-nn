# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Environment Setup
- GPU (Windows/Linux): `mamba env create -f environment.yml`
- CPU (Windows/Linux/Intel Mac): `mamba env create -f environment_cpu.yml`
- Apple Silicon (M1/M2 Mac): `mamba env create -f environment_osx-arm64.yml`
- Activate environment: `mamba activate sleap-nn`

## Architecture Overview

sleap-nn is a PyTorch-based neural network backend for animal pose estimation. The codebase follows a modular architecture:

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

### Model Types

The system supports multiple model architectures for pose estimation:
- Single Instance: One animal per frame
- Centered Instance: Crop-based single instance
- Centroid: Animal center detection
- Top-Down: Centroid → Instance detection
- Bottom-Up: Multi-instance with PAFs

Each model type has corresponding head modules, data processing, and inference pipelines.