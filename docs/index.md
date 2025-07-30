# SLEAP-NN

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![code](https://img.shields.io/github/stars/talmolab/sleap-nn)](https://github.com/talmolab/sleap-nn)
<!-- [![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
-->
sleap-nn is a PyTorch-based backend for animal pose estimation. It provides efficient neural network architectures for multi-instance pose tracking and serves as the deep learning engine for [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses).

## Features

- **Multiple Model Architectures**: Support for single-instance, top-down, and bottom-up approaches
- **Modern Backbones**: UNet, ConvNext, and Swin Transformer implementations
- **PyTorch Lightning**: Built on Lightning for efficient training and deployment
- **Flexible Configuration**: Hydra and Omegaconf-based configuration system
- **Multi-GPU Support**: Distributed training capabilities
- **Export Ready**: Easy model export for inference

## 🚀 Quick Start

### Installation

```bash
# Create and activate environment
mamba create -n sleap-nn-dev python=3.11
mamba activate sleap-nn-dev

# Install uv and dependencies
pip install uv
uv pip install -e ".[torch]"

# For GPU support (Windows/Linux with NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

Step-1 Set-up `config.yaml`:




Step-2 Train a model:
```bash
python -m sleap_nn.train --config-name config.yaml --config-path configs/
```

Step-3 Run inference:

To run inference on a topdown model:
```bash
python -m sleap_nn.predict --data-path video.mp4 --model-paths centroid/ --model-paths centered_instance/
```

## Architecture

sleap-nn follows a modular architecture with clear separation of concerns:

- **Data Pipeline**: Efficient data loading, augmentation, and preprocessing
- **Model Architectures**: Pluggable backbone and head modules
- **Training System**: Lightning-based training with custom callbacks
- **Inference Pipeline**: Optimized inference for different model types
- **Tracking**: Multi-instance tracking across frames using a flow-shift based tracker.

## Model Types

### Single Instance
For videos with exactly one animal per frame.

### Centered Instance
Crop-based approach for improved accuracy on single instances.

### Top-Down
Two-stage approach: detect centroids first, then predict instances.

### Bottom-Up
Direct multi-instance prediction using Part Affinity Fields (PAFs).

## Next Steps

- [Installation Guide](installation.md)
- [Configuration Guide](config.md)
- [Training Models](training.md)
- [API Reference](api/index.md)