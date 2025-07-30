# sleap-nn

A PyTorch-based neural network backend for animal pose estimation.

## Overview

sleap-nn provides high-performance implementations of neural network architectures for multi-instance pose tracking. It serves as the backend for [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses).

## Features

- **Multiple Model Architectures**: Support for single-instance, top-down, and bottom-up approaches
- **Modern Backbones**: UNet, ConvNext, and Swin Transformer implementations
- **PyTorch Lightning**: Built on Lightning for efficient training and deployment
- **Flexible Configuration**: Hydra-based configuration system
- **Multi-GPU Support**: Distributed training capabilities
- **Export Ready**: Easy model export for inference

## Quick Start

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

Train a model:
```bash
python -m sleap_nn.train --config-name config
```

Run inference:
```bash
python -m sleap_nn.predict --ckpt-path model.ckpt --data-path video.mp4
```

## Architecture

sleap-nn follows a modular architecture with clear separation of concerns:

- **Data Pipeline**: Efficient data loading, augmentation, and preprocessing
- **Model Architectures**: Pluggable backbone and head modules
- **Training System**: Lightning-based training with custom callbacks
- **Inference Pipeline**: Optimized inference for different model types
- **Tracking**: Multi-instance tracking across frames

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
- [Training Models](training.md)
- [Configuration Reference](configuration.md)
- [API Reference](api/index.md)