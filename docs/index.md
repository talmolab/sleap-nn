# SLEAP-NN

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![code](https://img.shields.io/github/stars/talmolab/sleap-nn)](https://github.com/talmolab/sleap-nn)
<!-- [![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
-->
sleap-nn is a PyTorch-based backend for animal pose estimation. It provides efficient neural network architectures for multi-instance pose tracking and serves as the deep learning engine for [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses).

## ✨ Features 

- **End-to-end workflow**: Streamlined training and inference pipelines, including a multi-instance tracking worklow with a flow-shift-based tracker.
- **Multiple model architectures**: Support for single-instance, top-down, and bottom-up pose estimation models using a variety of backbones, including highly-customizable UNet, ConvNeXt, and Swin Transformer.
- **PyTorch Lightning integration**: Built on PyTorch Lightning for fast and scalable training, with support for multi-GPU and distributed training.
- **Flexible configuration**: Hydra and OmegaConf based config system to validate training parameters and enable reproducible experiments.
- **Built-in wandb support**: Integrated Weights & Biases (wandb) logging for efficient experiment management and visualization.

---


## 🚀 Quick Start

#### 1. Install `sleap-nn`

```bash
# Create and activate environment
mamba create -n sleap-nn-dev python=3.11
mamba activate sleap-nn-dev
```

```bash
# Install uv and dependencies
pip install uv
uv pip install -e ".[torch]"
```

```bash
# For GPU support (Windows/Linux with NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Refer [Installation](installation.md) for more details on how to install sleap-nn package for your specific hardware.

#### 2. Set Up Your Configuration

Create a `config.yaml` file for your experiment.

> - Refer to the [Configuration Guide](config.md) for detailed options.
  - Or, use a sample config from [`docs/sample_configs`](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs).

#### 3. Train a model

```bash
python -m sleap_nn.train --config-name config.yaml --config-path configs/
```
> For more details on training, refer to the [Training Guide](training.md).

#### 4. Run inference on the trained model

To run inference on a topdown model:
```bash
python -m sleap_nn.predict --data-path video.mp4 --model-paths centroid/ --model-paths centered_instance/
```
> For more details on running inference + tracking, refer to the [Inference Guide](inference.md).

---


## Architecture

sleap-nn follows a modular architecture with clear separation of concerns:

- **Data Pipeline**: Efficient data loading, augmentation, and preprocessing
- **Model Architectures**: Pluggable backbone and head modules to support different backbone architectures and different pose estimation model types
- **Training System**: Lightning-based training with custom callbacks
- **Inference Pipeline**: Optimized inference for different model types
- **Tracking**: Multi-instance tracking across frames using a flow-shift based tracker

---


## Next Steps

- [Installation Guide](installation.md)
- [Configuration Guide](config.md)
- [Training Models](training.md)
- [Running Inference / Tracking](inference.md)
- [API Reference](api/index.md)