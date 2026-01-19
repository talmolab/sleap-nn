# SLEAP-NN

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![code](https://img.shields.io/github/stars/talmolab/sleap-nn)](https://github.com/talmolab/sleap-nn)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-nn)
[![PyPI](https://img.shields.io/pypi/v/sleap-nn?label=PyPI)](https://pypi.org/project/sleap-nn)

**SLEAP-NN** is the deep learning engine that powers [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses), providing neural network architectures for multi-instance animal pose estimation and tracking. Built on PyTorch, SLEAP-NN offers an end-to-end training workflow, supporting multiple model types (Single Instance, Top-Down, Bottom-Up, Multi-Class), and seamless integration with SLEAP's GUI and command-line tools.

## ✨ Features 

- **End-to-end workflow**: Streamlined training and inference pipelines, including a multi-instance tracking worklow with a flow-shift-based tracker.
- **Multiple model architectures**: Support for single-instance, top-down, and bottom-up pose estimation models using a variety of backbones, including highly-customizable UNet, ConvNeXt, and Swin Transformer.
- **PyTorch Lightning integration**: Built on PyTorch Lightning for fast and scalable training, with support for multi-GPU and distributed training.
- **Flexible configuration**: Hydra and OmegaConf based config system to validate training parameters and enable reproducible experiments.

---


## 🚀 Quick Start

Let's start SLEAPiNNg !!! 🐭🐭

**Prerequisite: uv installation**

Install [`uv`](https://github.com/astral-sh/uv), a fast Python package manager for modern projects:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step - 1 : Set Up Your Configuration

Create a `config.yaml` file for your experiment. Use a sample config from [`docs/sample_configs`](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs).

### Step - 2 : Train a Model

We use `uvx` here which automatically installs sleap-nn from PyPI with all dependencies and runs the command in a single step. `uvx` automatically installs sleap-nn and runs your command inside a temporary virtual environment (venv). This means each run is fully isolated and leaves no trace on your system—perfect for trying out sleap-nn without any permanent installation. Check out [Installation docs](installation.md) for different installation options. 

> **Quick Start Data:** Download sample training data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp) and validation data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp) for quick experimentation.

!!! warning "Python 3.14 is not yet supported"
    `sleap-nn` currently supports **Python 3.11, 3.12, and 3.13**. **Python 3.14 is not yet tested or supported.** If you have Python 3.14 installed, you must specify the Python version in the install commands by adding `--python 3.13`.  
    For example:
    ```bash
    uvx --python 3.13  ...
    ```
    Replace `...` with the rest of your install command as needed.

=== "Windows/Linux (GPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --torch-backend auto sleap-nn train --config-name config.yaml --config-dir /path/to/config_dir/ "data_config.train_labels_path=[train.pkg.slp]" "data_config.val_labels_path=[val.pkg.slp]"
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --torch-backend cpu sleap-nn train --config-name config.yaml --config-dir /path/to/config_dir/ "data_config.train_labels_path=[train.pkg.slp]" "data_config.val_labels_path=[val.pkg.slp]"
    ```

=== "macOS"
    ```bash
    uvx sleap-nn[torch] train --config-name config.yaml --config-dir /path/to/config_dir/ "data_config.train_labels_path=[train.pkg.slp]" "data_config.val_labels_path=[val.pkg.slp]"
    ```

!!! info
    - The `--torch-backend auto` option automatically detects your GPU and installs the optimal PyTorch version. Requires uv 0.9.20+.
    - For manual selection, use `--torch-backend cu130` (CUDA 13), `--torch-backend cu128` (CUDA 12.8), or `--torch-backend cpu`.
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.


### Step - 3 : Run Inference on the Trained Model

To run inference:

=== "Windows/Linux (GPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --torch-backend auto sleap-nn track --data-path video.mp4 --model-paths model_ckpt_dir/
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --torch-backend cpu sleap-nn track --data-path video.mp4 --model-paths model_ckpt_dir/
    ```

=== "macOS"
    ```bash
    uvx sleap-nn[torch] track --data-path video.mp4 --model-paths model_ckpt_dir/
    ```

!!! info
    - The `--torch-backend auto` option automatically detects your GPU and installs the optimal PyTorch version. Requires uv 0.9.20+.
    - For manual selection, use `--torch-backend cu130` (CUDA 13), `--torch-backend cu128` (CUDA 12.8), or `--torch-backend cpu`.
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.


!!! warning "Model Paths"
    `--model-paths` should be set to `<config.trainer_config.config_dir>/<config.trainer_config.run_name>`. Make sure the model checkpoint directory contains both `best.ckpt` (or legacy sleap `best_model.h5` - only UNet backbone is supported) and `training_config.yaml` (or legacy sleap `training_config.json` - only UNet backbone is supported) files. The inference will fail without these files.

---

## 📚 Documentation Structure

- **[Installation Guide](installation.md)** - Complete installation instructions with different options (CPU, GPU, development)

#### **How-to Guides**
- **[Configuration Guide](config.md)** - Detailed explanation of all configuration parameters and how to set up your config file for training
- **[Training Guide](training.md)** - How to train models using the CLI or Python API and advanced training options
- **[Inference Guide](inference.md)** - How to run inference and tracking with CLI/ APIs and evaluate the models
- **[Export Guide](export.md)** - Export models to ONNX/TensorRT for high-performance inference
- **[CLI Reference](cli.md)** - Quick reference for all command-line options

#### **Tutorials**
- **[Example Notebooks](example_notebooks.md)** - Interactive marimo-based tutorial notebooks
- **[Colab Notebooks](colab_notebooks/index.md)** - Jupyter notebooks to run on Google Colab
- **[Step-by-Step Tutorial](step_by_step_tutorial.md)** - Comprehensive walkthrough of the entire workflow

#### **Additional Resources**

- **[Model Architectures](models.md)** — Explore supported model types, backbone networks, and architecture details.
- **[Core Components](core_components.md)** — Main building blocks of the sleap-nn pipeline.
- **[API Reference](api/index.md)** — Complete API documentation for all `sleap-nn` modules and functions.


---


## Get Help

If you encounter issues or have questions about SLEAP-NN:

**Report Bugs**:

Found a bug? Please create an issue on GitHub:
- [Create a new issue](https://github.com/talmolab/sleap-nn/issues/new)
- Include details about your environment, error messages, and steps to reproduce

**Start a Discussion**:

Have questions about usage, feature requests, or want to share your experience?
- [Start a discussion](https://github.com/talmolab/sleap-nn/discussions)

**Additional Resources**:

- [SLEAP Documentation](https://sleap.ai) - Main SLEAP documentation
- [SLEAP-NN GitHub Repository](https://github.com/talmolab/sleap-nn) - Source code and releases
- [SLEAP Community](https://github.com/talmolab/sleap/discussions) - General SLEAP discussions

---

## Next Steps

- [Installation Guide](installation.md)
- [Configuration Guide](config.md)
- [Training Models](training.md)
- [Running Inference / Tracking](inference.md)
- [Model Export](export.md)
- [API Reference](api/index.md)