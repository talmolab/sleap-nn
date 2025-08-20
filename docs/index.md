# SLEAP-NN

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![code](https://img.shields.io/github/stars/talmolab/sleap-nn)](https://github.com/talmolab/sleap-nn)
<!-- [![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
-->
**SLEAP-NN** is the deep learning engine that powers [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses), providing neural network architectures for multi-instance animal pose estimation and tracking. Built on PyTorch, SLEAP-NN offers an end-to-end training workflow, supporting multiple model types (Single Instance, Top-Down, Bottom-Up, Multi-Class), and seamless integration with SLEAP's GUI and command-line tools.

## ✨ Features 

- **End-to-end workflow**: Streamlined training and inference pipelines, including a multi-instance tracking worklow with a flow-shift-based tracker.
- **Multiple model architectures**: Support for single-instance, top-down, and bottom-up pose estimation models using a variety of backbones, including highly-customizable UNet, ConvNeXt, and Swin Transformer.
- **PyTorch Lightning integration**: Built on PyTorch Lightning for fast and scalable training, with support for multi-GPU and distributed training.
- **Flexible configuration**: Hydra and OmegaConf based config system to validate training parameters and enable reproducible experiments.

---


## 🚀 Quick Start

Let's start SLEAPiNNg !!! 🐭🐭

#### 1. Install `sleap-nn`

- **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

```bash
pip install sleap-nn[torch-cuda118]
```

- **Windows/Linux with NVIDIA GPU (CUDA 12.8):**

```bash
pip install sleap-nn[torch-cuda128]
```

- **macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):** 
Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
```bash
pip install sleap-nn[torch-cpu]
```

> Refer [Installation](installation.md) for more details on how to install sleap-nn package for your specific hardware.

#### 2. Set Up Your Configuration

Create a `config.yaml` file for your experiment.

> - Refer to the [Configuration Guide](config.md) for detailed options.
  - Or, use a sample config from [`docs/sample_configs`](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs).

#### 3. Train a model

> Download sample training data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp) and validation data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp) for quick experimentation.

```bash
sleap-nn-train --config-name config.yaml --config-dir configs/ "data_config.train_labels_path=[labels.pkg.slp]"
```
> For detailed information on training workflows, configuration options, and advanced usage, please refer to the [Training Guide](training.md).

#### 4. Run inference on the trained model

To run inference:
```bash
sleap-nn-track --data-path video.mp4 --model-paths model_ckpt_dir/
```
> More options for running inference and tracking workflows are available in the [Inference Guide](inference.md).

!!! note
    **Note:** For step-by-step tutorial on our entire workflow, check out [Step--by-Step guide](step_by_step_guide.md).

!!! note
    **Note:** For tutorials, sample notebooks, and instructions on how to use the interactive notebooks, please see [Example notebooks](example_notebooks.md).

---


## 🛠️ Core Components

SLEAP-NN provides a modular, PyTorch-based architecture:

### **Data Pipeline**

- Efficient loading of data from SLEAP label files
- Parallelized data loading using PyTorch's multiprocessing for high throughput
- Caching (memory/ disk) to accelerate repeated data access and minimize I/O bottlenecks

### **Model System**

- **Pluggable Design**: Easy to add new backbones/ head modules
- **Backbone Networks**: UNet, ConvNeXt, Swin Transformer  
    - See [Backbone Architectures](models.md#backbone-architectures) for more details.
- **Model Types**: Single Instance, Top-Down, Bottom-Up, Multi-Class (Supervised ID models) variants

> - **Single Instance**: Direct pose prediction for single animals

> - **Top-Down**: Two-stage (centroid → centered-instance) for multi-animal scenarios

> - **Bottom-Up**: Simultaneous keypoint detection and association using Part Affinity Fields (PAFs).

> - **Supervised ID or Multi-Class**: Pose estimation + ID assignment for multi-instance scenarios

> Explore detailed descriptions of all supported architectures in the [Model Types Guide](models.md).

### **Training Engine**

- PyTorch Lightning integration with custom callbacks
- In-built multi-GPU and distributed training support
- Experiment tracking with visualizers and WandB

### **Inference Pipeline**

- Optimized inference workflow for different model types
- Integration with SLEAP's labeling interface

### **Tracking System**

- Multi-instance tracking across frames
- Flow-shift based tracker for robust tracking


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
- [API Reference](api/index.md)