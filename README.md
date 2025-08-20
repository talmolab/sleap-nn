# sleap-nn

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)

Neural network backend for training and inference for animal pose estimation.

This is the deep learning engine that powers [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses), providing neural network architectures for multi-instance animal pose estimation and tracking. Built on PyTorch, SLEAP-NN offers an end-to-end training workflow, supporting multiple model types (Single Instance, Top-Down, Bottom-Up, Multi-Class), and seamless integration with SLEAP's GUI and command-line tools.

## Documentation

**📚 [Documentation](https://nn.sleap.ai)** - Comprehensive guides and API reference

## Installation

**Prerequisites: Python 3.11**

### From PyPI

- **Create a mamba environment**
    ```bash
    mamba create -n sleap-nn python=3.11
    mamba activate sleap-nn
    ```

- **Install uv**
    ```bash
    pip install uv
    ```

- **Install sleap-nn dependencies**

    For Windows/Linux with NVIDIA GPU (CUDA 11.8):

    ```bash
    uv pip install sleap-nn[torch-cuda118]
    ```

    For Windows/Linux with NVIDIA GPU (CUDA 12.8):

    ```bash
    uv pip install sleap-nn[torch-cuda128]
    ```

    For macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):  
    Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
    ```bash
    uv pip install sleap-nn[torch-cpu]
    ```


### For development setup

1. **Clone the sleap-nn repo**

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

2. **Install [`uv`](https://github.com/astral-sh/uv) and development dependencies**  
   `uv` is a fast and modern package manager for `pyproject.toml`-based projects. Refer [installation docs](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

3. **Install sleap-nn dependencies based on your platform**\

   - Sync all dependencies based on your correct wheel using `uv sync`:
     - **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

      ```bash
      uv sync --extra dev --extra torch-cuda118
      ```

      - **Windows/Linux with NVIDIA GPU (CUDA 12.8):**

      ```bash
      uv sync --extra dev --extra torch-cuda128
      ```
     
     - **macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):** 
     Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
     ```bash
      uv sync --extra dev --extra torch-cpu
      ```

   You can find the correct wheel for your system at:\
   👉 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

4. **Run tests**  
   ```bash
   uv run pytest tests
   ```

5. **(Optional) Lint and format code**
   ```bash
   uv run black --check sleap_nn tests
   uv run ruff check sleap_nn/
   ```

## Quick Start

Let's start SLEAPiNNg !!! 🐭🐭

> For detailed information on setting up config, training/ inference workflows, please refer to our [docs](https://nn.sleap.ai).

#### 1. Set Up Your Configuration

Create a `config.yaml` file for your experiment.

> Use a sample config from [`docs/sample_configs`](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs).

#### 2. Train a model

> Download sample training data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp) and validation data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp) for quick experimentation.

```bash
sleap-nn-train --config-name config.yaml --config-dir configs/ "data_config.train_labels_path=[labels.pkg.slp]"
```

#### 3. Run inference on the trained model

To run inference:
```bash
sleap-nn-track --data-path video.mp4 --model-paths model_ckpt_dir/
```
