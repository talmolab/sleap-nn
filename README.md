# sleap-nn

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-nn?label=PyPI)](https://pypi.org/project/sleap-nn)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-nn)

Neural network backend for training and inference for animal pose estimation.

This is the deep learning engine that powers [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses), providing neural network architectures for multi-instance animal pose estimation and tracking. Built on PyTorch, SLEAP-NN offers an end-to-end training workflow, supporting multiple model types (Single Instance, Top-Down, Bottom-Up, Multi-Class), and seamless integration with SLEAP's GUI and command-line tools.

## Documentation

**📚 [Documentation](https://nn.sleap.ai)** - Comprehensive guides and API reference

## Quick Start

Let's start SLEAPiNNg !!! 🐭🐭

> For detailed information on setting up config, training/ inference workflows, please refer to our [docs](https://nn.sleap.ai).

#### 1. Install uv
Install [`uv`](https://github.com/astral-sh/uv) first - a fast Python package manager:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Set Up Your Configuration

Create a `config.yaml` file for your experiment.

> Use a sample config from [`docs/sample_configs`](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs).

#### 3. Train a model

> Download sample training data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp) and validation data from [here](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp) for quick experimentation.

```bash
uvx sleap-nn[torch-cpu] train --config-name config.yaml --config-dir /path/to/config_dir/ "data_config.train_labels_path=[labels.pkg.slp]"
```

#### 4. Run inference on the trained model

To run inference:
```bash
uvx sleap-nn[torch-cpu] track --data-path video.mp4 --model-paths model_ckpt_dir/
```


### For development setup

1. **Clone the sleap-nn repo**

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

2. **Install [`uv`](https://github.com/astral-sh/uv)**
Install [`uv`](https://github.com/astral-sh/uv) first - a fast Python package manager:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
  

3. **Install sleap-nn dependencies based on your platform**\

- Sync all dependencies based on your correct wheel using `uv sync`. `uv sync` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, you must run commands with `uv run` (e.g., `uv run sleap-nn train ...` or `uv run pytest ...`).
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

4. **Run tests**  
   ```bash
   uv run pytest tests
   ```

5. **(Optional) Lint and format code**
   ```bash
   uv run black --check sleap_nn tests
   uv run ruff check sleap_nn/
   ```
