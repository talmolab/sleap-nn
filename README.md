# sleap-nn

[![CI](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-nn/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-nn/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/sleap-nn)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-nn?label=Latest)](https://github.com/talmolab/sleap-nn/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-nn?label=PyPI)](https://pypi.org/project/sleap-nn)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleap-nn)

Neural network backend for training and inference for animal pose estimation.

This is the deep learning engine that powers [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses), providing neural network architectures for multi-instance animal pose estimation and tracking. Built on PyTorch, SLEAP-NN offers an end-to-end training workflow, supporting multiple model types (Single Instance, Top-Down, Bottom-Up, Multi-Class), and seamless integration with SLEAP's GUI and command-line tools.

Need a quick start? Refer to our [Quick Start guide](https://nn.sleap.ai/latest/#quick-start) in the docs.

## Documentation

**📚 [Documentation](https://nn.sleap.ai)** - Comprehensive guides and API reference

## For development setup

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

> **Python 3.14 is not yet supported**
>
> `sleap-nn` currently supports **Python 3.11, 3.12, and 3.13**.  
> **Python 3.14 is not yet tested or supported.**  
> By default, `uv` will use your system-installed Python.  
> If you have Python 3.14 installed, you must specify the Python version (≤3.13) in the install command.  
>
> For example:
>
> ```bash
> uv sync --python 3.13 ...
> ```
> Replace `...` with the rest of your install command as needed.

- Sync all dependencies based on your correct wheel using `uv sync`. `uv sync` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, you must run commands with `uv run` (e.g., `uv run sleap-nn train ...` or `uv run pytest ...`).
   - **Windows/Linux with NVIDIA GPU (CUDA 13.0):**

   ```bash
   uv sync --extra torch-cuda130
   ```

   - **Windows/Linux with NVIDIA GPU (CUDA 12.8):**

   ```bash
   uv sync --extra torch-cuda128
   ```

   - **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

   ```bash
   uv sync --extra torch-cuda118
   ```

   - **macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):**
   Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
   ```bash
   uv sync --extra torch-cpu
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

> **Upgrading All Dependencies**
> To ensure you have the latest versions of all dependencies, use the `--upgrade` flag with `uv sync`:
> ```bash
> uv sync --upgrade
> ```
> This will upgrade all installed packages in your environment to the latest available versions compatible with your `pyproject.toml`.