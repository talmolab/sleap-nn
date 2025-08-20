# Installation

## Prerequisites

- Python 3.11
- [Miniforge](https://github.com/conda-forge/miniforge) (optional, recommended for isolated Python environments with fast dependency resolution)
- [Mamba](https://mamba.readthedocs.io/) (optional, included with Miniforge, recommended for faster dependency resolution)

## Install sleap-nn

### 1. (Optional) Install Miniforge and create environment

We recommend using [Miniforge](https://github.com/conda-forge/miniforge) for an isolated Python environment with fast dependency resolution. (You may skip this step if you already have a suitable Python environment.)

To create and activate development environment

```bash
mamba create -n sleap-nn-dev python=3.11
mamba activate sleap-nn-dev
```

### 2. Install `uv` and Development Dependencies

[`uv`](https://github.com/astral-sh/uv) is a fast and modern package manager for `pyproject.toml`-based projects.

```bash
pip install uv
uv pip install -e ".[dev]"
```

### 3. Install PyTorch

You can either:

#### Option A: Install with Optional Dependencies (Recommended)

```bash
uv pip install -e ".[torch]"
```

This installs the default builds of `torch` and `torchvision` via PyPI for your OS. By default, this means:

- **Windows:** CPU-only build.

- **Linux:** CUDA-enabled (GPU) build (if a compatible NVIDIA GPU is detected).

- **macOS:** CPU-only build (with Metal backend support for Apple Silicon in recent versions)
If you need a different build (e.g., GPU support on Windows, or CPU-only on Linux), see the manual installation options below.

#### Option B: Manual Installation for Specific Platforms

Install the correct wheel for your system using PyTorch's index URL:

**Windows/Linux with NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**macOS with Apple Silicon (M1, M2, M3, M4):**
You don't need to do anything if you used the `[torch]` optional dependency or default PyPI install—the default wheels now include Metal backend support for Apple GPUs.

**CPU-only (no GPU or unsupported GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

You can find the correct wheel for your system at:
👉 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

### 4. Verify Installation

Test your installation:

```bash
# Run tests
pytest tests

# Optional: Lint and format code
black --check sleap_nn tests
ruff check sleap_nn/
```

----

## 🚀 Development Setup

1. **Install [`uv`](https://github.com/astral-sh/uv) and development dependencies**  
   `uv` is a fast and modern package manager for `pyproject.toml`-based projects. Refer [installation docs](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

2. **Install sleap-nn dependencies based on your platform**\

   - Sync all dependencies based on your correct wheel using `uv sync`:
     - **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

      ```bash
      uv sync --extra dev --extra torch-cu118
      ```

      - **Windows/Linux with NVIDIA GPU (CUDA 12.8):**

      ```bash
      uv sync --extra dev --extra torch-cu128
      ```
     
     - **macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):** 
     Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
     ```bash
      uv sync --extra dev --extra torch-cpu
      ```

   You can find the correct wheel for your system at:\
   👉 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

3. **Run tests**  
   ```bash
   uv run pytest tests
   ```

4. **(Optional) Lint and format code**
   ```bash
   uv run black --check sleap_nn tests
   uv run ruff check sleap_nn/
   ```

---


## Troubleshooting

### Import Errors

If you get import errors:

1. Ensure you've activated the conda environment: `mamba activate sleap-nn-dev`
2. Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
3. Try reinstalling with torch extras: `uv pip install -e ".[torch]"`

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your NVIDIA drivers are up to date
2. Check CUDA compatibility with PyTorch version
3. Try the CPU-only installation as a fallback


## Next Steps

- [Step-by-step guide on training models](step_by_step_guide.md)
- [Configuration Guide](config.md)
- [Training models](training.md)