# Installation

**Prerequisites:** Python 3.11+ (required for all installation methods)

!!! tip "Choose Your Installation Method"
    - **[Installation with uvx](#installation-with-uvx)**: Use `uvx` for one-off commands (no installation needed)
    - **[Installation with pip](#installation-with-pip)**: Use `pip` for persistent installation. (Recommended to use with a conda env)
    - **[Installation from source](#development-setup-with-uv)**: Use `uv sync` to install from source (for developmental purposes)

---

## Installation with uvx

`uvx` automatically installs sleap-nn and runs your command inside a temporary virtual environment (venv). This means each run is fully isolated and leaves no trace on your system—perfect for trying out sleap-nn without any permanent installation.

!!! warning "First Time Setup"
    Install [`uv`](https://github.com/astral-sh/uv) first - a fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Platform-Specific Commands

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    uvx "sleap-nn[torch-cuda118]" train --config-name myconfig --config-dir configs/
    uvx "sleap-nn[torch-cuda118]" track --data_path video.mp4 --model_paths models/
    ```

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uvx "sleap-nn[torch-cuda128]" train --config-name myconfig --config-dir configs/
    uvx "sleap-nn[torch-cuda128]" track --data_path video.mp4 --model_paths models/
    ```

=== "macOS/CPU Only"
    ```bash
    uvx "sleap-nn[torch-cpu]" train --config-name myconfig --config-dir configs/
    uvx "sleap-nn[torch-cpu]" track --data_path video.mp4 --model_paths models/
    ```

!!! tip "How uvx Works"
    - **Automatic Installation**: Downloads and installs sleap-nn with dependencies
    - **Isolated Environment**: Each command runs in a clean, temporary environment
    - **No Conflicts**: Won't interfere with your existing Python packages
    - **Uses recent pkgs**: Uses the latest version from PyPI

!!! note "Performance Note"
    `uvx` downloads packages each time, so it's slower than persistent installation. Use `pip` or `uv sync` for regular use.

---

## Installation with pip

For regular use, install sleap-nn permanently on your system.

### Platform-Specific Installation

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    pip install sleap-nn[torch-cuda118]
    ```

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    pip install sleap-nn[torch-cuda128]
    ```

=== "macOS/CPU Only"
    ```bash
    pip install sleap-nn[torch-cpu]
    ```

!!! info "macOS MPS Support"
    Even with `torch-cpu`, macOS automatically enables MPS (Metal Performance Shaders) for Apple Silicon acceleration.

### Verify Installation

```bash
# Test the installation
sleap-nn --help

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Development Setup with uv

For contributing to sleap-nn or development workflows.

!!! info "uv sync"
    `uv sync` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, you must run commands with `uv run` (e.g., `uv run sleap-nn train ...` or `uv run pytest ...`).

### 1. Clone the Repository

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

### 2. Install uv

=== "macOS/Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```bash
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### 3. Install Dependencies

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    uv sync --extra dev --extra torch-cuda118
    ```

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uv sync --extra dev --extra torch-cuda128
    ```

=== "macOS/CPU Only"
    ```bash
    uv sync --extra dev --extra torch-cpu
    ```

### 4. Verify Development Setup

```bash
# Run tests
uv run pytest tests

# Check code formatting
uv run black --check sleap_nn tests
uv run ruff check sleap_nn/
```

---


## Troubleshooting

### Import Errors

If you get import errors:

1. Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
2. Try reinstalling with torch extras

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your NVIDIA drivers are up to date
2. Check CUDA compatibility with PyTorch version
3. Try the CPU-only installation as a fallback


## Next Steps

- [Step-by-step guide on training models](step_by_step_guide.md)
- [Configuration Guide](config.md)
- [Training models](training.md)