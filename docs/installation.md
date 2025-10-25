# Installation

**Prerequisites:** 

Python 3.11 (or) 3.12 (or) 3.13 (required for all installation methods)

!!! warning "Python 3.14 is not yet supported"
    `sleap-nn` currently supports **Python 3.11, 3.12, and 3.13**. **Python 3.14 is not yet tested or supported.** If you have Python 3.14 installed, you must specify the Python version in all `uv` install commands by adding `--python 3.13`.  
    For example:
    ```bash
    # for uv tool install
    uv tool install --python 3.13 "sleap-nn[torch]"  ...

    # for uvx
    uvx --python 3.13 ...

    # for uv add setup; specify version in uv init
    uv init --python 3.13

    # for uv sync
    uv sync --python 3.13 ...
    ```
    Replace `...` with the rest of your install command as needed.


!!! tip "Choose Your Installation Method"
    - **[Installation as a system-wide tool with uv](#installation-as-a-system-wide-tool-with-uv)**: **(Recommended)** Use `uv tool install` to install sleap-nn globally as a CLI tool
    - **[Installation with uvx](#installation-with-uvx)**: Use `uvx` for one-off commands (no installation needed)
    - **[Installation with uv add](#installation-with-uv-add)**: Use `uv add` to install sleap-nn as a dependency in a uv virtual env. (useful for project-specific workspaces)
    - **[Installation with pip](#installation-with-pip)**: Use `pip` to install from pypi in a conda env. (Recommended to use with a conda env)
    - **[Installation from source](#installation-from-source)**: Use `uv sync` to install from source (for developmental purposes)

---

## Installation as a system-wide tool with uv

`uv tool install` installs sleap-nn globally as a system-wide CLI tool, making `sleap-nn` commands available from anywhere in your terminal.

!!! note "Install uv"
    Install [`uv`](https://github.com/astral-sh/uv) - a fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Platform-Specific Installation

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv tool install sleap-nn[torch] --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple

    # CUDA 11.8
    uv tool install sleap-nn[torch] --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv tool install sleap-nn[torch] --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv tool install "sleap-nn[torch]"
    ```

!!! info
    - For more information on which CUDA version to use for your system, see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).  
      The `--index` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cu118` for CUDA 11.8, `https://download.pytorch.org/whl/cu128` for CUDA 12.8, etc.).
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

### Verify installation

```bash
# Test the installation
sleap-nn --help
```

---

## Installation with uvx

`uvx` automatically installs sleap-nn and runs your command inside a temporary virtual environment (venv). This means each run is fully isolated and leaves no trace on your system—perfect for trying out sleap-nn without any permanent installation.

!!! note "Install uv"
    Install [`uv`](https://github.com/astral-sh/uv) - a fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    uvx --from "sleap-nn[torch]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap-nn train --config-name myconfig --config-dir /path/to/config_dir/
    uvx --from "sleap-nn[torch]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap-nn track --data_path video.mp4 --model_paths models/
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple sleap-nn train --config-name myconfig --config-dir /path/to/config_dir/
    uvx --from "sleap-nn[torch]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple sleap-nn track --data_path video.mp4 --model_paths models/
    ```

=== "macOS"
    ```bash
    uvx "sleap-nn[torch]" train --config-name myconfig --config-dir /path/to/config_dir/
    uvx "sleap-nn[torch]" track --data_path video.mp4 --model_paths models/
    ```

!!! note
    - For more information on which CUDA version to use for your system, see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).  
      The `--index` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cu118` for CUDA 11.8, `https://download.pytorch.org/whl/cu128` for CUDA 12.8, etc.).
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

!!! note "uvx Installation"
    Because `uvx` installs packages fresh on every run, it's ideal for quick tests or use in remote environments. For regular use, you could install with [`uv tool install`](#installation-as-a-system-wide-tool-with-uv) or setting up a development environment with [`uv sync`](#installation-from-source) to avoid repeated downloads.

---

## Installation with uv add

This method creates a dedicated project environment using uv's modern Python project management. It initializes a new project with `uv init`, creates an isolated virtual environment with `uv venv`, and adds sleap-nn as a dependency using `uv add`. To use all installed packages, you must run commands with `uv run` (e.g., `uv run sleap-nn train ...` or `uv run pytest ...`).

!!! note "Install and set-up uv"
    Step-1: Install [`uv`](https://github.com/astral-sh/uv) - a fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Step-2: Move to your project directory and initialize the virtual env.
    ```bash
    uv init
    uv venv
    ```

### Platform-Specific Installation

!!! tip "How `uv add` works"
    - When you run `uv init`, it creates a `pyproject.toml` file in your working directory to manage your project's dependencies.
    - When you use `uv add sleap-nn[torch]`, it adds `sleap-nn[torch]` as a dependency in your `pyproject.toml` and installs it in your virtual environment.
    - To add other packages, simply run `uv add <package>`. After adding new packages, you should run `uv sync` to update your environment with all dependencies specified in `pyproject.toml`. (or `uv sync --upgrade` to update all dependencies)
    - To install a local package (such as a local clone of sleap-nn) in editable mode, use:
      ```bash
      uv add --editable "path/to/sleap-nn[torch]" ...
      ```
      This is useful for development, as changes to the code are immediately reflected in your environment.

!!! warning "Windows: MarkupSafe Installation Issue"
    On **Windows**, you may encounter errors when running `uv add "sleap-nn[torch]" --index ...` due to an incompatibility with the MarkupSafe wheel (e.g., "failed to install MarkupSafe" or similar errors).  
    Similar issues: [#11532](https://github.com/astral-sh/uv/issues/11532) and [#12620](https://github.com/astral-sh/uv/issues/12620).

    **Workaround:**  
    Before running `uv add "sleap-nn[torch]" ...` on Windows, manually install a compatible version of MarkupSafe:

    ```bash
    uv add git+https://github.com/pallets/markupsafe@3.0.2
    ```

    Then proceed with:

    ```bash
    uv add "sleap-nn[torch]" ...
    ```

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv add sleap-nn[torch] --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple

    # CUDA 11.8
    uv add sleap-nn[torch] --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv add sleap-nn[torch] --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv add "sleap-nn[torch]"
    ```

!!! info
    - For more information on which CUDA version to use for your system, see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).  
      The `--index` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cu118` for CUDA 11.8, `https://download.pytorch.org/whl/cu128` for CUDA 12.8, etc.).
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

### Verify Installation

```bash
# Test the installation
uv run sleap-nn --help
```

!!! warning "sleap-nn not recognized after installation?"

    If running the verification step above gives an error like `sleap-nn: command not found` or `'sleap-nn' is not recognized as an internal or external command`, try the following workarounds:

    - Activate your virtual environment (the venv name should be the same as your current working dir name). If you used `uv`, activate it and then run:
      ```bash
      uv run --active sleap-nn --help
      ```
      This ensures the command runs in the correct environment.

    - **Another workaround (not recommended):**  
      Check if you have any *empty* `pyproject.toml` or `uv.lock` files in `Users/<your-user-name>`. If you find empty files with these names, delete them and try again. (Empty files here can sometimes interfere with uv's environment resolution.)

---

## Installation with pip

We recommend creating a dedicated environment with [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba/miniforge](https://github.com/conda-forge/miniforge) before installing `sleap-nn` with pip. This helps avoid dependency conflicts and keeps your Python setup clean. After installing Miniconda or Miniforge, create and activate an environment, then run the pip install commands below inside the activated environment.

!!! warning "Python 3.14 is not yet supported"
    SLEAP currently supports **Python 3.11, 3.12, and 3.13**. **Python 3.14 is not yet tested or supported.**

To create a conda environment, run:
```bash
conda create -n sleap-nn-env python=3.13
conda activate sleap-nn-env
```

### Platform-Specific Installation

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    pip install sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128

    # CUDA 11.8
    pip install sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118
    ```

=== "Windows/Linux (CPU)"
    ```bash
    pip install sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS"
    ```bash
    pip install "sleap-nn[torch]"
    ```

!!! info
    - For more information on which CUDA version to use for your system, see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).  
      The `--extra-index-url` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cu118` for CUDA 11.8, `https://download.pytorch.org/whl/cu128` for CUDA 12.8, etc.).
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.


### Verify Installation

```bash
# Test the installation
sleap-nn --help

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Installation from source

For contributing to sleap-nn or development workflows.

!!! info "uv sync"
    `uv sync` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, you must run commands with `uv run` (e.g., `uv run sleap-nn train ...` or `uv run pytest ...`).

#### 1. Clone the Repository

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

#### 2. Install uv

=== "macOS/Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```bash
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

#### 3. Install Dependencies

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

!!! tip "Upgrading All Dependencies"
    To ensure you have the latest versions of all dependencies, use the `--upgrade` flag with `uv sync`:
    ```bash
    uv sync --extra dev --upgrade
    ```
    This will upgrade all installed packages in your environment to the latest available versions compatible with your `pyproject.toml`.


### Verify Installation

In your working dir (where you ran `uv sync`):

```bash
# Run tests
uv run pytest tests

# Check code formatting
uv run black --check sleap_nn tests
uv run ruff check sleap_nn/

# Run CLI commands
uv run sleap-nn train ...
uv run sleap-nn track ...
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