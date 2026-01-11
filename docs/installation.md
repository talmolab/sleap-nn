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
    - **[Installing Pre-release Versions](#installing-pre-release-versions)**: For testing alpha/beta releases

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

=== "Windows/Linux (GPU - Auto-detect)"
    ```bash
    # Recommended: Automatically detects your GPU and installs optimal PyTorch
    uv tool install sleap-nn[torch] --torch-backend auto
    ```

=== "Windows/Linux (CUDA 13.0)"
    ```bash
    uv tool install sleap-nn[torch] --torch-backend cu130
    ```

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uv tool install sleap-nn[torch] --torch-backend cu128
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv tool install sleap-nn[torch] --torch-backend cpu
    ```

=== "macOS"
    ```bash
    uv tool install "sleap-nn[torch]"
    ```

!!! note "uv Version Requirement"
    The `--torch-backend` option requires **uv 0.9.20 or later**. Check your version:
    ```bash
    uv --version
    ```
    Update if needed:
    ```bash
    uv self update
    ```

!!! info
    - The `--torch-backend auto` option automatically detects your GPU and installs the optimal PyTorch version.
    - For manual selection, use `--torch-backend cu130` (CUDA 13), `--torch-backend cu128` (CUDA 12.8), `--torch-backend cu118` (CUDA 11.8), or `--torch-backend cpu`.
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

### Verify installation

```bash
# Test the installation
sleap-nn --help
```

### Updating Dependencies

To update sleap-nn and its dependencies (e.g., sleap-io) to their latest versions:

```bash
# Upgrade sleap-nn to the latest version
uv tool upgrade sleap-nn
```

!!! note
    When upgrading, uv respects any version constraints specified during installation. The upgrade will only update within those constraints. To change version constraints, reinstall with new specifications using `uv tool install`.

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

=== "Windows/Linux (GPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --torch-backend auto sleap-nn train --config-name myconfig --config-dir /path/to/config_dir/
    uvx --from "sleap-nn[torch]" --torch-backend auto sleap-nn track --data_path video.mp4 --model_paths models/
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uvx --from "sleap-nn[torch]" --torch-backend cpu sleap-nn train --config-name myconfig --config-dir /path/to/config_dir/
    uvx --from "sleap-nn[torch]" --torch-backend cpu sleap-nn track --data_path video.mp4 --model_paths models/
    ```

=== "macOS"
    ```bash
    uvx "sleap-nn[torch]" train --config-name myconfig --config-dir /path/to/config_dir/
    uvx "sleap-nn[torch]" track --data_path video.mp4 --model_paths models/
    ```

!!! note
    - The `--torch-backend auto` option automatically detects your GPU and installs the optimal PyTorch version. Requires uv 0.9.20+.
    - For manual selection, use `--torch-backend cu130` (CUDA 13), `--torch-backend cu128` (CUDA 12.8), or `--torch-backend cpu`.
    - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

!!! note "uvx Installation"
    Because `uvx` installs packages fresh on every run, it's ideal for quick tests or use in remote environments. For regular use, you could install with [`uv tool install`](#installation-as-a-system-wide-tool-with-uv) or setting up a development environment with [`uv sync`](#installation-from-source) to avoid repeated downloads.

### Updating Dependencies

With `uvx`, no separate update command is needed:

!!! tip "Automatic Updates"
    `uvx` automatically fetches and installs the latest version of sleap-nn and its dependencies (e.g., sleap-io) each time you run a command. This means you're always using the most recent version unless you specify version constraints like `uvx "sleap-nn[torch]==0.0.3" ...`.

    To ensure you're using the latest version, simply run your `uvx` command as usual - it will automatically download and use the newest available version.

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

### Updating Dependencies

To update sleap-nn and its dependencies to their latest versions:

=== "Upgrade a Specific Package"
    ```bash
    # Upgrade sleap-nn and update the lock file
    uv add "sleap-nn[torch]" --upgrade-package sleap-nn

    # Upgrade a specific dependency like sleap-io
    uv add sleap-io --upgrade-package sleap-io
    ```

=== "Upgrade All Dependencies"
    ```bash
    # Upgrade all packages to their latest compatible versions
    uv sync --upgrade
    ```

!!! note
    - `uv add --upgrade-package <package>` forces the specified package to update to its latest compatible version, even if a valid version is already installed.
    - `uv sync --upgrade` refreshes the entire lockfile and updates all dependencies to their newest compatible versions while maintaining compatibility with your `pyproject.toml` constraints.
    - By default, `uv add` only updates the locked version if necessary to satisfy new constraints. Use `--upgrade-package` to force an update.

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

### Updating Dependencies

To update sleap-nn and its dependencies to their latest versions:

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    pip install --upgrade sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128

    # CUDA 11.8
    pip install --upgrade sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118
    ```

=== "Windows/Linux (CPU)"
    ```bash
    pip install --upgrade sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS"
    ```bash
    pip install --upgrade "sleap-nn[torch]"
    ```

!!! tip "Upgrading Specific Dependencies"
    To upgrade a specific dependency like sleap-io independently:
    ```bash
    pip install --upgrade sleap-io
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

=== "Windows/Linux (CUDA 13.0)"
    ```bash
    uv sync --extra torch-cuda130
    ```

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uv sync --extra torch-cuda128
    ```

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    uv sync --extra torch-cuda118
    ```

=== "macOS/CPU Only"
    ```bash
    uv sync --extra torch-cpu
    ```

#### 4. Updating Dependencies

To update sleap-nn and its dependencies to their latest versions:

=== "Windows/Linux (CUDA 13.0)"
    ```bash
    uv sync --extra torch-cuda130 --upgrade
    ```

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uv sync --extra torch-cuda128 --upgrade
    ```

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    uv sync --extra torch-cuda118 --upgrade
    ```

=== "macOS/CPU Only"
    ```bash
    uv sync --extra torch-cpu --upgrade
    ```

!!! tip "How --upgrade Works"
    The `--upgrade` flag refreshes the lockfile and updates all dependencies to their newest compatible versions while maintaining compatibility with your `pyproject.toml` constraints. This ensures you have the latest versions of all dependency packages.


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

## Installing Pre-release Versions

Pre-release versions (alpha, beta, release candidates) require explicit opt-in since they are excluded by default per [PEP 440](https://peps.python.org/pep-0440/#handling-of-pre-releases).

### Why Pre-releases?

Pre-release versions like `0.1.0a0` let us test new features and breaking changes with a smaller group before a stable release. Users won't accidentally get pre-releases unless they explicitly opt in.

### Installation Commands

=== "uv tool install"
    ```bash
    # With GPU auto-detection
    uv tool install sleap-nn[torch] --torch-backend auto --prerelease=allow

    # With specific CUDA version
    uv tool install sleap-nn[torch] --torch-backend cu130 --prerelease=allow
    ```

=== "uvx"
    ```bash
    # Run commands with pre-release version
    uvx --from "sleap-nn[torch]" --torch-backend auto --prerelease=allow sleap-nn --help
    uvx --from "sleap-nn[torch]" --torch-backend auto --prerelease=allow sleap-nn track -i video.mp4 -m models/
    ```

=== "uv add"
    ```bash
    # Add pre-release to project
    uv add sleap-nn[torch] --prerelease=allow --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "pip"
    ```bash
    # Install pre-release with pip
    pip install --pre sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
    ```

### Pin to Specific Version

To install a specific pre-release version:

```bash
# uv tool install
uv tool install "sleap-nn[torch]==0.1.0a0" --torch-backend auto

# pip
pip install "sleap-nn[torch]==0.1.0a0" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
```

### Environment Variable

For repeated use, you can set the environment variable:

```bash
export UV_PRERELEASE=allow
```

This makes all `uv` commands default to allowing pre-releases.

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