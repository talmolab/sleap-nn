# Installation

## Before You Start

SLEAP-NN uses [**uv**](https://docs.astral.sh/uv/) for installation and environment management. If you're coming from conda/pip, this section explains why.

??? question "Why uv?"

    **GPU dependencies are complex.** PyTorch requires matching CUDA versions, platform-specific wheels, and careful index configuration. Traditional pip/conda installs often result in CPU-only PyTorch or version conflicts.

    **uv solves this** with the `--torch-backend` flag:

    - `auto` – Detects your GPU and installs the right PyTorch
    - `cu130` / `cu128` – Explicit CUDA versions
    - `cpu` – CPU-only (smaller install)

    **One command, correct GPU support.** No manual index URLs or environment debugging.

    !!! info "uv version requirement"
        The `--torch-backend` flag for `uv tool install` requires **uv 0.9.20+**. Run `uv self update` if you encounter errors.

??? question "Can I still use pip/conda?"

    Yes. See the [pip installation](#pip) section below. You'll need to manually configure PyTorch index URLs.

---

## Install SLEAP-NN

=== "Linux / Windows"

    **Step 1: Install uv**

    === "Linux / macOS / WSL"
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    === "Windows (PowerShell)"
        ```powershell
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    **Step 2: Install sleap-nn**

    !!! warning "Python 3.14 not supported"
        If you don't have Python installed, uv will automatically download the latest version (Python 3.14), which is not yet supported. Add `--python 3.13` to specify a compatible version:
        ```bash
        uv tool install --python 3.13 sleap-nn[torch] --torch-backend auto
        ```

    ```bash
    uv tool install sleap-nn[torch] --torch-backend auto
    ```

    **Step 3: Verify**

    ```bash
    sleap-nn system
    ```

=== "macOS"

    **Step 1: Install uv**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    **Step 2: Install sleap-nn**

    !!! warning "Python 3.14 not supported"
        If you don't have Python installed, uv will automatically download the latest version (Python 3.14), which is not yet supported. Add `--python 3.13` to specify a compatible version:
        ```bash
        uv tool install --python 3.13 "sleap-nn[torch]"
        ```

    ```bash
    uv tool install "sleap-nn[torch]"
    ```

    !!! note "Apple Silicon"
        PyTorch uses Metal Performance Shaders (MPS) for GPU acceleration on M1/M2/M3 Macs. No additional configuration needed.

    **Step 3: Verify**

    ```bash
    sleap-nn system
    ```

=== "CPU Only"

    **Step 1: Install uv**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    **Step 2: Install sleap-nn**

    !!! warning "Python 3.14 not supported"
        If you don't have Python installed, uv will automatically download the latest version (Python 3.14), which is not yet supported. Add `--python 3.13` to specify a compatible version:
        ```bash
        uv tool install --python 3.13 sleap-nn[torch] --torch-backend cpu
        ```

    ```bash
    uv tool install sleap-nn[torch] --torch-backend cpu
    ```

    **Step 3: Verify**

    ```bash
    sleap-nn system
    ```

---

## Updating


=== "Update to latest"
    ```bash
    uv tool upgrade sleap-nn
    ```

    !!! note
        This preserves the extras (`[torch]`) and torch backend from your original installation. If you need to change the torch backend, use the reinstall option.

=== "Update to specific version"
    ```bash
    uv tool install "sleap-nn[torch]==0.1.0" --torch-backend auto --force
    ```

=== "Reinstall (fix issues)"
    ```bash
    uv tool install sleap-nn[torch] --torch-backend auto --reinstall
    ```

    !!! tip "When to use `--reinstall`"
        Use this when you've updated CUDA drivers, changed GPUs, or have import errors.

=== "Downgrade"
    ```bash
    uv tool install "sleap-nn[torch]==0.0.5" --torch-backend auto --force
    ```

=== "Uninstall"
    ```bash
    uv tool uninstall sleap-nn
    ```

---

## Pre-release Versions

Install alpha/beta releases to test new features:

```bash
uv tool install sleap-nn[torch] --torch-backend auto --prerelease=allow
```

Install a specific pre-release:

```bash
uv tool install "sleap-nn[torch]==0.1.0a4" --torch-backend auto
```

---

## Alternative Methods

### uvx (No Install)

Run sleap-nn without permanent installation. Each command creates a temporary environment.

```bash
# Train
uvx --from "sleap-nn[torch]" --torch-backend auto sleap-nn train --config config.yaml

# Inference
uvx --from "sleap-nn[torch]" --torch-backend auto sleap-nn track -i video.mp4 -m models/
```

!!! tip "Always latest"
    uvx uses the latest version each run. Great for testing or one-off tasks.

---

### pip

Use pip when working within conda/mamba environments.

**Create environment:**

```bash
conda create -n sleap-nn python=3.13
conda activate sleap-nn
```

**Install with GPU support:**

=== "CUDA 12.8"
    ```bash
    pip install sleap-nn[torch] \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu128
    ```

=== "CUDA 11.8"
    ```bash
    pip install sleap-nn[torch] \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu118
    ```

=== "CPU Only"
    ```bash
    pip install sleap-nn[torch] \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS"
    ```bash
    pip install "sleap-nn[torch]"
    ```

!!! tip "Other CUDA versions"
    You can install any CUDA version supported by PyTorch by changing the index URL. Replace `cu128` with your desired version (e.g., `cu124`, `cu121`, `cu118`). See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for available versions.

**Verify:**

```bash
sleap-nn system
```

---

### From Source

For development and contributing.

**Step 1: Clone repository**

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

**Step 2: Install uv (if needed)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 3: Install in development mode**

```bash
uv sync --extra torch-cuda130
```

!!! tip "Other backends"
    Replace `torch-cuda130` with `torch-cuda128`, `torch-cpu`, or `torch` (macOS) as needed.

**Step 4: Run commands**

```bash
uv run sleap-nn --help
```

```bash
uv run pytest tests/
```

See [Contributing](https://github.com/talmolab/sleap-nn/blob/main/CONTRIBUTING.md) for development guidelines.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11 | 3.13 |
| RAM | 8 GB | 16+ GB |

!!! note "Apple Silicon"
    M1/M2/M3 Macs are fully supported via Metal Performance Shaders (MPS).

!!! warning "Python 3.14"
    Not yet supported. Use `--python 3.13` with uv commands.

---

## Troubleshooting

??? question "Command not found after install"

    Restart your terminal or source your shell config:

    ```bash
    source ~/.bashrc  # or ~/.zshrc on macOS
    ```

    Verify the tool is installed:

    ```bash
    uv tool list
    ```

??? question "CUDA not detected"

    1. **Check NVIDIA drivers:**
       ```bash
       nvidia-smi
       ```

    2. **Check sleap-nn detects CUDA:**
       ```bash
       sleap-nn system
       ```

    3. **Reinstall with explicit CUDA version:**
       ```bash
       uv tool install sleap-nn[torch] --torch-backend cu128 --reinstall
       ```

??? question "Import errors or missing modules"

    Reinstall with the torch extras:

    ```bash
    uv tool install sleap-nn[torch] --torch-backend auto --reinstall
    ```

??? question "Wrong Python version"

    Specify the Python version explicitly:

    ```bash
    uv tool install --python 3.13 sleap-nn[torch] --torch-backend auto
    ```

??? question "uv version too old"

    The `--torch-backend` flag requires uv 0.9.20+.

    ```bash
    # Check version
    uv --version

    # Update uv
    uv self update
    ```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch: **Quick Start**

    Train your first model in 5 minutes.

    [:octicons-arrow-right-24: Get started](getting-started/quickstart.md)

-   :material-school: **Your First Model**

    Complete walkthrough from data to predictions.

    [:octicons-arrow-right-24: Tutorial](getting-started/first-model.md)

</div>
