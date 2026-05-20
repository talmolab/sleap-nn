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
        uv tool install --python 3.13 sleap-nn --torch-backend auto
        ```

    ```bash
    uv tool install sleap-nn --torch-backend auto
    ```

    This auto-detects your GPU and installs the correct PyTorch build. You can also specify a backend explicitly:

    ```bash
    # Explicit CUDA 13.0
    uv tool install sleap-nn --torch-backend cu130

    # CPU-only (smaller install)
    uv tool install sleap-nn --torch-backend cpu
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
        uv tool install --python 3.13 sleap-nn
        ```

    ```bash
    uv tool install sleap-nn
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
        uv tool install --python 3.13 sleap-nn --torch-backend cpu
        ```

    ```bash
    uv tool install sleap-nn --torch-backend cpu
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
        This preserves the torch backend from your original installation. If you need to change the torch backend, use the reinstall option.

=== "Update to specific version"
    ```bash
    uv tool install "sleap-nn==0.1.0" --torch-backend auto --force
    ```

=== "Reinstall (fix issues)"
    ```bash
    uv tool install sleap-nn --torch-backend auto --reinstall
    ```

    !!! tip "When to use `--reinstall`"
        Use this when you've updated CUDA drivers, changed GPUs, or have import errors.

=== "Downgrade"
    ```bash
    uv tool install "sleap-nn==0.0.5" --torch-backend auto --force
    ```

=== "Uninstall"
    ```bash
    uv tool uninstall sleap-nn
    ```

---

## Pre-release Versions

Install alpha/beta releases to test new features:

```bash
uv tool install sleap-nn --torch-backend auto --prerelease=allow
```

Install a specific pre-release:

```bash
uv tool install "sleap-nn==0.1.0a4" --torch-backend auto
```

---

## Alternative Methods

### uvx (No Install)

Run sleap-nn without permanent installation. Each command creates a temporary environment.

```bash
# Train
uvx --from sleap-nn --torch-backend auto sleap-nn train --config config.yaml

# Inference
uvx --from sleap-nn --torch-backend auto sleap-nn track -i video.mp4 -m models/
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

=== "Default (PyPI)"
    ```bash
    pip install sleap-nn
    ```

    !!! note
        PyTorch wheels on PyPI include CUDA support on Linux and Windows. This is the simplest option and works for most users.

=== "CUDA 12.8"
    ```bash
    pip install sleap-nn \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu128
    ```

=== "CUDA 11.8"
    ```bash
    pip install sleap-nn \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu118
    ```

=== "Intel GPU (XPU)"
    ```bash
    pip install sleap-nn \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/xpu
    ```

    !!! info "Intel Arc / Battlemage"
        Linux x86_64 only. See **System Requirements** below for the Intel GPU
        userspace runtime (`libze1`, `libze-intel-gpu1`, `intel-opencl-icd`).

=== "CPU Only"
    ```bash
    pip install sleap-nn \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS"
    ```bash
    pip install sleap-nn
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
uv sync --extra gpu
```

This installs with CUDA 13.0 support. Other backends:

| Extra | Backend |
|-------|---------|
| `--extra gpu` | CUDA 13.0 (alias for `torch-cuda130`) |
| `--extra cpu` | CPU-only (alias for `torch-cpu`) |
| `--extra torch-cuda128` | CUDA 12.8 |
| `--extra torch-cuda118` | CUDA 11.8 |
| `--extra torch-xpu` | Intel GPU (Arc / Battlemage), Linux x86_64 only |

!!! note
    On macOS, use `--extra cpu` — the MPS backend is automatically available.

!!! info "Intel GPU prerequisites"
    `--extra torch-xpu` only pulls the PyTorch wheel. You also need the Intel
    GPU userspace runtime — see **System Requirements** below.

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

!!! note "Intel GPU (Arc / Battlemage)"
    Linux x86_64 only. PyTorch ≥ 2.5 ships native `torch.xpu` support; sleap-nn
    registers a Lightning `XPUAccelerator` on import so `trainer_accelerator="xpu"`
    just works.

    **Userspace runtime** (install once, system-wide):

    === "Ubuntu 24.04+"
        ```bash
        sudo apt install libze1 libze-intel-gpu1 intel-opencl-icd clinfo
        ```
        Ubuntu 26.04 ships these in the `universe` repo by default. For 24.04
        link the Intel [GPU compute apt repo](https://dgpu-docs.intel.com/driver/client/overview.html)
        first.

    === "Other distros"
        Install the equivalent of the Intel Level-Zero loader, the Intel L0
        backend for your GPU, and the Intel OpenCL ICD. The PyTorch XPU wheel
        bundles the SYCL runtime (`intel-sycl-rt`, `oneCCL`, `oneMKL`), so
        Intel oneAPI Base Toolkit is **not** required.

    **Headless / SSH / service users:** `/dev/dri/renderD*` is ACL-granted to
    the active desktop seat owner. For SSH or systemd-service users, add
    `$USER` to the `render` group:

    ```bash
    sudo gpasswd -a "$USER" render && newgrp render
    ```

    **Verify:**

    ```bash
    clinfo -l                  # should list "Intel(R) Arc(TM) ..."
    sleap-nn system            # should report "accelerator: xpu"
    ```

    **Kernel driver note:** the `i915` and `xe` DRM drivers both work for
    PyTorch XPU compute. On kernel 7.0 with Arc A-series (Alchemist, e.g.
    A770), the default is `i915`. To force `xe`, boot with
    `i915.force_probe=!56a0 xe.force_probe=56a0` — usually unnecessary.

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
       uv tool install sleap-nn --torch-backend cu128 --reinstall
       ```

??? question "Import errors or missing modules"

    Reinstall sleap-nn:

    ```bash
    uv tool install sleap-nn --torch-backend auto --reinstall
    ```

??? question "Wrong Python version"

    Specify the Python version explicitly:

    ```bash
    uv tool install --python 3.13 sleap-nn --torch-backend auto
    ```

??? question "Intel GPU (XPU) not detected"

    1. **Confirm the device is visible at the OS level:**
       ```bash
       clinfo -l                # should list Intel Arc / Battlemage
       ls /dev/dri/renderD*     # should exist; check `getfacl` for your user
       ```

    2. **Confirm torch sees it:**
       ```bash
       python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
       ```
       Expected: `2.x.y+xpu True`. If the version doesn't end in `+xpu`, your
       venv has the CPU wheel — install with `--extra torch-xpu` (or
       `--extra-index-url https://download.pytorch.org/whl/xpu` for pip).

    3. **Check sleap-nn detects XPU:**
       ```bash
       sleap-nn system
       ```

    !!! warning "torch can silently downgrade"
        Any `uv pip install` that resolves a torch-using package will replace
        the XPU wheel with the PyPI CPU wheel unless you re-pin the source.
        Always use `uv sync --extra torch-xpu` (from source) or include
        `--extra-index-url https://download.pytorch.org/whl/xpu` (with pip)
        when re-installing.

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
