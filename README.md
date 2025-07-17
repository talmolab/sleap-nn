# sleap-nn
Neural network backend for training and inference for animal pose estimation.

## 🚀 Development Setup

1. **Install [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers)**  
   We recommend using [miniforge](https://github.com/conda-forge/miniforge) for an isolated Python environment with fast dependency resolution.

2. **Create and activate the development environment using Python 3.11**  
   ```bash
   mamba create -n sleap-nn-dev python=3.11
   mamba activate sleap-nn-dev
   ```

3. **Install [`uv`](https://github.com/astral-sh/uv) and development dependencies**  
   `uv` is a fast and modern package manager for `pyproject.toml`-based projects.
   ```bash
   pip install uv
   uv pip install -e ".[dev]"
   ```

4. **Install PyTorch based on your platform**\
   You can either:

   - Install the optional dependencies:

     ```bash
     uv pip install -e ".[torch]"
     ```

     This installs the default builds of `torch` and `torchvision` via PyPI for your OS.

   - Or manually install the correct wheel for your system using PyTorch's index URL:

     - **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

       ```bash
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
       ```

     - **macOS with Apple Silicon (M1, M2, M3, M4):** You don’t need to do anything if you used the `[torch]` optional dependency or default PyPI install--the default wheels now include Metal backend support for Apple GPUs.

     - **CPU-only (no GPU or unsupported GPU):** You don’t need to do anything if you used the `[torch]` optional dependency or default PyPI install.

      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ```

   You can find the correct wheel for your system at:\
   👉 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

5. **Run tests**  
   ```bash
   pytest tests
   ```

6. **(Optional) Lint and format code**
   ```bash
   black --check sleap_nn tests
   ruff check sleap_nn/
   ```

---

## ⚠️ PyTorch is Required at Runtime

The `torch` and `torchvision` dependencies are now defined as **optional** in `pyproject.toml`. However, they are **required for the code to run**, and are imported at the top level in many modules. This means:

- You **must** install `sleap-nn` with the `[torch]` extras:

  ```bash
  uv pip install -e ".[torch]"
  ```

- **Or** manually install `torch` and `torchvision` with the appropriate build for your system.

> 🛑 If you install `sleap-nn` without `torch`, **any import of **``** will fail** with an `ImportError` until you install it manually.

---

## 🛠️ Future Improvements

To improve the flexibility and user experience, future versions of `sleap-nn` may:

- **Move **``** imports into functions** or use conditional imports to avoid top-level dependency failures.
- **Add clearer runtime error messages** when `torch` is required but not available.
- **Enable limited functionality** in environments without PyTorch (e.g. metadata inspection or CLI help).

This would allow `sleap-nn` to be installed and imported without `torch`, only requiring it when deep learning functionality is actually used.

---

## ⚙️ GPU Support Strategy

We intentionally **do not include GPU-specific `torch` or `torchvision` builds** in `pyproject.toml`. Instead, we recommend installing them manually based on your platform.

### ✅ Why this strategy works

- **Portability**: No CUDA version or hardware is assumed. This avoids broken installs on unsupported platforms.
- **Flexibility**: You can use the appropriate PyTorch build for your system.
- **Reliability**: All other dependencies are managed cleanly with `uv`.

> 💡 This makes `sleap-nn` compatible with both GPU-accelerated and CPU-only environments.

<details>
<summary>📦 Why not use `pyproject.toml` for GPU builds?</summary>

- GPU wheels are not on PyPI — they live at [https://download.pytorch.org/whl/](https://download.pytorch.org/whl/)
- These builds vary by platform, CUDA version, and GPU architecture.
- `uv` does not currently support CLI-based extra index URLs like pip’s `--index-url`.
- Hardcoding GPU wheels into `pyproject.toml` would break cross-platform support.

</details>

## GitHub Workflows

This repository uses GitHub Actions for continuous integration and publishing:

### CI Workflow (`.github/workflows/ci.yml`)
Runs on every pull request and performs the following:
- Sets up a Conda environment using Miniforge3 with Python 3.11.
- Installs `uv` and uses it to install the package in editable mode with dev dependencies.
- Runs code quality checks using `black` and `ruff`.
- Executes the test suite using `pytest` with coverage reporting.
- Uploads coverage results to Codecov.

Runs on all major operating systems (`ubuntu-latest`, `windows-latest`, `macos-14`).

### Release Workflow (`.github/workflows/uvpublish.yml`)
Triggered on GitHub Releases:

- For **pre-releases**, the package is published to [Test PyPI](https://test.pypi.org) for testing.
- For **final releases**, the package is published to the official [PyPI](https://pypi.org) registry using trusted publishing.

The `uv` tool is used for both building and publishing. You can create a pre-release by tagging your release with a version suffix like `1.0.0rc1` or `1.0.0b1`.

To test the pre-release in your development workflow:
```bash
uv pip install --index-url https://test.pypi.org/simple/ sleap-nn
```

Trusted publishing is handled automatically using GitHub OIDC, and no credentials are stored.