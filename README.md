# sleap-nn
Neural network backend for training and inference for animal pose estimation.

## Prerequisites

- Python 3.11

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

## ⚠️ PyTorch is Required at Runtime

The `torch` and `torchvision` dependencies are now defined as **optional** in `pyproject.toml`. However, they are **required for the code to run**, and are imported at the top level in many modules. This means:

> 🛑 If you install `sleap-nn` without `torch`, **any import of **``** will fail** with an `ImportError` until you install it manually.

---

## GitHub Workflows

This repository uses GitHub Actions for continuous integration and publishing:

### CI Workflow (`.github/workflows/ci.yml`)
Runs on every pull request and performs the following:
- Installs `uv` and uses it to install the package in editable mode with dev dependencies.
- Runs code quality checks using `black` and `ruff`.
- Executes the test suite using `pytest` with coverage reporting.
- Uploads coverage results to Codecov.

Runs on all major operating systems (`ubuntu-latest`, `windows-latest`, `macos-14`) and on `self-hosted-gpu` runners.

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