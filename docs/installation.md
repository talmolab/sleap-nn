# Installation

**Prerequisites: Python 3.11**

## From PyPI

- **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

```bash
pip install sleap-nn[torch-cuda118]
```

- **Windows/Linux with NVIDIA GPU (CUDA 12.8):**

```bash
pip install sleap-nn[torch-cuda128]
```

- **macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):** 
Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
```bash
pip install sleap-nn[torch-cpu]
```


## For development setup

1. **Clone the sleap-nn repo**

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

2. **Install [`uv`](https://github.com/astral-sh/uv) and development dependencies**  
   `uv` is a fast and modern package manager for `pyproject.toml`-based projects. Refer [installation docs](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

3. **Install sleap-nn dependencies based on your platform**\

   - Sync all dependencies based on your correct wheel using `uv sync`:
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

   You can find the correct wheel for your system at:\
   👉 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

4. **Run tests**  
   ```bash
   uv run pytest tests
   ```

5. **(Optional) Lint and format code**
   ```bash
   uv run black --check sleap_nn tests
   uv run ruff check sleap_nn/
   ```

---


## Troubleshooting

### Import Errors

If you get import errors:

1. Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
2. Try reinstalling with torch extras: `uv pip install -e ".[torch-cuda118]"`

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your NVIDIA drivers are up to date
2. Check CUDA compatibility with PyTorch version
3. Try the CPU-only installation as a fallback


## Next Steps

- [Step-by-step guide on training models](step_by_step_guide.md)
- [Configuration Guide](config.md)
- [Training models](training.md)