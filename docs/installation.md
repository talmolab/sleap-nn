# Installation

## Prerequisites

- Python 3.8 or higher
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Mamba](https://mamba.readthedocs.io/) (recommended for faster dependency resolution)

## Environment Setup

### GPU Support (Windows/Linux)

For systems with NVIDIA GPUs:

```bash
mamba env create -f environment.yml
mamba activate sleap-nn
```

### CPU Only (Windows/Linux/Intel Mac)

For systems without GPUs or when GPU support is not needed:

```bash
mamba env create -f environment_cpu.yml
mamba activate sleap-nn
```

### Apple Silicon (M1/M2/M3)

For Apple Silicon Macs:

```bash
mamba env create -f environment_osx-arm64.yml
mamba activate sleap-nn
```

## Development Installation

If you plan to contribute or modify the code:

```bash
# Clone the repository
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn

# Create environment (choose appropriate file for your platform)
mamba env create -f environment.yml
mamba activate sleap-nn

# Install in development mode
pip install -e ".[dev]"
```

## Verifying Installation

Test your installation:

```python
import sleap_nn
import torch

print(f"sleap-nn version: {sleap_nn.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your NVIDIA drivers are up to date
2. Check CUDA compatibility with PyTorch version
3. Try the CPU-only installation as a fallback

### Import Errors

If you get import errors:

1. Ensure you've activated the conda environment
2. Verify all dependencies installed correctly
3. Try reinstalling the package

### Memory Issues

For large models or datasets:

1. Consider using mixed precision training
2. Reduce batch size
3. Use gradient accumulation

## Next Steps

- [Training Your First Model](training.md)
- [Configuration Guide](configuration.md)
- [API Reference](api/index.md)