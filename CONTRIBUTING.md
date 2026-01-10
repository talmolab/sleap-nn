# Contributing to sleap-nn

Thank you for your interest in contributing to sleap-nn! This guide will help you get started with development and contribution.

## Development Setup

> **Python 3.14 is not yet supported**
>
> `sleap-nn` currently supports **Python 3.11, 3.12, and 3.13**.  
> **Python 3.14 is not yet tested or supported.**  
> By default, `uv` will use your system-installed Python.  
> If you have Python 3.14 installed, you must specify the Python version (≤3.13) in the install command.  
>
> For example:
>
> ```bash
> uv sync --python 3.13 ...
> ```
> Replace `...` with the rest of your install command as needed.

1. **Install [`uv`](https://github.com/astral-sh/uv) and development dependencies**  
   `uv` is a fast and modern package manager for `pyproject.toml`-based projects. Refer [installation docs](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

2. **Install sleap-nn dependencies based on your platform**\

   - Sync all dependencies based on your correct wheel using `uv sync`. `uv sync` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, you must run commands with `uv run` (e.g., `uv run sleap-nn train ...` or `uv run pytest ...`).
     - **Windows/Linux with NVIDIA GPU (CUDA 13.0):**

      ```bash
      uv sync --extra torch-cuda130
      ```

     - **Windows/Linux with NVIDIA GPU (CUDA 12.8):**

      ```bash
      uv sync --extra torch-cuda128
      ```

     - **Windows/Linux with NVIDIA GPU (CUDA 11.8):**

      ```bash
      uv sync --extra torch-cuda118
      ```

     - **macOS with Apple Silicon (M1, M2, M3, M4) or CPU-only (no GPU or unsupported GPU):**
     Note: Even if torch-cpu is used on macOS, the MPS backend will be available.
      ```bash
      uv sync --extra torch-cpu
      ```
> **Upgrading All Dependencies**
> To ensure you have the latest versions of all dependencies, use the `--upgrade` flag with `uv sync`:
> ```bash
> uv sync --upgrade
> ```
> This will upgrade all installed packages in your environment to the latest available versions compatible with your `pyproject.toml`.

## Code Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with these specifics:

- Use `black` for code formatting
- Use `ruff` for linting
- Maximum line length: 88 characters
- Use type hints where possible

### Running Code Quality Checks

```bash
# Format code
uv run black sleap_nn tests

# Check formatting
uv run black --check sleap_nn tests

# Run linter
uv run ruff check sleap_nn/
```

## Testing

We use `pytest` for testing. Please ensure all tests pass before submitting a PR:

```bash
# Run all tests
uv run pytest tests

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Run with coverage
uv run pytest --cov=sleap_nn --cov-report=xml tests/

# Run tests with duration info
uv run pytest --durations=-1 tests/
```

## Documentation

### Writing Documentation

- Use Google-style docstrings
- First line should fit within 88 characters
- Use imperative tense (e.g., "Load X..." not "Loads X...")
- Use backticks for potential auto-linking
- Document array shapes and data types

Example:
```python
def load_tracks(filepath: str) -> np.ndarray:
    """Load the tracks from a SLEAP Analysis HDF5 file.

    Args:
        filepath: Path to a SLEAP Analysis HDF5 file.
    
    Returns:
        The loaded tracks as a `np.ndarray` of shape `(n_tracks, n_frames, n_nodes, 2)`.
    """
```

### Documentation Website Workflow

To work with the documentation locally:

1. **Clone the sleap-nn repo**

```bash
git clone https://github.com/talmolab/sleap-nn.git
cd sleap-nn
```

2. Install `sleap-nn` with docs dependencies:
   ```bash
   uv sync --group docs --extra torch-cpu
   ```

3. Build and tag a new documentation version:
   ```bash
   uv run mike deploy --update-aliases 0.1.4 latest
   ```

4. Preview documentation locally:
   ```bash
   uv run mike serve
   ```

5. Push a specific version manually:
   ```bash
   uv run mike deploy --push --update-aliases --allow-empty 0.1.4 latest
   ```

The documentation is automatically deployed to https://nn.sleap.ai/ when changes are pushed to the main branch or when a new release is published.

## Submitting Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b your-username/your-feature-name
   ```

2. Make your changes and ensure tests pass

3. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

4. Push your branch and create a pull request

5. Ensure all CI checks pass

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update tests and documentation as needed
- Follow the existing code style
- Add yourself to the contributors list if this is your first contribution

## Questions?

If you have questions or need help, please:
- Check the [documentation](https://nn.sleap.ai/)
- Open an issue on GitHub
- Join our community discussions

Thank you for contributing to sleap-nn!