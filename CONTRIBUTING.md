# Contributing to sleap-nn

Thank you for your interest in contributing to sleap-nn! This guide will help you get started with development and contribution.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sleap-nn.git
   cd sleap-nn
   ```

2. Create a conda environment based on your platform:
   ```bash
   # GPU (Windows/Linux)
   mamba env create -f environment.yml
   
   # CPU (Windows/Linux/Intel Mac)
   mamba env create -f environment_cpu.yml
   
   # Apple Silicon (M1/M2 Mac)
   mamba env create -f environment_osx-arm64.yml
   ```

3. Activate the environment:
   ```bash
   mamba activate sleap-nn
   ```

4. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with these specifics:

- Use `black` for code formatting
- Use `ruff` for linting
- Maximum line length: 88 characters
- Use type hints where possible

### Running Code Quality Checks

```bash
# Format code
black sleap_nn tests

# Check formatting
black --check sleap_nn tests

# Run linter
ruff check sleap_nn/
```

## Testing

We use `pytest` for testing. Please ensure all tests pass before submitting a PR:

```bash
# Run all tests
pytest tests

# Run specific test file
pytest tests/path/to/test_file.py

# Run with coverage
pytest --cov=sleap_nn --cov-report=xml tests/

# Run tests with duration info
pytest --durations=-1 tests/
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

1. Install `sleap-nn` with docs dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build and tag a new documentation version:
   ```bash
   mike deploy --update-aliases 0.1.4 latest
   ```

3. Preview documentation locally:
   ```bash
   mike serve
   ```

4. Push a specific version manually:
   ```bash
   mike deploy --push --update-aliases --allow-empty 0.1.4 latest
   ```

The documentation is automatically deployed to https://nn.sleap.ai/ when changes are pushed to the main branch or when a new release is published.

## Submitting Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
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