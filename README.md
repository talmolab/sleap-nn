# sleap-nn
Neural network backend for training and inference for animal pose estimation.

## Development

1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation.html) ([Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) is recommended).

2. Create and activate the environment (Python 3.11):
   ```bash
   mamba create -n sleap-nn-dev python=3.11
   mamba activate sleap-nn-dev
   ```

3. Install [uv](https://github.com/astral-sh/uv) and the package with development dependencies:
   ```bash
   pip install uv
   uv pip install -e ".[dev]"
   ```

4. Run the tests:
   ```bash
   pytest tests
   ```

Optional: You can run the linter and formatter checks manually:
```bash
black --check sleap_nn tests
ruff check sleap_nn/
```

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