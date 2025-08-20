# Example Notebooks

This folder contains all tutorial notebooks created with marimo. These notebooks are **sandboxed**, which means you don't need to create a separate sleap-nn environment. The sandbox environment will handle all dependencies automatically. Just install `uv`, add `marimo`, and run the notebooks with the commands below. 

## Getting Started with Marimo

### Prerequisites

1. **Install [uv](https://github.com/astral-sh/uv)**: 
   ```bash
   pip install uv
   ```

2. **Create a new folder**:

   Clone the `sleap-nn` repo
   ```bash
   cd sleap-nn/example_notebooks
   ```
   
   Or make a new directory
   ```bash
   mkdir sleap-nn-tutorials
   ```

3. **Initialize uv**:

   Ensure the working directory where you run `uv init` doesn't have an existing `pyproject.toml`.
   ```bash
   uv init
   ```

### Setup

1. **Add [marimo](https://github.com/marimo-team/marimo) to your project**:

   Ensure the working directory where you run `uv init` doesn't have an existing `pyproject.toml` as `uv add` will try to add marimo to the existing `.toml`.

   ```bash
   uv add marimo
   ```

2. **Run the marimo notebooks**

Ensure the python scripts are in your current working directory!

(i) Training Demo Notebook
- **Description**: End-to-end demo on creating config files and running training, inference, and evaluation using sleap-nn APIs.

```bash
   uvx marimo edit --sandbox training_demo.py
```

(ii) Augmentation Guide Notebook
- **Description**: Visualize the different data augmentation techniques available in sleap-nn.

```bash
   uvx marimo edit --sandbox augmentation_guide.py
```

(iii) Receptive Field Guide Notebook
- **Description**: Visualize how the receptive field could be set by changing the config parameters.

```bash
   uvx marimo edit --sandbox receptive_field_guide.py
```

