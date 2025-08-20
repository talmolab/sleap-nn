# Example Notebooks

This page provides information on how to access and use the example notebooks from the sleap-nn GitHub repository.

## Accessing Example Notebooks

The example notebooks are available in the [sleap-nn GitHub repository](https://github.com/talmolab/sleap-nn) in the `example_notebooks/` folder. These notebooks are created with [marimo](https://github.com/marimo-team/marimo) and provide interactive tutorials for various sleap-nn workflows.

### Getting Started with Marimo

The example notebooks use marimo, which provides a **sandboxed environment** that automatically handles all dependencies. You don't need to create a separate sleap-nn environment.

#### Prerequisites

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

#### Setup

1. **Add [marimo](https://github.com/marimo-team/marimo) to your project**:

   Ensure the working directory where you run `uv init` doesn't have an existing `pyproject.toml` as `uv add` will try to add marimo to the existing `.toml`.

   ```bash
   uv add marimo
   ```

2. **Run the marimo notebooks**

Ensure the python scripts are in your current working directory!

(i) Training Demo Notebook

- **Description**: End-to-end demo on creating config files and running training, inference, and evaluation using sleap-nn APIs.

> **Note:** Marimo notebooks are designed for a seamless, automated workflow. After you select the model type, all cells will execute automatically—no need to run them one by one. Training will only start when you click the  **Run Training** button, giving you full control over when to begin model training.

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
