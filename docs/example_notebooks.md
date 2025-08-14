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

2. **Create a new folder and initialize uv**:
   ```bash
   uv init
   ```

#### Setup

1. **Add [marimo](https://github.com/marimo-team/marimo) to your project**:
   ```bash
   uv add marimo
   ```

2. **Run a marimo notebook**:
   ```bash
   uvx marimo edit --sandbox <filename.py>
   ```

   For example:
   ```bash
   uvx marimo edit --sandbox "notebook.py"
   ```

## 📖 Available Tutorials

### 1. Training Demo Notebook
- **File**: [training_demo.py](https://github.com/talmolab/sleap-nn/blob/main/example_notebooks/training_demo.py)
- **Description**: End-to-end demo on creating config files and running training, inference, and evaluation using sleap-nn APIs.

### 2. Augmentation Guide Notebook
- **File**: [augmentation_guide.py](https://github.com/talmolab/sleap-nn/blob/main/example_notebooks/augmentation_guide.py)
- **Description**: Visualize the different data augmentation techniques available in sleap-nn.
