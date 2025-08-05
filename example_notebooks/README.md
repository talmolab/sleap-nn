# Example Notebooks

This folder contains all tutorial notebooks created with marimo. These notebooks are **sandboxed**, which means you don't need to create a separate sleap-nn environment. The sandbox environment will handle all dependencies automatically. Just install `uv`, add `marimo`, and run the notebooks with the commands below. 

## Getting Started with Marimo

### Prerequisites

1. **Install [uv](https://github.com/astral-sh/uv)**: 
   ```bash
   pip install uv
   ```

2. **Create a new folder and initialize uv**:
   ```bash
   uv init
   ```

### Setup

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

## Available Tutorials

- **augmentation_guide.py**: Visualize the different data augmentation techniques in SLEAP-NN

