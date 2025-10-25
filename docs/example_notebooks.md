# Example Notebooks

This page provides information on how to access and use the example notebooks from the sleap-nn GitHub repository.

## Accessing Example Notebooks

The example notebooks are available in the [sleap-nn GitHub repository](https://github.com/talmolab/sleap-nn) in the `example_notebooks/` folder. These notebooks are created with [marimo](https://docs.marimo.io/) and provide interactive tutorials for various sleap-nn workflows.

## Getting Started with Marimo

The example notebooks use marimo, which provides a **sandboxed environment** that automatically handles all dependencies. You don't need to create a separate sleap-nn environment.

### Prerequisites

**Install [uv](https://github.com/astral-sh/uv)**: 
!!! info "First Time Setup"
      Install [`uv`](https://github.com/astral-sh/uv) first - a fast Python package manager:
      ```bash
      # macOS/Linux
      curl -LsSf https://astral.sh/uv/install.sh | sh
      
      # Windows
      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
      ```

### Setup

Download the notebooks from [here](https://github.com/talmolab/sleap-nn/tree/main/example_notebooks)!

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
