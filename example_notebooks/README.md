# Example Notebooks

This document provides information on how to access and use the example notebooks from the sleap-nn GitHub repository.

## Accessing Example Notebooks

The example notebooks are available in the [sleap-nn GitHub repository](https://github.com/talmolab/sleap-nn) in the `example_notebooks/` folder. These notebooks are created with [marimo](https://docs.marimo.io/) and provide interactive tutorials for various sleap-nn workflows.

## Getting Started with Marimo

The example notebooks use marimo, which provides a **sandboxed environment** that automatically handles all dependencies. You don't need to create a separate sleap-nn environment.

### Prerequisites

**Step-1 : Install [uv](https://github.com/astral-sh/uv)**: 
Install [`uv`](https://github.com/astral-sh/uv) first - a fast Python package manager:
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```


**Step-2 : Run the marimo notebooks**

Download the notebooks from [here](https://github.com/talmolab/sleap-nn/tree/main/example_notebooks)!

(i) Training Demo Notebook

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/talmolab/sleap-nn/blob/main/example_notebooks/training_demo.py)

- **Description**: End-to-end **top-down** demo — downloads a sample dataset, trains a centroid + centered-instance model, shows how to download the trained model artifacts for local inference, and runs the new inference API (`sleap_nn.inference.predict`) on a fresh video.
- **Runs on [molab](https://molab.marimo.io)** with a GPU: open the badge above and enable the GPU from the notebook-specs button in the app header. Or run locally:

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