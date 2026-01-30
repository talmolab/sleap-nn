# Example Notebooks

Interactive tutorials using marimo notebooks.

---

## Accessing Notebooks

The example notebooks are in the [sleap-nn GitHub repository](https://github.com/talmolab/sleap-nn) in the `example_notebooks/` folder. They use [marimo](https://docs.marimo.io/) which provides a **sandboxed environment** that automatically handles all dependencies.

---

## Prerequisites

Install [uv](https://github.com/astral-sh/uv):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Available Notebooks

Download notebooks from [here](https://github.com/talmolab/sleap-nn/tree/main/example_notebooks).

### Training Demo

End-to-end demo on creating config files and running training, inference, and evaluation.

```bash
uvx marimo edit --sandbox training_demo.py
```

### Augmentation Guide

Visualize different data augmentation techniques available in sleap-nn.

```bash
uvx marimo edit --sandbox augmentation_guide.py
```

### Receptive Field Guide

Visualize how receptive field changes with different config parameters.

```bash
uvx marimo edit --sandbox receptive_field_guide.py
```

---

## Running Notebooks

Each notebook runs in a sandboxed environment - no need to install sleap-nn separately:

```bash
# Navigate to your notebooks folder
cd example_notebooks/

# Run any notebook
uvx marimo edit --sandbox <notebook_name>.py
```

The `--sandbox` flag creates an isolated environment with all required dependencies.
