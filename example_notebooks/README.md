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

2. **Download and run the marimo notebooks**

(i) Training Demo Notebook
- **Description**: End-to-end demo on creating config files and running training, inference, and evaluation using sleap-nn APIs.

**Note:** Marimo notebooks are designed for a seamless, automated workflow. After you select the model type, all cells will execute automatically—no need to run them one by one. Training will only start when you click the **Run Training** button, giving you full control over when to begin model training.

Download the notebook [training_demo.py](https://github.com/talmolab/sleap-nn/blob/main/example_notebooks/training_demo.py) and run the below command:
```bash
   uvx marimo edit --sandbox <training_demo.py>
```

(ii) Augmentation Guide Notebook
- **Description**: Visualize the different data augmentation techniques available in sleap-nn.
Download the notebook: [augmentation_guide.py](https://github.com/talmolab/sleap-nn/blob/main/example_notebooks/augmentation_guide.py) and run the below command:
```bash
   uvx marimo edit --sandbox <augmentation_guide.py>
```

(iii) Receptive Field Guide Notebook
- **Description**: Visualize how the receptive field could be set by changing the config parameters.
Download the notebook: [augmentation_guide.py](https://github.com/talmolab/sleap-nn/blob/main/example_notebooks/augmentation_guide.py) and run the below command:
```bash
   uvx marimo edit --sandbox <receptive_field_guide.py>
```

