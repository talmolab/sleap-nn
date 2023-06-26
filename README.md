# sleap-nn
Neural network backend for training and inference for animal pose estimation.

## Development

1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation.html) ([mambaforge](https://github.com/conda-forge/miniforge#mambaforge) is a good replacement for miniconda).
2. Create the environment:
    *With GPU (Windows/Linux):*
    ```
    mamba env create -f environment.yml
    ```
    *With CPU (Windows/Linux/Intel Mac):*
    ```
    mamba env create -f environment_cpu.yml
    ```
    *Apple Silicon (M1/M2 Mac):*
    ```
    mamba env create -f environment_osx-arm64.yml
    ```
3. Make sure you can activate the environment:
    ```
    mamba activate sleap-nn
    ```
 4. And run the tests:
    ```
    pytest tests
    ```