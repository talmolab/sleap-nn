"""Utilities for loading legacy SLEAP models.

This module provides functions to convert SLEAP models trained with the
TensorFlow/Keras backend to PyTorch format compatible with sleap-nn.
"""

import h5py
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional


def convert_keras_to_pytorch_conv2d(keras_weight: np.ndarray) -> torch.Tensor:
    """Convert Keras Conv2D weights to PyTorch format.

    Args:
        keras_weight: Numpy array with shape (H, W, C_in, C_out) from Keras

    Returns:
        PyTorch tensor with shape (C_out, C_in, H, W)
    """
    if keras_weight.ndim != 4:
        raise ValueError(
            f"Expected 4D array for Conv2D weights, got shape {keras_weight.shape}"
        )

    # Keras: (H, W, C_in, C_out) -> PyTorch: (C_out, C_in, H, W)
    pytorch_weight = keras_weight.transpose(3, 2, 0, 1)
    return torch.from_numpy(pytorch_weight.copy()).float()


def convert_keras_to_pytorch_conv2d_transpose(keras_weight: np.ndarray) -> torch.Tensor:
    """Convert Keras Conv2DTranspose weights to PyTorch format.

    Args:
        keras_weight: Numpy array with shape (H, W, C_out, C_in) from Keras

    Returns:
        PyTorch tensor with shape (C_in, C_out, H, W)

    Note:
        Keras stores transposed conv weights differently than regular conv.
    """
    if keras_weight.ndim != 4:
        raise ValueError(
            f"Expected 4D array for Conv2DTranspose weights, got shape {keras_weight.shape}"
        )

    # Keras: (H, W, C_out, C_in) -> PyTorch: (C_in, C_out, H, W)
    pytorch_weight = keras_weight.transpose(3, 2, 0, 1)
    return torch.from_numpy(pytorch_weight.copy()).float()


def load_keras_weights(h5_path: str) -> Dict[str, np.ndarray]:
    """Load all weights from a Keras HDF5 model file.

    Args:
        h5_path: Path to the .h5 model file

    Returns:
        Dictionary mapping layer paths to weight arrays
    """
    weights = {}

    with h5py.File(h5_path, "r") as f:

        def extract_weights(name, obj):
            if isinstance(obj, h5py.Dataset) and name.startswith("model_weights/"):
                # Skip optimizer weights
                if "optimizer_weights" in name:
                    return
                weights[name] = obj[:]

        f.visititems(extract_weights)

    return weights


def parse_keras_layer_name(layer_path: str) -> Dict[str, Any]:
    """Parse a Keras layer path to extract information.

    Args:
        layer_path: Full path like "model_weights/stack0_enc0_conv0/stack0_enc0_conv0/kernel:0"

    Returns:
        Dictionary with parsed information:
        - layer_name: Base layer name (e.g., "stack0_enc0_conv0")
        - weight_type: "kernel" or "bias"
        - is_encoder: True if encoder layer
        - is_decoder: True if decoder layer
        - is_head: True if output head layer
        - block_idx: Block index if applicable
        - conv_idx: Conv index within block if applicable
    """
    # Remove model_weights prefix and split
    clean_path = layer_path.replace("model_weights/", "")
    parts = clean_path.split("/")

    if len(parts) < 2:
        raise ValueError(f"Invalid layer path: {layer_path}")

    layer_name = parts[0]
    weight_name = parts[-1]  # e.g., "kernel:0" or "bias:0"

    info = {
        "layer_name": layer_name,
        "weight_type": "kernel" if "kernel" in weight_name else "bias",
        "is_encoder": "enc" in layer_name,
        "is_decoder": "dec" in layer_name,
        "is_head": "Head" in layer_name,
        "block_idx": None,
        "conv_idx": None,
    }

    # Extract indices for encoder/decoder layers
    if info["is_encoder"] or info["is_decoder"]:
        # Pattern: stack0_enc0_conv1 or stack0_dec0_s8_to_s4_refine_conv0
        import re

        if info["is_encoder"]:
            match = re.search(r"enc(\d+)_conv(\d+)", layer_name)
            if match:
                info["block_idx"] = int(match.group(1))
                info["conv_idx"] = int(match.group(2))
        elif info["is_decoder"]:
            # Decoder naming is more complex
            match = re.search(r"dec(\d+)_.*conv(\d+)", layer_name)
            if match:
                info["block_idx"] = int(match.group(1))
                info["conv_idx"] = int(match.group(2))

    return info
