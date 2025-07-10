"""Utilities for loading legacy SLEAP models.

This module provides functions to convert SLEAP models trained with the
TensorFlow/Keras backend to PyTorch format compatible with sleap-nn.
"""

import h5py
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
from omegaconf import OmegaConf
import re

from sleap_nn.architectures.model import Model
from sleap_nn.config.training_job_config import TrainingJobConfig


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
            # Check for middle blocks first
            if "middle" in layer_name:
                # Middle blocks don't follow the standard pattern
                # Extract conv index from the layer name
                match = re.search(r"conv(\d+)", layer_name)
                if match:
                    info["conv_idx"] = int(match.group(1))
                # Block idx will remain None for middle blocks
            else:
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


def map_legacy_to_pytorch_layers(
    legacy_weights: Dict[str, np.ndarray], pytorch_model: torch.nn.Module
) -> Dict[str, str]:
    """Create mapping between legacy Keras layers and PyTorch model layers.

    Args:
        legacy_weights: Dictionary of legacy weights from load_keras_weights()
        pytorch_model: PyTorch model instance to map to

    Returns:
        Dictionary mapping legacy layer paths to PyTorch parameter names
    """
    mapping = {}

    # Get all PyTorch layers with their shapes
    pytorch_params = {}
    for name, param in pytorch_model.named_parameters():
        pytorch_params[name] = param.shape

    # Parse legacy layers
    legacy_info = {}
    for path, weight in legacy_weights.items():
        info = parse_keras_layer_name(path)
        info["shape"] = weight.shape
        info["path"] = path
        legacy_info[path] = info

    # Map encoder layers
    encoder_layers = []
    for path, info in legacy_info.items():
        if info["is_encoder"]:
            # Use a high value for None block_idx so middle blocks sort last
            block_idx = info["block_idx"] if info["block_idx"] is not None else 999
            conv_idx = info["conv_idx"] if info["conv_idx"] is not None else 0
            encoder_layers.append((block_idx, conv_idx, path, info))
    encoder_layers.sort()

    # Map decoder layers
    decoder_layers = []
    for path, info in legacy_info.items():
        if info["is_decoder"]:
            block_idx = info["block_idx"] if info["block_idx"] is not None else 0
            conv_idx = info["conv_idx"] if info["conv_idx"] is not None else 0
            decoder_layers.append((block_idx, conv_idx, path, info))
    decoder_layers.sort()

    # Map head layers
    head_layers = [
        (path, info) for path, info in legacy_info.items() if info["is_head"]
    ]

    # Separate middle/bottleneck blocks from regular encoder blocks
    middle_blocks = []
    regular_encoder_blocks = []

    for block_idx, conv_idx, path, info in encoder_layers:
        # Check if this is a middle block (usually has "middle" in the name)
        if "middle" in info["layer_name"]:
            middle_blocks.append((block_idx, conv_idx, path, info))
        elif block_idx is not None:
            regular_encoder_blocks.append((block_idx, conv_idx, path, info))

    # Mapping logic for regular encoder layers
    for block_idx, conv_idx, path, info in regular_encoder_blocks:
        weight_type = info["weight_type"]

        # In PyTorch UNet:
        # - Stack 0: blocks.0 and blocks.2
        # - Stack 1+: blocks.1 and blocks.3

        # PyTorch uses "weight" instead of "kernel"
        pytorch_weight_type = "weight" if weight_type == "kernel" else weight_type

        if block_idx == 0:
            # First encoder stack
            if conv_idx == 0:
                pytorch_name = (
                    f"backbone.enc.encoder_stack.0.blocks.0.{pytorch_weight_type}"
                )
            elif conv_idx == 1:
                pytorch_name = (
                    f"backbone.enc.encoder_stack.0.blocks.2.{pytorch_weight_type}"
                )
        else:
            # Subsequent encoder stacks
            if conv_idx == 0:
                pytorch_name = f"backbone.enc.encoder_stack.{block_idx}.blocks.1.{pytorch_weight_type}"
            elif conv_idx == 1:
                pytorch_name = f"backbone.enc.encoder_stack.{block_idx}.blocks.3.{pytorch_weight_type}"

        # Check if this parameter exists in the PyTorch model
        if pytorch_name in pytorch_params:
            mapping[path] = pytorch_name

    # Map middle blocks
    # Group middle blocks by layer name to handle them in order
    middle_layers = {}
    for _, _, path, info in middle_blocks:
        layer_name = info["layer_name"]
        if layer_name not in middle_layers:
            middle_layers[layer_name] = []
        middle_layers[layer_name].append((path, info))

    # Sort layers - in legacy models, "expand" comes before "contract"
    sorted_layer_names = sorted(
        middle_layers.keys(), key=lambda x: (0 if "expand" in x else 1, x)
    )

    # Middle blocks in PyTorch start after regular encoder blocks
    middle_idx = (
        len(regular_encoder_blocks) // 2
    )  # Divide by 2 because each block has kernel and bias

    for layer_name in sorted_layer_names:
        for path, info in middle_layers[layer_name]:
            weight_type = info["weight_type"]
            pytorch_weight_type = "weight" if weight_type == "kernel" else weight_type

            # Middle blocks use blocks.1 in PyTorch
            pytorch_name = f"backbone.enc.encoder_stack.{middle_idx}.blocks.1.{pytorch_weight_type}"

            if pytorch_name in pytorch_params:
                mapping[path] = pytorch_name
                # Only increment after both weight and bias are mapped
                if weight_type == "bias":
                    middle_idx += 1

    # Mapping logic for decoder layers
    # In PyTorch, decoder stacks are numbered in reverse order
    max_decoder_block = (
        max([block_idx for block_idx, _, _, _ in decoder_layers])
        if decoder_layers
        else 0
    )

    for block_idx, conv_idx, path, info in decoder_layers:
        weight_type = info["weight_type"]
        pytorch_weight_type = "weight" if weight_type == "kernel" else weight_type

        # Reverse the block index for PyTorch
        pytorch_block_idx = max_decoder_block - block_idx

        # In PyTorch decoder:
        # - First conv in block: blocks.1
        # - Second conv in block: blocks.3
        if conv_idx == 0:
            pytorch_name = f"backbone.dec.decoder_stack.{pytorch_block_idx}.blocks.1.{pytorch_weight_type}"
        elif conv_idx == 1:
            pytorch_name = f"backbone.dec.decoder_stack.{pytorch_block_idx}.blocks.3.{pytorch_weight_type}"

        if pytorch_name in pytorch_params:
            mapping[path] = pytorch_name

    # Mapping logic for head layers
    # Filter out unsupported heads like OffsetRefinementHead
    supported_heads = []
    for path, info in head_layers:
        layer_name = info["layer_name"]
        # Skip offset refinement heads as they're not supported in current architecture
        if "OffsetRefinement" not in layer_name:
            supported_heads.append((path, info))

    head_idx = 0
    current_head_type = None
    for path, info in supported_heads:
        weight_type = info["weight_type"]
        pytorch_weight_type = "weight" if weight_type == "kernel" else weight_type
        layer_name = info["layer_name"]

        # Track when we move to a new head type
        head_type = layer_name.split("_")[0]  # e.g., "CentroidConfmapsHead"
        if current_head_type != head_type:
            if current_head_type is not None:
                head_idx += 1
            current_head_type = head_type

        # Most heads have a single conv2d layer at index 0
        pytorch_name = f"head_layers.{head_idx}.0.{pytorch_weight_type}"

        if pytorch_name in pytorch_params:
            mapping[path] = pytorch_name

    return mapping


def load_legacy_model_weights(
    pytorch_model: torch.nn.Module,
    h5_path: str,
    mapping: Optional[Dict[str, str]] = None,
) -> None:
    """Load legacy Keras weights into a PyTorch model.

    Args:
        pytorch_model: PyTorch model to load weights into
        h5_path: Path to the legacy .h5 model file
        mapping: Optional manual mapping of layer names. If None,
                 will attempt automatic mapping.
    """
    # Load legacy weights
    legacy_weights = load_keras_weights(h5_path)

    if mapping is None:
        # Attempt automatic mapping
        mapping = map_legacy_to_pytorch_layers(legacy_weights, pytorch_model)

    # Apply weights
    for legacy_path, pytorch_name in mapping.items():
        if legacy_path not in legacy_weights:
            print(f"Warning: Legacy weight not found: {legacy_path}")
            continue

        weight = legacy_weights[legacy_path]
        info = parse_keras_layer_name(legacy_path)

        # Convert weight format if needed
        if info["weight_type"] == "kernel":
            if "trans_conv" in legacy_path:
                weight = convert_keras_to_pytorch_conv2d_transpose(weight)
            else:
                weight = convert_keras_to_pytorch_conv2d(weight)
        else:
            # Bias weights don't need conversion
            weight = torch.from_numpy(weight).float()

        # Set the parameter using state_dict
        try:
            state_dict = pytorch_model.state_dict()
            if pytorch_name not in state_dict:
                print(f"Warning: PyTorch parameter not found: {pytorch_name}")
                continue

            # Check shape compatibility
            pytorch_shape = state_dict[pytorch_name].shape
            if weight.shape != pytorch_shape:
                print(
                    f"Warning: Shape mismatch for {pytorch_name}: "
                    f"legacy {weight.shape} vs pytorch {pytorch_shape}"
                )
                continue

            # Update the parameter in the model
            with torch.no_grad():
                param = pytorch_model
                for attr in pytorch_name.split(".")[:-1]:
                    param = getattr(param, attr)
                param_name = pytorch_name.split(".")[-1]
                setattr(param, param_name, torch.nn.Parameter(weight))

        except Exception as e:
            print(f"Error loading {pytorch_name}: {e}")


def create_model_from_legacy_config(config_path: str) -> Model:
    """Create a PyTorch model from a legacy training config.

    Args:
        config_path: Path to the legacy training_config.json file

    Returns:
        Model instance configured to match the legacy architecture
    """
    # Load config using existing functionality
    config_path = Path(config_path)
    if config_path.is_dir():
        config_path = config_path / "training_config.json"

    # Use the existing config loader
    config = TrainingJobConfig.load_sleap_config(str(config_path))

    # Determine backbone type from config
    backbone_type = "unet"  # Default for legacy models

    # Get backbone config (should be under the unet key for legacy models)
    backbone_config = config.model_config.backbone_config.unet

    # Determine model type from head configs
    head_configs = config.model_config.head_configs
    model_type = None
    active_head_config = None

    if head_configs.centroid is not None:
        model_type = "centroid"
        active_head_config = head_configs.centroid
    elif head_configs.centered_instance is not None:
        model_type = "centered_instance"
        active_head_config = head_configs.centered_instance
    elif head_configs.single_instance is not None:
        model_type = "single_instance"
        active_head_config = head_configs.single_instance
    elif head_configs.bottomup is not None:
        model_type = "bottomup"
        active_head_config = head_configs.bottomup
    else:
        raise ValueError("Could not determine model type from head configs")

    # Create model using the from_config method
    model = Model.from_config(
        backbone_type=backbone_type,
        backbone_config=backbone_config,
        head_configs=active_head_config,
        model_type=model_type,
    )

    return model


def load_legacy_model(model_dir: str, load_weights: bool = True) -> Model:
    """Load a complete legacy SLEAP model including weights.

    Args:
        model_dir: Path to the legacy model directory containing
                   training_config.json and best_model.h5
        load_weights: Whether to load the weights. If False, only
                      creates the model architecture.

    Returns:
        Model instance with loaded weights
    """
    model_dir = Path(model_dir)

    # Create model from config
    model = create_model_from_legacy_config(str(model_dir))

    # Load weights if requested
    if load_weights:
        h5_path = model_dir / "best_model.h5"
        if h5_path.exists():
            load_legacy_model_weights(model, str(h5_path))
        else:
            print(f"Warning: Model weights not found at {h5_path}")

    return model
