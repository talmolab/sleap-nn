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
from loguru import logger

from sleap_nn.architectures.model import Model
from sleap_nn.config.training_job_config import TrainingJobConfig


def get_keras_first_layer_channels(h5_path: str) -> Optional[int]:
    """Extract the number of input channels from the first layer of a Keras model.

    Args:
        h5_path: Path to the .h5 model file

    Returns:
        Number of input channels in the first layer, or None if not found
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Look for the first convolutional layer weights
            kernel_weights = []

            def collect_kernel_weights(name, obj):
                if isinstance(obj, h5py.Dataset) and name.startswith("model_weights/"):
                    # Skip optimizer weights
                    if "optimizer_weights" in name:
                        return

                    # Look for kernel weights (not bias)
                    if "kernel" in name and obj.ndim == 4:
                        kernel_weights.append((name, obj.shape))

            # Visit all items to collect kernel weights
            f.visititems(collect_kernel_weights)

            if not kernel_weights:
                return None

            # Look for the known first layer patterns (stem0_conv0 or stack0_enc0_conv0)
            for name, shape in kernel_weights:
                input_channels = shape[2]
                layer_name = name.split("/")[1] if len(name.split("/")) > 1 else name

                # Check for the known first layer patterns
                if "stem0_conv0" in layer_name or "stack0_enc0_conv0" in layer_name:
                    logger.info(
                        f"Found first layer '{name}' with {input_channels} input channels"
                    )
                    return input_channels

            # If no known first layer patterns are found, return None
            logger.warning(
                f"No known first layer patterns (stem0_conv0 or stack0_enc0_conv0) found in {h5_path}"
            )
            return None

    except Exception as e:
        logger.warning(f"Could not extract first layer channels from {h5_path}: {e}")
        return None


def update_backbone_in_channels(backbone_config, keras_in_channels: int):
    """Update the backbone configuration's in_channels if it's different from the Keras model.

    Args:
        backbone_config: The backbone configuration object
        keras_in_channels: Number of input channels from the Keras model
    """
    if backbone_config.in_channels != keras_in_channels:
        logger.info(
            f"Updating backbone in_channels from {backbone_config.in_channels} to {keras_in_channels}"
        )
        backbone_config.in_channels = keras_in_channels

    return backbone_config


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
    """Parse a Keras layer path to extract basic information.

    Args:
        layer_path: Full path like "model_weights/stack0_enc0_conv0/stack0_enc0_conv0/kernel:0"

    Returns:
        Dictionary with parsed information:
        - layer_name: Base layer name (e.g., "stack0_enc0_conv0")
        - weight_type: "kernel" or "bias"
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
    }

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

    # Get all PyTorch parameters with their shapes
    pytorch_params = {}
    for name, param in pytorch_model.named_parameters():
        pytorch_params[name] = param.shape

    # For each legacy weight, find the corresponding PyTorch parameter
    for legacy_path, weight in legacy_weights.items():
        # Extract the layer name from the legacy path
        # Legacy path format: "model_weights/stack0_enc0_conv0/stack0_enc0_conv0/kernel:0"
        clean_path = legacy_path.replace("model_weights/", "")
        parts = clean_path.split("/")

        if len(parts) < 2:
            continue

        layer_name = parts[0]  # e.g., "stack0_enc0_conv0" or "CentroidConfmapsHead_0"
        weight_name = parts[-1]  # e.g., "kernel:0" or "bias:0"

        # Convert Keras weight type to PyTorch weight type
        weight_type = "weight" if "kernel" in weight_name else "bias"

        # For head layers, strip numeric suffixes (e.g., "CentroidConfmapsHead_0" -> "CentroidConfmapsHead")
        # This handles cases where Keras uses suffixes like _0, _1, etc.
        if "Head" in layer_name:
            # Remove trailing _N where N is a number
            import re

            layer_name_clean = re.sub(r"_\d+$", "", layer_name)
        else:
            layer_name_clean = layer_name

        # Find the PyTorch parameter that contains this layer name
        # PyTorch names will be like: "backbone.enc.encoder_stack.0.blocks.0.stack0_enc0_conv0.weight"
        matching_pytorch_name = None

        for pytorch_name in pytorch_params.keys():
            # Check if the PyTorch parameter name contains the layer name (or cleaned layer name for heads)
            # and has the correct weight type
            search_name = layer_name_clean if "Head" in layer_name else layer_name
            if search_name in pytorch_name and pytorch_name.endswith(f".{weight_type}"):
                # For kernel weights, we need to check shape after conversion
                if weight_type == "weight" and weight.ndim == 4:
                    # Convert Keras kernel to PyTorch format for shape comparison
                    if "trans_conv" in legacy_path:
                        converted_weight = convert_keras_to_pytorch_conv2d_transpose(
                            weight
                        )
                    else:
                        converted_weight = convert_keras_to_pytorch_conv2d(weight)
                    shape_to_check = converted_weight.shape
                elif weight_type == "weight" and weight.ndim == 2:
                    # for linear weights, we need to transpose the shape
                    shape_to_check = weight.shape[::-1]
                else:
                    # Bias weights don't need conversion
                    shape_to_check = weight.shape

                # Verify shape compatibility
                if shape_to_check == pytorch_params[pytorch_name]:
                    matching_pytorch_name = pytorch_name
                    break

        if matching_pytorch_name:
            mapping[legacy_path] = matching_pytorch_name
        else:
            logger.warning(f"No matching PyTorch parameter found for {legacy_path}")

    # Log mapping results
    if not mapping:
        logger.info(
            f"No mappings could be created between legacy weights and PyTorch model. "
            f"Legacy weights: {len(legacy_weights)}, PyTorch parameters: {len(pytorch_params)}"
        )
    else:
        logger.info(
            f"Successfully mapped {len(mapping)}/{len(legacy_weights)} legacy weights to PyTorch parameters"
        )

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
        try:
            mapping = map_legacy_to_pytorch_layers(legacy_weights, pytorch_model)
        except Exception as e:
            logger.error(f"Failed to create weight mappings: {e}")
            return

    # Apply weights
    loaded_count = 0
    errors = []

    for legacy_path, pytorch_name in mapping.items():
        if legacy_path not in legacy_weights:
            logger.warning(f"Legacy weight not found: {legacy_path}")
            continue

        weight = legacy_weights[legacy_path]
        info = parse_keras_layer_name(legacy_path)

        # Convert weight format if needed
        if info["weight_type"] == "kernel" and weight.ndim == 4:
            if "trans_conv" in legacy_path:
                weight = convert_keras_to_pytorch_conv2d_transpose(weight)
            else:
                weight = convert_keras_to_pytorch_conv2d(weight)
        elif info["weight_type"] == "kernel" and weight.ndim != 4:
            # for linear weights, we need to transpose the shape
            weight = torch.from_numpy(weight.transpose(1, 0)).float()
        else:
            # Bias weights don't need conversion
            weight = torch.from_numpy(weight).float()
        # Set the parameter using state_dict
        try:
            state_dict = pytorch_model.state_dict()
            if pytorch_name not in state_dict:
                logger.warning(f"PyTorch parameter not found: {pytorch_name}")
                continue

            # Check shape compatibility
            pytorch_shape = state_dict[pytorch_name].shape
            if weight.shape != pytorch_shape:
                logger.warning(
                    f"Shape mismatch for {pytorch_name}: "
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

            loaded_count += 1

        except Exception as e:
            error_msg = f"Error loading {pytorch_name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    # Log summary
    if loaded_count == 0:
        logger.info(
            f"No weights were successfully loaded. "
            f"Attempted to load {len(mapping)} weights, but all failed."
        )
    else:
        logger.info(
            f"Successfully loaded {loaded_count}/{len(mapping)} weights from legacy model"
        )

    # Log any errors that occurred
    if errors:
        logger.info(
            f"Weight loading completed with {len(errors)} errors: {'; '.join(errors[:5])}"
        )

    # Verify all loaded weights by comparing means
    logger.info("Verifying weight assignments...")
    verification_errors = []

    for legacy_path, pytorch_name in mapping.items():
        if legacy_path not in legacy_weights:
            continue

        try:
            original_weight = legacy_weights[legacy_path]
            info = parse_keras_layer_name(legacy_path)

            if info["weight_type"] == "kernel" and original_weight.ndim == 4:
                # Convert Keras to PyTorch format
                torch_weight = convert_keras_to_pytorch_conv2d(original_weight)
                # Keras: (H, W, C_in, C_out), PyTorch: (C_out, C_in, H, W)
                keras_cout = original_weight.shape[-1]
                torch_cout = torch_weight.shape[0]
                assert (
                    keras_cout == torch_cout
                ), f"Output channel mismatch: {keras_cout} vs {torch_cout}"

                # Check each output channel
                channel_errors = []
                for i in range(keras_cout):
                    keras_ch_mean = np.mean(original_weight[..., i])
                    torch_ch_mean = torch.mean(torch_weight[i]).item()
                    diff = abs(keras_ch_mean - torch_ch_mean)
                    if diff > 1e-6:
                        channel_errors.append(
                            f"channel {i}: keras={keras_ch_mean:.6f}, torch={torch_ch_mean:.6f}, diff={diff:.6e}"
                        )

                if channel_errors:
                    message = f"Channel verification failed for {pytorch_name}: {'; '.join(channel_errors)}"
                    logger.error(message)
                    verification_errors.append(message)
            elif info["weight_type"] == "kernel" and original_weight.ndim == 2:
                # for linear weights, we need to transpose the shape
                keras_mean = np.mean(original_weight.transpose(1, 0))
                torch_mean = torch.mean(
                    torch.from_numpy(original_weight.transpose(1, 0)).float()
                ).item()
                diff = abs(keras_mean - torch_mean)
                if diff > 1e-6:
                    message = f"Weight verification failed for {pytorch_name} linear): keras={keras_mean:.6f}, torch={torch_mean:.6f}, diff={diff:.6e}"
            else:
                # Bias : just compare all values
                keras_mean = np.mean(original_weight)
                torch_mean = torch.mean(
                    torch.from_numpy(original_weight).float()
                ).item()
                diff = abs(keras_mean - torch_mean)
                if diff > 1e-6:
                    message = f"Weight verification failed for {pytorch_name} bias): keras={keras_mean:.6f}, torch={torch_mean:.6f}, diff={diff:.6e}"
                    logger.error(message)
                    verification_errors.append(message)

        except Exception as e:
            error_msg = f"Error verifying {pytorch_name}: {e}"
            logger.error(error_msg)
            verification_errors.append(error_msg)

    if not verification_errors:
        logger.info("✓ All weight assignments verified successfully")


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

    # Check if there's a corresponding .h5 file to extract input channels
    model_dir = config_path.parent
    h5_path = model_dir / "best_model.h5"

    if h5_path.exists():
        keras_in_channels = get_keras_first_layer_channels(str(h5_path))
        if keras_in_channels is not None:
            backbone_config = update_backbone_in_channels(
                backbone_config, keras_in_channels
            )

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
    elif head_configs.multi_class_topdown is not None:
        model_type = "multi_class_topdown"
        active_head_config = head_configs.multi_class_topdown
    elif head_configs.multi_class_bottomup is not None:
        model_type = "multi_class_bottomup"
        active_head_config = head_configs.multi_class_bottomup
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
    model.eval()

    # Load weights if requested
    if load_weights:
        h5_path = model_dir / "best_model.h5"
        if h5_path.exists():
            load_legacy_model_weights(model, str(h5_path))

        else:
            message = f"Model weights not found at {h5_path}"
            logger.error(message)
            raise ValueError(message)

    return model
