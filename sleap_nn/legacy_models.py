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
                if weight_type == "weight":
                    # Convert Keras kernel to PyTorch format for shape comparison
                    if "trans_conv" in legacy_path:
                        converted_weight = convert_keras_to_pytorch_conv2d_transpose(
                            weight
                        )
                    else:
                        converted_weight = convert_keras_to_pytorch_conv2d(weight)
                    shape_to_check = converted_weight.shape
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

            if info["weight_type"] == "kernel":
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
                    verification_errors.append(
                        f"{pytorch_name}: {'; '.join(channel_errors)}"
                    )

            else:
                # Bias: just compare all values
                keras_mean = np.mean(original_weight)
                torch_mean = torch.mean(
                    torch.from_numpy(original_weight).float()
                ).item()
                diff = abs(keras_mean - torch_mean)
                if diff > 1e-6:
                    verification_errors.append(
                        f"{pytorch_name} (bias): keras={keras_mean:.6f}, torch={torch_mean:.6f}, diff={diff:.6e}"
                    )

        except Exception as e:
            error_msg = f"Error verifying {pytorch_name}: {e}"
            logger.error(error_msg)
            verification_errors.append(error_msg)

    if verification_errors:
        logger.error(
            f"Weight verification completed with {len(verification_errors)} errors"
        )
        for error in verification_errors[:5]:  # Show first 5 errors
            logger.error(f"  {error}")
    else:
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
    try:
        model = create_model_from_legacy_config(str(model_dir))
        model.eval()
    except Exception as e:
        logger.error(f"Failed to create model from legacy config: {e}")
        raise

    # Load weights if requested
    if load_weights:
        h5_path = model_dir / "best_model.h5"
        if h5_path.exists():
            try:
                load_legacy_model_weights(model, str(h5_path))
            except Exception as e:
                logger.error(f"Failed to load legacy weights: {e}")
        else:
            logger.warning(f"Model weights not found at {h5_path}")

    return model
