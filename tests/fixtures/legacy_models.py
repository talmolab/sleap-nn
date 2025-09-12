"""Fixtures for legacy SLEAP model files.

This module provides paths and utilities for accessing legacy SLEAP models
that were trained with the TensorFlow/Keras backend. These models are used
for testing the legacy model import functionality.

The models are stored in tests/assets/legacy_models/ and include:
- Centroid models for detecting animal centers
- Centered instance models for single animal pose in cropped frames
- Single instance models for single animal pose in full frames
- Bottom-up models for multi-animal pose estimation
- Multiclass bottom-up models for multi-animal pose with identity tracking
- Multiclass top-down models for multi-animal pose with identity tracking

Each model directory contains:
- best_model.h5: Trained Keras model weights (only UNet backbone is supported)
- training_config.json: Full configuration used for training
- initial_config.json: Initial model architecture configuration
- training_log.csv: Training metrics over epochs
- labels_gt.*.slp: Ground truth labels used for training/validation
- labels_pr.*.slp: Model predictions on training/validation sets (when available)
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import json


# Base path for all legacy models
LEGACY_MODELS_DIR = Path(__file__).parent.parent / "assets" / "legacy_models"


@pytest.fixture
def sleap_centroid_model_path() -> Path:
    """Path to a trained centroid model.

    This model detects animal centroids in 384x384 grayscale images.
    Architecture: UNet with 3 down blocks, 2 up blocks
    Heads: CentroidConfmapsHead + OffsetRefinementHead
    """
    return LEGACY_MODELS_DIR / "minimal_instance.UNet.centroid"


@pytest.fixture
def sleap_centered_instance_model_path() -> Path:
    """Path to a trained centered instance model.

    This model predicts poses for cropped single animals in 96x96 grayscale images.
    Architecture: UNet with 3 down blocks, 2 up blocks
    Heads: CenteredInstanceConfmapsHead + OffsetRefinementHead
    Parts: head, thorax
    """
    return LEGACY_MODELS_DIR / "minimal_instance.UNet.centered_instance"


@pytest.fixture
def sleap_single_instance_model_path() -> Path:
    """Path to a trained single instance model.

    This model predicts poses for single animals in 192x192 RGB images.
    Architecture: UNet with 2 down blocks, 0 up blocks
    Heads: SingleInstanceConfmapsHead
    Parts: left, middle, right (robot markers)
    """
    return LEGACY_MODELS_DIR / "minimal_robot.UNet.single_instance"


@pytest.fixture
def sleap_bottomup_model_path() -> Path:
    """Path to a trained bottom-up model.

    This model predicts multi-animal poses in 384x384 grayscale images.
    Architecture: UNet with 3 down blocks, 2 up blocks
    Heads: MultiInstanceConfmapsHead + PartAffinityFieldsHead
    Parts: head, thorax
    """
    return LEGACY_MODELS_DIR / "minimal_instance.UNet.bottomup"


@pytest.fixture
def sleap_bottomup_multiclass_model_path() -> Path:
    """Path to a trained multiclass bottom-up model.

    This model predicts multi-animal poses with identity classes in 384x384 grayscale images.
    Architecture: UNet with 3 down blocks, 2 up blocks
    Heads: MultiInstanceConfmapsHead + PartAffinityFieldsHead + ClassMapsHead
    Parts: node1, node2
    Classes: 3 identity classes
    """
    return LEGACY_MODELS_DIR / "min_tracks_2node.UNet.bottomup_multiclass"


@pytest.fixture
def sleap_topdown_multiclass_model_path() -> Path:
    """Path to a trained multiclass top-down model.

    This model predicts multi-animal poses with identity classes in 384x384 grayscale images.
    Architecture: UNet with 3 down blocks, 2 up blocks
    Heads: CentroidConfmapsHead + ClassVectorsHead + CenteredInstanceConfmapsHead
    Parts: node1, node2
    Classes: 3 identity classes
    """
    return LEGACY_MODELS_DIR / "min_tracks_2node.UNet.topdown_multiclass"


def load_legacy_config(model_path: Path) -> Dict[str, Any]:
    """Load the training configuration for a legacy model.

    Args:
        model_path: Path to the model directory containing training_config.json

    Returns:
        Dictionary containing the full training configuration
    """
    config_path = model_path / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def get_model_architecture(model_path: Path) -> Dict[str, Any]:
    """Extract architecture parameters from a legacy model config.

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary containing architecture parameters like:
        - backbone_type: "unet"
        - down_blocks: Number of downsampling blocks
        - up_blocks: Number of upsampling blocks
        - filters: Initial number of filters
        - heads: List of output head configurations
    """
    config = load_legacy_config(model_path)
    model_config = config.get("model", {})

    # Extract backbone config
    backbone = model_config.get("backbone", {})
    arch_config = {
        "backbone_type": backbone.get("backbone_type", "unet"),
        "down_blocks": backbone.get("down_blocks", 3),
        "up_blocks": backbone.get("up_blocks", 2),
        "filters": backbone.get("filters", 16),
        "filters_rate": backbone.get("filters_rate", 1.5),
    }

    # Extract head configs
    heads_config = model_config.get("heads", {})
    arch_config["heads"] = []

    for head_type, head_params in heads_config.items():
        if head_params:  # Skip None values
            arch_config["heads"].append({"type": head_type, "config": head_params})

    return arch_config
