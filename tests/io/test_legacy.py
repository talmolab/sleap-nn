"""Tests for legacy model conversion utilities."""

import numpy as np
import torch
import torch.nn as nn
import pytest
from pathlib import Path

from sleap_nn.io.legacy import (
    convert_keras_to_pytorch_conv2d,
    convert_keras_to_pytorch_conv2d_transpose,
    load_keras_weights,
    parse_keras_layer_name,
)


class TestWeightConversion:
    """Test weight format conversion from Keras to PyTorch."""

    def test_conv2d_weight_conversion(self):
        """Test Conv2D weight conversion maintains values correctly."""
        # Create test weight in Keras format (H, W, C_in, C_out)
        keras_weight = np.random.randn(3, 3, 16, 32)

        # Convert to PyTorch format
        pytorch_weight = convert_keras_to_pytorch_conv2d(keras_weight)

        # Check shape: should be (C_out, C_in, H, W)
        assert pytorch_weight.shape == (32, 16, 3, 3)

        # Check values are preserved (just transposed)
        # Take a specific element and verify it's in the right place
        assert keras_weight[1, 2, 5, 10] == pytorch_weight[10, 5, 1, 2]

        # Test with actual PyTorch layer
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        conv.weight.data = pytorch_weight

        # Create test input
        x = torch.randn(1, 16, 10, 10)
        output = conv(x)
        assert output.shape == (1, 32, 10, 10)  # Same size with padding=1

    def test_conv2d_transpose_weight_conversion(self):
        """Test Conv2DTranspose weight conversion."""
        # Create test weight in Keras format (H, W, C_out, C_in)
        keras_weight = np.random.randn(3, 3, 32, 16)

        # Convert to PyTorch format
        pytorch_weight = convert_keras_to_pytorch_conv2d_transpose(keras_weight)

        # Check shape: should be (C_in, C_out, H, W)
        assert pytorch_weight.shape == (16, 32, 3, 3)

        # Check values are preserved
        assert keras_weight[1, 2, 10, 5] == pytorch_weight[5, 10, 1, 2]

        # Test with actual PyTorch layer
        conv_transpose = nn.ConvTranspose2d(16, 32, kernel_size=3)
        conv_transpose.weight.data = pytorch_weight

        # Create test input
        x = torch.randn(1, 16, 10, 10)
        output = conv_transpose(x)
        assert output.shape[1] == 32  # Output channels

    def test_invalid_weight_shapes(self):
        """Test that invalid weight shapes raise errors."""
        # Wrong number of dimensions
        with pytest.raises(ValueError):
            convert_keras_to_pytorch_conv2d(np.random.randn(3, 3, 16))

        with pytest.raises(ValueError):
            convert_keras_to_pytorch_conv2d_transpose(np.random.randn(3, 3))


class TestLayerNameParsing:
    """Test parsing of Keras layer names."""

    def test_encoder_layer_parsing(self):
        """Test parsing encoder layer names."""
        info = parse_keras_layer_name(
            "model_weights/stack0_enc0_conv1/stack0_enc0_conv1/kernel:0"
        )

        assert info["layer_name"] == "stack0_enc0_conv1"
        assert info["weight_type"] == "kernel"
        assert info["is_encoder"] is True
        assert info["is_decoder"] is False
        assert info["is_head"] is False
        assert info["block_idx"] == 0
        assert info["conv_idx"] == 1

    def test_decoder_layer_parsing(self):
        """Test parsing decoder layer names."""
        info = parse_keras_layer_name(
            "model_weights/stack0_dec0_s8_to_s4_refine_conv0/stack0_dec0_s8_to_s4_refine_conv0/bias:0"
        )

        assert info["layer_name"] == "stack0_dec0_s8_to_s4_refine_conv0"
        assert info["weight_type"] == "bias"
        assert info["is_encoder"] is False
        assert info["is_decoder"] is True
        assert info["is_head"] is False
        assert info["block_idx"] == 0
        assert info["conv_idx"] == 0

    def test_head_layer_parsing(self):
        """Test parsing head layer names."""
        info = parse_keras_layer_name(
            "model_weights/CentroidConfmapsHead_0/CentroidConfmapsHead_0/kernel:0"
        )

        assert info["layer_name"] == "CentroidConfmapsHead_0"
        assert info["weight_type"] == "kernel"
        assert info["is_encoder"] is False
        assert info["is_decoder"] is False
        assert info["is_head"] is True


class TestWeightLoading:
    """Test loading weights from actual legacy model files."""

    def test_load_centroid_weights(self, centroid_model_path):
        """Test loading weights from centroid model."""
        h5_path = centroid_model_path / "best_model.h5"
        weights = load_keras_weights(str(h5_path))

        # Check that we loaded weights
        assert len(weights) > 0

        # Check for expected layers
        encoder_layers = [k for k in weights.keys() if "enc" in k and "kernel" in k]
        assert len(encoder_layers) > 0

        # Check weight shapes
        for path, weight in weights.items():
            info = parse_keras_layer_name(path)

            if info["weight_type"] == "kernel":
                if info["is_head"]:
                    # Head layers use 1x1 convs
                    assert weight.shape[0] == 1 and weight.shape[1] == 1
                else:
                    # Regular conv layers use 3x3
                    assert weight.shape[0] == 3 and weight.shape[1] == 3

    def test_weight_conversion_integration(self, centroid_model_path):
        """Test converting actual model weights to PyTorch format."""
        h5_path = centroid_model_path / "best_model.h5"
        weights = load_keras_weights(str(h5_path))

        # Convert a specific conv layer
        conv_path = "model_weights/stack0_enc0_conv0/stack0_enc0_conv0/kernel:0"
        if conv_path in weights:
            keras_weight = weights[conv_path]
            pytorch_weight = convert_keras_to_pytorch_conv2d(keras_weight)

            # Original shape: (3, 3, 1, 16) -> PyTorch: (16, 1, 3, 3)
            assert keras_weight.shape == (3, 3, 1, 16)
            assert pytorch_weight.shape == (16, 1, 3, 3)

            # Create PyTorch layer and verify it works
            conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            conv.weight.data = pytorch_weight

            # Test forward pass
            x = torch.randn(1, 1, 32, 32)
            output = conv(x)
            assert output.shape == (1, 16, 32, 32)
