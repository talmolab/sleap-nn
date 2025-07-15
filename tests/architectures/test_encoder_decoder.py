import torch

import sleap_nn
from sleap_nn.architectures.encoder_decoder import (
    SimpleConvBlock,
    SimpleUpsamplingBlock,
)


def test_simple_conv_block():
    block = SimpleConvBlock(
        in_channels=1,
        pool=True,
        pooling_stride=2,
        pool_before_convs=False,
        num_convs=2,
        filters=32,
        kernel_size=3,
        use_bias=True,
        batch_norm=True,
        activation="relu",
    )

    block.blocks[0].__class__ == torch.nn.modules.conv.Conv2d
    block.blocks[1].__class__ == torch.nn.modules.batchnorm.BatchNorm2d
    block.blocks[2].__class__ == torch.nn.modules.activation.ReLU
    block.blocks[3].__class__ == torch.nn.modules.conv.Conv2d
    block.blocks[4].__class__ == torch.nn.modules.batchnorm.BatchNorm2d
    block.blocks[5].__class__ == torch.nn.modules.activation.ReLU
    block.blocks[6].__class__ == sleap_nn.architectures.common.MaxPool2dWithSamePadding


def test_simple_upsampling_block():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with interpolation (default behavior)
    block = SimpleUpsamplingBlock(
        x_in_shape=5,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=64,
        refine_convs_kernel_size=3,
        refine_convs_use_bias=True,
        refine_convs_batch_norm=True,
        refine_convs_batch_norm_before_activation=True,
        refine_convs_activation="relu",
    )

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 64, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 64, 200, 200)

    # Test with transpose convolution and batch norm before activation
    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=5,
        refine_convs_kernel_size=3,
        refine_convs_use_bias=True,
        refine_convs_batch_norm=True,
        refine_convs_batch_norm_before_activation=True,
        refine_convs_activation="relu",
        up_interpolate=False,
        transpose_convs_filters=5,
        transpose_convs_batch_norm=True,
        transpose_convs_batch_norm_before_activation=True,
    )

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 10, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 5, 200, 200)

    # Test with transpose convolution and batch norm after activation
    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=5,
        refine_convs_kernel_size=3,
        refine_convs_use_bias=True,
        refine_convs_batch_norm=True,
        refine_convs_batch_norm_before_activation=False,
        refine_convs_activation="relu",
        up_interpolate=False,
        transpose_convs_filters=5,
        transpose_convs_batch_norm=True,
        transpose_convs_batch_norm_before_activation=False,
    )

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 10, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 5, 200, 200)

    # Test with transpose convolution and no batch norm
    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=5,
        refine_convs_kernel_size=3,
        refine_convs_use_bias=True,
        refine_convs_batch_norm=True,
        refine_convs_batch_norm_before_activation=False,
        refine_convs_activation="relu",
        up_interpolate=False,
        transpose_convs_filters=5,
        transpose_convs_batch_norm=False,
        transpose_convs_batch_norm_before_activation=False,
    )

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 10, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 5, 200, 200)

    # Test with transpose convolution and batch norm before activation
    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=5,
        refine_convs_kernel_size=3,
        refine_convs_use_bias=True,
        refine_convs_batch_norm=True,
        refine_convs_batch_norm_before_activation=False,
        refine_convs_activation="relu",
        up_interpolate=False,
        transpose_convs_filters=5,
        transpose_convs_batch_norm=True,
        transpose_convs_batch_norm_before_activation=True,
    )

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 10, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 5, 200, 200)


def test_simple_upsampling_block_transpose_conv_channels():
    """Test that transpose convolution uses correct channel configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with specific channel configuration
    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        up_interpolate=False,
        refine_convs_filters=64,
        transpose_convs_filters=5,
        transpose_convs_batch_norm=True,
        transpose_convs_batch_norm_before_activation=True,
    )

    block = block.to(device)
    block.eval()

    # Check that transpose conv has correct in/out channels
    transpose_conv = None
    for name, module in block.blocks.named_modules():
        if isinstance(module, torch.nn.ConvTranspose2d):
            transpose_conv = module
            break

    assert transpose_conv is not None
    assert transpose_conv.in_channels == 10  # transpose_convs_filters
    assert transpose_conv.out_channels == 5  # refine_convs_filters

    # Check batch norm layers have correct features
    bn_before = None
    bn_after = None
    for name, module in block.blocks.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if "trans_conv_bn" in name:
                bn_before = module
            elif "trans_conv_bn_after" in name:
                bn_after = module

    # With batch_norm_before_activation=True, we should have bn_before but not bn_after
    assert bn_before is not None
    assert bn_after is None


def test_simple_upsampling_block_transpose_conv_channels_after_activation():
    """Test transpose convolution with batch norm after activation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        up_interpolate=False,
        refine_convs_filters=64,
        transpose_convs_filters=5,
        transpose_convs_batch_norm=True,
        transpose_convs_batch_norm_before_activation=False,
    )

    block = block.to(device)
    block.eval()

    # Check batch norm layers have correct features
    bn_before = None
    bn_after = None
    for name, module in block.blocks.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if "trans_conv_bn" in name:
                bn_before = module
            if "trans_conv_bn_after" in name:
                bn_after = module

    # With batch_norm_before_activation=False, we should have bn_after but not bn_before
    assert bn_after is not None
