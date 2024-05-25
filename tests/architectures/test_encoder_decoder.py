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

    block = SimpleUpsamplingBlock(
        x_in_shape=10,
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

    x = torch.rand(5, 5, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 64, 200, 200)

    block = SimpleUpsamplingBlock(
        x_in_shape=10,
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

    x = torch.rand(5, 5, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 64, 200, 200)

    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=64,
        refine_convs_kernel_size=3,
        refine_convs_use_bias=True,
        refine_convs_batch_norm=True,
        refine_convs_batch_norm_before_activation=False,
        refine_convs_activation="relu",
    )

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 5, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 64, 200, 200)

    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=64,
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
    print(block)

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 5, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 64, 200, 200)

    block = SimpleUpsamplingBlock(
        x_in_shape=10,
        current_stride=1,
        upsampling_stride=2,
        interp_method="bilinear",
        refine_convs=2,
        refine_convs_filters=64,
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
    print(block)

    block = block.to(device)
    block.eval()

    x = torch.rand(5, 5, 100, 100).to(device)
    feature = torch.rand(5, 5, 200, 200).to(device)

    z = block(x, feature=feature)

    assert z.shape == (5, 64, 200, 200)
