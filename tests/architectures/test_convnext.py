import torch
from torch import nn

from sleap_nn.architectures.encoder_decoder import Encoder
from sleap_nn.architectures.convnext import ConvNextWrapper, ConvNeXtEncoder
from sleap_nn.architectures.utils import get_children_layers


def test_convnext_reference():

    arch = {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]}
    in_channels = 1
    kernel_size = 3
    stem_patch_kernel = 4
    stem_patch_stride = 2
    filters_rate = 2
    up_blocks = 3
    convs_per_block = 2

    convnext = ConvNextWrapper(
        arch=arch,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stem_patch_kernel=stem_patch_kernel,
        stem_patch_stride=stem_patch_stride,
        filters_rate=filters_rate,
        up_blocks=up_blocks,
        convs_per_block=convs_per_block,
    )

    down_blocks = len(arch["depths"]) - 1
    in_channels = int(
        arch["channels"][0] * (filters_rate ** (down_blocks - 1 - up_blocks + 1))
    )

    # Test final output shape.
    convnext.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = convnext(x)
    assert type(y) is dict
    assert "outputs" in y
    assert "strides" in y
    assert y["outputs"][-1].shape == (1, 96, 192, 192)
    assert type(y["strides"]) is list
    assert len(y["strides"]) == 4

    conv2d = nn.Conv2d(
        in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
    )

    conv2d.eval()
    with torch.no_grad():
        z = conv2d(y["outputs"][-1])
    assert z.shape == (1, 13, 192, 192)

    # Test number of intermediate features outputted from encoder.
    enc = ConvNeXtEncoder(
        blocks={"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels=1,
        stem_kernel=4,
        stem_stride=2,
    )

    enc = enc
    enc.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        out = enc(x)
        x, features = out[-1], out[::2]
        features = features[:-1][::-1]

    assert x.shape == (1, 768, 12, 12)
    assert len(features) == 4
    assert features[0].shape == (1, 384, 24, 24)
    assert features[1].shape == (1, 192, 48, 48)
    assert features[2].shape == (1, 96, 96, 96)

    # stride=4
    enc = ConvNeXtEncoder(
        blocks={"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels=1,
        stem_kernel=4,
        stem_stride=4,
    )

    enc = enc
    enc.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        out = enc(x)
        x, features = out[-1], out[::2]
        features = features[:-1][::-1]

    assert x.shape == (1, 768, 6, 6)
    assert len(features) == 4
    assert features[0].shape == (1, 384, 12, 12)
    assert features[1].shape == (1, 192, 24, 24)
    assert features[2].shape == (1, 96, 48, 48)
    assert features[3].shape == (1, 1, 192, 192)
