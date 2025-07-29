import torch
from torch import nn

from sleap_nn.architectures.encoder_decoder import Encoder
from sleap_nn.architectures.convnext import ConvNextWrapper, ConvNeXtEncoder
from sleap_nn.architectures.utils import get_children_layers
from omegaconf import OmegaConf


def test_convnext_reference():
    """Test ConvNextEncoder and ConvNextWrapper."""
    config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    convnext = ConvNextWrapper.from_config(config)

    in_channels = convnext.final_dec_channels
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
    assert len(features) == 3
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
    assert len(features) == 3
    assert features[0].shape == (1, 384, 12, 12)
    assert features[1].shape == (1, 192, 24, 24)
    assert features[2].shape == (1, 96, 48, 48)

    config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 4,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    convnext = ConvNextWrapper.from_config(config)

    convnext.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = convnext(x)
    out = y["outputs"]
    assert out[0].shape == (1, 768, 6, 6)
    assert out[1].shape == (1, 384, 12, 12)
    assert out[2].shape == (1, 192, 24, 24)
    assert out[3].shape == (1, 96, 48, 48)
    assert out[4].shape == (1, 96, 96, 96)

    # arch as None. select `tiny` architecture by default

    config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": None,
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    convnext = ConvNextWrapper.from_config(config)

    in_channels = convnext.final_dec_channels
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
    assert len(y["strides"]) == 5

    conv2d = nn.Conv2d(
        in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
    )

    conv2d.eval()
    with torch.no_grad():
        z = conv2d(y["outputs"][-1])
    assert z.shape == (1, 13, 192, 192)

    # custom architecture

    config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": None,
            "arch": {"depths": [3, 3, 9, 3], "channels": [32, 64, 128, 256]},
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    convnext = ConvNextWrapper.from_config(config)

    in_channels = convnext.final_dec_channels
    # Test final output shape.
    convnext.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = convnext(x)
    assert type(y) is dict
    assert "outputs" in y
    assert "strides" in y
    assert y["outputs"][-1].shape == (1, 32, 192, 192)
    assert type(y["strides"]) is list
    assert len(y["strides"]) == 5

    conv2d = nn.Conv2d(
        in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
    )

    conv2d.eval()
    with torch.no_grad():
        z = conv2d(y["outputs"][-1])
    assert z.shape == (1, 13, 192, 192)
