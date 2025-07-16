import torch
from torch import nn
import math
from sleap_nn.architectures.swint import SwinTransformerEncoder, SwinTWrapper
from omegaconf import OmegaConf


def test_swint_reference():
    """Test SwinTWrapper and SwinTransformerEncoder module."""
    depths = [2, 2, 6, 2]
    embed_dim = 96

    config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "patch_size": 4,
            "window_size": 7,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_stride": 2,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    swint = SwinTWrapper.from_config(config)

    in_channels = swint.final_dec_channels
    model = nn.Sequential(
        *[
            swint,
            nn.Conv2d(
                in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
            ),
        ]
    )

    # Test final output shape.
    swint.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = swint(x)
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

    # Test number of intermediate features outputted from encoder.
    enc = SwinTransformerEncoder(
        in_channels=1,
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stem_stride=2,
        stochastic_depth_prob=0.1,
        norm_layer="",
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

    # stride = 4

    enc = SwinTransformerEncoder(
        in_channels=1,
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stem_stride=4,
        stochastic_depth_prob=0.1,
        norm_layer="",
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
            "patch_size": 4,
            "window_size": 7,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_stride": 4,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    swint = SwinTWrapper.from_config(config)

    swint.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = swint(x)
    out = y["outputs"]
    assert out[0].shape == (1, 768, 6, 6)
    assert out[1].shape == (1, 384, 12, 12)
    assert out[2].shape == (1, 192, 24, 24)
    assert out[3].shape == (1, 96, 48, 48)
    assert out[4].shape == (1, 96, 96, 96)

    # without providing arch and model type
    config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": None,
            "arch": None,
            "patch_size": 4,
            "window_size": 7,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_stride": 2,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    swint = SwinTWrapper.from_config(config)

    in_channels = swint.final_dec_channels

    # Test final output shape.
    swint.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = swint(x)
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
            "arch": {"embed": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
            "patch_size": 4,
            "window_size": 7,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_stride": 2,
            "output_stride": 1,
            "max_stride": 32,
        }
    )

    swint = SwinTWrapper.from_config(config)

    in_channels = swint.final_dec_channels

    # Test final output shape.
    swint.eval()

    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        y = swint(x)
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
