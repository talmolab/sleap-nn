import torch
from torch import nn

from sleap_nn.architectures.encoder_decoder import Encoder
from sleap_nn.architectures.swint import SwinTransformerEncoder, SwinTWrapper
from sleap_nn.architectures.utils import get_children_layers


def test_swint_reference():

    depths = [2, 2, 6, 2]
    up_blocks = 3
    embed_dim = 96
    filters_rate = 2

    swint = SwinTWrapper(
        in_channels=1,
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stem_stride=2,
        stochastic_depth_prob=0.1,
        norm_layer="",
        kernel_size=3,
        filters_rate=filters_rate,
        up_blocks=up_blocks,
        convs_per_block=2,
    )

    down_blocks = len(depths) - 1
    in_channels = int(embed_dim * (filters_rate ** (down_blocks - 1 - up_blocks + 1)))
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
    assert len(y["strides"]) == 4

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
    assert len(features) == 4
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
    assert len(features) == 4
    assert features[0].shape == (1, 384, 12, 12)
    assert features[1].shape == (1, 192, 24, 24)
    assert features[2].shape == (1, 96, 48, 48)
    assert features[3].shape == (1, 1, 192, 192)
