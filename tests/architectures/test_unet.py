import torch
from torch import nn

from sleap_nn.architectures.encoder_decoder import Encoder
from sleap_nn.architectures.unet import UNet
from sleap_nn.architectures.utils import get_children_layers
from omegaconf import OmegaConf


def test_unet_reference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_channels = 1
    filters = 16
    filters_rate = 2
    kernel_size = 3
    down_blocks = 4
    convs_per_block = 2

    config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 16,
            "filters_rate": 2,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "block_contraction": False,
            "output_stride": 1,
        }
    )

    unet = UNet.from_config(config=config)

    in_channels = unet.final_dec_channels
    model = nn.Sequential(
        *[
            unet,
            nn.Conv2d(
                in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
            ),
        ]
    )
    # Test number of layers.
    flattened_layers = get_children_layers(model)
    assert len(flattened_layers) == 45

    # Test number of trainable weights.
    trainable_weights_count = sum(
        [1 if p.requires_grad else 0 for p in model.parameters()]
    )
    assert trainable_weights_count == 38

    # Test trainable parameter count.
    pytorch_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert pytorch_trainable_params == 1962541

    # Test total parameter count.
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    assert pytorch_total_params == 1962541

    # Test final output shape.
    unet = unet.to(device)
    unet.eval()

    x = torch.rand(1, 1, 192, 192).to(device)
    with torch.no_grad():
        y = unet(x)
    assert type(y) is dict
    assert "outputs" in y
    assert "strides" in y
    assert y["outputs"][-1].shape == (1, 16, 192, 192)
    assert type(y["strides"]) is list
    assert len(y["strides"]) == 4

    conv2d = nn.Conv2d(
        in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
    ).to(device)

    conv2d.eval()
    with torch.no_grad():
        z = conv2d(y["outputs"][-1])
    assert z.shape == (1, 13, 192, 192)

    # Test number of intermediate features outputted from encoder.
    enc = Encoder(
        in_channels=1,
        filters=filters,
        down_blocks=down_blocks,
        filters_rate=filters_rate,
        current_stride=2,
        convs_per_block=convs_per_block,
        kernel_size=kernel_size,
    )

    enc = enc.to(device)
    enc.eval()

    x = torch.rand(1, 1, 192, 192).to(device)
    with torch.no_grad():
        y, features = enc(x)

    assert y.shape == (1, 128, 12, 12)
    assert len(features) == 4
    assert features[0].shape == (1, 128, 24, 24)
    assert features[1].shape == (1, 64, 48, 48)
    assert features[2].shape == (1, 32, 96, 96)
    assert features[3].shape == (1, 16, 192, 192)

    # with stem stride
    config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 16,
            "filters_rate": 2,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": 2,
            "middle_block": True,
            "up_interpolate": True,
            "block_contraction": False,
            "output_stride": 1,
        }
    )

    unet = UNet.from_config(config=config)

    in_channels = unet.final_dec_channels
    model = nn.Sequential(
        *[
            unet,
            nn.Conv2d(
                in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
            ),
        ]
    )

    # Test final output shape.
    unet = unet.to(device)
    unet.eval()

    x = torch.rand(1, 1, 192, 192).to(device)
    with torch.no_grad():
        y = unet(x)
    assert type(y) is dict
    assert "outputs" in y
    assert "strides" in y
    assert y["outputs"][-1].shape == (1, 16, 192, 192)
    assert type(y["strides"]) is list

    conv2d = nn.Conv2d(
        in_channels=in_channels, out_channels=13, kernel_size=1, padding="same"
    ).to(device)

    conv2d.eval()
    with torch.no_grad():
        z = conv2d(y["outputs"][-1])
    assert z.shape == (1, 13, 192, 192)

    # block contraction.
    enc = Encoder(
        in_channels=1,
        filters=filters,
        down_blocks=down_blocks,
        filters_rate=filters_rate,
        current_stride=2,
        convs_per_block=convs_per_block,
        kernel_size=kernel_size,
    )

    enc = enc.to(device)
    enc.eval()

    x = torch.rand(1, 1, 192, 192).to(device)
    with torch.no_grad():
        y, features = enc(x)

    assert y.shape == (1, 128, 12, 12)
    assert len(features) == 4
    assert features[0].shape == (1, 128, 24, 24)
    assert features[1].shape == (1, 64, 48, 48)
    assert features[2].shape == (1, 32, 96, 96)
    assert features[3].shape == (1, 16, 192, 192)
