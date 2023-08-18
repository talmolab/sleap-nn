import torch
from torch import nn

from sleap_nn.architectures.common import get_children_layers
from sleap_nn.architectures.encoder_decoder import Encoder
from sleap_nn.architectures.unet import UNet


def test_unet_reference():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        in_channels = 1
        filters = 64
        filters_rate = 2
        kernel_size = 3
        down_blocks = 4
        stem_blocks = 0
        up_blocks = 4
        convs_per_block = 2
        middle_block = True
        block_contraction = False

        unet = UNet(
            in_channels=in_channels,
            filters=filters, 
            filters_rate=filters_rate, 
            down_blocks=down_blocks, 
            stem_blocks=stem_blocks, 
            up_blocks=up_blocks
        )

        in_channels = int(
            filters
            * (
                filters_rate
                ** (down_blocks + stem_blocks - 1 - up_blocks + 1)
            )
        )
        model = nn.Sequential(*[
            unet,
            nn.Conv2d(in_channels=in_channels, out_channels=13, kernel_size=1, padding="same")
        ])

        # Test number of layers.
        flattened_layers = get_children_layers(model)
        assert len(flattened_layers) == 45

        # Test number of trainable weights.
        trainable_weights_count = sum([1 if p.requires_grad else 0 for p in model.parameters()])
        assert trainable_weights_count == 38
        
        # Test trainable parameter count.
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert pytorch_trainable_params == 31378573

        # Test total parameter count.
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        assert pytorch_total_params == 31378573

        # Test final output shape.
        model = model.to(device)
        _ = model.eval()

        x = torch.rand(1, 1, 192, 192).to(device)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 13, 192, 192)

        # Test number of intermediate features outputted from encoder.
        enc = Encoder(
        in_channels = 1,
        filters = filters,
        down_blocks = down_blocks,
        filters_rate = filters_rate,
        current_stride = 2,
        stem_blocks = stem_blocks,
        convs_per_block = convs_per_block,
        kernel_size = kernel_size,
        middle_block = middle_block,
        block_contraction = block_contraction,
        )

        enc = enc.to(device)
        _ = enc.eval()

        x = torch.rand(1, 1, 192, 192).to(device)
        with torch.no_grad():
            y, features = enc(x)

        assert y.shape == (1, 1024, 12, 12)
        assert len(features) == 4
        assert features[0].shape == (1, 512, 24, 24)
        assert features[1].shape == (1, 256, 48, 48)
        assert features[2].shape == (1, 128, 96, 96)
        assert features[3].shape == (1, 64, 192, 192)