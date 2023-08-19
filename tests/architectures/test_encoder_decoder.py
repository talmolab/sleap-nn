import torch

import sleap_nn
from sleap_nn.architectures.encoder_decoder import SimpleConvBlock


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
