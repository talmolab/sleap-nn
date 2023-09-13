import pytest
import torch

from sleap_nn.architectures.common import MaxPool2dWithSamePadding


def test_maxpool2d_with_same_padding():
    pooling = MaxPool2dWithSamePadding(
        kernel_size=3, stride=2, dilation=2, padding="same"
    )

    x = torch.rand(1, 10, 100, 100)
    z = pooling(x)
    assert z.shape == (1, 10, 50, 50)
