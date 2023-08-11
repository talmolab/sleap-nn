import torch

from sleap_nn.data.utils import make_grid_vectors


def test_make_grid_vectors():
    xv, yv = make_grid_vectors(image_height=100, image_width=80, output_stride=2)
    assert xv.shape == torch.Size([40])
    assert yv.shape == torch.Size([50])

    xv, yv = make_grid_vectors(image_height=40, image_width=20, output_stride=1)
    assert xv.shape == torch.Size([20])
    assert yv.shape == torch.Size([40])
