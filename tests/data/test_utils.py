import torch

from sleap_nn.data.utils import make_grid_vectors, expand_to_rank


def test_make_grid_vectors():
    xv, yv = make_grid_vectors(image_height=100, image_width=80, output_stride=2)
    assert xv.shape == torch.Size([40])
    assert yv.shape == torch.Size([50])

    xv, yv = make_grid_vectors(image_height=40, image_width=20, output_stride=1)
    assert xv.shape == torch.Size([20])
    assert yv.shape == torch.Size([40])

def test_expand_to_rank():
    out = expand_to_rank(torch.arange(3), target_rank=2, prepend=True)
    assert out.numpy().tolist() == [[0, 1, 2]]

    out = expand_to_rank(torch.arange(3), target_rank=3, prepend=True)
    assert out.numpy().tolist() == [[[0, 1, 2]]]

    out = expand_to_rank(torch.arange(3), target_rank=2, prepend=False)
    assert out.numpy().tolist() == [[0], [1], [2]]

    out = expand_to_rank(torch.reshape(torch.arange(3), (1, 3)), target_rank=2, prepend=True)
    assert out.numpy().tolist() == [[0, 1, 2]]

    gt = torch.reshape(torch.arange(2 * 3 * 4), (2, 3, 4))
    out = expand_to_rank(torch.arange(2* 3 * 4).reshape(2, 3, 4), target_rank=2, prepend=True)
    assert gt.numpy().tolist() == out.numpy().tolist()
