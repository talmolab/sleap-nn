import torch

from sleap_nn.data.utils import ensure_list, make_grid_vectors, expand_to_rank, gaussian_pdf


def test_ensure_list():
    assert ensure_list([0, 1, 2]) == [0, 1, 2]
    assert ensure_list(0) == [0]
    assert ensure_list([0]) == [0]

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

def test_gaussian_pdf():
    assert gaussian_pdf(torch.tensor([0]), sigma=1) == 1.0
    assert gaussian_pdf(torch.tensor([1]), sigma=1) == 0.6065306597126334
    assert gaussian_pdf(torch.tensor([1]), sigma=2) == 0.8824969025845955