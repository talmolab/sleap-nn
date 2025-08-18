import torch
import pytest
import sleap_io as sio
import psutil
import itertools

from types import SimpleNamespace

from sleap_nn.data.utils import (
    ensure_list,
    make_grid_vectors,
    expand_to_rank,
    gaussian_pdf,
    check_memory,
    check_cache_memory,
)


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

    out = expand_to_rank(
        torch.reshape(torch.arange(3), (1, 3)), target_rank=2, prepend=True
    )
    assert out.numpy().tolist() == [[0, 1, 2]]

    gt = torch.reshape(torch.arange(2 * 3 * 4), (2, 3, 4))
    out = expand_to_rank(
        torch.arange(2 * 3 * 4).reshape(2, 3, 4), target_rank=2, prepend=True
    )
    assert gt.numpy().tolist() == out.numpy().tolist()


def test_gaussian_pdf():
    assert gaussian_pdf(torch.tensor([0]), sigma=1) == 1.0
    assert gaussian_pdf(torch.tensor([1]), sigma=1) == 0.6065306597126334
    assert gaussian_pdf(torch.tensor([1]), sigma=2) == 0.8824969025845955


@pytest.mark.parametrize(
    "labels_path_fixture",
    [("minimal_instance"), ("small_robot_minimal")],
)
def test_check_memory_ok(labels_path_fixture, request):
    labels_path = request.getfixturevalue(labels_path_fixture)
    labels = sio.load_slp(labels_path)

    # Compute expected bytes from the real labels
    expected = sum(lf.image.nbytes for lf in labels if lf.image is not None)
    # Re-iterate because sleap_io.Labels is iterable; make a fresh iterator
    labels = sio.load_slp(labels_path)

    assert check_memory(labels) == expected


@pytest.mark.parametrize(
    "labels_path_fixture",
    [("minimal_instance"), ("small_robot_minimal")],
)
def test_check_memory_raises_when_one_label_has_no_image(labels_path_fixture, request):
    labels_path = request.getfixturevalue(labels_path_fixture)
    labels = sio.load_slp(labels_path)

    # Build an iterable that yields: one fake with image=None, then all real frames.
    missing = SimpleNamespace(image=None)
    adapted = itertools.chain(
        [missing], (SimpleNamespace(image=lf.image) for lf in labels)
    )

    with pytest.raises(ValueError, match="no image data"):
        check_memory(adapted)


def test_check_memory_no_labels():
    """Test memory check when no labels are present."""
    labels = sio.Labels()
    memory_required = check_memory(labels)
    assert memory_required == 0
    # check_cache_memory should return True for empty labels
    assert check_cache_memory([labels], [labels]) == True


@pytest.mark.parametrize(
    "labels_path_fixture",
    [("minimal_instance"), ("small_robot_minimal")],
)
def test_check_cache_memory(labels_path_fixture, request):
    """Test memory check for caching image samples."""
    labels_path = request.getfixturevalue(labels_path_fixture)
    labels = sio.load_slp(labels_path)
    assert isinstance(labels, sio.Labels)
    available_memory = psutil.virtual_memory().available
    memory_required = check_memory(labels)
    assert isinstance(memory_required, int)
    assert memory_required > 0
    if memory_required < available_memory:
        assert check_cache_memory([labels], [labels]) == True
    else:
        assert check_cache_memory([labels], [labels]) == False
