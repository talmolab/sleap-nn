import torch
import pytest
import sleap_io as sio
import psutil

from sleap_nn.data.utils import (
    ensure_list,
    make_grid_vectors,
    expand_to_rank,
    gaussian_pdf,
    check_memory,
    check_cache_memory,
    estimate_cache_memory,
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
def test_check_memory_raises_when_shape_is_none(labels_path_fixture, request):
    labels_path = request.getfixturevalue(labels_path_fixture)
    labels = sio.load_slp(labels_path)

    # Patch the video's shape to None to simulate an inaccessible video.
    from unittest.mock import PropertyMock, patch

    video = labels[0].video
    with patch.object(
        type(video), "shape", new_callable=PropertyMock, return_value=None
    ):
        with pytest.raises(ValueError, match="Cannot determine video shape"):
            check_memory(labels)


def test_check_memory_multi_video(minimal_instance, small_robot_minimal):
    """Test per-video sampling logic with two real videos of different sizes."""
    labels_a = sio.load_slp(minimal_instance)
    labels_b = sio.load_slp(small_robot_minimal)

    # Compute expected from video shapes.
    shape_a = labels_a[0].video.shape
    shape_b = labels_b[0].video.shape
    bytes_per_frame_a = shape_a[1] * shape_a[2] * shape_a[3]
    bytes_per_frame_b = shape_b[1] * shape_b[2] * shape_b[3]
    expected = len(labels_a) * bytes_per_frame_a + len(labels_b) * bytes_per_frame_b

    # Combine labeled frames from both into a single list.
    combined = list(labels_a) + list(labels_b)
    assert check_memory(combined) == expected


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


@pytest.mark.parametrize(
    "labels_path_fixture",
    [("minimal_instance"), ("small_robot_minimal")],
)
def test_estimate_cache_memory(labels_path_fixture, request):
    """Test detailed memory estimation for caching image samples."""
    labels_path = request.getfixturevalue(labels_path_fixture)
    labels = sio.load_slp(labels_path)

    # Test basic estimation without workers
    estimate = estimate_cache_memory([labels], [labels], num_workers=0)

    assert isinstance(estimate, dict)
    assert "raw_cache_bytes" in estimate
    assert "python_overhead_bytes" in estimate
    assert "worker_overhead_bytes" in estimate
    assert "buffer_bytes" in estimate
    assert "total_bytes" in estimate
    assert "available_bytes" in estimate
    assert "sufficient" in estimate
    assert "num_samples" in estimate

    assert estimate["raw_cache_bytes"] > 0
    assert estimate["python_overhead_bytes"] > 0
    assert estimate["worker_overhead_bytes"] == 0  # No workers
    assert estimate["buffer_bytes"] > 0
    assert estimate["total_bytes"] > estimate["raw_cache_bytes"]
    assert estimate["num_samples"] == len(labels) * 2  # train + val


@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_estimate_cache_memory_with_workers(minimal_instance, num_workers):
    """Test memory estimation accounts for DataLoader workers."""
    labels = sio.load_slp(minimal_instance)

    estimate = estimate_cache_memory([labels], [labels], num_workers=num_workers)

    if num_workers == 0:
        assert estimate["worker_overhead_bytes"] == 0
    else:
        # Workers should add overhead
        assert estimate["worker_overhead_bytes"] > 0
        # More workers = more overhead
        estimate_fewer = estimate_cache_memory(
            [labels], [labels], num_workers=num_workers - 1
        )
        assert (
            estimate["worker_overhead_bytes"] >= estimate_fewer["worker_overhead_bytes"]
        )


@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_check_cache_memory_with_workers(minimal_instance, num_workers):
    """Test check_cache_memory accepts num_workers parameter."""
    labels = sio.load_slp(minimal_instance)

    # Should not raise - just verify it accepts the parameter
    result = check_cache_memory([labels], [labels], num_workers=num_workers)
    assert isinstance(result, bool)
