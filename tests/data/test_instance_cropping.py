import math

import sleap_io as sio
import torch

from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import (
    compute_augmentation_padding,
    find_instance_crop_size,
    find_max_instance_bbox_size,
    generate_crops,
    make_centered_bboxes,
)
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.providers import process_lf


def test_find_instance_crop_size(minimal_instance):
    """Test `find_instance_crop_size` function."""
    labels = sio.load_slp(minimal_instance)
    crop_size = find_instance_crop_size(labels)
    assert crop_size == 74

    crop_size = find_instance_crop_size(labels, min_crop_size=100)
    assert crop_size == 100

    crop_size = find_instance_crop_size(labels, padding=10)
    assert crop_size == 84


def test_make_centered_bboxes():
    # Test bounding box calculation.
    gt = torch.Tensor(
        [
            [72.9970474243164, 131.07481384277344],
            [171.99703979492188, 131.07481384277344],
            [171.99703979492188, 230.07481384277344],
            [72.9970474243164, 230.07481384277344],
        ]
    )

    centroid = torch.Tensor([122.49704742431640625000, 180.57481384277343750000])
    bbox = make_centered_bboxes(centroid, 100, 100)
    assert torch.equal(gt, bbox)


def test_generate_crops(minimal_instance):
    """Test `generate_crops` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"])

    centroids = generate_centroids(ex["instances"], 0)
    cropped_ex = generate_crops(
        ex["image"], ex["instances"][0, 0], centroids[0, 0], crop_size=(100, 100)
    )

    assert cropped_ex["instance"].shape == (1, 2, 2)
    assert cropped_ex["centroid"].shape == (1, 2)
    assert cropped_ex["instance_image"].shape == (1, 1, 100, 100)
    assert cropped_ex["instance_bbox"].shape == (1, 4, 2)


def test_compute_augmentation_padding_no_augmentation():
    """Test that no padding is added when no augmentation is applied."""
    # No rotation, no scaling
    padding = compute_augmentation_padding(bbox_size=100, rotation_max=0, scale_max=1.0)
    assert padding == 0

    # No rotation, scale < 1 (shrinking)
    padding = compute_augmentation_padding(bbox_size=100, rotation_max=0, scale_max=0.9)
    assert padding == 0


def test_compute_augmentation_padding_scale_only():
    """Test padding with scale augmentation only."""
    # 10% scale increase on 100px bbox = 10px expansion (ceil gives 11 due to float)
    padding = compute_augmentation_padding(bbox_size=100, rotation_max=0, scale_max=1.1)
    assert padding == 11

    # 20% scale increase on 100px bbox = 20px expansion
    padding = compute_augmentation_padding(bbox_size=100, rotation_max=0, scale_max=1.2)
    assert padding == 20


def test_compute_augmentation_padding_rotation_only():
    """Test padding with rotation augmentation only."""
    # 45 degree rotation is worst case: bbox expands by sqrt(2)
    # For 100px bbox: 100 * sqrt(2) - 100 = 41.4
    padding = compute_augmentation_padding(
        bbox_size=100, rotation_max=45, scale_max=1.0
    )
    assert padding == math.ceil(100 * (math.sqrt(2) - 1))  # 42

    # 90 degree rotation: same as 45 for worst case
    padding = compute_augmentation_padding(
        bbox_size=100, rotation_max=90, scale_max=1.0
    )
    assert padding == math.ceil(100 * (math.sqrt(2) - 1))  # 42

    # 180 degree rotation: same as 45 for worst case
    padding = compute_augmentation_padding(
        bbox_size=100, rotation_max=180, scale_max=1.0
    )
    assert padding == math.ceil(100 * (math.sqrt(2) - 1))  # 42

    # Small rotation (15 degrees)
    # cos(15) + sin(15) = 0.966 + 0.259 = 1.225
    padding = compute_augmentation_padding(
        bbox_size=100, rotation_max=15, scale_max=1.0
    )
    rotation_factor = abs(math.cos(math.radians(15))) + abs(math.sin(math.radians(15)))
    expected = math.ceil(100 * (rotation_factor - 1))
    assert padding == expected


def test_compute_augmentation_padding_combined():
    """Test padding with both rotation and scale augmentation."""
    # 45 degree rotation + 1.1x scale
    # Expansion factor = sqrt(2) * 1.1 = 1.556
    # Padding = 100 * (1.556 - 1) = 55.6 -> 56
    padding = compute_augmentation_padding(
        bbox_size=100, rotation_max=45, scale_max=1.1
    )
    expected = math.ceil(100 * (math.sqrt(2) * 1.1 - 1))
    assert padding == expected

    # 180 degree rotation + 1.1x scale on larger bbox
    padding = compute_augmentation_padding(
        bbox_size=228.2, rotation_max=180, scale_max=1.1
    )
    expected = math.ceil(228.2 * (math.sqrt(2) * 1.1 - 1))
    assert padding == expected  # Should be ~127


def test_find_max_instance_bbox_size(minimal_instance):
    """Test `find_max_instance_bbox_size` function."""
    labels = sio.load_slp(minimal_instance)
    max_bbox = find_max_instance_bbox_size(labels)

    # The minimal_instance has a known bbox size
    # Based on test_find_instance_crop_size, crop_size=74 with stride=2
    # So max_length should be around 73-74
    assert max_bbox > 0
    assert max_bbox <= 74  # Should be less than or equal to crop size with stride=2
