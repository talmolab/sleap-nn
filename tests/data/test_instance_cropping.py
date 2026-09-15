import math

import numpy as np
import pytest
import sleap_io as sio
import torch

from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import (
    compute_augmentation_padding,
    count_clipped_instances,
    find_instance_crop_size,
    find_max_instance_bbox_size,
    generate_crops,
    iter_required_crop_sizes,
    make_centered_bboxes,
)
from sleap_nn.data.resizing import apply_sizematcher, compute_eff_scale
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


def _labels_with_videos(specs, node_names=("head", "mid", "tail")):
    """Build labels whose videos have differing resolutions.

    Args:
        specs: One ``(video_hw, points)`` per labeled frame, where ``points`` is
            an ``(n_nodes, 2)`` nested sequence in that video's native pixels.
        node_names: Skeleton node names.

    Returns:
        A `sio.Labels` with one video and one labeled frame per spec.
    """
    skel = sio.Skeleton(list(node_names))
    videos = []
    lfs = []
    for idx, (video_hw, points) in enumerate(specs):
        video = sio.Video(
            filename=f"v{idx}.mp4",
            backend_metadata={"shape": (1, video_hw[0], video_hw[1], 1)},
            open_backend=False,
        )
        videos.append(video)
        lfs.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[sio.Instance.from_numpy(np.array(points), skeleton=skel)],
            )
        )
    return sio.Labels(videos=videos, skeletons=[skel], labeled_frames=lfs)


def test_compute_eff_scale_matches_sizematcher():
    """`compute_eff_scale` must agree with the scale `apply_sizematcher` applies."""
    # Square target from a square source: no scaling.
    assert compute_eff_scale((384, 384), (384, 384)) == 1.0
    # No target at all, or a missing axis, leaves that axis native.
    assert compute_eff_scale((384, 512), None) == 1.0
    assert compute_eff_scale((384, 512), (None, None)) == 1.0

    # Non-square cases, where hratio != wratio, exercise the min() branch that
    # preserves aspect ratio. Assert against the value `apply_sizematcher`
    # actually returns rather than restating the formula.
    for img_hw, max_hw in [
        ((100, 200), (400, 400)),  # width-limited
        ((200, 100), (400, 400)),  # height-limited
        ((512, 384), (256, 256)),  # downscale
        ((128, 128), (256, 512)),  # upscale, height-limited
        ((90, 160), (180, 320)),  # exact 2x, both axes
    ]:
        image = torch.zeros(1, 1, img_hw[0], img_hw[1])
        _, eff_scale = apply_sizematcher(image, max_hw[0], max_hw[1])
        assert compute_eff_scale(img_hw, max_hw) == pytest.approx(
            eff_scale
        ), f"mismatch for img={img_hw} max={max_hw}"


def test_find_instance_crop_size_accounts_for_sizematcher():
    """A crop size must cover instances after the size matcher rescales them.

    The #2862 worked example: the larger video sets ``max_hw``, so the smaller
    video is upscaled and its instance grows past a natively-measured crop.
    """
    # A 200px-wide instance on a 2048px video, and a 150px-wide instance on a
    # 1024px video. Points are laid out symmetrically so the center_of_mass
    # centroid sits at the bbox midpoint and only scaling is under test.
    labels = _labels_with_videos(
        [
            ((2048, 2048), [[900, 1000], [1000, 1000], [1100, 1000]]),
            ((1024, 1024), [[425, 500], [500, 500], [575, 500]]),
        ]
    )

    # Measured natively, the 200px instance wins and the 150px one looks safe.
    assert find_instance_crop_size(labels, maximum_stride=2) == 200

    # But size matching to the larger video upscales the 1024px video 2x, so its
    # instance actually spans 300px and needs the bigger crop.
    scaled = find_instance_crop_size(labels, maximum_stride=2, max_hw=(2048, 2048))
    assert scaled == 300

    # `find_max_instance_bbox_size` feeds the augmentation padding and must
    # measure in the same space, or the margin is under-sized for the same reason.
    assert find_max_instance_bbox_size(labels) == pytest.approx(200.0)
    assert find_max_instance_bbox_size(labels, max_hw=(2048, 2048)) == pytest.approx(
        300.0
    )


def test_find_instance_crop_size_is_anchor_aware():
    """Crops are centered on the centroid, so an off-center anchor needs more room."""
    # A 100px-wide instance: nodes at x=0, 50, 100 on a single 512px video.
    labels = _labels_with_videos([((512, 512), [[0, 100], [50, 100], [100, 100]])])

    # center_of_mass sits at the middle, so the crop only needs the bbox extent.
    assert find_instance_crop_size(labels, maximum_stride=2) == 100

    # Anchoring on the middle node is equivalent.
    assert find_instance_crop_size(labels, maximum_stride=2, anchor_ind=1) == 100

    # Anchoring on either end puts the far node a full extent away from the crop
    # center, so the crop must be twice as wide to reach it. This is the
    # long-tail case from #2862.
    assert find_instance_crop_size(labels, maximum_stride=2, anchor_ind=0) == 200
    assert find_instance_crop_size(labels, maximum_stride=2, anchor_ind=2) == 200


def test_iter_required_crop_sizes_skips_undefined_centroids():
    """An instance with no visible points contributes no requirement."""
    labels = _labels_with_videos(
        [
            ((512, 512), [[0, 100], [50, 100], [100, 100]]),
            ((512, 512), [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]),
        ]
    )
    required = list(iter_required_crop_sizes(labels))
    assert required == [pytest.approx(100.0)]


def test_find_instance_crop_size_min_crop_size_is_a_floor():
    """`min_crop_size` must not short-circuit the measurement (it did).

    A `min_crop_size` divisible by `maximum_stride` used to return immediately,
    ignoring the labels entirely -- so a project of large animals could be given
    a tiny crop. It is documented as a floor, so the larger value must win.
    """
    # A 400px instance, with min_crop_size=96 divisible by maximum_stride=16.
    labels = _labels_with_videos([((1024, 1024), [[100, 500], [300, 500], [500, 500]])])

    crop_size = find_instance_crop_size(labels, maximum_stride=16, min_crop_size=96)
    assert crop_size == 400

    # The floor still applies when it is the larger of the two.
    small = _labels_with_videos([((1024, 1024), [[500, 500], [510, 500], [520, 500]])])
    assert find_instance_crop_size(small, maximum_stride=16, min_crop_size=96) == 96


def test_crop_size_helpers_honor_user_instances_only():
    """Predicted instances must not size a crop they are excluded from training."""
    skel = sio.Skeleton(["head", "mid", "tail"])
    video = sio.Video(
        filename="v.mp4",
        backend_metadata={"shape": (1, 1024, 1024, 1)},
        open_backend=False,
    )
    labels = sio.Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[
            sio.LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    sio.Instance.from_numpy(
                        np.array([[450, 500], [500, 500], [550, 500]]), skeleton=skel
                    ),
                    # A much wider PREDICTED instance.
                    sio.PredictedInstance.from_numpy(
                        np.array([[300, 500], [500, 500], [700, 500]]),
                        skeleton=skel,
                        point_scores=np.ones(3),
                        score=1.0,
                    ),
                ],
            )
        ],
    )

    # Including predictions, the 400px predicted instance sets the size.
    assert find_instance_crop_size(labels, maximum_stride=2) == 400
    # Excluding them, only the 100px user instance counts.
    assert (
        find_instance_crop_size(labels, maximum_stride=2, user_instances_only=True)
        == 100
    )
    assert find_max_instance_bbox_size(
        labels, user_instances_only=True
    ) == pytest.approx(100.0)


def test_count_clipped_instances():
    """Counting clipped instances reports the size that would contain them all."""
    labels = _labels_with_videos(
        [
            ((512, 512), [[0, 100], [50, 100], [100, 100]]),  # needs 100
            ((512, 512), [[0, 100], [100, 100], [200, 100]]),  # needs 200
        ]
    )

    # Big enough for both.
    assert count_clipped_instances(labels, crop_size=200) == (
        0,
        2,
        pytest.approx(200.0),
    )
    # Clips only the wider one.
    n_clipped, n_total, max_required = count_clipped_instances(labels, crop_size=150)
    assert (n_clipped, n_total) == (1, 2)
    assert max_required == pytest.approx(200.0)
    # Anchored at an end, both need twice the room and both clip.
    n_clipped, _, max_required = count_clipped_instances(
        labels, crop_size=150, anchor_ind=0
    )
    assert n_clipped == 2
    assert max_required == pytest.approx(400.0)
