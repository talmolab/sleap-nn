import sleap_io as sio
import torchvision.transforms as T
from sleap_nn.data.get_data_chunks import (
    bottomup_data_chunks,
    centered_instance_data_chunks,
    centroid_data_chunks,
    single_instance_data_chunks,
)

from sleap_nn.data.providers import get_max_height_width


def test_bottomup_data_chunks(minimal_instance, config):
    """Test `bottomup_data_chunks` function."""
    labels = sio.load_slp(minimal_instance)
    transform = T.ToTensor()
    max_hw = get_max_height_width(labels)
    samples = []
    for idx, lf in enumerate(labels):
        samples.append(
            bottomup_data_chunks(
                (lf, idx),
                data_config=config.data_config,
                max_instances=4,
                max_hw=max_hw,
            )
        )

    assert len(samples) == 1

    gt_keys = [
        "image",
        "instances",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["image"]).shape == (1, 384, 384)
    assert samples[0]["instances"].shape == (1, 4, 2, 2)

    # test `is_rgb`
    config.data_config.preprocessing.is_rgb = True
    samples = []
    for idx, lf in enumerate(labels):
        samples.append(
            bottomup_data_chunks(
                (lf, idx),
                data_config=config.data_config,
                max_instances=2,
                max_hw=max_hw,
            )
        )

    gt_keys = [
        "image",
        "instances",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["image"]).shape == (3, 384, 384)
    assert samples[0]["instances"].shape == (1, 2, 2, 2)


def test_centered_instance_data_chunks(minimal_instance, config):
    """Test `centered_instance_data_chunks` function."""
    labels = sio.load_slp(minimal_instance)
    transform = T.ToTensor()
    max_hw = get_max_height_width(labels)
    samples = []
    for idx, lf in enumerate(labels):
        res = centered_instance_data_chunks(
            (lf, idx),
            data_config=config.data_config,
            anchor_ind=0,
            crop_size=(160, 160),
            max_instances=4,
            max_hw=max_hw,
        )
        samples.extend(res)

    assert len(samples) == 2

    gt_keys = [
        "instance_image",
        "instance",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
        "instance_bbox",
        "centroid",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["instance_image"]).shape == (1, 226, 226)
    assert samples[0]["instance"].shape == (1, 2, 2)

    # test `is_rgb`
    config.data_config.preprocessing.scale = 1.0
    config.data_config.preprocessing.is_rgb = True
    samples = []
    for idx, lf in enumerate(labels):
        res = centered_instance_data_chunks(
            (lf, idx),
            data_config=config.data_config,
            anchor_ind=0,
            crop_size=(160, 160),
            max_instances=2,
            max_hw=max_hw,
        )
        samples.extend(res)

    gt_keys = [
        "instance_image",
        "instance",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
        "instance_bbox",
        "centroid",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["instance_image"]).shape == (3, 226, 226)
    assert samples[0]["instance"].shape == (1, 2, 2)


def test_centroid_data_chunks(minimal_instance, config):
    """Test `centroid_data_chunks` function."""
    labels = sio.load_slp(minimal_instance)
    transform = T.ToTensor()
    max_hw = get_max_height_width(labels)
    samples = []
    for idx, lf in enumerate(labels):
        samples.append(
            centroid_data_chunks(
                (lf, idx),
                data_config=config.data_config,
                max_instances=4,
                anchor_ind=0,
                max_hw=max_hw,
            )
        )

    assert len(samples) == 1

    gt_keys = [
        "image",
        "instances",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
        "centroids",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["image"]).shape == (1, 384, 384)
    assert samples[0]["instances"].shape == (1, 4, 2, 2)
    assert samples[0]["centroids"].shape == (1, 4, 2)

    # test  `is_rgb`
    samples = []
    config.data_config.preprocessing.is_rgb = True
    for idx, lf in enumerate(labels):
        samples.append(
            centroid_data_chunks(
                (lf, idx),
                data_config=config.data_config,
                max_instances=2,
                anchor_ind=0,
                max_hw=max_hw,
            )
        )

    gt_keys = [
        "image",
        "instances",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
        "centroids",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["image"]).shape == (3, 384, 384)
    assert samples[0]["instances"].shape == (1, 2, 2, 2)
    assert samples[0]["centroids"].shape == (1, 2, 2)


def test_single_instance_data_chunks(minimal_instance, config):
    """Test `single_instance_data_chunks` function."""
    labels = sio.load_slp(minimal_instance)
    transform = T.ToTensor()
    max_hw = get_max_height_width(labels)
    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    samples = []
    for idx, lf in enumerate(labels):
        samples.append(
            single_instance_data_chunks(
                (lf, idx), data_config=config.data_config, max_hw=max_hw
            )
        )

    assert len(samples) == 1

    gt_keys = [
        "image",
        "instances",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
    ]
    for k in gt_keys:
        assert k in samples[0]

    assert transform(samples[0]["image"]).shape == (1, 384, 384)
    assert samples[0]["instances"].shape == (1, 1, 2, 2)

    # test `is_rgb`
    config.data_config.preprocessing.is_rgb = True
    samples = []
    for idx, lf in enumerate(labels):
        samples.append(
            single_instance_data_chunks(
                (lf, idx), data_config=config.data_config, max_hw=max_hw
            )
        )

    gt_keys = [
        "image",
        "instances",
        "frame_idx",
        "video_idx",
        "num_instances",
        "orig_size",
    ]
    for k in gt_keys:
        assert k in samples[0]
    assert transform(samples[0]["image"]).shape == (3, 384, 384)
    assert samples[0]["instances"].shape == (1, 1, 2, 2)
