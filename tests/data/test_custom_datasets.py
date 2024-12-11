from omegaconf import DictConfig, OmegaConf
import sleap_io as sio
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    CenteredInstanceDataset,
    CentroidDataset,
    SingleInstanceDataset,
)


def test_bottomup_dataset(minimal_instance):
    """Test BottomUpDataset class."""
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
    pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

    dataset = BottomUpDataset(
        data_config=base_bottom_config,
        max_stride=32,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=sio.load_slp(minimal_instance),
        apply_aug=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["part_affinity_fields"].shape == (2, 96, 96)

    # with scaling
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 0.5,
                "is_rgb": True,
            },
            "use_augmentations_train": False,
        }
    )

    dataset = BottomUpDataset(
        data_config=base_bottom_config,
        max_stride=32,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=sio.load_slp(minimal_instance),
        apply_aug=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 3, 192, 192)
    assert sample["confidence_maps"].shape == (1, 2, 96, 96)
    assert sample["part_affinity_fields"].shape == (2, 48, 48)

    # with padding
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "intensity": {
                    "uniform_noise_min": 0.0,
                    "uniform_noise_max": 0.04,
                    "uniform_noise_p": 0.5,
                    "gaussian_noise_mean": 0.02,
                    "gaussian_noise_std": 0.004,
                    "gaussian_noise_p": 0.5,
                    "contrast_min": 0.5,
                    "contrast_max": 2.0,
                    "contrast_p": 0.5,
                    "brightness": 0.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation": 15.0,
                    "scale": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda": None,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    dataset = BottomUpDataset(
        data_config=base_bottom_config,
        max_stride=256,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=sio.load_slp(minimal_instance),
        apply_aug=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]

    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 512, 512)
    assert sample["confidence_maps"].shape == (1, 2, 256, 256)
    assert sample["part_affinity_fields"].shape == (2, 128, 128)


def test_centered_instance_dataset(minimal_instance):
    """Test the CenteredInstanceDataset."""
    crop_hw = (160, 160)
    base_topdown_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": False,
        },
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    dataset = CenteredInstanceDataset(
        data_config=base_topdown_data_config,
        max_stride=16,
        confmap_head_config=confmap_head,
        crop_hw=crop_hw,
        labels=sio.load_slp(minimal_instance),
        apply_aug=base_topdown_data_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "frame_idx",
        "video_idx",
        "orig_size",
        "num_instances",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)

    base_topdown_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": True,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "intensity": {
                    "uniform_noise_min": 0.0,
                    "uniform_noise_max": 0.04,
                    "uniform_noise_p": 0.5,
                    "gaussian_noise_mean": 0.02,
                    "gaussian_noise_std": 0.004,
                    "gaussian_noise_p": 0.5,
                    "contrast_min": 0.5,
                    "contrast_max": 2.0,
                    "contrast_p": 0.5,
                    "brightness": 0.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation": 15.0,
                    "scale": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda": None,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    dataset = CenteredInstanceDataset(
        data_config=base_topdown_data_config,
        max_stride=8,
        confmap_head_config=confmap_head,
        crop_hw=(100, 100),
        labels=sio.load_slp(minimal_instance),
        apply_aug=base_topdown_data_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "frame_idx",
        "video_idx",
        "orig_size",
        "num_instances",
    ]

    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 3, 104, 104)
    assert sample["confidence_maps"].shape == (1, 2, 52, 52)

    # Test with resizing and padding
    base_topdown_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 2.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "intensity": {
                    "uniform_noise_min": 0.0,
                    "uniform_noise_max": 0.04,
                    "uniform_noise_p": 0.5,
                    "gaussian_noise_mean": 0.02,
                    "gaussian_noise_std": 0.004,
                    "gaussian_noise_p": 0.5,
                    "contrast_min": 0.5,
                    "contrast_max": 2.0,
                    "contrast_p": 0.5,
                    "brightness": 0.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation": 15.0,
                    "scale": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda": None,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    dataset = CenteredInstanceDataset(
        data_config=base_topdown_data_config,
        max_stride=16,
        confmap_head_config=confmap_head,
        crop_hw=(100, 100),
        labels=sio.load_slp(minimal_instance),
        apply_aug=base_topdown_data_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "frame_idx",
        "video_idx",
        "orig_size",
        "num_instances",
    ]

    sample = next(iter(dataset))
    assert len(dataset) == 2
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 208, 208)
    assert sample["confidence_maps"].shape == (1, 2, 104, 104)


def test_centroid_dataset(minimal_instance):
    """Test CentroidDataset class."""
    base_centroid_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": True,
            },
            "use_augmentations_train": False,
        }
    )
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    dataset = CentroidDataset(
        data_config=base_centroid_data_config,
        max_stride=32,
        confmap_head_config=confmap_head,
        apply_aug=base_centroid_data_config.use_augmentations_train,
        labels=sio.load_slp(minimal_instance),
    )

    gt_sample_keys = [
        "image",
        "instances",
        "centroids",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 3, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)

    base_centroid_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "intensity": {
                    "uniform_noise_min": 0.0,
                    "uniform_noise_max": 0.04,
                    "uniform_noise_p": 0.5,
                    "gaussian_noise_mean": 0.02,
                    "gaussian_noise_std": 0.004,
                    "gaussian_noise_p": 0.5,
                    "contrast_min": 0.5,
                    "contrast_max": 2.0,
                    "contrast_p": 0.5,
                    "brightness": 0.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation": 15.0,
                    "scale": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda": None,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    dataset = CentroidDataset(
        data_config=base_centroid_data_config,
        max_stride=32,
        confmap_head_config=confmap_head,
        apply_aug=base_centroid_data_config.use_augmentations_train,
        labels=sio.load_slp(minimal_instance),
    )

    gt_sample_keys = [
        "image",
        "instances",
        "centroids",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
    ]

    sample = next(iter(dataset))
    assert len(dataset) == 1
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)


def test_single_instance_dataset(minimal_instance):
    """Test the SingleInstanceDataset."""
    labels = sio.load_slp(minimal_instance)

    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    base_singleinstance_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 2.0,
                "is_rgb": True,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    dataset = SingleInstanceDataset(
        data_config=base_singleinstance_data_config,
        max_stride=8,
        confmap_head_config=confmap_head,
        labels=labels,
        apply_aug=base_singleinstance_data_config.use_augmentations_train,
    )

    sample = next(iter(dataset))
    assert len(dataset) == 1

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "num_instances",
        "confidence_maps",
        "orig_size",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 3, 768, 768)
    assert sample["confidence_maps"].shape == (1, 2, 384, 384)

    base_singleinstance_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "intensity": {
                    "uniform_noise_min": 0.0,
                    "uniform_noise_max": 0.04,
                    "uniform_noise_p": 0.5,
                    "gaussian_noise_mean": 0.02,
                    "gaussian_noise_std": 0.004,
                    "gaussian_noise_p": 0.5,
                    "contrast_min": 0.5,
                    "contrast_max": 2.0,
                    "contrast_p": 0.5,
                    "brightness": 0.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation": 15.0,
                    "scale": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda": None,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    dataset = SingleInstanceDataset(
        data_config=base_singleinstance_data_config,
        max_stride=8,
        confmap_head_config=confmap_head,
        labels=labels,
        apply_aug=base_singleinstance_data_config.use_augmentations_train,
    )

    sample = next(iter(dataset))

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["instances"].shape == (1, 1, 2, 2)
