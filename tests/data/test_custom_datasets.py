from omegaconf import DictConfig, OmegaConf
import sleap_io as sio
import torch
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    BottomUpMultiClassDataset,
    CenteredInstanceDataset,
    TopDownCenteredInstanceMultiClassDataset,
    CentroidDataset,
    SingleInstanceDataset,
    InfiniteDataLoader,
    get_steps_per_epoch,
)


def test_bottomup_dataset(minimal_instance, tmp_path):
    """Test BottomUpDataset class."""
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": False,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
    pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

    dataset = BottomUpDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=[sio.load_slp(minimal_instance)],
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
        "labels_idx",
        "eff_scale",
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
                "ensure_rgb": True,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )

    dataset = BottomUpDataset(
        max_stride=32,
        scale=0.5,
        ensure_rgb=True,
        ensure_grayscale=False,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=[sio.load_slp(minimal_instance)],
        cache_img="memory",
        apply_aug=base_bottom_config.use_augmentations_train,
    )
    dataset._fill_cache([sio.load_slp(minimal_instance)])

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
        "labels_idx",
        "eff_scale",
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
                "ensure_rgb": False,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    dataset = BottomUpDataset(
        max_stride=256,
        scale=1.0,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=[sio.load_slp(minimal_instance)],
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
        "labels_idx",
        "eff_scale",
    ]

    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 512, 512)
    assert sample["confidence_maps"].shape == (1, 2, 256, 256)
    assert sample["part_affinity_fields"].shape == (2, 128, 128)

    ## test with disk caching
    dataset = BottomUpDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=[sio.load_slp(minimal_instance)],
        apply_aug=base_bottom_config.use_augmentations_train,
        cache_img="disk",
        cache_img_path=f"{tmp_path}/cache_imgs",
    )
    dataset._fill_cache([sio.load_slp(minimal_instance)])

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
        "labels_idx",
        "eff_scale",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["part_affinity_fields"].shape == (2, 96, 96)


def test_bottomup_multiclass_dataset(minimal_instance, tmp_path):
    """Test BottomUpDataset class."""
    tracked_labels = sio.load_slp(minimal_instance)
    tracks = 0
    for lf in tracked_labels:
        for instance in lf.instances:
            instance.track = sio.Track(f"{tracks}")
            tracks += 1
    tracked_labels.update()
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": False,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
    class_maps_head = DictConfig(
        {
            "sigma": 4,
            "output_stride": 4,
            "classes": [x.name for x in tracked_labels.tracks],
        }
    )

    dataset = BottomUpMultiClassDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        class_maps_head_config=class_maps_head,
        labels=[tracked_labels],
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
        "class_maps",
        "labels_idx",
        "num_tracks",
        "eff_scale",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["class_maps"].shape == (1, 2, 96, 96)

    # with scaling
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 0.5,
                "ensure_rgb": True,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )

    dataset = BottomUpMultiClassDataset(
        max_stride=32,
        scale=0.5,
        ensure_rgb=True,
        ensure_grayscale=False,
        confmap_head_config=confmap_head,
        class_maps_head_config=class_maps_head,
        labels=[tracked_labels],
        cache_img="memory",
        apply_aug=base_bottom_config.use_augmentations_train,
    )
    dataset._fill_cache([tracked_labels])

    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 3, 192, 192)
    assert sample["confidence_maps"].shape == (1, 2, 96, 96)
    assert sample["class_maps"].shape == (1, 2, 48, 48)

    # with padding
    base_bottom_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": False,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )
    dataset = BottomUpMultiClassDataset(
        max_stride=256,
        scale=1.0,
        confmap_head_config=confmap_head,
        class_maps_head_config=class_maps_head,
        labels=[tracked_labels],
        apply_aug=base_bottom_config.use_augmentations_train,
    )

    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 512, 512)
    assert sample["confidence_maps"].shape == (1, 2, 256, 256)
    assert sample["class_maps"].shape == (1, 2, 128, 128)

    ## test with disk caching
    dataset = BottomUpMultiClassDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        class_maps_head_config=class_maps_head,
        labels=[tracked_labels],
        apply_aug=base_bottom_config.use_augmentations_train,
        cache_img="disk",
        cache_img_path=f"{tmp_path}/cache_imgs",
    )
    dataset._fill_cache([tracked_labels])

    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["class_maps"].shape == (1, 2, 96, 96)


def test_centered_instance_dataset(minimal_instance, tmp_path):
    """Test the CenteredInstanceDataset."""
    crop_size = 160
    base_topdown_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": False,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        },
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    ## save imgs
    dataset = CenteredInstanceDataset(
        max_stride=16,
        scale=1.0,
        confmap_head_config=confmap_head,
        crop_size=crop_size,
        labels=[
            sio.load_slp(minimal_instance),
            sio.load_slp(minimal_instance),
            sio.load_slp(minimal_instance),
        ],
        apply_aug=base_topdown_data_config.use_augmentations_train,
        cache_img="disk",
        cache_img_path=f"{tmp_path}/cache_imgs",
    )
    dataset._fill_cache(
        [
            sio.load_slp(minimal_instance),
            sio.load_slp(minimal_instance),
            sio.load_slp(minimal_instance),
        ]
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
        "labels_idx",
        "eff_scale",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)
    assert len(dataset) == 6

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)

    ## memory caching

    dataset = CenteredInstanceDataset(
        max_stride=16,
        scale=1.0,
        confmap_head_config=confmap_head,
        crop_size=crop_size,
        labels=[sio.load_slp(minimal_instance)],
        cache_img="memory",
        apply_aug=base_topdown_data_config.use_augmentations_train,
    )
    dataset._fill_cache([sio.load_slp(minimal_instance)])

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
        "labels_idx",
        "eff_scale",
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
                "ensure_rgb": True,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )

    dataset = CenteredInstanceDataset(
        max_stride=8,
        scale=1.0,
        ensure_rgb=True,
        ensure_grayscale=False,
        confmap_head_config=confmap_head,
        crop_size=100,
        labels=[sio.load_slp(minimal_instance)],
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
        "labels_idx",
        "eff_scale",
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
                "ensure_rgb": False,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )

    dataset = CenteredInstanceDataset(
        max_stride=16,
        scale=2.0,
        confmap_head_config=confmap_head,
        crop_size=100,
        labels=[sio.load_slp(minimal_instance)],
        apply_aug=base_topdown_data_config.use_augmentations_train,
        intensity_aug=base_topdown_data_config.augmentation_config.intensity,
        geometric_aug=base_topdown_data_config.augmentation_config.geometric,
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
        "labels_idx",
        "eff_scale",
    ]

    sample = next(iter(dataset))
    assert len(dataset) == 2
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 208, 208)
    assert sample["confidence_maps"].shape == (1, 2, 104, 104)


def test_centered_multiclass_dataset(minimal_instance, tmp_path):
    """Test the TopDownCenteredInstanceMultiClassDataset."""
    tracked_labels = sio.load_slp(minimal_instance)
    tracks = 0
    for lf in tracked_labels:
        for instance in lf.instances:
            instance.track = sio.Track(f"{tracks}")
            tracks += 1
    tracked_labels.update()

    crop_size = 160
    base_topdown_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": False,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        },
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})
    class_vectors_head = DictConfig(
        {"classes": [x.name for x in tracked_labels.tracks]}
    )

    ## save imgs
    dataset = TopDownCenteredInstanceMultiClassDataset(
        max_stride=16,
        scale=1.0,
        confmap_head_config=confmap_head,
        class_vectors_head_config=class_vectors_head,
        crop_size=crop_size,
        labels=[
            tracked_labels,
            tracked_labels,
            tracked_labels,
        ],
        apply_aug=base_topdown_data_config.use_augmentations_train,
        cache_img="disk",
        cache_img_path=f"{tmp_path}/cache_imgs",
    )
    dataset._fill_cache([tracked_labels, tracked_labels, tracked_labels])

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
        "labels_idx",
        "class_vectors",
        "eff_scale",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)
    assert len(dataset) == 6

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)
    assert sample["class_vectors"].shape == (2,)

    ## memory caching

    dataset = TopDownCenteredInstanceMultiClassDataset(
        max_stride=16,
        scale=1.0,
        confmap_head_config=confmap_head,
        class_vectors_head_config=class_vectors_head,
        crop_size=crop_size,
        labels=[tracked_labels],
        cache_img="memory",
        apply_aug=base_topdown_data_config.use_augmentations_train,
    )
    dataset._fill_cache([tracked_labels])

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
                "ensure_rgb": True,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )

    dataset = TopDownCenteredInstanceMultiClassDataset(
        max_stride=8,
        scale=1.0,
        ensure_rgb=True,
        ensure_grayscale=False,
        confmap_head_config=confmap_head,
        class_vectors_head_config=class_vectors_head,
        crop_size=100,
        labels=[tracked_labels],
        apply_aug=base_topdown_data_config.use_augmentations_train,
    )

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
                "ensure_rgb": False,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )

    dataset = TopDownCenteredInstanceMultiClassDataset(
        max_stride=16,
        scale=2.0,
        confmap_head_config=confmap_head,
        class_vectors_head_config=class_vectors_head,
        crop_size=100,
        labels=[tracked_labels],
        apply_aug=base_topdown_data_config.use_augmentations_train,
        intensity_aug=["uniform_noise", "gaussian_noise"],
    )

    sample = next(iter(dataset))
    assert len(dataset) == 2
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 208, 208)
    assert sample["confidence_maps"].shape == (1, 2, 104, 104)
    assert sample["class_vectors"].shape == (2,)


def test_centroid_dataset(minimal_instance, tmp_path):
    """Test CentroidDataset class."""
    base_centroid_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": True,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    ## save imgs
    dataset = CentroidDataset(
        max_stride=32,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=base_centroid_data_config.use_augmentations_train,
        labels=[sio.load_slp(minimal_instance)],
        cache_img="disk",
        cache_img_path=f"{tmp_path}/cache_imgs",
    )
    dataset._fill_cache([sio.load_slp(minimal_instance)])

    gt_sample_keys = [
        "image",
        "instances",
        "centroids",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
        "labels_idx",
        "eff_scale",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 3, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)

    ## no saving imgs

    dataset = CentroidDataset(
        max_stride=32,
        ensure_rgb=False,
        ensure_grayscale=True,
        scale=1.0,
        confmap_head_config=confmap_head,
        cache_img="memory",
        apply_aug=base_centroid_data_config.use_augmentations_train,
        labels=[sio.load_slp(minimal_instance)],
    )
    dataset._fill_cache([sio.load_slp(minimal_instance)])

    gt_sample_keys = [
        "image",
        "instances",
        "centroids",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
        "labels_idx",
        "eff_scale",
    ]
    sample = next(iter(dataset))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)

    base_centroid_data_config = OmegaConf.create(
        {
            "user_instances_only": True,
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "ensure_rgb": False,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )

    dataset = CentroidDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=base_centroid_data_config.use_augmentations_train,
        labels=[sio.load_slp(minimal_instance)],
        intensity_aug=base_centroid_data_config.augmentation_config.intensity,
        geometric_aug=base_centroid_data_config.augmentation_config.geometric,
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
        "labels_idx",
        "eff_scale",
    ]

    sample = next(iter(dataset))
    assert len(dataset) == 1
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)


def test_single_instance_dataset(minimal_instance, tmp_path):
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
                "ensure_rgb": True,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

    ## saving imgs
    dataset = SingleInstanceDataset(
        max_stride=8,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=2.0,
        confmap_head_config=confmap_head,
        labels=[labels, labels, labels],
        apply_aug=base_singleinstance_data_config.use_augmentations_train,
        cache_img="disk",
        cache_img_path=f"{tmp_path}/cache_imgs",
    )
    dataset._fill_cache([labels, labels, labels])
    sample = next(iter(dataset))
    assert len(dataset) == 3

    gt_sample_keys = [
        "image",
        "instances",
        "video_idx",
        "frame_idx",
        "num_instances",
        "confidence_maps",
        "orig_size",
        "labels_idx",
        "eff_scale",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 3, 768, 768)
    assert sample["confidence_maps"].shape == (1, 2, 384, 384)

    ## no saving imgs

    dataset = SingleInstanceDataset(
        max_stride=8,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=2.0,
        confmap_head_config=confmap_head,
        labels=[labels],
        cache_img="memory",
        apply_aug=base_singleinstance_data_config.use_augmentations_train,
    )
    dataset._fill_cache([labels])

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
        "labels_idx",
        "eff_scale",
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
                "ensure_rgb": False,
                "ensure_grayscale": False,
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
                    "brightness_min": 1.0,
                    "brightness_max": 1.0,
                    "brightness_p": 0.5,
                },
                "geometric": {
                    "rotation_min": -15.0,
                    "rotation_max": 15.0,
                    "scale_min": 0.05,
                    "scale_max": 0.05,
                    "translate_width": 0.02,
                    "translate_height": 0.02,
                    "affine_p": 0.5,
                    "erase_scale_min": 0.0001,
                    "erase_scale_max": 0.01,
                    "erase_ratio_min": 1,
                    "erase_ratio_max": 1,
                    "erase_p": 0.5,
                    "mixup_lambda_min": 0.01,
                    "mixup_lambda_max": 0.05,
                },
            },
        }
    )

    dataset = SingleInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=confmap_head,
        labels=[labels],
        apply_aug=base_singleinstance_data_config.use_augmentations_train,
        intensity_aug="uniform_noise",
        geometric_aug="rotation",
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
        "labels_idx",
        "eff_scale",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["instances"].shape == (1, 1, 2, 2)


def test_infinite_dataloader(minimal_instance, tmp_path):
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
                "ensure_rgb": True,
                "ensure_grayscale": False,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    dataset = SingleInstanceDataset(
        max_stride=8,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=2.0,
        confmap_head_config=confmap_head,
        labels=[labels],
        apply_aug=base_singleinstance_data_config.use_augmentations_train,
        cache_img=None,
    )

    assert len(list(iter(dataset))) == 1

    dl = InfiniteDataLoader(dataset=dataset, batch_size=1, num_workers=0)
    loader = iter(dl)

    for _ in range(10):
        _ = next(loader)

    assert get_steps_per_epoch(dataset=dataset, batch_size=1) == 1
    assert len(dl) == 1

    dl = InfiniteDataLoader(dataset=dataset, batch_size=1, len_dataloader=15)
    assert len(dl) == 15
