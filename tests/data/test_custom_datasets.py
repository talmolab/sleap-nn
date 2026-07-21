from omegaconf import DictConfig, OmegaConf
import sleap_io as sio
import torch
import pytest
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    BottomUpMultiClassDataset,
    CenteredInstanceDataset,
    TopDownCenteredInstanceMultiClassDataset,
    CentroidDataset,
    SingleInstanceDataset,
    InfiniteDataLoader,
    get_steps_per_epoch,
    labels_have_user_centroids,
    resolve_centroid_source,
)
from sleap_nn.data.instance_centroids import generate_centroids


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


def test_single_instance_dataset_filters_out_of_frame(minimal_instance):
    """Nodes pushed off-frame by augmentation get an empty confmap."""
    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = lf.instances[:1]

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

    # Deterministic 4x zoom about the image center pushes the off-center node
    # outside the 384x384 frame; all other augmentations are disabled.
    geometric_aug = {
        "scale_min": 4.0,
        "scale_max": 4.0,
        "scale_p": 1.0,
    }
    dataset = SingleInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=confmap_head,
        labels=[labels],
        apply_aug=True,
        geometric_aug=geometric_aug,
        cache_img="memory",
    )
    dataset._fill_cache([labels])
    sample = next(iter(dataset))

    instances = sample["instances"][0]  # (n_instances, n_nodes, 2)
    cms = sample["confidence_maps"][0]  # (n_nodes, h, w)

    oob = torch.isnan(instances[0]).any(dim=-1)  # single instance
    assert oob.any()  # at least one node was pushed off-frame and masked
    for n in range(oob.shape[0]):
        if oob[n]:
            assert cms[n].sum() == 0  # empty target for off-frame node


def test_centered_instance_dataset_filters_out_of_crop(minimal_instance):
    """Nodes outside the crop get an empty confmap; in-crop nodes are preserved."""
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    def build(crop_size):
        dataset = CenteredInstanceDataset(
            max_stride=2,
            scale=1.0,
            confmap_head_config=confmap_head,
            crop_size=crop_size,
            labels=[sio.load_slp(minimal_instance)],
            apply_aug=False,
            cache_img="memory",
        )
        dataset._fill_cache([sio.load_slp(minimal_instance)])
        return next(iter(dataset))

    # Tiny crop: both nodes (~30px from the centroid) fall outside the crop, so they
    # are masked to NaN and their confidence maps are empty (all-zero).
    small = build(20)
    inst_s = small["instance"][0]  # (n_nodes, 2)
    cms_s = small["confidence_maps"][0]  # (n_nodes, h, w)
    oob = torch.isnan(inst_s).any(dim=-1)
    assert oob.any()
    for n in range(inst_s.shape[0]):
        if oob[n]:
            assert cms_s[n].sum() == 0

    # Large crop: both nodes stay inside, so nothing is masked and every node has a
    # non-empty confidence map.
    large = build(160)
    inst_l = large["instance"][0]
    cms_l = large["confidence_maps"][0]
    assert not torch.isnan(inst_l).any()
    assert (cms_l.sum(dim=(-1, -2)) > 0).all()


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


def test_centroid_dataset_user_centroids(minimal_instance):
    """CentroidDataset targets first-class user centroids, else falls back.

    When a frame carries ``sio.UserCentroid`` annotations, the centroid target
    (and its confidence map) must be built at those *annotated* locations, not
    at the instance-keypoint-derived centroid. Frames without user centroids
    must still fall back to the anchor/mean-of-visible-nodes path.
    """
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    # --- Frame WITH user centroids: annotated points far from the pose means.
    # minimal_instance has 2 instances with keypoint means ~ (122, 180) and
    # (242, 195); pick centroids clearly distinct from those.
    user_xy = [[50.0, 60.0], [300.0, 320.0]]
    labels = sio.load_slp(minimal_instance)
    labels[0].centroids = [
        sio.UserCentroid(x=user_xy[0][0], y=user_xy[0][1]),
        sio.UserCentroid(x=user_xy[1][0], y=user_xy[1][1]),
    ]

    dataset = CentroidDataset(
        max_stride=32,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=False,
        labels=[labels],
    )
    sample = next(iter(dataset))

    # No extra keys leak into the sample (parity with test_centroid_dataset).
    assert set(sample.keys()) == {
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
    }

    eff_scale = float(sample["eff_scale"])
    expected = torch.tensor(user_xy, dtype=torch.float32) * eff_scale

    # num_instances tracks the user-centroid count.
    assert sample["num_instances"] == 2
    # Target sits at the ANNOTATED centroids, not the instance-keypoint means.
    produced = sample["centroids"][0, :2]
    assert torch.allclose(produced, expected, atol=1e-4)
    # Sanity: annotated centroids are far from the instance-derived fallback.
    fallback = generate_centroids(sample["instances"], anchor_ind=None)[0, :2]
    assert not torch.allclose(produced, fallback, atol=5.0)

    # The confidence map peaks at each annotated centroid (in stride coords).
    output_stride = confmap_head["output_stride"]
    cmap = sample["centroids_confidence_maps"][0, 0]
    for cx, cy in expected.tolist():
        r = int(round(cy / output_stride))
        c = int(round(cx / output_stride))
        assert cmap[r, c] > 0.9

    # --- Frame WITHOUT user centroids: fall back to the anchor/mean path.
    labels_fallback = sio.load_slp(minimal_instance)
    assert labels_fallback[0].centroids == []
    dataset_fb = CentroidDataset(
        max_stride=32,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=False,
        labels=[labels_fallback],
    )
    sample_fb = next(iter(dataset_fb))
    fb_expected = generate_centroids(sample_fb["instances"], anchor_ind=None)
    assert torch.allclose(
        sample_fb["centroids"], fb_expected, atol=1e-4, equal_nan=True
    )
    # The fallback target differs from the annotated-centroid target.
    assert not torch.allclose(sample_fb["centroids"][0, :2], expected, atol=5.0)


def test_centroid_dataset_centroid_only_frame(minimal_instance):
    """CentroidDataset keeps frames with user centroids but NO pose instances.

    Pure-centroid seeding: a frame annotated only with ``sio.UserCentroid``
    (no pose instance) is dropped by the instance-required filter for every
    other model, but the centroid model keeps it and targets the annotated
    centroids. The placeholder ``instances`` tensor stays node-consistent so a
    batch that mixes these with normal frames still collates.
    """
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    user_xy = [[70.0, 90.0], [260.0, 300.0]]
    labels = sio.load_slp(minimal_instance)
    # Strip every pose instance, then annotate one frame with user centroids
    # only -> the other frames (no instances, no centroids) are dropped.
    for lf in labels:
        lf.instances = []
    labels[0].centroids = [
        sio.UserCentroid(x=user_xy[0][0], y=user_xy[0][1]),
        sio.UserCentroid(x=user_xy[1][0], y=user_xy[1][1]),
    ]

    dataset = CentroidDataset(
        max_stride=32,
        ensure_rgb=True,
        ensure_grayscale=False,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=False,
        labels=[labels],
    )

    # The centroid-only frame is kept (every other dataset would drop it).
    assert len(dataset.lf_idx_list) == 1
    sample = next(iter(dataset))

    # Same schema as a normal centroid sample.
    assert set(sample.keys()) == {
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
    }

    eff_scale = float(sample["eff_scale"])
    expected = torch.tensor(user_xy, dtype=torch.float32) * eff_scale
    assert sample["num_instances"] == 2
    produced = sample["centroids"][0, :2]
    assert torch.allclose(produced, expected, atol=1e-4)

    # Placeholder instances: all-NaN, shaped to the pose skeleton's node count
    # so batches collate uniformly with instance-bearing frames.
    n_nodes = len(labels.skeletons[0].nodes)
    assert tuple(sample["instances"].shape[1:]) == (dataset.max_instances, n_nodes, 2)
    assert torch.isnan(sample["instances"]).all()

    # Confmap peaks at each annotated centroid (stride coords).
    output_stride = confmap_head["output_stride"]
    cmap = sample["centroids_confidence_maps"][0, 0]
    for cx, cy in expected.tolist():
        r = int(round(cy / output_stride))
        c = int(round(cx / output_stride))
        assert cmap[r, c] > 0.9


def test_resolve_centroid_source(minimal_instance):
    """The centroid source resolves to ONE dataset-wide mode; unset -> inferred."""
    labels_plain = sio.load_slp(minimal_instance)  # no user centroids
    labels_user = sio.load_slp(minimal_instance)
    labels_user[0].centroids = [sio.UserCentroid(x=10.0, y=20.0)]

    # Explicit config wins regardless of what the labels contain.
    assert resolve_centroid_source("user", [labels_plain]) is True
    assert resolve_centroid_source("computed", [labels_user]) is False
    assert resolve_centroid_source("anchor", [labels_user]) is False  # alias
    assert resolve_centroid_source(" User ", [labels_plain]) is True  # normalized

    # Unset -> inferred from the (train) labels.
    assert resolve_centroid_source(None, [labels_user]) is True
    assert resolve_centroid_source(None, [labels_plain]) is False

    # A non-empty unrecognized value is a hard error, not a silent fallback.
    with pytest.raises(ValueError):
        resolve_centroid_source("centroid", [labels_plain])

    # Detection helper agrees.
    assert labels_have_user_centroids([labels_user]) is True
    assert labels_have_user_centroids([labels_plain]) is False


def test_centroid_dataset_source_overrides_per_frame(minimal_instance):
    """The dataset-wide source wins over any per-frame user centroid.

    A frame carrying ``UserCentroid`` annotations must still produce the
    COMPUTED target in computed mode (no per-frame preference) and the user
    target in user mode.
    """
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})
    user_xy = [[50.0, 60.0], [300.0, 320.0]]

    def _labels_with_user_centroids():
        lbls = sio.load_slp(minimal_instance)
        lbls[0].centroids = [
            sio.UserCentroid(x=user_xy[0][0], y=user_xy[0][1]),
            sio.UserCentroid(x=user_xy[1][0], y=user_xy[1][1]),
        ]
        return lbls

    common = dict(
        max_stride=32,
        ensure_rgb=True,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=False,
    )

    # Computed mode ignores the user centroids -> target is the keypoint-derived
    # centroid, NOT the annotated one.
    ds_computed = CentroidDataset(
        labels=[_labels_with_user_centroids()], use_user_centroids=False, **common
    )
    assert ds_computed.use_user_centroids is False
    s = next(iter(ds_computed))
    fb = generate_centroids(s["instances"], anchor_ind=None)
    assert torch.allclose(s["centroids"], fb, atol=1e-4, equal_nan=True)
    user_target = torch.tensor(user_xy, dtype=torch.float32) * float(s["eff_scale"])
    assert not torch.allclose(s["centroids"][0, :2], user_target, atol=5.0)

    # User mode -> target sits at the annotated centroids.
    ds_user = CentroidDataset(
        labels=[_labels_with_user_centroids()], use_user_centroids=True, **common
    )
    assert ds_user.use_user_centroids is True
    s2 = next(iter(ds_user))
    user_target2 = torch.tensor(user_xy, dtype=torch.float32) * float(s2["eff_scale"])
    assert torch.allclose(s2["centroids"][0, :2], user_target2, atol=1e-4)


def test_centroid_dataset_no_mix_frame_dropping(minimal_instance):
    """No mix-and-match: frames that can't supply the chosen target are dropped.

    Two single-frame Labels (each an embedded-image copy of the fixture) let us
    control which frames carry user centroids vs. only pose instances.
    """
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    def _pose_only():
        return sio.load_slp(minimal_instance)  # 2 pose instances, no centroids

    def _with_centroids():
        lbls = sio.load_slp(minimal_instance)  # pose instances + a user centroid
        lbls[0].centroids = [sio.UserCentroid(x=50.0, y=60.0)]
        return lbls

    def _centroid_only():
        lbls = sio.load_slp(minimal_instance)  # user centroid, no pose instance
        for lf in lbls:
            lf.instances = []
        lbls[0].centroids = [sio.UserCentroid(x=50.0, y=60.0)]
        return lbls

    common = dict(
        max_stride=32,
        ensure_rgb=True,
        scale=1.0,
        confmap_head_config=confmap_head,
        apply_aug=False,
    )

    # User mode: the pose-only frame (no user centroid) is dropped.
    ds_user = CentroidDataset(
        labels=[_with_centroids(), _pose_only()], use_user_centroids=True, **common
    )
    assert len(ds_user.lf_idx_list) == 1
    assert all(e.get("user_centroids") for e in ds_user.lf_idx_list)

    # Computed mode: the centroid-only frame (no pose) is dropped, and the
    # surviving frame's user centroid is ignored in favor of a computed target.
    ds_computed = CentroidDataset(
        labels=[_with_centroids(), _centroid_only()], use_user_centroids=False, **common
    )
    assert len(ds_computed.lf_idx_list) == 1
    assert all(e.get("has_pose_instances") for e in ds_computed.lf_idx_list)
    s = next(iter(ds_computed))
    assert not torch.isnan(s["centroids"][0, :2]).all()


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


def test_uint8_pipeline_bottomup(minimal_instance):
    """Test that BottomUpDataset returns uint8 images for GPU normalization."""
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
    pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

    dataset = BottomUpDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=[sio.load_slp(minimal_instance)],
        apply_aug=False,
    )

    sample = next(iter(dataset))
    # Image should be uint8 for GPU normalization (4x bandwidth savings)
    assert (
        sample["image"].dtype == torch.uint8
    ), f"Expected uint8 image for GPU normalization, got {sample["image"].dtype}"


def test_uint8_pipeline_singleinstance(minimal_instance):
    """Test that SingleInstanceDataset returns uint8 images for GPU normalization."""
    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = lf.instances[:1]

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    dataset = SingleInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=confmap_head,
        labels=[labels],
        apply_aug=False,
    )

    sample = next(iter(dataset))
    assert (
        sample["image"].dtype == torch.uint8
    ), f"Expected uint8 image for GPU normalization, got {sample["image"].dtype}"


def test_uint8_pipeline_centroid(minimal_instance):
    """Test that CentroidDataset returns uint8 images for GPU normalization."""
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    dataset = CentroidDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=confmap_head,
        labels=[sio.load_slp(minimal_instance)],
        apply_aug=False,
    )

    sample = next(iter(dataset))
    assert (
        sample["image"].dtype == torch.uint8
    ), f"Expected uint8 image for GPU normalization, got {sample["image"].dtype}"


def test_uint8_pipeline_centered_instance(minimal_instance):
    """Test that CenteredInstanceDataset returns uint8 images for GPU normalization."""
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": None})

    dataset = CenteredInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=confmap_head,
        crop_size=100,
        labels=[sio.load_slp(minimal_instance)],
        apply_aug=False,
    )

    sample = next(iter(dataset))
    assert (
        sample["instance_image"].dtype == torch.uint8
    ), f"Expected uint8 image for GPU normalization, got {sample["instance_image"].dtype}"


def test_flip_augmentation_resolves_symmetries(minimal_instance):
    """BottomUpDataset resolves symmetric_inds and applies flip end-to-end."""
    labels = sio.load_slp(minimal_instance)
    # Add a symmetry between the two nodes of the minimal skeleton.
    skel = labels.skeletons[0]
    node_names = [n.name for n in skel.nodes]
    skel.add_symmetry(node_names[0], node_names[1])

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
    pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

    dataset = BottomUpDataset(
        max_stride=32,
        scale=1.0,
        confmap_head_config=confmap_head,
        pafs_head_config=pafs_head,
        labels=[labels],
        geometric_aug=DictConfig({"flip_p": 1.0, "affine_p": 0.0}),
        apply_aug=True,
    )

    # Symmetric pair resolved from the raw skeleton.
    assert sorted(tuple(sorted(p)) for p in dataset.symmetric_inds) == [(0, 1)]

    # Sample is produced without error and image is the expected shape.
    sample = next(iter(dataset))
    assert sample["image"].shape == (1, 1, 384, 384)


def test_flip_augmentation_warns_without_symmetries(minimal_instance):
    """A warning is emitted when flip is enabled but skeleton has no symmetries."""
    from loguru import logger

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
    pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        dataset = BottomUpDataset(
            max_stride=32,
            scale=1.0,
            confmap_head_config=confmap_head,
            pafs_head_config=pafs_head,
            labels=[sio.load_slp(minimal_instance)],
            geometric_aug=DictConfig({"flip_p": 1.0, "affine_p": 0.0}),
            apply_aug=True,
        )
    finally:
        logger.remove(sink_id)

    assert dataset.symmetric_inds == []
    assert any("no symmetries" in m for m in messages)
