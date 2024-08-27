from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig

import sleap_io as sio

from sleap_nn.data.confidence_maps import ConfidenceMapGenerator
from sleap_nn.data.general import KeyFilter
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import SizeMatcher, Resizer
from sleap_nn.data.pipelines import (
    find_instance_crop_size,
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
    CentroidConfmapsPipeline,
    BottomUpPipeline,
)
from sleap_nn.data.providers import LabelsReader


def test_find_instance_crop_size(minimal_instance):
    """Test `find_instance_crop_size` function."""
    labels = sio.load_slp(minimal_instance)
    crop_size = find_instance_crop_size(labels)
    assert crop_size == 74

    crop_size = find_instance_crop_size(labels, min_crop_size=100)
    assert crop_size == 100

    crop_size = find_instance_crop_size(labels, padding=10)
    assert crop_size == 84


def test_key_filter(minimal_instance):
    """Test KeyFilter module."""
    datapipe = LabelsReader.from_filename(filename=minimal_instance)
    datapipe = Normalizer(datapipe)
    datapipe = SizeMatcher(datapipe)
    datapipe = Resizer(datapipe)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = InstanceCropper(datapipe, (160, 160))
    datapipe = ConfidenceMapGenerator(
        datapipe,
        sigma=1.5,
        output_stride=2,
        image_key="instance_image",
        instance_key="instance",
    )
    datapipe = KeyFilter(datapipe, keep_keys=None)

    gt_sample_keys = [
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "video_idx",
        "frame_idx",
        "num_instances",
        "orig_size",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)
    assert sample["frame_idx"] == 0
    assert sample["video_idx"] == 0
    assert sample["num_instances"] == 2

    datapipe = LabelsReader.from_filename(filename=minimal_instance)
    datapipe = Normalizer(datapipe)
    datapipe = SizeMatcher(datapipe)
    datapipe = Resizer(datapipe, keep_original=True)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = InstanceCropper(datapipe, (160, 160))
    datapipe = ConfidenceMapGenerator(
        datapipe,
        sigma=1.5,
        output_stride=2,
        image_key="instance_image",
        instance_key="instance",
    )
    datapipe = KeyFilter(datapipe, keep_keys=None)

    gt_sample_keys = [
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "video_idx",
        "frame_idx",
        "num_instances",
        "orig_size",
        "original_image",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)


def test_topdownconfmapspipeline(minimal_instance):
    """Test the TopdownConfmapsPipeline."""
    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
                "crop_hw": (160, 160),
            },
            "use_augmentations_train": False,
        },
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=16, confmap_head=confmap_head
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_topdown_data_config.use_augmentations_train,
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
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)

    # test crop size
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
                "crop_hw": None,
            },
            "use_augmentations_train": False,
        },
    )
    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=16, confmap_head=confmap_head
    )

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_topdown_data_config.use_augmentations_train,
    )
    sample = next(iter(datapipe))
    assert sample["instance_image"].shape == (1, 1, 80, 80)

    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
                "crop_hw": None,
                "min_crop_size": 100,
            },
            "use_augmentations_train": False,
        },
    )
    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=16, confmap_head=confmap_head
    )

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_topdown_data_config.use_augmentations_train,
    )
    sample = next(iter(datapipe))
    assert sample["instance_image"].shape == (1, 1, 112, 112)

    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
                "crop_hw": (100, 100),
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "random_crop": {
                    "random_crop_p": 0.0,
                    "crop_height": 160,
                    "crop_width": 160,
                },
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

    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=8, confmap_head=confmap_head
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_topdown_data_config.use_augmentations_train,
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

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 104, 104)
    assert sample["confidence_maps"].shape == (1, 2, 52, 52)

    # Test with resizing and padding
    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 2.0,
                "is_rgb": False,
                "crop_hw": (100, 100),
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "random_crop": {
                    "random_crop_p": 0.0,
                    "crop_height": 160,
                    "crop_width": 160,
                },
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

    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=16, confmap_head=confmap_head
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_topdown_data_config.use_augmentations_train,
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

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 208, 208)
    assert sample["confidence_maps"].shape == (1, 2, 104, 104)


def test_singleinstanceconfmapspipeline(minimal_instance):
    """Test the SingleInstanceConfmapsPipeline."""
    labels = sio.load_slp(minimal_instance)

    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    base_singleinstance_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 2.0,
                "is_rgb": False,
            },
            "use_augmentations_train": False,
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    pipeline = SingleInstanceConfmapsPipeline(
        data_config=base_singleinstance_data_config,
        max_stride=8,
        confmap_head=confmap_head,
    )
    data_provider = LabelsReader(labels=labels)

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_singleinstance_data_config.use_augmentations_train,
    )

    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "instances",
        "confidence_maps",
        "orig_size",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 768, 768)
    assert sample["confidence_maps"].shape == (1, 2, 384, 384)

    base_singleinstance_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "random_crop": {
                    "random_crop_p": 1.0,
                    "crop_height": 160,
                    "crop_width": 160,
                },
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

    pipeline = SingleInstanceConfmapsPipeline(
        data_config=base_singleinstance_data_config,
        max_stride=8,
        confmap_head=confmap_head,
    )

    data_provider = LabelsReader(labels=labels)
    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_singleinstance_data_config.use_augmentations_train,
    )

    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "instances",
        "confidence_maps",
        "orig_size",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)
    assert sample["instances"].shape == (1, 1, 2, 2)


def test_centroidconfmapspipeline(minimal_instance):
    """Test CentroidConfmapsPipeline class."""
    base_centroid_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": False,
        }
    )
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    pipeline = CentroidConfmapsPipeline(
        data_config=base_centroid_data_config, max_stride=32, confmap_head=confmap_head
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_centroid_data_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
    ]
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)

    base_centroid_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "random_crop": {
                    "random_crop_p": 1.0,
                    "crop_height": 160,
                    "crop_width": 160,
                },
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

    pipeline = CentroidConfmapsPipeline(
        data_config=base_centroid_data_config, max_stride=32, confmap_head=confmap_head
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_centroid_data_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 160, 160)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 80, 80)


def test_bottomuppipeline(minimal_instance):
    """Test BottomUpPipeline class."""
    base_bottom_config = OmegaConf.create(
        {
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

    pipeline = BottomUpPipeline(
        data_config=base_bottom_config,
        max_stride=32,
        confmap_head=confmap_head,
        pafs_head=pafs_head,
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)
    assert sample["part_affinity_fields"].shape == (96, 96, 2)

    # with scaling
    base_bottom_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 0.5,
                "is_rgb": False,
            },
            "use_augmentations_train": False,
        }
    )

    pipeline = BottomUpPipeline(
        data_config=base_bottom_config,
        max_stride=32,
        confmap_head=confmap_head,
        pafs_head=pafs_head,
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 192, 192)
    assert sample["confidence_maps"].shape == (1, 2, 96, 96)
    assert sample["part_affinity_fields"].shape == (48, 48, 2)

    # with padding
    base_bottom_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "random_crop": {
                    "random_crop_p": 1.0,
                    "crop_height": 100,
                    "crop_width": 100,
                },
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

    pipeline = BottomUpPipeline(
        data_config=base_bottom_config,
        max_stride=32,
        confmap_head=confmap_head,
        pafs_head=pafs_head,
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 128, 128)
    assert sample["confidence_maps"].shape == (1, 2, 64, 64)
    assert sample["part_affinity_fields"].shape == (32, 32, 2)

    # with random crop
    base_bottom_config = OmegaConf.create(
        {
            "preprocessing": {
                "max_height": None,
                "max_width": None,
                "scale": 1.0,
                "is_rgb": False,
            },
            "use_augmentations_train": True,
            "augmentation_config": {
                "random_crop": {
                    "random_crop_p": 1.0,
                    "crop_height": 160,
                    "crop_width": 160,
                },
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

    pipeline = BottomUpPipeline(
        data_config=base_bottom_config,
        max_stride=32,
        confmap_head=confmap_head,
        pafs_head=pafs_head,
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(
        data_provider=data_provider,
        use_augmentations=base_bottom_config.use_augmentations_train,
    )

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "part_affinity_fields",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)
