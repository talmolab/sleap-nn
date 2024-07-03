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
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
    CentroidConfmapsPipeline,
    BottomUpPipeline,
)
from sleap_nn.data.providers import LabelsReader


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
        "image",
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "video_idx",
        "frame_idx",
        "num_instances",
        "orig_size",
        "scale",
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
        "image",
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "video_idx",
        "frame_idx",
        "num_instances",
        "orig_size",
        "scale",
        "original_image",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)


def test_topdownconfmapspipeline(minimal_instance):
    """Test the TopdownConfmapsPipeline."""
    base_topdown_data_config = OmegaConf.create(
        {
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {
                "crop_hw": (160, 160),
            },
            "augmentation_config": {
                "random_crop": {"random_crop_p": 1.0, "random_crop_hw": (160, 160)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=16, confmap_head=confmap_head
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "frame_idx",
        "video_idx",
        "orig_size",
        "num_instances",
        "scale",
    ]
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)

    base_topdown_data_config = OmegaConf.create(
        {
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {
                "crop_hw": (100, 100),
            },
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0.0, "random_crop_hw": (160, 160)},
                "use_augmentations": True,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
        }
    )

    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=8, confmap_head=confmap_head
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "frame_idx",
        "video_idx",
        "orig_size",
        "num_instances",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 2.0,
            "is_rgb": False,
            "preprocessing": {
                "crop_hw": (100, 100),
            },
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0.0, "random_crop_hw": (160, 160)},
                "use_augmentations": True,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
        }
    )

    pipeline = TopdownConfmapsPipeline(
        data_config=base_topdown_data_config, max_stride=16, confmap_head=confmap_head
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "confidence_maps",
        "frame_idx",
        "video_idx",
        "orig_size",
        "num_instances",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 2.0,
            "is_rgb": False,
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0.0, "random_crop_hw": (160, 160)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
        }
    )

    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    pipeline = SingleInstanceConfmapsPipeline(
        data_config=base_singleinstance_data_config,
        max_stride=8,
        confmap_head=confmap_head,
    )
    data_provider = LabelsReader(labels=labels)

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "instances",
        "confidence_maps",
        "orig_size",
        "scale",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 768, 768)
    assert sample["confidence_maps"].shape == (1, 2, 384, 384)

    base_singleinstance_data_config = OmegaConf.create(
        {
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "augmentation_config": {
                "random_crop": {"random_crop_p": 1.0, "random_crop_hw": (160, 160)},
                "use_augmentations": True,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
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
    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "instances",
        "confidence_maps",
        "orig_size",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {},
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0.0, "random_crop_hw": (160, 160)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
        }
    )
    confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2, "anchor_part": 0})

    pipeline = CentroidConfmapsPipeline(
        data_config=base_centroid_data_config, max_stride=32, confmap_head=confmap_head
    )
    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
        "scale",
    ]
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)

    base_centroid_data_config = OmegaConf.create(
        {
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {},
            "augmentation_config": {
                "random_crop": {"random_crop_p": 1.0, "random_crop_hw": (160, 160)},
                "use_augmentations": True,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
        }
    )

    pipeline = CentroidConfmapsPipeline(
        data_config=base_centroid_data_config, max_stride=32, confmap_head=confmap_head
    )

    data_provider = LabelsReader(labels=sio.load_slp(minimal_instance))
    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "centroids_confidence_maps",
        "orig_size",
        "num_instances",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {},
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0.0, "random_crop_hw": (160, 160)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
                },
            },
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

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 0.5,
            "is_rgb": False,
            "preprocessing": {},
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0.0, "random_crop_hw": (160, 160)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
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

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {},
            "augmentation_config": {
                "random_crop": {"random_crop_p": 1.0, "random_crop_hw": (100, 100)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
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

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "scale",
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
            "max_height": None,
            "max_width": None,
            "scale": 1.0,
            "is_rgb": False,
            "preprocessing": {},
            "augmentation_config": {
                "random_crop": {"random_crop_p": 1.0, "random_crop_hw": (160, 160)},
                "use_augmentations": True,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0.5,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0.5,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0.5,
                        "brightness": 0.0,
                        "brightness_p": 0.5,
                    },
                    "geometric": {
                        "rotation": 15.0,
                        "scale": 0.05,
                        "translate": (0.02, 0.02),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0.5,
                        "mixup_lambda": None,
                        "mixup_p": 0.5,
                    },
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
    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "confidence_maps",
        "orig_size",
        "num_instances",
        "scale",
        "part_affinity_fields",
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)
