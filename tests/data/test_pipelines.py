from omegaconf import OmegaConf

import sleap_io as sio

from sleap_nn.data.confidence_maps import ConfidenceMapGenerator
from sleap_nn.data.general import KeyFilter
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
)
from sleap_nn.data.providers import LabelsReader


def test_key_filter(minimal_instance):
    datapipe = LabelsReader.from_filename(filename=minimal_instance, max_instances=30)
    datapipe = Normalizer(datapipe)
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
        "centroids",
        "instances",
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


def test_topdownconfmapspipeline(minimal_instance):
    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "anchor_ind": None,
                "crop_hw": (160, 160),
                "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
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

    pipeline = TopdownConfmapsPipeline(data_config=base_topdown_data_config)
    data_provider = LabelsReader(
        labels=sio.load_slp(minimal_instance), max_instances=30
    )

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
    ]
    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)

    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "anchor_ind": None,
                "crop_hw": (160, 160),
                "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
            },
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

    pipeline = TopdownConfmapsPipeline(data_config=base_topdown_data_config)

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
    ]

    sample = next(iter(datapipe))
    assert len(sample.keys()) == len(gt_sample_keys)

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance_image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)


def test_singleinstanceconfmapspipeline(minimal_instance):
    labels = sio.load_slp(minimal_instance)

    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    base_singleinstance_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
            },
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

    pipeline = SingleInstanceConfmapsPipeline(
        data_config=base_singleinstance_data_config
    )
    data_provider = LabelsReader(labels=labels, max_instances=1)

    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "instances",
        "confidence_maps",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 384, 384)
    assert sample["confidence_maps"].shape == (1, 2, 192, 192)

    base_singleinstance_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
            },
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
        data_config=base_singleinstance_data_config
    )

    data_provider = LabelsReader(labels=labels, max_instances=1)
    datapipe = pipeline.make_training_pipeline(data_provider=data_provider)

    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "video_idx",
        "frame_idx",
        "instances",
        "confidence_maps",
    ]

    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["image"].shape == (1, 1, 160, 160)
    assert sample["confidence_maps"].shape == (1, 2, 80, 80)
    assert sample["instances"].shape == (1, 1, 2, 2)
