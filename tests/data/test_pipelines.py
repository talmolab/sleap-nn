import torch
from omegaconf import OmegaConf

from sleap_nn.data.confidence_maps import ConfidenceMapGenerator
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.pipelines import SleapDataset, TopdownConfmapsPipeline
from sleap_nn.data.providers import LabelsReader


def test_sleap_dataset(minimal_instance):
    datapipe = LabelsReader.from_filename(filename=minimal_instance)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = InstanceCropper(datapipe, (160, 160))
    datapipe = ConfidenceMapGenerator(datapipe, sigma=1.5, output_stride=2)
    datapipe = SleapDataset(datapipe)

    sample = next(iter(datapipe))
    assert len(sample) == 2
    assert sample[0].shape == (1, 160, 160)
    assert sample[1].shape == (2, 80, 80)


def test_topdownconfmapspipeline(minimal_instance):
    base_topdown_data_config = OmegaConf.create(
        {
            "preprocessing": {
                "crop_hw": (160, 160),
                "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
            },
            "augmentation_config": {
                "random_crop": {"random_crop_p": 1.0, "random_crop_hw": (160, 160)},
                "augmentations": {
                    "rotation": 15.0,
                    "scale": 0.05,
                    "translate": (0.02, 0.02),
                    "affine_p": 0.5,
                    "uniform_noise": (0.0, 0.04),
                    "uniform_noise_p": 0.5,
                    "gaussian_noise_mean": 0.02,
                    "gaussian_noise_std": 0.004,
                    "gaussian_noise_p": 0.5,
                    "contrast": (0.5, 2.0),
                    "contrast_p": 0.5,
                    "brightness": 0.0,
                    "brightness_p": 0.5,
                    "erase_scale": (0.0001, 0.01),
                    "erase_ratio": (1, 1),
                    "erase_p": 0.5,
                    "mixup_lambda": None,
                    "mixup_p": 0.5,
                },
            },
        }
    )

    pipeline = TopdownConfmapsPipeline(data_config=base_topdown_data_config)
    datapipe = pipeline.make_base_pipeline(
        data_provider=LabelsReader, filename=minimal_instance
    )

    sample = next(iter(datapipe))
    assert len(sample) == 2
    assert sample[0].shape == (1, 160, 160)
    assert sample[1].shape == (2, 80, 80)

    pipeline = TopdownConfmapsPipeline(data_config=base_topdown_data_config)
    datapipe = pipeline.make_training_pipeline(
        data_provider=LabelsReader, filename=minimal_instance
    )

    sample = next(iter(datapipe))
    assert len(sample) == 2
    assert sample[0].shape == (1, 160, 160)
    assert sample[1].shape == (2, 80, 80)
