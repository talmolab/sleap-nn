"""This module implements base configurations for data pipelines."""

from omegaconf import OmegaConf

# Base TopDownConfmapsPipeline data config.
base_topdown_data_config = OmegaConf.create(
    {
        "preprocessing": {
            "crop_hw": (160, 160),
            "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
        },
        "augmentation_config": {"random_crop": 0.0, "random_crop_hw": (160, 160)},
    }
)
