"""Custom `litdata.StreamingDataset`s for different models."""
from omegaconf import DictConfig
import litdata as ld
import torch

from sleap_nn.data.augmentation import apply_geometric_augmentation, apply_intensity_augmentation
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs

class BottomUpStreamingDataset(ld.StreamingDataset):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
    
    def __getitem__(self, index):
        ex = super().__getitem__(index)

        # Augmentation
        ex["image"], ex["instances"] = apply_intensity_augmentation(ex["image"],
                                                                    ex["instances"],)
                                                                    # TODO)

        img_hw = ex["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_multiconfmaps(ex["instances"],
                                                 img_hw=img_hw,
                                                 num_instances=ex["num_instances"],
                                                 sigma=sigma,
                                                 output_stride=output_stride)

        # pafs
        pafs = generate_pafs(ex["instances"],
                             img_hw=img_hw,
                             sigma=sigma,
                             output_stride=output_stride,
                             edge_inds=edge_inds,
                             flatten_channels=True,
                             )

        sample = ex
        del sample["instances"]
        sample["confidence_maps"] = confidence_maps
        sample["part_affinity_fields"] = pafs

        return sample