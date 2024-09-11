"""Custom `litdata.StreamingDataset`s for different models."""
from kornia.geometry.transform import crop_and_resize
from omegaconf import DictConfig
from typing import List, Optional, Tuple
import litdata as ld
import torch

from sleap_nn.data.augmentation import apply_geometric_augmentation, apply_intensity_augmentation
from sleap_nn.data.confidence_maps import generate_confmaps, generate_multiconfmaps
from sleap_nn.data.edge_maps import generate_pafs
from sleap_nn.data.instance_cropping import make_centered_bboxes

class BottomUpStreamingDataset(ld.StreamingDataset):
    def __init__(self, augmentation_config: DictConfig,
                 confmap_head: DictConfig,
                 pafs_head: DictConfig,
                 edge_inds: list,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_config = augmentation_config
        self.confmap_head = confmap_head
        self.pafs_head = pafs_head
        self.edge_inds = edge_inds

    def __getitem__(self, index):
        ex = super().__getitem__(index)

        # Augmentation
        if "intensity" in self.aug_config:
            ex["image"], ex["instances"] = apply_intensity_augmentation(ex["image"],
                                                                        ex["instances"],
                                                                        **self.aug_config.intensity)

        if "geometric" in self.aug_config:
            ex["image"], ex["instances"] = apply_geometric_augmentation(ex["image"],
                                                                        ex["instances"],
                                                                        **self.aug_config.geometric)

        img_hw = ex["image"].shape[-2:]

        # Generate confidence maps
        confidence_maps = generate_multiconfmaps(ex["instances"],
                                                 img_hw=img_hw,
                                                 num_instances=ex["num_instances"],
                                                 sigma=self.confmap_head.sigma,
                                                 output_stride=self.confmap_head.output_stride,
                                                 is_centroids=False)

        # pafs
        pafs = generate_pafs(ex["instances"],
                             img_hw=img_hw,
                             sigma=self.pafs_head.sigma,
                             output_stride=self.pafs_head.output_stride,
                             edge_inds=torch.Tensor(self.edge_inds),
                             flatten_channels=True,
                             )

        sample = ex
        del sample["instances"]
        sample["confidence_maps"] = confidence_maps
        sample["part_affinity_fields"] = pafs

        return sample
    
class CenteredInstanceStreamingDataset(ld.StreamingDataset):
    def __init__(self, augmentation_config: DictConfig,
                 confmap_head: DictConfig,
                 crop_hw: Tuple[int],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_config = augmentation_config
        self.confmap_head = confmap_head
        self.crop_hw = crop_hw

    def __getitem__(self, index):
        ex = super().__getitem__(index)

        # Augmentation
        if "intensity" in self.aug_config:
            ex["image"], ex["instances"] = apply_intensity_augmentation(ex["image"],
                                                                        ex["instances"],
                                                                        **self.aug_config.intensity)

        if "geometric" in self.aug_config:
            ex["image"], ex["instances"] = apply_geometric_augmentation(ex["image"],
                                                                        ex["instances"],
                                                                        **self.aug_config.geometric)

        img_hw = ex["image"].shape[-2:]

        # Re-crop to original crop size
        ex["instance"] = ex["instance"].squeeze(dim=0) # n_samples
        ex["instance_bbox"] = torch.unsqueeze(
            make_centered_bboxes(ex["centroid"], self.crop_hw[0], self.crop_hw[1]), 0
        )
        ex["instance_image"] = crop_and_resize(
            ex["instance_image"],
            boxes=ex["instance_bbox"],
            size=self.crop_hw
        )

        # Generate confidence maps
        confidence_maps = generate_confmaps(ex["instances"],
                                                 img_hw=img_hw,
                                                 sigma=self.confmap_head.sigma,
                                                 output_stride=self.confmap_head.output_stride,)

        sample = ex
        del sample["instances"]
        sample["confidence_maps"] = confidence_maps

        return sample