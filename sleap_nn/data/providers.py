"""This module implements pipeline blocks for reading input data such as labels."""

from typing import Dict, Iterator

import numpy as np
import sleap_io as sio
import torch
import copy
from torch.utils.data.datapipes.datapipe import IterDataPipe


def get_max_instances(labels: sio.Labels):
    """Function to get the maximum number of instances in a single Labeled Frame."""
    max_instances = -1
    for lf in labels:
        num_inst = len(lf.instances)
        if num_inst > max_instances:
            max_instances = num_inst
    return max_instances


class LabelsReader(IterDataPipe):
    """IterDataPipe for reading frames from Labels object.

    This IterDataPipe will produce examples containing a frame and an sleap_io.Instance
    from a sleap_io.Labels instance.

    Attributes:
        labels: sleap_io.Labels object that contains LabeledFrames that will be
            accessed through a torchdata DataPipe
        max_height: Maximum height the image should be padded to. If not provided, the original image size will be retained.
        max_width: Maximum width the image should be padded to. If not provided, the original image size will be retained.
        user_instances_only: True if filter labels only to user instances else False. Default value True
        is_rgb: True if the image has 3 channels (RGB image)
    """

    def __init__(
        self,
        labels: sio.Labels,
        max_height: int = None,
        max_width: int = None,
        user_instances_only: bool = True,
        is_rgb: bool = False,
    ):
        """Initialize labels attribute of the class."""
        self.labels = copy.deepcopy(labels)
        self.max_instances = get_max_instances(labels)
        self.max_width = max_width
        self.max_height = max_height
        self.is_rgb = is_rgb

        # Filter to user instances
        if user_instances_only:
            filtered_lfs = []
            for lf in self.labels:
                if lf.user_instances is not None and len(lf.user_instances) > 0:
                    lf.instances = lf.user_instances
                    filtered_lfs.append(lf)
            self.labels = sio.Labels(
                videos=self.labels.videos,
                skeletons=self.labels.skeletons,
                labeled_frames=filtered_lfs,
            )

    @classmethod
    def from_filename(
        cls,
        filename: str,
        user_instances_only: bool = True,
        max_height: int = None,
        max_width: int = None,
        is_rgb=False,
    ):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        return cls(labels, max_height, max_width, user_instances_only, is_rgb)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary containing the following elements.

        "image": A torch.Tensor containing full raw frame image as a uint8 array
            of shape (1, channels, height, width).
        "instances": Keypoint coordinates for all instances in the frame as a
            float32 torch.Tensor of shape (1, num_instances, num_nodes, 2).
        """
        for lf in self.labels:
            image = np.transpose(lf.image, (2, 0, 1))  # HWC -> CHW

            instances = []
            for inst in lf:
                if not np.all(np.isnan(inst.numpy())):
                    instances.append(inst.numpy())
            instances = np.stack(instances, axis=0)

            # Add singleton time dimension for single frames.
            image = np.expand_dims(image, axis=0)  # (1, C, H, W)
            instances = np.expand_dims(
                instances, axis=0
            )  # (1, num_instances, num_nodes, 2)

            instances = torch.from_numpy(instances.astype("float32"))
            num_instances, nodes = instances.shape[1:3]
            nans = torch.full(
                (1, np.abs(self.max_instances - num_instances), nodes, 2), torch.nan
            )
            instances = torch.cat([instances, nans], dim=1)
            img_height, img_width = image.shape[-2:]

            # pad images to max_height and max_width
            if self.max_height is not None:  # only if user provides
                pad_height = (self.max_height - img_height) // 2
                pad_width = (self.max_width - img_width) // 2
                image = np.pad(
                    image,
                    ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                    mode="constant",
                ).astype("float32")
                instances = instances + torch.Tensor([pad_height, pad_width])

            # convert to rgb
            if self.is_rgb and image.shape[-3] != 3:
                image = np.concatenate([image, image, image], axis=-3)

            yield {
                "image": torch.from_numpy(image),
                "instances": instances,
                "video_idx": torch.tensor(
                    self.labels.videos.index(lf.video), dtype=torch.int32
                ),
                "frame_idx": torch.tensor(lf.frame_idx, dtype=torch.int32),
                "num_instances": num_instances,
                "orig_size": torch.Tensor([img_height, img_width]),
            }
