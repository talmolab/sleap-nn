"""This module implements pipeline blocks for reading input data such as labels."""
from typing import Dict, Iterator

import numpy as np
import sleap_io as sio
import torch
import copy
from torch.utils.data.datapipes.datapipe import IterDataPipe


class LabelsReader(IterDataPipe):
    """Datapipe for reading frames from Labels object.

    This DataPipe will produce examples containing a frame and an sleap_io.Instance
    from a sleap_io.Labels instance.

    Attributes:
        labels: sleap_io.Labels object that contains LabeledFrames that will be
            accessed through a torchdata DataPipe
        user_instances_only: True if filter labels only to user instances else False. Default value True
    """

    def __init__(self, labels: sio.Labels, user_instances_only: bool = True):
        """Initialize labels attribute of the class."""
        self.labels = copy.deepcopy(labels)

        # Filter to user instances
        if user_instances_only:
            filtered_lfs = []
            for lf in self.labels:
                if lf.user_instances is not None and len(lf.user_instances) > 0:
                    lf.instances = lf.user_instances
                    filtered_lfs.append(lf)
            self.labels = sio.Labels(
                videos=labels.videos,
                skeletons=[labels.skeleton],
                labeled_frames=filtered_lfs,
            )

    @classmethod
    def from_filename(cls, filename: str, user_instances_only: bool = True):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        return cls(labels, user_instances_only)

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
                instances.append(inst.numpy())
            instances = np.stack(instances, axis=0)

            # Add singleton time dimension for single frames.
            image = np.expand_dims(image, axis=0)  # (1, C, H, W)
            instances = np.expand_dims(
                instances, axis=0
            )  # (1, num_instances, num_nodes, 2)

            yield {
                "image": torch.from_numpy(image),
                "instances": torch.from_numpy(instances.astype("float32")),
            }
