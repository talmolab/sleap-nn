"""Handle distribution of sleap data."""
from typing import Tuple, Union, Optional

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

import sleap_io as sio


class SleapDataset(Dataset):
    """Dataset for loading items from Labels object."""

    def __init__(
        self,
        labels: sio.Labels,
        output_stride: int = 2,
        sigma: int = 2.5,
    ):
        """Initialize SleapDataset.

        Args:
            labels: sleap_io.Labels object that contains LabeledFrames
            output_stride: Relative stride of the generated confidence maps. This is
                effectively the reciprocal of the output scale, i.e., increase this to
                generate confidence maps that are smaller than the input images.
            sigma: Standard deviation of the 2D Gaussian distribution sampled to
                generate confidence maps. This defines the spread in units of the input
                image’s grid, i.e., it does not take scaling in previous steps into account.
        """
        self.labels = labels
        self.sigma = sigma
        self.output_stride = output_stride
        self.data = []

        # Store data in self.data -> format [(LabeledFrame index, Instance index),
        #     (LabeledFrame index, Next Instance index)...]

        for lf_idx, lf in enumerate(labels):
            inst_amt = len(lf.instances)
            self.data.extend([(lf_idx, inst_idx) for inst_idx in range(inst_amt)])

        # Define object augmentations

    def __len__(self):
        """Return the number of instances available."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a single item."""
        lf_idx, inst_idx = self.data[index]
        lf = self.labels[lf_idx]
        inst = lf[inst_idx]
        img = lf.image  # must be permuted

        return lf, inst, img
