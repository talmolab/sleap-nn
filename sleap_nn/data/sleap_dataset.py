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
    ):
        """Initialize SleapDataset.

        Args:
            labels: sleap_io.Labels object that contains LabeledFrames
        """
        self.labels = labels
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
        """Return a single item corresponding to a single instance in a single frame."""
        lf_idx, inst_idx = self.data[index]
        lf = self.labels[lf_idx]
        inst = lf[inst_idx]
        img = lf.image  # must be permuted

        return lf, inst, img
