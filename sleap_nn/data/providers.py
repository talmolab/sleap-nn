"""Handle importing of sleap data."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl
import torch
import sleap_io as sio
import numpy as np


class LabelsReader(dp.IterDataPipe):
    """Datapipe for reading frames from Labels object.

    This DataPipe will produce examples containing a frame and an sleap_io.Instance
    from a sleap_io.Labels instance.

    Attributes:
        labels: sleap_io.Labels object that contains LabeledFrames that will be
            accessed through a torchdata DataPipe
    """

    def __init__(self, labels: sio.Labels):
        """Initialize labels attribute of the class."""
        self.labels = labels

    @classmethod
    def from_filename(cls, filename: str):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        return cls(labels)

    def __iter__(self):
        """Return an example dictionary containing the following elements:

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
                "instances": torch.from_numpy(instances),
            }
