"""Handle importing of sleap data."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl
import torch
import sleap_io as sio


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
        """Return a sample containing the following elements.

        - a torch.Tensor representing an instance
        - a torch.Tensor representing the corresponding image
        """
        for lf in self.labels:
            for inst in lf:
                instance = torch.from_numpy(inst.numpy())
                image = torch.from_numpy(lf.image)
                # kornia takes input with shape (batch, channels, height, width)
                image = image.permute(2, 0, 1)
                yield {"image": image, "instance": instance}
