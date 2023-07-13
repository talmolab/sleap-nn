"""Handle importing of sleap data."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl

import sleap_io as sio


class LabelsReader(dp.IterDataPipe):
    """Datapipe for reading frames from Labels object."""

    def __init__(self, labels: sio.Labels):
        """Initialize LabelsReader.

        Args:
            labels: sleap_io.Labels object that contains LabeledFrames that will be
                accessed through a torchdata DataPipe
        """
        self.labels = labels

    @classmethod
    def from_filename(cls, filename: str):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        return cls(labels)

    def __iter__(self):
        """Return labeledframe sample."""
        for lf in self.labels:
            yield lf
