"""Handle importing of sleap data."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl

import sleap_io as sio


class LabelsReader(dp.IterDataPipe):
    """Datapipe for reading frames from Labels object."""

    def __init__(self, labels: sio.Labels):
        self.labels = labels

    @classmethod
    def from_filename(cls, filename: str):
        labels = sio.load_slp(filename)
        return cls(labels)

    def __iter__(self):
        for lf in self.labels:
            yield lf
