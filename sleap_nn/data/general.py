"""General purpose transformers for common pipeline processing tasks."""
from typing import Callable, Dict, List, Text

from torch.utils.data.datapipes.datapipe import IterDataPipe


class KeyFilter(IterDataPipe):
    """Transformer for filtering example keys."""

    def __init__(self, source_dp: IterDataPipe, keep_keys: List[Text] = None) -> None:
        """Initialize KeyFilter with the source `DataPipe."""
        self.dp = source_dp
        self.keep_keys = keep_keys

    def __iter__(self):
        """Return a dictionary filtered for the relevant outputs.

        The input dictionary includes:
        - image: the full frame image
        - instances: all keypoints of all instances in the frame image
        - centroids: all centroids of all instances in the frame image
        - instance: the individual instance's keypoints
        - instance_bbox: the individual instance's bbox
        - instance_image: the individual instance's cropped image
        - confidence_maps: the individual instance's heatmap
        """
        for example in self.dp:
            if self.keep_keys is None:
                # If keep_keys is not provided, yield the entire example.
                yield example
            else:
                # Filter the example dictionary based on keep_keys.
                filtered_example = {
                    key: value
                    for key, value in example.items()
                    if key in self.keep_keys
                }
                yield filtered_example