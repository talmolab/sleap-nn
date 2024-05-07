"""General purpose transformers for common pipeline processing tasks."""

from typing import Dict, Iterator, List, Text

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe


class KeyFilter(IterDataPipe):
    """Transformer for filtering example keys."""

    def __init__(self, source_dp: IterDataPipe, keep_keys: List[Text] = None) -> None:
        """Initialize KeyFilter with the source `DataPipe."""
        self.dp = source_dp
        self.keep_keys = set(keep_keys) if keep_keys else None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return a dictionary filtered for the relevant outputs.

        The input dictionary includes:
        - image: the full frame image
        - video_idx: the index of the source video in the list of videos
        - frame_idx: the frame idx of the image in video[`video_idx`]
        - instance: the individual instance's keypoints
        - instance_bbox: the individual instance's bbox
        - instance_image: the individual instance's cropped image
        - confidence_maps: the individual instance's heatmap
        - orig_size: Original Image size
        - num_instances: Number of instances in the frame
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
