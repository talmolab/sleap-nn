"""This module implements pipeline blocks for reading input data such as labels."""

from typing import Dict, Iterator, Optional

import numpy as np
import sleap_io as sio
from queue import Queue
from threading import Thread
import torch
import copy
from torch.utils.data.datapipes.datapipe import IterDataPipe


def get_max_instances(labels: sio.Labels):
    """Get the maximum number of instances in a single LabeledFrame.

    Args:
        labels: sleap_io.Labels object that contains LabeledFrames.

    Returns:
        Maximum number of instances that could occur in a single LabeledFrame.
    """
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
                accessed through a torchdata DataPipe.
        user_instances_only: True if filter labels only to user instances else False.
                Default value True
        instances_key: True if `instances` key needs to be present in the data pipeline.
                When this is set to True, the instances are appended with NaNs to have same
                number of instances to enable batching. This is useful when running
                inference.inference.FindInstancePeaksGroundTruth where we need the
                `instances` key. Default: False.
    """

    def __init__(
        self,
        labels: sio.Labels,
        user_instances_only: bool = True,
        instances_key: bool = False,
    ):
        """Initialize labels attribute of the class."""
        self.labels = copy.deepcopy(labels)
        self.max_instances = get_max_instances(labels)
        self.instances_key = instances_key

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
        instances_key: bool = False,
    ):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        return cls(labels, user_instances_only, instances_key)

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
                if not inst.is_empty:
                    instances.append(inst.numpy())
            instances = np.stack(instances, axis=0)

            # Add singleton time dimension for single frames.
            image = np.expand_dims(image, axis=0)  # (1, C, H, W)
            instances = np.expand_dims(
                instances, axis=0
            )  # (1, num_instances, num_nodes, 2)

            instances = torch.from_numpy(instances.astype("float32"))
            num_instances, nodes = instances.shape[1:3]

            if self.instances_key:
                nans = torch.full(
                    (1, np.abs(self.max_instances - num_instances), nodes, 2), torch.nan
                )
                instances = torch.cat([instances, nans], dim=1)

            yield {
                "image": torch.from_numpy(image),
                "instances": instances,
                "video_idx": torch.tensor(
                    self.labels.videos.index(lf.video), dtype=torch.int32
                ),
                "frame_idx": torch.tensor(lf.frame_idx, dtype=torch.int32),
                "num_instances": num_instances,
            }


class VideoReader(Thread):
    """Thread module for reading frames from sleap-io Video object.

    This module will load the frames from video and pushes them as Tensors into a buffer
    queue as a tuple in the format (image, frame index, (height, width)) which are then
    batched and consumed during the inference process.

    Attributes:
        video: sleap_io.Video object that contains LabeledFrames that will be
                accessed through a torchdata DataPipe.
        frame_buffer: Maximum height the image should be padded to. If not provided,
                the original image size will be retained.
        start_idx: start index of the frames to read. If None, 0 is set as the default.
        end_idx: end index of the frames to read. If None, length of the video is set as
                the default.
    """

    def __init__(
        self,
        video: sio.Video,
        frame_buffer: Queue,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """Initialize attribute of the class."""
        super().__init__()
        self.video = video
        self.frame_buffer = frame_buffer
        self.start_idx = start_idx
        self.end_idx = end_idx
        if self.start_idx is None:
            self.start_idx = 0
        if self.end_idx is None:
            self.end_idx = self.video.shape[0]

    def total_len(self):
        """Returns the total number of frames in the video."""
        return self.end_idx - self.start_idx

    @classmethod
    def from_filename(
        cls,
        filename: str,
        frame_buffer: Queue,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """Create LabelsReader from a .slp filename."""
        video = sio.load_video(filename)
        return cls(video, frame_buffer, start_idx, end_idx)

    def run(self):
        """Adds frames to the buffer queue."""
        try:
            for idx in range(self.start_idx, self.end_idx):
                img = self.video[idx]
                img = np.transpose(img, (2, 0, 1))  # convert H,W,C to C,H,W
                img = np.expand_dims(img, axis=0)  # (1, C, H, W)

                self.frame_buffer.put(
                    (
                        torch.from_numpy(img),
                        torch.tensor(idx, dtype=torch.int32),
                        torch.Tensor(img.shape[-2:]),
                    )
                )

        except Exception as e:
            print(f"Error when reading video frame. Stopping video reader.\n{e}")

        finally:
            self.frame_buffer.put((None, None, None))
