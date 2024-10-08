"""This module implements pipeline blocks for reading input data such as labels."""

from typing import Any, Dict, Iterator, Optional, Tuple

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


def get_max_height_width(labels: sio.Labels) -> Tuple[int, int]:
    """Return `(height, width)` that is the maximum of all videos."""
    return max(video.shape[1] for video in labels.videos), max(
        video.shape[2] for video in labels.videos
    )


def process_lf(
    lf: sio.LabeledFrame,
    video_idx: int,
    max_instances: int,
    user_instances_only: bool = True,
) -> Dict[str, Any]:
    """Get sample dict from `sio.LabeledFrame`.

    Args:
        lf: Input `sio.LabeledFrame`.
        video_idx: Video index of the given lf.
        max_instances: Maximum number of instances that could occur in a single LabeledFrame.
        user_instances_only: True if filter labels only to user instances else False.
            Default: True.

    Returns:
        Dict with image, instancs, frame index, video index, original image size and
        number of instances.

    """
    # Filter to user instances
    if user_instances_only:
        if lf.user_instances is not None and len(lf.user_instances) > 0:
            lf.instances = lf.user_instances

    image = np.transpose(lf.image, (2, 0, 1))  # HWC -> CHW

    instances = []
    for inst in lf:
        if not inst.is_empty:
            instances.append(inst.numpy())
    instances = np.stack(instances, axis=0)

    # Add singleton time dimension for single frames.
    image = np.expand_dims(image, axis=0)  # (n_samples=1, C, H, W)
    instances = np.expand_dims(
        instances, axis=0
    )  # (n_samples=1, num_instances, num_nodes, 2)

    instances = torch.from_numpy(instances.astype("float32"))

    num_instances, nodes = instances.shape[1:3]
    img_height, img_width = image.shape[-2:]

    # append with nans for broadcasting
    if max_instances != 1:
        nans = torch.full(
            (1, np.abs(max_instances - num_instances), nodes, 2), torch.nan
        )
        instances = torch.cat(
            [instances, nans], dim=1
        )  # (n_samples, max_instances, num_nodes, 2)

    ex = {
        "image": torch.from_numpy(image),
        "instances": instances,
        "video_idx": torch.tensor(video_idx, dtype=torch.int32),
        "frame_idx": torch.tensor(lf.frame_idx, dtype=torch.int32),
        "orig_size": torch.Tensor([img_height, img_width]),
        "num_instances": num_instances,
    }

    return ex


class LabelsReaderDP(IterDataPipe):
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
                number of instances to enable batching. Default: False.
    """

    def __init__(
        self,
        labels: sio.Labels,
        user_instances_only: bool = True,
        instances_key: bool = True,
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

    @property
    def edge_inds(self) -> list:
        """Returns list of edge indices."""
        return self.labels.skeletons[0].edge_inds

    @property
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` that is the maximum of all videos."""
        return max(video.shape[1] for video in self.labels.videos), max(
            video.shape[2] for video in self.labels.videos
        )

    @classmethod
    def from_filename(
        cls,
        filename: str,
        user_instances_only: bool = True,
        instances_key: bool = True,
    ):
        """Create LabelsReaderDP from a .slp filename."""
        labels = sio.load_slp(filename)
        return cls(labels, user_instances_only, instances_key)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary containing the following elements.

        "image": A torch.Tensor containing full raw frame image as a uint8 array
            of shape (n_samples, channels, height, width).
        "instances": Keypoint coordinates for all instances in the frame as a
            float32 torch.Tensor of shape (n_samples, n_instances, n_nodes, 2).
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
            img_height, img_width = image.shape[-2:]
            instances = np.expand_dims(
                instances, axis=0
            )  # (1, num_instances, num_nodes, 2)

            instances = torch.from_numpy(instances.astype("float32"))
            num_instances, nodes = instances.shape[1:3]
            ex = {
                "image": torch.from_numpy(image),
                "video_idx": torch.tensor(
                    self.labels.videos.index(lf.video), dtype=torch.int32
                ),
                "frame_idx": torch.tensor(lf.frame_idx, dtype=torch.int32),
                "num_instances": num_instances,
            }
            ex["orig_size"] = torch.Tensor([img_height, img_width])

            if self.instances_key:
                nans = torch.full(
                    (1, np.abs(self.max_instances - num_instances), nodes, 2), torch.nan
                )
                ex["instances"] = torch.cat([instances, nans], dim=1)

            yield ex


class VideoReader(Thread):
    """Thread module for reading frames from sleap-io Video object.

    This module will load the frames from video and pushes them as Tensors into a buffer
    queue as a dictionary with (image, frame index, video index, (height, width))
    which are then batched and consumed during the inference process.

    Attributes:
        video: sleap_io.Video object that contains images that will be
                accessed through a torchdata DataPipe.
        frame_buffer: Frame buffer queue.
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

    @property
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` of frames in the video."""
        return self.video.shape[1], self.video.shape[2]

    @classmethod
    def from_filename(
        cls,
        filename: str,
        queue_maxsize: int,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """Create VideoReader from a .slp filename."""
        video = sio.load_video(filename)
        frame_buffer = Queue(maxsize=queue_maxsize)
        return cls(video, frame_buffer, start_idx, end_idx)

    def run(self):
        """Adds frames to the buffer queue."""
        try:
            for idx in range(self.start_idx, self.end_idx):
                img = self.video[idx]
                img = np.transpose(img, (2, 0, 1))  # convert H,W,C to C,H,W
                img = np.expand_dims(img, axis=0)  # (1, C, H, W)

                self.frame_buffer.put(
                    {
                        "image": torch.from_numpy(img),
                        "frame_idx": torch.tensor(idx, dtype=torch.int32),
                        "video_idx": torch.tensor(0, dtype=torch.int32),
                        "orig_size": torch.Tensor(img.shape[-2:]),
                    }
                )

        except Exception as e:
            print(f"Error when reading video frame. Stopping video reader.\n{e}")

        finally:
            self.frame_buffer.put(
                {
                    "image": None,
                    "frame_idx": None,
                    "video_idx": None,
                    "orig_size": None,
                }
            )


class LabelsReader(Thread):
    """Thread module for reading images from sleap-io Labels object.

    This module will load the images from `.slp` files and pushes them as Tensors into a
    buffer queue as a dictionary with (image, frame index, video index, (height, width))
    which are then batched and consumed during the inference process.

    Attributes:
        labels: sleap_io.Labels object that contains LabeledFrames that will be
                accessed through a torchdata DataPipe.
        frame_buffer: Frame buffer queue.
        instances_key: If `True`, then instances are appended to the output dictionary.
    """

    def __init__(
        self, labels: sio.Labels, frame_buffer: Queue, instances_key: bool = False
    ):
        """Initialize attribute of the class."""
        super().__init__()
        self.labels = labels
        self.frame_buffer = frame_buffer
        self.instances_key = instances_key
        self.max_instances = get_max_instances(self.labels)

    def total_len(self):
        """Returns the total number of frames in the video."""
        return len(self.labels)

    @property
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` of frames in the video."""
        return max(video.shape[1] for video in self.labels.videos), max(
            video.shape[2] for video in self.labels.videos
        )

    @classmethod
    def from_filename(
        cls, filename: str, queue_maxsize: int, instances_key: bool = False
    ):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        frame_buffer = Queue(maxsize=queue_maxsize)
        return cls(labels, frame_buffer, instances_key)

    def run(self):
        """Adds frames to the buffer queue."""
        try:
            for idx in range(self.total_len()):
                lf = self.labels[idx]
                img = lf.image
                img = np.transpose(img, (2, 0, 1))  # convert H,W,C to C,H,W
                img = np.expand_dims(img, axis=0)  # (1, C, H, W)

                sample = {
                    "image": torch.from_numpy(img),
                    "frame_idx": torch.tensor(lf.frame_idx, dtype=torch.int32),
                    "video_idx": torch.tensor(
                        self.labels.videos.index(lf.video), dtype=torch.int32
                    ),
                    "orig_size": torch.Tensor(img.shape[-2:]),
                }

                if self.instances_key:
                    instances = []
                    for inst in lf:
                        if not inst.is_empty:
                            instances.append(inst.numpy())
                    instances = np.stack(instances, axis=0)

                    # Add singleton time dimension for single frames.
                    instances = np.expand_dims(
                        instances, axis=0
                    )  # (n_samples=1, num_instances, num_nodes, 2)

                    instances = torch.from_numpy(instances.astype("float32"))

                    num_instances, nodes = instances.shape[1:3]

                    # append with nans for broadcasting
                    if self.max_instances != 1:
                        nans = torch.full(
                            (1, np.abs(self.max_instances - num_instances), nodes, 2),
                            torch.nan,
                        )
                        instances = torch.cat(
                            [instances, nans], dim=1
                        )  # (n_samples, max_instances, num_nodes, 2)

                    sample["instances"] = instances

                self.frame_buffer.put(sample)

        except Exception as e:
            print(f"Error when reading labelled frame. Stopping labels reader.\n{e}")

        finally:
            self.frame_buffer.put(
                {
                    "image": None,
                    "frame_idx": None,
                    "video_idx": None,
                    "orig_size": None,
                }
            )
