"""This module implements pipeline blocks for reading input data such as labels."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sleap_io as sio
from queue import Queue
from threading import Thread
import torch
from copy import deepcopy
from loguru import logger


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
    return int(max(video.shape[1] for video in labels.videos)), int(
        max(video.shape[2] for video in labels.videos)
    )


def process_lf(
    instances_list: List[sio.Instance],
    img: np.ndarray,
    frame_idx: int,
    video_idx: int,
    max_instances: int,
    user_instances_only: bool = True,
) -> Dict[str, Any]:
    """Get sample dict from `sio.LabeledFrame`.

    Args:
        instances_list: List of `sio.Instance` objects.
        img: Input image.
        frame_idx: Frame index of the given lf.
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
        user_instances = [inst for inst in instances_list if type(inst) is sio.Instance]
        if len(user_instances) > 0:
            instances_list = user_instances

    image = np.transpose(img, (2, 0, 1))  # HWC -> CHW

    instances = []
    for inst in instances_list:
        if not inst.is_empty:
            instances.append(inst.numpy())
    if len(instances) == 0:
        return None
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
        "image": torch.from_numpy(image.copy()),
        "instances": instances,
        "video_idx": torch.tensor(video_idx, dtype=torch.int32),
        "frame_idx": torch.tensor(frame_idx, dtype=torch.int32),
        "orig_size": torch.Tensor([img_height, img_width]).unsqueeze(0),
        "num_instances": num_instances,
    }

    return ex


class VideoReader(Thread):
    """Thread module for reading frames from sleap-io Video object.

    This module will load the frames from video and pushes them as Tensors into a buffer
    queue as a dictionary with (image, frame index, video index, (height, width))
    which are then batched and consumed during the inference process.

    Attributes:
        video: sleap_io.Video object that contains images that will be
                accessed through a torchdata DataPipe.
        frame_buffer: Frame buffer queue.
        frames: List of frames indices. If `None`, all frames in the video are used.
    """

    def __init__(
        self,
        video: sio.Video,
        frame_buffer: Queue,
        frames: Optional[list] = None,
    ):
        """Initialize attribute of the class."""
        super().__init__()
        self.video = video
        self.frame_buffer = frame_buffer
        self.frames = frames
        self._daemonic = True  # needs to be set to True for graceful stop; all threads should be killed when main thread is killed
        if self.frames is None:
            self.frames = [x for x in range(0, len(self.video))]

        # Close the backend
        self.video.close()
        self.backend_status = self.video.open_backend
        self.video.open_backend = False

        # Make a thread-local copy
        self.local_video_copy = deepcopy(self.video)

        # Set it to open the backend on first read
        self.local_video_copy.open_backend = True

    def total_len(self):
        """Returns the total number of frames in the video."""
        return len(self.frames)

    @property
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` of frames in the video."""
        return self.video.shape[1], self.video.shape[2]

    @classmethod
    def from_filename(
        cls,
        filename: str,
        queue_maxsize: int,
        frames: Optional[list] = None,
        dataset: Optional[str] = None,
        input_format: str = "channels_last",
    ):
        """Create VideoReader from a .slp filename."""
        video = sio.load_video(filename, dataset=dataset, input_format=input_format)
        frame_buffer = Queue(maxsize=queue_maxsize)
        return cls(video, frame_buffer, frames)

    @classmethod
    def from_video(
        cls,
        video: sio.Video,
        queue_maxsize: int,
        frames: Optional[list] = None,
    ):
        """Create VideoReader from a video object."""
        frame_buffer = Queue(maxsize=queue_maxsize)
        return cls(video, frame_buffer, frames)

    def run(self):
        """Adds frames to the buffer queue."""
        try:
            for idx in self.frames:
                img = self.local_video_copy[idx]
                img = np.transpose(img, (2, 0, 1))  # convert H,W,C to C,H,W
                img = np.expand_dims(img, axis=0)  # (1, C, H, W)

                self.frame_buffer.put(
                    {
                        "image": torch.from_numpy(img.copy()),
                        "frame_idx": torch.tensor(idx, dtype=torch.int32),
                        "video_idx": torch.tensor(0, dtype=torch.int32),
                        "orig_size": torch.Tensor(img.shape[-2:]).unsqueeze(0),
                    }
                )

        except Exception as e:
            logger.error(f"Error when reading video frame. Stopping video reader.\n{e}")

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
        only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
        only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
    """

    def __init__(
        self,
        labels: sio.Labels,
        frame_buffer: Queue,
        instances_key: bool = False,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
    ):
        """Initialize attribute of the class."""
        super().__init__()
        self.labels = labels
        self.frame_buffer = frame_buffer
        self.instances_key = instances_key
        self.max_instances = get_max_instances(self.labels)

        self._daemonic = True  # needs to be set to True for graceful stop; all threads should be killed when main thread is killed

        self.only_labeled_frames = only_labeled_frames
        self.only_suggested_frames = only_suggested_frames
        self.exclude_user_labeled = exclude_user_labeled
        self.only_predicted_frames = only_predicted_frames

        # Filter to only user labeled instances
        if self.only_labeled_frames:
            self.filtered_lfs = []
            for lf in self.labels:
                if lf.user_instances is not None and len(lf.user_instances) > 0:
                    lf.instances = lf.user_instances
                    self.filtered_lfs.append(lf)

        # Filter to only unlabeled suggested instances
        elif self.only_suggested_frames:
            self.filtered_lfs = []
            for suggestion in self.labels.suggestions:
                lf = self.labels.find(suggestion.video, suggestion.frame_idx)
                if len(lf) == 0 or not lf[0].has_user_instances:
                    new_lf = sio.LabeledFrame(
                        video=suggestion.video, frame_idx=suggestion.frame_idx
                    )
                    self.filtered_lfs.append(new_lf)

        # Filter out user labeled frames
        elif self.exclude_user_labeled:
            self.filtered_lfs = []
            for lf in self.labels:
                if not lf.has_user_instances:
                    self.filtered_lfs.append(lf)

        # Filter to only predicted frames
        elif self.only_predicted_frames:
            self.filtered_lfs = []
            for lf in self.labels:
                if lf.has_predicted_instances:
                    self.filtered_lfs.append(lf)

        else:
            self.filtered_lfs = [lf for lf in self.labels]

        # Close the backend
        self.local_video_copy = []
        for video in self.labels.videos:
            video.close()
            self.backend_status = video.open_backend
            video.open_backend = False

            # make a thread-local copy
            self.local_video_copy.append(deepcopy(video))

            # Set it to open the backend on first read
            self.local_video_copy[-1].open_backend = True

    def total_len(self):
        """Returns the total number of frames in the video."""
        return len(self.filtered_lfs)

    @property
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` of frames in the video."""
        return max(video.shape[1] for video in self.labels.videos), max(
            video.shape[2] for video in self.labels.videos
        )

    @classmethod
    def from_filename(
        cls,
        filename: str,
        queue_maxsize: int,
        instances_key: bool = False,
        only_labeled_frames: bool = False,
        only_suggested_frames: bool = False,
        exclude_user_labeled: bool = False,
        only_predicted_frames: bool = False,
    ):
        """Create LabelsReader from a .slp filename."""
        labels = sio.load_slp(filename)
        frame_buffer = Queue(maxsize=queue_maxsize)
        return cls(
            labels,
            frame_buffer,
            instances_key,
            only_labeled_frames,
            only_suggested_frames,
            exclude_user_labeled,
            only_predicted_frames,
        )

    def run(self):
        """Adds frames to the buffer queue."""
        try:
            for lf in self.filtered_lfs:
                video_idx = self.labels.videos.index(lf.video)
                img = self.local_video_copy[video_idx][lf.frame_idx]
                img = np.transpose(img, (2, 0, 1))  # convert H,W,C to C,H,W
                img = np.expand_dims(img, axis=0)  # (1, C, H, W)

                sample = {
                    "image": torch.from_numpy(img.copy()),
                    "frame_idx": torch.tensor(lf.frame_idx, dtype=torch.int32),
                    "video_idx": torch.tensor(video_idx, dtype=torch.int32),
                    "orig_size": torch.Tensor(img.shape[-2:]).unsqueeze(0),
                }

                if self.instances_key:
                    instances = []
                    for inst in lf:
                        if not inst.is_empty:
                            instances.append(inst.numpy())
                    if len(instances) == 0:
                        continue
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
            logger.error(
                f"Error when reading labelled frame. Stopping labels reader.\n{e}"
            )

        finally:
            self.frame_buffer.put(
                {
                    "image": None,
                    "frame_idx": None,
                    "video_idx": None,
                    "orig_size": None,
                }
            )
