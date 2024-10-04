"""Handles generating data chunks for training."""

from typing import Dict, Iterator, Optional, Tuple
from omegaconf import DictConfig
import numpy as np
import torch

import sleap_io as sio
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import generate_crops
from sleap_nn.data.normalization import (
    apply_normalization,
    convert_to_grayscale,
    convert_to_rgb,
)
from sleap_nn.data.providers import process_lf
from sleap_nn.data.resizing import apply_sizematcher


def bottomup_data_chunks(
    x: Tuple[sio.LabeledFrame, int],
    data_config: DictConfig,
    max_instances: int,
    max_hw: Tuple[int, int],
    user_instances_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate dict from `sio.LabeledFrame`.

    This function processes the input `sio.LabeledFrame`, applies data pre-processing
    operations (except augmentation and confmaps generation). This function is passed
    to `litdata.optimize()` which applies this function on all the `sio.LabeledFrame`s
    in the training `.slp` file and saves these dictionaries as `.bin` files.

    Args:
        x: Tuple (lf, video_idx) where lf is a `sio.LabeledFrame` and video_idx is the
            index of lf.video in the source sio.labels.videos.
        data_config: Data-related configuration. (`data_config` section in the config file)
        max_instances: Maximum number of instances that could occur in a single LabeledFrame.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        user_instances_only: True if filter labels only to user instances else False.
            Default: True.

    Returns:
        Dict with image, instances, frame index, video index, original image size and
        number of instances.

    """
    lf, video_idx = x

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])

    if data_config.preprocessing.is_rgb:
        sample["image"] = convert_to_rgb(sample["image"])
    else:
        sample["image"] = convert_to_grayscale(sample["image"])

    # size matcher
    max_height, max_width = (
        data_config.preprocessing.max_height,
        data_config.preprocessing.max_width,
    )
    sample["image"] = apply_sizematcher(
        sample["image"],
        max_height=max_height if max_height is not None else max_hw[0],
        max_width=max_width if max_width is not None else max_hw[1],
    )

    return sample


def centered_instance_data_chunks(
    x: Tuple[sio.LabeledFrame, int],
    data_config: DictConfig,
    max_instances: int,
    crop_size: Tuple[int],
    anchor_ind: Optional[int],
    max_hw: Tuple[int, int],
    user_instances_only: bool = True,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Generate dict from `sio.LabeledFrame`.

    This function processes the input `sio.LabeledFrame`, applies data pre-processing
    operations (except augmentation and confmaps generation). This function is passed
    to `litdata.optimize()` which applies this function on all the `sio.LabeledFrame`s
    in the training `.slp` file and saves these dictionaries as `.bin` files.

    Args:
        x: Tuple (lf, video_idx) where lf is a `sio.LabeledFrame` and video_idx is the
            index of lf.video in the source sio.labels.videos.
        data_config: Data-related configuration. (`data_config` section in the config file)
        max_instances: Maximum number of instances that could occur in a single LabeledFrame.
        crop_size: Height and width of the crop in pixels.
        anchor_ind: The index of the node to use as the anchor for the centroid. If not
            provided or if not present in the instance, the midpoint of the bounding box
            is used instead.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        user_instances_only: True if filter labels only to user instances else False.
            Default: True.

    Returns:
        Dict with image, instances, frame index, video index, original image size and
        number of instances.

    """
    lf, video_idx = x

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])

    if data_config.preprocessing.is_rgb:
        sample["image"] = convert_to_rgb(sample["image"])
    else:
        sample["image"] = convert_to_grayscale(sample["image"])

    # size matcher
    max_height, max_width = (
        data_config.preprocessing.max_height,
        data_config.preprocessing.max_width,
    )
    sample["image"] = apply_sizematcher(
        sample["image"],
        max_height=max_height if max_height is not None else max_hw[0],
        max_width=max_width if max_width is not None else max_hw[1],
    )

    # get the centroids based on the anchor idx
    centroids = generate_centroids(sample["instances"], anchor_ind=anchor_ind)

    crop_size = np.array(crop_size) * np.sqrt(2)  # crop extra
    crop_size = crop_size.astype(np.int32).tolist()

    sample["instances"], centroids = sample["instances"][0], centroids[0]  # n_samples=1

    for cnt, (instance, centroid) in enumerate(zip(sample["instances"], centroids)):
        if cnt == sample["num_instances"]:
            break

        res = generate_crops(sample["image"], instance, centroid, crop_size)

        res["frame_idx"] = sample["frame_idx"]
        res["video_idx"] = sample["video_idx"]
        res["num_instances"] = sample["num_instances"]
        res["orig_size"] = sample["orig_size"]

        yield res


def centroid_data_chunks(
    x: Tuple[sio.LabeledFrame, int],
    data_config: DictConfig,
    max_instances: int,
    anchor_ind: Optional[int],
    max_hw: Tuple[int, int],
    user_instances_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate dict from `sio.LabeledFrame`.

    This function processes the input `sio.LabeledFrame`, applies data pre-processing
    operations (except augmentation and confmaps generation). This function is passed
    to `litdata.optimize()` which applies this function on all the `sio.LabeledFrame`s
    in the training `.slp` file and saves these dictionaries as `.bin` files.

    Args:
        x: Tuple (lf, video_idx) where lf is a `sio.LabeledFrame` and video_idx is the
            index of lf.video in the source sio.labels.videos.
        data_config: Data-related configuration. (`data_config` section in the config file)
        max_instances: Maximum number of instances that could occur in a single LabeledFrame.
        anchor_ind: The index of the node to use as the anchor for the centroid. If not
            provided or if not present in the instance, the midpoint of the bounding box
            is used instead.
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        user_instances_only: True if filter labels only to user instances else False.
            Default: True.

    Returns:
        Dict with image, instances, frame index, video index, original image size and
        number of instances.

    """
    lf, video_idx = x

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])

    if data_config.preprocessing.is_rgb:
        sample["image"] = convert_to_rgb(sample["image"])
    else:
        sample["image"] = convert_to_grayscale(sample["image"])

    # size matcher
    max_height, max_width = (
        data_config.preprocessing.max_height,
        data_config.preprocessing.max_width,
    )
    sample["image"] = apply_sizematcher(
        sample["image"],
        max_height=max_height if max_height is not None else max_hw[0],
        max_width=max_width if max_width is not None else max_hw[1],
    )

    # get the centroids based on the anchor idx
    centroids = generate_centroids(sample["instances"], anchor_ind=anchor_ind)

    sample["centroids"] = centroids

    return sample


def single_instance_data_chunks(
    x: Tuple[sio.LabeledFrame, int],
    data_config: DictConfig,
    max_hw: Tuple[int, int],
    user_instances_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate dict from `sio.LabeledFrame`.

    This function processes the input `sio.LabeledFrame`, applies data pre-processing
    operations (except augmentation and confmaps generation). This function is passed
    to `litdata.optimize()` which applies this function on all the `sio.LabeledFrame`s
    in the training `.slp` file and saves these dictionaries as `.bin` files.

    Args:
        x: Tuple (lf, video_idx) where lf is a `sio.LabeledFrame` and video_idx is the
            index of lf.video in the source sio.labels.videos.
        data_config: Data-related configuration. (`data_config` section in the config file).
        max_hw: Maximum height and width of images across the labels file. If `max_height` and
           `max_width` in the config is None, then `max_hw` is used (computed with
            `sleap_nn.data.providers.get_max_height_width`). Else the values in the config
            are used.
        user_instances_only: True if filter labels only to user instances else False.
            Default: True.

    Returns:
        Dict with image, instances, frame index, video index, original image size and
        number of instances.

    """
    lf, video_idx = x

    sample = process_lf(
        lf, video_idx, user_instances_only=user_instances_only, max_instances=1
    )

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])

    if data_config.preprocessing.is_rgb:
        sample["image"] = convert_to_rgb(sample["image"])
    else:
        sample["image"] = convert_to_grayscale(sample["image"])

    # size matcher
    max_height, max_width = (
        data_config.preprocessing.max_height,
        data_config.preprocessing.max_width,
    )
    sample["image"] = apply_sizematcher(
        sample["image"],
        max_height=max_height if max_height is not None else max_hw[0],
        max_width=max_width if max_width is not None else max_hw[1],
    )

    return sample
