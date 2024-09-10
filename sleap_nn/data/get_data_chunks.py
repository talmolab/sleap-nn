"""Handles generating data chunks for training."""

import numpy as np
import torch

from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.instance_cropping import generate_crops
from sleap_nn.data.normalization import apply_normalization, convert_to_grayscale, convert_to_rgb
from sleap_nn.data.providers import process_lf
from sleap_nn.data.resizing import apply_pad_to_stride, apply_resizer, apply_sizematcher

def bottomup_data_chunks(lf):

    image, instances_n, frame_idx, video_idx = lf

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])
    sample["image"] = convert_to_grayscale(sample["image"]) # TODO: how to set config params
    sample["image"] = convert_to_rgb(sample["image"])

    # size matcher
    sample["image"] = apply_sizematcher(sample["image"], max_height, max_width)

    # resize the image
    sample["image"], sample["instances"] = apply_resizer(sample["image"], sample["instances"],
                                                         scale=scale)
    
    # Pad the image (if needed) according max stride
    sample["image"] = apply_pad_to_stride(sample["image"], max_stride)

    return sample

def centered_data_chunks(lf):

    image, instances_n, frame_idx, video_idx = lf

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])
    sample["image"] = convert_to_grayscale(sample["image"]) # TODO: how to set config params
    sample["image"] = convert_to_rgb(sample["image"])

    # size matcher
    sample["image"] = apply_sizematcher(sample["image"], max_height, max_width)

    # resize the image
    sample["image"], sample["instances"] = apply_resizer(sample["image"], sample["instances"],
                                                         scale=scale)
    
    # Pad the image (if needed) according max stride
    sample["image"] = apply_pad_to_stride(sample["image"], max_stride)

    # get the centroids based on the anchor idx
    centroids = generate_centroids(sample["instances"], anchor_ind = anchor_ind)

    crop_size = np.array(crop_size) * np.sqrt(2) # crop extra
    crop_size = crop_size.astype(np.int32).tolist()

    for cnt, (instance, centroid) in enumerate(zip(sample["instances"], sample["centroids"])):
        if cnt==sample["num_instances"]:
            break

        res = generate_crops(sample["image"], instance, centroid, crop_size)

        res["frame_idx"] = sample["frame_idx"]
        res["video_idx"] = sample["video_idx"]
        res["num_instances"] = sample["num_instances"]
        res["orig_size"] = sample["orig_size"]

        yield res

def centroid_data_chunks(lf):

    image, instances_n, frame_idx, video_idx = lf

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])
    sample["image"] = convert_to_grayscale(sample["image"]) # TODO: how to set config params
    sample["image"] = convert_to_rgb(sample["image"])

    # size matcher
    sample["image"] = apply_sizematcher(sample["image"], max_height, max_width)

    # resize the image
    sample["image"], sample["instances"] = apply_resizer(sample["image"], sample["instances"],
                                                         scale=scale)
    
    # Pad the image (if needed) according max stride
    sample["image"] = apply_pad_to_stride(sample["image"], max_stride)

    # get the centroids based on the anchor idx
    centroids = generate_centroids(sample["instances"], anchor_ind = anchor_ind)

    sample["centroids"] = centroids

    return sample

def single_instance_data_chunks(lf):

    image, instances_n, frame_idx, video_idx = lf

    sample = process_lf(lf, video_idx, max_instances, user_instances_only)

    # Normalize image
    sample["image"] = apply_normalization(sample["image"])
    sample["image"] = convert_to_grayscale(sample["image"]) # TODO: how to set config params
    sample["image"] = convert_to_rgb(sample["image"])

    # size matcher
    sample["image"] = apply_sizematcher(sample["image"], max_height, max_width)

    # resize the image
    sample["image"], sample["instances"] = apply_resizer(sample["image"], sample["instances"],
                                                         scale=scale)
    
    # Pad the image (if needed) according max stride
    sample["image"] = apply_pad_to_stride(sample["image"], max_stride)

    return sample