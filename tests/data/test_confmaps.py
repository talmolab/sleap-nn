import torch
import sleap_io as sio
from sleap_nn.data.confidence_maps import (
    make_multi_confmaps,
    make_confmaps,
    generate_confmaps,
    generate_multiconfmaps,
)
from sleap_nn.data.providers import process_lf
from sleap_nn.data.utils import make_grid_vectors
import numpy as np


def test_generate_confmaps(minimal_instance):
    """Test `generate_confmaps` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    confmaps = generate_confmaps(
        ex["instances"][:, 0].unsqueeze(dim=1), img_hw=(384, 384)
    )
    assert confmaps.shape == (1, 2, 192, 192)


def test_generate_multiconfmaps(minimal_instance):
    """Test `generate_multiconfmaps` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    confmaps = generate_multiconfmaps(
        ex["instances"], img_hw=(384, 384), num_instances=ex["num_instances"]
    )
    assert confmaps.shape == (1, 2, 192, 192)

    confmaps = generate_multiconfmaps(
        ex["instances"][:, :, 0, :],
        img_hw=(384, 384),
        num_instances=ex["num_instances"],
        is_centroids=True,
    )
    assert confmaps.shape == (1, 1, 192, 192)
