import torch
import sleap_io as sio
from sleap_nn.data.instance_centroids import (
    generate_centroids,
)
from sleap_nn.data.providers import process_lf


def test_generate_centroids(minimal_instance):
    """Test `generate_centroids` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    centroids = generate_centroids(ex["instances"], 1).int()
    gt = torch.Tensor([[[152, 158], [278, 203]]]).int()
    assert torch.equal(centroids, gt)

    partial_instance = torch.Tensor(
        [
            [
                [[92.6522, 202.7260], [152.3419, 158.4236], [97.2618, 53.5834]],
                [[205.9301, 187.8896], [torch.nan, torch.nan], [201.4264, 75.2373]],
                [
                    [torch.nan, torch.nan],
                    [torch.nan, torch.nan],
                    [torch.nan, torch.nan],
                ],
            ]
        ]
    )
    centroids = generate_centroids(partial_instance, 1).int()
    gt = torch.Tensor([[[152, 158], [203, 131], [torch.nan, torch.nan]]]).int()
    assert torch.equal(centroids, gt)
