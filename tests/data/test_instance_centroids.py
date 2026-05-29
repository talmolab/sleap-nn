import torch
import sleap_io as sio
from sleap_nn.data.instance_centroids import (
    find_points_bbox_midpoint,
    find_points_mean,
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


def test_generate_centroids_anchor_none_uses_mean_of_visible_nodes():
    """`anchor_ind=None` falls back to mean of visible nodes (not bbox midpoint).

    The two diverge on skewed instances (e.g., long tails / sprawled limbs).
    """
    # One node at (10, 10) skews bbox-midpoint vs mean.
    points = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]]],  # (1 instance, 3 nodes, 2)
    )

    centroids = generate_centroids(points, anchor_ind=None)
    expected_mean = torch.tensor([[10.0 / 3, 10.0 / 3]])
    torch.testing.assert_close(centroids, expected_mean)

    # And distinct from what bbox-midpoint would return.
    bbox_mid = find_points_bbox_midpoint(points)
    assert not torch.allclose(centroids, bbox_mid)


def test_generate_centroids_missing_anchor_node_fallback():
    """A NaN (occluded) anchor node falls back to the mean of visible nodes.

    Pins the post-#530 anchor-fallback convention (mean of visible nodes, not
    the bbox midpoint) on the per-instance missing-anchor path — distinct from
    the ``anchor_ind=None`` path. Uses a skewed instance (one far node) so the
    mean and the bbox midpoint differ, locking the intended behavior (#582).
    This is a shared module (training + GT-centroid inference) kept consistent
    between the two; the bbox-midpoint vs mean-of-visible choice is tracked in
    #586 for a later revisit.
    """
    # 4 nodes; anchor (index 0) is NaN. Visible nodes (0,0),(0,0),(12,12).
    points = torch.tensor(
        [[[float("nan"), float("nan")], [0.0, 0.0], [0.0, 0.0], [12.0, 12.0]]]
    )
    centroids = generate_centroids(points, anchor_ind=0)
    # mean of visible = (4, 4); bbox midpoint would be (6, 6).
    torch.testing.assert_close(centroids, torch.tensor([[4.0, 4.0]]))
    bbox_mid = find_points_bbox_midpoint(points)
    assert not torch.allclose(centroids, bbox_mid)


def test_find_points_mean_ignores_nan():
    """`find_points_mean` excludes NaN nodes; all-NaN → NaN."""
    points = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [torch.nan, torch.nan]],
            [[torch.nan, torch.nan], [torch.nan, torch.nan], [torch.nan, torch.nan]],
        ]
    )
    means = find_points_mean(points)
    assert torch.allclose(means[0], torch.tensor([2.0, 3.0]))
    assert torch.isnan(means[1]).all()
