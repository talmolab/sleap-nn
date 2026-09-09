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


# ─────────────────────────────────────────────────────────────────────────
# #586 — centroid methods, and object/tensor-level parity
# ─────────────────────────────────────────────────────────────────────────


import numpy as np
import pytest

from sleap_nn.data.instance_centroids import (
    CENTROID_METHODS,
    REDUCE_METHODS,
    add_centroids_from_masks,
    centroid_method_from_config,
    degrade_anchor_if_unresolved,
    find_points_geometric_median,
    reduce_points,
    resolve_centroid_method,
)


def _rand_instances(n=24, n_nodes=13, seed=0, missing=True):
    """Random instances with all-or-nothing NaN nodes, as sleap-io writes them."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(300.0, 60.0, size=(n, n_nodes, 2))
    if missing:
        for i in range(n):
            k = rng.integers(0, n_nodes - 1)
            if k:
                pts[i, rng.choice(n_nodes, size=k, replace=False)] = np.nan
    return pts


def test_default_is_byte_identical_to_the_pre_586_behavior():
    """`method=None` must reproduce exactly what every existing caller got."""
    pts = torch.from_numpy(_rand_instances()).to(torch.float32)
    # No anchor -> mean of visible nodes.
    assert torch.equal(generate_centroids(pts), find_points_mean(pts))
    # Anchor -> that node, falling back to the mean when it is missing.
    with_anchor = generate_centroids(pts, anchor_ind=3)
    expected = pts[..., 3, :].clone()
    missing = torch.isnan(expected).any(dim=-1)
    expected[missing] = find_points_mean(pts[missing])
    assert torch.equal(with_anchor, expected)


@pytest.mark.parametrize("method", REDUCE_METHODS)
def test_object_and_tensor_levels_agree(method):
    """The two centroid derivations must not drift (the whole point of #586).

    ``sio.Instance.to_centroid`` (object level, used for annotations) and
    ``generate_centroids`` (tensor level, used for training targets, crop centers
    and GT-centroid inference) must give the same answer for the same instance.
    """
    pts = _rand_instances(n=32, seed=1)
    skeleton = sio.Skeleton([f"n{i}" for i in range(pts.shape[1])])

    tensor_level = reduce_points(torch.from_numpy(pts).to(torch.float32), method)
    for i in range(len(pts)):
        instance = sio.Instance.from_numpy(pts[i].astype(np.float32), skeleton=skeleton)
        centroid = instance.to_centroid(method=method)
        mine = tensor_level[i].numpy()
        if np.isnan(centroid.x):
            assert np.isnan(mine).all()
            continue
        # float32 accumulation order differs between numpy and torch; the
        # geometric median is computed in float64 internally and matches exactly.
        np.testing.assert_allclose(mine, [centroid.x, centroid.y], rtol=0, atol=1e-3)


def test_object_and_tensor_levels_agree_for_the_anchor_method():
    """Anchor + fallback, the one method that combines a node and a reduction."""
    pts = _rand_instances(n=32, seed=2)
    skeleton = sio.Skeleton([f"n{i}" for i in range(pts.shape[1])])
    anchor_ind = 4

    for fallback in REDUCE_METHODS:
        tensor_level = generate_centroids(
            torch.from_numpy(pts).to(torch.float32),
            anchor_ind=anchor_ind,
            method="anchor",
            fallback=fallback,
        )
        for i in range(len(pts)):
            instance = sio.Instance.from_numpy(
                pts[i].astype(np.float32), skeleton=skeleton
            )
            centroid = instance.to_centroid(
                method="anchor", node=anchor_ind, fallback=fallback
            )
            mine = tensor_level[i].numpy()
            if np.isnan(centroid.x):
                assert np.isnan(mine).all()
                continue
            np.testing.assert_allclose(
                mine, [centroid.x, centroid.y], rtol=0, atol=1e-3
            )


def test_geometric_median_is_robust_to_an_outlying_node():
    """The reason `geometric_median` is worth having at all."""
    body = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    with_outlier = torch.cat(
        [body, torch.tensor([[[100.0, 100.0]]])], dim=1
    )  # a flung-out tail node

    mean_shift = (find_points_mean(with_outlier) - find_points_mean(body)).norm()
    median_shift = (
        find_points_geometric_median(with_outlier) - find_points_geometric_median(body)
    ).norm()
    assert median_shift < 0.05 * mean_shift


def test_geometric_median_edge_cases():
    """All-NaN, single visible point, and coincident points."""
    all_nan = torch.full((2, 5, 2), float("nan"))
    assert torch.isnan(find_points_geometric_median(all_nan)).all()

    one_visible = torch.full((1, 5, 2), float("nan"))
    one_visible[0, 2] = torch.tensor([7.0, -3.0])
    torch.testing.assert_close(
        find_points_geometric_median(one_visible), torch.tensor([[7.0, -3.0]])
    )

    # Every point identical: the Weiszfeld reweighting would divide by zero.
    coincident = torch.full((1, 4, 2), 5.0)
    torch.testing.assert_close(
        find_points_geometric_median(coincident), torch.tensor([[5.0, 5.0]])
    )


def test_geometric_median_preserves_dtype_and_batch_shape():
    pts = torch.from_numpy(_rand_instances(n=6)).to(torch.float32)
    batched = pts.unsqueeze(0).repeat(3, 1, 1, 1)  # (3, 6, 13, 2)
    out = find_points_geometric_median(batched)
    assert out.shape == (3, 6, 2)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out[0], out[1])


def test_geometric_median_is_batch_invariant():
    """An instance's centroid must not depend on its batch-mates.

    Weiszfeld converges at different rates for different instances. A batch-level
    stopping rule lets a slow slot keep iterating while its neighbours are done,
    so the same instance gets a different answer depending on what it was batched
    with (a real bug caught here: max delta 0.24 px). Convergence is tracked per
    slot instead.
    """
    pts = torch.from_numpy(_rand_instances(n=32, seed=1)).to(torch.float32)
    batched = find_points_geometric_median(pts)
    one_at_a_time = torch.stack(
        [find_points_geometric_median(pts[i : i + 1])[0] for i in range(len(pts))]
    )
    torch.testing.assert_close(batched, one_at_a_time, equal_nan=True)


def test_resolve_centroid_method_vocabulary():
    assert resolve_centroid_method() == ("center_of_mass", None)
    assert resolve_centroid_method(anchor_part="head") == ("anchor", "center_of_mass")
    assert resolve_centroid_method(
        anchor_part="head", centroid_fallback="bbox_center"
    ) == ("anchor", "bbox_center")
    assert resolve_centroid_method(centroid_method="geometric_median") == (
        "geometric_median",
        None,
    )
    # anchor_part + a non-anchor method name two different centroids.
    with pytest.raises(ValueError, match="Contradictory"):
        resolve_centroid_method(anchor_part="head", centroid_method="bbox_center")
    # "anchor" needs a node.
    with pytest.raises(ValueError, match="requires anchor_part"):
        resolve_centroid_method(centroid_method="anchor")
    with pytest.raises(ValueError, match="Unknown centroid_method"):
        resolve_centroid_method(centroid_method="centre_of_mass")
    # An anchor cannot fall back to another anchor.
    with pytest.raises(ValueError, match="Unknown centroid_fallback"):
        resolve_centroid_method(anchor_part="head", centroid_fallback="anchor")


def test_centroid_method_from_config_handles_every_config_shape():
    from omegaconf import OmegaConf

    assert centroid_method_from_config(None) == ("center_of_mass", None)
    assert centroid_method_from_config({}) == ("center_of_mass", None)
    # A config written before #586 keeps its historical meaning.
    assert centroid_method_from_config(
        OmegaConf.create({"anchor_part": "head", "sigma": 5.0})
    ) == ("anchor", "center_of_mass")
    assert centroid_method_from_config({"centroid_method": "bbox_center"}) == (
        "bbox_center",
        None,
    )


def test_degrade_anchor_if_unresolved():
    """An anchor_part absent from the skeleton degrades instead of raising."""
    assert degrade_anchor_if_unresolved("anchor", "bbox_center", 2) == (
        "anchor",
        "bbox_center",
    )
    assert degrade_anchor_if_unresolved("anchor", "bbox_center", None) == (
        "bbox_center",
        None,
    )
    assert degrade_anchor_if_unresolved("anchor", None, None) == (
        "center_of_mass",
        None,
    )
    assert degrade_anchor_if_unresolved("geometric_median", None, None) == (
        "geometric_median",
        None,
    )


def test_generate_centroids_rejects_bad_arguments():
    pts = torch.from_numpy(_rand_instances(n=2)).to(torch.float32)
    with pytest.raises(ValueError, match="requires anchor_ind"):
        generate_centroids(pts, method="anchor")
    with pytest.raises(ValueError, match="Unknown anchor fallback"):
        generate_centroids(pts, anchor_ind=0, method="anchor", fallback="anchor")
    with pytest.raises(ValueError, match="Unknown centroid reduce method"):
        reduce_points(pts, "median")


def test_every_method_is_reachable_through_generate_centroids():
    pts = torch.from_numpy(_rand_instances(n=4, seed=3)).to(torch.float32)
    for method in CENTROID_METHODS:
        kwargs = {"anchor_ind": 0} if method == "anchor" else {}
        out = generate_centroids(pts, method=method, **kwargs)
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all()


def test_evaluation_mirror_matches_the_tensor_op():
    """`compute_gt_centroids` must not drift from `generate_centroids` (#586)."""
    from sleap_nn.evaluation import compute_gt_centroids

    pts = _rand_instances(n=16, seed=4)
    for method in REDUCE_METHODS:
        np.testing.assert_allclose(
            compute_gt_centroids(pts, method=method),
            reduce_points(torch.from_numpy(pts), method).numpy(),
        )
    np.testing.assert_allclose(
        compute_gt_centroids(pts, anchor_ind=2),
        generate_centroids(torch.from_numpy(pts), anchor_ind=2).numpy(),
    )


def test_centroid_source_tag_follows_the_method():
    """The recorded `sio.Centroid.source` must not contradict the trained target."""
    from sleap_nn.inference.centroid_convert import centroid_source_for_anchor

    assert centroid_source_for_anchor(None) == "center_of_mass"
    assert centroid_source_for_anchor(1, ["a", "b"]) == "anchor:b"
    assert (
        centroid_source_for_anchor(None, None, "geometric_median") == "geometric_median"
    )
    assert centroid_source_for_anchor(None, None, "bbox_center") == "bbox_center"
    assert centroid_source_for_anchor(1, ["a", "b"], "anchor") == "anchor:b"


def test_add_centroids_from_masks(minimal_instance):
    """Mask-only labels gain the user centroids a centroid model needs (#674)."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    if not getattr(lf, "masks", None):
        # Build masks from the poses so this runs on the minimal fixture.
        labels.get_masks()
    if not getattr(labels[0], "masks", None):
        pytest.skip("fixture has no segmentation masks to derive centroids from")

    n_masks = sum(len(f.masks) for f in labels)
    n_added = add_centroids_from_masks(labels, method="center_of_mass")
    assert n_added == n_masks
    assert all(not c.is_predicted for f in labels for c in f.centroids)

    # A second pass must not duplicate: real annotations outrank derived ones.
    assert add_centroids_from_masks(labels, method="center_of_mass") == 0
    assert sum(len(f.centroids) for f in labels) == n_masks

    with pytest.raises(ValueError, match="unsupported method"):
        add_centroids_from_masks(labels, method="anchor")
    # `geometric_median` is a valid pose method but NOT a mask one -- reject it at
    # the call rather than letting sleap-io raise deep inside the load.
    with pytest.raises(ValueError, match="not offered for masks"):
        add_centroids_from_masks(labels, method="geometric_median")
