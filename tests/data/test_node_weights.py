"""Tests for per-node loss weights in the data pipeline."""

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import DictConfig

from sleap_nn.data.augmentation import (
    apply_flip_augmentation,
    apply_geometric_augmentation,
)
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    CenteredInstanceDataset,
    SingleInstanceDataset,
)
from sleap_nn.data.node_weights import (
    carry_labeled_nodes,
    finite_point_weights,
    get_labeled_nodes_weights,
    has_partial_labeling_info,
    instance_node_weights,
    node_weights_active,
)


def _labels(path, labeled_nodes=None, symmetric=False):
    labels = sio.load_slp(path)
    if labeled_nodes is not None:
        labels.provenance["labeled_nodes"] = labeled_nodes
    if symmetric:
        skel = labels.skeletons[0]
        skel.add_symmetry(skel.nodes[0], skel.nodes[1])
    return labels


def test_get_labeled_nodes_weights(minimal_instance):
    labels = _labels(minimal_instance)
    assert np.array_equal(get_labeled_nodes_weights(labels), [1, 1])
    assert not has_partial_labeling_info([labels])
    labels.provenance["labeled_nodes"] = ["B"]
    assert np.array_equal(get_labeled_nodes_weights(labels), [0, 1])
    assert has_partial_labeling_info([labels])
    labels.provenance["labeled_nodes"] = [0]
    assert np.array_equal(get_labeled_nodes_weights(labels), [1, 0])
    labels.provenance["labeled_nodes"] = ["A", "not_a_node"]
    assert np.array_equal(get_labeled_nodes_weights(labels), [1, 0])
    labels.provenance["labeled_nodes"] = []
    assert np.array_equal(get_labeled_nodes_weights(labels), [0, 0])
    # Explicit model node order.
    labels.provenance["labeled_nodes"] = ["A"]
    assert np.array_equal(get_labeled_nodes_weights(labels, ["B", "A"]), [0, 1])


def test_node_weights_active(minimal_instance):
    full = _labels(minimal_instance)
    partial = _labels(minimal_instance, labeled_nodes=["A"])
    assert not node_weights_active([full])
    assert node_weights_active([full, partial])
    assert node_weights_active([full], unlabeled_points="ignore")
    # node_loss_mask is the master switch.
    assert not node_weights_active([partial], node_loss_mask=False)
    assert not node_weights_active(
        [full], node_loss_mask=False, unlabeled_points="ignore"
    )
    with pytest.raises(ValueError, match="unlabeled_points"):
        node_weights_active([full], unlabeled_points="drop")


def test_carry_labeled_nodes(minimal_instance):
    source = _labels(minimal_instance, labeled_nodes=["A"])
    # sleap-io's splits build fresh provenance (the trap this guards against).
    train, val = source.make_training_splits(n_train=0.5, n_val=0.5, seed=0)
    assert "labeled_nodes" not in train.provenance
    carry_labeled_nodes(source, train, val)
    assert train.provenance["labeled_nodes"] == ["A"]
    assert val.provenance["labeled_nodes"] == ["A"]
    # A split's own value wins; a source without the key is a no-op.
    val.provenance["labeled_nodes"] = ["B"]
    carry_labeled_nodes(source, val)
    assert val.provenance["labeled_nodes"] == ["B"]
    other = sio.Labels()
    carry_labeled_nodes(_labels(minimal_instance), other)
    assert "labeled_nodes" not in other.provenance


def test_finite_point_weights():
    pts = torch.tensor([[1.0, 2.0], [np.nan, np.nan], [3.0, np.nan]])
    assert torch.equal(finite_point_weights(pts), torch.tensor([1.0, 0.0, 0.0]))
    # Several instances: a node counts if any instance has it.
    two = torch.stack([pts, torch.tensor([[np.nan, 0.0], [5.0, 5.0], [np.nan] * 2])])
    assert torch.equal(finite_point_weights(two[None]), torch.tensor([1.0, 1.0, 0.0]))


def test_flip_swaps_node_weights():
    image = torch.zeros(1, 1, 8, 10)
    inst = torch.tensor([[[1.0, 2.0], [7.0, 3.0], [4.0, 4.0]]])  # (1, 3, 2)
    w = torch.tensor([[1.0, 0.0, 1.0]])
    _, inst_f, w_f = apply_flip_augmentation(
        image, inst, symmetric_inds=[(0, 1)], flip_p=1.0, node_weights=w
    )
    assert torch.equal(w_f, torch.tensor([[0.0, 1.0, 1.0]]))
    assert torch.equal(w, torch.tensor([[1.0, 0.0, 1.0]]))  # input untouched
    # Node 0 slot now holds the mirrored node-1 point.
    assert torch.allclose(inst_f[0, 0], torch.tensor([9.0 - 7.0, 3.0]))
    # No flip: unchanged.
    _, _, w_n = apply_flip_augmentation(
        image, inst, symmetric_inds=[(0, 1)], flip_p=0.0, node_weights=w
    )
    assert torch.equal(w_n, w)
    # Through the geometric wrapper (flip before affine), with masks too.
    masks = torch.zeros(1, 2, 8, 10)
    out = apply_geometric_augmentation(
        image.to(torch.uint8),
        inst,
        flip_p=1.0,
        symmetric_inds=[(0, 1)],
        masks=masks,
        node_weights=w,
    )
    assert len(out) == 4
    assert torch.equal(out[-1], torch.tensor([[0.0, 1.0, 1.0]]))
    # Without node_weights the return arity is unchanged.
    assert len(apply_geometric_augmentation(image, inst, flip_p=1.0)) == 2
    assert len(apply_flip_augmentation(image, inst, flip_p=1.0, masks=masks)) == 3


def test_flip_weights_follow_keypoints_for_any_pair_list():
    """Weights are swapped by the same sequential swaps as the keypoints."""
    image = torch.zeros(1, 1, 8, 10)
    inst = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pairs = [(0, 1), (1, 0), (2, 3)]  # duplicated pair: swaps back
    _, inst_f, w_f = apply_flip_augmentation(
        image, inst, symmetric_inds=pairs, flip_p=1.0, node_weights=w
    )
    # Weight i still belongs to the keypoint that originally had weight i.
    orig_y = inst[0, :, 1]
    for slot in range(4):
        src = int(w_f[0, slot].item()) - 1
        assert inst_f[0, slot, 1] == orig_y[src]


def _ci_dataset(labels, flip_p=0.0, **kwargs):
    geo = (
        DictConfig(
            {
                "flip_p": flip_p,
                "affine_p": 0.0,
                "erase_p": 0.0,
                "mixup_p": 0.0,
            }
        )
        if flip_p > 0
        else None
    )
    return CenteredInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=DictConfig(
            {"sigma": 1.5, "output_stride": 2, "anchor_part": None}
        ),
        crop_size=160,
        labels=[labels],
        apply_aug=flip_p > 0,
        geometric_aug=geo,
        **kwargs,
    )


def test_centered_instance_dataset_node_weights_and_flip(minimal_instance):
    labels = _labels(minimal_instance, labeled_nodes=["A"], symmetric=True)
    ds = _ci_dataset(labels)
    s = ds[0]
    assert torch.equal(s["node_weights"], torch.tensor([[1.0, 0.0]]))

    ds_flip = _ci_dataset(labels, flip_p=1.0)
    s_f = ds_flip[0]
    # Weights swap with the node channels: B's slot now carries A's label.
    assert torch.equal(s_f["node_weights"], torch.tensor([[0.0, 1.0]]))
    # Point/weight registration: the weighted slot holds the original A point,
    # mirrored (crop is centered, so x -> W - 1 - x within the crop).
    w = s["instance_image"].shape[-1]
    a = s["instance"][0, 0]
    b_flipped = s_f["instance"][0, 1]
    assert torch.allclose(b_flipped[0], (w - 1) - a[0], atol=1e-3)
    assert torch.allclose(b_flipped[1], a[1], atol=1e-3)


def test_centered_instance_no_weights_by_default(minimal_instance):
    """Fully labeled data under the defaults keeps its exact sample dict."""
    labels = _labels(minimal_instance)
    s = _ci_dataset(labels)[0]
    assert "node_weights" not in s
    ref = _ci_dataset(labels, node_loss_mask=False)[0]
    assert s.keys() == ref.keys()
    assert torch.equal(s["confidence_maps"], ref["confidence_maps"])
    # The mask is off entirely without node_loss_mask.
    partial = _labels(minimal_instance, labeled_nodes=["A"])
    assert "node_weights" not in _ci_dataset(partial, node_loss_mask=False)[0]


def test_centered_instance_unlabeled_points(minimal_instance):
    labels = _labels(minimal_instance)
    labels[0].instances[0].points["xy"][1] = np.nan  # node B has no location
    labels[0].instances[0].points["visible"][1] = False

    ds_absent = _ci_dataset(labels)
    assert "node_weights" not in ds_absent[0]  # nothing differs from 1

    ds_ignore = _ci_dataset(labels, unlabeled_points="ignore")
    assert torch.equal(ds_ignore[0]["node_weights"], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(ds_ignore[1]["node_weights"], torch.tensor([[1.0, 1.0]]))
    # Targets are unchanged: weighting never alters the confmaps.
    assert torch.equal(ds_ignore[0]["confidence_maps"], ds_absent[0]["confidence_maps"])

    # Combined with the file mask.
    labels.provenance["labeled_nodes"] = ["B"]
    ds_both = _ci_dataset(labels, unlabeled_points="ignore")
    assert torch.equal(ds_both[0]["node_weights"], torch.tensor([[0.0, 0.0]]))
    assert torch.equal(ds_both[1]["node_weights"], torch.tensor([[0.0, 1.0]]))


def test_centered_instance_out_of_crop_point_ignored(minimal_instance):
    labels = _labels(minimal_instance)
    # Put node B of instance 0 far outside its centered crop (still in frame).
    pts = labels[0].instances[0].points["xy"]
    pts[1] = pts[0] + np.array([150.0, 0.0])
    ds = CenteredInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=DictConfig(
            {"sigma": 1.5, "output_stride": 2, "anchor_part": "A"}
        ),
        crop_size=64,
        anchor_ind=0,
        labels=[labels],
        unlabeled_points="ignore",
    )
    s = ds[0]
    assert torch.isnan(s["instance"][0, 1]).all()
    assert torch.equal(s["node_weights"], torch.tensor([[1.0, 0.0]]))


def _frame_datasets(labels, **kwargs):
    head = DictConfig({"sigma": 1.5, "output_stride": 2})
    si = SingleInstanceDataset(
        labels=[labels], confmap_head_config=head, max_stride=8, **kwargs
    )
    bu = BottomUpDataset(
        labels=[labels],
        confmap_head_config=head,
        pafs_head_config=DictConfig({"sigma": 4, "output_stride": 4}),
        max_stride=8,
        **kwargs,
    )
    return si, bu


def test_frame_level_datasets_emit_file_weights(minimal_instance):
    labels = _labels(minimal_instance, labeled_nodes=["B"])
    si, bu = _frame_datasets(labels)
    assert torch.equal(si[0]["node_weights"], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(bu[0]["node_weights"], torch.tensor([[0.0, 1.0]]))
    si_off, bu_off = _frame_datasets(labels, node_loss_mask=False)
    assert "node_weights" not in si_off[0]
    assert "node_weights" not in bu_off[0]


def test_frame_level_unlabeled_points(minimal_instance):
    labels = _labels(minimal_instance)
    # Node B missing in instance 0 only: bottom-up keeps it (instance 1 has it).
    labels[0].instances[0].points["xy"][1] = np.nan
    labels[0].instances[0].points["visible"][1] = False
    _, bu = _frame_datasets(labels, unlabeled_points="ignore")
    assert torch.equal(bu[0]["node_weights"], torch.tensor([[1.0, 1.0]]))
    # Missing in every instance: ignored.
    labels[0].instances[1].points["xy"][1] = np.nan
    labels[0].instances[1].points["visible"][1] = False
    _, bu = _frame_datasets(labels, unlabeled_points="ignore")
    assert torch.equal(bu[0]["node_weights"], torch.tensor([[1.0, 0.0]]))

    # Single instance: one instance per frame.
    single = _labels(minimal_instance)
    single[0].instances = single[0].instances[:1]
    single[0].instances[0].points["xy"][0] = np.nan
    single[0].instances[0].points["visible"][0] = False
    si, _ = _frame_datasets(single, unlabeled_points="ignore")
    assert torch.equal(si[0]["node_weights"], torch.tensor([[0.0, 1.0]]))
    si_absent, _ = _frame_datasets(single)
    assert "node_weights" not in si_absent[0]


def test_negative_frames_keep_unit_weights(minimal_instance):
    labels = _labels(minimal_instance, labeled_nodes=["A"])
    lf = labels[0]
    labels.append(sio.LabeledFrame(video=lf.video, frame_idx=0, is_negative=True))
    si, bu = _frame_datasets(
        labels, use_negative_frames=True, unlabeled_points="ignore"
    )
    for ds in (si, bu):
        negatives = [ds[i] for i in range(len(ds)) if ds[i].get("is_negative")]
        assert negatives, "expected a negative sample"
        for s in negatives:
            assert torch.equal(s["node_weights"], torch.ones(1, 2))


class _FakeInstance:
    """Instance whose points expose sleap-io >= 0.10 style flags."""

    def __init__(self, xy, visible, labeled):
        dt = [("xy", "<f8", (2,)), ("visible", "?"), ("labeled", "?")]
        self.points = np.zeros(len(xy), dtype=dt)
        self.points["xy"] = xy
        self.points["visible"] = visible
        self.points["labeled"] = labeled


def test_instance_node_weights_forward_compat():
    xy = np.array([[1.0, 1.0], [2.0, 2.0], [np.nan, np.nan], [3.0, 3.0]])
    inst = _FakeInstance(
        xy,
        visible=[True, False, False, False],
        labeled=[True, True, True, False],
    )
    pts_nan = xy.copy()
    pts_nan[1:] = np.nan  # as Instance.numpy() returns invisible points
    file_w = np.ones(4, np.float32)

    w, pts = instance_node_weights(inst, file_w, pts_nan, supervise_occluded=True)
    # labeled|visible; node 2 is labeled-absent (weight 1, NaN target).
    assert np.array_equal(w, [1, 1, 1, 0])
    assert np.allclose(pts[1], [2.0, 2.0])  # occluded estimate restored
    assert np.isnan(pts[2]).all()

    w, pts = instance_node_weights(inst, file_w, pts_nan, supervise_occluded=False)
    assert np.array_equal(w, [1, 0, 1, 0])
    assert np.isnan(pts[1]).all()

    # Today's sleap-io (no `labeled` field): only the file mask applies.
    class _Old:
        points = np.zeros(2, dtype=[("xy", "<f8", (2,)), ("visible", "?")])

    w, _ = instance_node_weights(_Old(), np.array([1, 0], np.float32), xy[:2])
    assert np.array_equal(w, [1, 0])
