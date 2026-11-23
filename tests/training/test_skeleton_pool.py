"""Frame models on labels files with different skeletons (experimental).

Covers ``data_config.merge_skeletons``, a list of centroid anchors, the balanced
per-file sampler for the bottom-up / centroid / single-instance datasets and
``preprocessing.file_scales``.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    CentroidDataset,
    SingleInstanceDataset,
    get_train_val_dataloaders,
    get_train_val_datasets,
)
from sleap_nn.data.topdown_warp import BalancedDataLoader
from sleap_nn.training.model_trainer import ModelTrainer
from tests.utils.cli import cpu_only_env, sleap_nn_cli

# Points of each node in the one frame of `minimal_instance.pkg.slp`: A and B
# are labelled there; C is the midpoint of the two.
_ALL_NODES = ("A", "B", "C")


# `sleap_nn.data.skeleton_pool` is imported inside the tests that use it, so the
# other tests here can run (and fail on behavior) against code without it.
def map_labels_to_skeleton(labels, skeleton):
    from sleap_nn.data.skeleton_pool import map_labels_to_skeleton as f

    return f(labels, skeleton)


def pool_skeleton(labels_list, node_names):
    from sleap_nn.data.skeleton_pool import pool_skeleton as f

    return f(labels_list, node_names)


def config_for_saving(config):
    from sleap_nn.data.skeleton_pool import config_for_saving as f

    return f(config)


def _pool_file(
    base: sio.Labels,
    out_dir: Path,
    name: str,
    nodes,
    prefix="",
    n_frames=1,
    one_instance=False,
):
    """``base``'s frame as a PNG-backed labels file on ``nodes`` (a subset of A, B, C).

    Node names get ``prefix``. The frame is repeated ``n_frames`` times (an image
    sequence of the same PNG); ``one_instance`` keeps its first instance only.
    Saved to ``<out_dir>/<name>.slp``; returns the path.
    """
    lf = base[0]
    img = lf.image
    png = str(out_dir / f"{name}.png")
    cv2.imwrite(png, img[..., 0] if img.shape[-1] == 1 else img)
    names = [prefix + n for n in nodes]
    sk = sio.Skeleton(
        nodes=names, edges=[(a, b) for a, b in zip(names[:-1], names[1:])]
    )
    video = sio.Video.from_filename([png] * n_frames)
    frames = []
    for frame_idx in range(n_frames):
        insts = []
        for inst in lf.user_instances[: 1 if one_instance else None]:
            a, b = inst.numpy()
            pts = {"A": a, "B": b, "C": (a + b) / 2}
            insts.append(sio.Instance.from_numpy(np.array([pts[n] for n in nodes]), sk))
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=insts)
        )
    labels = sio.Labels(frames, skeletons=[sk], videos=[video])
    path = out_dir / f"{name}.slp"
    labels.save(str(path))
    return path


@pytest.fixture
def pool_files(minimal_instance, tmp_path):
    """Two files: one on nodes A, B and one on nodes B, C."""
    base = sio.load_slp(minimal_instance)
    return (
        _pool_file(base, tmp_path, "ab", ["A", "B"]),
        _pool_file(base, tmp_path, "bc", ["B", "C"]),
    )


def _frame_config(config, tmp_path, model_type, paths, run_name):
    """A tiny CPU config of ``model_type`` on ``paths`` (train = val files)."""
    cfg = config.copy()
    cfg.data_config.train_labels_path = [str(p) for p in paths]
    cfg.data_config.val_labels_path = [str(p) for p in paths]
    cfg.data_config.augmentation_config.geometric.affine_p = 0.0
    heads = cfg.model_config.head_configs
    del heads.centered_instance
    if model_type == "bottomup":
        heads.bottomup = {
            "confmaps": {
                "part_names": None,
                "sigma": 1.5,
                "output_stride": 2,
                "loss_weight": 1.0,
            },
            "pafs": {
                "edges": None,
                "sigma": 4,
                "output_stride": 4,
                "loss_weight": 1.0,
            },
        }
    elif model_type == "centroid":
        heads.centroid = {
            "confmaps": {
                "anchor_part": None,
                "sigma": 1.5,
                "output_stride": 2,
            }
        }
    elif model_type == "single_instance":
        heads.single_instance = {
            "confmaps": {"part_names": None, "sigma": 1.5, "output_stride": 2}
        }
    tc = cfg.trainer_config
    tc.trainer_accelerator = "cpu"
    tc.max_epochs = 1
    tc.min_train_steps_per_epoch = 2
    tc.train_data_loader.batch_size = 2
    tc.val_data_loader.batch_size = 2
    tc.save_ckpt = True
    tc.ckpt_dir = str(tmp_path)
    tc.run_name = run_name
    tc.use_wandb = False
    tc.visualize_preds_during_training = False
    OmegaConf.update(cfg, "trainer_config.eval.post_train_splits", [], merge=True)
    return cfg


def _datasets(config, train, val=None):
    cfg = config.copy()
    cfg.data_config.preprocessing.max_height = 384
    cfg.data_config.preprocessing.max_width = 384
    return get_train_val_datasets(train, val if val is not None else train, cfg)


# --------------------------------------------------------------------------- #
# Mapping files onto the union skeleton
# --------------------------------------------------------------------------- #
def test_map_labels_marks_missing_nodes_unlabeled(pool_files):
    ab, bc = (sio.load_slp(p) for p in pool_files)
    union = pool_skeleton([ab, bc], list(_ALL_NODES))
    assert [(e.source.name, e.destination.name) for e in union.edges] == [
        ("A", "B"),
        ("B", "C"),
    ]
    mapped = map_labels_to_skeleton(bc, union)
    inst = mapped[0].instances[0]
    assert inst.skeleton is union and mapped.skeletons == [union]
    assert list(inst.points["labeled"]) == [False, True, True]
    assert np.isnan(inst.numpy()[0]).all()
    np.testing.assert_allclose(inst.numpy()[1:], bc[0].instances[0].numpy())
    assert mapped.provenance["labeled_nodes"] == ["B", "C"]
    # The input is left alone.
    assert bc[0].instances[0].skeleton is bc.skeletons[0]
    assert len(bc[0].instances[0].points) == 2


def test_map_labels_keeps_existing_mask(pool_files):
    ab = sio.load_slp(pool_files[0])
    ab.provenance["labeled_nodes"] = [1]  # index into the file's skeleton: "B"
    union = pool_skeleton([ab], list(_ALL_NODES))
    assert map_labels_to_skeleton(ab, union).provenance["labeled_nodes"] == ["B"]


def test_mixed_skeletons_still_refused_without_merge(config, tmp_path, pool_files):
    cfg = _frame_config(config, tmp_path, "bottomup", pool_files, "refuse")
    with pytest.raises(ValueError, match="does not match the skeleton"):
        ModelTrainer.get_model_trainer_from_config(cfg)


def test_merge_refused_for_other_model_types(config, tmp_path, pool_files):
    cfg = _frame_config(config, tmp_path, "bottomup", pool_files, "other")
    cfg.model_config.head_configs.bottomup = None
    cfg.model_config.head_configs.centered_instance = {
        "confmaps": {"part_names": None, "anchor_part": None, "sigma": 1.5}
    }
    cfg.data_config.merge_skeletons = True
    with pytest.raises(ValueError, match="merge_skeletons is supported"):
        ModelTrainer.get_model_trainer_from_config(cfg)


def test_bottomup_weights_on_merged_files(config, tmp_path, pool_files):
    cfg = _frame_config(config, tmp_path, "bottomup", pool_files, "weights")
    cfg.data_config.merge_skeletons = True
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    assert trainer.skeletons[0].node_names == list(_ALL_NODES)
    head = trainer.config.model_config.head_configs.bottomup
    assert list(head.confmaps.part_names) == list(_ALL_NODES)
    assert [list(e) for e in head.pafs.edges] == [["A", "B"], ["B", "C"]]
    train, _ = get_train_val_datasets(
        trainer.train_labels, trainer.val_labels, trainer.config
    )
    assert isinstance(train, BottomUpDataset)
    by_file = {}
    for i in range(len(train)):
        s = train[i]
        by_file[s["labels_idx"]] = s
    # File "ab" lacks C: no loss on C and on the B-C edge; "bc" lacks A.
    torch.testing.assert_close(by_file[0]["node_weights"], torch.tensor([[1, 1, 0.0]]))
    torch.testing.assert_close(by_file[0]["edge_weights"], torch.tensor([[1, 0.0]]))
    torch.testing.assert_close(by_file[1]["node_weights"], torch.tensor([[0, 1, 1.0]]))
    torch.testing.assert_close(by_file[1]["edge_weights"], torch.tensor([[0, 1.0]]))
    # The viz / eval treat each file as having only its own nodes.
    assert trainer._viz_file_nodes()[0] == [[0, 1], [1, 2]]


# --------------------------------------------------------------------------- #
# Centroid anchors given as a list
# --------------------------------------------------------------------------- #
def _centroid_datasets(config, labels_list, anchor_part):
    cfg = config.copy()
    del cfg.model_config.head_configs.centered_instance
    cfg.model_config.head_configs.centroid = {
        "confmaps": {
            "anchor_part": anchor_part,
            "sigma": 1.5,
            "output_stride": 2,
            "centroid_source": "computed",
        }
    }
    sk = labels_list[0].skeletons[0]
    cfg.data_config.skeletons = [
        {"name": sk.name, "nodes": [{"name": n} for n in sk.node_names]}
    ]
    cfg.data_config.use_augmentations_train = False
    return _datasets(cfg, labels_list)


def test_centroid_anchor_list_takes_first_labelled_node(config, pool_files):
    files = [sio.load_slp(p) for p in pool_files]
    union = pool_skeleton(files, list(_ALL_NODES))
    mapped = [map_labels_to_skeleton(lb, union) for lb in files]
    train, _ = _centroid_datasets(config, mapped, ["A", "B"])
    assert isinstance(train, CentroidDataset)
    for i in range(len(train)):
        s = train[i]
        lb = files[s["labels_idx"]]
        want = np.stack([inst.numpy()[0] for inst in lb[0].instances])
        # "ab" anchors on A (its node 0), "bc" on B (its node 0).
        np.testing.assert_allclose(s["centroids"][0, :2].numpy(), want, atol=1e-4)
        torch.testing.assert_close(s["node_weights"], torch.tensor([[1.0]]))


def test_centroid_anchor_list_masks_files_without_anchor(config, pool_files):
    files = [sio.load_slp(p) for p in pool_files]
    union = pool_skeleton(files, list(_ALL_NODES))
    mapped = [map_labels_to_skeleton(lb, union) for lb in files]
    train, _ = _centroid_datasets(config, mapped, ["C"])
    weights = {}
    for i in range(len(train)):
        s = train[i]
        weights[s["labels_idx"]] = float(s["node_weights"])
        if s["labels_idx"] == 0:
            # No centroid at a fallback location: the "ab" file labels no C.
            assert torch.isnan(s["centroids"]).all()
    assert weights == {0: 0.0, 1: 1.0}


def test_centroid_single_anchor_unchanged(config, pool_files):
    files = [sio.load_slp(p) for p in pool_files]
    union = pool_skeleton(files, list(_ALL_NODES))
    mapped = [map_labels_to_skeleton(lb, union) for lb in files]
    train, _ = _centroid_datasets(config, mapped, "C")
    s = train[0]
    assert "node_weights" not in s
    # Today's fallback: the mean of the located nodes.
    lb = files[s["labels_idx"]]
    want = np.stack([inst.numpy().mean(axis=0) for inst in lb[0].instances])
    if s["labels_idx"] == 1:
        want = np.stack([inst.numpy()[1] for inst in lb[0].instances])
    np.testing.assert_allclose(s["centroids"][0, :2].numpy(), want, atol=1e-4)


def test_centroid_loss_ignores_zero_weight_frames(config):
    from sleap_nn.training.lightning_modules import CentroidLightningModule

    cfg = config.copy()
    del cfg.model_config.head_configs.centered_instance
    cfg.model_config.head_configs.centroid = {
        "confmaps": {"anchor_part": None, "sigma": 1.5, "output_stride": 2}
    }
    model = CentroidLightningModule(
        model_type="centroid",
        backbone_config=cfg.model_config.backbone_config,
        backbone_type="unet",
        head_configs=cfg.model_config.head_configs,
        init_weights=cfg.model_config.init_weights,
        lr_scheduler=cfg.trainer_config.lr_scheduler,
    )
    g = torch.Generator().manual_seed(0)
    y, y_pr = torch.rand(3, 1, 8, 8, generator=g), torch.rand(3, 1, 8, 8, generator=g)
    batch = {"node_weights": torch.tensor([[[1.0]], [[0.0]], [[1.0]]])}
    keep = [0, 2]
    want = torch.nn.MSELoss()(y_pr[keep], y[keep])
    for stage in ("train", "val"):
        got = model._compute_loss(y_pr, y, batch, stage=stage)
        torch.testing.assert_close(got, want)
    # Without weights: exactly today's loss.
    assert torch.equal(model._compute_loss(y_pr, y, {}), torch.nn.MSELoss()(y_pr, y))


# --------------------------------------------------------------------------- #
# Balanced sampler and per-file scales
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", ["bottomup", "centroid", "single_instance"])
def test_balanced_sampler_for_frame_models(config, pool_files, model_type):
    files = [sio.load_slp(p) for p in pool_files]
    union = pool_skeleton(files, list(_ALL_NODES))
    mapped = [map_labels_to_skeleton(lb, union) for lb in files]
    # The "ab" file twice, so the sampler has an unequal pool to balance.
    mapped = [mapped[0], mapped[0], mapped[1]]
    cfg = config.copy()
    del cfg.model_config.head_configs.centered_instance
    leaf = {"part_names": list(_ALL_NODES), "sigma": 1.5, "output_stride": 2}
    if model_type == "bottomup":
        cfg.model_config.head_configs.bottomup = {
            "confmaps": leaf,
            "pafs": {"edges": [["A", "B"]], "sigma": 4, "output_stride": 4},
        }
    elif model_type == "centroid":
        cfg.model_config.head_configs.centroid = {
            "confmaps": {"anchor_part": None, "sigma": 1.5, "output_stride": 2}
        }
    else:
        cfg.model_config.head_configs.single_instance = {"confmaps": leaf}
        mapped = [
            sio.Labels(
                [sio.LabeledFrame(video=lf.video, frame_idx=0, instances=lf[:1])],
                videos=lb.videos,
                skeletons=lb.skeletons,
                provenance=lb.provenance,
            )
            for lb in mapped
            for lf in lb
        ]
    cfg.trainer_config.balanced_sampler = {"counts": [1, 0, 1]}
    cfg.trainer_config.train_data_loader.batch_size = 2
    train, val = _datasets(cfg, mapped)
    assert train.num_files == 3 and list(train.labels_idx) == [0, 1, 2]
    loader, _ = get_train_val_dataloaders(train, val, cfg, train_steps_per_epoch=3)
    assert isinstance(loader, BalancedDataLoader)
    for batch in loader:
        assert sorted(batch["labels_idx"].tolist()) == [0, 2]


def test_file_scales_pad_instead_of_stretch(config, pool_files):
    files = [sio.load_slp(p) for p in pool_files]
    union = pool_skeleton(files, list(_ALL_NODES))
    mapped = [map_labels_to_skeleton(lb, union) for lb in files]
    cfg = config.copy()
    del cfg.model_config.head_configs.centered_instance
    cfg.model_config.head_configs.single_instance = {
        "confmaps": {"part_names": list(_ALL_NODES), "sigma": 1.5, "output_stride": 2}
    }
    cfg.data_config.use_augmentations_train = False
    plain = [
        sio.Labels(
            [sio.LabeledFrame(video=lf.video, frame_idx=0, instances=lf[:1])],
            videos=lb.videos,
            skeletons=lb.skeletons,
            provenance=lb.provenance,
        )
        for lb in mapped
        for lf in lb
    ]
    ref, _ = _datasets(cfg, plain)
    cfg.data_config.preprocessing.file_scales = {"ab": 0.5}
    scaled, _ = _datasets(cfg, plain)
    assert isinstance(scaled, SingleInstanceDataset)
    assert scaled.file_scales == [0.5, 1.0]
    a, b = scaled[0], ref[0]
    # Same canvas; the frame is half size in its top-left corner, zero elsewhere.
    assert a["image"].shape == b["image"].shape
    assert float(a["eff_scale"]) == 0.5
    assert a["image"][..., 192:, :].abs().sum() == 0
    assert a["image"][..., :192, 192:].abs().sum() == 0
    torch.testing.assert_close(a["instances"], b["instances"] * 0.5, equal_nan=True)
    # The other file is untouched.
    torch.testing.assert_close(scaled[1]["image"], ref[1]["image"])
    # A key naming no file fails fast.
    cfg.data_config.preprocessing.file_scales = {"nope": 0.5}
    with pytest.raises(ValueError, match="match no training"):
        _datasets(cfg, plain)


# --------------------------------------------------------------------------- #
# Saved configs
# --------------------------------------------------------------------------- #
def test_saved_config_leaves_out_unused_keys():
    cfg = OmegaConf.structured(TrainingJobConfig())
    out = config_for_saving(cfg)
    assert "merge_skeletons" not in out.data_config
    assert "file_scales" not in out.data_config.preprocessing
    # Everything else is saved as before.
    ref = OmegaConf.to_container(cfg)
    del ref["data_config"]["merge_skeletons"]
    del ref["data_config"]["preprocessing"]["file_scales"]
    assert OmegaConf.to_container(out) == ref
    cfg.data_config.merge_skeletons = True
    cfg.data_config.preprocessing.file_scales = {"a": 0.5}
    out = config_for_saving(cfg)
    assert out.data_config.merge_skeletons is True
    assert dict(out.data_config.preprocessing.file_scales) == {"a": 0.5}


# --------------------------------------------------------------------------- #
# Train, save and predict from the checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", ["bottomup", "centroid", "single_instance"])
def test_merged_pool_trains_and_predicts(
    config, tmp_path, pool_files, model_type, minimal_instance
):
    if model_type == "single_instance":
        base = sio.load_slp(minimal_instance)
        pool_files = (
            _pool_file(base, tmp_path, "ab1", ["A", "B"], one_instance=True),
            _pool_file(base, tmp_path, "bc1", ["B", "C"], one_instance=True),
        )
    cfg = _frame_config(config, tmp_path, model_type, pool_files, model_type)
    cfg.data_config.merge_skeletons = True
    cfg.trainer_config.balanced_sampler = {"counts": [1, 1]}
    if model_type == "centroid":
        cfg.model_config.head_configs.centroid.confmaps.anchor_part = ["A", "C"]
    ModelTrainer.get_model_trainer_from_config(cfg).train()

    run = tmp_path / model_type
    saved = OmegaConf.load(run / "training_config.yaml")
    assert saved.data_config.merge_skeletons is True
    assert "file_scales" not in saved.data_config.preprocessing
    nodes = [n["name"] for n in saved.data_config.skeletons[0]["nodes"]]
    assert nodes == list(_ALL_NODES)
    if model_type == "centroid":
        assert list(saved.model_config.head_configs.centroid.confmaps.anchor_part) == [
            "A",
            "C",
        ]

    out = tmp_path / f"{model_type}.pr.slp"
    subprocess.run(
        sleap_nn_cli(
            "predict",
            "--data_path",
            str(pool_files[1]),
            "--model_paths",
            str(run),
            "--output_path",
            str(out),
            "--device",
            "cpu",
            "--peak_threshold",
            "0.0",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=cpu_only_env(),
    )
    pred = sio.load_slp(str(out))
    assert len(pred) == 1
    if model_type != "centroid":
        assert pred.skeletons[0].node_names == list(_ALL_NODES)


def test_default_run_saves_no_new_keys(config, tmp_path, minimal_instance):
    cfg = _frame_config(config, tmp_path, "bottomup", [minimal_instance], "plain")
    cfg.trainer_config.max_epochs = 1
    cfg.model_config.head_configs.bottomup.pafs.edges = [["A", "B"]]
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    for name in ("training_config.yaml", "initial_config.yaml"):
        text = (tmp_path / "plain" / name).read_text()
        assert "merge_skeletons" not in text and "file_scales" not in text


# --------------------------------------------------------------------------- #
# Review follow-ups
# --------------------------------------------------------------------------- #
def test_file_scales_with_default_validation_split(config, tmp_path, minimal_instance):
    """Keys match split files too (sleap-io records them as `source_labels`)."""
    base = sio.load_slp(minimal_instance)
    paths = [
        _pool_file(base, tmp_path, "ab", ["A", "B"], n_frames=4),
        _pool_file(base, tmp_path, "bc", ["B", "C"], n_frames=4),
    ]
    cfg = _frame_config(config, tmp_path, "bottomup", paths, "split")
    cfg.data_config.val_labels_path = None
    cfg.data_config.validation_fraction = 0.25
    cfg.data_config.merge_skeletons = True
    cfg.data_config.preprocessing.file_scales = {"ab": 0.5}
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    assert "filename" not in trainer.train_labels[0].provenance
    train, val = get_train_val_datasets(
        trainer.train_labels, trainer.val_labels, trainer.config
    )
    assert train.file_scales == [0.5, 1.0]
    assert val.file_scales == [0.5, 1.0]
    trainer.train()


def test_file_scales_refuses_ambiguous_key(tmp_path):
    from sleap_nn.data.custom_datasets import check_file_scales

    a, b = sio.Labels(), sio.Labels()
    a.provenance["filename"] = str(tmp_path / "x.v1.slp")
    b.provenance["source_labels"] = str(tmp_path / "x.v2.slp")
    with pytest.raises(ValueError, match="more than one file"):
        check_file_scales({"x": 0.5}, [a], [b])
    # A full file name or path is unique; the same file on both sides is one file.
    check_file_scales({"x.v1.slp": 0.5, str(tmp_path / "x.v2.slp"): 2.0}, [a], [b])
    check_file_scales({"x": 0.5}, [a], [a])


def _centroid_job(config, anchor_part):
    cfg = config.copy()
    del cfg.model_config.head_configs.centered_instance
    cfg.model_config.head_configs.centroid = {
        "confmaps": {"anchor_part": anchor_part, "sigma": 1.5, "output_stride": 2}
    }
    return cfg


def test_anchor_part_scalars_become_names_and_bad_values_fail(config):
    from sleap_nn.config.training_job_config import verify_training_cfg

    def anchor(value):
        cfg = verify_training_cfg(_centroid_job(config, value))
        return cfg.model_config.head_configs.centroid.confmaps.anchor_part

    assert anchor(1) == "1"  # as the former `str` field read it
    assert anchor("A") == "A"
    assert anchor(None) is None
    assert list(anchor(["A", 2])) == ["A", "2"]
    for bad in ([], {"a": 1}):
        with pytest.raises(Exception, match="anchor_part"):
            anchor(bad)


def test_centroid_anchor_list_falls_back_for_unlabelled_points(config, pool_files):
    """Only the FILE decides zero weight; an unlabelled point falls back."""
    ab = sio.load_slp(pool_files[0])
    ab[0].instances[0].set_unlabeled(["A"])
    for anchor in (["A"], "A"):
        train, _ = _centroid_datasets(config, [ab], anchor)
        s = train[0]
        insts = ab[0].instances
        # Instance 0 has no A: the fallback (mean of its located nodes = B).
        np.testing.assert_allclose(
            s["centroids"][0, :2].numpy(),
            np.stack([insts[0].numpy()[1], insts[1].numpy()[0]]),
            atol=1e-4,
        )
        if isinstance(anchor, list):
            torch.testing.assert_close(s["node_weights"], torch.tensor([[1.0]]))
        else:
            assert "node_weights" not in s


def test_file_without_merged_nodes_is_refused(config, tmp_path, pool_files):
    cfg = _frame_config(config, tmp_path, "single_instance", pool_files, "none")
    cfg.model_config.head_configs.single_instance.confmaps.part_names = ["A"]
    cfg.data_config.merge_skeletons = True
    with pytest.raises(ValueError, match=r"training file 1 \(.*bc\.slp\)"):
        ModelTrainer.get_model_trainer_from_config(cfg)


def test_explicit_paf_edges_with_merging(config, tmp_path, pool_files):
    cfg = _frame_config(config, tmp_path, "bottomup", pool_files, "edges")
    cfg.data_config.merge_skeletons = True
    cfg.model_config.head_configs.bottomup.pafs.edges = [["A", "C"]]
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    edges = trainer.skeletons[0].edges
    assert [(e.source.name, e.destination.name) for e in edges] == [("A", "C")]
    train, _ = get_train_val_datasets(
        trainer.train_labels, trainer.val_labels, trainer.config
    )
    for i in range(len(train)):
        # Neither file labels both A and C.
        torch.testing.assert_close(train[i]["edge_weights"], torch.tensor([[0.0]]))


def test_centroid_focal_loss_ignores_zero_weight_frames(config):
    from sleap_nn.training.lightning_modules import CentroidLightningModule
    from sleap_nn.training.losses import compute_centroid_focal_loss

    cfg = _centroid_job(config, None)
    cfg.model_config.head_configs.centroid.confmaps.use_sigmoid_activation = True
    model = CentroidLightningModule(
        model_type="centroid",
        backbone_config=cfg.model_config.backbone_config,
        backbone_type="unet",
        head_configs=cfg.model_config.head_configs,
        init_weights=cfg.model_config.init_weights,
        lr_scheduler=cfg.trainer_config.lr_scheduler,
        centroid_focal_loss_alpha=2.0,
    )
    assert model.centroid_focal_loss_alpha == 2.0
    g = torch.Generator().manual_seed(0)
    y, y_pr = torch.rand(3, 1, 8, 8, generator=g), torch.rand(3, 1, 8, 8, generator=g)
    batch = {"node_weights": torch.tensor([[[1.0]], [[0.0]], [[1.0]]])}
    keep = [0, 2]
    focal = compute_centroid_focal_loss(
        y_pr[keep],
        y[keep],
        alpha=2.0,
        beta=model.centroid_focal_loss_beta,
        pos_threshold=model.centroid_focal_loss_pos_threshold,
        reduction="none",
    ).mean()
    torch.testing.assert_close(model._compute_loss(y_pr, y, batch), focal)
    torch.testing.assert_close(
        model._compute_loss(y_pr, y, batch, stage="val"),
        torch.nn.MSELoss()(y_pr[keep], y[keep]),
    )


def test_tiled_single_instance_ignores_balanced_sampler(config, pool_files, caplog):
    cfg = config.copy()
    del cfg.model_config.head_configs.centered_instance
    cfg.model_config.head_configs.single_instance = {
        "confmaps": {"part_names": ["A", "B"], "sigma": 1.5, "output_stride": 2}
    }
    OmegaConf.update(
        cfg,
        "data_config.preprocessing.tiling",
        {"enabled": True, "tile_size": 64, "overlap": 16},
        merge=True,
    )
    cfg.trainer_config.balanced_sampler = {"counts": [1]}
    ab = sio.load_slp(pool_files[0])
    from loguru import logger

    messages = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        # This minimal tiling section lacks keys the tiled dataset reads later;
        # only the warning, given before the datasets are built, matters here.
        get_train_val_datasets([ab], [ab], cfg)
    except Exception:
        pass
    finally:
        logger.remove(handler)
    assert any("balanced_sampler is ignored for tiled" in m for m in messages)


def test_post_training_centroid_anchor_per_file(pool_files):
    from sleap_nn.data.skeleton_pool import file_anchor_part

    files = [sio.load_slp(p) for p in pool_files]
    union = pool_skeleton(files, list(_ALL_NODES))
    ab, bc = (map_labels_to_skeleton(lb, union) for lb in files)
    assert file_anchor_part(ab, ["C", "A"]) == "A"
    assert file_anchor_part(bc, ["C", "A"]) == "C"
    assert file_anchor_part(ab, ["C"]) is None
    # Unmapped files: their own skeleton's nodes.
    assert file_anchor_part(files[1], ["A", "B"]) == "B"
