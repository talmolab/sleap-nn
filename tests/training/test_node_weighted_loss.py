"""Tests for per-node weighted confmap losses (partially labeled skeletons)."""

import pytest
import sleap_io as sio
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import default_collate

from sleap_nn.data.custom_datasets import CenteredInstanceDataset
from sleap_nn.training.lightning_modules import (
    LightningModel,
    TopDownCenteredInstanceLightningModule,
)
from sleap_nn.training.losses import (
    compute_ohkm_loss,
    compute_weighted_mse_loss,
    compute_weighted_ohkm_loss,
)


def _maps(b=4, c=5, h=16, w=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(b, c, h, w, generator=g)
    y_pr = torch.rand(b, c, h, w, generator=g)
    return y, y_pr


def test_weighted_mse_equals_mse_with_unit_weights():
    y, y_pr = _maps()
    ref = F.mse_loss(y_pr, y)
    assert torch.allclose(compute_weighted_mse_loss(y_pr, y), ref)
    ones = torch.ones(4, 5)
    assert torch.allclose(compute_weighted_mse_loss(y_pr, y, ones), ref)
    # (B, 1, C) as collated from the dataset's (1, n_nodes) also works.
    assert torch.allclose(compute_weighted_mse_loss(y_pr, y, ones[:, None]), ref)


def test_weighted_mse_negative_weights_match_legacy():
    """Sample weights reproduce `_compute_negative_weighted_loss` exactly."""
    y, y_pr = _maps()
    sw = torch.tensor([1.0, 0.3, 1.0, 0.3])
    per_sample = (y_pr - y).pow(2).mean(dim=(1, 2, 3))
    legacy = (per_sample * sw).mean()
    assert torch.allclose(compute_weighted_mse_loss(y_pr, y, None, sw), legacy)
    assert torch.allclose(
        compute_weighted_mse_loss(y_pr, y, torch.ones(4, 5), sw), legacy
    )


def test_weighted_mse_normalizes_by_weight_sum():
    y, y_pr = _maps()
    w = torch.zeros(4, 5)
    w[:, :2] = 1.0
    ref = F.mse_loss(y_pr[:, :2], y[:, :2])
    assert torch.allclose(compute_weighted_mse_loss(y_pr, y, w), ref)
    # All-zero weights: zero loss, no NaN.
    z = compute_weighted_mse_loss(y_pr, y, torch.zeros(4, 5))
    assert z.item() == 0.0


@pytest.mark.parametrize("loss_fn", ["mse", "ohkm"])
def test_zero_gradient_on_unlabeled_channels(loss_fn):
    y, y_pr = _maps()
    y_pr.requires_grad_(True)
    w = torch.ones(4, 5)
    w[0, 1] = 0.0
    w[2, 3] = 0.0
    w[1, :] = 0.0
    if loss_fn == "mse":
        loss = compute_weighted_mse_loss(y_pr, y, w)
    else:
        loss = compute_weighted_ohkm_loss(
            y, y_pr, w, min_hard_keypoints=5, hard_to_easy_ratio=1.0
        )
    loss.backward()
    assert torch.isfinite(y_pr.grad).all()
    g = y_pr.grad.abs().sum(dim=(2, 3))
    assert torch.all(g[w == 0] == 0)
    assert torch.all(g[w > 0] > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(hard_to_easy_ratio=1.2, min_hard_keypoints=1),
        dict(max_hard_keypoints=2),
        dict(hard_to_easy_ratio=100.0, min_hard_keypoints=3),
        dict(min_hard_keypoints=20),
    ],
)
def test_ohkm_per_sample_matches_legacy_for_one_sample(kwargs):
    y, y_pr = _maps(b=1, c=7)
    legacy = compute_ohkm_loss(y, y_pr, **kwargs)
    new = compute_weighted_ohkm_loss(y, y_pr, None, **kwargs)
    assert torch.allclose(legacy, new, rtol=1e-5)


def test_ohkm_per_sample_ranks_within_each_sample():
    """Each sample mines its own worst node; the batch-level version does not."""
    y = torch.zeros(2, 3, 4, 4)
    y_pr = torch.full((2, 3, 4, 4), 0.1)
    y_pr[0, 0] = 2.0  # sample 0: node 0 is hard
    y_pr[1, 2] = 1.0  # sample 1: node 2 is hard
    loss = compute_weighted_ohkm_loss(
        y, y_pr, hard_to_easy_ratio=2.0, min_hard_keypoints=1, loss_scale=1.0
    )
    assert torch.allclose(loss, torch.tensor((4.0 + 1.0) / 2))
    legacy = compute_ohkm_loss(
        y, y_pr, hard_to_easy_ratio=2.0, min_hard_keypoints=1, loss_scale=1.0
    )
    assert not torch.allclose(loss, legacy)


def test_ohkm_finite_with_perfect_channel():
    y, y_pr = _maps(b=2, c=4)
    y_pr = y_pr.clone()
    y_pr[:, 0] = y[:, 0]  # zero loss channel: easiest loss == 0
    assert torch.isfinite(compute_weighted_ohkm_loss(y, y_pr))
    # A sample with no weighted node contributes nothing.
    w = torch.ones(2, 4)
    w[1] = 0.0
    one = compute_weighted_ohkm_loss(y[:1], y_pr[:1])
    assert torch.allclose(compute_weighted_ohkm_loss(y, y_pr, w), one)
    assert compute_weighted_ohkm_loss(y, y_pr, torch.zeros(2, 4)).item() == 0.0


def test_ohkm_mixed_dataset_batch():
    """Per-sample ranking only over each sample's labeled nodes."""
    b, c, h, w = 4, 6, 8, 8
    y = torch.zeros(b, c, h, w)
    y_pr = torch.zeros(b, c, h, w)
    # Dataset A (samples 0, 1) labels nodes 0-2; dataset B (2, 3) nodes 3-5.
    wts = torch.zeros(b, c)
    wts[:2, :3] = 1.0
    wts[2:, 3:] = 1.0
    # Huge error on unlabeled channels: must be ignored by the masked loss.
    y_pr[:2, 3:] = 10.0
    y_pr[2:, :3] = 10.0
    # One distinct hard node per sample among its labeled nodes.
    hard = {0: 0, 1: 2, 2: 4, 3: 5}
    for s, n in hard.items():
        y_pr[s, n] = 1.0
    # Small error everywhere else that is labeled (so ratios are finite).
    y_pr = y_pr + 0.01 * wts[:, :, None, None]

    loss = compute_weighted_ohkm_loss(
        y, y_pr, wts, hard_to_easy_ratio=2.0, min_hard_keypoints=1, loss_scale=1.0
    )
    expected = torch.stack(
        [((y_pr[s, n] - y[s, n]) ** 2).mean() for s, n in hard.items()]
    ).mean()
    assert torch.allclose(loss, expected)

    # Legacy ranking (sum over batch, unmasked) picks the unlabeled channels.
    legacy = compute_ohkm_loss(
        y, y_pr, hard_to_easy_ratio=2.0, min_hard_keypoints=1, loss_scale=1.0
    )
    assert legacy > 10 * loss

    # Masked MSE ignores the unlabeled channels entirely.
    mse = compute_weighted_mse_loss(y_pr, y, wts)
    ref = (((y_pr - y) ** 2).mean(dim=(2, 3)) * wts).sum() / wts.sum()
    assert torch.allclose(mse, ref)
    assert mse < 1.0


def _labels(minimal_instance, labeled_nodes=None):
    labels = sio.load_slp(minimal_instance)
    if labeled_nodes is not None:
        labels.provenance["labeled_nodes"] = labeled_nodes
    return labels


def _centered_dataset(labels_list):
    return CenteredInstanceDataset(
        max_stride=8,
        scale=1.0,
        confmap_head_config=DictConfig(
            {"sigma": 1.5, "output_stride": 2, "anchor_part": None}
        ),
        crop_size=160,
        labels=labels_list,
    )


def _ci_model(config, **kwargs):
    model = TopDownCenteredInstanceLightningModule(
        model_type="centered_instance",
        backbone_config=config.model_config.backbone_config,
        backbone_type="unet",
        head_configs=config.model_config.head_configs,
        init_weights=config.model_config.init_weights,
        lr_scheduler=config.trainer_config.lr_scheduler,
        **kwargs,
    )
    logged = {}
    model.log = lambda name, value, **k: logged.__setitem__(name, value)
    return model, logged


def test_masked_loss_without_weights_is_byte_identical(config, minimal_instance):
    """No `node_weights` in the batch: exactly today's loss, bit for bit."""
    model, logged = _ci_model(config)
    y, y_pr = _maps(b=3, c=2)
    assert torch.equal(
        model._compute_masked_confmap_loss(y_pr, y, {}), torch.nn.MSELoss()(y_pr, y)
    )
    # Negative-frame weighting path (single-instance / bottom-up) too.
    model.negative_loss_weight = 0.5
    batch = {"is_negative": torch.tensor([True, False, False])}
    for stage in ("train", "val"):
        assert torch.equal(
            model._compute_masked_confmap_loss(y_pr, y, batch, stage=stage),
            model._compute_negative_weighted_loss(y_pr, y, batch, stage=stage),
        )
    # And the training step on a fully labeled dataset.
    batch = default_collate([_centered_dataset([_labels(minimal_instance)])[0]] * 2)
    assert "node_weights" not in batch
    with torch.no_grad():
        preds = model(batch["instance_image"])
        loss = model.training_step(batch, 0)
        assert torch.equal(
            loss, torch.nn.MSELoss()(preds, batch["confidence_maps"].squeeze(1))
        )
        model.validation_step(batch, 0)
    assert "val/loss_unmasked" not in logged


def test_node_weights_width_mismatch_raises(config):
    model, _ = _ci_model(config)
    y, y_pr = _maps(b=2, c=3)
    with pytest.raises(ValueError, match="node_weights"):
        model._compute_masked_confmap_loss(
            y_pr, y, {"node_weights": torch.ones(2, 1, 2)}
        )


def test_centered_instance_module_masks_unlabeled_nodes(config, minimal_instance):
    ds = _centered_dataset([_labels(minimal_instance, labeled_nodes=["A"])])
    batch = default_collate([ds[0], ds[1]])
    assert batch["node_weights"].shape == (2, 1, 2)
    assert torch.equal(batch["node_weights"][:, 0], torch.tensor([[1.0, 0.0]] * 2))

    model, _ = _ci_model(config, online_mining=True, min_hard_keypoints=1)
    y = batch["confidence_maps"].squeeze(1)
    preds = model(batch["instance_image"])
    loss = model.training_step(batch, 0)
    mse_a = F.mse_loss(preds[:, :1], y[:, :1])
    # Masked MSE + per-sample OHKM over the single labeled node (k=1 per sample).
    assert torch.allclose(loss, mse_a + 5.0 * mse_a, rtol=1e-4)

    model.zero_grad()
    loss.backward()
    head = [p for n, p in model.model.named_parameters() if "head" in n]
    w_last = [p for p in head if p.ndim == 4][-1]
    assert w_last.shape[0] == 2
    assert torch.all(w_last.grad[1] == 0)
    assert w_last.grad[0].abs().sum() > 0


def test_mixed_dataset_batch_with_ohkm(config, minimal_instance):
    """Two files labeling disjoint node sets, one batch, per-sample OHKM."""
    ds = _centered_dataset(
        [
            _labels(minimal_instance, labeled_nodes=["A"]),
            _labels(minimal_instance, labeled_nodes=["B"]),
        ]
    )
    batch = default_collate([ds[i] for i in range(len(ds))])
    w = batch["node_weights"][:, 0]
    assert w.shape == (4, 2)
    assert {tuple(r.tolist()) for r in w} == {(1.0, 0.0), (0.0, 1.0)}

    model, logged = _ci_model(
        config, online_mining=True, min_hard_keypoints=2, loss_scale=5.0
    )
    y = batch["confidence_maps"].squeeze(1)
    with torch.no_grad():
        preds = model(batch["instance_image"])
        loss = model.training_step(batch, 0)
        mse_bc = ((preds - y) ** 2).mean(dim=(2, 3))
        masked = (mse_bc * w).sum() / w.sum()
        # min_hard_keypoints=2 is capped at the 1 labeled node per sample.
        assert torch.allclose(loss, masked + 5.0 * masked, rtol=1e-4)

        model.validation_step(batch, 0)
        assert torch.allclose(logged["val/loss"], loss, rtol=1e-4)
        assert torch.allclose(logged["val/loss_unmasked"], F.mse_loss(preds, y))


def test_ohkm_per_sample_false_uses_legacy(config, minimal_instance):
    ds = _centered_dataset([_labels(minimal_instance, labeled_nodes=["A"])])
    batch = default_collate([ds[0], ds[1]])
    model, _ = _ci_model(config, online_mining=True)
    model.ohkm_per_sample = False
    y = batch["confidence_maps"].squeeze(1)
    with torch.no_grad():
        preds = model(batch["instance_image"])
        assert torch.equal(
            model._compute_ohkm(preds, y, batch), compute_ohkm_loss(y, preds)
        )
    assert model._warned_legacy_ohkm


def test_lightning_model_reads_per_sample_from_config(config):
    assert LightningModel.ohkm_per_sample  # default for directly built models
    model = LightningModel.get_lightning_model_from_config(config)
    assert model.ohkm_per_sample  # key absent from the config: default True
    OmegaConf.update(
        config, "trainer_config.online_hard_keypoint_mining.per_sample", False
    )
    model = LightningModel.get_lightning_model_from_config(config)
    assert not model.ohkm_per_sample


def test_train_with_labeled_nodes_and_auto_split(config, minimal_instance, tmp_path):
    """No val file: the automatic split keeps the file's node mask."""
    from sleap_nn.training.model_trainer import ModelTrainer

    path = (tmp_path / "partial.pkg.slp").as_posix()
    _labels(minimal_instance, labeled_nodes=["A"]).save(path, embed=True)

    OmegaConf.update(config, "data_config.train_labels_path", [path])
    OmegaConf.update(config, "data_config.val_labels_path", None)
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "partial_labels")
    OmegaConf.update(config, "trainer_config.max_epochs", 1)
    OmegaConf.update(config, "trainer_config.trainer_accelerator", "cpu")
    OmegaConf.update(
        config, "trainer_config.online_hard_keypoint_mining.online_mining", True
    )
    trainer = ModelTrainer.get_model_trainer_from_config(config)
    for split in trainer.train_labels + trainer.val_labels:
        assert split.provenance["labeled_nodes"] == ["A"]
    trainer.train()
    assert trainer.lightning_model.ohkm_per_sample
    metrics = trainer.trainer.callback_metrics
    assert "val/loss" in metrics and "val/loss_unmasked" in metrics
    assert torch.isfinite(metrics["val/loss"])


def test_group_split_keeps_labeled_nodes(config, minimal_instance, monkeypatch):
    """The group-aware splitter also builds fresh provenance; it is carried over."""
    import sleap_nn.data.splitting as splitting
    from sleap_nn.training.model_trainer import ModelTrainer

    def fake_split(labels_list, split_config):
        # Same frames on both sides (one-frame fixture), provenance dropped.
        fresh = [sio.Labels(labeled_frames=list(x.labeled_frames)) for x in labels_list]
        return fresh, [
            sio.Labels(labeled_frames=list(x.labeled_frames)) for x in labels_list
        ]

    monkeypatch.setattr(splitting, "split_labels_list_train_val", fake_split)
    OmegaConf.update(config, "data_config.val_labels_path", None)
    OmegaConf.update(
        config,
        "data_config.split",
        {"split_by": "frame", "n_folds": 2, "fold": 0, "seed": 0},
    )
    trainer = ModelTrainer.get_model_trainer_from_config(
        config,
        train_labels=[_labels(minimal_instance, labeled_nodes=["B"])],
    )
    for split in trainer.train_labels + trainer.val_labels:
        assert split.provenance["labeled_nodes"] == ["B"]


def _frame_level_config(config, model_type):
    """Turn the centered-instance `config` fixture into single-instance/bottom-up."""
    cfg = config.copy()
    head = cfg.model_config.head_configs.centered_instance
    del cfg.model_config.head_configs.centered_instance
    OmegaConf.update(cfg, f"model_config.head_configs.{model_type}", head)
    heads = cfg.model_config.head_configs[model_type]
    del heads.confmaps.anchor_part
    if model_type == "bottomup":
        heads.confmaps.loss_weight = 1.0
        heads["pafs"] = {
            "edges": [("A", "B")],
            "sigma": 4,
            "output_stride": 4,
            "loss_weight": 1.0,
        }
    OmegaConf.update(
        cfg, "trainer_config.online_hard_keypoint_mining.online_mining", True
    )
    OmegaConf.update(
        cfg, "trainer_config.online_hard_keypoint_mining.min_hard_keypoints", 1
    )
    OmegaConf.update(cfg, "data_config.negative_loss_weight", 0.5)
    return cfg


@pytest.mark.parametrize("model_type", ["single_instance", "bottomup"])
def test_frame_level_modules_mask_unlabeled_nodes(config, minimal_instance, model_type):
    """Single-instance / bottom-up steps: masked confmap loss + per-sample OHKM."""
    from sleap_nn.data.custom_datasets import BottomUpDataset, SingleInstanceDataset

    labels = _labels(minimal_instance, labeled_nodes=["A"])
    lf = labels[0]
    labels.append(sio.LabeledFrame(video=lf.video, frame_idx=0, is_negative=True))
    head = DictConfig({"sigma": 1.5, "output_stride": 2})
    geo = DictConfig({"flip_p": 1.0, "affine_p": 0.0, "erase_p": 0.0, "mixup_p": 0.0})
    common = dict(
        labels=[labels],
        confmap_head_config=head,
        max_stride=8,
        use_negative_frames=True,
        apply_aug=True,
        geometric_aug=geo,
    )
    if model_type == "single_instance":
        lf.instances = lf.instances[:1]
        ds = SingleInstanceDataset(**common)
        key = "SingleInstanceConfmapsHead"
    else:
        ds = BottomUpDataset(
            pafs_head_config=DictConfig({"sigma": 4, "output_stride": 4}), **common
        )
        key = "MultiInstanceConfmapsHead"
    samples = [ds[i] for i in range(len(ds))]
    # No symmetries: the flip leaves the weights in place.
    weights = {bool(s["is_negative"]): s["node_weights"] for s in samples}
    assert torch.equal(weights[False], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(weights[True], torch.ones(1, 2))
    batch = default_collate(samples)

    model = LightningModel.get_lightning_model_from_config(
        _frame_level_config(config, model_type)
    )
    logged = {}
    model.log = lambda name, value, **k: logged.__setitem__(name, value)
    with torch.no_grad():
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss)
        model.validation_step(batch, 0)
        assert torch.isfinite(logged["val/loss"])
    if model_type == "single_instance":
        assert "val/loss_unmasked" in logged

    # The confmap term matches the weighted MSE (negative frame scaled by 0.5).
    from sleap_nn.data.normalization import normalize_on_gpu

    X = normalize_on_gpu(batch["image"].squeeze(1))
    y = batch["confidence_maps"].squeeze(1)
    with torch.no_grad():
        preds = model.model(X)[key]
        sw = torch.where(batch["is_negative"], 0.5, 1.0)
        expected = compute_weighted_mse_loss(preds, y, batch["node_weights"], sw)
        got = model._compute_masked_confmap_loss(preds, y, batch, stage="train")
        assert torch.allclose(got, expected)
