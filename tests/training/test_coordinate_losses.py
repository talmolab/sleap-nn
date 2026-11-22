"""Tests for the experimental coordinate losses (`trainer_config.coordinate_losses`)."""

from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.config.trainer_config import CoordinateLossesConfig
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.coordinate_losses import (
    CoordinateLosses,
    KeypointAutoencoder,
    build_coordinate_losses,
    check_not_saved_decoded_config,
    consistency_loss,
    dataset_config,
    decoded_loss,
    label_node_names,
    latent_names,
    load_keypoint_autoencoder,
    setup_coordinate_losses,
    soft_argmax,
)
from sleap_nn.training.lightning_modules import LightningModel

HEAD = "CenteredInstanceConfmapsHead"


def _save_ae(path, w_enc, w_dec, names, latent_partner=None, **extra):
    arrays = dict(
        W_enc=np.asarray(w_enc, np.float32),
        W_dec=np.asarray(w_dec, np.float32),
        names=np.asarray(names),
        **extra,
    )
    if latent_partner is not None:
        arrays["latent_partner"] = np.asarray(latent_partner, np.int64)
    np.savez(path, **arrays)
    return str(path)


@pytest.fixture
def midpoint_ae(tmp_path):
    """Two channels A, B and one latent point at their midpoint."""
    return _save_ae(tmp_path / "mid.npz", [[0.5, 0.5]], [[1.0], [1.0]], ["A", "B"])


@pytest.fixture
def latent_ae(tmp_path):
    """Channels A, B, C; two latent points (A and B), C = their midpoint."""
    return _save_ae(
        tmp_path / "lat.npz",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        ["A", "B", "C"],
        latent_partner=[1, 0],
        latent_side=np.asarray(["L", "R"]),
    )


def _gaussian_maps(points, h, w, stride, sigma=1.0):
    """``(1, K, h, w)`` unit-height Gaussians at ``points`` (input px)."""
    ys = torch.arange(h, dtype=torch.float32)[:, None] * stride
    xs = torch.arange(w, dtype=torch.float32)[None, :] * stride
    maps = [
        torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * (sigma * stride) ** 2))
        for x, y in points
    ]
    return torch.stack(maps)[None]


# ---------------------------------------------------------------- soft-argmax


def test_soft_argmax_recovers_known_peaks():
    pts = [(10.3, 20.7), (50.0, 3.2), (33.6, 60.1)]
    maps = _gaussian_maps(pts, 40, 40, stride=2, sigma=2.5)
    out = soft_argmax(maps, output_stride=2, beta=20.0)
    assert out.shape == (1, 3, 2)
    assert torch.allclose(out[0], torch.tensor(pts), atol=0.05)
    # Narrow peaks: a large beta over the whole map rounds towards the grid, a
    # window around the maximum allows a smaller beta.
    narrow = _gaussian_maps(pts, 40, 40, stride=2, sigma=1.5)
    windowed = soft_argmax(narrow, output_stride=2, beta=10.0, window=5)
    assert torch.allclose(windowed[0], torch.tensor(pts), atol=0.1)
    assert not torch.allclose(
        soft_argmax(narrow, output_stride=2, beta=10.0)[0], torch.tensor(pts), atol=1
    )
    # A single hot cell with a large beta lands on that cell's input pixel.
    hot = torch.zeros(1, 1, 8, 8)
    hot[0, 0, 5, 2] = 1.0
    assert torch.allclose(
        soft_argmax(hot, output_stride=4, beta=100.0), torch.tensor([[[8.0, 20.0]]])
    )
    # Uniform maps give the grid centre (stride 1: (w - 1) / 2, (h - 1) / 2).
    flat = soft_argmax(torch.zeros(2, 1, 5, 9))
    assert torch.allclose(flat, torch.tensor([4.0, 2.0]).expand(2, 1, 2))


def test_soft_argmax_is_differentiable_and_float32():
    maps = _gaussian_maps([(6.0, 7.0)], 16, 16, 1).to(torch.bfloat16)
    maps.requires_grad_(True)
    out = soft_argmax(maps, beta=5.0)
    assert out.dtype == torch.float32
    out[0, 0, 0].backward()
    assert maps.grad is not None and maps.grad.abs().sum() > 0
    # Moving mass to the right raises x: the gradient is larger on the right.
    g = maps.grad[0, 0].float()
    assert g[:, 12].sum() > g[:, 2].sum()


# --------------------------------------------------------------- loss terms


def test_consistency_loss_value_and_gradient():
    proj = torch.tensor([[0.5, 0.5], [0.5, 0.5]])  # W_dec W_enc of the midpoint AE
    p = torch.tensor([[[0.0, 0.0], [2.0, 4.0]]], requires_grad=True)
    loss = consistency_loss(p, proj)
    # Residuals (-1, -2) and (1, 2): |.|_1 = 3 per channel.
    assert loss.item() == pytest.approx(3.0)
    loss.backward()
    expected = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]]) * 0.5  # d/dP of the mean
    assert torch.allclose(p.grad, expected)
    # Points the autoencoder reproduces cost nothing.
    same = torch.tensor([[[3.0, 1.0], [3.0, 1.0]]])
    assert consistency_loss(same, proj).item() == 0.0
    # Sample weights leave samples out.
    two = torch.cat([p.detach(), same])
    assert consistency_loss(two, proj).item() == pytest.approx(1.5)
    assert consistency_loss(two, proj, torch.tensor([1.0, 0.0])).item() == (
        pytest.approx(3.0)
    )
    assert torch.autograd.gradcheck(
        lambda x: consistency_loss(x, proj.double()),
        (
            torch.tensor(
                [[[0.1, 0.7], [2.3, 4.1]]], dtype=torch.float64
            ).requires_grad_(),
        ),
    )


def test_decoded_loss_value_weights_and_gradient():
    dec = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    q = torch.tensor([[[0.0, 0.0], [4.0, 2.0]]], requires_grad=True)
    # Decoded: (0, 0), (4, 2), (2, 1).
    y = torch.tensor([[[1.0, 0.0], [4.0, 2.0], [float("nan"), 1.0]]])
    loss = decoded_loss(q, dec, y)
    # C has no finite target: mean over A (|1|) and B (0) = 0.5.
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.allclose(q.grad, torch.tensor([[[-0.5, 0.0], [0.0, 0.0]]]))
    # Per-channel weights: A off, C labelled.
    y2 = torch.tensor([[[1.0, 0.0], [4.0, 2.0], [3.0, 1.0]]])
    w = torch.tensor([[0.0, 1.0, 1.0]])
    assert decoded_loss(q.detach(), dec, y2, w).item() == pytest.approx(0.5)
    # Nothing labelled: zero loss with a zero (finite) gradient.
    q.grad = None
    zero = decoded_loss(q, dec, y2, torch.zeros(1, 3))
    zero.backward()
    assert zero.item() == 0.0 and (q.grad == 0).all()
    assert torch.autograd.gradcheck(
        lambda x: decoded_loss(x, dec.double(), y2.double(), w.double()),
        (
            torch.tensor(
                [[[0.2, 0.3], [3.1, 2.4]]], dtype=torch.float64
            ).requires_grad_(),
        ),
    )


def test_coordinate_losses_module_reads_the_batch():
    dec = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], np.float32)
    module = CoordinateLosses("decoded", 0.1, dec, output_stride=2, beta=20.0)
    assert module.state_dict() == {}  # frozen matrix stays out of checkpoints
    maps = _gaussian_maps([(10.0, 6.0), (20.0, 14.0)], 16, 16, stride=2)
    maps = maps.repeat(2, 1, 1, 1)
    batch = {
        # Decoded: (10, 6), (20, 14), (15, 10).
        "instance": torch.tensor([[[[11.0, 6.0], [20.0, 14.0], [15.0, 10.0]]]]).repeat(
            2, 1, 1, 1
        ),
        "node_weights": torch.tensor([[[1.0, 1.0, 0.0]], [[0.0, 1.0, 1.0]]]),
        "is_negative": torch.tensor([False, True]),
    }
    term = module(maps, batch)
    # Sample 2 is a negative; sample 1 counts A (off by 1) and B (0).
    assert term.item() == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError, match="keypoint channels"):
        module(maps, {"instance": torch.zeros(2, 1, 2, 2)})


def test_term_under_bf16_autocast_is_float32():
    proj = np.full((2, 2), 0.5, np.float32)
    module = CoordinateLosses("consistency", 1.0, proj, output_stride=2)
    maps = _gaussian_maps([(10.0, 6.0), (20.0, 14.0)], 16, 16, stride=2)
    maps = maps.to(torch.bfloat16).requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        term = module(maps, {})
    assert term.dtype == torch.float32
    term.backward()
    assert torch.isfinite(maps.grad.float()).all() and maps.grad.abs().sum() > 0


# ------------------------------------------------------------- autoencoder


def test_load_autoencoder_format(latent_ae, tmp_path):
    ae = load_keypoint_autoencoder(latent_ae)
    assert ae.n_latent == 2 and ae.names == ["A", "B", "C"]
    assert ae.latent_names == ["latent_00", "latent_01"]
    assert ae.latent_partner.tolist() == [1, 0]
    skel = ae.latent_skeleton()
    assert skel.node_names == ["latent_00", "latent_01"]
    assert len(skel.symmetries) == 1
    assert latent_names(120)[-1] == "latent_119"

    def bad(name, **kw):
        args = dict(w_enc=[[0.5, 0.5]], w_dec=[[1.0], [1.0]], names=["A", "B"], **kw)
        return _save_ae(tmp_path / name, args.pop("w_enc"), args.pop("w_dec"), **args)

    with pytest.raises(ValueError, match="sum to 1"):
        load_keypoint_autoencoder(
            _save_ae(tmp_path / "s.npz", [[1.0, 1.0]], [[1.0], [1.0]], ["A", "B"])
        )
    with pytest.raises(ValueError, match="W_dec must be"):
        load_keypoint_autoencoder(
            _save_ae(tmp_path / "t.npz", [[0.5, 0.5]], [[1.0, 0.0]], ["A", "B"])
        )
    with pytest.raises(ValueError, match="names for"):
        load_keypoint_autoencoder(
            _save_ae(tmp_path / "n.npz", [[0.5, 0.5]], [[1.0], [1.0]], ["A"])
        )
    with pytest.raises(ValueError, match="latent_partner"):
        load_keypoint_autoencoder(bad("p.npz", latent_partner=[3]))
    np.savez(tmp_path / "k.npz", W_enc=np.ones((1, 2)) / 2)
    with pytest.raises(ValueError, match="lacks"):
        load_keypoint_autoencoder(tmp_path / "k.npz")
    with pytest.raises(FileNotFoundError):
        load_keypoint_autoencoder(tmp_path / "missing.npz")


# ------------------------------------------------------------------ config


def test_config_block_validation(midpoint_ae):
    with pytest.raises(ValueError, match="needs `autoencoder`"):
        CoordinateLossesConfig(consistency=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        CoordinateLossesConfig(autoencoder=midpoint_ae)
    with pytest.raises(ValueError, match="exactly one"):
        CoordinateLossesConfig(autoencoder=midpoint_ae, consistency=1.0, decoded=1.0)
    cfg = CoordinateLossesConfig(autoencoder=midpoint_ae, consistency=0.1)
    assert cfg.soft_argmax_beta == 20.0 and cfg.data_head is None


def _setup_cfg(ae, part_names, flip_p=0.0, model_type="centered_instance", **block):
    heads = {
        "centered_instance": None,
        "centroid": None,
    }
    heads[model_type] = {
        "confmaps": {
            "part_names": list(part_names),
            "anchor_part": part_names[0],
            "centroid_method": None,
            "centroid_fallback": None,
            "sigma": 1.5,
            "output_stride": 2,
        }
    }
    return OmegaConf.create(
        {
            "data_config": {
                "use_augmentations_train": True,
                "augmentation_config": {"geometric": {"flip_p": flip_p}},
                "skeletons": [],
            },
            "model_config": {"head_configs": heads},
            "trainer_config": {
                "coordinate_losses": {"autoencoder": ae, **block},
                "eval": {"enabled": False},
                "use_wandb": False,
            },
        }
    )


def _skeleton(names, symmetries=()):
    return sio.Skeleton(nodes=list(names), symmetries=[list(s) for s in symmetries])


def test_setup_consistency_checks_names_and_flips(midpoint_ae, latent_ae, tmp_path):
    skel = _skeleton(["A", "B"], [("A", "B")])
    cfg = _setup_cfg(midpoint_ae, ["B", "A"], flip_p=0.5, consistency=0.1)
    assert setup_coordinate_losses(cfg, [skel]) == [skel]
    with pytest.raises(ValueError, match="part names must be the autoencoder"):
        setup_coordinate_losses(
            _setup_cfg(latent_ae, ["A", "B"], consistency=0.1), [skel]
        )
    # A projection that is not symmetric under the A <-> B swap.
    lopsided = _save_ae(tmp_path / "lop.npz", [[0.8, 0.2]], [[1.0], [1.0]], ["A", "B"])
    cfg = _setup_cfg(lopsided, ["A", "B"], flip_p=0.5, consistency=0.1)
    with pytest.raises(ValueError, match="does not commute"):
        setup_coordinate_losses(cfg, [skel])
    # Without flips it is accepted.
    cfg = _setup_cfg(lopsided, ["A", "B"], flip_p=0.0, consistency=0.1)
    assert setup_coordinate_losses(cfg, [skel]) == [skel]
    with pytest.raises(ValueError, match="only centered_instance"):
        setup_coordinate_losses(
            _setup_cfg(midpoint_ae, ["A", "B"], model_type="centroid", consistency=1),
            [skel],
        )


def test_setup_decoded_resolves_the_latent_head(latent_ae, tmp_path):
    skel = _skeleton(["A", "B"], [("A", "B")])
    cfg = _setup_cfg(latent_ae, ["A", "B"], flip_p=0.5, decoded=0.01)
    out = setup_coordinate_losses(cfg, [skel])
    assert out[0].node_names == ["latent_00", "latent_01"]
    head = cfg.model_config.head_configs.centered_instance.confmaps
    assert list(head.part_names) == ["latent_00", "latent_01"]
    assert head.anchor_part is None
    assert [n["name"] for n in cfg.data_config.skeletons[0]["nodes"]] == [
        "latent_00",
        "latent_01",
    ]
    data_head = cfg.trainer_config.coordinate_losses.data_head
    assert list(data_head.part_names) == ["A", "B"] and data_head.anchor_part == "A"
    # The data loader sees the labelled channels again.
    view = dataset_config(cfg)
    vhead = view.model_config.head_configs.centered_instance.confmaps
    assert list(vhead.part_names) == ["A", "B"] and vhead.anchor_part == "A"
    assert list(head.part_names) == ["latent_00", "latent_01"]  # unchanged
    # The module decodes into the batch's channel order.
    module = build_coordinate_losses(cfg)
    assert module.mode == "decoded" and module.weight == 0.01
    assert module.matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]

    # A labelled channel the autoencoder does not know.
    with pytest.raises(ValueError, match="not autoencoder channels"):
        setup_coordinate_losses(
            _setup_cfg(latent_ae, ["A", "Z"], decoded=0.01), [_skeleton(["A", "Z"])]
        )
    # A decoder that no latent swap makes symmetric under A <-> B.
    skew = _save_ae(
        tmp_path / "skew.npz",
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.3, 0.7]],
        ["A", "B"],
        latent_partner=[1, 0],
    )
    with pytest.raises(ValueError, match="no exact latent target"):
        setup_coordinate_losses(_setup_cfg(skew, ["A", "B"], 0.5, decoded=1), [skel])
    # Options the latent head cannot run with.
    for key, value, match in [
        ("eval", {"enabled": True}, "eval.enabled"),
        ("visualize_preds_during_training", True, "visualize_preds"),
        ("distillation", {"teacher": "x"}, "distillation"),
    ]:
        cfg = _setup_cfg(latent_ae, ["A", "B"], decoded=0.01)
        cfg.trainer_config[key] = value
        with pytest.raises(ValueError, match=match):
            setup_coordinate_losses(cfg, [skel])
    for monitor in ("val/macro/loss", "val/fileA/loss"):
        cfg = _setup_cfg(latent_ae, ["A", "B"], decoded=0.01)
        cfg.trainer_config.model_ckpt = {"monitor": monitor}
        with pytest.raises(ValueError, match="no per-file"):
            setup_coordinate_losses(cfg, [skel])
    # The W&B gallery stays on (it draws the decoded points); post-training
    # evaluation is turned off.
    cfg = _setup_cfg(latent_ae, ["A", "B"], decoded=0.01)
    cfg.trainer_config.use_wandb = True
    setup_coordinate_losses(cfg, [skel])
    assert list(cfg.trainer_config.eval.post_train_splits) == []
    assert label_node_names(cfg, [skel]) == ["A", "B"]
    with pytest.raises(ValueError, match="data_head is set"):
        check_not_saved_decoded_config(cfg)


def test_unset_block_is_dropped_and_changes_nothing(config):
    cfg = verify_training_cfg(config)
    assert "coordinate_losses" in cfg.trainer_config  # schema default (null)
    initial = cfg.copy()
    skel = _skeleton(["A", "B"])
    assert setup_coordinate_losses(cfg, [skel], initial) == [skel]
    assert "coordinate_losses" not in cfg.trainer_config
    assert "coordinate_losses" not in initial.trainer_config
    assert dataset_config(cfg) is cfg
    assert build_coordinate_losses(cfg) is None


# -------------------------------------------------------- lightning module


def _ci_cfg(ckpt_dir):
    cfg = OmegaConf.load(Path(ckpt_dir) / "training_config.yaml")
    cfg.model_config.backbone_config.unet.filters = 8
    return cfg


def test_training_step_adds_the_consistency_term(
    minimal_instance_centered_instance_ckpt, midpoint_ae
):
    cfg = _ci_cfg(minimal_instance_centered_instance_ckpt)
    torch.manual_seed(0)
    plain = LightningModel.get_lightning_model_from_config(cfg)
    assert plain.coord_losses is None
    cfg = verify_training_cfg(cfg)
    cfg.trainer_config.coordinate_losses = OmegaConf.structured(
        CoordinateLossesConfig(autoencoder=midpoint_ae, consistency=0.5)
    )
    torch.manual_seed(0)
    lm = LightningModel.get_lightning_model_from_config(cfg)
    assert isinstance(lm.coord_losses, CoordinateLosses)
    assert lm.state_dict().keys() == plain.state_dict().keys()
    logged = {}
    lm.log = lambda name, value, **kw: logged.__setitem__(name, value)
    plain.log = lambda *a, **kw: None
    g = torch.Generator().manual_seed(0)
    batch = {
        "instance_image": torch.rand(2, 1, 1, 64, 64, generator=g),
        "confidence_maps": torch.rand(2, 1, 2, 32, 32, generator=g),
    }
    total = lm.training_step(batch, 0)
    supervised = plain.training_step(batch, 0)
    term = logged["train/coord_consistency"]
    assert term.item() > 0
    assert total.item() == pytest.approx(supervised.item() + 0.5 * term.item(), 1e-5)
    assert logged["loss"].item() == pytest.approx(supervised.item(), 1e-6)
    total.backward()
    head = lm.model.head_layers[0]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    lm.validation_step(batch, 0)
    assert "val/coord_consistency" in logged


# --------------------------------------------------------------- end to end


def _e2e_config(config, tmp_path, run_name, block=None):
    cfg = config.copy()
    cfg.data_config.preprocessing.crop_size = 64
    cfg.model_config.backbone_config.unet.filters = 8
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = run_name
    cfg.trainer_config.trainer_accelerator = "cpu"
    if block is not None:
        cfg.trainer_config.coordinate_losses = block
    return cfg


def _plain_reload_and_predict(run_dir, minimal_instance):
    """The run's checkpoint loads strictly into a plain model and predicts."""
    from sleap_nn.inference.run import predict

    saved = OmegaConf.load(run_dir / "training_config.yaml")
    plain = LightningModel.get_lightning_model_from_config(saved)
    best = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    plain.load_state_dict(best["state_dict"], strict=True)
    labels = predict(
        str(minimal_instance),
        model_paths=[str(run_dir)],
        device="cpu",
        # Two training steps leave plain maps below 0; decoded-term maps have no
        # fixed height at all, so their documented threshold is very low.
        peak_threshold=-1e9,
    )
    insts = [i for lf in labels for i in lf.predicted_instances]
    assert insts
    return saved, insts


def test_train_with_consistency_term(config, tmp_path, minimal_instance, midpoint_ae):
    from sleap_nn.training.model_trainer import ModelTrainer

    block = {"autoencoder": midpoint_ae, "consistency": 0.01}
    cfg = _e2e_config(config, tmp_path, "cons", block)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    assert trainer.lightning_model.coord_losses.mode == "consistency"
    metrics = trainer.trainer.callback_metrics
    assert {"train/coord_consistency", "val/coord_consistency", "val/loss"} <= set(
        metrics
    )
    assert all(torch.isfinite(metrics[k]) for k in metrics if "coord" in k)
    saved, insts = _plain_reload_and_predict(tmp_path / "cons", minimal_instance)
    assert saved.trainer_config.coordinate_losses.consistency == 0.01
    assert list(insts[0].skeleton.node_names) == ["A", "B"]


def test_train_with_decoded_term(config, tmp_path, minimal_instance, latent_ae):
    from sleap_nn.training.model_trainer import ModelTrainer

    block = {"autoencoder": latent_ae, "decoded": 0.01}
    cfg = _e2e_config(config, tmp_path, "dec", block)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    lm = trainer.lightning_model
    assert lm.coord_losses.mode == "decoded"
    # The head predicts the two latent points.
    assert lm.model(torch.zeros(1, 1, 64, 64))[HEAD].shape[1] == 2
    metrics = trainer.trainer.callback_metrics
    assert {"train/coord_decoded", "val/coord_decoded", "val/loss"} <= set(metrics)
    assert metrics["val/loss"].item() == pytest.approx(
        0.01 * metrics["val/coord_decoded"].item(), rel=1e-5
    )
    # The batches carried the labelled channels A, B.
    assert trainer.trainer.train_dataloader.dataset[0]["instance"].shape[-2] == 2
    saved, insts = _plain_reload_and_predict(tmp_path / "dec", minimal_instance)
    head = saved.model_config.head_configs.centered_instance.confmaps
    assert list(head.part_names) == ["latent_00", "latent_01"]
    assert head.anchor_part is None
    assert list(saved.trainer_config.coordinate_losses.data_head.part_names) == [
        "A",
        "B",
    ]
    assert list(insts[0].skeleton.node_names) == ["latent_00", "latent_01"]


def test_train_with_decoded_term_warp_crops(
    config, tmp_path, minimal_instance, latent_ae
):
    """The warp data path, which reads its channels from the head config."""
    from sleap_nn.training.model_trainer import ModelTrainer

    block = {"autoencoder": latent_ae, "decoded": 0.01}
    cfg = _e2e_config(config, tmp_path, "decw", block)
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [64, 48]
    pre.gpu_confmaps = True
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.anchor_part = None
    head.centroid_method = "bbox_center"
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    sample = trainer.trainer.val_dataloaders.dataset[0]  # no augmentation
    assert sample["instance"].shape[-2] == 2
    assert torch.isfinite(sample["instance"]).all()
    assert torch.isfinite(trainer.trainer.callback_metrics["train/coord_decoded"])


def test_old_config_trains_and_saves_as_before(config, tmp_path):
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _e2e_config(config, tmp_path, "plain")
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    assert trainer.lightning_model.coord_losses is None
    for name in ("training_config.yaml", "initial_config.yaml"):
        text = (tmp_path / "plain" / name).read_text()
        assert "coordinate_losses" not in text
    assert not any("coord" in k for k in trainer.trainer.callback_metrics)


def _labels_copy(src, dst, symmetric=False, labeled_nodes=None):
    """``minimal_instance`` with A <-> B symmetric and/or a ``labeled_nodes`` mask."""
    labels = sio.load_slp(str(src))
    if symmetric:
        labels.skeletons[0].add_symmetry("A", "B")
    if labeled_nodes is not None:
        labels.provenance["labeled_nodes"] = list(labeled_nodes)
    labels.save(str(dst), embed=True)
    return str(dst)


def _with_labels(cfg, path):
    cfg.data_config.train_labels_path = [path]
    cfg.data_config.val_labels_path = [path]
    return cfg


@pytest.mark.parametrize("term", ["consistency", "decoded"])
def test_run_training_entry_point(term, config, tmp_path, midpoint_ae, latent_ae):
    """`sleap_nn.train.run_training` (what `sleap-nn train` runs), including the
    post-training step, completes with each term."""
    from sleap_nn.train import run_training

    ae = midpoint_ae if term == "consistency" else latent_ae
    cfg = _e2e_config(config, tmp_path, term, {"autoencoder": ae, term: 0.01})
    run_training(cfg)
    saved = OmegaConf.load(tmp_path / term / "training_config.yaml")
    assert (tmp_path / term / "best.ckpt").exists()
    if term == "decoded":
        assert list(saved.trainer_config.eval.post_train_splits) == []


def test_flip_augmentation_end_to_end(
    config, tmp_path, minimal_instance, midpoint_ae, latent_ae
):
    """Flips on a symmetric skeleton: both terms train, the loader swaps the
    channels the setup check assumes, and a lopsided autoencoder is refused."""
    from sleap_nn.data.custom_datasets import get_train_val_datasets
    from sleap_nn.training.model_trainer import ModelTrainer

    path = _labels_copy(minimal_instance, tmp_path / "sym.pkg.slp", symmetric=True)

    def flip_cfg(name, block):
        cfg = _with_labels(_e2e_config(config, tmp_path, name, block), path)
        cfg.data_config.preprocessing.crop_size = 160  # both points in the crop
        geo = cfg.data_config.augmentation_config.geometric
        geo.rotation_min = geo.rotation_max = 0.0
        geo.flip_p = 1.0
        return cfg

    for name, block in [
        ("fc", {"autoencoder": midpoint_ae, "consistency": 0.01}),
        ("fd", {"autoencoder": latent_ae, "decoded": 0.01}),
    ]:
        trainer = ModelTrainer.get_model_trainer_from_config(flip_cfg(name, block))
        trainer.train()
        term = f"train/coord_{name == 'fc' and 'consistency' or 'decoded'}"
        assert torch.isfinite(trainer.trainer.callback_metrics[term])

    # Every training crop is flipped: B - A mirrors to (dx, -dy) only when the
    # loader also swaps A and B (without the swap it would be (-dx, dy)).
    flipped, plain = get_train_val_datasets(
        trainer.val_labels, trainer.val_labels, trainer.config
    )
    assert len(plain) > 0
    for i in range(len(plain)):
        assert torch.isfinite(plain[i]["instance"]).all()
        d0 = plain[i]["instance"][0, 1] - plain[i]["instance"][0, 0]
        d1 = flipped[i]["instance"][0, 1] - flipped[i]["instance"][0, 0]
        assert torch.allclose(d1, d0 * torch.tensor([1.0, -1.0]), atol=1e-3)

    lopsided = _save_ae(tmp_path / "lop.npz", [[0.8, 0.2]], [[1.0], [1.0]], ["A", "B"])
    trainer_cfg = flip_cfg("fl", {"autoencoder": lopsided, "consistency": 0.01})
    with pytest.raises(ValueError, match="does not commute"):
        ModelTrainer.get_model_trainer_from_config(trainer_cfg).train()


def test_decoded_term_uses_labeled_nodes(config, tmp_path, minimal_instance, latent_ae):
    """A file's `labeled_nodes` provenance reaches the decoded term as weights."""
    from sleap_nn.training.model_trainer import ModelTrainer

    path = _labels_copy(minimal_instance, tmp_path / "a.pkg.slp", labeled_nodes=["A"])
    block = {"autoencoder": latent_ae, "decoded": 0.01}
    cfg = _with_labels(_e2e_config(config, tmp_path, "ln", block), path)
    cfg.data_config.preprocessing.crop_size = 160  # both points in the crop
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    lm = trainer.lightning_model.eval()
    sample = trainer.trainer.val_dataloaders.dataset[0]
    assert sample["node_weights"].tolist() == [[1.0, 0.0]]
    batch = {k: v[None] for k, v in sample.items() if torch.is_tensor(v)}
    with torch.no_grad():
        maps = lm.forward(batch["instance_image"])
        term = lm.coord_losses(maps, batch)
        q = lm.coord_losses.points(maps)
    y = batch["instance"].reshape(1, -1, 2)
    assert torch.isfinite(y).all()  # B has a location; only its weight is 0
    only_a = decoded_loss(q, lm.coord_losses.matrix, y, torch.tensor([[1.0, 0.0]]))
    both = decoded_loss(q, lm.coord_losses.matrix, y)
    assert term.item() == pytest.approx(only_a.item(), rel=1e-5)
    assert term.item() != pytest.approx(both.item(), rel=1e-3)


def test_decoded_gallery_draws_decoded_points(config, tmp_path, latent_ae):
    """The W&B gallery is allowed for the decoded term and draws the soft-argmax
    latent points decoded into the labelled channels."""
    from sleap_nn.training import gallery
    from sleap_nn.training.model_trainer import ModelTrainer

    block = {"autoencoder": latent_ae, "decoded": 0.01}
    trainer = ModelTrainer.get_model_trainer_from_config(
        _e2e_config(config, tmp_path, "gal", block)
    )
    trainer.train()
    lm = trainer.lightning_model.eval()
    images = trainer.trainer.val_dataloaders.dataset[0]["instance_image"][None]
    pts, vals = gallery.predict_keypoints(lm, images)
    with torch.no_grad():
        q = lm.coord_losses.points(lm.forward(images))
    expected = torch.einsum("tl,bld->btd", lm.coord_losses.matrix, q).numpy()
    assert pts.shape == (1, 2, 2) and np.allclose(pts, expected, atol=1e-4)
    assert (vals == 1).all()
    # Model nodes for the gallery's per-file lists are the labelled channels.
    assert trainer._viz_file_nodes() == (None, None, None)
    trainer.config.trainer_config.use_wandb = True
    callback = trainer._gallery_callback()
    assert callback is not None


def test_restart_from_saved_decoded_config_is_refused(config, tmp_path, latent_ae):
    from sleap_nn.training.model_trainer import ModelTrainer

    block = {"autoencoder": latent_ae, "decoded": 0.01}
    ModelTrainer.get_model_trainer_from_config(
        _e2e_config(config, tmp_path, "first", block)
    ).train()
    saved = OmegaConf.load(tmp_path / "first" / "training_config.yaml")
    saved.trainer_config.run_name = "again"
    with pytest.raises(ValueError, match="data_head is set"):
        ModelTrainer.get_model_trainer_from_config(saved).train()
