"""Model semantics of the ``embedding`` model type, through `ModelTrainer.train`.

Covers three fixes from the embedding-stack review (emb-review F1):

* ``freeze_backbone`` freezes the backbone's pretrained ENCODER only, and keeps it in
  eval mode for the whole run. It used to set ``requires_grad=False`` on the whole
  backbone: the encoder stayed in train mode (BatchNorm statistics drifted,
  stochastic depth stayed on) and the randomly initialized middle blocks of a native
  convnext / swint were frozen at their random init.
* The GeM exponent ``p`` is bounded to ``[1, 6]`` in the forward pass and its
  gradient can always bring it back into range. The old ``clamp(min=1)`` had zero
  gradient below 1 and no upper bound.
* A UNet ``stem_stride`` is rejected at config time: it builds decoder blocks the
  pooled head never reads.

The BatchNorm half of the ``freeze_backbone`` fix needs a BatchNorm encoder, which only
the HuggingFace ``pretrained`` backbone has; that test lives in
``tests/architectures/test_pretrained.py`` (the ``backbones`` extra).
"""

import numpy as np
import pytest
import sleap_io as sio
import torch
from _pytest.logging import LogCaptureFixture
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.architectures.heads import GeM
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.lightning_modules import EmbeddingLightningModule
from sleap_nn.training.model_trainer import ModelTrainer

N_FRAMES = 16
N_ANIMALS = 2
P, K = 2, 4  # P x K = 8 crops per step -> 4 steps per epoch

GEM_KEY = "model.head_layers.0.pre_embedding_pool.p"

# A native convnext small enough for a CPU test. `model_type` outside the named
# sizes makes the wrapper use `arch`.
SMALL_CONVNEXT = {
    "model_type": "custom",
    "arch": {"depths": [1, 1, 1, 1], "channels": [8, 16, 32, 64]},
    "in_channels": 1,
    "filters_rate": 1.5,
}


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru into pytest's caplog."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def tracked_slp(centered_instance_video, tmp_path):
    """Two tracked (and identified) animals in 16 frames of a real video."""
    video = sio.load_video(centered_instance_video.as_posix())
    skeleton = sio.Skeleton(["a", "b"])
    tracks = [sio.Track(name=f"track_{i}") for i in range(N_ANIMALS)]
    identities = [sio.Identity(name=f"animal_{i}") for i in range(N_ANIMALS)]
    rng = np.random.default_rng(0)
    frames = []
    for frame_idx in range(N_FRAMES):
        instances = []
        for track, identity, (cx, cy) in zip(
            tracks, identities, [(120, 180), (260, 200)]
        ):
            points = np.array([[cx - 10, cy], [cx + 10, cy]], float)
            instance = sio.Instance.from_numpy(
                points + rng.normal(0, 2, 2), skeleton=skeleton, track=track
            )
            instance.identity = identity
            instances.append(instance)
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )
    labels = sio.Labels(
        labeled_frames=frames, videos=[video], skeletons=[skeleton], tracks=tracks
    )
    path = tmp_path / "tracked.slp"
    labels.save(path.as_posix())
    return path


def _config(tracked_slp, tmp_path, run_name, backbone=None, **updates):
    """A tiny CPU embedding training config (UNet unless `backbone` is given)."""
    if backbone is None:
        backbone = {
            "unet": {
                "in_channels": 1,
                "filters": 8,
                "max_stride": 16,
                "output_stride": 16,
            }
        }
    cfg = OmegaConf.create(
        {
            "data_config": {
                "train_labels_path": [tracked_slp.as_posix()],
                "val_labels_path": [tracked_slp.as_posix()],
                "data_pipeline_fw": "torch_dataset",
                "preprocessing": {"crop_size": 64},
                "identity": {"track_names_are_global": True},
            },
            "model_config": {
                "backbone_config": backbone,
                "head_configs": {
                    "embedding": {
                        "embedding": {
                            "embedding_dim": 16,
                            "output_stride": 16,
                            "objective": {
                                "sampler": {
                                    "kind": "pk",
                                    "groups_per_batch": P,
                                    "samples_per_group": K,
                                }
                            },
                        }
                    }
                },
            },
            "trainer_config": {
                "train_data_loader": {"batch_size": 4, "num_workers": 0},
                "val_data_loader": {"batch_size": 16, "num_workers": 0},
                "trainer_accelerator": "cpu",
                "trainer_devices": 1,
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 1,
                "max_epochs": 1,
                "save_ckpt": True,
                "ckpt_dir": tmp_path.as_posix(),
                "run_name": run_name,
                "lr_scheduler": None,
                "early_stopping": {"stop_training_on_plateau": False},
                "model_ckpt": {"save_last": False},
                "optimizer": {"lr": 1e-2},
            },
        }
    )
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return verify_training_cfg(cfg)


def _train(cfg):
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    return trainer


def _encoder(backbone):
    """The encoder modules, spelled with attributes old and new code both have."""
    if hasattr(backbone, "encoders"):  # unet
        stem = [backbone.stem] if backbone.stem is not None else []
        return stem + list(backbone.encoders)
    return [backbone.enc]


def _encoder_param_names(module):
    """Qualified names (in `module.named_parameters()`) of the encoder's params."""
    ids = {id(p) for m in _encoder(module.model.backbone) for p in m.parameters()}
    return {n for n, p in module.named_parameters() if id(p) in ids}


@pytest.fixture
def step_spy(monkeypatch):
    """Record the model's state at every real training step.

    Wraps `EmbeddingLightningModule.training_step`, which Lightning calls once per
    batch before that batch's optimizer step -- so the first record holds the weights
    training started from.
    """
    records = []
    original = EmbeddingLightningModule.training_step

    def spy(self, batch, batch_idx):
        encoder = _encoder(self.model.backbone)
        records.append(
            {
                "encoder_modes": {m.training for enc in encoder for m in enc.modules()},
                "params": (
                    {n: p.detach().clone() for n, p in self.named_parameters()}
                    if not records
                    else None
                ),
            }
        )
        return original(self, batch, batch_idx)

    monkeypatch.setattr(EmbeddingLightningModule, "training_step", spy)
    return records


def _changed(before, after, names):
    return {n for n in names if not torch.equal(before[n], after[n])}


# --------------------------------------------------------------- freeze_backbone


@pytest.mark.parametrize(
    "backbone",
    [
        {"convnext": SMALL_CONVNEXT},
        # swint-tiny (its size is fixed): the one native encoder whose train mode
        # differs from eval (stochastic depth 0.1), so "frozen but in train mode"
        # computed different features at every step.
        {"swint": {"in_channels": 1}},
    ],
    ids=["convnext", "swint"],
)
def test_freeze_backbone_freezes_only_the_encoder_of_a_native_backbone(
    tracked_slp, tmp_path, step_spy, caplog, backbone
):
    """The encoder is frozen and in eval mode; the middle blocks and head train.

    The old code froze every backbone parameter, so on a native convnext / swint the
    randomly initialized middle blocks (31.8M parameters on convnext-tiny) stayed at
    their random init and only the ~50k head parameters trained -- and the "frozen"
    encoder stayed in train mode.
    """
    cfg = _config(
        tracked_slp,
        tmp_path,
        "freeze_native",
        backbone=backbone,
        **{"model_config.head_configs.embedding.embedding.freeze_backbone": True},
    )
    trainer = _train(cfg)
    module = trainer.lightning_model

    encoder = _encoder_param_names(module)
    middle = {n for n, _ in module.named_parameters() if "middle_blocks" in n}
    head = {n for n, _ in module.named_parameters() if "head_layers" in n}
    assert encoder and middle and head

    frozen = {n for n, p in module.named_parameters() if not p.requires_grad}
    assert frozen == encoder, "only the encoder may be frozen"

    before = step_spy[0]["params"]
    after = {n: p.detach() for n, p in module.named_parameters()}
    assert not _changed(before, after, encoder), "a frozen encoder weight moved"
    assert _changed(before, after, middle), "the middle blocks did not train"
    assert _changed(before, after, head), "the head did not train"

    assert len(step_spy) >= 2
    for step, record in enumerate(step_spy):
        assert record["encoder_modes"] == {
            False
        }, f"step {step}: frozen encoder was in train mode"

    # No pretrained weights at all: the encoder stays at its random init. Allowed,
    # but almost certainly a mistake, so it is flagged.
    assert "freeze_backbone=True" in caplog.text
    assert "random initialization" in caplog.text

    # A native backbone keeps GeM even with a frozen encoder: what it pools is the
    # output of its trainable, ReLU-terminated middle blocks. The resolved default
    # is written into the saved config.
    saved = OmegaConf.load(tmp_path / "freeze_native" / "training_config.yaml")
    assert saved.model_config.head_configs.embedding.embedding.pool == "gem"
    assert isinstance(module.model.head_layers[0].pre_embedding_pool, GeM)


def test_freeze_backbone_on_a_checkpoint_initialized_unet(
    tracked_slp, tmp_path, step_spy, caplog
):
    """Fine-tune from a sleap-nn checkpoint with a frozen encoder.

    `model_config.pretrained_backbone_weights` makes the backbone pretrained, so no
    warning; the encoder keeps exactly the checkpoint's weights through training
    while the UNet middle blocks and the head keep training.
    """
    _train(_config(tracked_slp, tmp_path, "unet_first"))
    ckpt = tmp_path / "unet_first" / "best.ckpt"
    assert ckpt.exists()
    step_spy.clear()
    caplog.clear()

    cfg = _config(
        tracked_slp,
        tmp_path,
        "unet_frozen",
        **{
            "model_config.pretrained_backbone_weights": ckpt.as_posix(),
            "model_config.head_configs.embedding.embedding.freeze_backbone": True,
        },
    )
    module = _train(cfg).lightning_model

    encoder = _encoder_param_names(module)
    middle = {n for n, _ in module.named_parameters() if "middle_blocks" in n}
    assert encoder and middle
    frozen = {n for n, p in module.named_parameters() if not p.requires_grad}
    assert frozen == encoder

    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    after = {n: p.detach() for n, p in module.named_parameters()}
    for name in encoder:
        assert torch.equal(after[name], loaded[name]), f"{name} left the checkpoint"
    assert _changed(step_spy[0]["params"], after, middle), "middle blocks froze"
    assert all(r["encoder_modes"] == {False} for r in step_spy)
    assert "random initialization" not in caplog.text


def test_freeze_backbone_off_trains_the_whole_backbone(tracked_slp, tmp_path, step_spy):
    """The default is unchanged: everything trains, the encoder in train mode."""
    module = _train(_config(tracked_slp, tmp_path, "unet_free")).lightning_model
    assert all(p.requires_grad for p in module.parameters())
    assert all(r["encoder_modes"] == {True} for r in step_spy)
    after = {n: p.detach() for n, p in module.named_parameters()}
    assert _changed(step_spy[0]["params"], after, _encoder_param_names(module))


# ------------------------------------------------------------------ UNet stem_stride


def test_unet_stem_stride_is_rejected_at_config_time(tracked_slp, tmp_path):
    """A stem makes the UNet build decoder blocks the pooled head never reads.

    `up_blocks = log2(max_stride / output_stride) + stem_blocks`, so even at
    output_stride == max_stride a stem adds decoder blocks, and the embedding head
    reads the bottleneck: those parameters never get a gradient (a DDP
    unused-parameter error). Same hazard as the rejected `pretrained.mode='decoder'`.
    """
    cfg = _config(
        tracked_slp,
        tmp_path,
        "stem",
        **{"model_config.backbone_config.unet.stem_stride": 2},
    )
    with pytest.raises(ValueError, match="stem_stride"):
        ModelTrainer.get_model_trainer_from_config(cfg)


# ------------------------------------------------------------------ GeM exponent


def _set_gem_p(ckpt_path, value, out_path):
    """Copy a checkpoint with its GeM exponent overwritten (as an old one could be)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["state_dict"][GEM_KEY] = torch.tensor(float(value))
    torch.save(ckpt, out_path)
    return out_path


def _gem_reference(x, p, eps=1e-6):
    """The GeM formula, written out."""
    return x.clamp(min=eps).pow(p).mean((2, 3)).pow(1.0 / p)


def test_gem_exponent_above_the_cap_is_clamped_through_training(
    tracked_slp, tmp_path, caplog
):
    """A head checkpoint whose GeM `p` is out of range trains with `p` capped.

    The old code had no upper bound; above p ~ 7.5 a channel that is non-positive
    everywhere pools to exactly 0 in float32 and the gradient of the 1/p-th root is
    NaN. The stored value keeps its meaning, so the cap applies to the value used, a
    warning names the difference, and training never pushes `p` further out.
    """
    _train(_config(tracked_slp, tmp_path, "gem_base"))
    head_ckpt = _set_gem_p(
        tmp_path / "gem_base" / "best.ckpt", 20.0, tmp_path / "gem20.ckpt"
    )
    caplog.clear()

    cfg = _config(
        tracked_slp,
        tmp_path,
        "gem_capped",
        **{"model_config.pretrained_head_weights": head_ckpt.as_posix()},
    )
    module = _train(cfg).lightning_model
    assert "above the cap" in caplog.text

    gem = module.model.head_layers[0].pre_embedding_pool
    assert isinstance(gem, GeM)
    p = float(gem.p.detach())
    assert np.isfinite(p) and p <= 20.0, f"p={p}: NaN, or pushed further out"

    x = torch.rand(3, 5, 4, 4) * 4.0
    with torch.no_grad():
        pooled = gem.eval()(x)
    torch.testing.assert_close(pooled, _gem_reference(x, 6.0))


@pytest.mark.parametrize("stored_p, used_p", [(2.9382, 2.9382), (0.7, 1.0)])
def test_old_gem_checkpoints_load_with_their_stored_exponent(
    tracked_slp, tmp_path, stored_p, used_p
):
    """Old checkpoints keep their state-dict key and pool exactly as they did.

    2.9382 is the exponent of a real checkpoint trained before the cap existed
    (`emb_convnext_frozen`); every checkpoint trained so far learned p in
    [2.94, 3.03]. An old checkpoint's p below 1 was used as 1 by the old
    `clamp(min=1)` and still is. Loaded through the inference loader `predict` uses.
    """
    from sleap_nn.inference.loaders import _load_lightning_module

    _train(_config(tracked_slp, tmp_path, "gem_old"))
    ckpt = tmp_path / "gem_old" / "best.ckpt"
    _set_gem_p(ckpt, stored_p, ckpt)

    module, _, _ = _load_lightning_module(
        EmbeddingLightningModule,
        (tmp_path / "gem_old").as_posix(),
        model_type="embedding",
        device="cpu",
    )
    gem = module.model.head_layers[0].pre_embedding_pool
    assert float(gem.p) == pytest.approx(stored_p)
    x = torch.rand(3, 5, 4, 4) * 4.0
    with torch.no_grad():
        pooled = gem(x)
    torch.testing.assert_close(pooled, _gem_reference(x, used_p))


def test_gem_exponent_below_the_floor_recovers_through_training(tracked_slp, tmp_path):
    """A GeM `p` below 1 moves back up during training instead of sticking.

    The old `clamp(min=1)` had exactly zero gradient below 1, so a `p` that stepped
    under the floor (or a checkpoint that stored one) never moved again. On this data
    the loss asks for a sharper pool than p=1 (average pooling), so `p` climbs.
    """
    _train(_config(tracked_slp, tmp_path, "floor_base"))
    head_ckpt = _set_gem_p(
        tmp_path / "floor_base" / "best.ckpt", 0.5, tmp_path / "gem05.ckpt"
    )
    cfg = _config(
        tracked_slp,
        tmp_path,
        "floor",
        **{
            "model_config.pretrained_head_weights": head_ckpt.as_posix(),
            "trainer_config.max_epochs": 3,
        },
    )
    module = _train(cfg).lightning_model
    p = float(module.model.head_layers[0].pre_embedding_pool.p.detach())
    assert p > 0.5, f"p stuck at {p} below the floor"


class TestGeMExponentGradient:
    """The clamp on `p` must not trap it: its gradient can always lead back in range.

    A descent step moves `p` by `-grad`. The generalized mean increases with `p`
    (power-mean inequality), so for a positive, non-constant input the gradient of
    `-sum(GeM(x))` w.r.t. `p` is negative (it asks for a larger `p`) and that of
    `+sum(GeM(x))` is positive (it asks for a smaller one).
    """

    @staticmethod
    def _grad(p0, sign):
        gem = GeM(p=p0).train()
        x = torch.rand(2, 4, 6, 6) + 0.1
        (sign * gem(x).sum()).backward()
        return gem.p.grad.item()

    def test_below_the_floor_the_way_back_up_has_a_gradient(self):
        # Old: clamp(min=1) -> exactly zero here, so p < 1 could never recover.
        assert self._grad(0.5, sign=-1.0) < 0

    def test_below_the_floor_a_push_further_down_is_blocked(self):
        assert self._grad(0.5, sign=1.0) == 0.0

    def test_above_the_cap_the_way_back_down_has_a_gradient(self):
        assert self._grad(20.0, sign=1.0) > 0

    def test_above_the_cap_a_push_further_up_is_blocked(self):
        assert self._grad(20.0, sign=-1.0) == 0.0

    def test_in_range_the_gradient_is_the_true_one(self):
        x = torch.rand(2, 4, 6, 6) + 0.1
        gem = GeM(p=3.0).train()
        gem(x).sum().backward()
        p = torch.tensor(3.0, requires_grad=True)
        _gem_reference(x, p).sum().backward()
        torch.testing.assert_close(gem.p.grad, p.grad)

    def test_out_of_range_values_are_clamped_in_the_forward(self):
        x = torch.rand(2, 4, 6, 6) + 0.1
        for p0, used in [(0.5, 1.0), (-2.0, 1.0), (20.0, 6.0), (3.0, 3.0)]:
            gem = GeM(p=p0)
            for mode in (True, False):
                gem.train(mode)
                torch.testing.assert_close(gem(x), _gem_reference(x, used))
