"""The ``pretrained`` (HF/timm plain ViT: DINOv3, DINOv2) ``bbox_detr`` backbone.

Tiny random DINOv3 / DINOv2 configs are written to a temp directory and built with
``weights: false``, so nothing touches the network: tap shapes and indices, prefix
tokens dropped, drop path, the frozen post-tap blocks, RF-DETR's layer-decay depths
and weight-decay rule on these parameter names, a CPU training step of a tiny
``bbox_detr``, and the M3 sample config round trip.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

transformers = pytest.importorskip("transformers")

from sleap_nn.architectures.detr import (  # noqa: E402
    DetrBackbone,
    PretrainedDetrBackbone,
    build_detr_backbone,
)
from sleap_nn.architectures.model import Model  # noqa: E402
from sleap_nn.config.get_config import get_head_configs  # noqa: E402

requires_timm = pytest.mark.skipif(
    importlib.util.find_spec("timm") is None, reason="needs the `timm` package"
)

SAMPLE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "sample_configs"
    / "config_bbox_detr_dinov3_b_person.yaml"
)
TINY_DETR = dict(
    categories=["person"],
    hidden_dim=16,
    num_queries=12,
    group_detr=3,
    dec_layers=2,
    sa_nheads=2,
    ca_nheads=4,
    dec_n_points=2,
    dim_feedforward=32,
    num_select=12,
)


def _write_config(root: Path, kind: str, layers: int = 4) -> str:
    """Save a tiny random ViT config of ``kind`` and return its directory."""
    from transformers import Dinov2Config, DINOv3ViTConfig, TimmWrapperConfig

    if kind == "dinov3":
        config = DINOv3ViTConfig(
            hidden_size=32,
            num_hidden_layers=layers,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=16,
            num_register_tokens=4,
            image_size=64,
        )
    elif kind == "dinov2":
        config = Dinov2Config(
            hidden_size=32,
            num_hidden_layers=layers,
            num_attention_heads=2,
            mlp_ratio=2,
            patch_size=14,
            image_size=56,
        )
    else:  # timm DINOv3 (EVA blocks, RoPE, CLS + 4 register tokens)
        config = TimmWrapperConfig(
            architecture="vit_base_patch16_dinov3",
            model_args=dict(embed_dim=32, depth=layers, num_heads=2),
        )
    path = root / f"{kind}_{layers}"
    config.save_pretrained(path)
    return str(path)


def _cfg(model_name, **kw):
    base = dict(
        model_name=model_name,
        weights=False,
        in_channels=3,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        max_stride=32,
        output_stride=16,
    )
    base.update(kw)
    return OmegaConf.create(base)


KINDS = ["dinov3", "dinov2", pytest.param("timm", marks=requires_timm)]


@pytest.mark.parametrize("kind", KINDS)
def test_taps_shapes_and_interface(tmp_path, kind):
    """Default taps, stride-16 maps of a non-square input, the DETR interface."""
    bb = build_detr_backbone("pretrained", _cfg(_write_config(tmp_path, kind)))
    assert isinstance(bb, PretrainedDetrBackbone) and isinstance(bb, DetrBackbone)
    assert bb.mode == "tokens"
    assert bb.layers == [1, 2, 3, 4]
    assert (bb.num_taps, bb.stride, bb.out_channels) == (4, 16, 32)
    # Every tap goes through the pretrained final norm (RF-DETR's convention).
    assert bb._final_norm_taps == [True] * 4
    maps = bb.eval()(torch.rand(2, 3, 64, 96))
    assert [tuple(m.shape) for m in maps] == [(2, 32, 4, 6)] * 4
    with pytest.raises(ValueError, match="multiples of 16"):
        bb(torch.rand(1, 3, 64, 90))


def test_default_taps_scale_with_depth(tmp_path):
    bb = build_detr_backbone("pretrained", _cfg(_write_config(tmp_path, "dinov3", 8)))
    assert bb.layers == [2, 4, 6, 8]
    bb = build_detr_backbone("pretrained", _cfg(_write_config(tmp_path, "dinov3", 24)))
    assert bb.layers == [6, 12, 18, 24]


@pytest.mark.parametrize("kind", ["dinov3", "dinov2"])
def test_last_tap_is_the_normed_patch_tokens(tmp_path, kind):
    """CLS / register tokens are dropped: the last tap is HF's output grid."""
    bb = build_detr_backbone("pretrained", _cfg(_write_config(tmp_path, kind))).eval()
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        last = bb(x)[-1]
        pixels = bb._normalize(x)
        if bb.patch_size != 16:  # patch-14 DINOv2: resampled onto the 16-px grid
            pixels = torch.nn.functional.interpolate(
                pixels, size=(4 * 14, 6 * 14), mode="bilinear", align_corners=False
            )
        tokens = bb.enc(pixel_values=pixels).last_hidden_state
    grid = tokens[:, -24:].transpose(1, 2).reshape(1, 32, 4, 6)
    torch.testing.assert_close(last, grid)


def test_post_tap_blocks_frozen_and_drop_path(tmp_path):
    cfg = _cfg(_write_config(tmp_path, "dinov3"), layers=[1, 2], drop_path_rate=0.3)
    bb = build_detr_backbone("pretrained", cfg)
    frozen = {n for n, p in bb.named_parameters() if not p.requires_grad}
    assert any(n.startswith("enc.model.layer.2.") for n in frozen)
    assert not any(n.startswith("enc.model.layer.1.") for n in frozen)
    # Stochastic depth ramps 0 -> 0.3 over the blocks (HF gives every block 0.3).
    rates = [
        float(m.drop_prob)
        for n, m in bb.enc.named_modules()
        if "droppath" in type(m).__name__.lower()
    ]
    assert rates and min(rates) == 0.0 and max(rates) == pytest.approx(0.3)


@requires_timm
def test_drop_path_reaches_timm(tmp_path):
    cfg = _cfg(_write_config(tmp_path, "timm"), drop_path_rate=0.2)
    bb = build_detr_backbone("pretrained", cfg)
    # timm ramps 0 -> rate; a zero rate is an Identity (block 0).
    rates = [
        float(getattr(b.drop_path1, "drop_prob", 0.0)) for b in bb.enc.timm_model.blocks
    ]
    assert rates[0] == 0.0 and rates[-1] == pytest.approx(0.2)


def test_explicit_non_token_modes_are_overridden(tmp_path):
    bb = build_detr_backbone(
        "pretrained", _cfg(_write_config(tmp_path, "dinov3"), mode="decoder")
    )
    assert bb.mode == "tokens" and not hasattr(bb, "dec")


def _expected_depth(name, num_layers):
    """RF-DETR depth of a backbone parameter name: embeddings 0, block k k + 1."""
    match = re.search(r"\.(?:layer|blocks)\.(\d+)\.", name)
    if match:
        return int(match.group(1)) + 1
    head = name.split(".", 2)[-1] if name.startswith("enc.timm_model.") else name
    if re.search(r"(^|\.)(embeddings|cls_token|reg_token|patch_embed)", head):
        return 0
    return num_layers + 1


@pytest.mark.parametrize("kind", KINDS)
def test_layer_decay_depths_follow_rfdetr(tmp_path, kind):
    """Block k of a tap-at-4 encoder: lr x 0.5 ** (D + 1 - (k + 1)), D = 5."""
    bb = build_detr_backbone("pretrained", _cfg(_write_config(tmp_path, kind)))
    groups = bb.encoder_param_groups(1.0, encoder_lr_scale=2.0, layer_decay=0.5)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    trainable = {n: p for n, p in bb.named_parameters() if p.requires_grad}
    assert set(lr_of) == {id(p) for p in trainable.values()}
    num_layers = max(bb.layers) + 1
    for name, p in trainable.items():
        depth = _expected_depth(name, num_layers)
        assert lr_of[id(p)] == pytest.approx(
            2.0 * 0.5 ** (num_layers + 1 - depth)
        ), name
    # The final norm trains undecayed, the first block decays the most.
    names = list(trainable)
    final = [
        n for n in names if re.fullmatch(r"enc\.(timm_model\.)?(layer)?norm\.weight", n)
    ]
    assert final and lr_of[id(trainable[final[0]])] == pytest.approx(2.0)
    assert bb.encoder_param_groups(1.0, encoder_lr_scale=1.0, layer_decay=None) is None


def _training_cfg(model_dir, tmp_path):
    from sleap_nn.config.training_job_config import verify_training_cfg

    cfg = OmegaConf.load(SAMPLE)
    cfg.data_config.train_labels_path = []
    cfg.data_config.val_labels_path = []
    cfg.data_config.preprocessing.max_height = 64
    cfg.data_config.preprocessing.max_width = 64
    cfg.model_config.backbone_config.pretrained = dict(
        _cfg(model_dir, drop_path_rate=0.1, mode="auto")
    )
    heads = OmegaConf.structured(get_head_configs("bbox_detr"))
    heads.bbox_detr.detr.update(TINY_DETR)
    cfg.model_config.head_configs = heads
    cfg.trainer_config.trainer_accelerator = "cpu"
    cfg.trainer_config.trainer_devices = 1
    cfg.trainer_config.use_wandb = False
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    return verify_training_cfg(cfg)


def _module(cfg):
    from sleap_nn.config.utils import check_output_strides
    from sleap_nn.training.lightning_modules import LightningModel

    return LightningModel.get_lightning_model_from_config(check_output_strides(cfg))


@pytest.mark.parametrize("kind", KINDS)
def test_weight_decay_exempts_tokens_norms_and_biases(tmp_path, kind):
    """RF-DETR's rule plus position embeddings / special tokens by name."""
    module = _module(_training_cfg(_write_config(tmp_path, kind), tmp_path))
    assert module.model.backbone.mode == "tokens"
    optimizer = module._build_optimizer()
    wd_of = {
        id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]
    }
    trainable = {n: p for n, p in module.named_parameters() if p.requires_grad}
    assert set(wd_of) == {id(p) for p in trainable.values()}
    for name, p in trainable.items():
        if not name.startswith("model.backbone."):
            assert wd_of[id(p)] == 1e-4, name
            continue
        exempt = any(
            k in name
            for k in ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")
        ) or re.search(r"\.(cls_token|reg_token|register_tokens|mask_token)$", name)
        assert wd_of[id(p)] == (0.0 if exempt else 1e-4), name
    special = [n for n in trainable if re.search(r"(cls_token|reg_token)$", n)]
    assert special and all(wd_of[id(trainable[n])] == 0.0 for n in special)


@pytest.mark.parametrize("kind", KINDS)
def test_cpu_training_step(tmp_path, kind):
    """A tiny bbox_detr on this backbone: finite loss, gradients everywhere."""
    from sleap_nn.data.bboxes import SetTargetEncoder

    module = _module(_training_cfg(_write_config(tmp_path, kind), tmp_path)).train()
    encoder = SetTargetEncoder(max_targets=3)
    samples = [
        encoder(torch.tensor([[4.0, 6.0, 40.0, 60.0]]), torch.tensor([0]), (64, 64)),
        encoder(
            torch.tensor([[10.0, 10.0, 30.0, 50.0], [33.0, 2.0, 60.0, 40.0]]),
            torch.tensor([0, 0]),
            (64, 64),
        ),
    ]
    batch = torch.utils.data.default_collate(samples)
    loss, terms = module._compute_losses(module.model(torch.rand(2, 3, 64, 64)), batch)
    assert torch.isfinite(loss) and {"loss_ce", "loss_bbox", "loss_giou"} <= set(terms)
    loss.backward()
    missing = [
        n
        for n, p in module.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not missing, missing[:5]
    assert any(
        p.grad.abs().sum() > 0
        for p in module.model.backbone.parameters()
        if p.requires_grad
    )


def test_model_builds_and_pose_models_still_take_pretrained(tmp_path):
    """`pretrained` is a DETR backbone and still a pose backbone."""
    model_dir = _write_config(tmp_path, "dinov3")
    heads = OmegaConf.structured(get_head_configs("bbox_detr")).bbox_detr
    heads.detr.update(TINY_DETR)
    model = Model("pretrained", _cfg(model_dir), heads, "bbox_detr").eval()
    out = model(torch.rand(1, 3, 64, 64))
    assert out["pred_logits"].shape == (1, 12, 2)  # person + the extra logit
    pose = Model(
        "pretrained",
        _cfg(model_dir, output_stride=2, mode="decoder"),
        OmegaConf.create(
            {"confmaps": {"part_names": ["a"], "sigma": 2.5, "output_stride": 2}}
        ),
        "single_instance",
    )
    assert pose.backbone.mode == "decoder"


def test_config_rejects_pooled_or_decoder_modes(tmp_path):
    from sleap_nn.config.utils import check_output_strides

    cfg = _training_cfg(_write_config(tmp_path, "dinov3"), tmp_path)
    cfg.model_config.backbone_config.pretrained.mode = "encoder"
    with pytest.raises(ValueError, match="tokens"):
        check_output_strides(cfg)


def test_m3_sample_config_round_trip(tmp_path):
    """The M3 stage-1 sample config survives a TrainingJobConfig round trip."""
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.config.utils import check_output_strides

    cfg = verify_training_cfg(OmegaConf.load(SAMPLE))
    pretrained = cfg.model_config.backbone_config.pretrained
    assert pretrained.model_name == "timm/vit_base_patch16_dinov3.lvd1689m"
    assert list(pretrained.layers) == [3, 6, 9, 12]
    assert pretrained.mode == "tokens" and pretrained.drop_path_rate == 0.1
    assert cfg.data_config.preprocessing.resize_mode == "letterbox"
    assert cfg.data_config.preprocessing.max_height == 704
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert detr.dec_layers == 4 and list(detr.categories) == ["person"]
    assert check_output_strides(cfg).model_config.backbone_config.pretrained.mode == (
        "tokens"
    )

    out = tmp_path / "round_trip.yaml"
    OmegaConf.save(cfg, out)
    again = verify_training_cfg(OmegaConf.load(out))
    assert OmegaConf.to_container(again) == OmegaConf.to_container(cfg)
