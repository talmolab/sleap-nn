"""Tests for the ``bbox_detr`` model modules (``sleap_nn.architectures.detr``)."""

import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.architectures.detr import (
    DETR_BACKBONES,
    BBoxDetrHead,
    DetrBackbone,
    GroupSelfAttention,
    MultiScaleDeformableAttention,
    TinyDetrBackbone,
    build_detr_backbone,
    multi_scale_deformable_attention,
    refine_boxes,
    register_detr_backbone,
    token_proposals,
)
from sleap_nn.architectures.model import Model
from sleap_nn.config.get_config import get_head_configs
from sleap_nn.training.detr import SetCriterion, build_set_criterion

TINY = dict(
    num_classes=2,
    hidden_dim=32,
    num_queries=10,
    group_detr=3,
    dec_layers=2,
    sa_nheads=2,
    ca_nheads=4,
    dec_n_points=2,
    dim_feedforward=64,
)


def _tiny(**overrides):
    torch.manual_seed(0)
    backbone = TinyDetrBackbone(out_channels=16, stride=16, num_blocks=2, num_taps=2)
    head = BBoxDetrHead(in_channels=32, **{**TINY, **overrides})
    return backbone, head


def _targets(batch_size):
    targets = []
    for i in range(batch_size):
        n = i + 1
        boxes = torch.rand(n, 4) * 0.4 + 0.3
        targets.append({"labels": torch.arange(n) % 2, "boxes": boxes})
    return targets


def test_tiny_backbone_interface():
    backbone = TinyDetrBackbone(out_channels=16, stride=8, num_blocks=3, num_taps=2)
    assert isinstance(backbone, DetrBackbone)
    taps = backbone(torch.rand(2, 3, 64, 48))
    assert len(taps) == backbone.num_taps == 2
    assert all(t.shape == (2, 16, 8, 6) for t in taps)
    with pytest.raises(ValueError, match="num_taps"):
        TinyDetrBackbone(num_blocks=1, num_taps=2)


@pytest.mark.parametrize("training", [True, False])
def test_output_shapes_train_vs_eval(training):
    """Training emits every query group; eval only group 0."""
    backbone, head = _tiny()
    head.train(training)
    out = head(backbone(torch.rand(2, 3, 96, 80)))
    groups = TINY["group_detr"] if training else 1
    q = groups * TINY["num_queries"]
    c = TINY["num_classes"] + 1
    assert out["pred_logits"].shape == (2, q, c)
    assert out["pred_boxes"].shape == (2, q, 4)
    assert len(out["aux_outputs"]) == TINY["dec_layers"] - 1
    for aux in out["aux_outputs"]:
        assert aux["pred_logits"].shape == (2, q, c)
        assert aux["pred_boxes"].shape == (2, q, 4)
    assert out["enc_outputs"]["pred_logits"].shape == (2, q, c)
    assert out["enc_outputs"]["pred_boxes"].shape == (2, q, 4)


def test_eval_matches_training_group_zero():
    """Group 0 of a training forward is the eval forward (groups don't mix)."""
    backbone, head = _tiny()
    feats = backbone(torch.rand(1, 3, 96, 80))
    head.train()
    train_out = head(feats)
    head.eval()
    eval_out = head(feats)
    q = TINY["num_queries"]
    torch.testing.assert_close(train_out["pred_logits"][:, :q], eval_out["pred_logits"])
    torch.testing.assert_close(train_out["pred_boxes"][:, :q], eval_out["pred_boxes"])


def test_group_self_attention_isolates_groups():
    torch.manual_seed(0)
    attn = GroupSelfAttention(d_model=16, n_heads=2)
    x = torch.randn(2, 12, 16)
    pos = torch.randn(2, 12, 16)
    out = attn(x, pos, num_groups=3)
    x2 = x.clone()
    x2[:, 4:] += 1.0  # perturb groups 1 and 2
    out2 = attn(x2, pos, num_groups=3)
    torch.testing.assert_close(out[:, :4], out2[:, :4])
    assert not torch.allclose(out[:, 4:], out2[:, 4:])
    # One group over the same 4 queries is the same computation.
    torch.testing.assert_close(attn(x[:, :4], pos[:, :4], num_groups=1), out[:, :4])


def test_refine_boxes():
    ref = torch.tensor([[0.5, 0.4, 0.2, 0.1]])
    assert torch.equal(refine_boxes(ref, torch.zeros(1, 4)), ref)
    out = refine_boxes(
        ref, torch.tensor([[1.0, -1.0, 0.0, torch.log(torch.tensor(2.0))]])
    )
    torch.testing.assert_close(out, torch.tensor([[0.7, 0.3, 0.2, 0.2]]))


def test_token_proposals():
    proposals, invalid = token_proposals([(2, 4)], torch.device("cpu"))
    assert proposals.shape == (8, 4) and not invalid.any()
    torch.testing.assert_close(proposals[0], torch.tensor([0.125, 0.25, 0.05, 0.05]))
    torch.testing.assert_close(proposals[-1], torch.tensor([0.875, 0.75, 0.05, 0.05]))
    # Wider than 50 tokens: edge centers fall outside (0.01, 0.99).
    proposals, invalid = token_proposals([(1, 64)], torch.device("cpu"))
    assert invalid[0, 0] and invalid[-1, 0] and not invalid[1:-1].any()
    assert (proposals[invalid[:, 0]] == 0).all()


def test_deformable_attention_samples_reference_point():
    """With all points at the reference center, the output is the bilinear value."""
    value = torch.arange(12.0).view(1, 12, 1, 1)  # one head, a 3x4 map
    loc = torch.tensor([0.5, 0.5]).view(1, 1, 1, 1, 1, 2).expand(1, 1, 1, 1, 2, 2)
    weights = torch.full((1, 1, 1, 1, 2), 0.5)
    out = multi_scale_deformable_attention(value, [(3, 4)], loc, weights)
    # Center of the 3x4 map: midway between columns 1-2 of row 1 -> (5 + 6) / 2.
    torch.testing.assert_close(out, torch.tensor([[[5.5]]]))

    attn = MultiScaleDeformableAttention(d_model=32, n_levels=1, n_heads=4, n_points=2)
    out = attn(
        torch.randn(2, 5, 32), torch.rand(2, 5, 4), torch.randn(2, 12, 32), [(3, 4)]
    )
    assert out.shape == (2, 5, 32)


def test_init_matches_rf_detr():
    _, head = _tiny()
    assert torch.all(head.reference_point_embed.weight == 0)
    torch.testing.assert_close(
        head.class_embed.bias.sigmoid(), torch.full((TINY["num_classes"] + 1,), 0.01)
    )
    assert torch.all(head.bbox_embed.layers[-1].weight == 0)
    for g in range(TINY["group_detr"]):
        torch.testing.assert_close(
            head.enc_out_class_embed[g].weight, head.class_embed.weight
        )
        assert head.enc_out_class_embed[g].weight is not head.class_embed.weight


def test_too_few_tokens_raises():
    backbone, head = _tiny(num_queries=50)
    with pytest.raises(ValueError, match="fewer tokens than num_queries"):
        head(backbone(torch.rand(1, 3, 64, 64)))  # 4x4 = 16 tokens


def _criterion_loss(backbone, head, x, targets):
    criterion = SetCriterion(TINY["num_classes"], group_detr=TINY["group_detr"])
    criterion.train(head.training)
    out = head(backbone(x))
    losses = criterion(out, targets)
    return criterion.total_loss(losses), losses


def test_outputs_feed_set_criterion_with_gradients():
    """Train-mode outputs give a finite loss and a finite grad on every parameter."""
    backbone, head = _tiny()
    x = torch.rand(2, 3, 96, 80)
    targets = _targets(2)
    loss, losses = _criterion_loss(backbone, head, x, targets)
    assert torch.isfinite(loss)
    assert "loss_ce_enc" in losses and "loss_giou_0" in losses
    loss.backward()
    params = list(backbone.named_parameters()) + list(head.named_parameters())
    for name, p in params:
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name

    # Zero-inits (last box layer, reference deltas) block some gradients at init;
    # with every parameter random, every one of them gets a non-zero gradient.
    with torch.no_grad():
        for _, p in params:
            p.normal_(0, 0.1)
            p.grad = None
    loss, _ = _criterion_loss(backbone, head, x, targets)
    loss.backward()
    for name, p in params:
        assert p.grad.abs().sum() > 0, name


def test_eval_outputs_feed_set_criterion():
    backbone, head = _tiny()
    head.eval()
    with torch.no_grad():
        loss, _ = _criterion_loss(backbone, head, torch.rand(2, 3, 96, 80), _targets(2))
    assert torch.isfinite(loss)


def test_build_set_criterion_from_config():
    leaf = OmegaConf.structured(get_head_configs("bbox_detr")).bbox_detr.detr
    leaf.categories = ["a", "b", "c"]
    leaf.giou_loss_coef = 3.0
    leaf.set_cost_bbox = 4.0
    criterion = build_set_criterion(leaf)
    assert criterion.num_classes == 3
    assert criterion.group_detr == 13
    assert criterion.weight_dict == {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 3.0}
    assert criterion.matcher.cost_bbox == 4.0


class _Logits(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out["pred_logits"], out["pred_boxes"]


def test_eval_export_trace():
    """The eval graph (incl. grid_sample deformable attention) exports and runs."""
    backbone, head = _tiny()
    model = _Logits(backbone, head).eval()
    x = torch.rand(1, 3, 96, 80)
    program = torch.export.export(model, (x,))
    ops = {str(node.target) for node in program.graph.nodes}
    assert any("grid_sampler" in op for op in ops)
    with torch.no_grad():
        expected = model(x)
        actual = program.module()(x)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@pytest.fixture
def tiny_unet_backbone():
    """Register the tiny backbone as the DETR adapter of the `unet` config type."""
    assert "unet" not in DETR_BACKBONES

    @register_detr_backbone("unet")
    def _build(cfg):
        return TinyDetrBackbone(
            in_channels=cfg.in_channels, out_channels=16, stride=16, num_taps=2
        )

    yield
    del DETR_BACKBONES["unet"]


def test_unregistered_backbone_raises():
    with pytest.raises(ValueError, match="no bbox_detr adapter"):
        build_detr_backbone("convnext", OmegaConf.create({}))


def test_model_builds_bbox_detr(tiny_unet_backbone):
    """`Model` builds a DETR from the config and returns the set outputs."""
    head_configs = OmegaConf.structured(get_head_configs("bbox_detr"))
    leaf = head_configs.bbox_detr.detr
    leaf.update(
        categories=["mouse"],
        hidden_dim=32,
        num_queries=10,
        group_detr=2,
        dec_layers=2,
        sa_nheads=2,
        ca_nheads=4,
        dim_feedforward=64,
    )
    model = Model(
        backbone_type="unet",
        backbone_config=OmegaConf.create({"in_channels": 1, "output_stride": 16}),
        head_configs=head_configs.bbox_detr,
        model_type="bbox_detr",
    )
    assert isinstance(model.detr, BBoxDetrHead)
    assert len(model.head_layers) == 0
    model.eval()
    out = model(torch.rand(2, 3, 96, 64))  # RGB input coerced to the 1-channel config
    assert out["pred_logits"].shape == (2, 10, 2)
    assert out["pred_boxes"].shape == (2, 10, 4)
    keys = set(model.state_dict())
    assert any(k.startswith("backbone.") for k in keys)
    assert any(k.startswith("detr.decoder.") for k in keys)
