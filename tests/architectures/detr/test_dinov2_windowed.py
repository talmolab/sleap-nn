"""Tests of the ``dinov2_windowed`` DETR backbone (RF-DETR's windowed DINOv2).

Parity with HF transformers' ``RfDetrDinov2Backbone`` is checked end to end in
``tests/convert/test_rfdetr.py``; these test the pieces.
"""

import json

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from sleap_nn.architectures.detr import DETR_BACKBONES, WindowedDinov2Backbone
from sleap_nn.architectures.detr.dinov2_windowed import (
    default_window_block_indexes,
    dinov2_state_dict,
    load_dinov2_weights,
    resample_patch_embed,
    resize_pos_embed,
    window_partition,
    window_unpartition,
)
from sleap_nn.architectures.model import Model, get_backbone
from sleap_nn.config.model_config import Dinov2WindowedConfig


def _tiny(**kwargs) -> WindowedDinov2Backbone:
    defaults = dict(
        hidden_size=16,
        num_layers=4,
        num_heads=2,
        patch_size=8,
        pos_embed_grid=4,
        out_indices=[2, 4],
        num_windows=2,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return WindowedDinov2Backbone(**defaults).eval()


def test_default_window_block_indexes_match_rfdetr():
    """RF-DETR's DINOv2-S: blocks 3, 6 and 9 (after the taps) attend globally."""
    assert default_window_block_indexes([3, 6, 9, 12], 12) == [
        0,
        1,
        2,
        4,
        5,
        7,
        8,
        10,
        11,
    ]


@pytest.mark.parametrize("grid", [(4, 4), (4, 6)])
def test_window_partition_roundtrip(grid):
    h, w = grid
    tokens = torch.randn(3, 1 + h * w, 5)
    windows = window_partition(tokens, grid, 2)
    assert windows.shape == (3 * 4, 1 + h * w // 4, 5)
    # Every window starts with its image's class token.
    assert torch.equal(windows[:, 0], tokens[:, 0].repeat_interleave(4, dim=0))
    # The first window is the top-left block of the grid.
    grid_tokens = tokens[:, 1:].reshape(3, h, w, 5)
    top_left = grid_tokens[0, : h // 2, : w // 2].reshape(-1, 5)
    assert torch.equal(windows[0, 1:], top_left)
    restored = window_unpartition(windows[:, 1:], grid, 2)
    assert torch.equal(restored, grid_tokens.permute(0, 3, 1, 2))


def test_forward_shapes_and_interface():
    backbone = _tiny()
    taps = backbone(torch.rand(2, 3, 32, 48))
    assert len(taps) == backbone.num_taps == 2
    assert backbone.stride == 8 and backbone.out_channels == 16
    assert all(t.shape == (2, 16, 4, 6) for t in taps)


def test_input_must_split_into_windows():
    with pytest.raises(ValueError, match="multiples of 16"):
        _tiny()(torch.rand(1, 3, 24, 32))


def test_windowed_blocks_attend_within_their_window():
    """With only windowed blocks before the tap, windows do not see each other."""
    backbone = _tiny(num_layers=2, out_indices=[2], window_block_indexes=[0, 1])
    x = torch.rand(1, 3, 32, 32)
    x2 = x.clone()
    x2[..., 16:, 16:] = torch.rand(1, 3, 16, 16)  # bottom-right window only
    with torch.no_grad():
        a, b = backbone(x)[0], backbone(x2)[0]
    torch.testing.assert_close(a[..., :2, :], b[..., :2, :])  # top windows
    torch.testing.assert_close(a[..., 2:, :2], b[..., 2:, :2])  # bottom-left
    assert not torch.allclose(a[..., 2:, 2:], b[..., 2:, 2:])

    # A global block mixes them.
    backbone = _tiny(num_layers=2, out_indices=[2], window_block_indexes=[0])
    with torch.no_grad():
        a, b = backbone(x)[0], backbone(x2)[0]
    assert not torch.allclose(a[..., :2, :], b[..., :2, :])


def test_position_embedding_interpolation():
    backbone = _tiny()
    emb = backbone.embeddings
    assert emb.interpolate_pos_encoding(4, 4) is emb.position_embeddings
    resized = emb.interpolate_pos_encoding(6, 8)
    assert resized.shape == (1, 1 + 48, 16)
    torch.testing.assert_close(resized[:, :1], emb.position_embeddings[:, :1])
    assert resize_pos_embed(emb.position_embeddings, 6).shape == (1, 37, 16)


def test_flexivit_resample_preserves_patch_responses():
    """<resize(x), resampled kernel> == <x, kernel> for an upsampled patch."""
    torch.manual_seed(0)
    kernel = torch.randn(4, 3, 14, 14)
    new = resample_patch_embed(kernel, 16)
    assert new.shape == (4, 3, 16, 16)
    patch = torch.randn(2, 3, 14, 14)
    up = torch.nn.functional.interpolate(
        patch, size=(16, 16), mode="bicubic", antialias=True
    )
    torch.testing.assert_close(
        torch.einsum("bchw,dchw->bd", up, new),
        torch.einsum("bchw,dchw->bd", patch, kernel),
        rtol=1e-4,
        atol=1e-4,
    )


def _fake_dinov2_dir(tmp_path, hidden=16, layers=4, heads=2, registers=False):
    """A local HF ``Dinov2Model`` checkpoint (patch 14, 37x37 position grid)."""
    ref = WindowedDinov2Backbone(
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        patch_size=14,
        pos_embed_grid=37,
        out_indices=[layers],
    )
    renames = {
        "attention.q_proj.": "attention.attention.query.",
        "attention.k_proj.": "attention.attention.key.",
        "attention.v_proj.": "attention.attention.value.",
        "attention.o_proj.": "attention.output.dense.",
    }
    state = {}
    for key, value in ref.state_dict().items():
        for new, old in renames.items():
            key = key.replace(new, old)
        state[key] = torch.randn_like(value)
    state["pooler.dense.weight"] = torch.zeros(1)  # dropped
    if registers:
        state["embeddings.register_tokens"] = torch.zeros(1, 4, hidden)
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                hidden_size=hidden,
                num_hidden_layers=layers,
                num_attention_heads=heads,
                patch_size=14,
            )
        )
    )
    return state


def test_load_dinov2_weights_from_local_checkpoint(tmp_path):
    state = _fake_dinov2_dir(tmp_path)
    backbone = _tiny(patch_size=16)
    load_dinov2_weights(backbone, tmp_path.as_posix())
    layer = backbone.encoder.layer[1]
    assert torch.equal(
        layer.attention.q_proj.weight,
        state["encoder.layer.1.attention.attention.query.weight"],
    )
    proj = backbone.embeddings.patch_embeddings.projection.weight
    assert proj.shape == (16, 3, 16, 16)
    assert backbone.embeddings.position_embeddings.shape == (1, 17, 16)

    # Re-applied after a re-initialization (the trainer's xavier init).
    torch.nn.init.zeros_(layer.attention.q_proj.weight)
    backbone.reload_pretrained_weights()
    assert torch.equal(
        layer.attention.q_proj.weight,
        state["encoder.layer.1.attention.attention.query.weight"],
    )


def test_load_dinov2_weights_rejects_mismatches(tmp_path):
    _fake_dinov2_dir(tmp_path, hidden=32, heads=4)
    with pytest.raises(ValueError, match="does not match"):
        load_dinov2_weights(_tiny(), tmp_path.as_posix())
    with pytest.raises(ValueError, match="register tokens"):
        dinov2_state_dict(
            {"embeddings.register_tokens": torch.zeros(1)},
            patch_size=16,
            pos_embed_grid=4,
        )


def test_registered_and_built_from_config(tmp_path):
    assert "dinov2_windowed" in DETR_BACKBONES
    _fake_dinov2_dir(tmp_path)
    cfg = OmegaConf.structured(
        Dinov2WindowedConfig(
            hidden_size=16,
            num_layers=4,
            num_heads=2,
            pos_embed_grid=4,
            out_indices=[2, 4],
            weights=tmp_path.as_posix(),
        )
    )
    head = OmegaConf.create(
        {
            "detr": dict(
                categories=["a", "b"],
                output_stride=16,
                hidden_dim=16,
                num_queries=4,
                group_detr=2,
                dec_layers=1,
                sa_nheads=2,
                ca_nheads=2,
                dec_n_points=2,
                dim_feedforward=16,
                dropout=0.0,
                c2f_num_blocks=1,
                c2f_expansion=0.5,
            )
        }
    )
    model = Model("dinov2_windowed", cfg, head, "bbox_detr").eval()
    assert model.backbone._pretrained_state is not None
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert out["pred_logits"].shape == (1, 4, 3)

    with pytest.raises(KeyError, match="only supported for bbox_detr"):
        get_backbone("dinov2_windowed", cfg)
