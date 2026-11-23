""" "Same" max-pool padding in traced (exported) graphs follows the input size.

``MaxPool2dWithSamePadding`` decided whether to pad with a Python ``if``, which
tracing freezes: an exported UNet padded as its traced input needed, whatever size
it later ran at. Eager (checkpoint) outputs must not change at all.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sleap_nn.architectures.common import MaxPool2dWithSamePadding

_CKPT = Path(__file__).resolve().parents[1] / "assets/model_ckpts"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("hw", [(8, 8), (9, 13), (17, 6), (1, 5)])
def test_eager_pool_is_the_padded_max_pool(hw, stride):
    """Eager output is bit-identical to F.pad + max_pool2d with the same amounts."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, *hw)
    pool = MaxPool2dWithSamePadding(kernel_size=2, stride=stride, padding="same")
    pads = []
    for i in reversed(hw):  # width first, as F.pad takes them
        p = max((-(-i // stride) - 1) * stride + 2 - i, 0)
        pads += [p // 2, p - p // 2]
    want = F.max_pool2d(F.pad(x, pads), 2, stride)
    assert torch.equal(pool(x), want)


@pytest.mark.parametrize("stride", [1, 2])
def test_traced_pool_pads_for_each_input_size(stride):
    pool = MaxPool2dWithSamePadding(kernel_size=2, stride=stride, padding="same")
    traced = torch.jit.trace(pool, torch.randn(1, 1, 8, 8))  # needs no padding
    for hw in [(8, 8), (9, 13), (17, 6), (5, 7)]:
        x = torch.randn(1, 2, *hw)
        assert torch.equal(traced(x), pool(x)), hw


def test_exported_unet_matches_the_checkpoint_at_other_sizes(tmp_path):
    """A UNet exported at one size runs at sizes needing other padding in ONNX Runtime
    with the checkpoint's confidence maps."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from sleap_nn.export.cli import _load_lightning_model
    from sleap_nn.export.exporters import export_to_onnx
    from sleap_nn.export.utils import (
        load_training_config,
        resolve_backbone_type,
        resolve_model_type,
    )
    from sleap_nn.export.wrappers.network import NetworkONNXWrapper

    d = _CKPT / "minimal_instance_centered_instance"
    cfg = load_training_config(d)
    model = _load_lightning_model(
        model_type=resolve_model_type(cfg),
        backbone_type=resolve_backbone_type(cfg),
        cfg=cfg,
        ckpt_path=d / "best.ckpt",
        device="cpu",
    ).model.eval()
    wrapper = NetworkONNXWrapper(model, head_names=["CenteredInstanceConfmapsHead"])
    wrapper.eval()
    path = export_to_onnx(
        wrapper,
        tmp_path / "model.onnx",
        input_shape=(1, 1, 64, 64),
        input_dtype=torch.float32,
        output_names=["CenteredInstanceConfmapsHead"],
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # 68 -> 34 -> 17 and 100 -> 50 -> 25: odd at the last pooling input, which the
    # traced size (64 -> 32 -> 16) never pads.
    for hw in [(64, 64), (68, 100), (100, 68)]:
        x = rng.uniform(0, 255, (2, 1, *hw)).astype(np.float32)
        (got,) = session.run(None, {session.get_inputs()[0].name: x})
        with torch.no_grad():
            want = wrapper(torch.from_numpy(x))
        want = want if isinstance(want, torch.Tensor) else next(iter(want.values()))
        np.testing.assert_allclose(got, want.numpy(), atol=1e-4, rtol=1e-4)
