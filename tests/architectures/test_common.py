import pytest
import torch

from sleap_nn.architectures.common import MaxPool2dWithSamePadding


def test_maxpool2d_with_same_padding():
    pooling = MaxPool2dWithSamePadding(
        kernel_size=3, stride=2, dilation=2, padding="same"
    )

    x = torch.rand(1, 10, 100, 100)
    z = pooling(x)
    assert z.shape == (1, 10, 50, 50)


def _same_pad_reference(i, k, s, d):
    # The tensor-based formula this layer used before (#783).
    return int(
        max((torch.ceil(torch.tensor(i / s)).item() - 1) * s + (k - 1) * d + 1 - i, 0)
    )


def test_maxpool2d_same_padding_unchanged():
    """``_calc_same_pad`` gives the same padding as the old tensor formula."""
    pooling = MaxPool2dWithSamePadding(kernel_size=2, stride=2, padding="same")
    for i in range(1, 2049):
        for k in (1, 2, 3, 5):
            for s in (1, 2, 3, 4):
                for d in (1, 2):
                    assert pooling._calc_same_pad(i, k, s, d) == _same_pad_reference(
                        i, k, s, d
                    ), (i, k, s, d)


def test_maxpool2d_same_padding_compiles_as_one_graph():
    """No ``Tensor.item()`` in the forward, so dynamo traces it in one graph (#783)."""
    pooling = MaxPool2dWithSamePadding(kernel_size=2, stride=2, padding="same")
    x = torch.rand(2, 4, 81, 160)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(pooling)(x)
    assert explanation.graph_break_count == 0, explanation.break_reasons

    torch._dynamo.reset()
    compiled = torch.compile(pooling, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), pooling(x))
    torch._dynamo.reset()
