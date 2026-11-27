"""Unit tests for offline inference performance optimizations."""

from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# torch.compile Power-of-2 Bucket Ladder
# ─────────────────────────────────────────────────────────────────────────────


def test_torch_backend_power_of_two_bucket_ladder_and_burst_dropback(monkeypatch):
    """A single burst frame never locks padding permanently; >cap chunks cleanly."""
    import torch._dynamo

    from sleap_nn.inference.layers.backends.torch_backend import (
        COMPILE_BATCH_BUCKETS,
        TorchBackend,
    )

    class _LinearNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x.float())

    real_compile = torch.compile
    compiled_shapes = []

    def counting_backend(gm, example_inputs):
        compiled_shapes.append(
            next(
                tuple(t.shape)
                for t in example_inputs
                if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[-1] in (4, 16)
            )
        )
        return gm.forward

    def fake_compile(model, mode=None, dynamic=None):
        return real_compile(model, backend=counting_backend, dynamic=dynamic)

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", fake_compile)
    try:
        torch.manual_seed(0)
        net = _LinearNet()
        eager = TorchBackend(model=_LinearNet(), device="cpu")
        eager.model.load_state_dict(net.state_dict())
        compiled = TorchBackend(model=net, device="cpu", use_compile=True)
        assert compiled.compile_batch_buckets == COMPILE_BATCH_BUCKETS

        # Sequence with a big crowd burst (150 -> chunked 64+64+32, _padded_batch=64)
        # followed by small batches (5 -> 8, 1 -> 1) and another burst (270 -> chunked).
        seq = [(5, 8), (150, 64), (40, 64), (5, 8), (1, 1), (270, 64), (7, 8)]
        for n, expected_padded in seq:
            x = torch.randn(n, 1, 4, 4)
            out = compiled(x)["output"]
            assert out.shape == (n, 2, 4, 4)
            assert compiled._padded_batch == expected_padded
            torch.testing.assert_close(out, eager(x)["output"])

        # Only in-ladder buckets (<= 64) were compiled: 8 (for 5, 7), 64 (chunks),
        # 32 (for 150 % 64 = 22), 1 (for 1), and 16 (for 270 % 64 = 14).
        # No oversized 128/256/270 graph was ever compiled, and small batches
        # dropped cleanly back down to 8 and 1 after bursts!
        assert compiled_shapes == [
            (8, 1, 4, 4),
            (64, 1, 4, 4),
            (32, 1, 4, 4),
            (1, 1, 4, 4),
            (16, 1, 4, 4),
        ]
        assert torch._dynamo.config.cache_size_limit >= 64
        if hasattr(torch._dynamo.config, "recompile_limit"):
            assert torch._dynamo.config.recompile_limit >= 64

        # Verify default mode compiles with dynamic=True (single symbolic graph
        # across buckets >= 2, preventing mid-clip recompiles on tail/burst batches).
        torch._dynamo.reset()
        compiled_shapes.clear()
        dyn_compiled = TorchBackend(
            model=net, device="cpu", use_compile=True, compile_mode="default"
        )
        for n in [8, 4, 9, 24, 32, 150]:
            x = torch.randn(n, 1, 16, 16)
            out = dyn_compiled(x)["output"]
            assert out.shape == (n, 2, 16, 16)
            torch.testing.assert_close(out, eager(x)["output"])
        # Buckets 32 and 64 reuse the symbolic graph without recompiling
        # (CPU conv2d has an internal batch<=15 vs >=16 dispatch split, so <=2
        # graphs across 5 distinct bucket sizes {4, 8, 16, 32, 64}).
        assert len(compiled_shapes) <= 2
        assert (32, 1, 16, 16) not in compiled_shapes
        assert (64, 1, 16, 16) not in compiled_shapes
    finally:
        torch._dynamo.reset()
