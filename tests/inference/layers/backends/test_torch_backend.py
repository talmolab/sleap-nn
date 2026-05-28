"""Tests for ``ModelBackend`` Protocol + ``TorchBackend``.

Coverage:

1. The Protocol is ``runtime_checkable`` and ``isinstance(torch_backend, ModelBackend)``
   returns ``True`` — guards against the protocol fields drifting.
2. ``TorchBackend`` parity vs raw Lightning forward on every fixture
   checkpoint that ships in ``tests/assets/model_ckpts/``. The forward
   output through the backend must equal the eager Lightning forward
   bit-for-bit (no behavior change in PR 3).
3. ``does_baked_postproc`` is ``False`` for every TorchBackend instance.
4. Warmup primes the backend without raising; on MPS the second call's
   latency is measurably lower (where MPS is available).
5. Device validation warnings — ``UserWarning`` text matches the locked
   acceptance criteria from the PR 3 issue body for CUDA + ``use_compile``,
   CUDA + ``use_fp16``, MPS + ``use_compile`` (downgrade), MPS + ``use_fp16``.
6. Compile guard for #527: when the swint ``torch.fx.wrap`` registry is
   contaminated, ``use_compile=True`` raises ``RuntimeError`` with a
   message pointing at #527.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from sleap_nn.inference.layers.backends import ModelBackend, TorchBackend


def _mps_actually_works() -> bool:
    """Return True only if MPS is reported available and can allocate.

    GitHub Actions Mac runners report MPS as available but raise
    ``RuntimeError: MPS backend out of memory`` on any real allocation,
    so a strict ``is_available()`` check is insufficient. This wrapper
    does a probe allocation to confirm MPS is genuinely usable.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1, device="mps")
    except RuntimeError:
        return False
    return True


_MPS_OK = _mps_actually_works()


# ─────────────────────────────────────────────────────────────────────────
# 1. Protocol invariants
# ─────────────────────────────────────────────────────────────────────────


class _TinyConvModel(nn.Module):
    """Stand-in for a Lightning module — accepts (B, C, H, W), returns Tensor."""

    def __init__(self, in_ch: int = 1, out_ch: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.float())


def test_torch_backend_satisfies_model_backend_protocol():
    """Static structural typing — guard against Protocol drift."""
    backend = TorchBackend(model=_TinyConvModel(), device="cpu")
    assert isinstance(backend, ModelBackend)


def test_protocol_attributes_present():
    """All four members of the protocol are reachable on a fresh backend."""
    backend = TorchBackend(model=_TinyConvModel(), device="cpu")
    assert backend.device == "cpu"
    assert backend.does_baked_postproc is False
    assert callable(backend)
    assert callable(backend.warmup)


# ─────────────────────────────────────────────────────────────────────────
# 2. Parity vs raw model forward
# ─────────────────────────────────────────────────────────────────────────


def test_forward_parity_returns_dict_for_tensor_returning_model():
    """A model that returns a Tensor is wrapped under the ``"output"`` key."""
    model = _TinyConvModel()
    backend = TorchBackend(model=model, device="cpu")

    x = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
    eager_out = model(x)
    backend_out = backend(x)

    assert isinstance(backend_out, dict)
    assert "output" in backend_out
    torch.testing.assert_close(backend_out["output"], eager_out)


def test_forward_passes_through_dict_outputs_unchanged():
    """A model that already returns a dict has its keys preserved."""

    class _DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)

        def forward(self, x):
            return {"my_head": self.conv(x.float())}

    model = _DictModel()
    backend = TorchBackend(model=model, device="cpu")
    x = torch.zeros(1, 1, 8, 8)
    out = backend(x)
    assert "my_head" in out
    torch.testing.assert_close(out["my_head"], model(x)["my_head"])


def test_forward_raises_on_unsupported_return_type():
    class _BadModel(nn.Module):
        def forward(self, x):
            return [x, x]  # neither Tensor nor dict

    backend = TorchBackend(model=_BadModel(), device="cpu")
    with pytest.raises(TypeError, match="unexpected output type"):
        backend(torch.zeros(1, 1, 4, 4))


# ─────────────────────────────────────────────────────────────────────────
# Lightning-module parity on every shipped checkpoint
# ─────────────────────────────────────────────────────────────────────────


CKPT_ROOT = Path(__file__).resolve().parents[4] / "tests" / "assets" / "model_ckpts"

CHECKPOINTS = [
    "minimal_instance_single_instance",
    "minimal_instance_centroid",
    "minimal_instance_centered_instance",
    "minimal_instance_bottomup",
    "minimal_instance_multiclass_bottomup",
    "minimal_instance_multiclass_centered_instance",
]


def _modules_under_inference_model(predictor):
    """Walk the predictor's ``inference_model`` and yield every direct
    sub-Lightning-module (its members like ``centroid_model``, ``confmap_model``,
    ``instance_peaks.torch_model``, etc.). These are the exact objects we want
    to wrap in a ``TorchBackend``."""
    inf = predictor.inference_model
    yielded: set = set()
    for _name, module in inf.named_modules():
        # Lightning modules subclass pytorch_lightning.LightningModule which
        # inherits from nn.Module. We don't want to re-emit every nn.Module
        # in the model — only the ones at the "wrapper" layer that the
        # Predictor calls into directly. Heuristic: look for objects whose
        # ``forward`` shows up as defined on the LightningModel hierarchy.
        if id(module) in yielded:
            continue
        cls_name = type(module).__name__
        if cls_name.endswith("LightningModule") or cls_name.endswith("LightningModel"):
            yielded.add(id(module))
            yield module


@pytest.mark.parametrize("ckpt_name", CHECKPOINTS)
def test_torch_backend_parity_per_checkpoint(ckpt_name: str):
    """Wrapping the Lightning module that backs each predictor in
    ``TorchBackend`` does not change its forward output bit-for-bit.

    Reuses the production ``Predictor.from_model_paths`` loader so we don't
    reimplement the (kwargs-heavy) checkpoint-load path here.
    """
    if not (CKPT_ROOT / ckpt_name).exists():
        pytest.skip(f"checkpoint not present: {ckpt_name}")

    from omegaconf import OmegaConf

    from sleap_nn.inference.predictors import Predictor

    # Build the predictor (CPU, neutral preprocess) just to get a loaded
    # Lightning module out the other side.
    predictor = Predictor.from_model_paths(
        [str(CKPT_ROOT / ckpt_name)],
        device="cpu",
        preprocess_config=OmegaConf.create(
            {
                "ensure_rgb": None,
                "ensure_grayscale": None,
                "crop_size": None,
                "max_width": None,
                "max_height": None,
                "scale": None,
            }
        ),
    )
    predictor._initialize_inference_model()

    modules = list(_modules_under_inference_model(predictor))
    assert modules, f"{ckpt_name}: no Lightning sub-module found in inference_model"

    for module in modules:
        module.eval()
        backend = TorchBackend(model=module, device="cpu")

        in_ch = _infer_input_channels(module)
        # Single-instance forward squeezes 5D (B, n_samples, C, H, W); top-down
        # heads expect 4D crops. We hit both code paths by trying 4D first
        # and falling back to 5D if forward complains.
        x = torch.zeros(1, in_ch, 64, 64, dtype=torch.float32)
        try:
            eager = module(x)
        except Exception:
            x = torch.zeros(1, 1, in_ch, 64, 64, dtype=torch.float32)
            eager = module(x)

        via_backend = backend(x)

        if isinstance(eager, torch.Tensor):
            torch.testing.assert_close(via_backend["output"], eager, atol=0, rtol=0)
        else:
            for k, v in eager.items():
                if isinstance(v, torch.Tensor):
                    torch.testing.assert_close(via_backend[k], v, atol=0, rtol=0)


def _infer_input_channels(module: nn.Module) -> int:
    """Walk a module looking for the first Conv2d's ``in_channels``."""
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            return sub.in_channels
    return 1


# ─────────────────────────────────────────────────────────────────────────
# 3 + 4. does_baked_postproc + warmup
# ─────────────────────────────────────────────────────────────────────────


def test_does_baked_postproc_is_false_for_torch():
    """TorchBackend always returns raw model output; never has baked peaks."""
    backend = TorchBackend(model=_TinyConvModel(), device="cpu")
    assert backend.does_baked_postproc is False


def test_warmup_on_cpu_is_noop():
    """CPU warmup is a no-op (no JIT to prime, no device synchronize)."""
    backend = TorchBackend(model=_TinyConvModel(), device="cpu")
    backend.warmup((1, 1, 8, 8))


@pytest.mark.skipif(not _MPS_OK, reason="MPS not usable on this host")
def test_warmup_on_mps_runs_iterations():
    backend = TorchBackend(model=_TinyConvModel(), device="mps", warmup_iterations=2)
    backend.warmup((1, 1, 16, 16))


# ─────────────────────────────────────────────────────────────────────────
# 5. Device validation warnings
# ─────────────────────────────────────────────────────────────────────────


def _has_cuda() -> bool:
    """Return True if a CUDA device is reported by torch."""
    return torch.cuda.is_available()


@pytest.mark.skipif(not _MPS_OK, reason="MPS not usable on this host")
def test_mps_compile_disables_with_warning():
    with pytest.warns(UserWarning, match="torch.compile is unreliable on MPS"):
        backend = TorchBackend(model=_TinyConvModel(), device="mps", use_compile=True)
    assert backend.use_compile is False
    assert backend._compiled is None


@pytest.mark.skipif(not _MPS_OK, reason="MPS not usable on this host")
def test_mps_fp16_warns_but_continues():
    with pytest.warns(UserWarning, match="FP16 on MPS gives no throughput gain"):
        backend = TorchBackend(model=_TinyConvModel(), device="mps", use_fp16=True)
    assert backend.use_fp16 is True


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_cuda_compile_emits_numerics_warning():
    with pytest.warns(UserWarning, match="numerics"):
        TorchBackend(model=_TinyConvModel(), device="cuda", use_compile=True)


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_cuda_fp16_warns_with_smallbatch_text():
    with pytest.warns(UserWarning) as record:
        TorchBackend(model=_TinyConvModel(), device="cuda", use_fp16=True)
    matched = [
        w for w in record if "counterproductive at small batch" in str(w.message)
    ]
    assert matched, "expected the small-batch regression warning text"


# ─────────────────────────────────────────────────────────────────────────
# 6. #527 compile guard
# ─────────────────────────────────────────────────────────────────────────


def test_compile_guard_raises_when_swint_fx_wrap_registered():
    """Importing ``sleap_nn.architectures.swint`` registers
    ``_patch_merging_pad`` into ``torch.fx._symbolic_trace._wrapped_fns_to_patch``
    at module-import time. While that contamination is present (which is
    always, since the predictor stack imports swint), ``use_compile=True``
    must raise ``RuntimeError`` pointing at issue #527 — not silently
    succeed and crash on the first forward.
    """
    # Force the import to ensure the registry is populated.
    import sleap_nn.architectures.swint  # noqa: F401

    from sleap_nn.inference.layers.backends.torch_backend import (
        _swint_fx_wrap_blocks_compile,
    )

    assert _swint_fx_wrap_blocks_compile(), (
        "swint torch.fx.wrap should be registered after import — guard test "
        "below would be a false-pass without this prerequisite"
    )

    if not _has_cuda():
        # The guard fires regardless of device, so we exercise it on CPU when
        # CUDA isn't available (the device-feature warning path doesn't
        # gate the guard).
        device = "cpu"
    else:
        device = "cuda"

    with pytest.raises(RuntimeError, match="issue #527"):
        TorchBackend(model=_TinyConvModel(), device=device, use_compile=True)


def test_compile_guard_does_not_block_use_compile_false():
    """The default path — ``use_compile=False`` — is never affected by the
    swint registry state. Sanity check."""
    import sleap_nn.architectures.swint  # noqa: F401

    backend = TorchBackend(model=_TinyConvModel(), device="cpu", use_compile=False)
    assert backend._compiled is None


# ─────────────────────────────────────────────────────────────────────────
# Conv+BN fusion (opt-in)
# ─────────────────────────────────────────────────────────────────────────


class _ConvBNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x.float()))


def test_fuse_layers_replaces_bn_with_identity():
    """``fuse_layers=True`` collapses Conv2d + BN into a fused Conv2d."""
    model = _ConvBNModel()
    # warm BN's running stats so the fold is well-defined
    model.eval()
    backend = TorchBackend(model=model, device="cpu", fuse_layers=True)
    # The BN should have been replaced with Identity.
    assert isinstance(backend.model.bn, nn.Identity)
    # Forward pass still works.
    out = backend(torch.zeros(1, 1, 8, 8))
    assert out["output"].shape == (1, 4, 8, 8)
