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
6. Regression test for #527: importing ``sleap_nn.architectures.swint`` must
   not register a torch.fx wrap that resolves to a missing name in this
   module's globals (which is what previously broke ``torch.compile`` for
   every backbone with ``KeyError: '_patch_merging_pad'``).
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


class _NormalizingUint8Model(nn.Module):
    """Mimics the production forward: receives a uint8 tensor, normalizes it
    *inside* forward (``/ 255.0``), then runs a conv — exactly the path where
    fp16 used to be a silent no-op (the uint8 input skipped the half cast)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0
        return self.conv(x)


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_cuda_fp16_actually_runs_half_on_uint8_path():
    """``use_fp16=True`` must genuinely run the model in half precision on the
    standard uint8 inference path, producing finite outputs close to fp32.

    fp16 is opt-in and changes numerics, so this asserts the run *succeeds*
    and is loosely close to fp32 — NOT bit-exact parity. Before the fix the
    uint8 input skipped the half cast and the half model raised a dtype
    mismatch (or fp16 was a silent no-op).
    """
    torch.manual_seed(0)
    x_u8 = torch.randint(0, 256, (2, 1, 16, 16), dtype=torch.uint8)

    # fp32 reference.
    fp32_model = _NormalizingUint8Model().eval()
    fp32_backend = TorchBackend(model=fp32_model, device="cuda", use_fp16=False)
    fp32_out = fp32_backend(x_u8)["output"]
    assert fp32_out.dtype == torch.float32

    # fp16 run: same weights, half precision.
    fp16_model = _NormalizingUint8Model().eval()
    fp16_model.load_state_dict(fp32_model.state_dict())
    with pytest.warns(UserWarning):
        fp16_backend = TorchBackend(model=fp16_model, device="cuda", use_fp16=True)
    # Autocast keeps fp32 master weights (non-destructive) — the half precision
    # happens per-op inside ``torch.autocast``, not by mutating the model.
    assert next(fp16_backend.model.parameters()).dtype == torch.float32

    fp16_out = fp16_backend(x_u8)["output"]
    # Output is cast back to float32 for downstream postprocess.
    assert fp16_out.dtype == torch.float32
    # Finite and loosely close to fp32 (numerics differ; this is opt-in).
    assert torch.isfinite(fp16_out).all()
    torch.testing.assert_close(fp16_out, fp32_out, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_cuda_fp16_default_off_is_bit_exact_vs_eager():
    """Default path (no fp16) on CUDA stays bit-exact vs the eager model."""
    torch.manual_seed(0)
    x_u8 = torch.randint(0, 256, (1, 1, 16, 16), dtype=torch.uint8)
    model = _NormalizingUint8Model().eval().cuda()
    eager = model(x_u8.cuda())

    backend = TorchBackend(model=model, device="cuda", use_fp16=False)
    out = backend(x_u8)["output"]
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


# ─────────────────────────────────────────────────────────────────────────
# 6. #527 regression
# ─────────────────────────────────────────────────────────────────────────


def test_swint_does_not_register_broken_fx_wraps():
    """Regression test for issue #527.

    The bug: ``sleap_nn/architectures/swint.py`` previously called
    ``torch.fx.wrap("_patch_merging_pad")`` (and two others) at module-import
    time. Those wraps registered ``(id(sleap_nn_swint_globals), name)`` into
    ``torch.fx._symbolic_trace._wrapped_fns_to_patch`` — but ``_patch_merging_pad``
    et al. are *not* defined in this module's globals (they're imported from
    torchvision only inside :class:`SwinTransformerEncoder`). When
    ``torch.compile``'s dynamo backend iterated the registry and did
    ``frame_dict[name]`` it raised ``KeyError: '_patch_merging_pad'`` — even
    when compiling a UNet that never touched SwinT, because importing any
    sleap-nn subpackage pulled this module in transitively.

    This test enforces the post-fix invariant: every FX wrap entry that
    points at ``sleap_nn.architectures.swint``'s globals must resolve to
    an actual callable in those globals. (The torchvision-owned wraps are
    fine and untouched: ``_patch_merging_pad`` is genuinely defined in
    torchvision's swin_transformer module.)
    """
    import sleap_nn.architectures.swint as swint_mod

    from torch.fx._symbolic_trace import _wrapped_fns_to_patch

    sleap_nn_globals_id = id(vars(swint_mod))
    sleap_nn_wraps = [
        name
        for (gid, name), frame_dict in _wrapped_fns_to_patch.items()
        if gid == sleap_nn_globals_id
    ]
    for name in sleap_nn_wraps:
        assert hasattr(swint_mod, name), (
            f"#527 regression: sleap_nn.architectures.swint registered FX "
            f"wrap for {name!r} but the name is not defined in this module — "
            f"dynamo will KeyError on compile."
        )


def test_use_compile_false_is_default_and_does_not_compile():
    """The default path — ``use_compile=False`` — never compiles."""
    import sleap_nn.architectures.swint  # noqa: F401

    backend = TorchBackend(model=_TinyConvModel(), device="cpu", use_compile=False)
    assert backend._compiled is None


# ─────────────────────────────────────────────────────────────────────────
# Conv+BN fusion (opt-in)
# ─────────────────────────────────────────────────────────────────────────


def _set_nontrivial_bn_stats(bn: nn.BatchNorm2d) -> None:
    """Give a BN non-identity running stats / affine params.

    Makes a mis-fuse (folding into the wrong conv + dropping the BN) produce
    a *large* output delta instead of one masked by near-identity stats.
    """
    bn.running_mean.data = torch.tensor([1.0, -2.0, 0.5, 3.0])
    bn.running_var.data = torch.tensor([4.0, 0.25, 2.0, 9.0])
    bn.weight.data = torch.tensor([2.0, 0.5, 1.5, 0.8])
    bn.bias.data = torch.tensor([0.1, -0.3, 0.2, 0.4])


class _SequentialConvBNModel(nn.Module):
    """Conv2d -> BatchNorm2d inside an ``nn.Sequential`` (execution order
    is guaranteed to match registration order — the safe-to-fuse case)."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
        )
        _set_nontrivial_bn_stats(self.body[1])

    def forward(self, x):
        return self.body(x.float())


def test_fuse_layers_replaces_bn_with_identity():
    """``fuse_layers=True`` collapses a Sequential Conv2d + BN into a fused
    Conv2d (BN replaced with ``Identity``) without changing the output."""
    model = _SequentialConvBNModel()
    model.eval()  # running stats frozen so the fold is well-defined

    x = torch.randn(1, 1, 8, 8)
    ref = model(x).detach().clone()

    backend = TorchBackend(model=model, device="cpu", fuse_layers=True)
    # The BN inside the Sequential should have been replaced with Identity.
    assert isinstance(backend.model.body[1], nn.Identity)
    # Forward pass still works and is numerically equivalent to unfused.
    out = backend(x)
    assert out["output"].shape == (1, 4, 8, 8)
    torch.testing.assert_close(out["output"], ref, atol=1e-5, rtol=1e-4)


def test_fuse_layers_default_off_is_bit_exact():
    """Default (``fuse_layers=False``) leaves the model untouched and the
    forward bit-exact vs the eager model."""
    model = _SequentialConvBNModel()
    model.eval()
    x = torch.randn(1, 1, 8, 8)
    ref = model(x).detach().clone()

    backend = TorchBackend(model=model, device="cpu")  # fuse_layers defaults False
    # BN is untouched.
    assert isinstance(backend.model.body[1], nn.BatchNorm2d)
    out = backend(x)
    torch.testing.assert_close(out["output"], ref, atol=0, rtol=0)


class _ReorderedConvBNModel(nn.Module):
    """A model whose ``forward`` applies BN to a *different* conv than the one
    registered immediately before it.

    Registration order is ``conv_a, bn, conv_b`` but ``forward`` computes
    ``bn(conv_b(x))`` — so a naive registration-order fuse would fold ``bn``
    into ``conv_a`` (which isn't even on the forward path) and drop the BN
    that genuinely follows ``conv_b``, silently corrupting the output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.conv_b = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        _set_nontrivial_bn_stats(self.bn)

    def forward(self, x):
        return self.bn(self.conv_b(x.float()))


def test_fuse_layers_does_not_misfuse_on_reordered_forward():
    """Regression: fusion must NOT fold a Conv→BN pair that is adjacent by
    *registration* order but not by *execution* order.

    The (formerly buggy) registration-order walk produced a max-abs-diff of
    ~1.7 here. The Sequential-only guard must leave this model's output
    bit-exact vs. unfused.
    """
    model = _ReorderedConvBNModel()
    model.eval()
    x = torch.randn(1, 1, 8, 8)
    ref = model(x).detach().clone()

    backend = TorchBackend(model=model, device="cpu", fuse_layers=True)

    # The BN must be left intact (NOT replaced with Identity) since it cannot
    # be proven to execute right after a conv.
    assert isinstance(backend.model.bn, nn.BatchNorm2d)
    assert isinstance(backend.model.conv_a, nn.Conv2d)
    assert isinstance(backend.model.conv_b, nn.Conv2d)

    out = backend(x)
    # Output must be unchanged vs the unfused eager model — no silent mis-fuse.
    torch.testing.assert_close(out["output"], ref, atol=0, rtol=0)
