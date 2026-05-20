"""Lightning ``XPUAccelerator`` for Intel GPUs via native ``torch.xpu``.

Background
----------
PyTorch ≥ 2.5 ships native Intel GPU (XPU) support via the ``torch.xpu``
module. PyTorch Lightning, however, does **not** ship an
``XPUAccelerator`` in any released version through 2.6.x; multiple
attempts to add one have stalled or been closed without merging
(Lightning-AI/pytorch-lightning#16834, #17700, #19443, #20349). The
tracking issue is Lightning-AI/pytorch-lightning#20938. Historically
``intel-extension-for-pytorch`` (IPEX) registered an XPU accelerator
with Lightning as a side-effect, but Intel discontinued IPEX after v2.8
(EOL end of March 2026) and directs users to native ``torch.xpu``.

The Lightning docs (`extensions/accelerator.html`_) suggest writing a
custom ``Accelerator`` subclass and registering it. That is exactly what
this module does. Importing the module registers the accelerator with
Lightning's ``AcceleratorRegistry`` as ``"xpu"``.

.. _extensions/accelerator.html: https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html

Strategy
--------
Even with an ``XPUAccelerator`` registered, Lightning's strategy chooser
(``accelerator_connector._choose_strategy``) hardcodes CUDA/MPS as the
only "GPU" types and falls back to a CPU ``SingleDeviceStrategy`` for
anything else. The fix lives in ``model_trainer.py``: when the user
selects ``accelerator="xpu"`` and leaves ``strategy="auto"``, we
explicitly construct ``SingleDeviceStrategy(device=torch.device("xpu",
N))`` and pass it to ``L.Trainer``. We do **not** monkey-patch Lightning
at runtime.

If/when Lightning ships native XPU support, this module should become a
no-op (or be deleted entirely).
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch
from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry


class XPUAccelerator(Accelerator):
    """Lightning Accelerator wrapping ``torch.xpu``.

    Mirrors the shape of ``lightning.pytorch.accelerators.CUDAAccelerator``
    closely enough that the Trainer treats XPU like any other single-device
    GPU accelerator. Multi-XPU / DDP is out of scope here.
    """

    def setup_device(self, device: torch.device) -> None:
        """Pin the current process to ``device`` (must be an XPU)."""
        if device.type != "xpu":
            raise RuntimeError(f"Device should be XPU, got {device}")
        torch.xpu.set_device(device)

    def get_device_stats(self, device) -> dict[str, Any]:
        """Return ``torch.xpu.memory_stats(device)`` if available."""
        try:
            return torch.xpu.memory_stats(device)
        except Exception:
            return {}

    def teardown(self) -> None:
        """Empty the XPU cache on teardown."""
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Parse the ``devices=`` flag passed to the Trainer into a list of ints."""
        if isinstance(devices, list):
            return devices
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            if devices in ("auto", "-1"):
                return list(range(torch.xpu.device_count()))
            try:
                return [int(x) for x in devices.split(",") if x.strip()]
            except ValueError:
                return [int(devices)]
        return None

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Map a list of XPU indices to ``torch.device`` objects."""
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Number of XPUs visible to torch, or 0 if torch.xpu is unavailable."""
        return torch.xpu.device_count() if torch.xpu.is_available() else 0

    @staticmethod
    def is_available() -> bool:
        """Whether at least one XPU is visible to torch."""
        return torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        """Accelerator name registered with Lightning."""
        return "xpu"

    @classmethod
    def register_accelerators(cls, accelerator_registry) -> None:
        """Register this accelerator under the name ``xpu``."""
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )


# Register on import so that `Trainer(accelerator="xpu")` resolves. Safe to
# import on a no-XPU box — `is_available()` is checked by Lightning before
# the accelerator is actually used.
if "xpu" not in AcceleratorRegistry.keys():
    XPUAccelerator.register_accelerators(AcceleratorRegistry)
