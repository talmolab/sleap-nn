"""Runtime backends for inference layers.

Currently exported:

- :class:`ModelBackend` — Protocol every backend implements.
- :class:`TorchBackend` — PyTorch ``nn.Module`` runtime with optional
  compile / FP16 / Conv-BN fusion.

PR 7 (#515) adds ``ONNXBackend`` and ``TensorRTBackend``.
"""

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.backends.torch_backend import TorchBackend

__all__ = ["ModelBackend", "TorchBackend"]
