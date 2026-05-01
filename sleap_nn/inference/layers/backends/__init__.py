"""Runtime backends for inference layers.

Currently exported:

- :class:`ModelBackend` — Protocol every backend implements.
- :class:`TorchBackend` — PyTorch ``nn.Module`` runtime with optional
  compile / FP16 / Conv-BN fusion.
- :class:`ONNXBackend` — ONNX Runtime backend (PR 7 / #515). Wraps an
  exported ``.onnx`` file; peak finding is baked into the graph.

PR 7 also adds ``TensorRTBackend`` (CUDA-only, requires ``tensorrt``
extra) — landing as a follow-up commit on the same branch.
"""

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.backends.onnx_backend import ONNXBackend
from sleap_nn.inference.layers.backends.torch_backend import TorchBackend

__all__ = ["ModelBackend", "ONNXBackend", "TorchBackend"]
