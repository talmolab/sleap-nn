"""Runtime backends for inference layers.

Exported:

- :class:`ModelBackend` — Protocol every backend implements.
- :class:`TorchBackend` — PyTorch ``nn.Module`` runtime with optional
  compile / FP16 / Conv-BN fusion.
- :class:`ONNXBackend` — ONNX Runtime backend. Wraps an exported
  ``.onnx`` file; peak finding is baked into the graph.
- :class:`TensorRTBackend` — TensorRT backend (CUDA-only, requires
  ``tensorrt`` extra).
"""

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.backends.onnx_backend import ONNXBackend
from sleap_nn.inference.layers.backends.tensorrt_backend import TensorRTBackend
from sleap_nn.inference.layers.backends.torch_backend import TorchBackend

__all__ = ["ModelBackend", "ONNXBackend", "TensorRTBackend", "TorchBackend"]
