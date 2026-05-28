"""Inference layers — model-type-aware wrappers around a runtime backend.

PR 3 (#511) ships the ``ModelBackend`` protocol + ``TorchBackend`` that
every layer delegates its forward pass to. PR 4 (#512) adds
``InferenceLayer`` (ABC) + ``SingleInstanceLayer`` (the proof-of-pattern).

Layers are model-type-aware (peak finding, NMS, multi-class identity
grouping). Backends are runtime-aware (PyTorch, ONNX, TensorRT). Crossing
the two gives 6 × 3 = 18 conceptual variants — but with this protocol-based
split we only ship 6 + 3 = 9 classes total, with zero duplication.
"""

from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer

__all__ = [
    "InferenceLayer",
    "PostprocessConfig",
    "PreprocessConfig",
    "SingleInstanceLayer",
]
