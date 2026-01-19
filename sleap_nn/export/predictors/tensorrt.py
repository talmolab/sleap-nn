"""TensorRT predictor for exported models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from sleap_nn.export.predictors.base import ExportPredictor


class TensorRTPredictor(ExportPredictor):
    """TensorRT inference for exported models.

    This predictor loads a native TensorRT engine file (.trt) and provides
    inference capabilities using CUDA for high-throughput predictions.

    Args:
        engine_path: Path to the TensorRT engine file (.trt).
        device: Device to run inference on (only "cuda" supported for TRT).

    Example:
        >>> predictor = TensorRTPredictor("model.trt")
        >>> outputs = predictor.predict(images)  # uint8 [B, C, H, W]
    """

    def __init__(
        self,
        engine_path: str | Path,
        device: str = "cuda",
    ) -> None:
        """Initialize TensorRT predictor with a serialized engine.

        Args:
            engine_path: Path to the TensorRT engine file.
            device: Device for inference ("cuda" or "auto"). TensorRT requires CUDA.
        """
        import tensorrt as trt

        if device not in ("cuda", "auto"):
            raise ValueError(f"TensorRT only supports CUDA devices, got: {device}")

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(self.engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, tuple] = {}
        self.output_shapes: Dict[str, tuple] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape

        self.device = torch.device("cuda")

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run TensorRT inference.

        Args:
            image: Input image(s) as numpy array [B, C, H, W] with uint8 dtype.

        Returns:
            Dict mapping output names to numpy arrays.
        """
        import tensorrt as trt

        # Convert input to torch tensor on GPU
        input_tensor = torch.from_numpy(image).to(self.device)

        # Check if engine expects uint8 or float32 and convert if needed
        input_name = self.input_names[0]
        expected_dtype = self.engine.get_tensor_dtype(input_name)
        if expected_dtype == trt.DataType.UINT8:
            # Engine expects uint8 - keep as uint8
            if input_tensor.dtype != torch.uint8:
                input_tensor = input_tensor.to(torch.uint8)
        else:
            # Engine expects float - convert uint8 to float32
            if input_tensor.dtype == torch.uint8:
                input_tensor = input_tensor.to(torch.float32)

        # Ensure contiguous memory
        input_tensor = input_tensor.contiguous()

        # Set input shape for dynamic dimensions
        input_name = self.input_names[0]
        self.context.set_input_shape(input_name, tuple(input_tensor.shape))

        # Allocate output tensors
        outputs: Dict[str, torch.Tensor] = {}
        bindings: Dict[str, int] = {}

        # Set input binding
        bindings[input_name] = input_tensor.data_ptr()

        # Allocate outputs
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            bindings[name] = outputs[name].data_ptr()

        # Set tensor addresses
        for name, ptr in bindings.items():
            self.context.set_tensor_address(name, ptr)

        # Run inference
        stream = torch.cuda.current_stream().cuda_stream
        success = self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()

        if not success:
            raise RuntimeError("TensorRT inference failed")

        # Convert outputs to numpy
        return {name: tensor.cpu().numpy() for name, tensor in outputs.items()}

    def benchmark(
        self, image: np.ndarray, n_warmup: int = 50, n_runs: int = 200
    ) -> Dict[str, float]:
        """Benchmark inference performance.

        Args:
            image: Input image(s) as numpy array [B, C, H, W].
            n_warmup: Number of warmup runs (not timed).
            n_runs: Number of timed runs.

        Returns:
            Dict with timing statistics:
                - mean_ms: Mean inference time in milliseconds
                - std_ms: Standard deviation of inference time
                - min_ms: Minimum inference time
                - max_ms: Maximum inference time
                - fps: Frames per second (based on mean time and batch size)
        """
        batch_size = image.shape[0]

        # Warmup
        for _ in range(n_warmup):
            _ = self.predict(image)

        # Timed runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.predict(image)
            times.append((time.perf_counter() - start) * 1000)

        times_arr = np.array(times)
        mean_ms = float(np.mean(times_arr))

        return {
            "mean_ms": mean_ms,
            "std_ms": float(np.std(times_arr)),
            "min_ms": float(np.min(times_arr)),
            "max_ms": float(np.max(times_arr)),
            "fps": (batch_size * 1000) / mean_ms if mean_ms > 0 else 0.0,
        }

    @staticmethod
    def _trt_dtype_to_torch(trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        import tensorrt as trt

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)


class TensorRTEngine:
    """Low-level wrapper for native TensorRT engine files (.trt).

    Provides a callable interface similar to PyTorch models, returning
    torch.Tensor outputs directly.

    Args:
        engine_path: Path to the TensorRT engine file (.trt).

    Example:
        >>> engine = TensorRTEngine("model.trt")
        >>> input_tensor = torch.randn(1, 1, 512, 512, device="cuda")
        >>> outputs = engine(input_tensor)  # Dict[str, torch.Tensor]
    """

    def __init__(self, engine_path: str | Path) -> None:
        """Initialize TensorRT engine wrapper.

        Args:
            engine_path: Path to the serialized TensorRT engine file.
        """
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, tuple] = {}
        self.output_shapes: Dict[str, tuple] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            *args: Input tensors (positional) or
            **kwargs: Input tensors by name

        Returns:
            Dict of output tensors on the same device as input.
        """
        import tensorrt as trt

        # Handle inputs
        if args:
            inputs = {self.input_names[i]: arg for i, arg in enumerate(args)}
        else:
            inputs = kwargs

        # Set input shapes and allocate outputs
        outputs: Dict[str, torch.Tensor] = {}
        bindings: Dict[str, int] = {}

        for name in self.input_names:
            tensor = inputs[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Input {name} must be a torch.Tensor")
            # Set actual shape for dynamic dimensions
            self.context.set_input_shape(name, tuple(tensor.shape))
            bindings[name] = tensor.contiguous().data_ptr()

        # Allocate output tensors
        device = next(iter(inputs.values())).device
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(tuple(shape), dtype=dtype, device=device)
            bindings[name] = outputs[name].data_ptr()

        # Set tensor addresses
        for name, ptr in bindings.items():
            self.context.set_tensor_address(name, ptr)

        # Run inference
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()

        return outputs

    @staticmethod
    def _trt_dtype_to_torch(trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        import tensorrt as trt

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)
