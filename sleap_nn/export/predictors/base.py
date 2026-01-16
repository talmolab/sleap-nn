"""Predictor base class for exported models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class ExportPredictor(ABC):
    """Base interface for exported model inference."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a batch of images."""

    @abstractmethod
    def benchmark(
        self, image: np.ndarray, n_warmup: int = 50, n_runs: int = 200
    ) -> Dict[str, float]:
        """Benchmark inference latency and throughput."""
