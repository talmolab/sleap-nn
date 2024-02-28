"""Base Tracker class."""

import attrs
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from sleap_nn.tracking.core.instance import Instance

@attrs.define(auto_attribs=True)
class BaseTracker(ABC):
    """Abstract base class for tracker."""

    @property
    def is_valid(self):
        return False

    @abstractmethod
    def track(
        self,
        untracked_instances: List[Instance],
        img: Optional[np.ndarray] = None,
        t: int = None,
    ):
        pass

    @property
    @abstractmethod
    def uses_image(self):
        pass

    # @abstractmethod
    # def final_pass(self, frames: List[LabeledFrame]):
    #     pass

    @abstractmethod
    def get_name(self):
        pass