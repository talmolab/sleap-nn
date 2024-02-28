"""Instance classes for metadata."""

import attrs
from typing import List, Optional
import numpy as np

class Instance:
    """A single instance in a frame."""

@attrs.define(auto_attribs=True)
class MatchedFrameInstances:
    t: int
    instances_t: List[Instance]
    img_t: Optional[np.ndarray] = None