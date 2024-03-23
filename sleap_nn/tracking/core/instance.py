"""Instance classes for metadata."""

import attrs
from typing import List, Optional
import numpy as np

from sleap_nn.tracking.core.track import Track

class Instance:
    track: Track
    n_visible_points: int
    bounding_box: np.ndarray

@attrs.define(auto_attribs=True)
class MatchedFrameInstances:
    t: int
    instances_t: List[Instance]
    img_t: Optional[np.ndarray] = None