"""Base candidate selector/maker class."""

import attrs
from typing import Deque, List

from sleap_nn.tracking.core.instance import MatchedFrameInstances, Instance

@attrs.define(auto_attribs=True)
class BaseCandidateMaker:
    """Base class for producing list of matching candidates from prior frames."""

    min_points: int = 0

    @property
    def uses_image(self):
        return False

    def get_candidates(
        self, track_matching_queue: Deque[MatchedFrameInstances], *args, **kwargs
    ) -> List[Instance]:
        pass
