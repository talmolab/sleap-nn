"""Candidate selector/maker for the SimpleTracker."""

import attrs
from typing import List
from typing import Deque

from sleap_nn.tracking.core.instance import MatchedFrameInstances, Instance
from sleap_nn.tracking.candidate.base import BaseCandidateMaker

@attrs.define(auto_attribs=True)
class SimpleCandidateMaker(BaseCandidateMaker):
    """Class for producing list of matching candidates from prior frames."""

    min_points: int = 0

    @property
    def uses_image(self):
        return False

    def get_candidates(
        self, track_matching_queue: Deque[MatchedFrameInstances], *args, **kwargs
    ) -> List[Instance]:
        # Build a pool of matchable candidate instances.
        candidate_instances = []
        for matched_item in track_matching_queue:
            _, ref_instances = matched_item.t, matched_item.instances_t
            for ref_instance in ref_instances:
                if ref_instance.n_visible_points >= self.min_points:
                    candidate_instances.append(ref_instance)
        return candidate_instances
