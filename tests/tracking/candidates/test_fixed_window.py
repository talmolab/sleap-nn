from typing import DefaultDict, Deque
from sleap_nn.tracking.candidates.fixed_window import FixedWindowCandidates


def test_fixed_window_candidates():
    fixed_window_candidates = FixedWindowCandidates(8, 20)
    assert isinstance(fixed_window_candidates.tracker_queue, DefaultDict)
