"""Identity linking: object-level long-term memory for tracked videos.

:func:`link_tracklets` runs once per video AFTER tracking (any tracker, the
``classic`` one included). It treats every track as an identity made of
tracklets and:

1. **Corrects swaps at crossings** (needs calibrated appearance): around each
   ambiguity event (two or more identities' boxes overlapping), it compares each
   identity's appearance gallery from before the event with the vectors right
   after it, and applies a permutation only when appearance prefers it by the
   calibrated margin.
2. **Links tracklets across gaps** (re-identification): an identity that ends
   and one that starts later, that never co-occur (cannot-link) and are
   spatially plausible given the gap, are merged when their appearance agrees
   (calibrated threshold and margin), or -- without usable appearance -- when the
   match is spatially unique within a short gap.
3. **Enforces a closed set** when the number of animals is known: keeps merging
   the best feasible pairs until at most ``max_tracks`` identities remain;
   identities that co-occur with all others (duplicates / false positives) are
   untracked, duplicates first.

Every vector already exists when tracking runs in sleap-nn (offline flow), so
linking decides with full tracklets on both sides of every gap -- more evidence
than an online re-ID decision at track confirmation could use.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import attrs
import sleap_io as sio

from sleap_nn.tracking.mot.calibration import VideoCalibration


@attrs.define
class LinkingResult:
    """Outcome of :func:`link_tracklets` for one video.

    Attributes:
        n_identities_before: Identities (tracks) before linking.
        n_identities: Identities after linking.
        n_merges: Gap links applied (appearance or spatial).
        n_forced_merges: Closed-set merges applied below the open-set rule.
        n_swaps: Swap corrections applied.
        n_untracked: Detections set to untracked by the closed-set rule.
        notes: Human-readable decisions worth logging.
    """

    n_identities_before: int = 0
    n_identities: int = 0
    n_merges: int = 0
    n_forced_merges: int = 0
    n_swaps: int = 0
    n_untracked: int = 0
    notes: List[str] = attrs.field(factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict."""
        return attrs.asdict(self)


def link_tracklets(
    frames: Sequence[Tuple[int, Sequence[Any]]],
    *,
    carrier: str = "pose",
    calibration: Optional[VideoCalibration] = None,
    max_tracks: Optional[int] = None,
    track_buffer: int = 30,
    swap_correction: bool = True,
    spatial_gate_frames: int = 300,
    box_padding: float = 0.1,
    high_threshold: float = 0.5,
    score_mode: str = "auto",
    track_objects: Optional[Dict[int, sio.Track]] = None,
) -> LinkingResult:
    """Link a tracked video's identities across gaps and crossings, in place.

    Args:
        frames: ``(frame_idx, objects)`` per frame in temporal order: the tracked
            detections (``.track`` set or ``None``) of ONE video.
        carrier: ``"pose"`` or ``"mask"``.
        calibration: The video's calibration (computes one from ``frames`` if
            ``None``). Appearance is used only where it reports
            ``appearance_for_linking``.
        max_tracks: Known number of animals: closed-set linking.
        track_buffer: Gap (steps) below which a track's detections form one
            tracklet; also scales the spatial-only rule.
        swap_correction: Correct swaps at crossings (appearance only).
        spatial_gate_frames: Gaps longer than this (steps) are not spatially gated.
        box_padding: Box padding used for overlap tests.
        high_threshold: Minimum score for a vector to enter a gallery.
        score_mode: Detection score mode (see ``MOTParams.score_mode``).
        track_objects: The tracker's ``{public_id: sio.Track}``; updated to drop
            merged-away / untracked identities when given.

    Returns:
        A :class:`LinkingResult`. Merged identities keep the EARLIER identity's
        ``sio.Track``. Never raises on degenerate input.
    """
    raise NotImplementedError
