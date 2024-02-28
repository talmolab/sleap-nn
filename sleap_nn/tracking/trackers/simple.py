"""SimpleTracker."""

import attrs
from typing import Optional, List, Callable, Deque
import numpy as np

from sleap_nn.tracking.core.instance import Instance, MatchedFrameInstances
from sleap_nn.tracking.candidate.simple import SimpleCandidateMaker

class SimpleTracker:

    pre_cull_function: Optional[Callable] = None
    candidate_maker: object = attrs.field(factory=SimpleCandidateMaker)
    track_matching_queue: Deque[MatchedFrameInstances] = attrs.field()

    def track(
        self,
        untracked_instances: List[Instance],
        img: Optional[np.ndarray] = None,
        t: int = None,
    ) -> List[Instance]:
        """Performs a single step of tracking.

        Args:
            untracked_instances: List of instances to assign to tracks.
            img: Image data of the current frame for flow shifting.
            t: Current timestep. If not provided, increments from the internal queue.

        Returns:
            A list of the instances that were tracked.
        """

        if self.candidate_maker is None:
            return untracked_instances

        # Infer timestep if not provided.
        if t is None:
            if len(self.track_matching_queue) > 0:

                # Default to last timestep + 1 if available.
                t = self.track_matching_queue[-1].t + 1

            else:
                t = 0

        # Initialize containers for tracked instances at the current timestep.
        tracked_instances = []

        # Make cache so similarity function doesn't have to recompute everything.
        # similarity_cache = dict()

        # Process untracked instances.
        if untracked_instances:

            if self.pre_cull_function:
                self.pre_cull_function(untracked_instances)

            # Build a pool of matchable candidate instances.
            candidate_instances = self.candidate_maker.get_candidates(
                track_matching_queue=self.track_matching_queue,
                t=t,
                img=img,
            )

            # Determine matches for untracked instances in current frame.
            frame_matches = FrameMatches.from_candidate_instances(
                untracked_instances=untracked_instances,
                candidate_instances=candidate_instances,
                similarity_function=self.similarity_function,
                matching_function=self.matching_function,
                robust_best_instance=self.robust_best_instance,
            )

            # Store the most recent match data (for outside inspection).
            self.last_matches = frame_matches

            # Set track for each of the matched instances.
            tracked_instances.extend(
                self.update_matched_instance_tracks(frame_matches.matches)
            )

            # Spawn a new track for each remaining untracked instance.
            tracked_instances.extend(
                self.spawn_for_untracked_instances(frame_matches.unmatched_instances, t)
            )

        # Add the tracked instances to the matching buffer.
        self.track_matching_queue.append(
            MatchedFrameInstances(t, tracked_instances, img)
        )

        # Save tracked instances internally.
        if self.save_tracked_instances:
            self.tracked_instances[t] = tracked_instances

        return tracked_instances
