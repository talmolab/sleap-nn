"""Matches."""

import attrs
import numpy as np
from typing import List, Optional, Callable
from collections import defaultdict

@attrs.define(auto_attribs=True)
class Match:
    """Stores a match between a specific instance and specific track."""

    track: Track
    instance: Instance
    score: Optional[float] = None
    is_first_choice: bool = False


@attrs.define(auto_attribs=True)
class FrameMatches:
    """
    Calculates (and stores) matches for a frame.

    This class encapsulates the logic to generate matches (using a custom
    matching function) from a cost matrix. One key feature is that it retains
    additional information, such as whether all the matches were first-choice
    (i.e., if each instance got the instance it would have matched to if there
    weren't other instances).

    Typically this will be created using the `from_candidate_instances` method
    which creates the cost matrix and then uses the matching function to find
    matches.

    Attributes:
        matches: the list of `Match` objects.
        cost_matrix: the cost matrix, shape is
            (number of untracked instances, number of candidate tracks).
        unmatched_instances: the instances for which we are finding matches.

    """

    matches: List[Match]
    cost_matrix: np.ndarray
    unmatched_instances: List[InstanceType] = attrs.field(factory=list)

    @property
    def has_only_first_choice_matches(self) -> bool:
        """Whether all the matches were first-choice.

        A match is a 'first-choice' for an instance if that instance would have
        matched to the same track even if there were no other instances.
        """
        return all(match.is_first_choice for match in self.matches)

    @classmethod
    def from_candidate_instances(
        cls,
        untracked_instances: List[InstanceType],
        candidate_instances: List[InstanceType],
        similarity_function: Callable,
        matching_function: Callable,
        robust_best_instance: float = 1.0,
    ):
        """Calculates (and stores) matches for a frame from candidate instance.

        Args:
            untracked_instances: list of untracked instances in the frame.
            candidate_instances: list of instances use as match.
            similarity_function: a function that returns the similarity between
                two instances (untracked and candidate).
            matching_function: function used to find the best match from the
                cost matrix. See the classmethod `from_cost_matrix`.
            robust_best_instance (float): if the value is between 0 and 1
                (excluded), use a robust quantile similarity score for the
                track. If the value is 1, use the max similarity (non-robust).
                For selecting a robust score, 0.95 is a good value.

        """
        cost = np.ndarray((0,))
        candidate_tracks = []

        if candidate_instances:

            # Group candidate instances by track.
            candidate_instances_by_track = defaultdict(list)
            for instance in candidate_instances:
                candidate_instances_by_track[instance.track].append(instance)

            # Compute similarity matrix between untracked instances and best
            # candidate for each track.
            candidate_tracks = list(candidate_instances_by_track.keys())
            dims = (len(untracked_instances), len(candidate_tracks))
            matching_similarities = np.full(dims, np.nan)

            for i, untracked_instance in enumerate(untracked_instances):

                for j, candidate_track in enumerate(candidate_tracks):
                    # Compute similarity between untracked instance and all track
                    # candidates.
                    track_instances = candidate_instances_by_track[candidate_track]
                    track_matching_similarities = [
                        similarity_function(
                            untracked_instance,
                            candidate_instance,
                        )
                        for candidate_instance in track_instances
                    ]

                    if 0 < robust_best_instance < 1:
                        # Robust, use the similarity score in the q-quantile for matching.
                        best_similarity = np.quantile(
                            track_matching_similarities,
                            robust_best_instance,
                        )
                    else:
                        # Non-robust, use the max similarity score for matching.
                        best_similarity = np.max(track_matching_similarities)
                    # Keep the best similarity score for this track.
                    matching_similarities[i, j] = best_similarity

            # Perform matching between untracked instances and candidates.
            cost = -matching_similarities
            cost[np.isnan(cost)] = np.inf

        return cls.from_cost_matrix(
            cost,
            untracked_instances,
            candidate_tracks,
            matching_function,
        )

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.ndarray,
        instances: List[InstanceType],
        tracks: List[Track],
        matching_function: Callable,
    ):
        matches = []
        match_instance_inds = []

        if instances and tracks:
            match_inds = matching_function(cost_matrix)

            # Determine the first-choice match for each instance since we want
            # to know whether or not all the matches in the frame were
            # uncontested.
            best_matches_vector = cost_matrix.argmin(axis=1)

            # Assign each matched instance.
            for i, j in match_inds:
                match_instance_inds.append(i)
                match_instance = instances[i]
                match_track = tracks[j]
                match_similarity = -cost_matrix[i, j]

                is_first_choice = best_matches_vector[i] == j

                # return matches as tuples
                matches.append(
                    Match(
                        instance=match_instance,
                        track=match_track,
                        score=match_similarity,
                        is_first_choice=is_first_choice,
                    )
                )

        # Make list of untracked instances which we didn't match to anything
        unmatched_instances = [
            untracked_instance
            for i, untracked_instance in enumerate(instances)
            if i not in match_instance_inds
        ]

        return cls(
            cost_matrix=cost_matrix,
            matches=matches,
            unmatched_instances=unmatched_instances,
        )