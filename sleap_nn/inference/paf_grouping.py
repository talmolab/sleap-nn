"""Backward-compatibility re-export shim for PAF grouping.

The implementations live in :mod:`sleap_nn.inference.ops.paf` after PR 1
of #508. This module preserves the old import path for existing callers;
it is scheduled for deletion in #519.
"""

from sleap_nn.inference.ops.paf import (
    EdgeConnection,
    EdgeType,
    PAFScorer,
    PeakID,
    assign_connections_to_instances,
    compute_distance_penalty,
    get_connection_candidates,
    get_paf_lines,
    group_instances_batch,
    group_instances_sample,
    make_line_subs,
    make_predicted_instances,
    match_candidates_batch,
    match_candidates_sample,
    score_paf_lines,
    score_paf_lines_batch,
    toposort_edges,
)

__all__ = [
    "EdgeConnection",
    "EdgeType",
    "PAFScorer",
    "PeakID",
    "assign_connections_to_instances",
    "compute_distance_penalty",
    "get_connection_candidates",
    "get_paf_lines",
    "group_instances_batch",
    "group_instances_sample",
    "make_line_subs",
    "make_predicted_instances",
    "match_candidates_batch",
    "match_candidates_sample",
    "score_paf_lines",
    "score_paf_lines_batch",
    "toposort_edges",
]
