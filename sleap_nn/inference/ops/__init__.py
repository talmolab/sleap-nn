"""Pure-ops library for inference (PR 1 of #508).

Stateless tensor operations grouped by concern. Every function here is:

- a pure function (no module state, no I/O)
- safe to import without pulling in Lightning, sleap-io, or video-reading deps
- the single source of truth — Lightning modules and (later) ``InferenceLayer``
  subclasses both call into this namespace, so they can never drift

Sub-modules:

- :mod:`peaks`      — global / local peak finding, integral refinement,
                      morphological NMS
- :mod:`paf`        — PAF scoring + grouping (``PAFScorer``)
- :mod:`coord`      — coordinate-ladder reversers (stride / scale / crop)
- :mod:`filters`    — keypoint and instance-level post-inference filters
- :mod:`identity`   — multi-class peak grouping
- :mod:`crops`      — bbox creation + cropping helpers

The original modules (``peak_finding.py``, ``paf_grouping.py`` etc.) are
preserved as thin re-export shims so existing callers keep working through
the deprecation window.
"""
