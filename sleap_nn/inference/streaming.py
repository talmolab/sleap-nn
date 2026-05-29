"""Streaming primitives for the new inference stack.

Two value types and one worker pool sit here so they can be imported
both by ``BottomUpLayer`` (which produces ``ScoredBatch`` on the GPU
side) and by the worker process (which consumes it).

* :class:`ScoredBatch` — the picklable output of the GPU stage of
  bottom-up inference (peaks + line scores). Contains every tensor
  the CPU grouping stage needs and nothing more.
* :class:`GroupingParams` — the picklable layer-level configuration
  needed to convert a :class:`ScoredBatch` into an :class:`Outputs`
  (PAFScorer kwargs, NaN-pad target, ``return_*`` flags).
* :func:`group_scored_batch` — the pure CPU function called inline or
  in a worker process. Reconstructs a :class:`PAFScorer` from the
  kwargs in ``GroupingParams`` and runs match + group + NaN-pad +
  scale-correction.
* :class:`PafGroupingPool` — context-managed ``ProcessPoolExecutor``
  wrapper that submits ``ScoredBatch`` instances and yields completed
  ``Outputs`` in submission order.

PR 9 (#517). The pool is opt-in via ``Predictor(paf_workers=N)`` —
``N=0`` keeps the inline single-process path (default, matches today).
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Iterator, List, Optional, Tuple

import attrs
import numpy as np
import torch

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo

# ─────────────────────────────────────────────────────────────────────────
# Value types — picklable handoff between GPU and CPU stages
# ─────────────────────────────────────────────────────────────────────────


@attrs.frozen(eq=False)
class ScoredBatch:
    """GPU-stage output of bottom-up inference, ready for CPU grouping.

    Carries everything :func:`group_scored_batch` needs and nothing
    more. Tensors should be detached + on CPU before this is shipped to
    a worker (``ProcessPoolExecutor`` pickles via shared-memory but a
    GPU tensor in an off-process worker is undefined). The
    :class:`BottomUpLayer` ensures CPU residency before submission.

    Attributes:
        cms_peaks: Per-sample list of ``(n_peaks, 2)`` keypoint coords
            in the scaled-input pixel space.
        cms_peak_vals: Per-sample list of ``(n_peaks,)`` confmap values.
        cms_peak_channel_inds: Per-sample list of ``(n_peaks,)`` node
            indices.
        edge_inds: Per-sample list of ``(n_candidates,)`` edge indices
            for each candidate connection.
        edge_peak_inds: Per-sample list of ``(n_candidates, 2)`` source
            and destination peak indices.
        line_scores: Per-sample list of ``(n_candidates,)`` PAF line
            scores.
        info: The :class:`PreprocInfo` capturing input scale + size
            (needed for the scale undo in :func:`group_scored_batch`).
        n_samples: Batch size; cached for cheap unpacking.
        n_nodes: Number of nodes in the skeleton.
        skip_paf: ``True`` iff the GPU stage tripped the
            ``max_peaks_per_node`` guard. The grouping stage short-
            circuits to all-NaN ``Outputs`` when this is set.
        cms: Optional confmaps tensor for ``return_confmaps``.
        pafs: Optional PAFs tensor for ``return_pafs``.
    """

    cms_peaks: List[torch.Tensor]
    cms_peak_vals: List[torch.Tensor]
    cms_peak_channel_inds: List[torch.Tensor]
    edge_inds: List[torch.Tensor]
    edge_peak_inds: List[torch.Tensor]
    line_scores: List[torch.Tensor]
    info: PreprocInfo
    n_samples: int
    n_nodes: int
    skip_paf: bool = False
    cms: Optional[torch.Tensor] = None
    pafs: Optional[torch.Tensor] = None

    def to_cpu(self) -> "ScoredBatch":
        """Detach + move every tensor field to CPU (idempotent on CPU).

        Includes ``info.eff_scale``, which is a device-resident tensor on
        cuda/mps after PR 26 made layer preprocess buffers device-aware.
        Pre-PR-26 ``eff_scale`` was always CPU so the original ``to_cpu``
        ignored ``info``; that assumption silently broke worker-pool
        submissions on CUDA (spawn can't unpickle a CUDA tensor without
        a shared CUDA context → deadlock on ``ProcessPoolExecutor.submit``).
        """
        new_info = attrs.evolve(self.info, eff_scale=self.info.eff_scale.detach().cpu())
        return attrs.evolve(
            self,
            cms_peaks=[t.detach().cpu() for t in self.cms_peaks],
            cms_peak_vals=[t.detach().cpu() for t in self.cms_peak_vals],
            cms_peak_channel_inds=[
                t.detach().cpu() for t in self.cms_peak_channel_inds
            ],
            edge_inds=[t.detach().cpu() for t in self.edge_inds],
            edge_peak_inds=[t.detach().cpu() for t in self.edge_peak_inds],
            line_scores=[t.detach().cpu() for t in self.line_scores],
            info=new_info,
            cms=self.cms.detach().cpu() if self.cms is not None else None,
            pafs=self.pafs.detach().cpu() if self.pafs is not None else None,
        )


@attrs.frozen(eq=False)
class GroupingParams:
    """Layer-level params needed to turn a ``ScoredBatch`` into ``Outputs``.

    Picklable (every field is a plain Python value or a small dict),
    so this can travel to a worker process by value.

    Attributes:
        paf_scorer_kwargs: Kwargs for the :class:`PAFScorer` constructor
            (``part_names``, ``edges``, ``pafs_stride``, etc.). The
            worker reconstructs a fresh scorer instance.
        max_instances: Cap on instances per frame. When ``None`` the
            grouping stage uses the per-batch maximum.
        return_confmaps: Echo confmaps into ``Outputs.pred_confmaps``.
        return_pafs: Echo PAFs into ``Outputs.pred_pafs``.
        return_paf_graph: Echo the per-batch PAF graph (peaks + edge
            inds + edge peak inds + line scores) into
            ``Outputs.pred_paf_graph``.
    """

    paf_scorer_kwargs: dict
    max_instances: Optional[int] = None
    return_confmaps: bool = False
    return_pafs: bool = False
    return_paf_graph: bool = False


# ─────────────────────────────────────────────────────────────────────────
# Pure CPU grouping function — same path inline and in worker
# ─────────────────────────────────────────────────────────────────────────


def group_scored_batch(scored: ScoredBatch, params: GroupingParams) -> Outputs:
    """Run the CPU grouping stage on a :class:`ScoredBatch`.

    This is the worker entry point for :class:`PafGroupingPool` and
    also the function that :class:`BottomUpLayer.postprocess` calls
    inline so the two paths share one source of truth. Pure CPU,
    no autograd, no model state.

    Args:
        scored: Output of the GPU stage (peaks + scored PAF lines).
        params: Layer-level grouping configuration.

    Returns:
        The per-batch :class:`Outputs` with NaN-padded ``(B, I, N, 2)``
        keypoints.
    """
    from sleap_nn.inference.ops.paf import PAFScorer

    B = scored.n_samples
    n_nodes = scored.n_nodes
    info = scored.info

    if scored.skip_paf:
        return _empty_outputs(B, n_nodes, info, scored, params)

    paf_scorer = PAFScorer(**params.paf_scorer_kwargs)
    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = paf_scorer.match_candidates(
        scored.edge_inds, scored.edge_peak_inds, scored.line_scores
    )
    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = paf_scorer.group_instances(
        scored.cms_peaks,
        scored.cms_peak_vals,
        scored.cms_peak_channel_inds,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )

    # Apply input-scale + eff-scale corrections per sample.
    if info.input_scale != 1.0:
        predicted_instances = [p / info.input_scale for p in predicted_instances]
    eff = info.eff_scale
    if not torch.all(eff == 1.0):
        predicted_instances = [
            p / eff[i].to(p.device) for i, p in enumerate(predicted_instances)
        ]

    # NaN-pad each batch's instance list into a uniform (B, I, N, 2) tensor.
    max_instances = params.max_instances or _infer_max_instances(predicted_instances)
    if max_instances == 0:
        max_instances = 1
    full_kpts = torch.full((B, max_instances, n_nodes, 2), float("nan"))
    full_vals = torch.full((B, max_instances, n_nodes), float("nan"))
    full_scores = torch.full((B, max_instances), float("nan"))
    for b in range(B):
        instances_b = predicted_instances[b]  # (n_b, n_nodes, 2)
        vals_b = predicted_peak_scores[b]  # (n_b, n_nodes)
        scores_b = predicted_instance_scores[b]  # (n_b,)
        # When an explicit cap is set and there are more instances than the
        # cap, keep the TOP-N by instance score (legacy parity — both the live
        # predictors.py path and the export top-k select by score descending),
        # not the first-N in grouping/assembly order. Only reorder when we are
        # actually truncating, so uncapped output keeps its assembly order
        # (preserves the parity goldens). NaN scores sort last.
        if (
            params.max_instances is not None
            and int(instances_b.shape[0]) > max_instances
        ):
            # Top-N by instance score, descending. Replicate legacy exactly
            # (``np.argsort(instance_scores)[::-1][:max]``) so tie-order and
            # NaN handling match the legacy live + export truncation. scores_b
            # is already CPU here (group_instances returns CPU tensors).
            order_np = np.argsort(scores_b.detach().cpu().numpy())[::-1]
            order = torch.from_numpy(order_np.copy())
            instances_b = instances_b[order]
            vals_b = vals_b[order]
            scores_b = scores_b[order]
        n = min(int(instances_b.shape[0]), max_instances)
        if n == 0:
            continue
        full_kpts[b, :n] = instances_b[:n]
        full_vals[b, :n] = vals_b[:n]
        full_scores[b, :n] = scores_b[:n]

    outputs = Outputs(
        pred_keypoints=full_kpts,
        pred_peak_values=full_vals,
        instance_scores=full_scores,
        preprocess_info=info,
    )
    if params.return_confmaps and scored.cms is not None:
        outputs = attrs.evolve(outputs, pred_confmaps=scored.cms)
    if params.return_pafs and scored.pafs is not None:
        outputs = attrs.evolve(
            outputs, pred_pafs=scored.pafs.permute(0, 3, 1, 2).contiguous()
        )
    if params.return_paf_graph:
        outputs = attrs.evolve(
            outputs,
            pred_paf_graph=(
                (
                    torch.cat(list(scored.cms_peaks), dim=0)
                    if scored.cms_peaks
                    else torch.empty(0, 2)
                ),
                (
                    torch.cat(list(scored.edge_inds), dim=0)
                    if scored.edge_inds
                    else torch.empty(0, dtype=torch.int32)
                ),
                (
                    torch.cat(list(scored.edge_peak_inds), dim=0)
                    if scored.edge_peak_inds
                    else torch.empty(0, 2, dtype=torch.int32)
                ),
                (
                    torch.cat(list(scored.line_scores), dim=0)
                    if scored.line_scores
                    else torch.empty(0)
                ),
            ),
        )
    return outputs


def _empty_outputs(
    B: int,
    n_nodes: int,
    info: PreprocInfo,
    scored: ScoredBatch,
    params: GroupingParams,
) -> Outputs:
    """Return all-NaN ``Outputs`` for the ``skip_paf`` short-circuit."""
    max_instances = params.max_instances or 1
    outputs = Outputs(
        pred_keypoints=torch.full((B, max_instances, n_nodes, 2), float("nan")),
        pred_peak_values=torch.full((B, max_instances, n_nodes), float("nan")),
        instance_scores=torch.full((B, max_instances), float("nan")),
        preprocess_info=info,
    )
    if params.return_confmaps and scored.cms is not None:
        outputs = attrs.evolve(outputs, pred_confmaps=scored.cms)
    if params.return_pafs and scored.pafs is not None:
        outputs = attrs.evolve(
            outputs, pred_pafs=scored.pafs.permute(0, 3, 1, 2).contiguous()
        )
    return outputs


def _infer_max_instances(predicted_instances: List[torch.Tensor]) -> int:
    """Largest per-batch instance count, or 0 if all empty."""
    return max((int(p.shape[0]) for p in predicted_instances), default=0)


# ─────────────────────────────────────────────────────────────────────────
# Worker pool — opt-in via Predictor(paf_workers=N)
# ─────────────────────────────────────────────────────────────────────────


@attrs.define
class PafGroupingPool:
    """Multiprocessing pool for the CPU grouping stage of bottom-up.

    Lifetime: a context manager. Use ``with PafGroupingPool(...) as pool``.
    Calls outside the ``with`` block raise — the executor is None.

    Submission order is FIFO; results are yielded in submission order
    (a future blocks ``iter_completed`` until earlier ones finish, which
    matches the frame-ordering contract of ``Predictor`` outputs).

    Args:
        n_workers: Number of worker processes. ``0`` is a config error
            here — the predictor short-circuits to the inline path
            before reaching this class.
        grouping_params: Layer-level params for the grouping function;
            shipped as a per-call argument so it remains picklable
            (the executor's submit copies it once per call but the
            data is small).

    Notes:
        On macOS / Windows the pool uses spawn and each worker pays a
        ~1s startup cost. Users running on those platforms should
        either keep the default ``paf_workers=0`` for short videos or
        size the pool small.
    """

    n_workers: int
    grouping_params: GroupingParams
    _executor: Optional[ProcessPoolExecutor] = attrs.field(default=None, init=False)
    _pending: "list[Tuple[int, Future]]" = attrs.field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        """Validate ``n_workers``."""
        if self.n_workers < 1:
            raise ValueError(
                f"n_workers must be >= 1; got {self.n_workers}. "
                f"Use the inline path (paf_workers=0) for single-process."
            )

    def __enter__(self) -> "PafGroupingPool":
        """Start the pool's worker processes.

        Always uses the ``spawn`` start method. ``ProcessPoolExecutor`` defaults
        to ``fork`` on Linux, which inherits the parent's already-initialized
        CUDA context and deadlocks the first worker call. ``spawn`` is the
        same start method already used on macOS / Windows by default.
        """
        import multiprocessing

        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        """Tear down workers; cancel pending futures on exception."""
        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=exc is not None)
        self._executor = None

    def submit(self, frame_idx: int, scored: ScoredBatch) -> None:
        """Enqueue a :class:`ScoredBatch` for grouping.

        Args:
            frame_idx: Caller-supplied submission ordinal (0, 1, 2, ...);
                used only for ``iter_completed`` ordering, not for any
                semantic frame index in :class:`Outputs` (those land via
                ``Predictor._stamp_metadata``).
            scored: The GPU-stage payload.
        """
        if self._executor is None:
            raise RuntimeError(
                "PafGroupingPool.submit called outside `with` block; "
                "use the context manager to start workers."
            )
        future = self._executor.submit(group_scored_batch, scored, self.grouping_params)
        self._pending.append((frame_idx, future))

    def iter_completed(self) -> Iterator[Tuple[int, Outputs]]:
        """Drain all submitted batches, yielding ``(frame_idx, Outputs)`` in order.

        Blocks on each future in submission order so the caller observes
        the same frame ordering as the inline path. Cleans the internal
        pending list as it goes.
        """
        while self._pending:
            frame_idx, future = self._pending.pop(0)
            yield frame_idx, future.result()

    def __len__(self) -> int:
        """Number of submitted-but-not-yet-drained batches (in-flight depth)."""
        return len(self._pending)

    def drain_one(self) -> "Optional[Tuple[int, Outputs]]":
        """Pop + block on the OLDEST pending batch (FIFO); ``None`` if empty.

        Lets a caller interleave ``submit`` and drain to bound the number of
        in-flight batches (true O(window) streaming) while preserving the same
        submission-order yield as :meth:`iter_completed`. Because submission is
        sequential on the main thread, the oldest future is always real, so
        this never deadlocks.
        """
        if not self._pending:
            return None
        frame_idx, future = self._pending.pop(0)
        return frame_idx, future.result()
