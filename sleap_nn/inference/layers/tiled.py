"""Sliding-window tiled inference wrappers.

This module holds the **composed wrapper** layers that drive an inner
:class:`~sleap_nn.inference.layers.base.InferenceLayer` over a frame that is
larger than the model's tile input by splitting it into overlapping tiles and
stitching the per-tile predictions back into a single full-frame output. Like
:class:`~sleap_nn.inference.layers.topdown.TopDownLayer` these are *not*
``InferenceLayer`` subclasses, have no ``preprocess`` / ``postprocess``, and
re-implement ``predict(image) -> Outputs``.

- :class:`TiledLayer` (Phase A single-instance): tiles → forward → stitch
  confmaps → decode one global peak per node → one pose per frame.
- :class:`TiledSegmentationLayer` (bottom-up instance segmentation): tiles →
  forward → stitch the three seg heads (foreground / instance-center /
  center-offset) into ONE 4-channel Gaussian-weighted ACC/CNT canvas → hand the
  stitched heads to the inner ``SegmentationLayer.postprocess`` verbatim (which
  groups foreground pixels into per-instance masks). The offset field is a
  per-pixel *displacement* to the instance center, so it is translation
  invariant and correct to Gaussian-weighted-average in overlaps.
- :class:`TiledSemanticSegmentationLayer` (whole-frame semantic segmentation):
  the single-head twin of :class:`TiledSegmentationLayer`. Stitches ONLY the
  foreground head into a 1-channel Gaussian-weighted ACC/CNT canvas (no
  4-channel cat/split), then hands it to the inner
  ``SemanticSegmentationLayer.postprocess`` which thresholds it into one
  whole-frame mask.

Per frame both preprocess the whole frame at **native** resolution (only
``input_scale`` applies — the sizematcher is bypassed via
``skip_sizematcher=True``), split the frame into overlapping square tiles on the
``output_stride`` grid (:func:`~sleap_nn.data.tiling.generate_tile_grid`), run
the inner backend on the tiles in batches, and stitch per-tile maps into one
Gaussian-weighted ACC/CNT canvas
(:class:`~sleap_nn.inference.tile_merger.TileMerger`). The tile offset is baked
into the paste coordinates at merge time, so there is **no** ``add_crop_offset``
step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import attrs
import torch

from sleap_nn.data.tiling import generate_tile_grid
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.ops.coord import (
    undo_eff_scale,
    undo_input_scale,
    undo_stride,
)
from sleap_nn.inference.ops.peaks import find_global_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.tile_merger import TileMerger, build_importance_window

# ──────────────────────────────────────────────────────────────────────────
# Shared tiling helpers (used by every tiled wrapper in this module)
# ──────────────────────────────────────────────────────────────────────────


def _extract_square_tile(
    frame: torch.Tensor, y0: int, x0: int, tile_size: int
) -> torch.Tensor:
    """Slice a ``tile_size`` window at ``(y0, x0)``, zero-padding overruns.

    Args:
        frame: ``(C, Hs, Ws)`` preprocessed frame.
        y0: Top row of the tile in scaled-frame pixels.
        x0: Left column of the tile in scaled-frame pixels.
        tile_size: Square tile side length in pixels.

    Returns:
        A ``(C, tile_size, tile_size)`` tile (bottom/right zero-padded when the
        window overruns the frame).
    """
    C, H, W = frame.shape
    ts = int(tile_size)
    ye = min(H, y0 + ts)
    xe = min(W, x0 + ts)
    tile = frame.new_zeros((C, ts, ts))
    if ye > y0 and xe > x0:
        tile[:, : ye - y0, : xe - x0] = frame[:, y0:ye, x0:xe]
    return tile


def _resolve_accumulator_device(
    policy: str,
    cpu_thresh: float,
    n_channels: Optional[int],
    out_hw: Tuple[int, int],
    model_device: str,
) -> str:
    """Resolve the ACC/CNT device per the ``accumulator_device`` policy.

    ``"cpu"`` / ``"cuda"`` are honored directly; ``"auto"`` predicts placement
    on a CUDA model device (spilling to CPU when the buffers would exceed
    ``cpu_thresh`` of free GPU memory) and otherwise keeps the buffers on the
    model device.
    """
    if policy == "cpu":
        return "cpu"
    if policy == "cuda":
        return str(model_device)

    # "auto": non-CUDA model device -> keep buffers with the model.
    dev = str(model_device)
    if not dev.startswith("cuda"):
        return dev
    try:
        free, _total = torch.cuda.mem_get_info(dev)
    except Exception:  # noqa: BLE001 — mem query is best-effort
        return dev
    H, W = out_hw
    n = n_channels if n_channels is not None else 1
    est = (n + 1) * H * W * 4  # ACC(N) + CNT(1) float32 bytes
    if est > cpu_thresh * free:
        return "cpu"
    return dev


def _make_tile_merger(
    out_hw: Tuple[int, int],
    channels: int,
    window: torch.Tensor,
    device: str,
) -> TileMerger:
    """Build a ``TileMerger`` on ``device``, spilling to CPU on CUDA OOM."""
    try:
        return TileMerger(out_hw, channels, window, device=device)
    except torch.cuda.OutOfMemoryError:  # pragma: no cover — GPU-only path
        return TileMerger(out_hw, channels, window, device="cpu")


class TiledLayer:
    """Sliding-window tiled inference wrapping a lone confmap layer.

    Composes an inner ``InferenceLayer`` (Phase A: ``SingleInstanceLayer``).
    Tiles each frame at native resolution, runs the inner backend per tile in
    batches, stitches per-tile confmaps into one Gaussian-weighted ACC/CNT
    canvas per frame (stitch), then decodes a single global peak per node
    (detect) -> one pose per frame. Not an ``InferenceLayer`` subclass.

    Args:
        inner_layer: Pre-built inner layer (Phase A: ``SingleInstanceLayer``).
            Its ``backend`` / ``output_stride`` / ``max_stride`` /
            ``preprocess_config`` / ``postprocess_config`` / ``_extract_confmaps``
            are reused verbatim.
        tile_size: Square tile side length in pixels; a multiple of both
            ``max_stride`` and ``output_stride``.
        overlap: Tile overlap in pixels.
        blend: Importance-window mode: ``"gaussian"`` (default), ``"pyramid"``,
            or ``"constant"``.
        sigma_scale: Gaussian window std as a fraction of the tile side.
        min_overlap_fraction: Minimum overlap as a fraction of ``tile_size``,
            enforced by :func:`generate_tile_grid`.
        tile_batch_size: Number of tiles forwarded per backend call.
        accumulator_device: Device for the per-frame ACC/CNT buffers.
            ``"auto"`` (predictive placement + OOM spill), ``"cpu"``, or
            ``"cuda"``.
        cpu_thresh: Spill ACC/CNT to CPU when they would exceed this fraction
            of free GPU memory (``"auto"`` only).
    """

    def __init__(
        self,
        inner_layer: InferenceLayer,
        tile_size: int,
        overlap: int,
        *,
        blend: str = "gaussian",
        sigma_scale: float = 0.125,
        min_overlap_fraction: float = 0.25,
        tile_batch_size: int = 8,
        accumulator_device: str = "auto",
        cpu_thresh: float = 0.40,
    ) -> None:
        """Stash the inner layer and tiling knobs."""
        self.inner = inner_layer
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.output_stride = inner_layer.output_stride
        self.max_stride = inner_layer.max_stride
        self.tile_batch_size = int(tile_batch_size)
        self.accumulator_device = accumulator_device
        self.cpu_thresh = cpu_thresh
        self._window_cache: dict = {}  # tile_hw -> window tensor
        self._blend = blend
        self._sigma_scale = sigma_scale
        self._min_overlap_fraction = min_overlap_fraction

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def predict(self, image: ImageInput) -> Outputs:
        """Tile → forward → stitch → detect → coord-undo, one pose per frame."""
        stride = self.output_stride

        # Step 0: whole-frame preprocess, sizematcher-bypassed. Only channel
        # coercion + input_scale + pad-to-output_stride apply (native res).
        x = InferenceLayer._to_4d_tensor(image)
        scaled, eff_scale, orig_hw = self.inner._apply_full_preprocess(
            x,
            max_stride=stride,
            unsqueeze_n_samples=False,
            skip_sizematcher=True,
        )  # scaled: (B, C, Hs, Ws); eff_scale == ones(B)
        input_scale = self.inner.preprocess_config.scale

        B = scaled.shape[0]
        peaks_all = []
        vals_all = []
        stitched_all = []
        first_proc_hw: Optional[Tuple[int, int]] = None

        for b in range(B):
            frame = scaled[b]  # (C, Hs, Ws)
            Hs, Ws = int(frame.shape[-2]), int(frame.shape[-1])
            if first_proc_hw is None:
                first_proc_hw = (Hs, Ws)

            origins = generate_tile_grid(
                (Hs, Ws),
                self.tile_size,
                self.overlap,
                stride,
                self.max_stride,
                self._min_overlap_fraction,
            )
            win = self._get_window((self.tile_size // stride, self.tile_size // stride))

            # Canvas covers max(frame, tile) per axis so a single tile larger
            # than the frame (tiny-frame case) still fits; cropped back to the
            # frame confmap size before peak detection.
            canvas_h = max(Hs, self.tile_size) // stride
            canvas_w = max(Ws, self.tile_size) // stride

            merger = None
            acc_dev = None
            for i in range(0, len(origins), self.tile_batch_size):
                chunk = origins[i : i + self.tile_batch_size]
                tiles = torch.stack(
                    [self._extract_tile(frame, y0, x0) for (y0, x0) in chunk]
                )  # (n, C, ts, ts)
                with torch.inference_mode():
                    raw = self.inner.backend(tiles.unsqueeze(1))  # (n, 1, C, ts, ts)
                cms = self.inner._extract_confmaps(raw)  # (n, N, ts//s, ts//s)
                cms = cms.detach()
                if merger is None:
                    N = int(cms.shape[1])
                    acc_dev = self._resolve_accumulator_device(
                        n_channels=N,
                        out_hw=(canvas_h, canvas_w),
                        model_device=self.inner.backend.device,
                    )
                    merger = self._make_merger((canvas_h, canvas_w), N, win, acc_dev)
                for j, (y0, x0) in enumerate(chunk):
                    merger.integrate(cms[j], y0 // stride, x0 // stride)

            # Stitch → crop back to frame confmap size → detect one global peak.
            stitched = merger.merge()  # (N, canvas_h, canvas_w)
            stitched = stitched[:, : Hs // stride, : Ws // stride]
            peaks, vals = find_global_peaks(
                stitched.unsqueeze(0),  # (1, N, h, w)
                threshold=self.inner.postprocess_config.peak_threshold,
                refinement=self.inner.postprocess_config.effective_refinement,
                integral_patch_size=self.inner.postprocess_config.integral_patch_size,
            )  # peaks (1, N, 2) in (x, y)

            # Coord undo — offset already baked at paste time (no add_crop_offset).
            peaks = undo_stride(peaks, stride)
            peaks = undo_input_scale(peaks, input_scale)
            peaks = undo_eff_scale(peaks, eff_scale[b : b + 1])

            peaks_all.append(peaks)  # (1, N, 2)
            vals_all.append(vals)  # (1, N)
            stitched_all.append(stitched)

        # Assemble ONE Outputs per batch, in original-image space.
        peaks_BN2 = torch.cat(peaks_all, dim=0)  # (B, N, 2)
        vals_BN = torch.cat(vals_all, dim=0)  # (B, N)
        info = PreprocInfo(
            original_size=orig_hw,
            processed_size=first_proc_hw or (0, 0),
            eff_scale=eff_scale,
            input_scale=input_scale,
            output_stride=stride,
        )
        outputs = Outputs(
            pred_keypoints=peaks_BN2.unsqueeze(1),  # (B, 1, N, 2)
            pred_peak_values=vals_BN.unsqueeze(1),  # (B, 1, N)
            preprocess_info=info,
        )
        if self.inner.postprocess_config.return_confmaps and stitched_all:
            shapes = {tuple(s.shape) for s in stitched_all}
            if len(shapes) == 1:
                outputs = attrs.evolve(
                    outputs,
                    pred_confmaps=torch.stack(stitched_all, dim=0).detach(),
                )
        return outputs

    def __call__(self, image: ImageInput) -> Outputs:
        """Alias for :meth:`predict`."""
        return self.predict(image)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_window(self, tile_hw: Tuple[int, int]) -> torch.Tensor:
        """Return (and cache) the importance window for a tile size."""
        key = tuple(tile_hw)
        win = self._window_cache.get(key)
        if win is None:
            win = build_importance_window(
                key, mode=self._blend, sigma_scale=self._sigma_scale
            )
            self._window_cache[key] = win
        return win

    def _extract_tile(self, frame: torch.Tensor, y0: int, x0: int) -> torch.Tensor:
        """Slice a ``tile_size`` window at ``(y0, x0)`` (see module helper)."""
        return _extract_square_tile(frame, y0, x0, self.tile_size)

    def _resolve_accumulator_device(
        self,
        n_channels: Optional[int],
        out_hw: Tuple[int, int],
        model_device: str,
    ) -> str:
        """Resolve the ACC/CNT device per the ``accumulator_device`` policy."""
        return _resolve_accumulator_device(
            self.accumulator_device,
            self.cpu_thresh,
            n_channels,
            out_hw,
            model_device,
        )

    def _make_merger(
        self,
        out_hw: Tuple[int, int],
        channels: int,
        window: torch.Tensor,
        device: str,
    ) -> TileMerger:
        """Build a ``TileMerger`` on ``device``, spilling to CPU on CUDA OOM."""
        return _make_tile_merger(out_hw, channels, window, device)


class TiledSegmentationLayer:
    """Sliding-window tiled inference wrapping a bottom-up ``SegmentationLayer``.

    Composes an inner
    :class:`~sleap_nn.inference.layers.segmentation.SegmentationLayer`. Tiles
    each frame at native resolution, runs the inner backend per tile in batches,
    and stitches the three predicted heads —foreground (``SegmentationHead``),
    instance-center (``InstanceCenterHead``), and center-offset
    (``CenterOffsetHead``)— into ONE 4-channel Gaussian-weighted ACC/CNT canvas
    per frame. The stitched heads are then handed to the inner layer's
    ``postprocess`` verbatim, which groups foreground pixels into per-instance
    masks and packages them into ``Outputs.pred_masks``. Not an
    ``InferenceLayer`` subclass.

    Why one 4-channel merger: ``foreground``/``center`` are probabilities, so the
    accumulate-normalize average is the correct blend in tile overlaps. The
    ``offset`` field is a per-pixel *displacement* to the instance center —
    translation invariant — so pasting + Gaussian-weighted-averaging in overlaps
    is also correct (an edge tile whose instance center is truncated is
    down-weighted by the window).

    Args:
        inner_layer: Pre-built inner ``SegmentationLayer``. Its ``backend`` /
            ``output_stride`` / ``max_stride`` / ``preprocess_config`` /
            ``postprocess_config`` / head keys / ``postprocess`` are reused
            verbatim.
        tile_size: Square tile side length in pixels; a multiple of both
            ``max_stride`` and ``output_stride``.
        overlap: Tile overlap in pixels.
        blend: Importance-window mode: ``"gaussian"`` (default), ``"pyramid"``,
            or ``"constant"``.
        sigma_scale: Gaussian window std as a fraction of the tile side.
        min_overlap_fraction: Minimum overlap as a fraction of ``tile_size``,
            enforced by :func:`generate_tile_grid`.
        tile_batch_size: Number of tiles forwarded per backend call.
        accumulator_device: Device for the per-frame ACC/CNT buffers.
            ``"auto"`` (predictive placement + OOM spill), ``"cpu"``, or
            ``"cuda"``.
        cpu_thresh: Spill ACC/CNT to CPU when they would exceed this fraction of
            free GPU memory (``"auto"`` only).
    """

    def __init__(
        self,
        inner_layer: InferenceLayer,
        tile_size: int,
        overlap: int,
        *,
        blend: str = "gaussian",
        sigma_scale: float = 0.125,
        min_overlap_fraction: float = 0.25,
        tile_batch_size: int = 8,
        accumulator_device: str = "auto",
        cpu_thresh: float = 0.40,
    ) -> None:
        """Stash the inner layer and tiling knobs."""
        self.inner = inner_layer
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.output_stride = inner_layer.output_stride
        self.max_stride = inner_layer.max_stride
        # Exposed so the Predictor's pipelined / metadata helpers can read the
        # backend off a tiled layer the same way they read it off a plain one.
        self.backend = inner_layer.backend
        self.tile_batch_size = int(tile_batch_size)
        self.accumulator_device = accumulator_device
        self.cpu_thresh = cpu_thresh
        self._window_cache: dict = {}  # tile_hw -> window tensor
        self._blend = blend
        self._sigma_scale = sigma_scale
        self._min_overlap_fraction = min_overlap_fraction

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def predict(self, image: ImageInput) -> Outputs:
        """Tile → forward → stitch 3 heads → group instances, per frame."""
        stride = self.output_stride
        seg_key = self.inner._SEG_KEY
        center_key = self.inner._CENTER_KEY
        offset_key = self.inner._OFFSET_KEY

        # Step 0: whole-frame preprocess, sizematcher-bypassed. Only channel
        # coercion + input_scale + pad-to-output_stride apply (native res).
        x = InferenceLayer._to_4d_tensor(image)
        scaled, eff_scale, orig_hw = self.inner._apply_full_preprocess(
            x,
            max_stride=stride,
            unsqueeze_n_samples=False,
            skip_sizematcher=True,
        )  # scaled: (B, C, Hs, Ws); eff_scale == ones(B)
        input_scale = self.inner.preprocess_config.scale

        B = scaled.shape[0]
        pred_masks_all: List[list] = []
        last_info: Optional[PreprocInfo] = None

        # One frame at a time (per-frame accumulator, mixed-resolution safe).
        for b in range(B):
            frame = scaled[b]  # (C, Hs, Ws)
            Hs, Ws = int(frame.shape[-2]), int(frame.shape[-1])

            origins = generate_tile_grid(
                (Hs, Ws),
                self.tile_size,
                self.overlap,
                stride,
                self.max_stride,
                self._min_overlap_fraction,
            )
            win = self._get_window((self.tile_size // stride, self.tile_size // stride))

            # Canvas covers max(frame, tile) per axis so a single tile larger
            # than the frame (tiny-frame case) still fits; cropped back to the
            # frame head-map size before grouping.
            canvas_h = max(Hs, self.tile_size) // stride
            canvas_w = max(Ws, self.tile_size) // stride

            merger = None
            for i in range(0, len(origins), self.tile_batch_size):
                chunk = origins[i : i + self.tile_batch_size]
                tiles = torch.stack(
                    [
                        _extract_square_tile(frame, y0, x0, self.tile_size)
                        for (y0, x0) in chunk
                    ]
                )  # (n, C, ts, ts)
                with torch.inference_mode():
                    raw = self.inner.backend(tiles.unsqueeze(1))  # squeeze(dim=1)
                fg = raw[seg_key]  # (n, 1, ts//s, ts//s), sigmoid
                cen = raw[center_key]  # (n, 1, ts//s, ts//s)
                off = raw[offset_key]  # (n, 2, ts//s, ts//s)
                # Stitch all three heads in one canvas: [fg(1), cen(1), off(2)].
                heads = torch.cat(
                    [fg, cen, off], dim=1
                ).detach()  # (n, 4, ts//s, ts//s)
                if merger is None:
                    acc_dev = _resolve_accumulator_device(
                        self.accumulator_device,
                        self.cpu_thresh,
                        n_channels=4,
                        out_hw=(canvas_h, canvas_w),
                        model_device=self.inner.backend.device,
                    )
                    merger = _make_tile_merger((canvas_h, canvas_w), 4, win, acc_dev)
                for j, (y0, x0) in enumerate(chunk):
                    merger.integrate(heads[j], y0 // stride, x0 // stride)

            # Stitch → crop back to the frame head-map size, then split the
            # 4-channel canvas back into the three head tensors.
            stitched = merger.merge()  # (4, canvas_h, canvas_w)
            stitched = stitched[:, : Hs // stride, : Ws // stride]
            fg_s = stitched[0:1].unsqueeze(0)  # (1, 1, H, W)
            cen_s = stitched[1:2].unsqueeze(0)  # (1, 1, H, W)
            off_s = stitched[2:4].unsqueeze(0)  # (1, 2, H, W)

            raw_out = {seg_key: fg_s, center_key: cen_s, offset_key: off_s}
            # Match ``SegmentationLayer.predict``'s PreprocInfo fields — offsets
            # are baked into paste coords, so no crop offset is needed.
            info = PreprocInfo(
                original_size=orig_hw,
                processed_size=(Hs, Ws),
                eff_scale=eff_scale[b : b + 1],
                input_scale=input_scale,
                output_stride=stride,
            )
            # Reuse ALL grouping / mask packaging verbatim.
            out_b = self.inner.postprocess(raw_out, info)
            pred_masks_all.extend(out_b.pred_masks)  # single-frame list per b
            last_info = info

        # Assemble ONE Outputs per batch (matches SegmentationLayer's shape).
        # Do NOT set frame/video indices — the Predictor stamps those.
        return Outputs(pred_masks=pred_masks_all, preprocess_info=last_info)

    def __call__(self, image: ImageInput) -> Outputs:
        """Alias for :meth:`predict`."""
        return self.predict(image)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_window(self, tile_hw: Tuple[int, int]) -> torch.Tensor:
        """Return (and cache) the importance window for a tile size."""
        key = tuple(tile_hw)
        win = self._window_cache.get(key)
        if win is None:
            win = build_importance_window(
                key, mode=self._blend, sigma_scale=self._sigma_scale
            )
            self._window_cache[key] = win
        return win


class TiledSemanticSegmentationLayer(TiledSegmentationLayer):
    """Sliding-window tiled inference wrapping a ``SemanticSegmentationLayer``.

    The single-head (foreground-only) twin of :class:`TiledSegmentationLayer`.
    Tiles each frame at native resolution, runs the inner backend per tile in
    batches, and stitches ONLY the foreground head (``SegmentationHead``, sigmoid
    probabilities) into a **1-channel** Gaussian-weighted ACC/CNT canvas per
    frame — there is no instance-center or center-offset head, so there is no
    4-channel cat/split. The stitched foreground map is handed to the inner
    ``SemanticSegmentationLayer.postprocess`` verbatim, which thresholds it into
    exactly one whole-frame mask. Not an ``InferenceLayer`` subclass.

    Averaging foreground *probabilities* across tile overlaps is the correct
    blend (identical to the ``foreground`` channel of the bottom-up path); with no
    offset field there is no translation-invariance subtlety.

    Reuses :class:`TiledSegmentationLayer`'s ``__init__`` / window cache /
    ``__call__`` / ``backend`` exposure verbatim; overrides only :meth:`predict`
    to stitch one channel instead of four. Also re-exposes the inner layer's
    ``mask_output`` / ``polygon_epsilon`` so the Predictor reads them off the
    tiled layer (the base ``TiledSegmentationLayer`` omits these).
    """

    def __init__(self, inner_layer, tile_size, overlap, **kwargs) -> None:
        """Stash the inner layer + tiling knobs, re-exposing packaging knobs."""
        super().__init__(inner_layer, tile_size, overlap, **kwargs)
        # Re-expose the packaging knobs so ``predictor`` reads them off the tiled
        # layer (via getattr) exactly as it does off the plain ``SegmentationLayer``.
        self.mask_output = getattr(inner_layer, "mask_output", "mask")
        self.polygon_epsilon = getattr(inner_layer, "polygon_epsilon", 0.01)

    def predict(self, image: ImageInput) -> Outputs:
        """Tile -> forward -> stitch fg -> threshold to one mask, per frame."""
        stride = self.output_stride
        seg_key = self.inner._SEG_KEY

        # Step 0: whole-frame preprocess, sizematcher-bypassed. Only channel
        # coercion + input_scale + pad-to-output_stride apply (native res).
        x = InferenceLayer._to_4d_tensor(image)
        scaled, eff_scale, orig_hw = self.inner._apply_full_preprocess(
            x,
            max_stride=stride,
            unsqueeze_n_samples=False,
            skip_sizematcher=True,
        )  # scaled: (B, C, Hs, Ws); eff_scale == ones(B)
        input_scale = self.inner.preprocess_config.scale

        B = scaled.shape[0]
        pred_masks_all: List[list] = []
        last_info: Optional[PreprocInfo] = None

        # One frame at a time (per-frame accumulator, mixed-resolution safe).
        for b in range(B):
            frame = scaled[b]  # (C, Hs, Ws)
            Hs, Ws = int(frame.shape[-2]), int(frame.shape[-1])

            origins = generate_tile_grid(
                (Hs, Ws),
                self.tile_size,
                self.overlap,
                stride,
                self.max_stride,
                self._min_overlap_fraction,
            )
            win = self._get_window((self.tile_size // stride, self.tile_size // stride))

            # Canvas covers max(frame, tile) per axis so a single tile larger than
            # the frame (tiny-frame case) still fits; cropped back to the frame
            # head-map size before thresholding.
            canvas_h = max(Hs, self.tile_size) // stride
            canvas_w = max(Ws, self.tile_size) // stride

            merger = None
            for i in range(0, len(origins), self.tile_batch_size):
                chunk = origins[i : i + self.tile_batch_size]
                tiles = torch.stack(
                    [
                        _extract_square_tile(frame, y0, x0, self.tile_size)
                        for (y0, x0) in chunk
                    ]
                )  # (n, C, ts, ts)
                with torch.inference_mode():
                    raw = self.inner.backend(tiles.unsqueeze(1))  # squeeze(dim=1)
                # Single foreground head only — a 1-channel canvas (no cen/off, so
                # no torch.cat / split like the bottom-up 4-channel path).
                fg = raw[seg_key].detach()  # (n, 1, ts//s, ts//s), sigmoid
                if merger is None:
                    acc_dev = _resolve_accumulator_device(
                        self.accumulator_device,
                        self.cpu_thresh,
                        n_channels=1,
                        out_hw=(canvas_h, canvas_w),
                        model_device=self.inner.backend.device,
                    )
                    merger = _make_tile_merger((canvas_h, canvas_w), 1, win, acc_dev)
                for j, (y0, x0) in enumerate(chunk):
                    merger.integrate(fg[j], y0 // stride, x0 // stride)

            # Stitch -> crop back to the frame head-map size. One channel, so there
            # is nothing to split: the stitched canvas IS the fg map.
            stitched = merger.merge()  # (1, canvas_h, canvas_w)
            stitched = stitched[:, : Hs // stride, : Ws // stride]
            fg_s = stitched[0:1].unsqueeze(0)  # (1, 1, H, W)

            raw_out = {seg_key: fg_s}
            # Match ``SemanticSegmentationLayer.predict``'s PreprocInfo fields —
            # tile offsets are baked into paste coords, so no crop offset.
            info = PreprocInfo(
                original_size=orig_hw,
                processed_size=(Hs, Ws),
                eff_scale=eff_scale[b : b + 1],
                input_scale=input_scale,
                output_stride=stride,
            )
            # Reuse the fg-threshold -> one-mask packaging verbatim.
            out_b = self.inner.postprocess(raw_out, info)
            pred_masks_all.extend(out_b.pred_masks)  # single-frame list per b
            last_info = info

        # Assemble ONE Outputs per batch (matches the non-tiled shape). Do NOT set
        # frame/video indices — the Predictor stamps those.
        return Outputs(pred_masks=pred_masks_all, preprocess_info=last_info)
