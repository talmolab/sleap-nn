"""``TiledLayer`` — sliding-window tiled inference (Phase A single-instance).

``TiledLayer`` is a **composed wrapper**, structured like
:class:`~sleap_nn.inference.layers.topdown.TopDownLayer`: it is *not* an
:class:`~sleap_nn.inference.layers.base.InferenceLayer` subclass, has no
``preprocess`` / ``postprocess``, and re-implements ``predict(image) -> Outputs``
by driving an inner ``InferenceLayer`` (Phase A: a ``SingleInstanceLayer``).

Per frame it:

1. Preprocesses the whole frame at **native** resolution (only ``input_scale``
   applies — the sizematcher is bypassed via ``skip_sizematcher=True``).
2. Splits the frame into overlapping square tiles on the ``output_stride`` grid
   (:func:`~sleap_nn.data.tiling.generate_tile_grid`).
3. Runs the inner backend on the tiles in batches.
4. Stitches per-tile confmaps into one Gaussian-weighted ACC/CNT canvas
   (:class:`~sleap_nn.inference.tile_merger.TileMerger`).
5. Decodes a single global peak per node on the stitched map
   (:func:`~sleap_nn.inference.ops.peaks.find_global_peaks`) → one pose per
   frame, then reverses the coord ladder into original-image space.

The tile offset is baked into the paste coordinates at merge time, so there is
**no** ``add_crop_offset`` step in the coordinate ladder.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
        """Slice a ``tile_size`` window at ``(y0, x0)``, zero-padding overruns.

        Args:
            frame: ``(C, Hs, Ws)`` preprocessed frame.
            y0: Top row of the tile in scaled-frame pixels.
            x0: Left column of the tile in scaled-frame pixels.

        Returns:
            A ``(C, tile_size, tile_size)`` tile (bottom/right zero-padded when
            the window overruns the frame).
        """
        C, H, W = frame.shape
        ts = self.tile_size
        ye = min(H, y0 + ts)
        xe = min(W, x0 + ts)
        tile = frame.new_zeros((C, ts, ts))
        if ye > y0 and xe > x0:
            tile[:, : ye - y0, : xe - x0] = frame[:, y0:ye, x0:xe]
        return tile

    def _resolve_accumulator_device(
        self,
        n_channels: Optional[int],
        out_hw: Tuple[int, int],
        model_device: str,
    ) -> str:
        """Resolve the ACC/CNT device per the ``accumulator_device`` policy.

        ``"cpu"`` / ``"cuda"`` are honored directly; ``"auto"`` predicts
        placement on a CUDA model device (spilling to CPU when the buffers
        would exceed ``cpu_thresh`` of free GPU memory) and otherwise keeps the
        buffers on the model device.
        """
        if self.accumulator_device == "cpu":
            return "cpu"
        if self.accumulator_device == "cuda":
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
        if est > self.cpu_thresh * free:
            return "cpu"
        return dev

    def _make_merger(
        self,
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
