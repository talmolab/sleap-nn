"""In-memory preview frame capture during inference.

Captures a single clean video frame already decoded by the inference provider
(preferring the first frame with at least one predicted detection, falling back
to the first frame seen) so the saved ``.slp`` file includes a ``/preview``
thumbnail with zero extra video decoding at save time.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch


def _to_hwc_uint8(img: Any) -> Optional[np.ndarray]:
    """Convert a single frame tensor/ndarray to a CPU ``(H, W)`` or ``(H, W, C)`` array."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim == 2:
        return np.array(arr, copy=True)
    if arr.ndim != 3:
        return None

    # Detect CHW vs HWC: channels-first has leading dim in {1, 3, 4} and larger spatial dims.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return np.array(arr, copy=True)


def _frame_has_prediction(outputs: Any, idx: int) -> bool:
    """Return True if frame ``idx`` in ``outputs`` has at least one detection."""
    valid = getattr(outputs, "instance_valid", None)
    if valid is not None and idx < len(valid):
        v = valid[idx]
        if isinstance(v, torch.Tensor):
            if bool(v.any().item()):
                return True
        elif bool(np.any(v)):
            return True

    for attr in ("pred_keypoints", "pred_centroids", "pred_bboxes"):
        val = getattr(outputs, attr, None)
        if val is None or idx >= len(val):
            continue
        item = val[idx]
        if isinstance(item, torch.Tensor):
            if item.numel() > 0 and bool(torch.isfinite(item).any().item()):
                return True
        else:
            arr = np.asarray(item)
            if arr.size > 0 and bool(np.isfinite(arr).any()):
                return True

    masks = getattr(outputs, "pred_masks", None)
    if masks is not None and idx < len(masks) and len(masks[idx]) > 0:
        return True

    return False


def _extract_index(arr: Any, idx: int, default: int = 0) -> int:
    if arr is None or idx >= len(arr):
        return default
    val = arr[idx]
    if isinstance(val, torch.Tensor):
        return int(val.item())
    return int(val)


def _crop_fitted_padding(
    img: np.ndarray,
    frame_size: Any,
    frame_scale: Any,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Strip right/bottom letterbox padding when a provider pre-fitted ``batch.images``."""
    orig_h = int(frame_size[0])
    orig_w = int(frame_size[1])
    if frame_scale is not None:
        sx = float(frame_scale[0])
        sy = float(frame_scale[1])
        scaled_w = max(1, min(img.shape[1], int(round(orig_w * sx))))
        scaled_h = max(1, min(img.shape[0], int(round(orig_h * sy))))
        img = np.ascontiguousarray(img[:scaled_h, :scaled_w])
    return img, (orig_h, orig_w)


class PreviewCollector:
    """Lightweight single-frame preview collector attached to an inference pass.

    Once a frame with at least one valid prediction has been observed, all
    subsequent calls to :meth:`observe` return immediately in O(1) time.
    """

    def __init__(self) -> None:
        """Initialize an empty preview collector."""
        self._selected: Optional[
            Tuple[int, int, np.ndarray, Optional[Tuple[int, int]]]
        ] = None
        self._fallback: Optional[
            Tuple[int, int, np.ndarray, Optional[Tuple[int, int]]]
        ] = None

    @property
    def has_selected(self) -> bool:
        """Whether a frame with at least one prediction has already been captured."""
        return self._selected is not None

    def observe(self, batch: Any, outputs: Any) -> None:
        """Inspect one inference batch and capture a preview frame if needed."""
        if self._selected is not None:
            return
        images = getattr(batch, "images", None)
        if images is None or len(images) == 0:
            return

        n = len(images)
        frame_indices = getattr(outputs, "frame_indices", None)
        if frame_indices is None:
            frame_indices = getattr(batch, "frame_indices", None)
        video_indices = getattr(outputs, "video_indices", None)
        if video_indices is None:
            video_indices = getattr(batch, "video_indices", None)
        frame_sizes = getattr(batch, "frame_sizes", None)
        frame_scales = getattr(batch, "frame_scales", None)

        def _capture_at(
            i: int,
        ) -> Optional[Tuple[int, int, np.ndarray, Optional[Tuple[int, int]]]]:
            arr = _to_hwc_uint8(images[i])
            if arr is None or arr.size == 0:
                return None
            orig_hw: Optional[Tuple[int, int]] = None
            if frame_sizes is not None and i < len(frame_sizes):
                scale = (
                    frame_scales[i]
                    if frame_scales is not None and i < len(frame_scales)
                    else None
                )
                arr, orig_hw = _crop_fitted_padding(arr, frame_sizes[i], scale)
            v_idx = _extract_index(video_indices, i, default=0)
            f_idx = _extract_index(frame_indices, i, default=i)
            return (v_idx, f_idx, arr, orig_hw)

        if self._fallback is None:
            self._fallback = _capture_at(0)

        for i in range(n):
            if _frame_has_prediction(outputs, i):
                captured = _capture_at(i)
                if captured is not None:
                    self._selected = captured
                    return

    def to_preview_frames(self, videos: Optional[List[Any]]) -> List[Any]:
        """Encode the captured frame into a ``[sio.PreviewFrame]`` list.

        Returns an empty list if nothing was captured, if the installed
        ``sleap_io`` does not yet expose ``PreviewFrame``, or if encoding fails.
        """
        entry = self._selected if self._selected is not None else self._fallback
        if entry is None or not videos:
            return []
        try:
            import sleap_io as sio

            preview_cls = getattr(sio, "PreviewFrame", None)
            if preview_cls is None:
                return []
            v_idx, f_idx, img, orig_hw = entry
            if not (0 <= v_idx < len(videos)):
                return []
            video = videos[v_idx]
            if video is None:
                return []
            pf = preview_cls.from_image(video=video, frame_idx=f_idx, image=img)
            if orig_hw is not None:
                pf.orig_height = int(orig_hw[0])
                pf.orig_width = int(orig_hw[1])
            return [pf]
        except Exception:  # noqa: BLE001 - preview capture must never fail inference
            return []
