"""Export-adapter layers — thin translators around exported ONNX/TRT models.

The exported wrappers in :mod:`sleap_nn.export.wrappers` bake every
preprocessing + postprocessing step into the ONNX graph, including
``input_scale`` resize, ``output_stride`` rescaling, peak finding,
and (top-down) crop extraction. By the time output keys arrive, peaks
are already in **original-image pixel space** with the right shapes.

The standard :class:`InferenceLayer` subclasses in
``sleap_nn/inference/layers/`` were designed for ``.ckpt`` checkpoints,
where the layer's own ``preprocess`` does the input-scale resize and
``postprocess`` runs the coord ladder (``undo_stride`` /
``undo_input_scale``). Reusing them for export-loaded backends would
double-apply both transforms.

This module ships dedicated adapter layers for the export path, one
per exported model type. Each one:

* skips ``input_scale`` resize (the wrapper did it)
* feeds raw uint8 ``(B, C, H, W)`` tensors to the backend
* skips peak finding (already baked)
* skips the coord ladder (already in original-image space)
* translates the wrapper's output dict into a structured :class:`Outputs`

The classes intentionally don't inherit from :class:`InferenceLayer`
— they're shaped like layers (``predict(image, **kwargs) -> Outputs``)
but the layered preprocess/forward/postprocess pipeline doesn't apply.
"""

from __future__ import annotations

from typing import Any, Optional

import attrs
import numpy as np
import torch

from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.outputs import Outputs


def _to_4d_uint8_tensor(image: Any) -> torch.Tensor:
    """Coerce an image input to ``(B, C, H, W)`` uint8 (channel-first).

    Mirrors :meth:`InferenceLayer._to_4d_float_tensor` shape-wise, but
    keeps the dtype as the runtime expects (``ONNXBackend`` casts to
    its declared input dtype, typically ``uint8``, before invoking the
    session).
    """
    return InferenceLayer._to_4d_float_tensor(image)


@attrs.define
class ExportedSingleInstanceLayer:
    """Adapter for an ONNX/TRT-exported single-instance model.

    The wrapper output schema is:

    * ``peaks``: ``(B, N, 2)`` in original-image (x, y) space
    * ``peak_vals``: ``(B, N)``
    * ``confmaps`` (optional): ``(B, N, H, W)``

    ``Outputs.pred_keypoints`` uses ``(B, I, N, 2)`` so the adapter
    inserts a singleton instance dim.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
        return_confmaps: When ``True`` and the wrapper exports a
            ``confmaps`` output, echo it onto :attr:`Outputs.pred_confmaps`.
    """

    backend: Any
    return_confmaps: bool = False

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        peaks = raw["peaks"]  # (B, N, 2)
        vals = raw["peak_vals"]  # (B, N)
        # Add singleton instance dim per the Outputs convention.
        out = Outputs(
            pred_keypoints=peaks.unsqueeze(1),  # (B, 1, N, 2)
            pred_peak_values=vals.unsqueeze(1),  # (B, 1, N)
        )
        if self.return_confmaps and "confmaps" in raw:
            out = attrs.evolve(out, pred_confmaps=raw["confmaps"])
        return out


@attrs.define
class ExportedCenteredInstanceLayer:
    """Adapter for an ONNX/TRT-exported centered-instance model.

    Standalone centered-instance is invoked on pre-cropped images. The
    wrapper outputs the same shape as single-instance:

    * ``peaks``: ``(B_crops, N, 2)`` in crop (x, y) space
    * ``peak_vals``: ``(B_crops, N)``

    ``Outputs.pred_keypoints`` of shape ``(B_crops, 1, N, 2)``.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
        return_confmaps: Echo ``confmaps`` onto
            :attr:`Outputs.pred_confmaps` when present.
    """

    backend: Any
    return_confmaps: bool = False

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        peaks = raw["peaks"]  # (B_crops, N, 2)
        vals = raw["peak_vals"]
        out = Outputs(
            pred_keypoints=peaks.unsqueeze(1),  # (B_crops, 1, N, 2)
            pred_peak_values=vals.unsqueeze(1),  # (B_crops, 1, N)
        )
        if self.return_confmaps and "confmaps" in raw:
            out = attrs.evolve(out, pred_confmaps=raw["confmaps"])
        return out


@attrs.define
class ExportedCentroidLayer:
    """Adapter for an ONNX/TRT-exported centroid model.

    The wrapper output schema:

    * ``centroids``: ``(B, I, 2)`` in original-image (x, y) space.
      Invalid slots are zero-filled and flagged in ``instance_valid``.
    * ``centroid_vals``: ``(B, I)``
    * ``instance_valid``: ``(B, I)`` bool

    Translates to ``Outputs.pred_centroids`` / ``pred_centroid_values``.
    Invalid slots are turned into ``NaN`` per the ``Outputs`` convention.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
    """

    backend: Any

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        centroids = raw["centroids"].clone()  # (B, I, 2)
        centroid_vals = raw["centroid_vals"].clone()  # (B, I)
        valid = raw["instance_valid"].bool()  # (B, I)

        # Replace invalid zero-padded slots with NaN per Outputs convention.
        invalid = ~valid
        centroids[invalid] = float("nan")
        centroid_vals[invalid] = float("nan")

        return Outputs(
            pred_centroids=centroids,
            pred_centroid_values=centroid_vals,
            instance_valid=valid,
        )


@attrs.define
class ExportedTopDownLayer:
    """Adapter for an ONNX/TRT-exported combined top-down model.

    The export bakes both centroid + centered-instance stages plus the
    crop extraction step into a single ONNX graph. Wrapper output:

    * ``centroids``: ``(B, I, 2)`` in original-image (x, y) space
    * ``centroid_vals``: ``(B, I)``
    * ``peaks``: ``(B, I, N, 2)`` in original-image (x, y) space
      (crop offset already added back inside the graph)
    * ``peak_vals``: ``(B, I, N)``
    * ``instance_valid``: ``(B, I)`` bool

    Invalid slots are zero-filled by the wrapper and flagged in
    ``instance_valid``. The adapter NaN-pads invalid slots per the
    ``Outputs`` convention.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
    """

    backend: Any

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        centroids = raw["centroids"].clone()
        centroid_vals = raw["centroid_vals"].clone()
        peaks = raw["peaks"].clone()
        peak_vals = raw["peak_vals"].clone()
        valid = raw["instance_valid"].bool()

        invalid = ~valid
        centroids[invalid] = float("nan")
        centroid_vals[invalid] = float("nan")
        peaks[invalid] = float("nan")
        peak_vals[invalid] = float("nan")

        return Outputs(
            pred_keypoints=peaks,
            pred_peak_values=peak_vals,
            pred_centroids=centroids,
            pred_centroid_values=centroid_vals,
            instance_valid=valid,
        )
