"""An exported network standing in for a checkpoint in the checkpoint pipeline.

A *network-only* export (``sleap-nn export --network-only``; every box detector
export is one) holds the network and nothing else: preprocessed image in, raw head
outputs out. :class:`ExportedNetwork` runs such a graph on ONNX Runtime or TensorRT
behind the same ``forward`` contract as the model's ``LightningModule``, so
``sleap-nn predict`` can load an export directory wherever it loads a checkpoint
directory (``-m det_export -m pose_export --part_model tail_export``) and keep every
host-side step its own: size matching and letterboxing, box decoding, fixed or
box-scaled crops, peak finding and refinement, node renaming, the part stage. Those
steps are the checkpoint path's code, so exported and checkpoint predictions differ
only by the runtime's numerics.

The runtime is chosen per load (:data:`EXPORT_RUNTIME`, set by
``load_model_assets(export_runtime=...)``): ``"auto"`` prefers ``model.trt`` when the
directory has one, else ``model.onnx``.
"""

from __future__ import annotations

import contextvars
import json
from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn

#: Runtime used for export directories met while loading model assets.
EXPORT_RUNTIME: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sleap_nn_export_runtime", default="auto"
)

#: Model types whose exports are network-only by construction.
_NETWORK_ONLY_TYPES = ("bbox_centernet", "bbox_ssd", "bbox_detr")


def is_network_export(path: Union[str, Path]) -> bool:
    """True if ``path`` is an export directory holding a network-only graph.

    That is a directory with ``export_metadata.json`` and ``model.onnx`` or
    ``model.trt`` whose metadata says ``network_only`` (``--network-only``) or whose
    model type is a box detector (detectors always export their network only).
    """
    p = Path(path)
    meta = p / "export_metadata.json"
    if not (p.is_dir() and meta.exists()):
        return False
    if not ((p / "model.onnx").exists() or (p / "model.trt").exists()):
        return False
    data = json.loads(meta.read_text())
    return bool(data.get("network_only")) or data.get("model_type") in (
        _NETWORK_ONLY_TYPES
    )


class ExportedNetwork(nn.Module):
    """A network-only export behind a ``LightningModule``-style ``forward``.

    ``forward`` takes what the checkpoint pipeline hands a ``LightningModule``
    (``(B, 1, C, H, W)`` or ``(B, C, H, W)`` frames or crops, preprocessed, values in
    ``[0, 255]``) and returns what that module's ``forward`` returns: the confidence
    maps tensor for a centered-instance model, the decoder-ready head dict for a box
    detector.

    Args:
        export_dir: The export directory.
        runtime: ``"auto"``, ``"onnx"`` or ``"tensorrt"``.
        device: Device the outputs are returned on (and the TensorRT engine's).
    """

    def __init__(
        self, export_dir: Union[str, Path], runtime: str = "auto", device: str = "cpu"
    ):
        """Load the graph."""
        super().__init__()
        from sleap_nn.inference.predictor import (
            _build_export_backend,
            _resolve_export_runtime,
        )

        self.export_dir = Path(export_dir)
        meta = json.loads((self.export_dir / "export_metadata.json").read_text())
        self.model_type = meta["model_type"]
        self.head_names = meta.get("output_names") or None
        # A uint8 graph gets the rounded image (as `ExportedBBoxLayer` sends it);
        # a float32 graph (`--input-dtype float32`) the image as it is.
        self.input_dtype = meta.get("input_dtype", "uint8")
        # The graph's channels are fixed at export; `Model.forward` coerces 1 <-> 3
        # channels inside the checkpoint's forward, so the same happens here.
        self.input_channels = int(meta.get("input_channels") or 0)
        # Inputs larger than the graph's batch limit (a TensorRT profile's max)
        # run in chunks; ONNX graphs take any batch.
        self.max_batch = int(meta.get("max_batch_size") or 1)
        self.runtime, path = _resolve_export_runtime(self.export_dir, runtime)
        if self.runtime == "tensorrt" and not str(device).startswith("cuda"):
            raise ValueError(
                f"{self.export_dir}: a TensorRT engine runs on CUDA only (device "
                f"{device!r}); use the ONNX runtime."
            )
        self.backend = _build_export_backend(self.runtime, path, device)
        self.out_device = torch.device("cuda" if str(device) == "cuda" else str(device))

    def forward(
        self, img: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the graph on a preprocessed batch (see the class docstring)."""
        if img.ndim == 5:
            img = img.squeeze(1)
        c = self.input_channels
        if c and img.shape[-3] != c:
            if img.shape[-3] == 1:
                img = img.repeat(1, c, 1, 1)
            elif img.shape[-3] == 3 and c == 1:
                import torchvision.transforms.v2.functional as F

                img = F.rgb_to_grayscale(img.float(), num_output_channels=1)
        if img.dtype == torch.uint8:
            img = img if self.input_dtype == "uint8" else img.float()
        else:
            img = img.float().clamp(0, 255)
            if self.input_dtype == "uint8":
                img = img.round().to(torch.uint8)
        n = img.shape[0]
        step = max(1, n) if self.runtime == "onnx" else max(1, self.max_batch)
        # No crops in a batch: run one dummy so the outputs keep their shapes.
        chunks = [img[i : i + step] for i in range(0, n, step)] or [
            torch.zeros((1,) + tuple(img.shape[1:]), dtype=img.dtype)
        ]
        parts = [self.backend(c) for c in chunks]
        for c, q in zip(chunks, parts):
            for k, v in q.items():
                if v.ndim == 0 or v.shape[0] != c.shape[0]:
                    raise RuntimeError(
                        f"{self.export_dir}: output '{k}' (shape {tuple(v.shape)}) is "
                        f"not batch-first for a batch of {c.shape[0]}; this export "
                        "cannot run in the checkpoint pipeline."
                    )
        out = {
            k: torch.cat([q[k].to(self.out_device).float() for q in parts])[:n]
            for k in parts[0]
        }
        if self.model_type == "centered_instance":
            name = (self.head_names or ["CenteredInstanceConfmapsHead"])[0]
            return out[name]
        return out
