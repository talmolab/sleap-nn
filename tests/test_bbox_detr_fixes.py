"""Unit tests for bbox_detr dataset, export, rescoring, and distillation fixes."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Negative/empty primary frames under zoom_out_canvas="frame"
# ─────────────────────────────────────────────────────────────────────────────


def test_detr_bbox_dataset_negative_frame_stays_empty_under_frame_canvas(monkeypatch):
    """Empty primary frames zoom onto a black canvas instead of pasting onto positive frames."""
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    ds = DetrBBoxDataset.__new__(DetrBBoxDataset)
    ds.apply_aug = True
    ds.detr_aug = {
        "crop_p": 0.0,
        "shortest_sides": [64],
        "crop_range": [32, 64],
        "flip_p": 0.0,
        "vflip_p": 0.0,
        "rot90_p": 0.0,
        "zoom_out_p": 1.0,
        "zoom_out_min": 0.5,
        "zoom_out_max": 0.8,
        "zoom_out_canvas": "frame",
        "paste_p": 0.0,
    }
    ds.train_hw = (64, 64)
    ds.resize_mode = "letterbox"
    ds._kpt_names = ()
    ds._kpt_flip_order = ()
    ds._capture_hulls = False

    zoom_on_frame_mock = MagicMock(
        side_effect=AssertionError("_zoom_out_on_frame must not run for empty frames")
    )
    ds._zoom_out_on_frame = zoom_on_frame_mock

    # Stub super().__getitem__ (BBoxDataset.__getitem__) to return an empty frame.
    empty_sample = {
        "image": torch.full((1, 3, 64, 64), 128, dtype=torch.uint8),
        "instances": torch.empty((1, 0, 4, 2), dtype=torch.float32),
        "_classes": torch.empty((0,), dtype=torch.long),
    }
    monkeypatch.setattr(
        DetrBBoxDataset.__mro__[1],
        "__getitem__",
        lambda self, idx: {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in empty_sample.items()
        },
    )

    out = ds[0]
    zoom_on_frame_mock.assert_not_called()
    assert out["instances"].shape == (1, 0, 4, 2)
    assert out["_classes"].shape == (0,)

    # Positive frame with boxes DOES use _zoom_out_on_frame when zoom_out_canvas="frame".
    called = []

    def _fake_zoom_out_on_frame(img, boxes, classes, zo_range, *pts):
        called.append((boxes.shape[0], zo_range))
        return (img, boxes, classes, *pts)

    ds._zoom_out_on_frame = _fake_zoom_out_on_frame
    pos_corners = torch.tensor(
        [[[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]]],
        dtype=torch.float32,
    )
    pos_sample = {
        "image": torch.full((1, 3, 64, 64), 128, dtype=torch.uint8),
        "instances": pos_corners,
        "_classes": torch.tensor([0], dtype=torch.long),
    }
    monkeypatch.setattr(
        DetrBBoxDataset.__mro__[1],
        "__getitem__",
        lambda self, idx: {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in pos_sample.items()
        },
    )
    out_pos = ds[0]
    assert len(called) == 1
    assert called[0][0] == 1
    assert out_pos["instances"].shape[1] == 1
