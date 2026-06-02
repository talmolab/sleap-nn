"""Tests for the ``Outputs`` instance-segmentation mask carrier (PR 2).

Covers the ``pred_masks`` field: repr, batch_size, pickle-safe ``slim()``,
``to_masks``, and ``to_labels`` mask attachment + ``.slp`` roundtrip.
"""

import pickle

import numpy as np
import torch

import sleap_io as sio

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.segmentation_convert import build_predicted_segmentation_mask


def _two_mask_outputs():
    m1 = np.zeros((32, 32), dtype=bool)
    m1[2:10, 2:10] = True
    m2 = np.zeros((32, 32), dtype=bool)
    m2[20:28, 20:28] = True
    pred_masks = [[{"mask": m1, "score": 0.9}, {"mask": m2, "score": 0.7}]]
    out = Outputs(
        pred_masks=pred_masks,
        frame_indices=torch.tensor([3]),
        video_indices=torch.tensor([0]),
    )
    return out, m1, m2


def test_build_predicted_segmentation_mask():
    """Helper produces a PredictedSegmentationMask carrying the score."""
    m = np.zeros((16, 16), dtype=bool)
    m[2:8, 2:8] = True
    pm = build_predicted_segmentation_mask(m, 0.42)
    assert isinstance(pm, sio.PredictedSegmentationMask)
    assert abs(pm.score - 0.42) < 1e-5
    np.testing.assert_array_equal(pm.data, m)


def test_outputs_pred_masks_repr_and_batch_size():
    """pred_masks gives a compact repr and a correct batch size."""
    out, _, _ = _two_mask_outputs()
    assert out.batch_size == 1
    r = repr(out)
    assert "pred_masks=list[1 frames, 2 masks]" in r
    # repr must not dump array contents
    assert "True" not in r and "array(" not in r


def test_outputs_slim_keeps_masks_and_is_picklable():
    """slim() retains masks (they are output, not heavy) and is pickle-safe."""
    out, _, _ = _two_mask_outputs()
    slim = out.slim()
    assert slim.pred_masks is not None and len(slim.pred_masks) == 1
    pickle.dumps(slim)  # must not raise


def test_outputs_to_masks():
    """to_masks builds PredictedSegmentationMask objects for a batch slot."""
    out, m1, m2 = _two_mask_outputs()
    masks = out.to_masks(batch_index=0)
    assert len(masks) == 2
    assert all(isinstance(m, sio.PredictedSegmentationMask) for m in masks)
    assert sorted(round(m.score, 2) for m in masks) == [0.7, 0.9]


def test_outputs_to_masks_skips_empty():
    """Empty masks are dropped by to_masks."""
    out = Outputs(pred_masks=[[{"mask": np.zeros((8, 8), bool), "score": 0.5}]])
    assert out.to_masks(0) == []


def test_outputs_to_labels_with_masks_roundtrip(tmp_path):
    """to_labels attaches masks; they roundtrip through .slp (no skeleton)."""
    out, m1, m2 = _two_mask_outputs()
    video = sio.Video.from_filename("dummy.mp4")
    labels = out.to_labels(skeleton=None, videos=[video])

    assert len(labels.masks) == 2
    assert labels[0].frame_idx == 3
    # No skeleton for a mask-only model.
    assert list(labels.skeletons) == []

    out_path = tmp_path / "seg_outputs.slp"
    labels.save(out_path.as_posix())
    reloaded = sio.load_slp(out_path.as_posix())
    assert len(reloaded.masks) == 2
    assert sorted(round(m.score, 2) for m in reloaded.masks) == [0.7, 0.9]
    assert sorted(int(m.data.sum()) for m in reloaded.masks) == sorted(
        [int(m1.sum()), int(m2.sum())]
    )


def test_outputs_to_labels_empty_masks_frame_skipped():
    """A frame whose only content is empty masks is skipped."""
    out = Outputs(
        pred_masks=[[{"mask": np.zeros((8, 8), bool), "score": 0.5}]],
        frame_indices=torch.tensor([0]),
        video_indices=torch.tensor([0]),
    )
    video = sio.Video.from_filename("dummy.mp4")
    labels = out.to_labels(skeleton=None, videos=[video])
    assert len(labels.labeled_frames) == 0
