"""GPU + real-transformers end-to-end smoke for the SAM3 mask backend.

Every other SAM3 test (``test_sam3_backend.py``) mocks the transformers
``Sam3TrackerModel`` / ``Sam3TrackerProcessor`` pair with ``FakeSam3Model`` /
``FakeSam3Processor``. Those fakes assert the assumed transformers SAM3 contract
*tautologically* — they hand back exactly the shapes the backend reads, so a
real drift in the transformers API would never trip them. This test is the
single place the *real* SAM3 image visual-prompt path is exercised end to end
(load gated ``facebook/sam3`` -> processor -> ``Sam3TrackerModel`` forward ->
``post_process_masks`` -> ``_pick`` -> ``_cleanup_speckle`` -> bool masks), and
where the literal class names ``Sam3TrackerModel`` / ``Sam3TrackerProcessor``
are confirmed to resolve (they are imported transitively by
``Sam3Backend.from_pretrained`` -> ``_load_sam3``).

Gating (mirrors ``test_e2e_gpu.py``) — it SKIPS CLEANLY in CI, which has none of:
  * ``torch.cuda.is_available()`` (SAM3 is GPU-only end to end);
  * ``transformers`` importable (the ``sleap_nn[sam3]`` extra; SAM3 needs
    ``transformers>=5``);
  * a resolvable gated-weights signal — ``HF_TOKEN`` in the env, OR a cached
    ``~/.cache/huggingface/token`` (from ``huggingface-cli login``), OR
    ``SLEAP_NN_SAM3_MODEL`` pointing at local/override weights (which also
    overrides the model id passed to ``from_pretrained``); and
  * the prototype val ``.slp`` present (reuses ``SLEAP_NN_SAM_VAL`` like
    ``test_e2e_gpu.py``).

CROSS-CHECK of the assumed transformers SAM3 contract
-----------------------------------------------------
The fakes encode a contract the backend (``Sam3Backend.masks``) depends on but
never verifies against the real library. When this test runs against real SAM3
it must FAIL LOUDLY if any of these break:

  * The processor output is a mapping carrying ``original_sizes`` — the backend
    passes ``inputs["original_sizes"]`` straight into ``post_process_masks``
    (``FakeSam3Processor.__call__`` returns ``original_sizes=[hw]``).
  * The model output carries ``pred_masks`` AND ``iou_scores`` (consumed as
    ``out.pred_masks`` / ``out.iou_scores``; ``FakeSam3Model.__call__`` returns
    a ``_FakeOut`` with exactly those two attributes).
  * ``processor.post_process_masks(..., binarize=True)`` returns a PER-IMAGE
    list whose first element is a ``(n_obj, n_cand, H, W)`` tensor
    (``FakeSam3Processor.post_process_masks`` returns ``[(n_obj, n_cand, H, W)]``;
    the backend takes ``[0]``).
  * ``iou_scores`` has shape ``(1, n_obj, n_cand)`` — the backend reads
    ``out.iou_scores.float().cpu().numpy()[0]`` to get ``(n_obj, n_cand)``
    (``FakeSam3Model`` emits ``self._scores[None, ...]``).

Each assertion below ties back to one of those contract points so a real-library
drift is caught here rather than silently passing the fakes.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import sleap_io as sio

# Reuse the same val asset env var/default as ``test_e2e_gpu.py`` so any GPU box
# can point both smokes at the same prototype data.
_VAL = os.environ.get(
    "SLEAP_NN_SAM_VAL",
    "/home/talmo/code/sleap-nn/scratch/2026-06-03-segmentation-methods/data/"
    "mice_of_seg_val.pkg.slp",
)

# ``facebook/sam3`` is GATED. Authentication is resolvable from any of:
#   * the ``HF_TOKEN`` env var,
#   * a cached token from ``huggingface-cli login``
#     (``~/.cache/huggingface/token``), or
#   * ``SLEAP_NN_SAM3_MODEL`` pointing at local/override weights (no auth needed),
# which ALSO overrides the model id handed to ``from_pretrained`` (so a box with
# local weights need not touch the gated hub at all). Do NOT import a module
# global for the model id (it was removed in the globals refactor); use
# ``Sam3Backend.from_pretrained``'s default ("facebook/sam3") unless overridden.
_SAM3_MODEL_OVERRIDE = os.environ.get("SLEAP_NN_SAM3_MODEL")
_HF_TOKEN = os.environ.get("HF_TOKEN")
_HF_CACHED_TOKEN = (Path("~/.cache/huggingface/token").expanduser()).is_file()
_HAS_GATED_WEIGHTS = bool(_SAM3_MODEL_OVERRIDE) or bool(_HF_TOKEN) or _HF_CACHED_TOKEN

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and importlib.util.find_spec("transformers") is not None
        and _HAS_GATED_WEIGHTS
        and os.path.exists(_VAL)
    ),
    reason=(
        "needs CUDA + transformers + a gated-weights signal "
        "(HF_TOKEN / cached HF token / SLEAP_NN_SAM3_MODEL) + val .slp"
    ),
)


def _pred_labels_two_frames():
    """Two-frame ``sio.Labels`` of PredictedInstance poses from the val .slp.

    Mirrors ``test_e2e_gpu.py``: the val file carries GT ``sio.Instance`` poses;
    convert them to ``PredictedInstance`` so the predicted-mask -> instance
    pairing is genuinely exercised. Two frames is plenty (the SAM3 forward is the
    slow part).
    """
    labels = sio.load_slp(_VAL)
    skel = labels.skeletons[0]
    pred_lfs = []
    for lf in labels.labeled_frames[:2]:
        pred_insts = []
        for inst in lf.instances:
            pts = inst.numpy()[:, :2]
            n = pts.shape[0]
            pred_insts.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    skeleton=skel,
                    point_scores=np.ones(n, np.float32),
                    score=1.0,
                )
            )
        pred_lfs.append(
            sio.LabeledFrame(
                video=lf.video, frame_idx=lf.frame_idx, instances=pred_insts
            )
        )
    return sio.Labels(
        videos=list(labels.videos), skeletons=[skel], labeled_frames=pred_lfs
    )


def _n_components(mask):
    """Number of connected components in a boolean mask (8-connectivity)."""
    from scipy import ndimage

    _, n = ndimage.label(np.asarray(mask, dtype=bool))
    return n


def test_sam3_backend_masks_real_contract():
    """Real SAM3 ``backend.masks`` honors the shape/cleanup contract the fakes assert.

    Builds pose prompts for one frame and drives ``Sam3Backend.masks`` directly so
    the cross-check (above) fails loudly if the transformers SAM3 surface drifts:
    the processor must expose ``original_sizes``, the model output must carry
    ``pred_masks`` + ``iou_scores``, ``post_process_masks(binarize=True)`` must be
    a per-image list of ``(n_obj, n_cand, H, W)``, and ``iou_scores`` must be
    ``(1, n_obj, n_cand)``. Any of those breaking raises inside ``backend.masks``
    rather than silently passing the tautological fakes.
    """
    from sleap_nn.inference.sam.backends import Sam3Backend
    from sleap_nn.inference.sam.prompts import pose_prompt

    small = _pred_labels_two_frames()

    # Construct the REAL backend. ``from_pretrained`` imports
    # ``Sam3TrackerModel`` / ``Sam3TrackerProcessor`` (the literal class names the
    # fakes stand in for); a NameError/ImportError here means those names no
    # longer resolve in the installed transformers. Use the default model id
    # ("facebook/sam3") unless an explicit local/override is supplied.
    model_id = _SAM3_MODEL_OVERRIDE or "facebook/sam3"
    backend = Sam3Backend.from_pretrained(model_id=model_id, device="cuda")

    # One frame's worth of pose prompts in full-frame pixel space.
    lf = small.labeled_frames[0]
    image = lf.image
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., 0]
    h, w = img.shape[:2]

    prompts = []
    for inst in lf.instances:
        pts = inst.numpy()[:, :2]
        # Skip an all-NaN instance (pose_prompt needs >=1 visible keypoint).
        if np.isfinite(pts).all(axis=1).any():
            prompts.append(pose_prompt(pts, (h, w)))
    assert prompts, "val frame must yield at least one pose prompt"

    masks, scores = backend.masks(image, prompts)

    # Contract: one mask + one score per prompt (the batched single-call path
    # returns ``len(prompts)`` of each).
    assert len(masks) == len(prompts) == len(scores)

    any_area = False
    for m, s in zip(masks, scores):
        m = np.asarray(m)
        # Each mask is a boolean (H, W) array matching the image HxW. This pins
        # the ``post_process_masks -> [0] -> (n_obj, n_cand, H, W)`` reshape: a
        # drift to per-object lists or transposed (H, W) would trip the shape
        # check (the backend itself also guards this and would raise).
        assert m.dtype == bool
        assert m.shape == (h, w)
        # The raw chosen-candidate iou score is finite (read from the
        # ``(1, n_obj, n_cand)`` ``iou_scores`` -> ``[0]`` -> per-object row).
        assert np.isfinite(s)
        if m.any():
            any_area = True
            # Mandatory SAM3 speckle cleanup ran: every non-empty returned mask
            # is a SINGLE connected component (raw SAM3 masks are ~14 components;
            # ``_cleanup_speckle`` collapses them to ~1). If cleanup were skipped
            # this would be >1 for the speckly real masks.
            assert _n_components(m) == 1
    # At least one mask has area (SAM3 found a real silhouette on a GT pose).
    assert any_area


def test_sam3_pose_masks_end_to_end(tmp_path):
    """Full ``run_sam_segmentation(mask_backend="sam3", ...)`` smoke (mirrors SAM1).

    Drives the SAM3 path through the same orchestration the SAM1 e2e test uses,
    asserting the emitted ``PredictedSegmentationMask`` objects carry finite
    scores, real area, are paired to their source predicted instance, and reload
    from the embedded ``.slp``.
    """
    from sleap_nn.inference.sam import run_sam_segmentation

    small = _pred_labels_two_frames()

    out_path = tmp_path / "sam3_pose_e2e.pkg.slp"
    overlay_path = tmp_path / "sam3_pose_e2e_overlay.png"
    model_id = _SAM3_MODEL_OVERRIDE or "facebook/sam3"
    out = run_sam_segmentation(
        small,
        mask_backend="sam3",
        prompt_mode="pose",
        sam3_model_id=model_id,
        device="cuda",
        output_path=out_path.as_posix(),
        overlay_path=overlay_path.as_posix(),
    )

    # Produced masks on at least one frame.
    assert any(len(lf.masks) > 0 for lf in out.labeled_frames)
    for lf in out.labeled_frames:
        for m in lf.masks:
            assert isinstance(m, sio.PredictedSegmentationMask)
            assert np.isfinite(m.score)
            # The mask is paired to its source predicted instance (PLAN L8).
            assert m.instance is not None
            data = np.asarray(m.data, dtype=bool)
            # A real mask has area...
            assert data.any()
            # ...and the SAM3 speckle cleanup left a single connected component.
            assert _n_components(data) == 1

    # The embedded SLP + overlay PNG are written and reload.
    assert out_path.exists()
    assert overlay_path.exists()
    sio.load_slp(out_path.as_posix())
