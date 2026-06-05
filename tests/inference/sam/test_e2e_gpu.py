"""GPU + real-SAM end-to-end smoke for the prompted mask producer.

Gated on (a) CUDA, (b) ``segment-anything`` installed, and (c) a ViT-H
checkpoint + the prototype val ``.slp`` being present — so it skips cleanly in
CI (which has none of these) and runs only on a GPU box with the SAM assets.
This is the single end-to-end check that the *real* SAM path
(load -> encode -> prompt -> pick -> PredictedSegmentationMask -> SLP) works;
every other test uses a fake backend.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch

import sleap_io as sio

# Asset paths are overridable via env vars so any GPU box can run this smoke
# (defaults point at the prototype assets on the dev machine).
_CKPT = os.environ.get(
    "SLEAP_NN_SAM_CKPT",
    "/home/talmo/code/sleap-nn/scratch/2026-06-03-segmentation-methods/"
    "experiments/07-sam-pseudomasks/models/sam_vit_h_4b8939.pth",
)
_VAL = os.environ.get(
    "SLEAP_NN_SAM_VAL",
    "/home/talmo/code/sleap-nn/scratch/2026-06-03-segmentation-methods/data/"
    "mice_of_seg_val.pkg.slp",
)

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and importlib.util.find_spec("segment_anything") is not None
        and os.path.exists(_CKPT)
        and os.path.exists(_VAL)
    ),
    reason="needs CUDA + segment-anything + the SAM ViT-H checkpoint + val .slp",
)


def test_sam_pose_masks_end_to_end(tmp_path):
    from sleap_nn.inference.sam import run_sam_segmentation

    labels = sio.load_slp(_VAL)
    skel = labels.skeletons[0]
    # The val file carries GT ``sio.Instance`` poses; convert them to predicted
    # instances so the PLAN-L8 ``mask.instance`` pairing is genuinely exercised
    # (a predicted mask is never pinned to a GT user annotation). Two frames is
    # plenty for a smoke test (SAM encode is the slow part).
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
    small = sio.Labels(
        videos=list(labels.videos), skeletons=[skel], labeled_frames=pred_lfs
    )

    out_path = tmp_path / "sam_pose_e2e.pkg.slp"
    overlay_path = tmp_path / "sam_pose_e2e_overlay.png"
    out = run_sam_segmentation(
        small,
        mask_backend="sam",
        prompt_mode="pose",
        sam_checkpoint=_CKPT,
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
            # PLAN L8: the mask is paired to its source predicted instance.
            assert m.instance is not None
            # A real mask has area.
            assert np.asarray(m.data, dtype=bool).any()

    # The embedded SLP + overlay PNG are written and reload.
    assert out_path.exists()
    assert overlay_path.exists()
    sio.load_slp(out_path.as_posix())
