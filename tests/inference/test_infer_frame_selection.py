"""Regression tests: `.slp` inference frame-selection + video attachment.

PR #530 audit (Cluster B): the new flow's ``.slp`` path returned ``videos=None``
(so predicted frames referenced no video) and relied on ``LabelsProvider``'s
``only_labeled_frames=True`` default, whereas legacy ``LabelsReader`` predicts
ALL frames. The GT-fallback paths legitimately need labeled-only frames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sleap_nn.inference.predictor import Predictor

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CKPT = ASSETS / "model_ckpts"
DATA = ASSETS / "datasets"

SLP = DATA / "minimal_instance.pkg.slp"
SINGLE = CKPT / "minimal_instance_single_instance"


@pytest.mark.skipif(not (SLP.exists() and SINGLE.exists()), reason="missing fixtures")
def test_make_provider_slp_attaches_videos_and_predicts_all_frames():
    """Normal (non-GT) model on a .slp path: videos attached, all frames."""
    pred = Predictor.from_model_paths([str(SINGLE)], device="cpu")
    provider, videos = pred._make_provider(str(SLP))
    # Blocker fix: the real video is attached for label packaging.
    assert videos is not None and len(videos) >= 1
    # Legacy parity: a real-model predictor predicts ALL frames, not just
    # user-labeled ones.
    assert getattr(provider, "only_labeled_frames") is False


@pytest.mark.skipif(not (SLP.exists() and SINGLE.exists()), reason="missing fixtures")
def test_predict_on_slp_path_output_references_video():
    """End-to-end: predicting from a .slp path yields frames with a real video."""
    pred = Predictor.from_model_paths([str(SINGLE)], device="cpu", batch_size=4)
    labels = pred.predict(str(SLP), make_labels=True)
    assert len(labels.videos) >= 1
    assert all(lf.video is not None for lf in labels.labeled_frames)
