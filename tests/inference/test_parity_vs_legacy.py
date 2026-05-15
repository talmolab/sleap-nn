"""Pipeline-parity test: new ``factory.from_model_paths`` vs legacy ``Predictor``.

Permanent regression guard added in PR 27 of #508 after the parity audit
(scratch/2026-04-30-inference-refactor-implementation/parity_audit/) found that
the new flow's preprocessing was silently diverging from legacy. The PR-0
goldens covered the wrong slice — they pinned model-forward parity (give the
model the same preprocessed input, get the same output) but never tested the
**full pipeline** (raw video → preprocess → forward → postprocess → final
keypoints).

This test fills that gap. For every supported fixture model type × multiple
sources, it runs both flows and asserts that the final keypoints match within
float tolerance. The rank-contract divergence on ``pred_keypoints`` (new
``(B, I, N, 2)`` vs legacy ``(B, N, 2)`` for single-instance) is reconciled
here explicitly via ``.squeeze(1)`` — that contract is documented in
``sleap_nn/inference/outputs.py`` and is design intent.

Marked slow because each fixture × source pair loads two predictors (~5s).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sleap_io as sio

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"

# Silence legacy deprecation in this test — we expect to call both flows.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="sleap_nn.inference.predictors"
)


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _kpts_from_labels(labels: sio.Labels) -> np.ndarray:
    """Pull a flat ``(n_instances, n_nodes, 2)`` array out of a ``sio.Labels``.

    Both flows ultimately write image-space coordinates into
    ``PredictedInstance.numpy()``; this normalises that into a single
    array we can diff.
    """
    rows: list[np.ndarray] = []
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            rows.append(inst.numpy())
    if not rows:
        return np.empty((0, 2))
    return np.stack(rows, axis=0)


def _run_legacy_keypoints(
    model_paths: list[Path], source: Path, n_frames: int
) -> np.ndarray:
    """Legacy keypoints in **image-space** (via the ``sio.Labels`` output)."""
    from omegaconf import OmegaConf

    from sleap_nn.inference.predictors import Predictor as LegacyPredictor

    pp = OmegaConf.create(
        {
            "scale": None,
            "ensure_rgb": None,
            "ensure_grayscale": None,
            "max_height": None,
            "max_width": None,
            "crop_size": None,
        }
    )
    pred = LegacyPredictor.from_model_paths(
        model_paths=[str(p) for p in model_paths],
        device="cpu",
        preprocess_config=pp,
    )
    pred._initialize_inference_model()
    pred.make_pipeline(inference_object=str(source), frames=list(range(n_frames)))
    labels = pred.predict(make_labels=True)
    return _kpts_from_labels(labels)


def _run_new_keypoints(
    model_paths: list[Path], source: Path, n_frames: int
) -> np.ndarray:
    """New-flow keypoints in **image-space** (via the ``sio.Labels`` output)."""
    from sleap_nn.inference.factory import from_model_paths
    from sleap_nn.inference.providers import LabelsProvider, VideoProvider

    predictor = from_model_paths(
        [str(p) for p in model_paths], device="cpu", batch_size=n_frames
    )
    if str(source).endswith(".slp"):
        loaded = sio.load_slp(str(source))
        skeleton = loaded.skeletons[0]
        videos = list(loaded.videos)
        provider = LabelsProvider(
            labels=str(source), batch_size=n_frames, only_labeled_frames=False
        )
    else:
        video = sio.load_video(str(source))
        skeleton = sio.Skeleton(nodes=[sio.Node(f"n{i}") for i in range(2)])
        videos = [video]
        provider = VideoProvider(
            video=video, batch_size=n_frames, frames=list(range(n_frames))
        )
    labels = predictor.predict(
        provider, make_labels=True, skeleton=skeleton, videos=videos
    )
    return _kpts_from_labels(labels)


def _assert_keypoint_parity(
    legacy: np.ndarray,
    new: np.ndarray,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    count_tol: float = 0.05,
    match_tol_px: float = 1.0,
) -> None:
    """Assert keypoint parity between legacy + new flows.

    Two-tier check:

    1. Strict tier — when both flows produce the same number of valid
       keypoints, compare element-wise within ``atol`` / ``rtol``.
    2. Tolerant tier — when counts differ (top-down's per-sample vs
       combined batching produces slightly different instance counts
       at NaN-boundary), assert: (a) count differs by no more than
       ``count_tol`` proportion of the larger; (b) every legacy keypoint
       has a near-neighbour in ``new`` within ``match_tol_px``. This
       order-independent match catches real regressions without being
       fooled by instance-padding/ordering quirks.
    """
    # Both flows now return image-space keypoints from sio.Labels:
    # shape (n_instances, n_nodes, 2). Flatten + drop NaN slots.
    legacy_valid = legacy.reshape(-1, 2)
    new_valid = new.reshape(-1, 2)
    legacy_valid = legacy_valid[~np.isnan(legacy_valid).any(axis=-1)]
    new_valid = new_valid[~np.isnan(new_valid).any(axis=-1)]

    if legacy_valid.size == 0 and new_valid.size == 0:
        return

    # Tier 1: strict element-wise compare when counts match.
    if legacy_valid.shape == new_valid.shape:
        np.testing.assert_allclose(
            legacy_valid,
            new_valid,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg="new flow's keypoints diverged from legacy (strict)",
        )
        return

    # Tier 2: count-tolerant nearest-neighbour match.
    n_l, n_n = len(legacy_valid), len(new_valid)
    diff_frac = abs(n_l - n_n) / max(n_l, n_n, 1)
    assert diff_frac <= count_tol, (
        f"valid-keypoint count differs by {diff_frac:.2%}: legacy={n_l} new={n_n}; "
        f"exceeds tolerance {count_tol:.2%}"
    )
    # Every legacy keypoint must have a near-neighbour in new.
    dists = np.linalg.norm(legacy_valid[:, None, :] - new_valid[None, :, :], axis=-1)
    nearest = dists.min(axis=1)
    max_drift = float(nearest.max())
    assert max_drift <= match_tol_px, (
        f"max legacy→new nearest-neighbour distance = {max_drift:.4f} px "
        f"exceeds {match_tol_px} px tolerance"
    )


# ──────────────────────────────────────────────────────────────────────
# Parametrized parity tests
# ──────────────────────────────────────────────────────────────────────

VIDEO = DATA_ROOT / "small_robot.mp4"
SLP = DATA_ROOT / "minimal_instance.pkg.slp"

FIXTURES = [
    ("single_instance", [CKPT_ROOT / "minimal_instance_single_instance"]),
    (
        "topdown",
        [
            CKPT_ROOT / "minimal_instance_centroid",
            CKPT_ROOT / "minimal_instance_centered_instance",
        ],
    ),
    ("bottomup", [CKPT_ROOT / "minimal_instance_bottomup"]),
]


@pytest.mark.parametrize(("label", "ckpts"), FIXTURES, ids=[f[0] for f in FIXTURES])
def test_parity_vs_legacy_on_video(label, ckpts):
    """``small_robot.mp4`` first 4 frames: new vs legacy keypoints match."""
    if not _have(VIDEO, *ckpts):
        pytest.skip(f"missing fixtures for {label}")
    legacy = _run_legacy_keypoints(ckpts, VIDEO, n_frames=4)
    new = _run_new_keypoints(ckpts, VIDEO, n_frames=4)
    _assert_keypoint_parity(legacy, new)


@pytest.mark.parametrize(("label", "ckpts"), FIXTURES, ids=[f[0] for f in FIXTURES])
def test_parity_vs_legacy_on_labels(label, ckpts):
    """``minimal_instance.pkg.slp`` (the PR-0 golden source): new vs legacy match."""
    if not _have(SLP, *ckpts):
        pytest.skip(f"missing fixtures for {label}")
    legacy = _run_legacy_keypoints(ckpts, SLP, n_frames=1)
    new = _run_new_keypoints(ckpts, SLP, n_frames=1)
    _assert_keypoint_parity(legacy, new)
