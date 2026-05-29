"""Strong full-pipeline parity: new ``Predictor`` vs legacy, on CPU AND CUDA.

The pre-existing ``test_parity_vs_legacy.py`` only ran on CPU (``device="cpu"``
hardcoded) and, for top-down, fell back to a 1.0 px nearest-neighbour tier that
did not assert scores, visibility, or per-frame instance counts. The PR body
nonetheless claimed "≤0.001 px parity on CPU and CUDA". This module closes that
gap: it runs both flows on each available device and asserts, per aligned
instance, tight coordinate parity PLUS instance-count, visibility, and
instance-score parity.

Multi-class track/score parity is covered separately in
``test_multiclass_identity_parity.py``.

Marked ``slow`` because each case loads two predictors.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sleap_io as sio

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="sleap_nn.inference.predictors"
)

CKPT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
DATA = Path(__file__).resolve().parents[1] / "assets" / "datasets"

FIXTURES = [
    (
        "single_instance",
        [CKPT / "minimal_instance_single_instance"],
        DATA / "small_robot.mp4",
        4,
        0.3,
        None,
    ),
    (
        "topdown",
        [
            CKPT / "minimal_instance_centroid",
            CKPT / "minimal_instance_centered_instance",
        ],
        DATA / "centered_pair_small.mp4",
        4,
        0.03,
        6,
    ),
    (
        "bottomup",
        [CKPT / "minimal_instance_bottomup"],
        DATA / "centered_pair_small.mp4",
        4,
        0.05,
        None,
    ),
]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _frames_map(labels: sio.Labels) -> dict:
    m: dict = {}
    for lf in labels.labeled_frames:
        rows = []
        for inst in lf.instances:
            sc = getattr(inst, "score", None)
            rows.append(
                (
                    np.asarray(inst.numpy(), dtype=float),
                    float(sc) if sc is not None else np.nan,
                )
            )
        m.setdefault(int(lf.frame_idx), []).extend(rows)
    return m


def _run_legacy(models, source, n, device, peak_threshold, max_instances):
    from omegaconf import OmegaConf

    from sleap_nn.inference.predictors import Predictor as Legacy

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
    kw = dict(peak_threshold=peak_threshold, preprocess_config=pp, device=device)
    if max_instances is not None:
        kw["max_instances"] = max_instances
    pred = Legacy.from_model_paths([str(p) for p in models], **kw)
    pred._initialize_inference_model()
    pred.make_pipeline(inference_object=str(source), frames=list(range(n)))
    return pred.predict(make_labels=True)


def _run_new(models, source, n, device, peak_threshold, max_instances):
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import VideoProvider

    pred = Predictor.from_model_paths(
        [str(p) for p in models],
        device=device,
        batch_size=n,
        peak_threshold=peak_threshold,
        max_instances=max_instances,
    )
    video = sio.load_video(str(source))
    skel = sio.Skeleton(nodes=[sio.Node(f"n{i}") for i in range(2)])
    provider = VideoProvider(video=video, batch_size=n, frames=list(range(n)))
    return pred.predict(provider, make_labels=True, skeleton=skel, videos=[video])


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("label", "models", "source", "n", "thr", "max_inst"),
    FIXTURES,
    ids=[f[0] for f in FIXTURES],
)
def test_full_pipeline_parity(label, models, source, n, thr, max_inst, device):
    """New flow reproduces legacy keypoints, counts, visibility, and score."""
    if not _have(source, *models):
        pytest.skip(f"missing fixtures for {label}")
    legacy = _frames_map(_run_legacy(models, source, n, device, thr, max_inst))
    new = _frames_map(_run_new(models, source, n, device, thr, max_inst))

    assert sorted(legacy) == sorted(new), "frame indices diverged"
    for f in legacy:
        L, N = legacy[f], new[f]
        assert len(L) == len(
            N
        ), f"[{label}/{device}] frame {f}: instance count {len(L)} vs {len(N)}"
        used = set()
        for lp, ls in L:
            # Align by nearest instance (identity-aware), then assert tightly.
            best, bestd = -1, 1e18
            for j, (npp, _) in enumerate(N):
                if j in used:
                    continue
                both = np.isfinite(lp).all(1) & np.isfinite(npp).all(1)
                d = (
                    float(np.linalg.norm(lp[both] - npp[both], axis=1).mean())
                    if both.any()
                    else 1e9
                )
                if d < bestd:
                    bestd, best = d, j
            assert best >= 0
            used.add(best)
            npp, ns = N[best]
            lvis, nvis = np.isfinite(lp).all(1), np.isfinite(npp).all(1)
            np.testing.assert_array_equal(
                lvis, nvis, err_msg=f"[{label}/{device}] visibility mismatch"
            )
            both = lvis & nvis
            if both.any():
                drift = float(np.abs(lp[both] - npp[both]).max())
                # Sub-pixel tolerance, ~100x tighter than the legacy test's 1px
                # tier. 1e-2 (not 5e-3) absorbs cross-platform float differences
                # (macOS Accelerate BLAS gives ~5e-3 on top-down) while still
                # catching any real coordinate-ladder regression.
                assert (
                    drift < 1e-2
                ), f"[{label}/{device}] coord drift {drift} px exceeds 1e-2"
            if np.isfinite(ls) and np.isfinite(ns):
                assert (
                    abs(ls - ns) < 1e-2
                ), f"[{label}/{device}] instance score drift {abs(ls - ns)}"
