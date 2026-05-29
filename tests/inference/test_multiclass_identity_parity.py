"""Identity-parity regression: new vs legacy multi-class (ID) packaging.

PR #530 (inference refactor) initially DROPPED multi-class identity: the new
pipeline assigned no ``sio.Track`` to predicted instances, set no
``tracking_score``, and used the class probability as the instance ``score``.
That made ID models useless. This module locks in EXACT legacy behavior:

* **TopDownMultiClass** (legacy ``predictors.py:3808-3880``):
  ``track = tracks[class_ind]``, ``score = centroid_val``,
  ``tracking_score = class_probability``.
* **BottomUpMultiClass** (legacy ``predictors.py:2987-3010``):
  ``track = tracks[i]`` (by instance order), ``score = np.nanmean(confs)``,
  ``tracking_score = np.nanmean(class_score)``.

For both multi-class fixtures we run the legacy
(``sleap_nn.inference.predictors.Predictor``) and new
(``sleap_nn.inference.predictor.Predictor``) flows on CPU and assert:

1. the same number of instances carry a track,
2. identical track names per matched instance, and
3. ``score`` / ``tracking_score`` parity within ``1e-3``.

Instance matching is done geometrically (nearest-keypoint), mirroring the
``scratch/inf-refactor-review/parity_probe.py`` probe.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import sleap_io as sio

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"

# Both flows are exercised here intentionally; silence the legacy deprecation.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="sleap_nn.inference.predictors"
)

SCORE_TOL = 1e-3
TRACKING_SCORE_TOL = 1e-3

FIXTURES = {
    "multiclass_topdown": dict(
        models=[
            CKPT_ROOT / "minimal_instance_centroid",
            CKPT_ROOT / "minimal_instance_multiclass_centered_instance",
        ],
        source=DATA_ROOT / "centered_pair_small.mp4",
        n_frames=4,
        peak_threshold=0.03,
        max_instances=6,
        expected_tracks=2,
    ),
    "multiclass_bottomup": dict(
        models=[CKPT_ROOT / "minimal_instance_multiclass_bottomup"],
        source=DATA_ROOT / "centered_pair_small.mp4",
        n_frames=4,
        peak_threshold=0.05,
        max_instances=None,
        expected_tracks=8,
    ),
}


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _frames_map(labels: sio.Labels) -> dict[int, list[dict]]:
    """``frame_idx -> list of instance dicts`` (points, score, track, tracking)."""
    out: dict[int, list[dict]] = {}
    for lf in labels.labeled_frames:
        insts: list[dict] = []
        for inst in lf.instances:
            pts = np.asarray(inst.numpy(), dtype=float)
            score = getattr(inst, "score", None)
            track = getattr(inst, "track", None)
            track_name = getattr(track, "name", None) if track is not None else None
            tscore = getattr(inst, "tracking_score", None)
            insts.append(
                dict(
                    pts=pts,
                    score=float(score) if score is not None else np.nan,
                    track=track_name,
                    tracking_score=float(tscore) if tscore is not None else np.nan,
                )
            )
        out.setdefault(int(lf.frame_idx), []).extend(insts)
    return out


def _run_legacy(fx: dict, device: str = "cpu") -> sio.Labels:
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
    kw: dict = dict(
        peak_threshold=fx["peak_threshold"], preprocess_config=pp, device=device
    )
    if fx["max_instances"] is not None:
        kw["max_instances"] = fx["max_instances"]
    pred = LegacyPredictor.from_model_paths(
        model_paths=[str(p) for p in fx["models"]], **kw
    )
    pred._initialize_inference_model()
    pred.make_pipeline(
        inference_object=str(fx["source"]), frames=list(range(fx["n_frames"]))
    )
    return pred.predict(make_labels=True)


def _run_new(fx: dict, device: str = "cpu") -> sio.Labels:
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import VideoProvider

    predictor = Predictor.from_model_paths(
        [str(p) for p in fx["models"]],
        device=device,
        batch_size=fx["n_frames"],
        peak_threshold=fx["peak_threshold"],
        max_instances=fx["max_instances"],
    )
    video = sio.load_video(str(fx["source"]))
    skeleton = sio.Skeleton(nodes=[sio.Node(f"n{i}") for i in range(2)])
    provider = VideoProvider(
        video=video, batch_size=fx["n_frames"], frames=list(range(fx["n_frames"]))
    )
    return predictor.predict(
        provider, make_labels=True, skeleton=skeleton, videos=[video]
    )


def _match_instances(L: list[dict], N: list[dict]) -> list[tuple[dict, dict]]:
    """Greedily pair legacy/new instances by mean visible-keypoint distance."""
    pairs: list[tuple[dict, dict]] = []
    used: set[int] = set()
    for dl in L:
        lp = dl["pts"]
        best_d, best_j = 1e18, -1
        for j, dn in enumerate(N):
            if j in used:
                continue
            npp = dn["pts"]
            both = np.isfinite(lp).all(1) & np.isfinite(npp).all(1)
            d = (
                float(np.linalg.norm(lp[both] - npp[both], axis=1).mean())
                if both.any()
                else 1e9
            )
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((dl, N[best_j]))
    return pairs


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_multiclass_identity_parity(fixture: str):
    """New flow restores legacy track / score / tracking_score for ID models."""
    fx = FIXTURES[fixture]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip(f"{fixture} checkpoints/data not present")

    legacy = _run_legacy(fx)
    new = _run_new(fx)

    lm, nm = _frames_map(legacy), _frames_map(new)

    # 1) The NEW flow must assign the SAME number of tracked instances as legacy.
    # The absolute count is NOT pinned to a hardcoded number: at the deliberately
    # low peak_threshold these tiny test models sit near the detection boundary,
    # so some platforms (e.g. macOS Accelerate BLAS) detect a different count.
    # Parity is "new == legacy", whatever legacy detects on this platform.
    n_legacy_tracks = sum(
        1 for insts in lm.values() for d in insts if d["track"] is not None
    )
    n_new_tracks = sum(
        1 for insts in nm.values() for d in insts if d["track"] is not None
    )
    # If legacy detected nothing here there is no identity to compare against —
    # skip rather than vacuously pass (or assert a platform-specific count).
    if n_legacy_tracks == 0:
        pytest.skip(
            f"{fixture}: legacy detected 0 tracked instances on this platform "
            f"at peak_threshold={fx['peak_threshold']} — nothing to compare"
        )
    assert n_new_tracks == n_legacy_tracks, (
        f"new flow produced {n_new_tracks} tracked instances, "
        f"legacy produced {n_legacy_tracks}"
    )

    # New labels must register the used tracks on the Labels object.
    assert len(new.tracks) > 0, "new Labels did not register any tracks"

    # 2) + 3) Per matched instance: identical track name, score + tracking_score
    # parity within tolerance.
    matched = 0
    for f in sorted(set(lm) | set(nm)):
        for dl, dn in _match_instances(lm.get(f, []), nm.get(f, [])):
            matched += 1
            assert dl["track"] == dn["track"], (
                f"frame {f}: track name mismatch legacy={dl['track']!r} "
                f"new={dn['track']!r}"
            )
            if np.isfinite(dl["score"]) and np.isfinite(dn["score"]):
                assert abs(dl["score"] - dn["score"]) < SCORE_TOL, (
                    f"frame {f}: score drift "
                    f"{abs(dl['score'] - dn['score'])} >= {SCORE_TOL}"
                )
            if np.isfinite(dl["tracking_score"]) and np.isfinite(dn["tracking_score"]):
                assert (
                    abs(dl["tracking_score"] - dn["tracking_score"])
                    < TRACKING_SCORE_TOL
                ), (
                    f"frame {f}: tracking_score drift "
                    f"{abs(dl['tracking_score'] - dn['tracking_score'])} "
                    f">= {TRACKING_SCORE_TOL}"
                )

    assert matched > 0, "no instances matched between legacy and new flows"
