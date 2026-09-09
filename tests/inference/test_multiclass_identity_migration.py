"""Multi-class predicted identities migrate onto ``sio.Identity`` (name-keyed).

A multi-class model (``multi_class_topdown`` via ``ClassVectorsHead``;
``multi_class_bottomup`` via ``ClassMapsHead``) **additively** assigns a canonical
``sio.Identity`` + ``identity_score`` to each predicted instance, alongside the
existing ``sio.Track`` + ``tracking_score`` it already emits, and registers
``labels.identities``.

The simplified sleap-io ``Identity`` (name + metadata, sleap-io #535) matches by
NAME across files and retrains, so the class NAME is the canonical identity key —
there is no per-class uuid bridge. These tests assert, on the new
(``sleap_nn.inference.predictor.Predictor``) pipeline:

1. each tracked ``PredictedInstance`` carries an ``identity`` (a ``sio.Identity``)
   whose ``name`` mirrors the class/track name, with ``identity_score ==
   tracking_score`` (the class prob),
2. the legacy ``track`` / ``tracking_score`` are STILL set (additive parity),
3. ``labels.identities`` is registered with the exact same canonical objects,
4. identity names are stable across runs (the class name is the deterministic key),
5. ``sio.save_slp`` -> ``sio.load_slp`` preserves identity (by name) +
   ``identity_score``,
6. default ``class_output='track'`` emits only the ``Track`` (no fabricated Identity).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import sleap_io as sio

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"

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
        # The model dir whose training_config.yaml carries the class head.
        class_head_dir="minimal_instance_multiclass_centered_instance",
        head_type="multi_class_topdown",
        sub_key="class_vectors",
    ),
    "multiclass_bottomup": dict(
        models=[CKPT_ROOT / "minimal_instance_multiclass_bottomup"],
        source=DATA_ROOT / "centered_pair_small.mp4",
        n_frames=4,
        peak_threshold=0.05,
        max_instances=None,
        class_head_dir="minimal_instance_multiclass_bottomup",
        head_type="multi_class_bottomup",
        sub_key="class_maps",
    ),
}


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _run_new(model_paths, source, n_frames, peak_threshold, max_instances):
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import VideoProvider

    predictor = Predictor.from_model_paths(
        [str(p) for p in model_paths],
        device="cpu",
        batch_size=n_frames,
        peak_threshold=peak_threshold,
        max_instances=max_instances,
    )
    video = sio.load_video(str(source))
    skeleton = sio.Skeleton(nodes=[sio.Node(f"n{i}") for i in range(2)])
    provider = VideoProvider(
        video=video, batch_size=n_frames, frames=list(range(n_frames))
    )
    return predictor.predict(
        provider, make_labels=True, skeleton=skeleton, videos=[video]
    )


def _tracked_instances(labels: sio.Labels):
    return [
        inst
        for lf in labels.labeled_frames
        for inst in lf.instances
        if getattr(inst, "track", None) is not None
    ]


def _prep_models(tmp_path, fx, *, class_output="identity"):
    """Copy fixture model dirs to tmp and set the class head's ``class_output``.

    The shipped fixtures have no ``class_output`` (so default to ``"track"`` =
    no Identity). Identity-emission tests opt in by writing ``class_output`` into
    the class head's ``training_config.yaml``. No per-class uuid is frozen — the
    class name is the canonical identity key (sleap-io #535).
    """
    new_models = []
    for src_dir in fx["models"]:
        dst = tmp_path / src_dir.name
        shutil.copytree(src_dir, dst)
        new_models.append(dst)
        if src_dir.name == fx["class_head_dir"]:
            cfg_path = dst / "training_config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            sub = cfg["model_config"]["head_configs"][fx["head_type"]][fx["sub_key"]]
            sub["class_output"] = class_output
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return new_models


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_identity_assigned_additively(fixture: str, tmp_path):
    """With class_output='identity': Identity + score assigned alongside the track."""
    fx = FIXTURES[fixture]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip(f"{fixture} checkpoints/data not present")

    labels = _run_new(
        _prep_models(tmp_path, fx, class_output="identity"),
        fx["source"],
        fx["n_frames"],
        fx["peak_threshold"],
        fx["max_instances"],
    )

    tracked = _tracked_instances(labels)
    if not tracked:
        pytest.skip(f"{fixture}: model detected 0 tracked instances on this platform")

    # labels.identities registered.
    assert len(labels.identities) > 0, "Labels.identities not registered"
    registered_ids = {id(i) for i in labels.identities}
    registered_names = {i.name for i in labels.identities}

    for inst in tracked:
        # (2) legacy track / tracking_score STILL set.
        assert inst.track is not None
        assert inst.tracking_score is not None
        # (1) identity additively assigned.
        assert inst.identity is not None, "predicted instance has no identity"
        assert isinstance(inst.identity, sio.Identity)
        # The class NAME is the canonical identity key (no uuid) and mirrors the track.
        assert inst.identity.name == inst.track.name
        # identity_score == tracking_score (the class probability).
        assert inst.identity_score == pytest.approx(inst.tracking_score, abs=1e-9)
        # (3) the assigned identity is the EXACT registered canonical object
        # (Identity compares by object identity -> the saver's registration
        # check only passes for the same object).
        assert id(inst.identity) in registered_ids
        assert inst.identity.name in registered_names


def test_identity_names_stable_across_runs(tmp_path):
    """Two runs of the same model emit identities with the SAME names.

    The class name is the deterministic cross-run/cross-file identity key (the role
    the frozen uuid used to serve), so predictions from separate runs join by name.
    """
    fx = FIXTURES["multiclass_topdown"]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip("multiclass_topdown checkpoints/data not present")

    # Re-prep into a distinct dir per run so the two runs are independent copies.
    def _run(subdir):
        labels = _run_new(
            _prep_models(tmp_path / subdir, fx, class_output="identity"),
            fx["source"],
            fx["n_frames"],
            fx["peak_threshold"],
            fx["max_instances"],
        )
        return sorted({i.name for i in labels.identities})

    names1 = _run("a")
    names2 = _run("b")
    if not names1 or not names2:
        pytest.skip("model detected 0 tracked instances on this platform")
    assert names1 == names2


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_default_track_emits_no_identity(fixture: str):
    """Default class_output='track': only the Track is emitted, NO Identity.

    A multi_class model whose classes are types/roles (e.g. male/female) must not
    fabricate a global Identity. The shipped fixtures carry no ``class_output`` ->
    default ``"track"`` -> no identities.
    """
    fx = FIXTURES[fixture]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip(f"{fixture} checkpoints/data not present")

    labels = _run_new(
        fx["models"],
        fx["source"],
        fx["n_frames"],
        fx["peak_threshold"],
        fx["max_instances"],
    )
    tracked = _tracked_instances(labels)
    if not tracked:
        pytest.skip(f"{fixture}: model detected 0 tracked instances on this platform")

    # No identities anywhere; tracks (the classification-tracker output) preserved.
    assert not (getattr(labels, "identities", None) or [])
    for inst in tracked:
        assert inst.identity is None
        assert inst.identity_score is None
        assert inst.track is not None


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_identity_round_trip(fixture: str, tmp_path):
    """save_slp -> load_slp preserves identity (by name) + identity_score."""
    fx = FIXTURES[fixture]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip(f"{fixture} checkpoints/data not present")

    labels = _run_new(
        _prep_models(tmp_path, fx, class_output="identity"),
        fx["source"],
        fx["n_frames"],
        fx["peak_threshold"],
        fx["max_instances"],
    )
    tracked = _tracked_instances(labels)
    if not tracked:
        pytest.skip(f"{fixture}: model detected 0 tracked instances on this platform")

    before = [
        (inst.identity.name, inst.identity_score)
        for inst in tracked
        if inst.identity is not None
    ]
    assert before, "no identities to round-trip"

    out = tmp_path / "out.slp"
    sio.save_slp(labels, str(out))
    reloaded = sio.load_slp(str(out))
    after = [
        (inst.identity.name, inst.identity_score)
        for inst in _tracked_instances(reloaded)
        if inst.identity is not None
    ]
    assert after == before, f"identity round-trip mismatch: {before} != {after}"
    # Reloaded catalog carries the same identity names.
    assert {i.name for i in reloaded.identities} == {b[0] for b in before}
