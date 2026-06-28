"""A2: multi-class predicted identities migrate onto ``sio.Identity``.

A multi-class model (``multi_class_topdown`` via ``ClassVectorsHead``;
``multi_class_bottomup`` via ``ClassMapsHead``) now **additively** assigns a
canonical ``sio.Identity`` + ``identity_score`` to each predicted instance,
alongside the existing ``sio.Track`` + ``tracking_score`` it already emits, and
registers ``labels.identities``. The per-class uuid is frozen at train time
(``class_uuids`` in the head config) so re-runs are stable and GT/predicted can
share the canonical id.

These tests assert, on the new (``sleap_nn.inference.predictor.Predictor``)
pipeline:

1. each tracked ``PredictedInstance`` carries an ``identity`` (a ``sio.Identity``)
   with a 32-hex uuid and ``identity_score == tracking_score`` (the class prob),
2. the legacy ``track`` / ``tracking_score`` are STILL set (additive parity),
3. ``labels.identities`` is registered with the exact same canonical objects,
4. a checkpoint whose ``training_config.yaml`` carries ``class_uuids`` re-emits
   those frozen uuids,
5. a legacy checkpoint with NO ``class_uuids`` still works (mints, no crash),
6. ``sio.save_slp`` -> ``sio.load_slp`` preserves identity (by uuid) +
   ``identity_score``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

import sleap_io as sio

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"

HEX32 = re.compile(r"^[0-9a-f]{32}$")

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


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_identity_assigned_additively(fixture: str):
    """Identity + identity_score assigned alongside the existing track."""
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

    # labels.identities registered.
    assert len(labels.identities) > 0, "Labels.identities not registered"
    registered_ids = {id(i) for i in labels.identities}
    registered_uuids = {i.uuid for i in labels.identities}

    for inst in tracked:
        # (2) legacy track / tracking_score STILL set.
        assert inst.track is not None
        assert inst.tracking_score is not None
        # (1) identity additively assigned.
        assert inst.identity is not None, "predicted instance has no identity"
        assert isinstance(inst.identity, sio.Identity)
        assert HEX32.match(inst.identity.uuid), f"bad uuid {inst.identity.uuid!r}"
        # identity_score == tracking_score (the class probability).
        assert inst.identity_score == pytest.approx(inst.tracking_score, abs=1e-9)
        # identity name mirrors the track name (same class).
        assert inst.identity.name == inst.track.name
        # (3) the assigned identity is the EXACT registered canonical object
        # (Identity compares by object identity -> the saver's registration
        # check only passes for the same object).
        assert id(inst.identity) in registered_ids
        assert inst.identity.uuid in registered_uuids


def test_frozen_uuid_reemitted(tmp_path):
    """A checkpoint carrying ``class_uuids`` re-emits those frozen uuids."""
    fx = FIXTURES["multiclass_topdown"]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip("multiclass_topdown checkpoints/data not present")

    # Copy both model dirs into tmp and inject class_uuids into the class head's
    # training_config.yaml (mirrors what train-time minting freezes).
    frozen = {"female": "a" * 32, "male": "b" * 32}
    new_models = []
    for src_dir in fx["models"]:
        dst = tmp_path / src_dir.name
        shutil.copytree(src_dir, dst)
        new_models.append(dst)
        if src_dir.name == fx["class_head_dir"]:
            cfg_path = dst / "training_config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            sub = cfg["model_config"]["head_configs"][fx["head_type"]][fx["sub_key"]]
            classes = list(sub["classes"])
            sub["class_uuids"] = [frozen[c] for c in classes]
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    labels = _run_new(
        new_models,
        fx["source"],
        fx["n_frames"],
        fx["peak_threshold"],
        fx["max_instances"],
    )
    tracked = _tracked_instances(labels)
    if not tracked:
        pytest.skip("model detected 0 tracked instances on this platform")

    for inst in tracked:
        assert inst.identity is not None
        assert inst.identity.uuid == frozen[inst.identity.name], (
            f"class {inst.identity.name!r} got uuid {inst.identity.uuid} != "
            f"frozen {frozen[inst.identity.name]}"
        )
    # Registered identities also carry the frozen uuids.
    for ident in labels.identities:
        assert ident.uuid == frozen[ident.name]


def test_legacy_checkpoint_without_class_uuids(tmp_path):
    """A checkpoint with NO class_uuids still works: mints/falls back, no crash."""
    fx = FIXTURES["multiclass_topdown"]
    if not _have(*fx["models"], fx["source"]):
        pytest.skip("multiclass_topdown checkpoints/data not present")

    # Sanity-check the asset truly lacks class_uuids (legacy shape).
    cfg_path = fx["models"][1] / "training_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    sub = cfg["model_config"]["head_configs"][fx["head_type"]][fx["sub_key"]]
    assert sub.get("class_uuids") is None

    labels = _run_new(
        fx["models"],
        fx["source"],
        fx["n_frames"],
        fx["peak_threshold"],
        fx["max_instances"],
    )
    tracked = _tracked_instances(labels)
    if not tracked:
        pytest.skip("model detected 0 tracked instances on this platform")
    for inst in tracked:
        # Minted fallback uuid: valid 32-hex, identity present, no crash.
        assert inst.identity is not None
        assert HEX32.match(inst.identity.uuid)


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_identity_round_trip(fixture: str, tmp_path):
    """save_slp -> load_slp preserves identity (by uuid) + identity_score."""
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

    before = [
        (inst.identity.uuid, inst.identity.name, inst.identity_score)
        for inst in tracked
        if inst.identity is not None
    ]
    assert before, "no identities to round-trip"

    out = tmp_path / "out.slp"
    sio.save_slp(labels, str(out))
    reloaded = sio.load_slp(str(out))
    after = [
        (inst.identity.uuid, inst.identity.name, inst.identity_score)
        for inst in _tracked_instances(reloaded)
        if inst.identity is not None
    ]
    assert after == before, f"identity round-trip mismatch: {before} != {after}"
    # Reloaded catalog carries the same uuids.
    assert {i.uuid for i in reloaded.identities} == {b[0] for b in before}
