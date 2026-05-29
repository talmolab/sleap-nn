"""Parity golden snapshots for the inference refactor (epic #508).

Each refactor PR (#509-#522) gates on byte-for-byte parity with the current
inference pipeline. This module captures golden outputs from
``Predictor.predict(make_labels=False)`` on a fixed input and stores them on
disk so later PRs can replay the same input through the new code path and
compare element-wise.

Layout under ``tests/inference/parity_golden/``::

    <checkpoint_name>.pkl    raw list-of-dicts from predict(make_labels=False)
    <checkpoint_name>.json   schema (keys + shapes + dtypes) for human review

Pickle is used because the raw dicts contain heterogeneous types (numpy arrays,
tuples-of-arrays for bottom-up PAF graphs); npz cannot represent the tuples
without flattening. JSON sidecar gives a diffable structural fingerprint.

Determinism (important):

* :func:`capture` seeds torch + numpy and runs on CPU; *within a single
  process*, calling ``capture(spec)`` on the same spec twice produces
  byte-identical output.
* Sequencing multiple specs in the same process is **not** guaranteed to
  reproduce a single-spec run, because the predictor stack maintains
  process-wide caches (sleap-io video objects, in particular) that bleed
  between specs. The original PR 0 ``regenerate_all`` ran every spec in one
  process and silently captured contaminated topdown outputs because of
  this. :func:`regenerate_all` and :func:`capture_in_subprocess` now isolate
  every spec in its own Python process to keep the on-disk goldens clean.

Re-run with ``python -m tests.utils.parity_goldens`` from the repo root after
intentional behavior changes.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

PICKLE_PROTOCOL = 4
SEED = 0
N_FRAMES = 8  # two batches at default batch_size=4
DEVICE = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "inference" / "parity_golden"
ASSETS = REPO_ROOT / "tests" / "assets"
CKPT_DIR = ASSETS / "model_ckpts"
DATA_DIR = ASSETS / "datasets"

# Preprocess config that lets each model use the values from its training_config.
NEUTRAL_PREPROCESS = OmegaConf.create(
    {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
)


@dataclass(frozen=True)
class GoldenSpec:
    """One checkpoint + input combination to capture."""

    name: str
    model_paths: Sequence[Path]
    video: Path
    peak_threshold: float = 0.2
    max_instances: int | None = None
    extra: Dict[str, Any] | None = None


def _specs() -> List[GoldenSpec]:
    return [
        GoldenSpec(
            name="single_instance",
            model_paths=[CKPT_DIR / "minimal_instance_single_instance"],
            video=DATA_DIR / "small_robot.mp4",
            peak_threshold=0.3,
        ),
        GoldenSpec(
            name="single_instance_with_metrics",
            model_paths=[CKPT_DIR / "single_instance_with_metrics"],
            video=DATA_DIR / "small_robot.mp4",
            peak_threshold=0.3,
        ),
        # Both standalone modes are supported today, but only with a
        # LabelsReader (``instances_key=True``) — neither works on a raw
        # video because the predictor needs GT instances:
        #
        # - **Standalone centered-instance** (only the centered_instance
        #   model loaded). The centered-instance model still runs; crops
        #   come from GT centroids via ``CentroidCrop(use_gt_centroids=True)``
        #   (the ``FindInstancePeaks`` model layer is used unchanged).
        #   In the new design this becomes ``CentroidLayer(use_gt_centroids=True)``
        #   composed with ``CenteredInstanceLayer`` — landing in PR 6 (#514).
        #
        # - **Standalone centroid** (only the centroid model loaded). The
        #   centroid model runs; keypoints come from GT via
        #   ``FindInstancePeaksGroundTruth`` (no second-stage model). In
        #   the new design this becomes ``CenteredInstanceLayer(use_gt_peaks=True)``
        #   — also landing in PR 6 (#514).
        #
        # Neither is captured here because doing so requires a labels file
        # and our experiment with ``minimal_instance.pkg.slp`` was
        # non-reproducible inside the full test suite — earlier specs that
        # read ``centered_pair_small.mp4`` leak state that makes a subsequent
        # ``.pkg.slp`` capture pick up the wrong video source. Tracking the
        # pollution down is a separate task. PR 6 ships per-test parity
        # coverage in isolation, so we don't lose assurance here.
        #
        # PR 14 (#522) is a *different* new capability: a saveable
        # ``.slp`` with centroid-only outputs (NaN keypoints). It will
        # bring its own golden when the feature lands.
        GoldenSpec(
            name="topdown",
            model_paths=[
                CKPT_DIR / "minimal_instance_centroid",
                CKPT_DIR / "minimal_instance_centered_instance",
            ],
            video=DATA_DIR / "centered_pair_small.mp4",
            peak_threshold=0.03,
            max_instances=6,
        ),
        GoldenSpec(
            name="bottomup",
            model_paths=[CKPT_DIR / "minimal_instance_bottomup"],
            video=DATA_DIR / "centered_pair_small.mp4",
            peak_threshold=0.05,
        ),
        GoldenSpec(
            name="multiclass_bottomup",
            model_paths=[CKPT_DIR / "minimal_instance_multiclass_bottomup"],
            video=DATA_DIR / "centered_pair_small.mp4",
            peak_threshold=0.05,
        ),
        GoldenSpec(
            name="multiclass_topdown",
            model_paths=[
                CKPT_DIR / "minimal_instance_centroid",
                CKPT_DIR / "minimal_instance_multiclass_centered_instance",
            ],
            video=DATA_DIR / "centered_pair_small.mp4",
            peak_threshold=0.03,
            max_instances=6,
        ),
    ]


def _seed() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Disable nondeterministic kernels where possible.
    torch.use_deterministic_algorithms(
        False
    )  # some sleap-nn ops aren't deterministic-marked
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_predictor(spec: GoldenSpec):
    """Build a ``Predictor`` from a spec.

    Imported lazily so this module imports cheaply without the inference stack.
    """
    from sleap_nn.inference.predictors import Predictor

    kwargs: Dict[str, Any] = dict(
        peak_threshold=spec.peak_threshold,
        preprocess_config=NEUTRAL_PREPROCESS,
        device=DEVICE,
    )
    if spec.max_instances is not None:
        kwargs["max_instances"] = spec.max_instances
    if spec.extra:
        kwargs.update(spec.extra)
    return Predictor.from_model_paths(
        [str(p) for p in spec.model_paths],
        **kwargs,
    )


def _arr_summary(value: Any) -> Any:
    """Build a JSON-friendly summary of an output value (recursive on tuples)."""
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_arr_summary(v) for v in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_arr_summary(v) for v in value]}
    return {"type": type(value).__name__, "repr": repr(value)[:80]}


def _summary(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_batches": len(batches),
        "batches": [
            {key: _arr_summary(val) for key, val in batch.items()} for batch in batches
        ],
    }


def capture(spec: GoldenSpec) -> List[Dict[str, Any]]:
    """Run inference for one spec and return the raw list-of-dicts output."""
    _seed()
    predictor = _build_predictor(spec)
    predictor.make_pipeline(str(spec.video), frames=list(range(N_FRAMES)))
    output = predictor.predict(make_labels=False)
    if not isinstance(output, list):
        raise RuntimeError(
            f"{spec.name}: expected list from predict(make_labels=False), got {type(output)}"
        )
    return output


def write_golden(spec: GoldenSpec, batches: List[Dict[str, Any]]) -> None:
    """Write a captured ``batches`` payload + JSON schema sidecar to disk."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = GOLDEN_DIR / f"{spec.name}.pkl"
    json_path = GOLDEN_DIR / f"{spec.name}.json"
    with pkl_path.open("wb") as fp:
        pickle.dump(batches, fp, protocol=PICKLE_PROTOCOL)
    with json_path.open("w") as fp:
        json.dump(_summary(batches), fp, indent=2, sort_keys=True)


def load_golden(name: str) -> List[Dict[str, Any]]:
    """Load the on-disk golden pickle for ``name`` (raises if missing)."""
    pkl_path = GOLDEN_DIR / f"{name}.pkl"
    with pkl_path.open("rb") as fp:
        return pickle.load(fp)


def capture_in_subprocess(spec: GoldenSpec) -> List[Dict[str, Any]]:
    """Run :func:`capture` in a fresh Python subprocess.

    Use this for parity testing and golden regeneration so process-wide
    sleap-io / predictor caches can't bleed between specs (see module
    docstring). The subprocess writes its pickled output to a temp file
    that we read back; we don't try to stream pickle through stdout
    because the predictor's progress bar contaminates that channel.
    """
    import os
    import subprocess
    import sys
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix=f"goldens_{spec.name}_")
    os.close(fd)
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import pickle\n"
                    "from tests.utils.parity_goldens import capture, all_specs\n"
                    f"spec = next(s for s in all_specs() if s.name == {spec.name!r})\n"
                    f"pickle.dump(capture(spec), open({tmp_path!r}, 'wb'), protocol=4)\n"
                ),
            ],
            check=True,
        )
        with open(tmp_path, "rb") as fp:
            return pickle.load(fp)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def regenerate_all(specs: Sequence[GoldenSpec] | None = None) -> None:
    """Regenerate every golden snapshot in subprocesses (one per spec).

    Subprocess isolation is required: the predictor stack caches sleap-io
    Video objects process-wide, so running multiple specs in sequence in a
    single process can leak the wrong video into a later spec's output. The
    original PR 0 sweep had this bug; this implementation does not.
    """
    for spec in specs or _specs():
        print(f"  capturing {spec.name} (subprocess) ...", flush=True)
        batches = capture_in_subprocess(spec)
        write_golden(spec, batches)
        print(f"    -> {len(batches)} batch(es), keys: {sorted(batches[0])}")


def all_specs() -> List[GoldenSpec]:
    """Return the canonical list of ``GoldenSpec`` entries."""
    return _specs()


if __name__ == "__main__":
    print(f"Regenerating goldens in {GOLDEN_DIR.relative_to(REPO_ROOT)}/")
    regenerate_all()
    print("done.")
