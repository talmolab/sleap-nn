"""Smoke tests for the parity golden snapshots used by the inference refactor.

Goldens themselves are produced by ``tests/utils/parity_goldens.py``. These
tests guarantee that:

1. Every spec listed in ``parity_goldens.all_specs()`` has a checked-in
   ``.pkl`` + ``.json`` pair (so a future refactor PR can't be merged with a
   missing baseline).
2. Each golden round-trips through pickle without surprise.
3. The ``.json`` schema matches the ``.pkl`` payload (catches a regenerated
   ``.pkl`` that wasn't accompanied by a regenerated schema).
4. Re-running the capture in a fresh process produces output that compares
   element-wise to the on-disk golden — i.e., the capture is reproducible.

The fourth check is heavy (it actually runs every predictor) so it is gated
behind ``RUN_GOLDEN_REGEN_CHECK=1`` in the environment. CI runs the cheap
checks every push and the expensive check on a slower lane.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.utils import parity_goldens as pg

CHEAP_SPECS = pg.all_specs()


def _flatten(value: Any, prefix: str = ""):
    """Yield ``(name, ndarray)`` pairs for every ndarray in a nested structure."""
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _flatten(v, f"{prefix}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)


@pytest.mark.parametrize("spec", CHEAP_SPECS, ids=lambda s: s.name)
def test_golden_files_present(spec: pg.GoldenSpec) -> None:
    """Every spec has both a ``.pkl`` payload and a ``.json`` schema."""
    pkl = pg.GOLDEN_DIR / f"{spec.name}.pkl"
    js = pg.GOLDEN_DIR / f"{spec.name}.json"
    assert pkl.exists(), f"missing golden pickle: {pkl}"
    assert js.exists(), f"missing golden schema:  {js}"


@pytest.mark.parametrize("spec", CHEAP_SPECS, ids=lambda s: s.name)
def test_golden_loads(spec: pg.GoldenSpec) -> None:
    """The pickled payload deserializes to a non-empty list of dicts."""
    batches = pg.load_golden(spec.name)
    assert isinstance(batches, list) and len(batches) > 0
    assert all(isinstance(b, dict) for b in batches)


@pytest.mark.parametrize("spec", CHEAP_SPECS, ids=lambda s: s.name)
def test_golden_schema_matches_payload(spec: pg.GoldenSpec) -> None:
    """The .json schema is a structural fingerprint of the .pkl. They must agree."""
    import json

    payload = pg.load_golden(spec.name)
    schema = json.loads((pg.GOLDEN_DIR / f"{spec.name}.json").read_text())
    fresh_summary = pg._summary(payload)
    assert schema == fresh_summary, (
        f"schema/payload drift for {spec.name}; regenerate goldens via "
        f"`python -m tests.utils.parity_goldens`"
    )


@pytest.mark.parametrize("spec", CHEAP_SPECS, ids=lambda s: s.name)
def test_golden_arrays_finite_or_documented_nan(spec: pg.GoldenSpec) -> None:
    """Every golden array should contain real numbers or documented NaNs.

    NaNs are legal as 'no peak' sentinels in pred_instance_peaks /
    pred_peak_values; everywhere else, NaN is a bug. This catches goldens that
    accidentally captured an exception state.
    """
    NAN_OK = {"pred_instance_peaks", "pred_peak_values", "pred_centroids", "centroid"}
    batches = pg.load_golden(spec.name)
    for b_i, batch in enumerate(batches):
        for key, val in batch.items():
            for arr_name, arr in _flatten(val, key):
                if arr.dtype.kind not in ("f", "c"):
                    continue
                if not np.isfinite(arr).all():
                    base = arr_name.split("[", 1)[0].split(".", 1)[0]
                    assert (
                        base in NAN_OK
                    ), f"{spec.name} batch{b_i} {arr_name} has non-finite values"


@pytest.mark.skipif(
    os.environ.get("RUN_GOLDEN_REGEN_CHECK") != "1",
    reason="set RUN_GOLDEN_REGEN_CHECK=1 to run the slow regeneration check",
)
@pytest.mark.parametrize("spec", CHEAP_SPECS, ids=lambda s: s.name)
def test_golden_is_reproducible(spec: pg.GoldenSpec) -> None:
    """Re-capture in a fresh subprocess and compare to the on-disk golden.

    Subprocess isolation is required because the predictor stack caches
    sleap-io ``Video`` objects process-wide, so running multiple specs in
    sequence in the same process can leak the wrong video source into a
    later spec's output. See the module docstring of
    :mod:`tests.utils.parity_goldens` for context.
    """
    expected = pg.load_golden(spec.name)
    actual = pg.capture_in_subprocess(spec)

    assert len(actual) == len(
        expected
    ), f"{spec.name}: batch count drift {len(actual)} vs golden {len(expected)}"
    for b_i, (got, want) in enumerate(zip(actual, expected)):
        assert sorted(got) == sorted(
            want
        ), f"{spec.name} batch{b_i} keys differ: {sorted(got)} vs {sorted(want)}"
        for key in want:
            for (a_name, a_val), (e_name, e_val) in zip(
                _flatten(got[key], key), _flatten(want[key], key)
            ):
                assert a_name == e_name
                # Goldens are captured in subprocess isolation alongside this
                # check, so they're bit-exact on the same machine. Subsequent
                # refactor PRs (#513 onward) will relax to a documented
                # tolerance budget at their own per-PR test sites.
                if a_val.dtype.kind == "f":
                    np.testing.assert_allclose(
                        a_val,
                        e_val,
                        equal_nan=True,
                        rtol=0,
                        atol=0,
                        err_msg=f"{spec.name} batch{b_i} {a_name} drifted",
                    )
                else:
                    np.testing.assert_array_equal(a_val, e_val)
