"""Spawn the ``sleap-nn`` CLI from a test WITHOUT re-syncing the project environment.

The CLI tests used to shell out to::

    uv run --frozen --no-group gpu --extra torch-cpu sleap-nn ...

``--frozen`` only pins the lockfile; ``uv run`` still **syncs the project's ``.venv``**
to the requested groups/extras before running. On CI that is a no-op (the job already
synced with exactly those flags), but on a developer machine whose default ``gpu``
group installed ``torch+cu130`` it silently uninstalled torch/torchvision and put the
``+cpu`` wheels in their place -- mid-suite, for every other process sharing the venv.
It is the "venv keeps flipping to CPU torch" hazard that cost several GPU sessions.

Run the CLI in *this* interpreter's environment instead: ``sleap_nn.cli`` has a
``__main__`` guard, so ``python -m sleap_nn.cli`` is the console script without ``uv``
(and without depending on ``uv`` being installed at all).
"""

from __future__ import annotations

import os
import sys


def sleap_nn_cli(*args: str) -> list[str]:
    """Command prefix running ``sleap-nn`` on ``sys.executable``: ``[python, -m, sleap_nn.cli, *args]``."""
    return [sys.executable, "-m", "sleap_nn.cli", *args]


def cpu_only_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a CLI subprocess that must stay off the GPU.

    ``--extra torch-cpu`` used to guarantee a CPU-only torch in the child; hiding the
    devices keeps that intent on a GPU box without touching the installed wheel.
    """
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env
