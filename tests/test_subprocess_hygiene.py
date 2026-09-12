"""No test may spawn ``uv run``: it re-syncs the shared project ``.venv`` as a side effect.

See ``tests/utils/cli.py`` for the history. Tests that need the ``sleap-nn`` CLI in a
subprocess use ``sleap_nn_cli()`` there, which runs it on ``sys.executable``.
"""

import re
from pathlib import Path

TESTS_DIR = Path(__file__).parent
UV_ARGV = re.compile(r"""\[\s*["']uv["']\s*,""", re.S)


def test_no_test_spawns_uv():
    """Every test file is free of a subprocess argv that starts with ``uv``."""
    offenders = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for m in UV_ARGV.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(TESTS_DIR)}:{line}")
    assert not offenders, (
        "These tests build a subprocess argv starting with 'uv'. `uv run` syncs the "
        "project venv before running (it swapped torch+cu130 for torch+cpu under a "
        "running suite); spawn the CLI with tests.utils.cli.sleap_nn_cli() instead:\n  "
        + "\n  ".join(offenders)
    )
