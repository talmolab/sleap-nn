"""Tests for the script that picks the docs alias of a release."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "docs_alias.py"
_spec = importlib.util.spec_from_file_location("docs_alias", _SCRIPT)
docs_alias = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docs_alias)

PUBLISHED = [
    {"version": "dev", "title": "dev", "aliases": []},
    {"version": "v0.4.0", "title": "v0.4.0", "aliases": ["latest"]},
    {"version": "v0.4.1a0", "title": "v0.4.1a0", "aliases": ["prerelease"]},
    {"version": "v0.3.5", "title": "v0.3.5", "aliases": []},
]


@pytest.mark.parametrize(
    "tag, alias",
    [
        ("v0.4.1", "latest"),  # Newer than every stable version.
        ("v0.4.0", "latest"),  # Re-publishing the newest one.
        ("v0.3.6", ""),  # An older line's patch doesn't take over.
        ("v0.5.0rc1", ""),  # Pre-releases never get "latest".
        ("v0.4.0a0", ""),  # Alphas, as this project names its pre-releases.
        ("0.4.0a0", ""),
        ("v0.5.0a1", ""),
        ("not-a-version", ""),
    ],
)
def test_docs_alias(tag, alias):
    """A stable release gets "latest" only if no newer stable one is published."""
    assert docs_alias.docs_alias(tag, PUBLISHED) == alias


def test_docs_alias_with_no_published_versions():
    """The first release gets "latest"."""
    assert docs_alias.docs_alias("v0.1.0", []) == "latest"


def test_docs_alias_cli(tmp_path):
    """CLI reads mike's versions.json and tolerates invalid JSON."""
    versions_file = tmp_path / "versions.json"
    versions_file.write_text(json.dumps(PUBLISHED))

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "v0.4.1", str(versions_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "latest"

    proc_old = subprocess.run(
        [sys.executable, str(_SCRIPT), "v0.3.6", str(versions_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc_old.stdout.strip() == ""

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("")
    proc_bad = subprocess.run(
        [sys.executable, str(_SCRIPT), "v0.4.0", str(bad_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc_bad.stdout.strip() == "latest"
