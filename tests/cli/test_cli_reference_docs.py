"""Keep ``docs/reference/cli.md`` in step with the inference/eval/export CLIs.

Every visible option of ``sleap-nn predict``, ``eval`` and ``export`` must be
mentioned in the CLI reference, so a new flag cannot ship undocumented.
"""

import re
from pathlib import Path

import click
import pytest

from sleap_nn.cli import cli

CLI_DOC = Path(__file__).resolve().parents[2] / "docs" / "reference" / "cli.md"


def _undocumented(command_name: str) -> list:
    command = cli.get_command(click.Context(cli), command_name)
    doc = CLI_DOC.read_text()
    missing = []
    for param in command.params:
        if not isinstance(param, click.Option) or param.hidden:
            continue
        spellings = [o for o in param.opts + param.secondary_opts if o.startswith("--")]
        if not any(re.search(re.escape(o) + r"(?![\w-])", doc) for o in spellings):
            missing.append(spellings[0])
    return missing


@pytest.mark.parametrize("command_name", ["predict", "eval", "export"])
def test_cli_reference_documents_every_option(command_name):
    """Every visible option of the command appears in the CLI reference."""
    assert (
        _undocumented(command_name) == []
    ), f"Options of `sleap-nn {command_name}` missing from docs/reference/cli.md"
