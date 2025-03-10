"""Tests for the utilities for config building and validation."""

import attr
import pytest
from typing import Optional, Text
from loguru import logger

from _pytest.logging import LogCaptureFixture

from sleap_nn.config import utils


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


def test_one_of(caplog):
    """Test of decorator."""

    @utils.oneof
    @attr.s(auto_attribs=True)
    class ExclusiveClass:
        a: Optional[Text] = None
        b: Optional[Text] = None

    c = ExclusiveClass(a="hello")

    assert c.which_oneof_attrib_name() == "a"
    assert c.which_oneof() == "hello"

    with pytest.raises(ValueError):
        c = ExclusiveClass(a="hello", b="too many values!")
    assert "Only one attribute" in caplog.text
