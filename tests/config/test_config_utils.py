"""Tests for the utilities for config building and validation."""

import attr
import pytest
from typing import Optional, Text

from sleap_nn.config import utils


def test_one_of():
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
