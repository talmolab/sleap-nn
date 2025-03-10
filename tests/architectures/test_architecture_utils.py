import pytest
from torch import nn
from loguru import logger

from _pytest.logging import LogCaptureFixture

from sleap_nn.architectures.utils import get_act_fn, get_children_layers


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


def test_get_act_fn(caplog):
    with pytest.raises(KeyError):
        get_act_fn("invalid_input")
    assert "invalid_input" in caplog.text

    assert isinstance(get_act_fn("relu"), nn.ReLU)
    assert isinstance(get_act_fn("softmax"), nn.Softmax)
    assert isinstance(get_act_fn("identity"), nn.Identity)


def test_get_children_layers():
    model = nn.Sequential(nn.Sequential(nn.Linear(5, 10)), nn.Linear(10, 5))

    layers = get_children_layers(model)

    assert len(layers) == 2
    assert layers[0].in_features == 5
    assert layers[0].out_features == 10
    assert layers[1].in_features == 10
    assert layers[1].out_features == 5
