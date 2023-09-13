import pytest
from torch import nn

from sleap_nn.architectures.utils import get_act_fn, get_children_layers


def test_get_act_fn():
    with pytest.raises(KeyError):
        get_act_fn("invalid_input")


def test_get_children_layers():
    model = nn.Sequential(nn.Sequential(nn.Linear(5, 10)), nn.Linear(10, 5))

    layers = get_children_layers(model)

    assert len(layers) == 2
    assert layers[0].in_features == 5
    assert layers[0].out_features == 10
    assert layers[1].in_features == 10
    assert layers[1].out_features == 5
