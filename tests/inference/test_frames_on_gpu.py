"""``frames_on_gpu``: top-down frames move to the GPU before stage 1."""

import numpy as np
import pytest
import torch

from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import _enable_frames_on_gpu


class _Stage1:
    """Records the device of the frames it is given; finds nothing."""

    use_gt_centroids = False

    def __init__(self):
        self.devices = []

    @staticmethod
    def _to_4d_tensor(image):
        t = torch.as_tensor(image)
        return t.permute(0, 3, 1, 2) if t.shape[-1] <= 4 else t

    def predict(self, image):
        self.devices.append(self._to_4d_tensor(image).device.type)
        return Outputs()


def _layer(on):
    layer = TopDownLayer(centroid_layer=_Stage1(), centered_instance_layer=None, crop_size=(8, 8))
    layer.frames_on_gpu = on
    return layer


@pytest.mark.parametrize("on", [False, True])
def test_frames_move_only_when_on(monkeypatch, on):
    layer = _layer(on)
    # "meta" stands in for the CUDA device so the test runs anywhere.
    monkeypatch.setattr(TopDownLayer, "_frames_device", lambda self: torch.device("meta"))
    layer.predict(np.zeros((2, 16, 16, 1), dtype=np.uint8))
    assert layer.centroid_layer.devices == (["meta"] if on else ["cpu"])


def test_no_device_leaves_frames(monkeypatch):
    layer = _layer(True)
    monkeypatch.setattr(TopDownLayer, "_frames_device", lambda self: None)
    layer.predict(np.zeros((1, 16, 16, 1), dtype=np.uint8))
    assert layer.centroid_layer.devices == ["cpu"]


def test_default_off():
    assert TopDownLayer.frames_on_gpu is False


def test_needs_top_down():
    with pytest.raises(ValueError, match="top-down"):
        _enable_frames_on_gpu(object())
    layer = _layer(False)
    _enable_frames_on_gpu(layer)
    assert layer.frames_on_gpu


def test_cli_flag():
    from sleap_nn.cli import cli

    opts = {o for p in cli.commands["predict"].params for o in p.opts}
    assert "--frames_on_gpu" in opts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_frames_device_finds_a_cuda_backend():
    from sleap_nn.inference.layers.backends.torch_backend import TorchBackend

    layer = _layer(True)
    backend = TorchBackend.__new__(TorchBackend)
    backend.device = "cuda"
    layer.centroid_layer.backend = backend
    assert layer._frames_device().type == "cuda"
