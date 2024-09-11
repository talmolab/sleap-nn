from omegaconf import OmegaConf
import torch

import sleap_io as sio
from sleap_nn.inference.utils import get_skeleton_from_config, interp1d


def test_get_skeleton_from_config(minimal_instance, minimal_instance_ckpt):
    """Test function for get_skeleton_from_config function."""
    training_config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    skeleton_config = training_config.data_config.skeletons
    skeletons = get_skeleton_from_config(skeleton_config)
    labels = sio.load_slp(f"{minimal_instance}")
    assert skeletons[0] == labels.skeletons[0]


def test_interp1d():
    """Test function for `interp()` function."""
    x = torch.linspace(0, 10, steps=10)
    y = torch.randint(10, 30, (10,), dtype=torch.float64)
    xq = torch.linspace(0, 10, steps=20)
    yq = interp1d(x, y, xq)
    assert yq.shape == (20,)

    x = torch.linspace(0, 10, steps=10).repeat(5, 1)
    y = torch.randint(10, 30, (5, 10), dtype=torch.float64)
    xq = torch.linspace(0, 10, steps=20).repeat(5, 1)
    yq = interp1d(x, y, xq)
    assert yq.shape == (5, 20)
