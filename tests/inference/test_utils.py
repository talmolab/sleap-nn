from omegaconf import OmegaConf
import sleap_io as sio
from sleap_nn.inference.utils import get_skeleton_from_config


def test_get_skeleton_from_config(minimal_instance, minimal_instance_ckpt):
    """Test function for get_skeleton_from_config function."""
    training_config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    skeleton_config = training_config.data_config.skeletons
    skeletons = get_skeleton_from_config(skeleton_config)
    labels = sio.load_slp(f"{minimal_instance}")
    assert skeletons[0] == labels.skeletons[0]
