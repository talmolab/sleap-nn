import torch
import numpy as np

import sleap_io as sio
from sleap_nn.data.identity import (
    make_class_vectors,
    make_class_maps,
    generate_class_maps,
)
from sleap_nn.data.confidence_maps import make_grid_vectors, make_confmaps


def test_make_class_vectors():
    """Test make class vectors function."""
    encoded_vectors = make_class_vectors(torch.Tensor([0, 2, 1, -1]), 3)
    np.testing.assert_array_equal(
        encoded_vectors, [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]]
    )


def test_make_class_maps():
    """Test make class maps function."""
    xv, yv = make_grid_vectors(32, 32, output_stride=1)
    pts = torch.Tensor([[[4, 6], [18, 24]]]).to(torch.float32)
    cms = make_confmaps(pts, xv, yv, sigma=2)
    class_maps = make_class_maps(
        cms, class_inds=torch.Tensor([1, 0]), n_classes=2, threshold=0.2
    )
    indices = torch.tensor([[6, 4], [24, 18]])
    gathered = class_maps[:, :, indices[:, 0], indices[:, 1]]

    # Now assert equality
    np.testing.assert_array_equal(gathered.cpu().numpy(), np.array([[[0, 1], [1, 0]]]))


def test_generate_class_maps(minimal_instance):
    """Test generate class maps function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    instances = []
    for inst in lf:
        if not inst.is_empty:
            instances.append(inst.numpy())
    instances = np.stack(instances, axis=0)
    class_maps = generate_class_maps(
        instances=torch.from_numpy(instances).unsqueeze(0),
        img_hw=lf.image.shape[:2],
        num_instances=2,
        class_inds=torch.Tensor([0, 1]),
        num_tracks=2,
        output_stride=2,
    )
    assert class_maps.shape == (1, 2, 192, 192)
