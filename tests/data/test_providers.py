import torch

from sleap_nn.data.providers import LabelsReader, VideoReader
from queue import Queue
import sleap_io as sio
import numpy as np
import sleap_io as sio
import pytest


def test_providers(minimal_instance):
    """Test LabelsReader module."""
    l = LabelsReader.from_filename(minimal_instance)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    labels = sio.load_slp(minimal_instance)
    org_image = labels[0].image
    image = image.squeeze().squeeze().unsqueeze(dim=-1)
    assert np.all(org_image == image.numpy())
<<<<<<< Updated upstream
    assert l.max_height_and_width == (384, 384)


def test_videoreader_provider(centered_instance_video):
    """Test VideoReader class."""
    video = sio.load_video(centered_instance_video)
    queue = Queue(maxsize=4)
    reader = VideoReader(video=video, frame_buffer=queue, start_idx=None, end_idx=4)
    assert reader.max_height_and_width == (384, 384)
    reader.start()
    batch_size = 4
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame[0] is None:
                break
            data.append(frame)
        assert len(data) == batch_size
        assert data[0][0].shape == (1, 1, 384, 384)
    except:
        raise
    finally:
        reader.join()
    assert reader.total_len() == 4

    # check graceful stop (video has 1100 frames)
    queue = Queue(maxsize=4)
    reader = VideoReader.from_filename(
        filename=centered_instance_video,
        frame_buffer=queue,
        start_idx=1099,
        end_idx=1104,
    )
    reader.start()
    batch_size = 4
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame[0] is None:
                break
            data.append(frame)
        assert len(data) == 1
        assert data[0][0].shape == (1, 1, 384, 384)
    except:
        raise
    finally:
        reader.join()
=======
>>>>>>> Stashed changes
