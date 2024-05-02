"""Test LabelsReader and VideoReader modules."""

import torch

from sleap_nn.data.providers import LabelsReader, VideoReader
import sleap_io as sio
import numpy as np
import sleap_io as sio
import pytest
from queue import Queue

def test_labelsreader_provider(minimal_instance):
    """Test LabelsReader class."""
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

    # check max_width, max_height and is_rgb
    l = LabelsReader.from_filename(
        minimal_instance, max_height=500, max_width=500, is_rgb=True
    )
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 3, 500, 500])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    l = LabelsReader.from_filename(
        minimal_instance, max_height=100, max_width=500, is_rgb=True
    )
    with pytest.raises(
        Exception,
        match=f"Max height {100} should be greater than the current image height: {384}",
    ):
        sample = next(iter(l))

    l = LabelsReader.from_filename(
        minimal_instance, max_height=500, max_width=100, is_rgb=True
    )
    with pytest.raises(
        Exception,
        match=f"Max width {100} should be greater than the current image width: {384}",
    ):
        sample = next(iter(l))

def test_videoreader_provider(centered_instance_video):
    """Test VideoReader class."""
    # is_rgb set to False
    video = sio.load_video(centered_instance_video)
    queue = Queue(maxsize=4)
    reader = VideoReader(video=video, frame_buffer=queue, is_rgb=False, 
                         start_idx=0, end_idx=4)
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

    # is_rgb set to True
    queue = Queue(maxsize=4)
    reader = VideoReader(video=video, frame_buffer=queue, is_rgb=True, 
                         start_idx=0, end_idx=4)
    reader.start()
    batch_size = 4
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame[0] is None:
                break
            data.append(frame)
        if data:
            assert len(data) == batch_size
            assert data[0][0].shape == (1, 3, 384, 384)
    except:
        raise
    finally:
        reader.join()

    # check graceful stop (video has 1100 frames)
    queue = Queue(maxsize=4)
    reader = VideoReader.from_filename(filename=centered_instance_video, 
                                       frame_buffer=queue, is_rgb=True, 
                                       start_idx=1099, end_idx=1104)
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
        assert data[0][0].shape == (1, 3, 384, 384)
    except:
        raise
    finally:
        reader.join()

