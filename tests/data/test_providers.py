import torch

from sleap_nn.data.providers import (
    LabelsReaderDP,
    LabelsReader,
    VideoReader,
    process_lf,
)
from queue import Queue
import sleap_io as sio
import numpy as np
import sleap_io as sio
import pytest


def test_providers(minimal_instance):
    """Test LabelsReaderDP module."""
    l = LabelsReaderDP.from_filename(minimal_instance)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    labels = sio.load_slp(minimal_instance)
    org_image = labels[0].image
    image = image.squeeze().squeeze().unsqueeze(dim=-1)
    assert np.all(org_image == image.numpy())
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
            if frame["image"] is None:
                break
            data.append(frame)
        assert len(data) == batch_size
        assert data[0]["image"].shape == (1, 1, 384, 384)
    except:
        raise
    finally:
        reader.join()
    assert reader.total_len() == 4

    # check graceful stop (video has 1100 frames)
    reader = VideoReader.from_filename(
        filename=centered_instance_video,
        queue_maxsize=4,
        start_idx=1099,
        end_idx=1104,
    )
    reader.start()
    batch_size = 4
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame["image"] is None:
                break
            data.append(frame)
        assert len(data) == 1
        assert data[0]["image"].shape == (1, 1, 384, 384)
    except:
        raise
    finally:
        reader.join()

    # end not specified
    queue = Queue(maxsize=4)
    reader = VideoReader(video=video, frame_buffer=queue, start_idx=1094, end_idx=None)
    assert reader.max_height_and_width == (384, 384)
    reader.start()
    batch_size = 4
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame["image"] is None:
                break
            data.append(frame)
        assert len(data) == batch_size
        assert data[0]["image"].shape == (1, 1, 384, 384)
    except:
        raise
    finally:
        reader.join()
    assert reader.total_len() == 6


def test_labelsreader_provider(minimal_instance):
    """Test LabelsReader class."""
    labels = sio.load_slp(minimal_instance)
    queue = Queue(maxsize=4)
    reader = LabelsReader(labels=labels, frame_buffer=queue, instances_key=False)
    assert reader.max_height_and_width == (384, 384)
    reader.start()
    batch_size = 1
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame["image"] is None:
                break
            data.append(frame)
        assert len(data) == batch_size
        assert data[0]["image"].shape == (1, 1, 384, 384)
        assert "instances" not in data[0]
    except:
        raise
    finally:
        reader.join()
    assert reader.total_len() == 1

    # with instances key
    reader = LabelsReader.from_filename(
        minimal_instance, queue_maxsize=4, instances_key=True
    )
    assert reader.max_height_and_width == (384, 384)
    reader.start()
    batch_size = 1
    try:
        data = []
        for i in range(batch_size):
            frame = reader.frame_buffer.get()
            if frame["image"] is None:
                break
            data.append(frame)
        assert len(data) == batch_size
        assert data[0]["image"].shape == (1, 1, 384, 384)
        assert "instances" in data[0]
    except:
        raise
    finally:
        reader.join()
    assert reader.total_len() == 1


def test_process_lf(minimal_instance):
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 4)

    assert ex["image"].shape == torch.Size([1, 1, 384, 384])
    assert ex["instances"].shape == torch.Size([1, 4, 2, 2])
    assert torch.isnan(ex["instances"][:, 2:, :, :]).all()
    assert not torch.is_floating_point(ex["image"])
