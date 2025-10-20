import torch

from sleap_nn.data.providers import (
    LabelsReader,
    VideoReader,
    process_lf,
)
from queue import Queue
import sleap_io as sio
import numpy as np


def test_videoreader_provider(centered_instance_video, minimal_instance):
    """Test VideoReader class."""
    video = sio.load_video(centered_instance_video)
    queue = Queue(maxsize=4)
    reader = VideoReader(video=video, frame_buffer=queue, frames=[0, 1, 2, 3])
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

    # test with from_video method
    labels = sio.load_slp(minimal_instance)
    reader = VideoReader.from_video(video=labels.videos[0], queue_maxsize=4, frames=[0])
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
    except:
        raise
    finally:
        reader.join()
    assert reader.total_len() == 1

    # check graceful stop (video has 1100 frames)
    reader = VideoReader.from_filename(
        filename=centered_instance_video,
        queue_maxsize=4,
        frames=[1099, 1100, 1101, 1102, 1103],
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

    # with 2 frames
    labels = sio.load_slp(minimal_instance)
    lfs = [lf for lf in labels]
    lfs.append(
        sio.LabeledFrame(
            video=labels.videos[0],
            frame_idx=1,
            instances=[
                sio.PredictedInstance.from_numpy(
                    points_data=np.array([[1.0, 2.0], [2.0, 3.0]]),
                    skeleton=labels.skeletons[0],
                    point_scores=[0.1],
                    score=1.0,
                )
            ],
        )
    )
    new_labels = sio.Labels(
        videos=labels.videos,
        skeletons=labels.skeletons,
        labeled_frames=lfs,
    )
    assert len(new_labels) == 2
    queue = Queue(maxsize=4)
    reader = LabelsReader(labels=new_labels, frame_buffer=queue, instances_key=False)
    assert reader.max_height_and_width == (384, 384)
    assert reader.total_len() == 2

    # test only user labelled instance
    labels = sio.load_slp(minimal_instance)
    lfs = [lf for lf in labels]
    lfs.append(
        sio.LabeledFrame(
            video=labels.videos[0],
            frame_idx=1,
            instances=[
                sio.PredictedInstance.from_numpy(
                    points_data=np.array([[1.0, 2.0], [2.0, 3.0]]),
                    skeleton=labels.skeletons[0],
                    point_scores=[0.1],
                    score=1.0,
                )
            ],
        )
    )
    new_labels = sio.Labels(
        videos=labels.videos,
        skeletons=labels.skeletons,
        labeled_frames=lfs,
    )
    assert len(new_labels) == 2
    queue = Queue(maxsize=4)
    reader = LabelsReader(
        labels=new_labels,
        frame_buffer=queue,
        instances_key=False,
        only_labeled_frames=True,
    )
    assert reader.max_height_and_width == (384, 384)
    assert reader.total_len() == 1

    # test for suggested frames (no suggested frames)
    labels_2 = sio.Labels(
        videos=labels.videos,
        skeletons=labels.skeletons,
        labeled_frames=[
            sio.LabeledFrame(
                video=labels.videos[0],
                frame_idx=1,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        points_data=np.array([[1.0, 2.0], [2.0, 3.0]]),
                        skeleton=labels.skeletons[0],
                        point_scores=[0.1],
                        score=1.0,
                    )
                ],
            )
        ],
    )
    assert len(labels_2) == 1
    queue = Queue(maxsize=4)
    reader = LabelsReader(
        labels=labels_2,
        frame_buffer=queue,
        instances_key=False,
        only_suggested_frames=True,
    )
    assert reader.max_height_and_width == (384, 384)
    assert reader.total_len() == 0

    # test for suggested frames (1 suggested frames)
    labels_2 = sio.Labels(
        videos=labels.videos,
        skeletons=labels.skeletons,
        labeled_frames=[
            sio.LabeledFrame(
                video=labels.videos[0],
                frame_idx=1,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        points_data=np.array([[1.0, 2.0], [2.0, 3.0]]),
                        skeleton=labels.skeletons[0],
                        point_scores=[0.1],
                        score=1.0,
                    )
                ],
            )
        ],
    )
    labels_2.suggestions.append(sio.SuggestionFrame(labels.videos[0], frame_idx=1))
    assert len(labels_2) == 1
    queue = Queue(maxsize=4)
    reader = LabelsReader(
        labels=labels_2,
        frame_buffer=queue,
        instances_key=False,
        only_suggested_frames=True,
    )
    assert reader.max_height_and_width == (384, 384)
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
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=4,
    )

    assert ex["image"].shape == torch.Size([1, 1, 384, 384])
    assert ex["instances"].shape == torch.Size([1, 4, 2, 2])
    assert torch.isnan(ex["instances"][:, 2:, :, :]).all()
    assert not torch.is_floating_point(ex["image"])
