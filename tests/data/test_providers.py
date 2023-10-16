import torch

from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import numpy as np


def test_providers(minimal_instance):
    l = LabelsReader.from_filename(minimal_instance)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])


def test_filter_user_instances(minimal_instance):
    # Create sample Labels object.

    # Create skeleton.
    skeleton = sio.Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )

    # Get video.
    min_labels = sio.load_slp(minimal_instance)
    video = min_labels.videos[0]

    # Create user labelled instance.
    user_inst = sio.Instance.from_numpy(
        points=np.array(
            [
                [11.4, 13.4],
                [13.6, 15.1],
                [0.3, 9.3],
            ]
        ),
        skeleton=skeleton,
    )

    # Create Predicted Instance.
    pred_inst = sio.PredictedInstance.from_numpy(
        points=np.array(
            [
                [10.2, 20.4],
                [5.8, 15.1],
                [0.3, 10.6],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.5, 0.6, 0.8]),
        instance_score=0.6,
    )

    # Create labeled frame.
    user_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[user_inst, pred_inst]
    )
    pred_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst])

    # Create labels.
    labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf, pred_lf]
    )

    l = LabelsReader(labels, user_instances_only=True)

    # Check user instance filtering.
    assert len(list(l)) == 1
    lf = next(iter(l))
    assert len(torch.squeeze(lf["instances"], dim=0)) == 1

    # Create labeled frame.
    user_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[user_inst, pred_inst]
    )
    pred_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst])
    # Create labels.
    labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf, pred_lf]
    )
    l = LabelsReader(labels, user_instances_only=False)
    assert len(list(l)) == 2
    lf = next(iter(l))
    assert len(torch.squeeze(lf["instances"], dim=0)) == 2

    # Test with only Predicted instance.
    pred_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst])
    labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[pred_lf])
    l = LabelsReader(labels, user_instances_only=True)
    assert len(list(l)) == 0
