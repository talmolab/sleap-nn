"""``.slp`` + ``--video_index`` selects the video's own frames (``VideoSelection``).

``--data_path project.slp --video_index N`` is "predict on video N": every frame
the video has (or ``--frames``), filtered by the project's annotations. Before,
it could only ever select frames that already had a ``LabeledFrame`` in the
``.slp`` -- so ``--exclude_user_labeled`` on a lightly-labeled project predicted
next to nothing, and a raw-video ``--data_path`` could not skip user-labeled
frames at all (the SLEAP GUI's "entire video" + "skip user labeled" bug).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner

from sleap_nn.cli import cli
from sleap_nn.inference.providers import (
    LabelsProvider,
    VideoProvider,
    VideoSelection,
)

N_FRAMES = 166  # small_robot.mp4
USER = [0, 79, 120]
PREDICTED_ONLY = 50
EMPTY = 60
SUGGESTED = [10, 79, 150]


def _user(skel):
    return sio.Instance.from_numpy(np.full((len(skel.nodes), 2), 20.0), skeleton=skel)


def _pred(skel):
    n = len(skel.nodes)
    return sio.PredictedInstance.from_numpy(
        np.full((n, 2), 20.0), skeleton=skel, point_scores=np.ones(n), score=1.0
    )


@pytest.fixture
def project(small_robot_minimal):
    """One 166-frame video: 3 user-labeled, 1 predicted-only, 1 empty, 3 suggested."""
    src = sio.load_slp(str(small_robot_minimal))
    video, skel = src.videos[0], src.skeletons[0]
    lfs = [
        sio.LabeledFrame(video=video, frame_idx=i, instances=[_user(skel)])
        for i in USER
    ]
    lfs[1].instances.append(_pred(skel))
    lfs += [
        sio.LabeledFrame(
            video=video, frame_idx=PREDICTED_ONLY, instances=[_pred(skel)]
        ),
        sio.LabeledFrame(video=video, frame_idx=EMPTY, instances=[]),
    ]
    return sio.Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=lfs,
        suggestions=[sio.SuggestionFrame(video=video, frame_idx=i) for i in SUGGESTED],
    )


@pytest.fixture
def pkg(project, tmp_path):
    """The project as a ``.pkg.slp`` with user frames + suggestions embedded."""
    path = tmp_path / "proj.pkg.slp"
    project.save(str(path), embed="user+suggestions")
    return sio.load_slp(str(path))


def _indices(selection, **kw):
    return list(selection.to_provider(**kw)._frame_indices)


def test_no_filters_is_the_whole_video(project):
    """No --frames and no filters reads every frame of the video."""
    provider = VideoSelection(project, 0).to_provider()
    assert isinstance(provider, VideoProvider)
    assert provider.num_frames() == N_FRAMES


def test_exclude_user_labeled_covers_unlabeled_frames(project):
    """--exclude_user_labeled keeps every frame except the user-labeled ones."""
    got = _indices(VideoSelection(project, 0, exclude_user_labeled=True))
    assert got == [i for i in range(N_FRAMES) if i not in USER]


def test_exclude_user_labeled_with_frames(project):
    """--exclude_user_labeled composes with --frames."""
    got = _indices(
        VideoSelection(
            project, 0, frames=list(range(75, 85)), exclude_user_labeled=True
        )
    )
    assert got == [75, 76, 77, 78, 80, 81, 82, 83, 84]


def test_only_labeled_frames_uses_labels_provider_with_gt(project):
    """--only_labeled_frames yields the user-labeled frames with their GT instances."""
    provider = VideoSelection(project, 0, only_labeled_frames=True).to_provider()
    assert isinstance(provider, LabelsProvider)
    assert [lf.frame_idx for lf in provider._labeled_frames] == USER
    assert all(lf.has_user_instances for lf in provider._labeled_frames)


def test_only_suggested_frames_skips_user_labeled(project):
    """--only_suggested_frames yields suggestions that are not yet user-labeled."""
    got = _indices(VideoSelection(project, 0, only_suggested_frames=True))
    assert got == [10, 150]


def test_only_predicted_frames(project):
    """--only_predicted_frames yields frames that already carry predictions."""
    got = _indices(VideoSelection(project, 0, only_predicted_frames=True))
    assert got == [PREDICTED_ONLY, 79]


def test_frames_sorted_deduped_and_out_of_range_dropped(project, caplog):
    """--frames are sorted, de-duplicated, and out-of-range indices dropped."""
    got = _indices(VideoSelection(project, 0, frames=[5, 3, 3, 1, 170, 165]))
    assert got == [1, 3, 5, 165]


def test_gt_layer_restricted_to_user_labeled(project):
    """A GT-fallback layer only runs on user-labeled frames."""
    provider = VideoSelection(project, 0).to_provider(needs_gt_instances=True)
    assert isinstance(provider, LabelsProvider)
    assert [lf.frame_idx for lf in provider._labeled_frames] == USER


def test_gt_layer_honors_exclude_user_labeled(project):
    # A GT-fallback layer can only run on user-labeled frames, and those are
    # excluded -- predict nothing rather than the frames the user excluded.
    """A GT-fallback layer never predicts frames the user excluded."""
    provider = VideoSelection(project, 0, exclude_user_labeled=True).to_provider(
        needs_gt_instances=True
    )
    assert provider.num_frames() == 0


def test_embedded_video_rejected(pkg):
    """A .pkg.slp video has no frames of its own beyond its labels; not selectable."""
    with pytest.raises(ValueError, match="embedded"):
        VideoSelection(pkg, 0)


@pytest.mark.parametrize("stream", [False, True])
def test_cli_pkg_video_index_keeps_labeled_frame_scoping(
    pkg, tmp_path, minimal_instance_single_instance_ckpt, stream
):
    """.pkg.slp + --video_index still predicts on the package's labeled frames."""
    out = tmp_path / "out.slp"
    args = [
        "predict",
        "--data_path",
        str(tmp_path / "proj.pkg.slp"),
        "--video_index",
        "0",
        "--model_paths",
        str(minimal_instance_single_instance_ckpt),
        "--device",
        "cpu",
    ]
    args += ["--stream-to-file", str(out)] if stream else ["-o", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    predicted = sorted(
        lf.frame_idx
        for lf in sio.load_slp(str(out)).labeled_frames
        if lf.has_predicted_instances
    )
    # Labeled frames that are stored in the package (50/60 aren't embedded).
    assert predicted == USER


def test_invalid_arguments(project):
    """Out-of-range video_index and contradictory filters raise."""
    with pytest.raises(IndexError):
        VideoSelection(project, 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        VideoSelection(project, 0, only_labeled_frames=True, exclude_user_labeled=True)


@pytest.mark.parametrize("stream", [False, True])
def test_cli_video_index_exclude_user_labeled_predicts_unlabeled_frames(
    project, tmp_path, minimal_instance_single_instance_ckpt, stream
):
    """The GUI's "entire video" + "skip user labeled" call, end to end."""
    slp = tmp_path / "proj.slp"
    project.save(str(slp))
    out = tmp_path / "out.slp"
    args = [
        "predict",
        "--data_path",
        str(slp),
        "--video_index",
        "0",
        "--frames",
        f"0-{N_FRAMES - 1}",
        "--exclude_user_labeled",
        "--model_paths",
        str(minimal_instance_single_instance_ckpt),
        "--device",
        "cpu",
    ]
    args += ["--stream-to-file", str(out)] if stream else ["-o", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output

    labels = sio.load_slp(str(out))
    predicted = sorted(
        lf.frame_idx for lf in labels.labeled_frames if lf.has_predicted_instances
    )
    assert predicted == [i for i in range(N_FRAMES) if i not in USER]
    assert Path(labels.videos[0].filename).name == "small_robot.mp4"


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"frames": [500, 501]}, "none of the requested --frames"),
        ({"frames": [1, 2], "only_labeled_frames": True}, "no user-labeled frames"),
        ({"frames": [1, 2], "only_suggested_frames": True}, "no unlabeled suggested"),
        ({"frames": [1, 2], "only_predicted_frames": True}, "no predicted frames"),
        ({"frames": [0, 79], "exclude_user_labeled": True}, "every frame"),
    ],
)
def test_empty_selection_warns_with_reason(project, kwargs, reason):
    """An empty selection logs why, instead of silently predicting nothing."""
    from loguru import logger

    messages = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        provider = VideoSelection(project, 0, **kwargs).to_provider()
    finally:
        logger.remove(sink_id)
    assert provider.num_frames() == 0
    assert any("No frames selected" in m and reason in m for m in messages)


def test_exclude_user_labeled_without_user_labels_is_whole_video(project):
    """--exclude_user_labeled on a video with no user labels keeps every frame."""
    for lf in list(project.labeled_frames):
        if lf.has_user_instances:
            project.labeled_frames.remove(lf)
    provider = VideoSelection(project, 0, exclude_user_labeled=True).to_provider()
    assert provider.num_frames() == N_FRAMES


@pytest.mark.parametrize("stream", [False, True])
def test_cli_video_index_out_of_range(
    project, tmp_path, minimal_instance_single_instance_ckpt, stream
):
    """An out-of-range --video_index is a usage error in both flows."""
    slp = tmp_path / "proj.slp"
    project.save(str(slp))
    out = str(tmp_path / "out.slp")
    args = [
        "predict",
        "--data_path",
        str(slp),
        "--video_index",
        "3",
        "--model_paths",
        str(minimal_instance_single_instance_ckpt),
        "--device",
        "cpu",
    ]
    args += ["--stream-to-file", out] if stream else ["-o", out]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    assert "out of range" in result.output
