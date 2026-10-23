"""``.slp`` + ``--video_index`` selects the video's own frames (``VideoSelection``).

``--data_path project.slp --video_index N`` is "predict on video N": every frame
the video has (or ``--frames``), filtered by the project's annotations. Before,
it could only ever select frames that already had a ``LabeledFrame`` in the
``.slp`` -- so ``--exclude_user_labeled`` on a lightly-labeled project predicted
next to nothing, and a raw-video ``--data_path`` could not skip user-labeled
frames at all (the SLEAP GUI's "entire video" + "skip user labeled" bug).
"""

from __future__ import annotations

import json
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


@pytest.fixture
def undecodable_project(project, tmp_path):
    """The project, with its video swapped for a file nothing can decode.

    The new video was never readable, so the ``.slp`` stores no shape for it and
    sleap-io reports it as having 0 frames. (A video that was readable when the
    project was saved keeps its stored frame count, and ``VideoProvider``'s own
    check already catches it.)
    """
    path = tmp_path / "undecodable.mp4"
    path.write_bytes(b"this is not a video")
    project.replace_videos(
        old_videos=[project.videos[0]], new_videos=[sio.Video.from_filename(str(path))]
    )
    # Reload so the video is opened the way the CLI opens it.
    slp = tmp_path / "undecodable.slp"
    project.save(str(slp))
    labels = sio.load_slp(str(slp))
    assert len(labels.videos[0]) == 0
    return labels


@pytest.mark.parametrize("frames", [None, [0, 1, 2]])
def test_undecodable_video_raises(undecodable_project, frames):
    """An undecodable video reports 0 frames: raise, don't select nothing.

    With ``frames``, every index used to be dropped as out of range for a
    0-frame video, so the run predicted nothing and exited 0.
    """
    selection = VideoSelection(undecodable_project, 0, frames=frames)
    with pytest.raises(ValueError, match="Could not read"):
        selection.to_provider()

    # An explicitly empty frame list asks for nothing, so there is nothing to check.
    assert _indices(VideoSelection(undecodable_project, 0, frames=[])) == []


@pytest.mark.parametrize(
    "kw, gt",
    [({"only_labeled_frames": True}, False), ({}, True)],
    ids=["only_labeled_frames", "needs_gt_instances"],
)
def test_undecodable_video_raises_on_gt_path(undecodable_project, kw, gt):
    """The LabelsProvider branch (GT instances ride along) checks the video too."""
    selection = VideoSelection(undecodable_project, 0, frames=[0, 1, 2], **kw)
    with pytest.raises(ValueError, match="Could not read"):
        selection.to_provider(needs_gt_instances=gt)


def test_frame_indices_raises_on_undecodable_video(undecodable_project):
    """Asking for the selection directly raises too, not only ``to_provider``."""
    selection = VideoSelection(undecodable_project, 0, frames=[0, 1, 2])
    with pytest.raises(ValueError, match="Could not read"):
        selection.frame_indices()


def test_readability_probe_uses_an_in_range_requested_frame(project, monkeypatch):
    """If frame 0 won't decode, probe the first requested frame that exists.

    The request's first index may be out of range; probing it would call a video
    whose requested frames all decode unreadable.
    """
    from sleap_io.io.video_reading import MediaVideo

    # Known length, as for a project video whose shape the .slp stored.
    assert len(project.videos[0]) == 166
    read_frame = MediaVideo._read_frame

    def _read_frame(self, frame_idx):
        if frame_idx == 0:
            raise IndexError("Failed to read frame index 0.")
        return read_frame(self, frame_idx)

    monkeypatch.setattr(MediaVideo, "_read_frame", _read_frame)

    selection = VideoSelection(project, 0, frames=[100000, 7, 5])

    assert _indices(selection) == [5, 7]


def test_falls_back_to_pyav_before_counting_frames(project, monkeypatch):
    """A video OpenCV can't decode is read with PyAV, not treated as empty."""
    pytest.importorskip("av")
    from sleap_io.io.video_reading import MediaVideo

    read_frame = MediaVideo._read_frame

    def _read_frame(self, frame_idx):
        if self.plugin == "opencv":
            raise IndexError(f"Failed to read frame index {frame_idx}.")
        return read_frame(self, frame_idx)

    monkeypatch.setattr(MediaVideo, "_read_frame", _read_frame)
    project.videos[0].set_video_plugin("opencv")

    selection = VideoSelection(project, 0, frames=[0, 1, 2])

    assert _indices(selection) == [0, 1, 2]
    assert project.videos[0].backend.plugin == "pyav"


@pytest.mark.parametrize("stream", [False, True])
def test_cli_video_index_frames_undecodable_video_fails(
    undecodable_project, tmp_path, minimal_instance_single_instance_ckpt, stream
):
    """The SLEAP GUI's call (``.slp`` + ``--video_index`` + ``--frames``) fails loudly.

    With ``--gui``, the error must also reach the GUI as the structured error line.
    """
    slp = tmp_path / "proj.slp"
    undecodable_project.save(str(slp))
    out = tmp_path / "out.slp"
    args = [
        "predict",
        "--data_path",
        str(slp),
        "--video_index",
        "0",
        "--frames",
        "0-9",
        "--model_paths",
        str(minimal_instance_single_instance_ckpt),
        "--device",
        "cpu",
    ]
    args += ["--stream-to-file", str(out)] if stream else ["-o", str(out)]
    result = CliRunner().invoke(cli, args + ["--gui"])

    assert result.exit_code != 0
    assert "Could not read" in str(result.exception)
    error_lines = [
        json.loads(line) for line in result.output.splitlines() if line.startswith("{")
    ]
    assert any(
        e.get("error") and "Could not read" in e["message"] for e in error_lines
    ), result.output
