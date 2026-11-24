"""Predicting on an HDF5 video whose ``frame_numbers`` table maps sparse frames.

An HDF5 file of extracted frames (encoded images, or a rank-4 array, plus a
``frame_numbers`` dataset beside them) is read by sleap-io's ``HDF5Video``
through ``frame_map``: ``video[n]`` reads frame NUMBER ``n``, while
``len(video)`` counts the stored images. The inference providers used to check
requested frames against ``range(len(video))``, so every frame number at or
above ``len(video)`` was skipped as "out of range", and ``frames=None`` asked
for positions (``1`` is not a frame number here, so reading it raised).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner
from loguru import logger

from sleap_nn.cli import cli
from sleap_nn.data.providers import check_video_readable, video_frame_numbers
from sleap_nn.inference.providers import VideoProvider, VideoSelection
from sleap_nn.inference.run import _video_embedded_in_slp

# Every 15th frame of small_robot.mp4 (166 frames), starting at 3 so frame 0 is
# not held either: 8 stored images numbered 3..108, almost all >= len(video).
FRAME_NUMBERS = list(range(3, 166, 15))[:8]
USER = 33


def _write_frames(path: Path, source: Path, encoded: bool) -> Path:
    """Write ``source``'s FRAME_NUMBERS frames as a frame-numbered HDF5 video."""
    src = sio.load_video(str(source))
    images = [np.asarray(src[n]) for n in FRAME_NUMBERS]
    src.close()
    with h5py.File(path, "w") as f:
        group = f.create_group("cam0")
        if encoded:
            ds = group.create_dataset(
                "video",
                (len(images),),
                dtype=h5py.special_dtype(vlen=np.dtype("uint8")),
            )
            for i, img in enumerate(images):
                ok, buf = cv2.imencode(".png", img)
                assert ok
                ds[i] = buf.ravel()
            ds.attrs["format"] = "png"
            ds.attrs["channel_order"] = "BGR"  # what cv2.imencode wrote
        else:
            group.create_dataset("video", data=np.stack(images))
        group.create_dataset("frame_numbers", data=np.asarray(FRAME_NUMBERS))
    return path


@pytest.fixture(params=["encoded", "rank4"])
def frames_h5(request, tmp_path, small_robot_minimal_video):
    """A frame-numbered HDF5 video: PNG-encoded images or a raw rank-4 array."""
    return _write_frames(
        tmp_path / f"frames_{request.param}.h5",
        small_robot_minimal_video,
        encoded=request.param == "encoded",
    )


@pytest.fixture
def source_frames(small_robot_minimal_video):
    """The source video's frames, by frame number."""
    src = sio.load_video(str(small_robot_minimal_video))
    frames = {n: np.asarray(src[n]) for n in FRAME_NUMBERS}
    src.close()
    return frames


def _read(provider):
    """``(frame_indices, images)`` of everything ``provider`` yields."""
    batches = list(provider)
    return (
        [int(i) for b in batches for i in b.frame_indices],
        [img for b in batches for img in b.images],
    )


def _warnings(fn):
    messages = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink_id)
    return result, messages


def test_fixture_is_frame_mapped(frames_h5):
    """The fixture is what the bug needs: frame numbers far past ``len(video)``."""
    video = sio.load_video(str(frames_h5))
    assert len(video) == len(FRAME_NUMBERS) == 8
    assert video_frame_numbers(video) == FRAME_NUMBERS
    with pytest.raises(IndexError):
        video[1]  # a position, not a frame number


def test_plain_video_has_no_frame_numbers(small_robot_minimal_video):
    """A video without a frame map keeps ``range(len(video))``."""
    assert video_frame_numbers(sio.load_video(str(small_robot_minimal_video))) is None


def test_frame_numbers_seen_through_a_crop(frames_h5):
    """A virtual crop of a frame-mapped video holds the same frame numbers."""
    cropped = sio.load_video(str(frames_h5)).crop((0, 0, 32, 32))
    assert video_frame_numbers(cropped) == FRAME_NUMBERS


@pytest.mark.parametrize("prefetch", [True, False])
def test_video_provider_reads_frame_numbers_above_len(
    frames_h5, source_frames, prefetch
):
    """Requested frame numbers >= len(video) are read, in the requested order."""
    wanted = [108, 18, 63]
    provider, messages = _warnings(
        lambda: VideoProvider(video=str(frames_h5), frames=wanted, prefetch=prefetch)
    )
    assert messages == []
    indices, images = _read(provider)
    assert indices == wanted
    for n, img in zip(indices, images):
        np.testing.assert_array_equal(img, source_frames[n])


@pytest.mark.parametrize("prefetch", [True, False])
def test_video_provider_without_frames_reads_every_frame_number(
    frames_h5, source_frames, prefetch
):
    """``frames=None`` reads the video's frame numbers, not positions."""
    provider = VideoProvider(video=str(frames_h5), batch_size=3, prefetch=prefetch)
    assert provider.num_frames() == len(FRAME_NUMBERS)
    assert len(provider) == 3
    indices, images = _read(provider)
    assert indices == FRAME_NUMBERS
    for n, img in zip(indices, images):
        np.testing.assert_array_equal(img, source_frames[n])


def test_video_provider_missing_frame_number_warns_and_skips(frames_h5):
    """A frame number the file doesn't hold is warned about and skipped."""
    provider, messages = _warnings(
        lambda: VideoProvider(video=str(frames_h5), frames=[18, 19, 500, 33, -1])
    )
    assert len(messages) == 1
    assert "not among the frame numbers" in messages[0]
    assert "numbered 3 to 108" in messages[0]
    assert str([-1, 19, 500]) in messages[0]
    assert _read(provider)[0] == [18, 33]


def test_legacy_video_reader_without_frames_reads_frame_numbers(frames_h5):
    """The legacy ``VideoReader`` default is the frame numbers too.

    The readability probe no longer fails on a file without frame 0, so a
    ``range(len(video))`` default would now stop quietly in the reader thread.
    """
    from queue import Queue

    from sleap_nn.data.providers import VideoReader

    reader = VideoReader(sio.load_video(str(frames_h5)), Queue(maxsize=4))
    assert reader.frames == FRAME_NUMBERS
    reader.start()
    read = []
    try:
        while True:
            frame = reader.frame_buffer.get(timeout=60)
            if frame["image"] is None:
                break
            read.append(int(frame["frame_idx"]))
    finally:
        reader.join(timeout=60)
    assert read == FRAME_NUMBERS


def test_readability_probe_uses_a_held_frame_number(frames_h5):
    """Frame 0 isn't a frame number here; the probe reads one that is."""
    video = sio.load_video(str(frames_h5))
    check_video_readable(video)
    check_video_readable(video, [19, 33])
    check_video_readable(video, [500])


@pytest.fixture
def project_slp(frames_h5, tmp_path, small_robot_minimal):
    """A ``.slp`` referencing the frames file, with one user-labeled frame."""
    skel = sio.load_slp(str(small_robot_minimal)).skeletons[0]
    video = sio.load_video(str(frames_h5))
    user = sio.Instance.from_numpy(np.full((len(skel.nodes), 2), 20.0), skeleton=skel)
    labels = sio.Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=USER, instances=[user])
        ],
    )
    path = tmp_path / "project.slp"
    labels.save(str(path))
    video.close()
    return path


def test_slp_keeps_the_frames_file_as_its_own_video(project_slp, frames_h5):
    """The ``.slp`` references the frames file; it is not a ``.pkg.slp`` embed."""
    video = sio.load_slp(str(project_slp)).videos[0]
    assert Path(video.filename).name == frames_h5.name
    assert video_frame_numbers(video) == FRAME_NUMBERS
    assert not _video_embedded_in_slp(video)


def test_video_selection_selects_frame_numbers(project_slp):
    """``VideoSelection`` candidates are the frame numbers, filters apply to them."""
    labels = sio.load_slp(str(project_slp))

    selection = VideoSelection(labels, 0)
    assert list(selection.candidate_indices()) == FRAME_NUMBERS
    provider = selection.to_provider()
    assert isinstance(provider, VideoProvider)
    assert _read(provider)[0] == FRAME_NUMBERS

    excluded = VideoSelection(labels, 0, exclude_user_labeled=True).to_provider()
    assert _read(excluded)[0] == [n for n in FRAME_NUMBERS if n != USER]


def test_video_selection_frames_above_len_and_missing(project_slp):
    """Requested frame numbers are kept; ones the file lacks are warned and dropped."""
    labels = sio.load_slp(str(project_slp))
    provider, messages = _warnings(
        lambda: VideoSelection(labels, 0, frames=[108, 18, 19, 18]).to_provider()
    )
    assert _read(provider)[0] == [18, 108]
    assert len(messages) == 1
    assert "1 of 3 requested" in messages[0]
    assert "not among the frame numbers" in messages[0]
    assert "[19]" in messages[0]


def test_pkg_slp_embed_is_still_rejected(project_slp, tmp_path):
    """A frame-numbered video embedded in a ``.pkg.slp`` keeps its old scoping."""
    pkg_path = tmp_path / "project.pkg.slp"
    sio.load_slp(str(project_slp)).save(str(pkg_path), embed="user")
    pkg = sio.load_slp(str(pkg_path))
    assert _video_embedded_in_slp(pkg.videos[0])
    with pytest.raises(ValueError, match="embedded"):
        VideoSelection(pkg, 0)


def _predict(args, out, model):
    result = CliRunner().invoke(
        cli,
        ["predict", *args, "--model_paths", str(model), "--device", "cpu"]
        + ["-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    labels = sio.load_slp(str(out))
    return labels, sorted(
        lf.frame_idx for lf in labels.labeled_frames if lf.has_predicted_instances
    )


@pytest.mark.parametrize(
    "frames, expected",
    [
        ("18,63,108", [18, 63, 108]),
        (None, FRAME_NUMBERS),
        ("18,19,108", [18, 108]),
    ],
)
def test_cli_predict_on_frames_file(
    frames_h5, tmp_path, minimal_instance_single_instance_ckpt, frames, expected
):
    """``sleap-nn predict`` on the frames file predicts its frame numbers."""
    args = ["--data_path", str(frames_h5)]
    if frames is not None:
        args += ["--frames", frames]
    labels, predicted = _predict(
        args, tmp_path / "out.slp", minimal_instance_single_instance_ckpt
    )
    assert predicted == expected
    assert Path(labels.videos[0].filename).name == frames_h5.name


def test_cli_slp_video_index_predicts_frame_numbers(
    project_slp, tmp_path, minimal_instance_single_instance_ckpt
):
    """``.slp`` + ``--video_index`` covers the frames file's unlabeled numbers too."""
    _, predicted = _predict(
        [
            "--data_path",
            str(project_slp),
            "--video_index",
            "0",
            "--exclude_user_labeled",
        ],
        tmp_path / "out.slp",
        minimal_instance_single_instance_ckpt,
    )
    assert predicted == [n for n in FRAME_NUMBERS if n != USER]
