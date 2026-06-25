"""Regression tests locking in sleap-io 0.8.0 behaviour relied on by sleap-nn.

sleap-nn tracks a sleap-io ``0.8.0`` dev build and is already migrated onto the
0.7.0 unified-annotation architecture. A compatibility audit (see
``scratch/2026-06-24-sleap-io-0.8.0-compat/``) found **no breakages**, but three
sleap-io behaviours are the "silent" kind (default-argument flips and
output-shape changes whose call-site signatures are unchanged) that a future
sleap-io patch could regress without a clear error. These tests pin them:

* **#483 / #490 / #495 — video-metadata serialisation.** ``Labels.save`` now
  serialises ``video.shape``/grayscale/fps from recorded backend metadata by
  default (``prefer_metadata=True``) instead of decoding frames, with companion
  fixes that invalidate stale metadata on relink/grayscale-flip. sleap-nn's
  prediction writer (:func:`sleap_nn.inference.run.save_predictions`) and the
  training GT save (``sleap_nn/training/model_trainer.py``) both round-trip
  videos through ``Labels.save``; these tests assert the saved/reloaded
  ``video.shape`` (height, width, **and** channel count) is preserved for both
  on-disk and embedded sources.
* **#368 / #480 — full-video-span analysis arrays.** ``save_analysis_h5`` now
  sizes its arrays to ``len(video)`` (trailing empty frames) rather than
  ``last_labeled_frame + 1``. sleap-nn's
  :func:`sleap_nn.inference.run.save_analysis_h5_files` wrapper must tolerate and
  emit the full-span output.

These run CPU-only and decode no video frames (only container metadata), so they
are fast and safe on the CI matrix.
"""

import h5py
import numpy as np
import sleap_io as sio

from sleap_nn.inference.run import save_analysis_h5_files, save_predictions


def _toy_labels_on_video(
    video: sio.Video, frame_idx: int = 0, with_track: bool = False
):
    """Build a tiny predicted ``Labels`` with one instance on ``video``."""
    skeleton = sio.Skeleton(["A", "B"])
    points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
    point_scores = np.array([0.9, 0.9], dtype="float32")
    track = sio.Track("track_0") if with_track else None
    inst = sio.PredictedInstance.from_numpy(
        points,
        skeleton=skeleton,
        point_scores=point_scores,
        score=0.9,
        track=track,
    )
    lf = sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst])
    labels = sio.Labels(labeled_frames=[lf])
    if with_track:
        labels.tracks = [track]
    return labels


def test_predict_save_roundtrip_preserves_ondisk_video_shape(
    small_robot_minimal_video, tmp_path
):
    """save_predictions -> reload keeps an on-disk (RGB) video's shape (#483/#490/#495).

    The on-disk, non-embedding save path is the primary #483 scenario: shape is
    serialised from recorded backend metadata rather than decoded.
    """
    video = sio.load_video(small_robot_minimal_video.as_posix())
    expected_shape = video.shape  # (frames, H, W, C); C == 3 (RGB)
    assert expected_shape is not None and expected_shape[-1] == 3

    labels = _toy_labels_on_video(video)
    out = tmp_path / "ondisk.predictions.slp"
    save_predictions(
        labels, out, output_format="slp", embed="false", restore_source_videos=True
    )

    reloaded = sio.load_slp(out.as_posix())
    assert reloaded.videos[0].shape == expected_shape


def test_predict_save_roundtrip_preserves_embedded_video_shape(
    minimal_instance, tmp_path
):
    """save_predictions -> reload keeps an embedded (grayscale) video's shape.

    Guards the grayscale channel count (C == 1) through the embedded ``.pkg.slp``
    backreference save path (#483/#495).
    """
    labels = sio.load_slp(minimal_instance.as_posix())
    expected_shape = labels.videos[0].shape  # (1, 384, 384, 1) grayscale
    assert expected_shape is not None and expected_shape[-1] == 1

    out = tmp_path / "embedded.predictions.slp"
    save_predictions(
        labels, out, output_format="slp", embed="false", restore_source_videos=False
    )

    reloaded = sio.load_slp(out.as_posix())
    assert reloaded.videos[0].shape == expected_shape


def test_save_analysis_h5_spans_full_video_length(small_robot_minimal_video, tmp_path):
    """Analysis HDF5 arrays span len(video), not last_labeled_frame + 1 (#368/#480).

    A single labelled frame at index 0 on a 166-frame video must still produce
    arrays whose frame axis covers all 166 frames, with only frame 0 occupied.
    """
    video = sio.load_video(small_robot_minimal_video.as_posix())
    n_frames = len(video)
    assert n_frames > 1

    labels = _toy_labels_on_video(video, frame_idx=0, with_track=True)
    out = tmp_path / "span.predictions.slp"

    written = save_analysis_h5_files(labels, out)
    assert len(written) == 1

    with h5py.File(written[0].as_posix(), "r") as f:
        # tracks: (n_tracks, n_nodes, 2, n_frames) -> frame axis is last.
        assert f["tracks"].shape[-1] == n_frames
        # track_occupancy: (n_frames, n_tracks).
        occupancy = f["track_occupancy"][:]
        assert occupancy.shape[0] == n_frames
        # Only the single labelled frame is occupied; the rest is the new
        # trailing all-empty span.
        assert int(occupancy[0].sum()) == 1
        assert int(occupancy[1:].sum()) == 0


def test_gt_save_reload_preserves_user_labels_and_shape(minimal_instance, tmp_path):
    """GT save (restore_original_videos=False) -> reload preserves eval inputs.

    Mirrors the training ground-truth save in
    ``sleap_nn/training/model_trainer.py`` (``labels_gt.{split}.{idx}.slp`` written
    with ``restore_original_videos=False``) that ``sleap_nn/train.py`` reloads for
    post-training evaluation. Locks the #483/#490/#495 video-metadata bundle plus
    user-instance/skeleton/point preservation against that round-trip.
    """
    labels = sio.load_slp(minimal_instance.as_posix())
    expected_shape = labels.videos[0].shape
    expected_counts = [len(lf.instances) for lf in labels.labeled_frames]
    expected_points = labels.labeled_frames[0].instances[0].numpy()
    expected_nodes = [n.name for n in labels.skeleton.nodes]

    out = tmp_path / "labels_gt.val.0.slp"
    labels.save(out.as_posix(), restore_original_videos=False)

    reloaded = sio.load_slp(out.as_posix())
    assert reloaded.videos[0].shape == expected_shape
    assert [len(lf.instances) for lf in reloaded.labeled_frames] == expected_counts
    assert [n.name for n in reloaded.skeleton.nodes] == expected_nodes
    np.testing.assert_allclose(
        reloaded.labeled_frames[0].instances[0].numpy(),
        expected_points,
        equal_nan=True,
    )
