"""CPU-only tests for the wired SAM inference-MVP surface (a fake backend).

No real SAM, no CUDA: every test injects a :class:`FakeBackend` (the
``backend=`` hook / a patched ``get_mask_backend``) so checkpoint loading and the
SAM encoder are never touched. Three target areas:

* :func:`sleap_nn.inference.sam.run_sam_segmentation` orchestration — masks
  emitted, ``.slp`` + overlay written, ``frames=`` subsetting, no-prompt skip,
  ``disjointify_masks``.
* :func:`sleap_nn.inference.run.predict` SAM short-circuit success path +
  forwarding of the ``sam_*`` / ``overlay_path`` / ``frames`` knobs.
* :func:`sleap_nn.inference.sam.overlay.save_mask_overlay` happy path + the two
  ``None`` returns (no frames / no masks).

The image source is the embedded-image fixture ``minimal_instance.pkg.slp`` so
``lf.image`` actually yields pixels (required by ``masks_for_frame`` / overlay),
with its GT ``sio.Instance`` poses rebuilt as ``sio.PredictedInstance`` so the
PLAN-L8 ``mask.instance`` pairing is exercised.
"""

from pathlib import Path

import numpy as np
import pytest

import sleap_io as sio

import sleap_nn.inference.sam as sam_pkg
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.run import predict
from sleap_nn.inference.sam import run_sam_segmentation
from sleap_nn.inference.sam.mask_layer import SamSegmentationLayer
from sleap_nn.inference.sam.overlay import save_mask_overlay


def _disk(h, w, cy, cx, r):
    """A boolean ``(h, w)`` disk of radius ``r`` centered at ``(cy, cx)``."""
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


class FakeBackend:
    """A :class:`MaskBackend`-shaped fake returning a fixed mask per prompt.

    ``mask_fn(image, prompt)`` produces one ``(H, W)`` bool mask; the score is a
    fixed value. Records each encoded image shape for assertions.
    """

    pred_iou_min = 0.88

    def __init__(self, mask_fn, score=0.77):
        """Stash the per-prompt ``mask_fn`` and the fixed score."""
        self._mask_fn, self._score, self.encoded = mask_fn, score, []

    def masks(self, image, prompts):
        """Return ``(masks, scores)`` for the prompts; record the image shape."""
        import numpy as np

        image = np.asarray(image)
        self.encoded.append(image.shape)
        ms, ss = [], []
        for p in prompts:
            ms.append(self._mask_fn(image, p).astype(bool))
            ss.append(float(self._score))
        return ms, ss


def _prompt_disk(radius=15):
    """A ``mask_fn`` placing a disk centered on the prompt's mean point."""

    def mask_fn(image, prompt):
        c = prompt.point_coords.mean(0)
        return _disk(image.shape[0], image.shape[1], c[1], c[0], radius)

    return mask_fn


def _pose_labels(minimal_instance, frame_indices=(0,)):
    """Build an in-memory ``sio.Labels`` of ``PredictedInstance`` poses.

    Loads the embedded-image fixture (so ``lf.image`` yields pixels), converts
    each GT ``sio.Instance`` to a ``sio.PredictedInstance`` (PLAN-L8 pairing), and
    rebuilds one ``LabeledFrame`` per requested ``frame_idx`` reusing the same
    video/poses (the fixture has a single frame; extra indices let us exercise
    ``frames=`` subsetting deterministically).
    """
    src = sio.load_slp(str(minimal_instance))
    skel = src.skeletons[0]
    gt_lf = src.labeled_frames[0]

    new_lfs = []
    for frame_idx in frame_indices:
        preds = []
        for inst in gt_lf.instances:
            pts = inst.numpy()[:, :2]
            n = pts.shape[0]
            preds.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    skeleton=skel,
                    point_scores=np.ones(n),
                    score=1.0,
                )
            )
        new_lfs.append(
            sio.LabeledFrame(video=gt_lf.video, frame_idx=frame_idx, instances=preds)
        )

    return sio.Labels(videos=list(src.videos), skeletons=[skel], labeled_frames=new_lfs)


# --------------------------------------------------------------------------- run_sam_segmentation


def test_run_sam_segmentation_emits_masks(minimal_instance):
    """It returns Labels with >=1 frame carrying a paired PredictedSegmentationMask."""
    labels = _pose_labels(minimal_instance)
    backend = FakeBackend(_prompt_disk())

    out = run_sam_segmentation(labels, "sam", backend=backend, prompt_mode="pose")

    assert isinstance(out, sio.Labels)
    assert len(out.labeled_frames) >= 1
    masked = [lf for lf in out.labeled_frames if lf.masks]
    assert masked, "expected >=1 frame carrying PredictedSegmentationMask"
    m = masked[0].masks[0]
    assert isinstance(m, sio.PredictedSegmentationMask)
    # PLAN-L8: predicted-instance pairing is exercised.
    assert m.instance is not None
    # The backend encoded the full frame.
    assert backend.encoded, "backend.masks() was never called"


def test_run_sam_segmentation_writes_output_slp(minimal_instance, tmp_path):
    """``output_path`` is written and reloads via ``sio.load_slp`` with masks."""
    labels = _pose_labels(minimal_instance)
    out_path = tmp_path / "o.slp"

    out = run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        output_path=out_path,
    )
    assert any(lf.masks for lf in out.labeled_frames)
    assert out_path.exists()

    reloaded = sio.load_slp(out_path.as_posix())
    rmasks = [m for lf in reloaded.labeled_frames for m in lf.masks]
    assert rmasks, "reloaded .slp has no masks"
    assert all(isinstance(m, sio.PredictedSegmentationMask) for m in rmasks)


def test_run_sam_segmentation_clean_empty_frames(minimal_instance):
    """clean_empty_frames drops 0-instance frames but keeps posed (mask-bearing) ones."""
    src = sio.load_slp(str(minimal_instance))
    skel = src.skeletons[0]
    gt = src.labeled_frames[0]
    preds = [
        sio.PredictedInstance.from_numpy(
            points_data=i.numpy()[:, :2],
            skeleton=skel,
            point_scores=np.ones(i.numpy().shape[0]),
            score=1.0,
        )
        for i in gt.instances
    ]
    posed = sio.LabeledFrame(video=gt.video, frame_idx=gt.frame_idx, instances=preds)
    # A 0-instance frame (reuse the embedded frame_idx so lf.image still resolves).
    empty = sio.LabeledFrame(video=gt.video, frame_idx=gt.frame_idx, instances=[])
    labels = sio.Labels(
        videos=list(src.videos), skeletons=[skel], labeled_frames=[posed, empty]
    )

    # Default keeps the empty frame (matches the regular path's default).
    kept = run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        clean_empty_frames=False,
    )
    assert len(kept.labeled_frames) == 2

    # clean_empty_frames=True drops only the 0-instance frame; the posed (and now
    # mask-bearing) frame is kept.
    cleaned = run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        clean_empty_frames=True,
    )
    assert len(cleaned.labeled_frames) == 1
    assert cleaned.labeled_frames[0].instances
    assert cleaned.labeled_frames[0].masks


def test_run_sam_segmentation_embed_true_self_contained(minimal_instance, tmp_path):
    """``embed="true"`` writes a self-contained .slp larger than ``embed="false"``.

    #652: the embedding policy is configurable. With ``embed="true"`` the output
    carries pixel data and reloads with a video reporting embedded images; it is
    strictly larger than the default ``embed="false"`` (backreference) output.
    """
    from sleap_nn.inference.run import _video_has_embedded_images

    out_false = tmp_path / "embed_false.slp"
    out_true = tmp_path / "embed_true.slp"

    run_sam_segmentation(
        str(minimal_instance),
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        output_path=out_false,
        embed="false",
    )
    run_sam_segmentation(
        str(minimal_instance),
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        output_path=out_true,
        embed="true",
    )
    assert out_false.exists() and out_true.exists()
    assert out_true.stat().st_size > out_false.stat().st_size
    reloaded = sio.load_slp(out_true.as_posix())
    assert any(_video_has_embedded_images(v) for v in reloaded.videos)
    # Masks still serialize in both cases.
    assert any(lf.masks for lf in reloaded.labeled_frames)


def test_run_sam_segmentation_restore_source_videos_controls_refs(
    minimal_instance, tmp_path
):
    """``restore_source_videos`` toggles the reloaded video filename reference.

    #652: on a non-embedding save, default ``True`` restores the original source
    video reference (the .mp4); ``False`` keeps the input ``.pkg.slp`` reference.
    """
    src = sio.load_slp(str(minimal_instance))
    source_name = Path(src.videos[0].source_video.filename).name
    input_name = Path(str(minimal_instance)).name

    out_restore = tmp_path / "restore_true.slp"
    out_keep = tmp_path / "restore_false.slp"

    run_sam_segmentation(
        str(minimal_instance),
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        output_path=out_restore,
        restore_source_videos=True,
    )
    run_sam_segmentation(
        str(minimal_instance),
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        output_path=out_keep,
        restore_source_videos=False,
    )

    restored = sio.load_slp(out_restore.as_posix())
    kept = sio.load_slp(out_keep.as_posix())
    # Default True -> the original source .mp4 reference.
    assert Path(restored.videos[0].filename).name == source_name
    # False -> the input .pkg.slp reference is kept.
    assert Path(kept.videos[0].filename).name == input_name


def test_run_sam_segmentation_writes_overlay_png(minimal_instance, tmp_path):
    """``overlay_path`` writes a review PNG to disk."""
    labels = _pose_labels(minimal_instance)
    overlay_path = tmp_path / "o.png"

    run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        overlay_path=overlay_path,
    )
    assert overlay_path.exists()


def test_run_sam_segmentation_frames_subset_existing(minimal_instance):
    """``frames=[0]`` masks only the requested (existing) frame."""
    # Two frames (idx 0 and 1); request only frame 0. The frames= filter runs
    # before any image is read, so the (image-less) idx-1 frame is dropped
    # before its pixels are ever fetched — only frame 0 (the embedded image)
    # is encoded and masked.
    labels = _pose_labels(minimal_instance, frame_indices=(0, 1))

    out = run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        frames=[0],
    )
    assert len(out.labeled_frames) == 1
    assert int(out.labeled_frames[0].frame_idx) == 0
    assert out.labeled_frames[0].masks


def test_run_sam_segmentation_frames_subset_nonexistent(minimal_instance):
    """A ``frames=`` index that matches no frame yields empty output."""
    labels = _pose_labels(minimal_instance, frame_indices=(0,))

    out = run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk()),
        prompt_mode="pose",
        frames=[999],  # no such frame_idx
    )
    assert out.labeled_frames == []


def test_run_sam_segmentation_emits_frame_without_prompt(minimal_instance):
    """A frame whose instances yield no prompt (all-NaN kpts) is emitted, masks=[].

    §3.5b: for review continuity the frame is retained (poses preserved) with an
    empty ``masks=[]`` rather than dropped when SAM yields no mask.
    """
    # A frame whose only instance has all-NaN keypoints yields no prompt -> no
    # mask, but the frame (and its pose) must survive.
    src = sio.load_slp(str(minimal_instance))
    skel = src.skeletons[0]
    gt_lf = src.labeled_frames[0]
    n = gt_lf.instances[0].numpy().shape[0]
    empty = sio.PredictedInstance.from_numpy(
        points_data=np.full((n, 2), np.nan),
        skeleton=skel,
        point_scores=np.zeros(n),
        score=0.0,
    )
    lf = sio.LabeledFrame(
        video=gt_lf.video, frame_idx=gt_lf.frame_idx, instances=[empty]
    )
    labels = sio.Labels(videos=list(src.videos), skeletons=[skel], labeled_frames=[lf])

    out = run_sam_segmentation(
        labels, "sam", backend=FakeBackend(_prompt_disk()), prompt_mode="pose"
    )
    assert len(out.labeled_frames) == 1
    out_lf = out.labeled_frames[0]
    assert int(out_lf.frame_idx) == int(gt_lf.frame_idx)
    assert list(out_lf.masks) == []  # no mask emitted...
    assert len(out_lf.instances) == 1  # ...but the pose is retained


def test_run_sam_segmentation_disjointify_multi_instance(minimal_instance):
    """``disjointify_masks=True`` runs without error on a >=2-instance frame."""
    # The fixture frame already has 2 instances; large overlapping disks +
    # disjointify must run without error and produce disjoint masks.
    labels = _pose_labels(minimal_instance)
    assert len(labels.labeled_frames[0].instances) >= 2

    out = run_sam_segmentation(
        labels,
        "sam",
        backend=FakeBackend(_prompt_disk(radius=80)),
        prompt_mode="pose",
        disjointify_masks=True,
    )
    lf = out.labeled_frames[0]
    assert len(lf.masks) == 2


# --------------------------------------------------------------------------- predict() short-circuit


def test_predict_sam_short_circuit_success(minimal_instance, tmp_path):
    """``predict(mask_backend='sam', ...)`` returns masks + writes the .slp."""
    labels = _pose_labels(minimal_instance)
    out_path = tmp_path / "p.slp"
    fb = FakeBackend(_prompt_disk())

    orig = sam_pkg.get_mask_backend
    sam_pkg.get_mask_backend = lambda *a, **k: fb
    try:
        out = predict(
            labels,
            mask_backend="sam",
            device="cpu",
            sam_prompt_mode="pose",
            output_path=out_path,
        )
    finally:
        sam_pkg.get_mask_backend = orig

    assert isinstance(out, sio.Labels)
    masked = [lf for lf in out.labeled_frames if lf.masks]
    assert masked, "predict() returned no masks"
    assert out_path.exists()
    reloaded = sio.load_slp(out_path.as_posix())
    assert any(lf.masks for lf in reloaded.labeled_frames)


def test_predict_sam_slp_output_not_embedded(minimal_instance, tmp_path):
    """§3.5a: the predict() SAM ``.slp`` output NEVER re-embeds images.

    Mirroring the regular prediction path, the output backreferences the source
    media via provenance instead of copying pixels (re-embedding is large and
    wasteful, and a ``.pkg.slp`` input stays matchable to its source videos
    regardless). The masks always serialize; only the image bytes are not
    duplicated.
    """
    out_path = tmp_path / "out.slp"
    src_pkg = Path(str(minimal_instance))
    fb = FakeBackend(_prompt_disk())

    orig = sam_pkg.get_mask_backend
    sam_pkg.get_mask_backend = lambda *a, **k: fb
    try:
        # Pass the .pkg.slp path directly (the common SAM input: embedded frames).
        predict(
            str(minimal_instance),
            mask_backend="sam",
            device="cpu",
            sam_prompt_mode="pose",
            output_path=out_path,
        )
    finally:
        sam_pkg.get_mask_backend = orig

    assert out_path.exists()
    reloaded = sio.load_slp(out_path.as_posix())
    # Masks serialize into the .slp ...
    assert any(lf.masks for lf in reloaded.labeled_frames)
    # ... but images are NOT re-embedded: the video does not point at the output
    # itself (no self-embed), and the output is smaller than the embedded input
    # (no pixel duplication).
    rv = reloaded.videos[0]
    assert Path(rv.filename).resolve() != out_path.resolve()
    assert out_path.stat().st_size < src_pkg.stat().st_size


def test_predict_sam_forwards_sam3_model_id(minimal_instance, tmp_path):
    """``sam3_model_id`` flows predict() -> run_sam_segmentation -> get_mask_backend.

    §3.2: capture the ``sam3_model_id`` reaching ``get_mask_backend`` (the real
    backend builder) to prove the value is threaded end-to-end. No real SAM3 is
    loaded — ``get_mask_backend`` is monkeypatched to a fake.
    """
    labels = _pose_labels(minimal_instance)
    captured = {}

    orig = sam_pkg.get_mask_backend

    def fake_get_mask_backend(mask_backend, **kwargs):
        captured["mask_backend"] = mask_backend
        captured["sam3_model_id"] = kwargs.get("sam3_model_id")
        return FakeBackend(_prompt_disk())

    sam_pkg.get_mask_backend = fake_get_mask_backend
    try:
        predict(
            labels,
            mask_backend="sam3",
            device="cpu",
            sam_prompt_mode="pose",
            sam3_model_id="acme/custom-sam3",
            output_path=tmp_path / "s3.slp",
        )
    finally:
        sam_pkg.get_mask_backend = orig

    assert captured["mask_backend"] == "sam3"
    assert captured["sam3_model_id"] == "acme/custom-sam3"


@pytest.mark.parametrize("output_format", ["analysis_h5", "both"])
def test_predict_sam_rejects_non_slp_output_format(
    minimal_instance, tmp_path, output_format
):
    """The SAM path rejects output formats that cannot represent masks.

    The SLEAP Analysis HDF5 format stores poses/tracks, not
    ``PredictedSegmentationMask`` — requesting ``analysis_h5``/``both`` would
    silently drop the masks (the actual output), so predict() raises instead of
    writing a mask-less ``.h5``.
    """
    out_path = tmp_path / "p.slp"
    fb = FakeBackend(_prompt_disk())

    orig = sam_pkg.get_mask_backend
    sam_pkg.get_mask_backend = lambda *a, **k: fb
    try:
        with pytest.raises(ValueError, match="only supports output_format='slp'"):
            predict(
                str(minimal_instance),
                mask_backend="sam",
                device="cpu",
                sam_prompt_mode="pose",
                output_path=out_path,
                output_format=output_format,
            )
    finally:
        sam_pkg.get_mask_backend = orig

    # Nothing was written (we rejected before producing any output).
    assert not out_path.exists()
    assert not list(tmp_path.glob("*.analysis.h5"))


def test_predict_sam_forwards_kwargs(minimal_instance, tmp_path, monkeypatch):
    """``predict`` forwards the SAM knobs / overlay_path / frames downstream."""
    labels = _pose_labels(minimal_instance, frame_indices=(0, 1))
    captured = {}

    def fake_run_sam_segmentation(source, mask_backend, **kwargs):
        captured["source"] = source
        captured["mask_backend"] = mask_backend
        captured.update(kwargs)
        # Return a minimal Labels so predict()'s downstream save path is happy.
        return sio.Labels(
            videos=list(labels.videos),
            skeletons=list(labels.skeletons),
            labeled_frames=[],
        )

    monkeypatch.setattr(
        "sleap_nn.inference.sam.run_sam_segmentation", fake_run_sam_segmentation
    )

    overlay_path = tmp_path / "ov.png"
    out = predict(
        labels,
        mask_backend="sam",
        device="cpu",
        sam_prompt_mode="centroid",
        sam_anchor_ind=1,
        sam_disjointify_masks=True,
        overlay_path=overlay_path,
        frames=[1],
    )

    assert isinstance(out, sio.Labels)
    assert captured["mask_backend"] == "sam"
    assert captured["prompt_mode"] == "centroid"
    assert captured["anchor_ind"] == 1
    assert captured["disjointify_masks"] is True
    assert captured["overlay_path"] == overlay_path
    assert captured["frames"] == [1]


def test_predict_sam_forwards_embed_restore(minimal_instance, tmp_path, monkeypatch):
    """``predict`` forwards embed / restore_source_videos into run_sam_segmentation."""
    labels = _pose_labels(minimal_instance)
    captured = {}

    def fake_run_sam_segmentation(source, mask_backend, **kwargs):
        captured.update(kwargs)
        return sio.Labels(
            videos=list(labels.videos),
            skeletons=list(labels.skeletons),
            labeled_frames=[],
        )

    monkeypatch.setattr(
        "sleap_nn.inference.sam.run_sam_segmentation", fake_run_sam_segmentation
    )

    predict(
        labels,
        mask_backend="sam",
        device="cpu",
        sam_prompt_mode="pose",
        output_path=tmp_path / "e.slp",
        embed="auto",
        restore_source_videos=False,
    )
    assert captured["embed"] == "auto"
    assert captured["restore_source_videos"] is False


def test_predict_sam_default_embed_restore_forwarded(
    minimal_instance, tmp_path, monkeypatch
):
    """Default predict() forwards the byte-for-byte defaults to run_sam_segmentation."""
    labels = _pose_labels(minimal_instance)
    captured = {}

    def fake_run_sam_segmentation(source, mask_backend, **kwargs):
        captured.update(kwargs)
        return sio.Labels(
            videos=list(labels.videos),
            skeletons=list(labels.skeletons),
            labeled_frames=[],
        )

    monkeypatch.setattr(
        "sleap_nn.inference.sam.run_sam_segmentation", fake_run_sam_segmentation
    )

    predict(
        labels,
        mask_backend="sam",
        device="cpu",
        sam_prompt_mode="pose",
        output_path=tmp_path / "e.slp",
    )
    assert captured["embed"] == "false"
    assert captured["restore_source_videos"] is True


# --------------------------------------------------------------------------- save_mask_overlay


def _frame_with_masks(minimal_instance):
    """A ``sio.Labels`` whose single frame carries real ``PredictedSegmentationMask``s.

    Masks are built through the standard path (``masks_for_frame`` ->
    ``Outputs.to_masks``) so they are genuine masks with scale/offset.
    """
    labels = _pose_labels(minimal_instance)
    lf = labels.labeled_frames[0]
    layer = SamSegmentationLayer(FakeBackend(_prompt_disk()), prompt_mode="pose")
    frame_masks = layer.masks_for_frame(lf.image, lf.instances)
    assert frame_masks, "expected non-empty frame masks for the overlay fixture"
    masks = Outputs(pred_masks=[frame_masks]).to_masks(0)
    masked_lf = sio.LabeledFrame(
        video=lf.video,
        frame_idx=lf.frame_idx,
        instances=list(lf.instances),
        masks=masks,
    )
    return sio.Labels(
        videos=list(labels.videos),
        skeletons=list(labels.skeletons),
        labeled_frames=[masked_lf],
    )


def test_save_mask_overlay_happy_path(minimal_instance, tmp_path):
    """It writes a PNG that ``cv2.imread`` reloads as an (H, W, 3) image."""
    import cv2

    labels = _frame_with_masks(minimal_instance)
    out_path = tmp_path / "overlay.png"

    result = save_mask_overlay(labels, out_path)
    assert result is not None
    assert result.exists()

    img = cv2.imread(result.as_posix())
    assert img is not None
    assert img.ndim == 3 and img.shape[-1] == 3


def test_save_mask_overlay_returns_none_no_frames(tmp_path):
    """It returns ``None`` when the labels have no frames."""
    labels = sio.Labels(videos=[], skeletons=[], labeled_frames=[])
    assert save_mask_overlay(labels, tmp_path / "none.png") is None


def test_save_mask_overlay_returns_none_no_masks(minimal_instance, tmp_path):
    """It returns ``None`` when the frame carries no masks."""
    # A frame with image + instances but no masks attached.
    labels = _pose_labels(minimal_instance)
    assert not labels.labeled_frames[0].masks
    assert save_mask_overlay(labels, tmp_path / "none.png") is None
