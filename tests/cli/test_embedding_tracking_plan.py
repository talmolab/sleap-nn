"""The embedding routes refuse an untrackable plan BEFORE their inference (emb-review F4).

`apply_tracking` is the last step of `sleap-nn predict -m <embedding> ... -t`: on the
fused route it runs after the detection stack AND the embedding pass, on the lone
(WF2) route after the embedding pass. The carrier rules it enforces need only the
carriers the labels hold and the carrier the vectors land on, both known beforehand:

- Fused route: a segmentation stack (top-down `centroid` + `centered_instance_
  segmentation`, `bottomup_segmentation`, ...) emits masks only, so tracking follows
  the masks, and a pose `--features` (e.g. with `--appearance_weight`) is refused --
  it used to be refused only after both passes, by a raw `ValueError` traceback.
- Lone route: a pose + mask file embedded by a mask-trained model puts the vectors on
  the masks, so an `--appearance_weight` blend that tracks the poses reads none.
"""

from __future__ import annotations

import numpy as np
import pytest
import sleap_io as sio
from omegaconf import OmegaConf

from tests.cli.test_embedding_carrier import N_FRAMES, _both_carriers_slp, _model_dir
from tests.cli.test_embedding_route_persistence import _flat, _predict
from tests.inference.test_embedding_persistence import _write_video

CENTROID_CKPT = "tests/assets/model_ckpts/minimal_instance_centroid"
CENTERED_CKPT = "tests/assets/model_ckpts/minimal_instance_centered_instance"
SLP = "tests/assets/datasets/minimal_instance.pkg.slp"

_SEG_HEADS = {
    "centered_instance_segmentation": {
        "segmentation": {"output_stride": 2, "loss_weight": 1.0, "anchor_part": None}
    },
    "bottomup_segmentation": {
        "segmentation": {"output_stride": 2, "loss_weight": 1.0},
        "center": {"sigma": 5.0, "output_stride": 2, "loss_weight": 1.0},
        "offsets": {"output_stride": 2, "loss_weight": 0.1},
    },
}


def _config_only_dir(tmp_path, name, heads):
    """A model dir holding only a `training_config.yaml` with these head configs.

    The early checks read the model type off the saved config; nothing is loaded.
    """
    dest = tmp_path / name
    dest.mkdir()
    cfg = OmegaConf.load(f"{CENTROID_CKPT}/training_config.yaml")
    head_configs = {key: None for key in cfg.model_config.head_configs}
    head_configs.update(heads)
    cfg.model_config.head_configs = OmegaConf.create(head_configs)
    OmegaConf.save(cfg, dest / "training_config.yaml")
    return dest.as_posix()


@pytest.fixture
def embedding_dir(tmp_path):
    return _config_only_dir(
        tmp_path,
        "embedding",
        {"embedding": {"embedding": {"embedding_dim": 16, "output_stride": 16}}},
    )


@pytest.fixture
def no_detection(monkeypatch):
    """Fail the test if the fused route runs its detection stack."""
    import sleap_nn.cli as cli_mod

    def detect(*args, **kwargs):
        pytest.fail("the detection stack ran before the plan was refused")

    monkeypatch.setattr(cli_mod, "_run_in_memory_new_flow", detect)


class _Detected(Exception):
    """Raised by a stand-in detection stack: validation let the run through."""


@pytest.fixture
def detection_reached(monkeypatch):
    import sleap_nn.cli as cli_mod

    def detect(*args, **kwargs):
        raise _Detected()

    monkeypatch.setattr(cli_mod, "_run_in_memory_new_flow", detect)


def _fused(model_dirs, *extra):
    args = []
    for model_dir in model_dirs:
        args += ["-m", model_dir]
    return _predict(*args, "-i", SLP, "-t", "--save_embeddings", "slp", *extra)


def _seg_stack(tmp_path, kind):
    seg = _config_only_dir(tmp_path, kind, {kind: _SEG_HEADS[kind]})
    return [CENTROID_CKPT, seg] if kind == "centered_instance_segmentation" else [seg]


# ── fused route: refused before the detection stack runs ────────────────────────


@pytest.mark.parametrize(
    "kind", ["centered_instance_segmentation", "bottomup_segmentation"]
)
@pytest.mark.parametrize(
    "extra",
    [
        ("--features", "keypoints", "--appearance_weight", "0.3"),
        ("--features", "centroids", "--appearance_weight", "0.3"),
        ("--features", "keypoints"),  # geometry only, vectors kept
        ("--scoring_method", "oks", "--appearance_weight", "0.3"),
    ],
    ids=["keypoints_blend", "centroids_blend", "keypoints", "oks_blend"],
)
def test_segmentation_stack_with_pose_options_is_refused_before_detecting(
    tmp_path, embedding_dir, no_detection, kind, extra
):
    """The masks a segmentation stack emits cannot be tracked by pose geometry."""
    result = _fused([embedding_dir, *_seg_stack(tmp_path, kind)], *extra)

    assert result.exit_code == 2, _flat(result)
    text = _flat(result)
    assert "emits mask detections only" in text
    assert "requires features='masks' and scoring_method='mask_iou'" in text


def test_segmentation_stack_with_pose_cleanup_is_refused_before_detecting(
    tmp_path, embedding_dir, no_detection
):
    """Appearance-only tracking of masks refuses the pose cull/clean/connect options."""
    result = _fused(
        [embedding_dir, *_seg_stack(tmp_path, "centered_instance_segmentation")],
        "--tracking_clean_instance_count",
        "2",
    )

    assert result.exit_code == 2, _flat(result)
    assert "does not support the pose cull/clean/connect options" in _flat(result)


@pytest.mark.parametrize(
    "extra",
    [
        ("--appearance_weight", "0.3"),
        ("--features", "masks", "--appearance_weight", "0.3"),
        (),  # appearance only
    ],
    ids=["blend_auto_features", "blend_masks", "appearance_only"],
)
def test_segmentation_stack_tracking_masks_gets_to_detection(
    tmp_path, embedding_dir, detection_reached, extra
):
    """...while a plan that tracks the masks is let through to the detection stack."""
    result = _fused(
        [embedding_dir, *_seg_stack(tmp_path, "centered_instance_segmentation")],
        *extra,
    )

    assert isinstance(result.exception, _Detected), _flat(result)


def test_pose_stack_with_pose_features_gets_to_detection(
    embedding_dir, detection_reached
):
    """A pose stack's poses are what a pose `--features` blend tracks: not refused."""
    result = _fused(
        [embedding_dir, CENTROID_CKPT, CENTERED_CKPT],
        "--features",
        "keypoints",
        "--appearance_weight",
        "0.3",
    )

    assert isinstance(result.exception, _Detected), _flat(result)


def test_trained_carrier_decides_when_both_carriers_will_be_present():
    """The prediction applies the embedder's rule (D2): the trained carrier when the
    labels will hold it, else the only carrier; unknown when it comes down to counts."""
    from sleap_nn.data.custom_datasets import predict_embedding_carrier

    assert predict_embedding_carrier({"mask"}, "pose") == "mask"
    assert predict_embedding_carrier({"pose"}, "mask") == "pose"
    assert predict_embedding_carrier({"pose", "mask"}, "pose") == "pose"
    assert predict_embedding_carrier({"pose", "mask"}, "mask") == "mask"
    assert predict_embedding_carrier({"pose", "mask"}, None) is None
    assert predict_embedding_carrier(set(), "mask") is None


# ── lone (WF2) route: refused before the embedding pass ─────────────────────────


@pytest.fixture
def no_embedding_pass(monkeypatch):
    import sleap_nn.inference.embedding as embedding

    def embed(*args, **kwargs):
        pytest.fail("the embedding pass ran before the plan was refused")

    monkeypatch.setattr(embedding, "embed_labels", embed)


def _masks_only_slp(tmp_path):
    """Predicted masks, no poses: what a segmentation model writes."""
    video = _write_video(tmp_path / "masks.mp4", n=N_FRAMES)
    yy, xx = np.ogrid[:64, :64]
    frames = []
    for fi in range(N_FRAMES):
        masks = [
            sio.PredictedSegmentationMask.from_numpy(
                ((yy - 24) ** 2 + (xx - x) ** 2) <= 8**2
            )
            for x in (19.0, 47.0)
        ]
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=fi, instances=[], masks=masks)
        )
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[])
    path = tmp_path / "masks.slp"
    sio.save_slp(labels, str(path), embed=False)
    return path


def test_blend_on_the_untrained_carrier_is_refused_before_embedding(
    tmp_path, no_embedding_pass
):
    """Mask-trained model on a pose + mask file: its vectors land on the masks, so a
    blend tracking the poses would read none."""
    model_dir = _model_dir(tmp_path, "mask_model", detection_mode="mask")
    slp = _both_carriers_slp(tmp_path)

    result = _predict(
        "-m", model_dir, "-i", slp, "-t", "--features", "keypoints",
        "--appearance_weight", "0.3", "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    text = _flat(result)
    assert "would be a silent no-op" in text
    assert "Tracking follows the pose carrier" in text
    assert "detection_mode=mask" in text
    assert "--features masks" in text


def test_masks_with_pose_features_are_refused_before_embedding(
    tmp_path, no_embedding_pass
):
    model_dir = _model_dir(tmp_path, "mask_model", detection_mode="mask")

    result = _predict(
        "-m", model_dir, "-i", _masks_only_slp(tmp_path), "-t", "--features",
        "keypoints", "--appearance_weight", "0.3", "--device", "cpu",
    )  # fmt: skip

    assert result.exit_code == 2, _flat(result)
    assert "requires features='masks' and scoring_method='mask_iou'" in _flat(result)


def test_blend_on_the_trained_carrier_tracks(tmp_path):
    """The way out the refusal names works: track the masks the vectors are on."""
    model_dir = _model_dir(tmp_path, "mask_model", detection_mode="mask")
    out = tmp_path / "tracked.slp"

    result = _predict(
        "-m", model_dir, "-i", _both_carriers_slp(tmp_path), "-t", "--features",
        "masks", "--appearance_weight", "0.3", "--device", "cpu", "-o", out,
    )  # fmt: skip

    assert result.exit_code == 0, _flat(result)
    tracked = sio.load_slp(str(out))
    masks = [m for lf in tracked.labeled_frames for m in lf.masks]
    assert len(masks) == 2 * N_FRAMES
    assert all(m.track is not None for m in masks)
