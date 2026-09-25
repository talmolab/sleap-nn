"""`sleap-nn predict` embeds the carrier the model was trained on (emb-review D2, D6).

A top-down segmentation or SAM output holds both carriers: every pose has a linked
mask. The two carriers crop differently (pose centroid vs mask center; an all-ones vs
a real mask for burn-in), so a model must embed the one it was trained on. Training
records it in `head_configs.embedding.embedding.detection_mode`; a model saved before
that field existed is taken to be mask-trained when it uses `burn_in`, and is
otherwise resolved by counting, as before.

Before this, the choice was a count with ties going to masks, whatever the model was
trained on.
"""

from __future__ import annotations

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from tests.cli.test_embedding_route_persistence import _flat, _predict, _vectors
from tests.inference.test_embedding_persistence import (
    _DIM,
    _build_embedding_config,
    _write_video,
)

N_FRAMES = 4


def _model_dir(tmp_path, name, *, detection_mode=None, burn_in=False):
    """A tiny random-weights embedding model whose config names its carrier."""
    from sleap_nn.training.lightning_modules import EmbeddingLightningModule

    cfg = _build_embedding_config()
    if detection_mode is not None:
        cfg.model_config.head_configs.embedding.embedding.detection_mode = (
            detection_mode
        )
    cfg.data_config.preprocessing.burn_in = burn_in
    module = EmbeddingLightningModule(
        model_type="embedding",
        backbone_type="unet",
        backbone_config=cfg.model_config.backbone_config,
        head_configs=cfg.model_config.head_configs,
        init_weights="xavier",
    ).eval()
    model_dir = tmp_path / name
    model_dir.mkdir()
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {},
            "pytorch-lightning_version": "2.0.0",
            "epoch": 0,
            "global_step": 0,
        },
        model_dir / "best.ckpt",
    )
    OmegaConf.save(cfg, model_dir / "training_config.yaml")
    return model_dir


def _both_carriers_slp(tmp_path, *, masks_per_frame=2):
    """Predicted poses, the first ``masks_per_frame`` of each frame with a linked mask."""
    video = _write_video(tmp_path / "v.mp4", n=N_FRAMES)
    skeleton = sio.Skeleton(nodes=["a", "b"])
    yy, xx = np.ogrid[:64, :64]
    frames = []
    for fi in range(N_FRAMES):
        instances, masks = [], []
        for k, x in enumerate((16.0, 44.0)):
            inst = sio.PredictedInstance.from_numpy(
                np.array([[x, 20.0], [x + 6, 28.0]]), skeleton=skeleton, score=0.9
            )
            instances.append(inst)
            if k < masks_per_frame:
                disk = ((yy - 24) ** 2 + (xx - (x + 3)) ** 2) <= 8**2
                mask = sio.PredictedSegmentationMask.from_numpy(disk)
                mask.instance = inst
                masks.append(mask)
        frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=fi, instances=instances, masks=masks
            )
        )
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])
    path = tmp_path / f"both_{masks_per_frame}.slp"
    sio.save_slp(labels, str(path), embed=False)
    return path


def _embedded_carriers(tmp_path, model_dir, slp):
    out = tmp_path / f"{model_dir.name}.{slp.stem}.out.slp"
    result = _predict(
        "-m", model_dir, "-i", slp, "--save_embeddings", "slp", "-o", out,
        "--device", "cpu",
    )  # fmt: skip
    assert result.exit_code == 0, _flat(result)
    vectors = _vectors(out)
    assert all(v is None or v.shape == (_DIM,) for _, _, v in vectors)
    return {
        carrier: sum(v is not None for _, c, v in vectors if c == carrier)
        for carrier in ("pose", "mask")
    }


@pytest.mark.parametrize("carrier", ["pose", "mask"])
def test_recorded_carrier_is_embedded(tmp_path, carrier):
    """Every pose has a mask (a tie): the recorded carrier decides, not the tie."""
    model_dir = _model_dir(tmp_path, f"{carrier}_model", detection_mode=carrier)
    slp = _both_carriers_slp(tmp_path)

    embedded = _embedded_carriers(tmp_path, model_dir, slp)

    other = "mask" if carrier == "pose" else "pose"
    assert embedded == {carrier: 2 * N_FRAMES, other: 0}


def test_burn_in_model_without_a_recorded_carrier_embeds_masks(tmp_path):
    """A model saved before `detection_mode` existed: burn-in means mask-trained.

    Here poses outnumber masks (one mask per two poses), so counting chose poses:
    a burn-in model then saw all-ones masks and no burn-in at all.
    """
    model_dir = _model_dir(tmp_path, "burn_in_model", burn_in=True)
    slp = _both_carriers_slp(tmp_path, masks_per_frame=1)

    assert _embedded_carriers(tmp_path, model_dir, slp) == {
        "pose": 0,
        "mask": N_FRAMES,
    }


def test_model_without_a_recorded_carrier_counts(tmp_path):
    """No recorded carrier and no burn-in: the carrier with more detections."""
    model_dir = _model_dir(tmp_path, "legacy_model")
    slp = _both_carriers_slp(tmp_path, masks_per_frame=1)

    assert _embedded_carriers(tmp_path, model_dir, slp) == {
        "pose": 2 * N_FRAMES,
        "mask": 0,
    }


def test_recorded_carrier_absent_falls_back_to_the_other(tmp_path):
    """A mask-trained model on a pose-only file (a fused detect -> embed run)."""
    model_dir = _model_dir(tmp_path, "mask_model", detection_mode="mask")
    slp = _both_carriers_slp(tmp_path, masks_per_frame=0)

    assert _embedded_carriers(tmp_path, model_dir, slp) == {
        "pose": 2 * N_FRAMES,
        "mask": 0,
    }
