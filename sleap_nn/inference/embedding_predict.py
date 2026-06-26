"""Inference for the ``embedding`` model type: crops -> appearance vectors.

Loads a trained embedding model and embeds the instance/mask crops of a ``.slp`` file,
writing the embeddings + index arrays (video / frame / detection / track) to a simple
``.h5`` for offline retrieval / clustering / re-ID experimentation (SPEC §6, the optional
stream). The crop pipeline (grayscale + mask burn-in + per-crop standardize) is IDENTICAL
to training, so the embeddings are consistent with the validation-time retrieval metrics.

This is the standalone (precropped / mask-driven) embedder path. Composing the embedder
with a centroid model for full top-down inference (the ``TopDownEmbeddingLayer`` /
``Outputs.pred_embeddings`` predictor integration) is a follow-up; this path covers the
re-ID experimentation the model is built for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
import sleap_io as sio

from sleap_nn.inference.loaders import _load_lightning_module
from sleap_nn.training.lightning_modules import EmbeddingLightningModule
from sleap_nn.data.custom_datasets import (
    EmbeddingDataset,
    resolve_embedding_class_names,
)


@torch.inference_mode()
def embed_labels(
    model_path: str,
    data_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> str:
    """Embed every tracked mask crop in ``data_path`` with a trained embedding model.

    Args:
        model_path: Trained embedding-model directory (with ``best.ckpt`` +
            ``training_config.yaml``).
        data_path: ``.slp`` file to embed.
        output_path: Output ``.h5`` path. Defaults to ``<data_path>.embeddings.h5``.
        device: Torch device.
        batch_size: Crops per forward pass.

    Returns:
        The output ``.h5`` path.
    """
    import h5py

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    module, config, _ = _load_lightning_module(
        EmbeddingLightningModule,
        model_path,
        model_type="embedding",
        device=device,
    )
    module.eval()

    crop_size = int(config.data_config.preprocessing.crop_size)
    max_stride = int(
        OmegaConf.select(
            config,
            f"model_config.backbone_config.{module.backbone_type}.max_stride",
            default=32,
        )
    )
    labels = sio.load_slp(data_path)
    class_names = resolve_embedding_class_names([labels])
    if not class_names:
        raise ValueError(f"No tracked masks found in {data_path} to embed.")

    emb_head = config.model_config.head_configs.embedding.embedding
    dataset = EmbeddingDataset(
        labels=[labels],
        crop_size=crop_size,
        class_names=class_names,
        embedding_head_config=emb_head,
        max_stride=max_stride,
        ensure_grayscale=True,
        cache_img=None,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb, vids, frames, tracks = [], [], [], []
    # per-(video, frame) running detection ordinal
    det_counter: dict = {}
    dets = []
    for batch in loader:
        gray = torch.squeeze(batch["instance_image"], dim=1).to(device).float()
        mask = torch.squeeze(batch["instance_mask"], dim=1).to(device).float()
        x = module._build_input(gray, mask)
        e = module.model(x)["EmbeddingHead"].cpu().numpy()
        all_emb.append(e)
        for i in range(e.shape[0]):
            v = int(batch["video_idx"][i])
            f = int(batch["frame_idx"][i])
            g = int(batch["group_id"][i])
            vids.append(v)
            frames.append(f)
            tracks.append(class_names[g])
            key = (v, f)
            dets.append(det_counter.get(key, 0))
            det_counter[key] = det_counter.get(key, 0) + 1

    emb = np.concatenate(all_emb).astype(np.float32)
    out = output_path or f"{data_path}.embeddings.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("embeddings", data=emb, compression="gzip")
        h.create_dataset("video", data=np.array(vids, np.int64))
        h.create_dataset("frame", data=np.array(frames, np.int64))
        h.create_dataset("detection", data=np.array(dets, np.int64))
        h.create_dataset(
            "track",
            data=np.array(tracks, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        h.attrs["embedding_dim"] = emb.shape[1]
        h.attrs["normalize"] = bool(emb_head.normalize)
        h.attrs["n"] = emb.shape[0]
    logger.info(f"Wrote {emb.shape[0]} embeddings (dim={emb.shape[1]}) to {out}")
    return out
