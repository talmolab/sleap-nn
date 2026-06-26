"""Inference for the ``embedding`` model type: crops -> appearance vectors.

This module hosts:

* :class:`EmbeddingInferenceModel` — the lightweight ``LoadedAssets.inference_model``
  holder that :func:`sleap_nn.inference.loaders.load_model_assets` builds for an
  ``embedding`` model and that :func:`sleap_nn.inference.predictor._build_embedding_layer`
  consumes.
* :func:`predict_embeddings_to_h5` — the offline re-ID stream: embed every tracked
  mask crop in a ``.slp`` and STREAM the vectors + index arrays
  (``video`` / ``frame`` / ``detection`` / ``track``) incrementally to a simple
  ``.h5`` (resizable datasets), so a long video never holds the whole set of
  vectors in RAM (SPEC M5). The forward routes through the native-framework
  :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`, so the crop
  pipeline (grayscale + mask burn-in + per-crop standardize) is IDENTICAL to
  training and the embeddings are consistent with the validation retrieval
  metrics.

Reachable from ``sleap-nn predict --embeddings_path <out.h5>`` (and the Python
``sleap_nn.inference.run.predict`` flow).
"""

from __future__ import annotations

from typing import List, Optional

import attrs
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
import sleap_io as sio


@attrs.define(eq=False, repr=False)
class EmbeddingInferenceModel:
    """Holder for a trained ``embedding`` model + the knobs its layer needs.

    Mirrors the ``*InferenceModel`` holders carried on ``LoadedAssets`` for the
    other model types; consumed by ``_build_embedding_layer`` to construct an
    :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`.
    """

    torch_model: object  # the EmbeddingLightningModule
    embedding_dim: int
    output_stride: int = 1
    max_stride: int = 1
    input_scale: float = 1.0
    crop_size: Optional[int] = None
    ensure_grayscale: bool = True


@torch.inference_mode()
def predict_embeddings_to_h5(
    model_paths,
    data_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 64,
    peak_threshold: Optional[float] = None,
) -> str:
    """Embed every tracked mask crop in ``data_path`` and STREAM them to ``.h5``.

    The forward routes through the native :class:`Predictor` /
    :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`; crops are
    enumerated per tracked mask (mask-COM centered, grayscale, mask burn-in)
    exactly as in training. Embeddings + index arrays are appended to resizable
    ``.h5`` datasets batch-by-batch (O(batch) RAM, never the whole video).

    Args:
        model_paths: Trained ``embedding`` model directory (or a list with one
            entry; the ``best.ckpt`` + ``training_config.yaml`` are resolved).
        data_path: ``.slp`` file to embed.
        output_path: Output ``.h5`` path. Defaults to ``<data_path>.embeddings.h5``.
        device: Torch device.
        batch_size: Crops per forward pass / write chunk.
        peak_threshold: Centroid peak threshold for the composed centroid + embedding
            path (ignored by the single-stage mask-driven path, which has no centroid
            stage). ``None`` keeps the model's default.

    Returns:
        The output ``.h5`` path.
    """
    import h5py

    from sleap_nn.config.utils import resolve_model_dir
    from sleap_nn.data.custom_datasets import (
        EmbeddingDataset,
        resolve_embedding_class_names,
    )
    from sleap_nn.inference.loaders import _load_training_config
    from sleap_nn.inference.predictor import Predictor

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    from sleap_nn.config.utils import get_model_type_from_cfg

    if isinstance(model_paths, (str, bytes)):
        model_paths = [model_paths]
    model_paths = list(model_paths)
    model_dirs = [resolve_model_dir(m) for m in model_paths]

    # A centroid + embedding pair composes the centroid -> crop -> embed path on the
    # RAW video (no masks needed): stream embeddings that ride on the predicted
    # centroids. A lone embedding dir is the single-stage, mask-driven case below.
    model_types = [
        get_model_type_from_cfg(config=_load_training_config(d)[0]) for d in model_dirs
    ]
    if "centroid" in model_types:
        return _stream_embeddings_centroid_driven(
            model_dirs,
            data_path,
            output_path,
            device=device,
            batch_size=batch_size,
            peak_threshold=peak_threshold,
        )

    model_dir = model_dirs[model_types.index("embedding")]
    config, _ = _load_training_config(model_dir)

    # Build the native Predictor; its layer is the EmbeddingLayer (mask-driven,
    # single-stage). The forward goes through this layer so it matches training.
    predictor = Predictor.from_model_paths(
        [model_dir], device=device, batch_size=batch_size
    )
    layer = predictor.layer

    crop_size = int(config.data_config.preprocessing.crop_size)
    backbone_type = next(
        k for k, v in config.model_config.backbone_config.items() if v is not None
    )
    max_stride = int(
        OmegaConf.select(
            config,
            f"model_config.backbone_config.{backbone_type}.max_stride",
            default=32,
        )
    )
    emb_head = config.model_config.head_configs.embedding.embedding

    labels = sio.load_slp(data_path)
    class_names = resolve_embedding_class_names([labels])
    if not class_names:
        raise ValueError(f"No tracked masks found in {data_path} to embed.")

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

    out = output_path or f"{data_path}.embeddings.h5"
    str_dt = h5py.string_dtype("utf-8")
    embedding_dim = int(emb_head.embedding_dim)
    # Per-(video, frame) running detection ordinal, stable across batches.
    det_counter: dict = {}
    n_written = 0

    with h5py.File(out, "w") as h:
        emb_ds = h.create_dataset(
            "embeddings",
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float32,
            chunks=(min(batch_size, 256), embedding_dim),
            compression="gzip",
        )
        vid_ds = h.create_dataset("video", shape=(0,), maxshape=(None,), dtype=np.int64)
        frame_ds = h.create_dataset(
            "frame", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        det_ds = h.create_dataset(
            "detection", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        track_ds = h.create_dataset("track", shape=(0,), maxshape=(None,), dtype=str_dt)

        for batch in loader:
            # EmbeddingDataset yields (b, 1, C, H, W); drop the n_samples axis to
            # (b, C, H, W) the same way the training/val steps do.
            crops = torch.squeeze(batch["instance_image"], dim=1)
            masks = torch.squeeze(batch["instance_mask"], dim=1)
            emb = layer.predict(crops, masks=masks).pred_embeddings
            emb = emb.squeeze(1).detach().cpu().numpy().astype(np.float32)  # (b, D)
            b = emb.shape[0]

            vids: List[int] = []
            frames: List[int] = []
            dets: List[int] = []
            tracks: List[str] = []
            for i in range(b):
                v = int(batch["video_idx"][i])
                f = int(batch["frame_idx"][i])
                g = int(batch["group_id"][i])
                key = (v, f)
                d = det_counter.get(key, 0)
                det_counter[key] = d + 1
                vids.append(v)
                frames.append(f)
                dets.append(d)
                tracks.append(class_names[g])

            new_n = n_written + b
            emb_ds.resize((new_n, embedding_dim))
            emb_ds[n_written:new_n] = emb
            for ds, vals, dt in (
                (vid_ds, vids, np.int64),
                (frame_ds, frames, np.int64),
                (det_ds, dets, np.int64),
            ):
                ds.resize((new_n,))
                ds[n_written:new_n] = np.array(vals, dt)
            track_ds.resize((new_n,))
            track_ds[n_written:new_n] = np.array(tracks, dtype=object)
            n_written = new_n

        h.attrs["embedding_dim"] = embedding_dim
        h.attrs["normalize"] = bool(emb_head.normalize)
        h.attrs["n"] = n_written

    logger.info(f"Wrote {n_written} embeddings (dim={embedding_dim}) to {out}")
    return out


@torch.inference_mode()
def _stream_embeddings_centroid_driven(
    model_dirs,
    data_path: str,
    output_path: Optional[str],
    *,
    device: str,
    batch_size: int,
    peak_threshold: Optional[float] = None,
) -> str:
    """Stream embeddings for a composed centroid + embedding pair to ``.h5``.

    Runs the composed
    :class:`~sleap_nn.inference.layers.embedding.TopDownEmbeddingLayer` over the RAW
    frames of ``data_path``: the centroid model finds instances, each crop is
    embedded, and the valid embeddings + their centroid coords + ``(video, frame,
    detection)`` index arrays are streamed to a simple ``.h5``. Unlike the
    mask-driven path there is NO ``track`` column — raw-frame inference has no GT
    identity — but a ``centroid_xy (N, 2)`` array records where each embedding rode.

    Args:
        model_dirs: Resolved model directories (one centroid + one embedding).
        data_path: ``.slp``/video source to embed.
        output_path: Output ``.h5`` path (defaults to ``<data_path>.embeddings.h5``).
        device: Torch device.
        batch_size: Frames per forward / write chunk.
        peak_threshold: Centroid peak threshold override for the stage-1 centroid
            model. ``None`` keeps the model's default.

    Returns:
        The output ``.h5`` path.
    """
    import h5py

    from sleap_nn.inference.layers.embedding import TopDownEmbeddingLayer
    from sleap_nn.inference.predictor import Predictor

    predictor = Predictor.from_model_paths(
        list(model_dirs), device=device, batch_size=batch_size
    )
    layer = predictor.layer
    if not isinstance(layer, TopDownEmbeddingLayer):
        raise ValueError(
            "Expected a composed centroid + embedding model for the centroid-driven "
            f"embedding stream (got layer {type(layer).__name__})."
        )
    embedding_dim = int(layer.centered_instance_layer.embedding_dim)

    out = output_path or f"{data_path}.embeddings.h5"
    det_counter: dict = {}
    n_written = 0

    with h5py.File(out, "w") as h:
        emb_ds = h.create_dataset(
            "embeddings",
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float32,
            chunks=(min(batch_size, 256), embedding_dim),
            compression="gzip",
        )
        vid_ds = h.create_dataset("video", shape=(0,), maxshape=(None,), dtype=np.int64)
        frame_ds = h.create_dataset(
            "frame", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        det_ds = h.create_dataset(
            "detection", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        cxy_ds = h.create_dataset(
            "centroid_xy", shape=(0, 2), maxshape=(None, 2), dtype=np.float32
        )

        stream_kwargs = {}
        if peak_threshold is not None:
            stream_kwargs["peak_threshold"] = peak_threshold
        for outputs in predictor.predict_streaming(data_path, **stream_kwargs):
            emb = outputs.pred_embeddings  # (B, I, D)
            if emb is None:
                continue
            emb = emb.detach().cpu().numpy()
            B, I = emb.shape[0], emb.shape[1]

            def _np(x, default):
                return x.detach().cpu().numpy() if x is not None else default

            valid = _np(outputs.instance_valid, np.ones((B, I), dtype=bool))
            cent = _np(
                outputs.pred_centroids, np.full((B, I, 2), np.nan, dtype=np.float32)
            )
            fidx = _np(outputs.frame_indices, np.zeros(B, dtype=np.int64))
            vidx = _np(outputs.video_indices, np.zeros(B, dtype=np.int64))

            embs, cxys, vids, frames, dets = [], [], [], [], []
            for b in range(B):
                v, f = int(vidx[b]), int(fidx[b])
                for i in range(I):
                    if not bool(valid[b, i]):
                        continue
                    key = (v, f)
                    d = det_counter.get(key, 0)
                    det_counter[key] = d + 1
                    embs.append(emb[b, i])
                    cxys.append(cent[b, i])
                    vids.append(v)
                    frames.append(f)
                    dets.append(d)
            if not embs:
                continue

            k = len(embs)
            new_n = n_written + k
            emb_ds.resize((new_n, embedding_dim))
            emb_ds[n_written:new_n] = np.asarray(embs, dtype=np.float32)
            cxy_ds.resize((new_n, 2))
            cxy_ds[n_written:new_n] = np.asarray(cxys, dtype=np.float32)
            for ds, vals in ((vid_ds, vids), (frame_ds, frames), (det_ds, dets)):
                ds.resize((new_n,))
                ds[n_written:new_n] = np.asarray(vals, dtype=np.int64)
            n_written = new_n

        h.attrs["embedding_dim"] = embedding_dim
        h.attrs["n"] = n_written
        h.attrs["centroid_driven"] = True

    logger.info(
        f"Wrote {n_written} embeddings (dim={embedding_dim}, centroid-driven) to {out}"
    )
    return out
