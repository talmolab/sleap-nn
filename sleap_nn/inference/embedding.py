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
  vectors in RAM. NOTE: ``detection`` is a per-``(video, frame)`` running
  ordinal over the *emitted* (tracked / valid) crops only — NOT the positional
  ``lf.instances`` / ``lf.masks`` index. Join an embedding back to its source on the
  ``(video, frame, track)`` key (or ``(video, frame, detection)`` within this file),
  not by positional index against the labels. The forward routes through the native-framework
  :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`, so the crop
  pipeline (grayscale + mask burn-in + per-crop standardize) is IDENTICAL to
  training and the embeddings are consistent with the validation retrieval
  metrics.

Reachable from ``sleap-nn predict --embeddings_path <out.h5>`` (the CLI routes
embedding models here); the pose-packaging ``sleap_nn.inference.run.predict`` flow
rejects embedding models and points back to this function.
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
    ensure_rgb: bool = False


@torch.inference_mode()
def predict_embeddings_to_h5(
    model_paths,
    data_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 64,
    peak_threshold: Optional[float] = None,
    save_embeddings: Optional[str] = None,
) -> str:
    """Embed every tracked mask crop in ``data_path`` and STREAM them to ``.h5``.

    The forward routes through the native :class:`Predictor` /
    :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`; crops are enumerated
    per tracked mask and built by the SAME ``EmbeddingDataset`` as training, so the
    crop pipeline (centering per ``crop_centering``, grayscale-by-default but
    RGB-capable via ``ensure_rgb``, mask burn-in + ``background_fill``) matches the
    trained model's config exactly. Embeddings + index arrays are appended to resizable
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
        save_embeddings: Optional ``.slp`` persistence of the appearance vectors,
            one of ``None`` / ``"none"`` / ``"slp"`` / ``"both"`` (default
            ``None``, OFF). When OFF the output is byte-identical to today: the
            vectors stream to ``.h5`` only and NO ``.slp`` is written. ``"slp"``
            attaches each crop's vector to its **source detection** (the exact
            ``sio.Instance`` / ``sio.SegmentationMask``) via
            :meth:`sio.Instance.set_embedding` (``name="reid"``) and writes a sibling
            ``.slp`` (no ``.h5``). ``"both"`` writes the ``.h5`` AND the ``.slp``.
            Pose detections (``Instance``) persist now; mask detections
            (``SegmentationMask``) attach in-memory but ``save_slp`` warns + drops the
            vectors until sleap-io#525 lands ``owner_type=3`` (the ``.h5`` still
            carries them) — this never crashes. Only honored by the single-stage
            mask-driven path; the centroid-driven stream warns and writes ``.h5`` only
            (predicted centroids have no source detection to attach to).

    Returns:
        The output path: the ``.h5`` path when one is written (OFF / ``"both"``),
        else the ``.slp`` path (``"slp"``).
    """
    import contextlib
    from pathlib import Path

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

    # Normalize the optional .slp persistence mode (A1). ``None``/``"none"`` keeps
    # today's behavior (stream to .h5 only, no .slp). ``"slp"`` attaches each vector
    # to its source detection and writes a .slp; ``"both"`` writes .h5 AND .slp.
    if save_embeddings is None:
        save_embeddings = "none"
    save_embeddings = str(save_embeddings).lower()
    if save_embeddings not in ("none", "slp", "both"):
        raise ValueError(
            "save_embeddings must be one of None|'none'|'slp'|'both', got "
            f"{save_embeddings!r}."
        )

    # A centroid + embedding pair composes the centroid -> crop -> embed path on the
    # RAW video (no masks needed): stream embeddings that ride on the predicted
    # centroids. A lone embedding dir is the single-stage, mask-driven case below.
    model_types = [
        get_model_type_from_cfg(config=_load_training_config(d)[0]) for d in model_dirs
    ]
    if "centroid" in model_types:
        if save_embeddings != "none":
            logger.warning(
                "--save_embeddings is not supported for the centroid-driven embedding "
                "stream (predicted centroids have no source detection to attach to); "
                "writing the .h5 only."
            )
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

    from sleap_nn.inference.loaders import _resolve_embedding_channels

    emb_ensure_rgb, emb_ensure_grayscale = _resolve_embedding_channels(config)
    crop_centering = OmegaConf.select(
        config, "data_config.preprocessing.crop_centering", default="auto"
    )
    dataset = EmbeddingDataset(
        labels=[labels],
        crop_size=crop_size,
        class_names=class_names,
        embedding_head_config=emb_head,
        max_stride=max_stride,
        crop_centering=crop_centering,
        ensure_rgb=emb_ensure_rgb,
        ensure_grayscale=emb_ensure_grayscale,
        cache_img=None,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    out = output_path or f"{data_path}.embeddings.h5"
    str_dt = h5py.string_dtype("utf-8")
    embedding_dim = int(emb_head.embedding_dim)
    normalize_flag = bool(emb_head.normalize)

    # A1 persistence: ``"none"`` -> .h5 only (today); ``"slp"`` -> attach + .slp only;
    # ``"both"`` -> .h5 + .slp. The .slp rides on the SAME source detection objects the
    # dataset indexed (``mask_idx_list[item_id]["mask_obj"]``), so attaching mutates the
    # in-memory ``labels`` that gets saved.
    write_h5 = save_embeddings in ("none", "both")
    attach_slp = save_embeddings in ("slp", "both")
    # The model dir name tags every vector's ``Embedding.source`` (provenance).
    model_id = Path(str(model_dir)).name
    # Canonical GT identities, deduped by track name across the file (pose-only:
    # ``SegmentationMask`` carries no ``identity`` attr -> naturally skipped). Seeded
    # from any identities already on the file so we reuse the catalog's objects.
    ident_by_name: dict = {
        i.name: i for i in (getattr(labels, "identities", None) or [])
    }
    # Only the identities minted here (never the pre-existing catalog) are appended
    # at save time, so a file already carrying same-named identities is not collapsed
    # by the name-keyed reuse map (``Identity.name`` is not unique).
    new_identities: list = []

    # Sibling .slp path: strip a known suffix off ``out`` and append ``.slp``.
    slp_stem = out
    for suffix in (".h5", ".slp"):
        if slp_stem.endswith(suffix):
            slp_stem = slp_stem[: -len(suffix)]
            break
    slp_out = slp_stem + ".slp"

    # Per-(video, frame) running detection ordinal, stable across batches. The stream is
    # frame-ordered (shuffle=False), so a frame's entry is evicted once the key changes
    # (``prev_key``) -> det_counter stays O(1), matching the documented O(batch) RAM.
    det_counter: dict = {}
    prev_key = None
    n_written = 0
    n_attached = 0

    h5_file = h5py.File(out, "w") if write_h5 else contextlib.nullcontext()
    with h5_file as h:
        if write_h5:
            emb_ds = h.create_dataset(
                "embeddings",
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
                chunks=(min(batch_size, 256), embedding_dim),
                compression="gzip",
            )
            vid_ds = h.create_dataset(
                "video", shape=(0,), maxshape=(None,), dtype=np.int64
            )
            frame_ds = h.create_dataset(
                "frame", shape=(0,), maxshape=(None,), dtype=np.int64
            )
            det_ds = h.create_dataset(
                "detection", shape=(0,), maxshape=(None,), dtype=np.int64
            )
            track_ds = h.create_dataset(
                "track", shape=(0,), maxshape=(None,), dtype=str_dt
            )

        for batch in loader:
            # EmbeddingDataset yields (b, 1, C, H, W); drop the n_samples axis to
            # (b, C, H, W) the same way the training/val steps do.
            crops = torch.squeeze(batch["instance_image"], dim=1)
            masks = torch.squeeze(batch["instance_mask"], dim=1)
            emb = layer.predict(crops, masks=masks).pred_embeddings
            emb = emb.squeeze(1).detach().cpu().numpy().astype(np.float32)  # (b, D)
            b = emb.shape[0]

            if attach_slp:
                # Map each emitted crop back to its OBJECT-EXACT source detection and
                # attach the vector. ``item_id`` is the dataset index carried per sample.
                for i in range(b):
                    item_id = int(batch["item_id"][i])
                    mask_obj = dataset.mask_idx_list[item_id]["mask_obj"]
                    mask_obj.set_embedding(
                        emb[i],
                        name="reid",
                        normalized=normalize_flag,
                        source=model_id,
                    )
                    n_attached += 1
                    # Optional: promote the source detection's GT track to a canonical
                    # GT ``sio.Identity`` (identity_score=None). Pose-only: a
                    # ``SegmentationMask`` has no ``identity`` attr, so masks skip this.
                    track = getattr(mask_obj, "track", None)
                    if track is not None and hasattr(mask_obj, "identity"):
                        name = str(track.name)
                        ident = ident_by_name.get(name)
                        if ident is None:
                            ident = sio.Identity(name=name)
                            ident_by_name[name] = ident
                            new_identities.append(ident)
                        mask_obj.identity = ident

            if not write_h5:
                continue

            vids: List[int] = []
            frames: List[int] = []
            dets: List[int] = []
            tracks: List[str] = []
            for i in range(b):
                v = int(batch["video_idx"][i])
                f = int(batch["frame_idx"][i])
                g = int(batch["group_id"][i])
                key = (v, f)
                if prev_key is not None and key != prev_key:
                    det_counter.pop(prev_key, None)  # completed frame (frame-ordered)
                d = det_counter.get(key, 0)
                det_counter[key] = d + 1
                prev_key = key
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

        if write_h5:
            h.attrs["embedding_dim"] = embedding_dim
            h.attrs["normalize"] = normalize_flag
            h.attrs["n"] = n_written

    if write_h5:
        logger.info(f"Wrote {n_written} embeddings (dim={embedding_dim}) to {out}")

    if attach_slp:
        # Register the canonical GT identities before saving so the producer-side
        # ``identity in labels.identities`` check passes (mirrors sleap-io#525 M1).
        # Append only the newly-minted identities to the existing catalog (never
        # overwrite it) so pre-existing same-named identities are preserved.
        if new_identities:
            labels.identities = (
                list(getattr(labels, "identities", None) or []) + new_identities
            )
        sio.save_slp(labels, slp_out, embed=False)
        # NOTE: mask detections (SegmentationMask, owner_type=3) warn + drop their
        # vectors here until sleap-io#525 lands; pose detections (Instance) persist.
        logger.info(
            f"Attached {n_attached} embeddings (dim={embedding_dim}) and wrote {slp_out}"
        )
        if not write_h5:
            return slp_out

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
    # The centroid-driven path crops from RAW frames, so there is no instance mask to
    # burn in: a mask-burn-in model is run with a whole-crop standardize, which diverges
    # from its (masked, foreground-only) training/native standardize and degrades the
    # embeddings. Warn so the divergence is not silent.
    emb_module = getattr(layer.centered_instance_layer, "embedding_module", None)
    if getattr(emb_module, "burn_in", False):
        logger.warning(
            "This embedding model was trained with mask burn-in, but the centroid-driven "
            "stream crops from raw frames (no masks): embeddings use a whole-crop "
            "standardize and will diverge from the masked training standardize. For exact "
            "parity, embed a mask-bearing .slp via the mask-driven path (pass only the "
            "embedding model dir, no centroid)."
        )
    embedding_dim = int(layer.centered_instance_layer.embedding_dim)

    out = output_path or f"{data_path}.embeddings.h5"
    # Frame-ordered stream: evict a frame's ordinal once the key changes (see the
    # mask-driven counterpart above) so det_counter stays O(1).
    det_counter: dict = {}
    prev_key = None
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
                    if prev_key is not None and key != prev_key:
                        det_counter.pop(prev_key, None)  # completed frame
                    d = det_counter.get(key, 0)
                    det_counter[key] = d + 1
                    prev_key = key
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
