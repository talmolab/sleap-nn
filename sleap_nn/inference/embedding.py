"""Inference for the ``embedding`` model type: crops -> appearance vectors.

This module hosts:

* :class:`EmbeddingInferenceModel` — the lightweight ``LoadedAssets.inference_model``
  holder that :func:`sleap_nn.inference.loaders.load_model_assets` builds for an
  ``embedding`` model and that :func:`sleap_nn.inference.predictor._build_embedding_layer`
  consumes.
* :func:`embed_labels` — embed every detection of an in-memory ``sio.Labels`` IN
  PLACE (attach an ``sio.Embedding`` ``"reid"`` vector to each source detection) and
  return the vectors + per-detection track names. The forward routes through the
  native-framework :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`, so the
  crop pipeline (grayscale + optional mask burn-in + per-crop standardize) is IDENTICAL
  to training and the embeddings are consistent with the validation retrieval metrics.
  Shared by the ``.slp`` writer below and the post-training retrieval eval in
  :mod:`sleap_nn.train`.
* :func:`predict_embeddings_to_slp` — the re-ID entry point: embed every detection in a
  ``.slp`` and persist the vectors via the sleap-io ``sio.Embedding`` data model back
  into a ``.slp``. With a ``tracker_config`` (WF2: "embed + track on the
  fly") every detection — tracked OR untracked — is embedded and
  :func:`~sleap_nn.inference.tracking.apply_tracking` assigns ``sio.Track``s by cosine
  similarity. Join an embedding back to its source on its host detection
  (``sio.Instance`` / ``sio.SegmentationMask``).

Reachable from ``sleap-nn predict`` (the CLI routes embedding models here); the
pose-packaging :func:`sleap_nn.inference.run.predict` flow rejects embedding models and
points back to this function. A fused ``-m centroid [-m centered_instance] -m
<embedding> -i <video> --tracking`` command first runs the detection stack to a
temporary ``.slp`` and then embeds + tracks it (see ``sleap_nn.cli._run_embeddings``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import attrs
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
import sleap_io as sio

if TYPE_CHECKING:
    from sleap_nn.inference.tracking import TrackerConfig


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
def embed_labels(
    model_dir,
    labels: sio.Labels,
    *,
    device: str = "cuda",
    batch_size: int = 64,
    include_untracked: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Embed every detection in ``labels`` IN PLACE and return the vectors.

    Builds the SAME :class:`~sleap_nn.data.custom_datasets.EmbeddingDataset` as
    training (centering per ``crop_centering``, grayscale-by-default but RGB-capable via
    ``ensure_rgb``, optional mask burn-in + ``background_fill``), runs the native
    :class:`~sleap_nn.inference.layers.embedding.EmbeddingLayer`, and attaches each
    crop's vector to its **object-exact** source detection (the exact ``sio.Instance`` /
    ``sio.SegmentationMask``) via the single ``identity_embedding`` slot (sleap-io #535).
    Both pose and mask (``owner_type=3``) detections persist their embeddings.

    This is the shared embedding kernel: :func:`predict_embeddings_to_slp` calls it then
    writes a ``.slp``; the post-training retrieval eval calls it for the ``(vectors,
    track names)`` arrays without writing anything.

    Args:
        model_dir: Trained ``embedding`` model directory (``best.ckpt`` +
            ``training_config.yaml`` resolved).
        labels: ``sio.Labels`` to embed; mutated in place (vectors attached to each
            detection).
        device: Torch device.
        batch_size: Crops per forward pass.
        include_untracked: When ``False`` (default) only tracked detections are
            embedded (track names are the identities, e.g. for retrieval eval). When
            ``True`` every detection is embedded regardless of track (WF2 tracking).

    Returns:
        ``(embeddings, track_names, n_attached, embedding_dim)`` where ``embeddings`` is
        ``(N, D) float32``, ``track_names`` is ``(N,)`` object (the per-detection class /
        track name; empty strings for untracked detections), ``n_attached`` is ``N``, and
        ``embedding_dim`` is ``D``.
    """
    from sleap_nn.config.utils import resolve_model_dir
    from sleap_nn.data.custom_datasets import (
        EmbeddingDataset,
        resolve_embedding_class_names,
    )
    from sleap_nn.inference.loaders import (
        _load_training_config,
        _resolve_embedding_channels,
    )
    from sleap_nn.inference.predictor import Predictor

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    model_dir = resolve_model_dir(model_dir)
    config, _ = _load_training_config(model_dir)

    # Build the native Predictor; its layer is the EmbeddingLayer (mask/pose-driven,
    # single-stage). The forward goes through this layer so it matches training.
    predictor = Predictor.from_model_paths(
        [str(model_dir)], device=device, batch_size=batch_size
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
    embedding_dim = int(emb_head.embedding_dim)

    class_names = resolve_embedding_class_names([labels])
    if not class_names and not include_untracked:
        raise ValueError("No tracked detections found to embed.")

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
        include_untracked=include_untracked,
        ensure_rgb=emb_ensure_rgb,
        ensure_grayscale=emb_ensure_grayscale,
        cache_img=None,
    )
    if len(dataset) == 0:
        return (
            np.zeros((0, embedding_dim), np.float32),
            np.zeros((0,), dtype=object),
            0,
            embedding_dim,
        )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb = []
    all_tracks = []
    n_attached = 0
    for batch in loader:
        # EmbeddingDataset yields (b, 1, C, H, W); drop the n_samples axis to
        # (b, C, H, W) the same way the training/val steps do.
        crops = torch.squeeze(batch["instance_image"], dim=1)
        masks = torch.squeeze(batch["instance_mask"], dim=1)
        emb = layer.predict(crops, masks=masks).pred_embeddings
        emb = emb.squeeze(1).detach().cpu().numpy().astype(np.float32)  # (b, D)
        b = emb.shape[0]
        for i in range(b):
            # Map each emitted crop back to its OBJECT-EXACT source detection and
            # attach the vector. ``item_id`` is the dataset index carried per sample.
            item_id = int(batch["item_id"][i])
            mask_obj = dataset.mask_idx_list[item_id]["mask_obj"]
            # Single re-ID slot on every detection modality (sleap-io #535): the vector
            # is implicitly the appearance embedding; provenance / normalized-flag /
            # space-name no longer have a home on the bare `Embedding` value object.
            mask_obj.identity_embedding = sio.Embedding(emb[i])
            g = int(batch["group_id"][i])
            all_tracks.append(
                class_names[g] if class_names and 0 <= g < len(class_names) else ""
            )
            n_attached += 1
        all_emb.append(emb)

    embeddings = (
        np.concatenate(all_emb, axis=0)
        if all_emb
        else np.zeros((0, embedding_dim), np.float32)
    )
    track_names = np.array(all_tracks, dtype=object)
    return embeddings, track_names, n_attached, embedding_dim


@torch.inference_mode()
def predict_embeddings_to_slp(
    model_paths,
    data_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 64,
    save_embeddings: str = "slp",
    tracker_config: Optional["TrackerConfig"] = None,  # noqa: F821
    include_untracked: Optional[bool] = None,
) -> str:
    """Embed every detection in ``data_path`` and persist the vectors into a ``.slp``.

    The appearance vectors persist via the sleap-io ``sio.Embedding`` data model
    (``name="reid"``, attached to each detection's host ``sio.Instance`` /
    ``sio.SegmentationMask``). The crop pipeline matches the trained model's config
    exactly (see :func:`embed_labels`).

    Args:
        model_paths: Trained ``embedding`` model directory (or a list; the embedding
            model is selected and any stray non-embedding dirs are ignored — the CLI
            runs the detection stack separately for the fused path).
        data_path: ``.slp`` file with the detections to embed.
        output_path: Output ``.slp`` path. Defaults to ``<data_path>.tracked.slp`` when
            tracking, else ``<data_path>.embeddings.slp``.
        device: Torch device.
        batch_size: Crops per forward pass.
        save_embeddings: ``"slp"`` persists the appearance vectors in the output
            ``.slp`` (attached to each source detection). ``"none"`` does not persist
            them; with ``tracker_config`` set, the vectors are still attached so the
            tracker can consume them, then stripped before the tracked ``.slp`` is
            written (tracks only).
        tracker_config: Optional :class:`~sleap_nn.inference.tracking.TrackerConfig`.
            When set (WF2), every detection — tracked OR untracked — is embedded and
            :func:`~sleap_nn.inference.tracking.apply_tracking` assigns ``sio.Track``s by
            appearance (``features="embeddings"`` / ``scoring_method="cosine_sim"``); the
            tracked ``.slp`` is written. Emits ``sio.Track`` only (no global
            ``sio.Identity`` — a track name is not a global animal identity).
        include_untracked: Whether to embed untracked detections too. ``None``
            (default) derives it from ``tracker_config`` (tracking embeds everything;
            non-tracking embeds only tracked detections, keyed by identity). The fused
            detect→embed path sets ``True`` because freshly-detected instances carry no
            tracks yet.

    Returns:
        The output ``.slp`` path.
    """
    from sleap_nn.config.utils import get_model_type_from_cfg, resolve_model_dir
    from sleap_nn.inference.loaders import _load_training_config

    if isinstance(model_paths, (str, bytes)):
        model_paths = [model_paths]
    model_dirs = [resolve_model_dir(m) for m in model_paths]

    save_embeddings = (save_embeddings or "none").lower()
    if save_embeddings not in ("none", "slp"):
        raise ValueError(
            f"save_embeddings must be one of 'none'|'slp', got {save_embeddings!r}."
        )

    # Select the embedding model dir (the detection stack, if any, is run by the CLI
    # before this point for the fused path).
    model_types = [
        get_model_type_from_cfg(config=_load_training_config(d)[0]) for d in model_dirs
    ]
    if "embedding" not in model_types:
        raise ValueError(
            "predict_embeddings_to_slp requires an `embedding` model directory."
        )
    model_dir = model_dirs[model_types.index("embedding")]

    tracking = tracker_config is not None
    inc_untracked = include_untracked if include_untracked is not None else tracking

    if not tracking and save_embeddings == "none":
        raise ValueError(
            "Nothing to persist: save_embeddings='none' without tracking. Pass "
            "save_embeddings='slp' to write the vectors into a .slp, or a tracker_config "
            "to track by appearance."
        )

    labels = sio.load_slp(data_path)

    # Tracking (and the fused detect→embed path) embed EVERY detection — tracked or not —
    # so untracked inputs can be tracked / persisted; otherwise only tracked detections
    # are embedded (keyed by identity).
    _, _, n_attached, embedding_dim = embed_labels(
        model_dir,
        labels,
        device=device,
        batch_size=batch_size,
        include_untracked=inc_untracked,
    )
    if n_attached == 0:
        raise ValueError(f"No detections found in {data_path} to embed.")

    if tracking:
        # WF2: track by appearance on the freshly-attached vectors, then write the
        # tracked .slp. apply_tracking re-assigns sio.Track from scratch (any prior
        # tracks are overwritten); it emits Tracks only (no global sio.Identity — a
        # track name is not a global animal identity).
        from sleap_nn.inference.tracking import apply_tracking

        logger.info(
            f"Attached {n_attached} embeddings (dim={embedding_dim}); tracking by "
            "appearance (cosine similarity)."
        )
        tracked = apply_tracking(labels, tracker_config)
        # save_embeddings controls whether the vectors persist in the tracked .slp;
        # "none" -> tracks only, so strip the attached vectors (belt-and-braces with
        # the explicit save_embedding_vectors flag below).
        persist_vectors = save_embeddings == "slp"
        if not persist_vectors:
            _strip_embeddings(tracked)
        tracked_out = output_path or f"{data_path}.tracked.slp"
        # #536 flipped the save_slp default to False; be explicit either way. Identity
        # links (sio.Track) always persist regardless of this flag.
        sio.save_slp(
            tracked, tracked_out, embed=False, save_embedding_vectors=persist_vectors
        )
        logger.info(f"Wrote tracked labels to {tracked_out}")
        return tracked_out

    # Non-tracking: the .slp of attached vectors IS the output. The embedding model
    # produces appearance vectors, not identities. Any ``sio.Identity`` already on the
    # input labels passes through untouched; we do NOT fabricate identities from track
    # names (a track/class name is not a global animal identity). Both Instance and
    # SegmentationMask (owner_type=3, sleap-io#527) embeddings persist here.
    slp_out = output_path or f"{data_path}.embeddings.slp"
    # This path exists to persist the vectors, so opt in explicitly (#536 default flip).
    sio.save_slp(labels, slp_out, embed=False, save_embedding_vectors=True)
    logger.info(
        f"Attached {n_attached} embeddings (dim={embedding_dim}) and wrote {slp_out}"
    )
    return slp_out


def _strip_embeddings(labels: sio.Labels) -> None:
    """Drop the re-ID appearance embedding from a labels' detections in place.

    Used when ``--tracking`` runs on an ``embedding`` model with
    ``save_embeddings="none"`` (the default): the vectors are attached only so the
    tracker can consume them, and are removed before the tracked ``.slp`` is written
    (tracks only). Both pose (``lf.instances``) and mask (``lf.masks``) carriers hold a
    single ``identity_embedding`` slot (sleap-io #535).
    """
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.identity_embedding = None
        for m in getattr(lf, "masks", None) or []:
            m.identity_embedding = None
