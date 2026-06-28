"""Group-aware train/val splitter, decided before training.

For the `embedding` (crop -> vector re-ID) model type, the train/val partition *is* the
generalization axis: the model must only ever see the training partition, with val/test
held out by a group key so there is no leakage of the held-out group into training.

Three group keys are supported (mirroring the standalone reference
`scratch/.../embedding/splits.py`):

- ``frame``    : stratified-random over ``LabeledFrame`` units (frames mixed, identity-balanced
                 via :class:`~sklearn.model_selection.StratifiedGroupKFold` grouped by frame).
                 Both train and val contain all identities -- the in-distribution headline.
- ``video``    : hold out whole videos (by sio video index) via
                 :class:`~sklearn.model_selection.GroupKFold` -- the honest cross-session
                 number.
- ``identity`` : hold out whole track names (true open-vocab) via
                 :class:`~sklearn.model_selection.GroupKFold` grouped by identity. Train and
                 val identity sets are disjoint (degenerate for few identities; supported for
                 completeness / verification-only).

The splitter operates at the ``LabeledFrame`` / detection level and returns *new*
``sio.Labels`` objects containing the selected detections, so it composes directly with the
existing dataset builders (which iterate ``sio.Labels`` -> detections). For ``frame``/``video``
each source frame stays whole on one side; for ``identity`` a source frame's detections are
filtered by identity, so a frame may contribute (disjoint) detections to both sides.

Mask-only labels (e.g. the gerbil instance-segmentation data) carry detections on
``lf.masks`` with ``lf.instances`` empty; identity grouping reads track names from
``lf.masks`` in that case.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import sleap_io as sio
from loguru import logger
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
)

# Sentinel used for detections with no track (so they form their own group / class
# instead of breaking sklearn's hashing of mixed None/str arrays).
_NO_TRACK = "__no_track__"


def _frame_detections(lf: sio.LabeledFrame) -> list:
    """Return the list of detection objects for a frame.

    Prefers ``lf.instances`` (pose/keypoint labels); falls back to ``lf.masks`` for
    mask-only (instance-segmentation) labels where ``lf.instances`` is empty.
    """
    instances = list(getattr(lf, "instances", None) or [])
    if instances:
        return instances
    masks = list(getattr(lf, "masks", None) or [])
    return masks


def _track_name(det) -> str:
    """Track name of a detection (mask or instance), or the no-track sentinel."""
    track = getattr(det, "track", None)
    if track is None:
        return _NO_TRACK
    name = getattr(track, "name", None)
    return name if name is not None else _NO_TRACK


def _build_pool(
    labels: sio.Labels,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a ``sio.Labels`` into per-detection arrays.

    Returns:
        Tuple of ``(lf_index, det_index, track_name, video_index)`` arrays, one entry per
        detection. ``lf_index`` indexes ``labels.labeled_frames``; ``det_index`` indexes
        within that frame's detection list (from :func:`_frame_detections`).
    """
    lf_idx, det_idx, track_names, video_idx = [], [], [], []
    for li, lf in enumerate(labels):
        vid = labels.videos.index(lf.video)
        for di, det in enumerate(_frame_detections(lf)):
            lf_idx.append(li)
            det_idx.append(di)
            track_names.append(_track_name(det))
            video_idx.append(vid)
    return (
        np.asarray(lf_idx, dtype=int),
        np.asarray(det_idx, dtype=int),
        np.asarray(track_names, dtype=object),
        np.asarray(video_idx, dtype=int),
    )


def _select_val_fold(
    n_detections: int,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    split_by: str,
    n_folds: int,
    fold: int,
    seed: int,
) -> np.ndarray:
    """Return the boolean mask (length ``n_detections``) of detections in the val fold.

    Uses the sklearn cross-validator appropriate for ``split_by`` and returns the
    ``fold``-th test partition as validation; everything else is training.
    """
    idx = np.arange(n_detections)

    if split_by == "frame":
        # Whole frames kept together (group=frame), identity-balanced across folds.
        n_classes = len(np.unique(y))
        n_groups = len(np.unique(groups))
        effective = min(n_folds, n_classes, n_groups)
        if effective < 2:
            # Can't stratify by identity (e.g. a single identity class), but still keep
            # frames WHOLE on one side via a frame-grouped split (a plain StratifiedKFold
            # would leak a frame's detections across train and val).
            effective = min(n_folds, n_groups)
            if effective < 2:
                message = (
                    "split_by='frame' requires at least 2 frames to hold one out, but "
                    f"found {n_groups}. Provide separate val_labels."
                )
                logger.error(message)
                raise ValueError(message)
            splitter = GroupKFold(n_splits=effective, shuffle=True, random_state=seed)
            splits = list(splitter.split(idx, y, groups=groups))
        else:
            splitter = StratifiedGroupKFold(
                n_splits=effective, shuffle=True, random_state=seed
            )
            splits = list(splitter.split(idx, y, groups=groups))
    elif split_by == "video":
        n_groups = len(np.unique(groups))
        effective = min(n_folds, n_groups)
        if effective < 2:
            message = (
                "split_by='video' requires at least 2 videos to hold one out, but found "
                f"{n_groups}. Provide separate val_labels or use split_by='frame'."
            )
            logger.error(message)
            raise ValueError(message)
        splitter = GroupKFold(n_splits=effective, shuffle=True, random_state=seed)
        splits = list(splitter.split(idx, y, groups=groups))
    elif split_by == "identity":
        n_groups = len(np.unique(groups))
        effective = min(n_folds, n_groups)
        if effective < 2:
            message = (
                "split_by='identity' requires at least 2 distinct track names to hold one "
                f"out, but found {n_groups}. Provide separate val_labels or use "
                "split_by='frame'."
            )
            logger.error(message)
            raise ValueError(message)
        splitter = GroupKFold(n_splits=effective, shuffle=True, random_state=seed)
        splits = list(splitter.split(idx, y, groups=groups))
    else:
        message = (
            f"Unknown split_by={split_by!r}; expected one of "
            "'frame' | 'video' | 'identity'."
        )
        logger.error(message)
        raise ValueError(message)

    # Surface silent reductions/wraps so a misconfigured fold/n_folds is visible.
    if len(splits) < n_folds:
        logger.warning(
            f"Group-aware split: only {len(splits)} fold(s) possible (requested "
            f"n_folds={n_folds}); the val partition is ~1/{len(splits)} of the data."
        )
    if int(fold) >= len(splits):
        logger.warning(
            f"Group-aware split: fold={fold} is out of range for {len(splits)} fold(s); "
            f"wrapping to fold {int(fold) % len(splits)}."
        )
    fold = int(fold) % len(splits)
    _, val_det_idx = splits[fold]
    mask = np.zeros(n_detections, dtype=bool)
    mask[val_det_idx] = True
    return mask


def _rebuild_labels(
    source: sio.Labels,
    lf_idx: np.ndarray,
    det_idx: np.ndarray,
    keep_mask: np.ndarray,
    extra_lf_idx: Optional[List[int]] = None,
) -> sio.Labels:
    """Build a new ``sio.Labels`` from the kept detections.

    Groups kept detections by source frame and constructs a new ``LabeledFrame`` per source
    frame containing only the selected detections (preserving video / frame_idx). Detections
    are attached on the same attribute (``instances`` or ``masks``) they came from.

    ``extra_lf_idx`` lists source-frame indices to carry through with NO detections (e.g.
    user-confirmed negative frames, which have no identity to fold) — emitted as empty
    ``LabeledFrame``s preserving ``is_negative``.
    """
    new_frames = []
    # Group kept detection rows by their source frame index, preserving frame order.
    kept_by_lf: dict = {}
    for li, di, keep in zip(lf_idx, det_idx, keep_mask):
        if not keep:
            continue
        kept_by_lf.setdefault(int(li), []).append(int(di))
    for li in extra_lf_idx or []:
        kept_by_lf.setdefault(int(li), [])  # carried negative/empty frame

    for li in sorted(kept_by_lf):
        src_lf = source.labeled_frames[li]
        dets = _frame_detections(src_lf)
        selected = [dets[di] for di in kept_by_lf[li]]
        # Re-attach on instances vs masks matching the source frame.
        use_masks = not (getattr(src_lf, "instances", None) or [])
        if use_masks:
            new_lf = sio.LabeledFrame(
                video=src_lf.video,
                frame_idx=src_lf.frame_idx,
                masks=selected,
                is_negative=src_lf.is_negative,
            )
        else:
            new_lf = sio.LabeledFrame(
                video=src_lf.video,
                frame_idx=src_lf.frame_idx,
                instances=selected,
                is_negative=src_lf.is_negative,
            )
        new_frames.append(new_lf)

    return sio.Labels(
        labeled_frames=new_frames,
        videos=list(source.videos),
        skeletons=list(source.skeletons),
        tracks=list(source.tracks),
    )


def split_labels_train_val(
    source: sio.Labels,
    *,
    split_by: str,
    n_folds: int,
    fold: int,
    seed: int,
) -> Tuple[sio.Labels, sio.Labels]:
    """Partition a single ``sio.Labels`` into (train, val) by a group key.

    Args:
        source: The ``sio.Labels`` to partition.
        split_by: One of ``'frame'`` | ``'video'`` | ``'identity'`` (see module docstring).
        n_folds: Number of CV folds; the val partition is ``1 / n_folds`` of the data.
        fold: Which fold (0-based) to hold out as validation.
        seed: Random seed for the (shuffled) splitter.

    Returns:
        Tuple of new ``sio.Labels`` ``(train_labels, val_labels)`` with no group leakage.
    """
    lf_idx, det_idx, track_names, video_idx = _build_pool(source)
    n = len(lf_idx)

    # Frames with zero detections (e.g. user-confirmed negatives) have no identity to
    # fold, so they would otherwise be silently dropped from BOTH sides. Carry them on
    # the train side (deterministic) — mirroring the non-split path's negative handling.
    frames_with_dets = set(int(i) for i in lf_idx)
    negative_lf_idx = [
        li for li in range(len(source.labeled_frames)) if li not in frames_with_dets
    ]

    if n == 0:
        # No detections at all (e.g. an all-negative or empty .slp). Don't abort the
        # run — keep any frames on the train side, leave val empty.
        if negative_lf_idx:
            logger.warning(
                "Group-aware split: a labels file has no detections; keeping its "
                f"{len(negative_lf_idx)} frame(s) on the train side (val empty)."
            )
        empty = np.zeros(0, dtype=bool)
        train_labels = _rebuild_labels(
            source, lf_idx, det_idx, empty, extra_lf_idx=negative_lf_idx
        )
        val_labels = _rebuild_labels(source, lf_idx, det_idx, empty)
        return train_labels, val_labels

    # Identity label per detection (used as stratification target and as identity group).
    y = track_names
    if split_by == "video":
        groups = video_idx
    elif split_by == "identity":
        groups = track_names
    else:  # frame
        groups = lf_idx

    val_mask = _select_val_fold(
        n,
        y,
        groups,
        split_by=split_by,
        n_folds=n_folds,
        fold=fold,
        seed=seed,
    )

    if negative_lf_idx:
        logger.info(
            f"Group-aware split: carried {len(negative_lf_idx)} zero-detection "
            "frame(s) to the train side."
        )
    train_labels = _rebuild_labels(
        source, lf_idx, det_idx, ~val_mask, extra_lf_idx=negative_lf_idx
    )
    val_labels = _rebuild_labels(source, lf_idx, det_idx, val_mask)
    return train_labels, val_labels


def split_labels_list_train_val(
    labels_list: List[sio.Labels],
    split_config,
) -> Tuple[List[sio.Labels], List[sio.Labels]]:
    """Apply :func:`split_labels_train_val` to each ``sio.Labels`` in a list.

    Splits each input ``sio.Labels`` independently (video indices and track names are scoped
    per file) and returns ``(train_labels_list, val_labels_list)`` aligned to the input list,
    so the result drops straight into ``ModelTrainer.train_labels`` / ``.val_labels``.

    Args:
        labels_list: List of source ``sio.Labels`` (typically the loaded train files).
        split_config: A ``SplitConfig`` (or any object exposing ``split_by`` / ``n_folds`` /
            ``fold`` / ``seed``).

    Returns:
        Tuple ``(train_labels_list, val_labels_list)`` of equal length to ``labels_list``.
    """
    split_by = getattr(split_config, "split_by", "frame")
    n_folds = int(getattr(split_config, "n_folds", 5))
    fold = int(getattr(split_config, "fold", 0))
    seed = int(getattr(split_config, "seed", 0))

    train_list, val_list = [], []
    for labels in labels_list:
        train_labels, val_labels = split_labels_train_val(
            labels,
            split_by=split_by,
            n_folds=n_folds,
            fold=fold,
            seed=seed,
        )
        train_list.append(train_labels)
        val_list.append(val_labels)

    logger.info(
        f"Group-aware split (split_by={split_by!r}, n_folds={n_folds}, fold={fold}, "
        f"seed={seed}): "
        f"train frames={sum(len(t) for t in train_list)}, "
        f"val frames={sum(len(v) for v in val_list)}."
    )
    return train_list, val_list
