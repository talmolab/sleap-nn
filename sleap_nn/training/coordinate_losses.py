"""Coordinate losses through a frozen linear keypoint autoencoder (experimental).

Configured by ``trainer_config.coordinate_losses`` (`CoordinateLossesConfig`), for
``centered_instance`` models only.

**Differentiable coordinates.** A soft-argmax (integral regression; Sun et al.,
"Integral Human Pose Regression", 2018) turns each predicted confidence map into
the expectation of the pixel coordinates under ``softmax(beta * map)``, in crop
pixels (the confidence-map grid ``i * output_stride`` of
`sleap_nn.data.utils.make_grid_vectors`), optionally only over a window around
each map's maximum. It is computed only when a coordinate loss is configured.

**The autoencoder.** An affine-combining autoencoder (Sárándi, Hermans and Leibe,
"Learning 3D Human Pose Estimation from Dozens of Datasets using a
Geometry-Aware Autoencoder to Bridge Between Skeleton Formats", WACV 2023):
``W_enc`` (``L x J``) maps ``J`` keypoint channels to ``L`` latent points and
``W_dec`` (``J x L``) maps them back, each output point an affine combination of
the inputs (rows sum to 1). It is read from an ``.npz`` with ``W_enc``,
``W_dec``, ``names`` (the ``J`` channel names) and, optionally,
``latent_partner`` (each latent point's left/right partner). It stays frozen and
outside the state dict, so checkpoints load into a plain model.

**The terms** (exactly one is set):

1. ``consistency``: ``|P - W_dec W_enc P|_1`` on the soft-argmax coordinates
   ``P`` of the head's channels, matched to the autoencoder's channels by name.
   Added to the unchanged confidence-map loss.
2. ``decoded``: the head predicts the ``L`` latent points ``Q`` and the loss is
   ``|W_dec Q - Y|_1`` on the channels each sample labels (the batch's crop
   keypoints ``Y`` and per-node weights, mapped to the autoencoder's rows by
   name). It is the whole loss. The saved model predicts the latent points.

Both are L1 distances in crop pixels (``|dx| + |dy|`` per channel), averaged over
the channels that count. They are computed in crop coordinates, after the crop's
rotation, scaling and flips. The map is equivariant to rotation and scaling
(linear in the coordinates), to translation (rows sum to 1), and to flips when
the autoencoder is symmetric under the same left/right swap the flip applies to
the channels; with flip augmentation on, training refuses an autoencoder that is
not (see `setup_coordinate_losses`).

Not built: the paper's hybrid student-teacher variant (two heads on one model
with a teacher term), and decoding at prediction time.
"""

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sleap_io as sio
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

#: Config path of the block.
CONFIG_KEY = "trainer_config.coordinate_losses"
#: The two terms, by their config weight names.
TERMS = ("consistency", "decoded")
#: How far rows may be from summing to 1, and the flip check's tolerance.
AFFINE_TOL = 1e-3
SYMMETRY_TOL = 1e-4


# --------------------------------------------------------------- autoencoder


def latent_names(n_latent: int) -> List[str]:
    """Channel names of the latent points: ``latent_00``, ``latent_01``, ..."""
    width = max(2, len(str(n_latent - 1)))
    return [f"latent_{i:0{width}d}" for i in range(n_latent)]


class KeypointAutoencoder:
    """A frozen linear keypoint autoencoder read from an ``.npz``.

    Attributes:
        w_enc: ``(L, J)`` float32 encoder.
        w_dec: ``(J, L)`` float32 decoder.
        names: The ``J`` channel names.
        latent_partner: ``(L,)`` int left/right partner of each latent point, or
            ``None`` when the file has none.
        path: The file it was read from.
    """

    def __init__(
        self,
        w_enc: np.ndarray,
        w_dec: np.ndarray,
        names: Sequence[str],
        latent_partner: Optional[np.ndarray] = None,
        path: str = "",
    ):
        """Validate the matrices (see `load_keypoint_autoencoder`)."""
        self.path = str(path)
        self.w_enc = np.asarray(w_enc, dtype=np.float32)
        self.w_dec = np.asarray(w_dec, dtype=np.float32)
        self.names = [str(n) for n in names]
        where = f"coordinate_losses.autoencoder {self.path!r}"
        if self.w_enc.ndim != 2 or self.w_dec.ndim != 2:
            raise ValueError(f"{where}: W_enc and W_dec must be 2-D matrices.")
        n_latent, n_channels = self.w_enc.shape
        if self.w_dec.shape != (n_channels, n_latent):
            raise ValueError(
                f"{where}: W_enc is {self.w_enc.shape} (L x J), so W_dec must be "
                f"{(n_channels, n_latent)} (J x L), got {self.w_dec.shape}."
            )
        if len(self.names) != n_channels:
            raise ValueError(
                f"{where}: {len(self.names)} names for {n_channels} channels."
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"{where}: channel names must be unique.")
        for label, w in (("W_enc", self.w_enc), ("W_dec", self.w_dec)):
            if not np.isfinite(w).all():
                raise ValueError(f"{where}: {label} has non-finite entries.")
            off = float(np.abs(w.sum(axis=1) - 1.0).max())
            if off > AFFINE_TOL:
                raise ValueError(
                    f"{where}: the rows of {label} must sum to 1 (affine "
                    f"combinations; off by up to {off:.3g}). Otherwise the losses "
                    "would depend on where the crop sits in the frame."
                )
        if latent_partner is not None:
            partner = np.asarray(latent_partner).astype(np.int64).reshape(-1)
            ok = (
                partner.shape == (n_latent,)
                and (partner >= 0).all()
                and (partner < n_latent).all()
                and (partner[partner] == np.arange(n_latent)).all()
            )
            if not ok:
                raise ValueError(
                    f"{where}: latent_partner must give each of the {n_latent} "
                    "latent points a partner index, pairing them both ways."
                )
            latent_partner = partner
        self.latent_partner = latent_partner

    @property
    def n_latent(self) -> int:
        """Number of latent points ``L``."""
        return self.w_enc.shape[0]

    @property
    def latent_names(self) -> List[str]:
        """Names of the latent channels (`latent_names`)."""
        return latent_names(self.n_latent)

    def channel_index(self, names: Sequence[str]) -> np.ndarray:
        """Row of each of ``names`` in the autoencoder (all must be present)."""
        pos = {n: i for i, n in enumerate(self.names)}
        return np.array([pos[n] for n in names], dtype=np.int64)

    def latent_skeleton(self, name: str = "latent") -> sio.Skeleton:
        """A skeleton of the latent points, symmetric by ``latent_partner``."""
        nodes = self.latent_names
        symmetries = []
        if self.latent_partner is not None:
            for i, j in enumerate(self.latent_partner.tolist()):
                if i < j:
                    symmetries.append([nodes[i], nodes[j]])
        return sio.Skeleton(nodes=nodes, edges=[], symmetries=symmetries, name=name)


def load_keypoint_autoencoder(path: str) -> KeypointAutoencoder:
    """Read a keypoint autoencoder from an ``.npz``.

    Args:
        path: File with ``W_enc`` (``L x J``), ``W_dec`` (``J x L``), ``names``
            (``J`` strings) and optionally ``latent_partner`` (``L`` ints).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a key is missing or the matrices do not fit together, are
            not affine (rows summing to 1), or the partners do not pair up.
    """
    try:
        data = np.load(str(path), allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"coordinate_losses.autoencoder: no such file {str(path)!r}."
        )
    with data:
        missing = [k for k in ("W_enc", "W_dec", "names") if k not in data.files]
        if missing:
            raise ValueError(
                f"coordinate_losses.autoencoder {str(path)!r} lacks {missing} "
                f"(has {sorted(data.files)})."
            )
        return KeypointAutoencoder(
            data["W_enc"],
            data["W_dec"],
            data["names"].tolist(),
            data["latent_partner"] if "latent_partner" in data.files else None,
            path=str(path),
        )


# ------------------------------------------------------------------- losses


def soft_argmax(
    confmaps: torch.Tensor,
    output_stride: int = 1,
    beta: float = 1.0,
    window: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable peak coordinates of confidence maps (integral regression).

    The expectation of the grid coordinates under ``softmax(beta * map)`` over
    each map, in input pixels: grid cell ``(i, j)`` sits at
    ``(j * output_stride, i * output_stride)``, as in the confidence-map targets.

    With ``window``, the softmax runs only over the cells within ``window`` cells
    (Chebyshev distance) of each map's maximum, which keeps the flat background of
    maps trained towards Gaussians from pulling the estimate towards the map's
    centre and so allows a smaller ``beta`` (a large ``beta`` narrows the peak
    below a cell and rounds the estimate towards the grid). The window's position
    carries no gradient.

    Args:
        confmaps: ``(B, C, h, w)`` maps (computed in float32).
        output_stride: Input pixels per grid cell.
        beta: Inverse temperature.
        window: Radius in grid cells of the window around each maximum, or
            ``None`` for the whole map.

    Returns:
        ``(B, C, 2)`` float32 ``(x, y)`` coordinates.
    """
    b, c, h, w = confmaps.shape
    logits = confmaps.float().reshape(b, c, h * w) * float(beta)
    if window is not None:
        peak = logits.detach().argmax(dim=-1)  # (B, C)
        rows = torch.arange(h, device=confmaps.device).repeat_interleave(w)
        cols = torch.arange(w, device=confmaps.device).repeat(h)
        inside = ((rows - (peak // w)[..., None]).abs() <= int(window)) & (
            (cols - (peak % w)[..., None]).abs() <= int(window)
        )
        logits = logits.masked_fill(~inside, float("-inf"))
    prob = torch.softmax(logits, dim=-1).reshape(b, c, h, w)
    xs = torch.arange(w, device=confmaps.device, dtype=torch.float32) * output_stride
    ys = torch.arange(h, device=confmaps.device, dtype=torch.float32) * output_stride
    x = (prob.sum(dim=2) * xs).sum(dim=-1)
    y = (prob.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``sum(w * v) / sum(w)``, 0 (with zero gradient) when every weight is 0."""
    total = weights.sum()
    return (values * weights).sum() / torch.where(
        total > 0, total, torch.ones_like(total)
    )


def consistency_loss(
    points: torch.Tensor,
    projection: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``|P - M P|_1`` per channel, averaged over channels and samples.

    Args:
        points: ``(B, C, 2)`` coordinates ``P``.
        projection: ``(C, C)`` matrix ``M = W_dec W_enc`` in the points' channel
            order.
        sample_weights: ``(B,)`` weights (0 leaves a sample out), or ``None``.

    Returns:
        Scalar loss in the points' units.
    """
    residual = points - torch.einsum("ij,bjd->bid", projection, points)
    per_channel = residual.abs().sum(dim=-1)  # (B, C)
    if sample_weights is None:
        return per_channel.mean()
    weights = sample_weights.to(per_channel.dtype)[:, None].expand_as(per_channel)
    return _weighted_mean(per_channel, weights)


def decoded_loss(
    latent_points: torch.Tensor,
    decoder: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``|W_dec Q - Y|_1`` over the target channels that count.

    Args:
        latent_points: ``(B, L, 2)`` latent points ``Q``.
        decoder: ``(T, L)`` rows of ``W_dec`` for the ``T`` target channels.
        targets: ``(B, T, 2)`` target coordinates ``Y``; NaN = no target.
        weights: ``(B, T)`` per-channel weights, or ``None`` (all 1). Channels
            without a finite target get weight 0.

    Returns:
        Scalar loss: the weighted mean over channels of ``|dx| + |dy|``.
    """
    decoded = torch.einsum("tl,bld->btd", decoder, latent_points)
    finite = torch.isfinite(targets).all(dim=-1)
    w = finite.to(decoded.dtype)
    if weights is not None:
        w = w * weights.to(decoded.dtype)
    err = (decoded - torch.nan_to_num(targets, nan=0.0)).abs().sum(dim=-1)
    return _weighted_mean(err, w)


class CoordinateLosses(nn.Module):
    """The configured coordinate term of a centered-instance training module.

    Holds the frozen matrix in non-persistent buffers: it follows the module
    across devices but is not in the state dict, the optimizer or checkpoints.

    Attributes:
        mode: ``"consistency"`` or ``"decoded"``.
        weight: The term's weight.
        beta: Soft-argmax inverse temperature.
        window: Soft-argmax window radius in grid cells, or ``None``.
        output_stride: The head's output stride.
    """

    def __init__(
        self,
        mode: str,
        weight: float,
        matrix: np.ndarray,
        output_stride: int,
        beta: float = 20.0,
        window: Optional[int] = None,
    ):
        """Initialize the term.

        Args:
            mode: ``"consistency"`` (``matrix`` is the ``(C, C)`` projection
                ``W_dec W_enc`` in the head's channel order) or ``"decoded"``
                (``matrix`` is the ``(T, L)`` rows of ``W_dec`` for the batch's
                ``T`` keypoint channels).
            weight: The term's weight.
            matrix: See ``mode``.
            output_stride: The head's output stride.
            beta: Soft-argmax inverse temperature.
            window: Soft-argmax window radius in grid cells (``None``: whole map).
        """
        super().__init__()
        if mode not in TERMS:
            raise ValueError(f"mode must be one of {TERMS}, got {mode!r}.")
        self.mode = mode
        self.weight = float(weight)
        self.beta = float(beta)
        self.window = None if window is None else int(window)
        self.output_stride = int(output_stride)
        self.register_buffer(
            "matrix", torch.as_tensor(np.asarray(matrix, np.float32)), persistent=False
        )

    @property
    def log_name(self) -> str:
        """Metric suffix: ``coord_consistency`` or ``coord_decoded``."""
        return f"coord_{self.mode}"

    def points(self, confmaps: torch.Tensor) -> torch.Tensor:
        """Soft-argmax coordinates ``(B, C, 2)`` of the head's maps, in crop px."""
        return soft_argmax(confmaps, self.output_stride, self.beta, self.window)

    def forward(self, confmaps: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """The unweighted term for a batch (float32, autocast off).

        Args:
            confmaps: ``(B, C, h, w)`` head output.
            batch: The batch; ``decoded`` reads ``instance`` (``(B, 1, T, 2)`` crop
                keypoints) and, when present, ``node_weights``. Samples marked
                ``is_negative`` are left out of both terms.
        """
        device_type = confmaps.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            points = self.points(confmaps)
            b = points.shape[0]
            sample_w = None
            is_negative = batch.get("is_negative", None)
            if is_negative is not None:
                sample_w = (
                    ~torch.as_tensor(is_negative, device=points.device)
                    .reshape(b)
                    .bool()
                ).float()
            if self.mode == "consistency":
                return consistency_loss(points, self.matrix, sample_w)
            targets = batch["instance"].reshape(b, -1, 2).to(points.device).float()
            n_targets = self.matrix.shape[0]
            if targets.shape[1] != n_targets:
                raise ValueError(
                    f"coordinate_losses (decoded): the batch has {targets.shape[1]} "
                    f"keypoint channels, the decoder rows {n_targets}."
                )
            weights = batch.get("node_weights", None)
            if weights is not None:
                weights = weights.reshape(b, -1).to(points.device).float()
            if sample_w is not None:
                weights = sample_w[:, None] * (
                    weights if weights is not None else torch.ones_like(targets[..., 0])
                )
            return decoded_loss(points, self.matrix, targets, weights)


# ------------------------------------------------------------ config & setup


def coordinate_losses_config(config: DictConfig) -> Optional[DictConfig]:
    """The ``trainer_config.coordinate_losses`` block, or ``None``."""
    return OmegaConf.select(config, CONFIG_KEY, default=None)


def coordinate_losses_mode(config: DictConfig) -> Optional[str]:
    """``"consistency"``, ``"decoded"`` or ``None`` (block unset)."""
    cfg = coordinate_losses_config(config)
    if cfg is None:
        return None
    return "decoded" if float(cfg.get("decoded", 0.0) or 0.0) > 0 else "consistency"


def drop_unset_coordinate_losses(config: Optional[DictConfig]) -> None:
    """Remove an unset (``null``) block so a saved config has no new key.

    Training calls it through `setup_coordinate_losses`; other writers of a
    ``training_config.yaml`` built from the schema call it before saving.
    """
    if config is None:
        return
    trainer = OmegaConf.select(config, "trainer_config", default=None)
    if trainer is None or "coordinate_losses" not in trainer:
        return
    if trainer.get("coordinate_losses") is None:
        with open_dict(trainer):
            trainer.pop("coordinate_losses")


def _flips_on(config: DictConfig) -> bool:
    """Whether training applies flip augmentation (stock or warp crops)."""
    if not OmegaConf.select(
        config, "data_config.use_augmentations_train", default=False
    ):
        return False
    for key in (
        "data_config.augmentation_config.geometric.flip_p",
        "data_config.augmentation_config.stage2.geometric.flip_p",
    ):
        p = OmegaConf.select(config, key, default=None)
        if p is not None and float(p) > 0:
            return True
    return False


def _symmetry_pairs(skeleton: Optional[sio.Skeleton]) -> List[Tuple[str, str]]:
    """Name pairs the flip swaps, from the skeleton the data loader uses."""
    if skeleton is None:
        return []
    pairs = []
    for symmetry in getattr(skeleton, "symmetries", []) or []:
        a, b = list(symmetry)
        pairs.append((a.name, b.name))
    return pairs


def _swap_permutation(n: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    perm = np.arange(n)
    for i, j in pairs:
        perm[i], perm[j] = j, i
    return perm


def _check_names(model_names: Sequence[str], ae: KeypointAutoencoder) -> None:
    """The consistency term needs the head's channels to be the autoencoder's."""
    model_set, ae_set = set(model_names), set(ae.names)
    if model_set == ae_set and len(model_names) == len(ae.names):
        return
    missing = [n for n in ae.names if n not in model_set]
    extra = [n for n in model_names if n not in ae_set]
    message = (
        "trainer_config.coordinate_losses (consistency): the head's part names must "
        f"be the autoencoder's {len(ae.names)} channel names (any order); the head "
        f"has {len(model_names)}."
    )
    if missing:
        message += (
            f" Not predicted: {missing[:10]}{' ...' if len(missing) > 10 else ''}."
        )
    if extra:
        message += (
            f" Not in the autoencoder: {extra[:10]}{' ...' if len(extra) > 10 else ''}."
        )
    logger.error(message)
    raise ValueError(message)


def _skeletons_config(skeletons: List[sio.Skeleton]) -> List[Dict[str, Any]]:
    """``data_config.skeletons`` entries, written as the trainer writes them."""
    from sleap_io.io.skeleton import SkeletonYAMLEncoder

    encoded = yaml.safe_load(SkeletonYAMLEncoder().encode(skeletons))
    out = []
    for name, skl in encoded.items():
        skl["name"] = name
        out.append(skl)
    return out


def _refuse(message: str) -> None:
    message = f"trainer_config.coordinate_losses: {message}"
    logger.error(message)
    raise ValueError(message)


def _check_decoded_compatible(config: DictConfig) -> None:
    """Options the latent-point head (decoded term) cannot run with."""

    def sel(key, default=None):
        return OmegaConf.select(config, key, default=default)

    if sel("trainer_config.distillation") is not None:
        _refuse("the decoded term cannot be combined with distillation.")
    if sel("trainer_config.online_hard_keypoint_mining.online_mining", False):
        _refuse(
            "the decoded term has no confidence-map loss; turn off "
            "online_hard_keypoint_mining."
        )
    if sel("trainer_config.eval.enabled", False):
        _refuse(
            "epoch-end evaluation compares predictions with labels channel by "
            "channel, but the decoded term's head predicts latent points; set "
            "trainer_config.eval.enabled: false."
        )
    if sel("trainer_config.visualize_preds_during_training", False):
        _refuse(
            "prediction plots pair predicted and labelled channels, but the decoded "
            "term's head predicts latent points; set "
            "visualize_preds_during_training: false."
        )
    monitor = str(sel("trainer_config.model_ckpt.monitor", "val/loss") or "")
    parts = monitor.split("/")
    if len(parts) >= 3 and parts[0] == "val" and parts[-1] == "loss":
        _refuse(
            f"model_ckpt.monitor={monitor!r}: the decoded term logs no per-file "
            "validation loss (only val/loss, the weighted decoded term, and "
            "val/coord_decoded); monitor val/loss."
        )


def check_not_saved_decoded_config(config: DictConfig) -> None:
    """Refuse to train from a config saved by a decoded-term run.

    That config describes the trained latent-point model (for prediction): its
    head lists the latent points, and the labelled channels' head settings sit
    in ``coordinate_losses.data_head``. Run by the trainer before the head is
    checked against the labels, so the error names the cause.
    """
    cfg = coordinate_losses_config(config)
    if cfg is None or cfg.get("data_head", None) is None:
        return
    _refuse(
        "data_head is set, so this config was saved by a run with the decoded term "
        "and describes the latent-point model it trained (for prediction). To "
        "train again, start from that run's original config (initial_config.yaml), "
        "or copy data_head's settings back into "
        "model_config.head_configs.centered_instance.confmaps, restore the "
        "labels' skeleton in data_config.skeletons and remove data_head."
    )


def label_node_names(
    config: DictConfig, skeletons: Sequence[sio.Skeleton]
) -> List[str]:
    """Names of the channels the labels and the batches carry.

    The model's skeleton nodes, except with the decoded term, whose model
    predicts latent points while the batches (and the gallery's drawn points,
    see `decoded_keypoints`) carry the labelled channels.
    """
    if coordinate_losses_mode(config) == "decoded":
        data_head = coordinate_losses_config(config).get("data_head", None)
        if data_head is not None:
            return list(data_head.part_names)
    return list(skeletons[0].node_names) if skeletons else []


def decoded_keypoints(module: nn.Module, confmaps: torch.Tensor):
    """Decoded keypoints of a decoded-term training module, or ``None``.

    For a centered-instance module training the decoded term: the soft-argmax
    latent points decoded into the labelled channels, ``(B, T, 2)`` crop pixels,
    i.e. exactly what the loss compares with the labels. ``None`` for any other
    module, which then draws its own peaks.
    """
    coord = getattr(module, "coord_losses", None)
    if coord is None or coord.mode != "decoded":
        return None
    with (
        torch.no_grad(),
        torch.autocast(device_type=confmaps.device.type, enabled=False),
    ):
        return torch.einsum("tl,bld->btd", coord.matrix, coord.points(confmaps))


def setup_coordinate_losses(
    config: DictConfig,
    skeletons: List[sio.Skeleton],
    initial_config: Optional[DictConfig] = None,
) -> List[sio.Skeleton]:
    """Validate ``trainer_config.coordinate_losses`` and resolve the model's head.

    Run once by the trainer after the rest of the config is resolved. Without the
    block it only removes the unset key from ``config`` and ``initial_config``,
    so the saved configs carry nothing new.

    With the consistency term it checks that the head's part names are the
    autoencoder's channels and, with flip augmentation on, that the autoencoder's
    projection commutes with the left/right swap the flip applies.

    With the decoded term it checks that every labelled channel is one of the
    autoencoder's (and, with flips on, that the decoder maps swapped latent
    points to swapped channels), then turns the head into the latent-point
    head: ``part_names`` become the latent names, the anchor settings are
    cleared, ``data_config.skeletons`` becomes the latent skeleton, and the
    labelled channels' head settings move to ``coordinate_losses.data_head``
    for the data loader (`dataset_config`).

    Args:
        config: The resolved training config (modified in place).
        skeletons: The trainer's skeletons (the data loader's channels).
        initial_config: The pre-setup config snapshot, if any.

    Returns:
        The skeletons for the rest of setup: the latent skeleton with the decoded
        term, else ``skeletons`` unchanged.

    Raises:
        ValueError: On any mismatch listed above, for a model type other than
            ``centered_instance``, or for options the latent head cannot run with.
    """
    from sleap_nn.config.utils import get_model_type_from_cfg

    cfg = coordinate_losses_config(config)
    if cfg is None:
        drop_unset_coordinate_losses(config)
        drop_unset_coordinate_losses(initial_config)
        return skeletons
    model_type = get_model_type_from_cfg(config)
    if model_type != "centered_instance":
        _refuse(f"only centered_instance models are supported, not {model_type!r}.")
    mode = coordinate_losses_mode(config)
    ae = load_keypoint_autoencoder(cfg.autoencoder)
    head = config.model_config.head_configs.centered_instance.confmaps
    names = list(head.part_names)
    skeleton = skeletons[0] if skeletons else None
    flips = _flips_on(config)

    if mode == "consistency":
        _check_names(names, ae)
        if flips:
            idx = {n: i for i, n in enumerate(ae.names)}
            perm = _swap_permutation(
                len(ae.names),
                [(idx[a], idx[b]) for a, b in _symmetry_pairs(skeleton)],
            )
            proj = ae.w_dec @ ae.w_enc
            off = float(np.abs(proj[perm][:, perm] - proj).max())
            if off > SYMMETRY_TOL:
                _refuse(
                    "flip augmentation swaps the skeleton's symmetric channels, but "
                    "the autoencoder's projection W_dec W_enc does not commute with "
                    f"that swap (off by up to {off:.3g}), so the consistency term "
                    "would treat flipped crops differently. Use an autoencoder "
                    "built with the same left/right pairs, or set flip_p: 0."
                )
        logger.info(
            f"Coordinate losses: consistency (weight {cfg.consistency}) through "
            f"{ae.n_latent} latent points of {cfg.autoencoder}."
        )
        return skeletons

    _check_decoded_compatible(config)
    unknown = [n for n in names if n not in set(ae.names)]
    if unknown:
        _refuse(
            "the decoded term compares W_dec Q with the labelled channels, but "
            f"{len(unknown)} of them are not autoencoder channels: "
            f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}."
        )
    if flips:
        rows = ae.channel_index(names)
        pos = {n: i for i, n in enumerate(names)}
        perm = _swap_permutation(
            len(names),
            [
                (pos[a], pos[b])
                for a, b in _symmetry_pairs(skeleton)
                if a in pos and b in pos
            ],
        )
        dec = ae.w_dec[rows]
        candidates = [np.arange(ae.n_latent)]
        if ae.latent_partner is not None:
            candidates.insert(0, ae.latent_partner)
        off = min(float(np.abs(dec[perm] - dec[:, p]).max()) for p in candidates)
        if off > SYMMETRY_TOL:
            _refuse(
                "flip augmentation swaps the skeleton's symmetric channels, but no "
                "swap of the latent points (latent_partner) maps the decoder onto "
                f"that swap (off by up to {off:.3g}), so flipped crops would have "
                "no exact latent target. Use an autoencoder built with the same "
                "left/right pairs, or set flip_p: 0."
            )

    eval_cfg = OmegaConf.select(config, "trainer_config.eval", default=None)
    if eval_cfg is not None:
        with open_dict(eval_cfg):
            eval_cfg.post_train_splits = []
        logger.info(
            "Coordinate losses (decoded): post-training evaluation is off "
            "(trainer_config.eval.post_train_splits: []): the model predicts "
            "latent points, which share no node with the labels."
        )
    data_head = {
        "part_names": names,
        "anchor_part": head.get("anchor_part", None),
        "centroid_method": head.get("centroid_method", None),
        "centroid_fallback": head.get("centroid_fallback", None),
    }
    with open_dict(cfg):
        cfg.data_head = data_head
    head.part_names = ae.latent_names
    head.anchor_part = None
    if head.get("centroid_method", None) == "anchor":
        head.centroid_method = None
    if "centroid_fallback" in head:
        head.centroid_fallback = None
    latent = ae.latent_skeleton()
    config.data_config.skeletons = _skeletons_config([latent])
    logger.info(
        f"Coordinate losses: decoded (weight {cfg.decoded}); the head predicts the "
        f"{ae.n_latent} latent points of {cfg.autoencoder}, trained against "
        f"{len(names)} labelled channels."
    )
    return [latent]


def dataset_config(config: DictConfig) -> DictConfig:
    """The config the data loader reads.

    With the decoded term the model's head describes the latent points, while the
    data loader still renders crops of the labelled channels: return a copy whose
    centered-instance head has the labelled channels' settings
    (``coordinate_losses.data_head``). Otherwise ``config`` itself.
    """
    if coordinate_losses_mode(config) != "decoded":
        return config
    data_head = coordinate_losses_config(config).get("data_head", None)
    if data_head is None:
        return config
    view = copy.deepcopy(config)
    head = view.model_config.head_configs.centered_instance.confmaps
    for key, value in OmegaConf.to_container(data_head).items():
        if key in head:
            head[key] = value
    return view


def build_coordinate_losses(config: DictConfig) -> Optional[CoordinateLosses]:
    """The training module's `CoordinateLosses` for ``config``, or ``None``.

    ``config`` is the resolved training config (`setup_coordinate_losses` has
    run, so with the decoded term the head already predicts the latent points
    and ``data_head`` names the batch's keypoint channels).
    """
    cfg = coordinate_losses_config(config)
    if cfg is None:
        return None
    mode = coordinate_losses_mode(config)
    ae = load_keypoint_autoencoder(cfg.autoencoder)
    head = config.model_config.head_configs.centered_instance.confmaps
    names = list(head.part_names)
    if mode == "consistency":
        _check_names(names, ae)
        rows = ae.channel_index(names)
        matrix = (ae.w_dec @ ae.w_enc)[rows][:, rows]
        weight = float(cfg.consistency)
    else:
        data_head = cfg.get("data_head", None)
        if data_head is None or names != ae.latent_names:
            _refuse(
                "the decoded term's head is set up by the trainer "
                "(setup_coordinate_losses); train through ModelTrainer."
            )
        matrix = ae.w_dec[ae.channel_index(list(data_head.part_names))]
        weight = float(cfg.decoded)
    return CoordinateLosses(
        mode=mode,
        weight=weight,
        matrix=matrix,
        output_stride=int(head.output_stride),
        beta=float(cfg.get("soft_argmax_beta", 20.0)),
        window=cfg.get("soft_argmax_window", None),
    )
