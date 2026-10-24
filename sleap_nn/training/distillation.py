"""DWPose-style knowledge distillation for confidence-map models.

After DWPose (Yang et al., "Effective Whole-body Pose Estimation with Two-stages
Distillation", 2023, arXiv 2307.15880), configured by
``trainer_config.distillation`` (`DistillationConfig`):

1. **Teacher -> student** (``stage: 1``): the supervised loss plus
   ``r(t) * (alpha * L_feat + beta * L_heatmap)``.

   * ``L_feat``: MSE between the teacher's deepest backbone feature and the
     student's, mapped student -> teacher channels by a trainable 1x1 conv
     projector. The feature is a plain ViT's last tapped token map
     (``"last_tap"`` in the backbone outputs), else the bottleneck
     (``"middle_output"``); the teacher's is resized to the student's grid.
   * ``L_heatmap``: MSE between teacher and student confidence maps over the
     nodes they share BY NAME (a COCO-17 teacher can supervise a larger union
     skeleton), weighted by the batch's ``node_weights`` unless
     ``mask_unlabeled_nodes`` is off.
   * ``r(t) = 1 - t / T`` decays both terms to 0 over the run (per optimizer
     step; DWPose Eq. 5).

2. **Head self-distillation** (``stage: 2``): the trained student is its own
   teacher. Everything outside ``trainable`` (default: the head) is frozen,
   ``trainable`` is re-initialized and learns ``gamma * L_heatmap`` against a
   frozen copy of the old weights. When the whole backbone is frozen, the
   teacher's head runs on the student's own (shared) backbone features, so the
   backbone runs once per step.

Design: a `Distiller` is a component of the `LightningModel` (its
``distiller`` attribute), not a subclass or a wrapper. The supported training
steps call ``LightningModel._forward_train`` (which asks the network for its
backbone outputs only while distilling) and return
``LightningModel._with_distillation(loss, ...)``; with no distiller both are
no-ops, so plain training is unchanged. A Lightning callback cannot change the
loss a training step returns, and a wrapper module would rename every
checkpoint key and hide ``pl_module.model`` from the callbacks.

The teacher is NOT a registered submodule: it is not in the ``state_dict``,
the optimizer, DDP's parameter set or checkpoints. It follows the student
across devices (``Distiller._apply``), stays in eval mode and frozen, and runs
under bf16 autocast on CUDA. Under DDP every rank loads its own copy from disk.
The projector IS a submodule (trained, all-reduced by DDP); checkpoints carry
it under ``checkpoint["distillation"]`` instead of ``state_dict`` so they load
strictly into a plain model, and resuming puts it back.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from sleap_nn.training.losses import compute_weighted_mse_loss
from sleap_nn.training.weight_transfer import (
    init_from_checkpoint,
    model_head_channels,
    reinit,
    select_params,
    set_requires_grad,
)

# Model types whose training step adds the distillation loss.
DISTILLATION_MODEL_TYPES = (
    "single_instance",
    "centered_instance",
    "multi_class_topdown",
)
# Checkpoint key that holds the distiller's own weights (the projector).
CHECKPOINT_KEY = "distillation"
# State-dict prefix of the distiller inside the LightningModel.
STATE_PREFIX = "distiller."
# Optional batch key: the student's crop rendered at the teacher's resolution
# (same window, ``teacher_input_scale`` x the pixels). A dual-resolution crop
# path can emit it; without it the student's crop is resampled.
TEACHER_IMAGE_KEY = "teacher_image"


def _resolve_model_dir(path: str) -> Tuple[Path, Path]:
    """``(checkpoint, training_config.yaml)`` for a model dir or a ``.ckpt``."""
    p = Path(path)
    ckpt = p / "best.ckpt" if p.is_dir() else p
    cfg = ckpt.parent / "training_config.yaml"
    if not ckpt.exists():
        raise FileNotFoundError(f"Distillation teacher checkpoint not found: {ckpt}")
    if not cfg.exists():
        raise FileNotFoundError(
            f"Distillation teacher needs its training_config.yaml next to {ckpt}."
        )
    return ckpt, cfg


def load_teacher_model(path: str) -> Tuple[nn.Module, DictConfig]:
    """Load a trained sleap-nn model as a frozen teacher network.

    Only ``model_config`` of the teacher's ``training_config.yaml`` is used, so
    teachers trained with data/trainer options this version lacks still load.

    Args:
        path: Model directory (``best.ckpt`` + ``training_config.yaml``) or a
            ``.ckpt`` file next to its ``training_config.yaml``.

    Returns:
        ``(model, config)``: the teacher `Model` (backbone + heads, its
        checkpoint ``state_dict`` -- the EMA weights when trained with EMA --
        loaded strictly), in eval mode with ``requires_grad=False``, and its
        training config.
    """
    from sleap_nn.architectures.model import Model
    from sleap_nn.config.utils import (
        get_backbone_type_from_cfg,
        get_model_type_from_cfg,
    )

    ckpt_path, cfg_path = _resolve_model_dir(path)
    cfg = OmegaConf.load(cfg_path.as_posix())
    model_type = get_model_type_from_cfg(cfg)
    backbone_type = get_backbone_type_from_cfg(cfg)
    backbone_cfg = copy.deepcopy(cfg.model_config.backbone_config[backbone_type])
    if backbone_type == "pretrained":
        backbone_cfg.weights = False  # the checkpoint has every weight
    model = Model(
        backbone_type=backbone_type,
        backbone_config=backbone_cfg,
        head_configs=cfg.model_config.head_configs[model_type],
        model_type=model_type,
    )
    ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    logger.info(
        f"Distillation teacher: {model_type} / {backbone_type} from {ckpt_path}"
    )
    return model, cfg


def _confmap_head_index(model: nn.Module) -> int:
    """Index of the model's (first) per-node confidence-map head."""
    for i, head in enumerate(model_head_channels(model)):
        if head["name"].endswith("ConfmapsHead") and head["channel_names"]:
            return i
    raise ValueError(
        "Distillation needs a per-node confidence-map head; found "
        f"{[h['name'] for h in model_head_channels(model)]}."
    )


def shared_node_channels(
    student_names: Sequence[str], teacher_names: Sequence[str]
) -> Tuple[List[int], List[int]]:
    """``(student idx, teacher idx)`` of the confmap channels shared by node name.

    Raises:
        ValueError: If the two skeletons share no node.
    """
    pairs = [
        (i, list(teacher_names).index(n))
        for i, n in enumerate(student_names)
        if n in teacher_names
    ]
    if not pairs:
        raise ValueError(
            f"Student and teacher share no node names ({list(student_names)} vs "
            f"{list(teacher_names)})."
        )
    s_idx, t_idx = zip(*pairs)
    return list(s_idx), list(t_idx)


def backbone_feature(backbone_outputs: Dict[str, Any]) -> torch.Tensor:
    """The distilled feature: a plain ViT's last tap, else the bottleneck."""
    feat = backbone_outputs.get("last_tap")
    return backbone_outputs["middle_output"] if feat is None else feat


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if OmegaConf.is_config(cfg):
        val = OmegaConf.select(cfg, key, default=default)
    elif isinstance(cfg, Mapping):
        val = cfg.get(key, default)
    else:
        val = getattr(cfg, key, default)
    if OmegaConf.is_config(val):
        val = OmegaConf.to_container(val)
    return default if val is None else val


class Distiller(nn.Module):
    """The distillation state and loss of one student `LightningModel`.

    Build it with `attach_distillation`. Registered as ``student.distiller``:
    only the projector is a parameter; the teacher network is held outside the
    module tree (see the module docstring).

    Attributes:
        stage: ``1`` or ``2``.
        alpha, beta, gamma: Loss weights (see `DistillationConfig`).
        supervised_weight: Weight of the student's own (supervised) loss.
        decay: ``"linear"`` or ``"none"`` (``None`` at construction: linear at
            stage 1, none at stage 2).
        teacher_input_scale: Student crop -> teacher input resampling factor.
        mask_unlabeled_nodes: Weight the heatmap term by ``node_weights``.
        teacher_bf16: Run the teacher under bf16 autocast on CUDA.
        student_head: Name of the student's confmap head.
        teacher_head: Name of the teacher's confmap head.
        student_idx, teacher_idx: Shared confmap channels.
        projector: 1x1 conv student -> teacher feature channels (stage 1 with
            ``alpha > 0``), else ``None``.
        shared_features: Stage 2 with a frozen backbone: the teacher's heads run
            on the student's backbone outputs.
        frozen_backbone: The student's backbone is frozen and kept in eval mode.
    """

    def __init__(
        self,
        teacher: nn.Module,
        stage: int,
        student_head: str,
        teacher_head: str,
        student_idx: List[int],
        teacher_idx: List[int],
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        supervised_weight: Optional[float] = None,
        decay: Optional[str] = None,
        teacher_input_scale: float = 1.0,
        mask_unlabeled_nodes: bool = True,
        teacher_bf16: bool = True,
        projector: Optional[nn.Module] = None,
        shared_features: bool = False,
        frozen_backbone: bool = False,
    ):
        """Initialize from resolved settings (see `attach_distillation`)."""
        super().__init__()
        # Not a submodule: outside state_dict / parameters() / train() / DDP.
        object.__setattr__(self, "teacher", teacher)
        self.stage = int(stage)
        self.student_head = student_head
        self.teacher_head = teacher_head
        self.student_idx = list(student_idx)
        self.teacher_idx = list(teacher_idx)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        if supervised_weight is None:
            supervised_weight = 1.0 if self.stage == 1 else 0.0
        self.supervised_weight = float(supervised_weight)
        if decay is None:
            decay = "linear" if self.stage == 1 else "none"
        self.decay = decay
        self.teacher_input_scale = float(teacher_input_scale)
        self.mask_unlabeled_nodes = bool(mask_unlabeled_nodes)
        self.teacher_bf16 = bool(teacher_bf16)
        self.projector = projector
        self.shared_features = bool(shared_features)
        self.frozen_backbone = bool(frozen_backbone)
        self._total_steps: Optional[int] = None

    def _apply(self, fn, *args, **kwargs):
        """Move/cast the (unregistered) teacher along with the student."""
        self.teacher._apply(fn, *args, **kwargs)
        return super()._apply(fn, *args, **kwargs)

    # ------------------------------------------------------------- schedule

    def weight(self, global_step: int, total_steps: Optional[int]) -> float:
        """``r(t)``: ``1 - t / T`` (``decay: linear``) or ``1``."""
        if self.decay == "none" or not total_steps:
            return 1.0
        return max(0.0, 1.0 - global_step / max(int(total_steps), 1))

    def total_steps(self, trainer) -> Optional[int]:
        """The run's optimizer steps ``T`` (cached; ``None`` without a trainer)."""
        if self._total_steps is None and trainer is not None:
            total = trainer.estimated_stepping_batches
            if total is not None and total != float("inf"):
                self._total_steps = int(total)
        return self._total_steps

    # ---------------------------------------------------------------- teacher

    def teacher_input(self, x: torch.Tensor, batch: Dict) -> torch.Tensor:
        """The teacher's input: ``batch["teacher_image"]`` or the resampled crop.

        Args:
            x: The student's (normalized) input ``(B, C, H, W)``.
            batch: The training batch.
        """
        from sleap_nn.data.normalization import normalize_on_gpu

        image = batch.get(TEACHER_IMAGE_KEY)
        if image is not None:
            if image.dim() == 5:
                image = torch.squeeze(image, dim=1)
            return normalize_on_gpu(image.to(x.device))
        if self.teacher_input_scale == 1.0:
            return x
        stride = _max_stride(self.teacher)
        h, w = x.shape[-2:]
        size = [
            max(stride, int(round(d * self.teacher_input_scale / stride)) * stride)
            for d in (h, w)
        ]
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def run_teacher(
        self, x: torch.Tensor, batch: Dict, backbone_outputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Teacher confmaps (shared channels, float32) and feature (or ``None``)."""
        use_bf16 = self.teacher_bf16 and x.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if self.shared_features:
                # Stage 2, frozen backbone: the old heads on the same features.
                out = self.teacher.forward_heads(backbone_outputs)
                t_bb = None
            else:
                out, t_bb = self.teacher(
                    self.teacher_input(x, batch), return_backbone_outputs=True
                )
        cms = out[self.teacher_head][:, self.teacher_idx].float()
        feat = None
        if self.projector is not None and t_bb is not None:
            feat = backbone_feature(t_bb).float()
        return cms, feat

    # ------------------------------------------------------------------- loss

    def loss_terms(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        backbone_outputs: Dict[str, Any],
        batch: Dict,
        node_weights: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Unweighted distillation terms ``{"heatmap": ..., "feat": ...}``.

        Args:
            x: The student's normalized input.
            outputs: The student's head outputs.
            backbone_outputs: The student's backbone output dict.
            batch: The training batch (for ``teacher_image``).
            node_weights: ``(B, n_student_nodes)`` per-node loss weights or
                ``None``.
        """
        t_cms, t_feat = self.run_teacher(x, batch, backbone_outputs)
        s_cms = outputs[self.student_head][:, self.student_idx]
        if t_cms.shape[-2:] != s_cms.shape[-2:]:
            t_cms = F.interpolate(
                t_cms, size=s_cms.shape[-2:], mode="bilinear", align_corners=False
            )
        w = None
        if self.mask_unlabeled_nodes and node_weights is not None:
            w = node_weights[:, self.student_idx]
        terms = {"heatmap": compute_weighted_mse_loss(s_cms.float(), t_cms, w)}
        if self.projector is not None and t_feat is not None:
            s_feat = backbone_feature(backbone_outputs)
            if t_feat.shape[-2:] != s_feat.shape[-2:]:
                t_feat = F.interpolate(
                    t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False
                )
            terms["feat"] = F.mse_loss(self.projector(s_feat).float(), t_feat)
        return terms

    def combine(
        self, supervised: torch.Tensor, terms: Dict[str, torch.Tensor], r: float
    ) -> torch.Tensor:
        """The loss to optimize from the supervised loss and the KD terms."""
        if self.stage == 2:
            kd = self.gamma * terms["heatmap"]
        else:
            kd = self.beta * terms["heatmap"]
            if "feat" in terms:
                kd = kd + self.alpha * terms["feat"]
        total = r * kd
        if self.supervised_weight:
            total = total + self.supervised_weight * supervised
        return total

    # ------------------------------------------------------------- checkpoint

    def pop_state(self, checkpoint: Dict[str, Any]) -> None:
        """Move the distiller's weights out of ``checkpoint["state_dict"]``."""
        sd = checkpoint.get("state_dict", {})
        own = {k: sd.pop(k) for k in [k for k in sd if k.startswith(STATE_PREFIX)]}
        checkpoint[CHECKPOINT_KEY] = {"stage": self.stage, "state_dict": own}

    def restore_state(self, checkpoint: Dict[str, Any], prefix_root: str) -> None:
        """Put the distiller's weights back into ``checkpoint["state_dict"]``.

        A checkpoint saved without distillation (or from the other stage) gets
        the current (fresh) projector, so strict loading still succeeds.
        """
        sd = checkpoint.setdefault("state_dict", {})
        saved = (checkpoint.get(CHECKPOINT_KEY) or {}).get("state_dict") or {}
        for k, v in self.state_dict(prefix=prefix_root).items():
            sd[k] = saved.get(k, v)


def drop_distiller_keys(checkpoint: Dict[str, Any]) -> None:
    """Remove stray distiller weights from a checkpoint's ``state_dict``."""
    sd = checkpoint.get("state_dict", {})
    for k in [k for k in sd if k.startswith(STATE_PREFIX)]:
        del sd[k]


def _max_stride(model: nn.Module) -> int:
    """The network's input-size multiple (the backbone's, else its config's)."""
    stride = getattr(model.backbone, "max_stride", None)
    if stride is None:
        stride = getattr(model.backbone_config, "max_stride", None)
    return int(stride or 1)


@torch.no_grad()
def _probe_features(model: nn.Module, size: int) -> Tuple[int, Tuple[int, int]]:
    """Channels and spatial size of a network's distilled feature at ``size``."""
    was_training = model.training
    model.eval()
    try:
        in_ch = int(model.backbone_config.in_channels)
        _, bb = model(torch.zeros(1, in_ch, size, size), return_backbone_outputs=True)
    finally:
        model.train(was_training)
    feat = backbone_feature(bb)
    return int(feat.shape[1]), tuple(feat.shape[-2:])


def _head_sigma_stride(head_cfg: Any) -> Tuple[Optional[float], Optional[int]]:
    """``(sigma, output_stride)`` of the confmap sub-config of a head config."""
    for sub in (head_cfg or {}).values():
        if sub is not None and "sigma" in sub and "part_names" in sub:
            return float(sub.sigma), int(sub.output_stride)
    return None, None


def _check_sigma(student: nn.Module, teacher_cfg: DictConfig, scale: float) -> None:
    """Warn when the teacher's heatmap width differs from the student's target."""
    from sleap_nn.config.utils import get_model_type_from_cfg

    s_sigma, s_stride = _head_sigma_stride(student.head_configs[student.model_type])
    t_type = get_model_type_from_cfg(teacher_cfg)
    t_sigma, t_stride = _head_sigma_stride(
        teacher_cfg.model_config.head_configs[t_type]
    )
    if not (s_sigma and t_sigma and s_stride and t_stride):
        return
    # sigma is in confmap pixels; the teacher's, in student confmap px after the
    # resize back to the student's grid:
    eff = t_sigma * t_stride / (scale * s_stride)
    if abs(eff / s_sigma - 1.0) > 0.2:
        logger.warning(
            f"Distillation: the teacher's heatmaps are sigma {eff:.2f} student "
            f"confmap px wide (sigma {t_sigma} at stride {t_stride}, input scale "
            f"{scale}) but the student's targets use sigma {s_sigma}."
        )


def attach_distillation(student: nn.Module, cfg: Any) -> Distiller:
    """Build a `Distiller` from ``trainer_config.distillation`` and attach it.

    Call before the optimizer is built (``trainer.fit``): stage 1 adds the
    projector's parameters and stage 2 changes what is trainable.

    Args:
        student: The `LightningModel` to train.
        cfg: A `DistillationConfig` (or its DictConfig).

    Returns:
        The distiller, also set as ``student.distiller``.

    Raises:
        ValueError: For an unsupported model type, skeletons with no shared node,
            or a stage-2 ``teacher`` that does not fully initialize the student.
    """
    if student.model_type not in DISTILLATION_MODEL_TYPES:
        raise ValueError(
            f"Distillation supports {list(DISTILLATION_MODEL_TYPES)}, not "
            f"{student.model_type!r}."
        )

    def get(key, default=None):
        return _cfg_get(cfg, key, default)

    stage = int(get("stage", 1))
    s_head = _confmap_head_index(student)
    s_heads = model_head_channels(student)

    if stage == 1:
        teacher, teacher_cfg = load_teacher_model(get("teacher"))
        t_head = _confmap_head_index(teacher)
        t_heads = model_head_channels(teacher)
        s_idx, t_idx = shared_node_channels(
            s_heads[s_head]["channel_names"], t_heads[t_head]["channel_names"]
        )
        scale = float(get("teacher_input_scale", 1.0))
        _check_sigma(student, teacher_cfg, scale)
        projector = None
        alpha = float(get("alpha", 1.0))
        if alpha > 0:
            size = 2 * max(_max_stride(student.model), _max_stride(teacher))
            cs, _ = _probe_features(student.model, size)
            ct, _ = _probe_features(teacher, size)
            projector = nn.Conv2d(cs, ct, kernel_size=1)
            feat_note = f"feature {cs}->{ct} ch"
        else:
            feat_note = "no feature term"
        distiller = Distiller(
            teacher,
            stage=1,
            student_head=s_heads[s_head]["name"],
            teacher_head=t_heads[t_head]["name"],
            student_idx=s_idx,
            teacher_idx=t_idx,
            alpha=alpha,
            beta=get("beta", 1.0),
            supervised_weight=get("supervised_weight"),
            decay=get("decay"),
            teacher_input_scale=scale,
            mask_unlabeled_nodes=get("mask_unlabeled_nodes", True),
            teacher_bf16=get("teacher_precision", "bf16") == "bf16",
            projector=projector,
        )
        logger.info(
            f"Distillation stage 1: {feat_note}, {len(s_idx)} shared nodes, "
            f"alpha={distiller.alpha}, beta={distiller.beta}, "
            f"supervised_weight={distiller.supervised_weight}, "
            f"decay={distiller.decay}, teacher_input_scale={scale}."
        )
    else:
        if get("teacher"):
            report = init_from_checkpoint(student, get("teacher"))
            if report.skipped or report.missing:
                raise ValueError(
                    "Distillation stage 2 refines the stage-1 model itself, but "
                    f"{get('teacher')} does not match the student: "
                    f"{len(report.skipped)} skipped, {len(report.missing)} missing "
                    f"(e.g. {(list(report.skipped) + report.missing)[:3]})."
                )
        trainable = list(get("trainable", ["head"]))
        if not select_params(student, trainable):
            raise ValueError(f"distillation.trainable {trainable} match no parameter.")
        teacher = copy.deepcopy(student.model).eval().requires_grad_(False)
        reinit(student, trainable)
        set_requires_grad(student, ["all"], False)
        set_requires_grad(student, trainable, True)
        frozen_backbone = not any(
            p.requires_grad for p in student.model.backbone.parameters()
        )
        if frozen_backbone:
            # The teacher's heads read the student's (identical) features.
            teacher.backbone = nn.Identity()
            student.model.backbone.eval()
        names = s_heads[s_head]["channel_names"]
        distiller = Distiller(
            teacher,
            stage=2,
            student_head=s_heads[s_head]["name"],
            teacher_head=s_heads[s_head]["name"],
            student_idx=list(range(len(names))),
            teacher_idx=list(range(len(names))),
            gamma=get("gamma", 1.0),
            supervised_weight=get("supervised_weight"),
            decay=get("decay"),
            mask_unlabeled_nodes=get("mask_unlabeled_nodes", True),
            teacher_bf16=get("teacher_precision", "bf16") == "bf16",
            shared_features=frozen_backbone,
            frozen_backbone=frozen_backbone,
        )
        n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
        logger.info(
            f"Distillation stage 2: training {trainable} ({n_train:,} params), "
            f"gamma={distiller.gamma}, supervised_weight="
            f"{distiller.supervised_weight}, shared backbone={frozen_backbone}."
        )
    student.distiller = distiller
    return distiller
