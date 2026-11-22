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

**Box detectors** (``bbox_centernet``, ``bbox_ssd``, ``bbox_detr``; experimental,
stage 1 only; `BoxDistiller`): a trained box detector of any of the three styles
teaches another. The loss is the supervised loss plus
``r(t) * (alpha * L_feat + beta * L_box)``, with ``L_feat`` as above (a DETR
backbone's feature is its last tapped token map) and ``L_box`` the student's own
detection loss against the teacher's detections on the same training input
(scores >= ``teacher_score_threshold``, categories matched by name) used as
targets, encoded by the student's own target encoder. A detector's outputs (set
queries, anchors, center heatmaps) do not line up between two models, so the
teacher's decoded boxes are the shared language: this works across styles and
input sizes.

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

# Batch key of the student's crop window rendered at the teacher's resolution
# (``teacher_crop_size``), emitted by the warp data path
# (`WarpCenteredInstanceDataset`). Without it the student's crop is resampled by
# ``teacher_input_scale``.
from sleap_nn.data.topdown_warp import TEACHER_IMAGE_KEY
from sleap_nn.training.losses import compute_weighted_mse_loss
from sleap_nn.training.weight_transfer import (
    init_from_checkpoint,
    model_head_channels,
    reinit,
    select_params,
    set_requires_grad,
)

# Confidence-map model types whose training step adds the distillation loss.
POSE_DISTILLATION_MODEL_TYPES = (
    "single_instance",
    "centered_instance",
    "multi_class_topdown",
)
# Box detectors that learn from a box-detector teacher (stage 1; `BoxDistiller`).
BOX_DISTILLATION_MODEL_TYPES = ("bbox_centernet", "bbox_ssd", "bbox_detr")
# Model types whose training step adds the distillation loss.
DISTILLATION_MODEL_TYPES = POSE_DISTILLATION_MODEL_TYPES + BOX_DISTILLATION_MODEL_TYPES
# Checkpoint key that holds the distiller's own weights (the projector).
CHECKPOINT_KEY = "distillation"
# State-dict prefix of the distiller inside the LightningModel.
STATE_PREFIX = "distiller."


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
    elif backbone_type == "dinov2_windowed" and "weights" in backbone_cfg:
        backbone_cfg.weights = None  # (a DETR backbone) likewise
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


def backbone_feature(backbone_outputs: Any) -> torch.Tensor:
    """The distilled feature: a plain ViT's last tap, else the bottleneck.

    A ``bbox_detr`` backbone returns its tapped token maps as a list: the last.
    """
    if isinstance(backbone_outputs, (list, tuple)):
        return backbone_outputs[-1]
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
        teacher_crop_size: ``(height, width)`` of the batch's ``teacher_image``
            (dual-resolution crops), which every training batch must then carry;
            ``None`` without.
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
        teacher_crop_size: Optional[Tuple[int, int]] = None,
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
        self.teacher_crop_size = (
            None
            if teacher_crop_size is None
            else tuple(int(v) for v in teacher_crop_size)
        )
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

        Raises:
            ValueError: With ``teacher_crop_size`` set, if the batch has no
                ``teacher_image`` of that size (the teacher would otherwise see the
                student's crop at the wrong resolution).
        """
        from sleap_nn.data.normalization import normalize_on_gpu

        image = batch.get(TEACHER_IMAGE_KEY)
        if image is not None:
            if image.dim() == 5:
                image = torch.squeeze(image, dim=1)
            if (
                self.teacher_crop_size is not None
                and tuple(image.shape[-2:]) != self.teacher_crop_size
            ):
                raise ValueError(
                    f"teacher_image is {tuple(image.shape[-2:])}, not "
                    f"distillation.teacher_crop_size {self.teacher_crop_size}."
                )
            return normalize_on_gpu(image.to(x.device))
        if self.teacher_crop_size is not None:
            raise ValueError(
                "distillation.teacher_crop_size is set but the batch has no "
                f"`{TEACHER_IMAGE_KEY}`: it is emitted by the warp crop path "
                "(preprocessing.crop_pipeline: warp, centered_instance)."
            )
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
        x = torch.zeros(1, in_ch, size, size)
        if getattr(model, "model_type", None) == "bbox_detr":
            # The backbone alone: the detector needs more tokens than queries.
            bb = model.backbone(x)
        else:
            _, bb = model(x, return_backbone_outputs=True)
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


class BoxDistiller(Distiller):
    """Stage-1 distillation of a box detector by a box-detector teacher.

    Experimental. Adds ``r(t) * (alpha * L_feat + beta * L_box)`` to the
    supervised loss (see the module docstring). Build it with
    `attach_distillation`.

    Attributes:
        teacher_type: The teacher's model type (``bbox_*``).
        teacher_decoder: The teacher's `BBoxDecoder` at ``score_threshold``.
        class_map: Per teacher category, the student category id (``-1``: none).
        score_threshold: Teacher detections below this score are not targets.
        target_encoder: The student's own target encoder.
    """

    def __init__(
        self,
        teacher: nn.Module,
        teacher_type: str,
        teacher_decoder: Any,
        class_map: List[int],
        target_encoder: Any,
        score_threshold: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        supervised_weight: Optional[float] = None,
        decay: Optional[str] = None,
        teacher_input_scale: float = 1.0,
        teacher_bf16: bool = True,
        projector: Optional[nn.Module] = None,
    ):
        """Initialize from resolved settings (see `attach_distillation`)."""
        super().__init__(
            teacher,
            stage=1,
            student_head="",
            teacher_head="",
            student_idx=[],
            teacher_idx=[],
            alpha=alpha,
            beta=beta,
            supervised_weight=supervised_weight,
            decay=decay,
            teacher_input_scale=teacher_input_scale,
            teacher_bf16=teacher_bf16,
            projector=projector,
        )
        self.teacher_type = teacher_type
        self.teacher_decoder = teacher_decoder
        self.class_map = [int(c) for c in class_map]
        self.target_encoder = target_encoder
        self.score_threshold = float(score_threshold)

    @torch.no_grad()
    def run_box_teacher(self, x: torch.Tensor) -> Tuple[List, Optional[torch.Tensor]]:
        """The teacher's detections on ``x`` (in ``x``'s pixels) and its feature.

        The teacher reads ``x`` resampled by ``teacher_input_scale`` to a multiple
        of its ``max_stride``; its boxes are scaled back to ``x``.

        Args:
            x: The student's normalized input ``(B, C, H, W)``.

        Returns:
            ``([(boxes (n, 4), classes (n,)) per image], feature or None)``:
            boxes in ``x`` pixels, classes as student category ids.
        """
        stride = _max_stride(self.teacher)
        h, w = x.shape[-2:]
        th, tw = (
            max(stride, int(round(d * self.teacher_input_scale / stride)) * stride)
            for d in (h, w)
        )
        xt = x
        if (th, tw) != (h, w):
            xt = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        use_bf16 = self.teacher_bf16 and x.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            raw, t_bb = self.teacher(xt, return_backbone_outputs=True)
        raw = _float_tree(raw)
        if self.teacher_type == "bbox_centernet":
            raw = dict(raw)
            raw["BBoxCenterHead"] = torch.sigmoid(raw["BBoxCenterHead"])
        boxes, scores, classes = self.teacher_decoder(raw, img_hw=(th, tw))
        scale = boxes.new_tensor([w / tw, h / th, w / tw, h / th])
        class_map = torch.as_tensor(self.class_map, device=classes.device)
        out = []
        for b, sc, c in zip(boxes.float(), scores.float(), classes.long()):
            keep = torch.isfinite(sc) & (sc >= self.score_threshold)
            b, c = b[keep] * scale, class_map[c[keep].clamp(min=0)]
            b = torch.stack(
                [b[:, 0].clamp(0, w), b[:, 1].clamp(0, h)]
                + [b[:, 2].clamp(0, w), b[:, 3].clamp(0, h)],
                dim=-1,
            )
            keep = (c >= 0) & (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
            out.append((b[keep], c[keep]))
        feat = None
        if self.projector is not None:
            feat = backbone_feature(t_bb).float()
        return out, feat

    def pseudo_targets(
        self, detections: List, img_hw: Tuple[int, int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """The teacher's detections as a batch of the student's training targets."""
        if hasattr(self.target_encoder, "max_targets"):
            self.target_encoder.max_targets = max([1] + [len(b) for b, _ in detections])
        encoded = [
            self.target_encoder(b.cpu(), c.cpu(), tuple(img_hw)) for b, c in detections
        ]
        return {k: torch.stack([e[k] for e in encoded]).to(device) for k in encoded[0]}

    def box_loss_terms(
        self,
        student: nn.Module,
        x: torch.Tensor,
        outputs: Dict[str, Any],
        backbone_outputs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Unweighted terms ``{"box": ..., "feat": ...}`` and the teacher's box count.

        Args:
            student: The student `_BBoxLightningModel` (its ``_compute_losses``).
            x: The student's normalized training input.
            outputs: The student's raw head outputs.
            backbone_outputs: The student's backbone outputs.

        Returns:
            ``(terms, mean teacher boxes per image)``.
        """
        detections, t_feat = self.run_box_teacher(x)
        targets = self.pseudo_targets(detections, tuple(x.shape[-2:]), x.device)
        box_loss, _ = student._compute_losses(outputs, targets)
        terms = {"box": box_loss}
        if self.projector is not None and t_feat is not None:
            s_feat = backbone_feature(backbone_outputs)
            if t_feat.shape[-2:] != s_feat.shape[-2:]:
                t_feat = F.interpolate(
                    t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False
                )
            terms["feat"] = F.mse_loss(self.projector(s_feat).float(), t_feat)
        n_boxes = sum(len(b) for b, _ in detections) / max(len(detections), 1)
        return terms, float(n_boxes)

    def combine(
        self, supervised: torch.Tensor, terms: Dict[str, torch.Tensor], r: float
    ) -> torch.Tensor:
        """The loss to optimize: ``supervised_weight * L + r * (beta L_box + alpha L_feat)``."""
        kd = self.beta * terms["box"]
        if "feat" in terms:
            kd = kd + self.alpha * terms["feat"]
        total = r * kd
        if self.supervised_weight:
            total = total + self.supervised_weight * supervised
        return total


def _float_tree(outputs: Any) -> Any:
    """Nested dicts / lists of tensors with half-precision floats as float32."""
    if isinstance(outputs, torch.Tensor):
        half = outputs.dtype in (torch.float16, torch.bfloat16)
        return outputs.float() if half else outputs
    if isinstance(outputs, dict):
        return {k: _float_tree(v) for k, v in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        return [_float_tree(v) for v in outputs]
    return outputs


def _attach_box_distillation(student: nn.Module, get) -> BoxDistiller:
    """`attach_distillation` for a box-detector student (see `BoxDistiller`)."""
    from sleap_nn.config.utils import get_model_type_from_cfg
    from sleap_nn.data.bboxes import bbox_head_categories, get_bbox_target_encoder
    from sleap_nn.inference.bbox import BBoxDecoder

    if int(get("stage", 1)) != 1:
        raise ValueError(
            "Distillation of box detectors is stage 1 (teacher -> student) only; "
            "stage 2 (head self-distillation) is for confidence-map models."
        )
    if get("teacher_crop_size") is not None:
        raise ValueError(
            "distillation.teacher_crop_size is for top-down pose crops; a box "
            "detector's teacher reads the student's frame (teacher_input_scale)."
        )
    teacher, teacher_cfg = load_teacher_model(get("teacher"))
    t_type = get_model_type_from_cfg(teacher_cfg)
    if t_type not in BOX_DISTILLATION_MODEL_TYPES:
        raise ValueError(
            f"A {student.model_type} student needs a box-detector teacher "
            f"({list(BOX_DISTILLATION_MODEL_TYPES)}), not {t_type!r}."
        )
    t_heads = teacher_cfg.model_config.head_configs
    s_cats = list(student.categories)
    t_cats = bbox_head_categories(t_type, t_heads)
    class_map = [s_cats.index(c) if c in s_cats else -1 for c in t_cats]
    if all(c < 0 for c in class_map):
        if len(s_cats) == 1 and len(t_cats) == 1:
            class_map = [0]
            logger.info(
                f"Distillation: the teacher's one category {t_cats[0]!r} teaches "
                f"the student's {s_cats[0]!r}."
            )
        else:
            raise ValueError(
                f"Student and teacher share no box category ({s_cats} vs {t_cats})."
            )
    threshold = float(get("teacher_score_threshold", 0.5))
    decoder = BBoxDecoder.from_head_configs(t_type, t_heads, score_threshold=threshold)
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
    distiller = BoxDistiller(
        teacher,
        teacher_type=t_type,
        teacher_decoder=decoder,
        class_map=class_map,
        target_encoder=get_bbox_target_encoder(
            student.model_type, student.head_configs
        ),
        score_threshold=threshold,
        alpha=alpha,
        beta=get("beta", 1.0),
        supervised_weight=get("supervised_weight"),
        decay=get("decay"),
        teacher_input_scale=float(get("teacher_input_scale", 1.0)),
        teacher_bf16=get("teacher_precision", "bf16") == "bf16",
        projector=projector,
    )
    shared = sum(c >= 0 for c in class_map)
    logger.info(
        f"Box distillation (experimental) from a {t_type} teacher: {feat_note}, "
        f"{shared} shared categories, teacher boxes >= {threshold}, "
        f"alpha={distiller.alpha}, beta={distiller.beta}, "
        f"supervised_weight={distiller.supervised_weight}, decay={distiller.decay}, "
        f"teacher_input_scale={distiller.teacher_input_scale}."
    )
    student.distiller = distiller
    return distiller


def attach_distillation(
    student: nn.Module, cfg: Any, student_crop_size: Any = None
) -> Distiller:
    """Build a `Distiller` from ``trainer_config.distillation`` and attach it.

    Call before the optimizer is built (``trainer.fit``): stage 1 adds the
    projector's parameters and stage 2 changes what is trainable.

    Args:
        student: The `LightningModel` to train.
        cfg: A `DistillationConfig` (or its DictConfig).
        student_crop_size: The student's input crop size (int or ``(h, w)``),
            for the heatmap-width check with ``teacher_crop_size``; ``None``
            skips that check.

    Returns:
        The distiller, also set as ``student.distiller``.

    Box detectors (``bbox_*``) get a `BoxDistiller` (stage 1, a box-detector
    teacher).

    Raises:
        ValueError: For an unsupported model type, skeletons with no shared node,
            or a stage-2 ``teacher`` that does not fully initialize the student,
            or a ``teacher_crop_size`` that is not a multiple of the teacher's
            ``max_stride`` or comes with a ``teacher_input_scale``.
    """
    if student.model_type not in DISTILLATION_MODEL_TYPES:
        raise ValueError(
            f"Distillation supports {list(DISTILLATION_MODEL_TYPES)}, not "
            f"{student.model_type!r}."
        )

    def get(key, default=None):
        return _cfg_get(cfg, key, default)

    if student.model_type in BOX_DISTILLATION_MODEL_TYPES:
        return _attach_box_distillation(student, get)

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
        teacher_crop = get("teacher_crop_size")
        if teacher_crop is not None:
            from sleap_nn.data.instance_cropping import crop_hw

            if scale != 1.0:
                raise ValueError(
                    "distillation: set teacher_crop_size or teacher_input_scale, "
                    "not both."
                )
            teacher_crop = crop_hw(teacher_crop)
            stride = _max_stride(teacher)
            if teacher_crop[0] % stride or teacher_crop[1] % stride:
                raise ValueError(
                    f"distillation.teacher_crop_size {list(teacher_crop)} must be a "
                    f"multiple of the teacher's max_stride {stride}."
                )
            student_hw = crop_hw(student_crop_size)
            # The teacher's outputs are resized from its crop to the student's.
            scale = teacher_crop[0] / student_hw[0] if student_hw[0] else None
        if scale is not None:
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
            teacher_input_scale=1.0 if teacher_crop is not None else scale,
            teacher_crop_size=teacher_crop,
            mask_unlabeled_nodes=get("mask_unlabeled_nodes", True),
            teacher_bf16=get("teacher_precision", "bf16") == "bf16",
            projector=projector,
        )
        logger.info(
            f"Distillation stage 1: {feat_note}, {len(s_idx)} shared nodes, "
            f"alpha={distiller.alpha}, beta={distiller.beta}, "
            f"supervised_weight={distiller.supervised_weight}, "
            f"decay={distiller.decay}, "
            + (
                f"teacher_crop_size={list(teacher_crop)}."
                if teacher_crop is not None
                else f"teacher_input_scale={scale}."
            )
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
