"""Import RF-DETR detection checkpoints as sleap-nn ``bbox_detr`` model directories.

Converts a whole ``Roboflow/rf-detr-{nano,small,medium,large}`` HF Hub checkpoint
(Apache-2.0) -- windowed DINOv2 backbone, projector, decoder and heads -- into a
directory that ``sleap-nn predict`` loads like any trained ``bbox_detr`` model:

* ``training_config.yaml``: a ``dinov2_windowed`` backbone and a ``bbox_detr`` head
  with the checkpoint's sizes, and ``resize_mode: squash`` to the checkpoint's
  square resolution (RF-DETR's preprocessing).
* ``best.ckpt``: the converted weights (``state_dict`` with ``model.`` keys).

Classes. RF-DETR's COCO head has 91 sigmoid logits indexed by COCO category id
(0 and the ten ids COCO skips are unused ``N/A`` slots), and every logit is a
class. sleap-nn's set head has ``len(categories) + 1`` logits and never predicts
the last one, so the importer keeps all 91 as categories -- category index =
COCO category id; unused slots are named ``N/A_<id>`` -- and appends one extra
logit with zero weights and a very negative bias.

Weights. The Hub checkpoints store the original ``rfdetr`` parameter names
(``backbone.0.encoder.encoder.*``, ``transformer.decoder.*``, ...); HF-format
names (``model.backbone.backbone.*``, as ``save_pretrained`` writes them) are
accepted too. sleap-nn's modules use HF's names, so both map by renaming, except
the decoder self-attention's packed ``in_proj`` which is split into q/k/v.

Only detection checkpoints with one projector level and the standard decoder are
supported; segmentation and keypoint checkpoints are refused.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from omegaconf import OmegaConf

#: Short names of the Apache-licensed RF-DETR detection checkpoints on the HF Hub.
RFDETR_MODELS = {
    "nano": "Roboflow/rf-detr-nano",
    "small": "Roboflow/rf-detr-small",
    "medium": "Roboflow/rf-detr-medium",
    "large": "Roboflow/rf-detr-large",
}

#: Bias of the appended never-predicted logit (sigmoid ~ 4e-44).
PAD_CLASS_LOGIT = -100.0

#: Renames inside the backbone (original ``rfdetr`` -> HF / sleap-nn names).
_BACKBONE_RENAMES = (
    ("attention.attention.query.", "attention.q_proj."),
    ("attention.attention.key.", "attention.k_proj."),
    ("attention.attention.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
)

#: Regex renames of the original ``rfdetr`` names after the backbone prefix.
_RFDETR_RENAMES = (
    (r"^backbone\.0\.projector\.stages\.0\.0\.", "detr.projector.projector_layer."),
    (r"^backbone\.0\.projector\.stages\.0\.1\.", "detr.projector.layer_norm."),
    (r"\.m\.(\d+)\.", r".bottlenecks.\1."),
    (r"\.cv1\.", ".conv1."),
    (r"\.cv2\.", ".conv2."),
    (r"\.bn\.", ".norm."),
    (r"^refpoint_embed\.", "detr.reference_point_embed."),
    (r"^query_feat\.", "detr.query_feat."),
    (r"^transformer\.decoder\.norm\.", "detr.decoder.layernorm."),
    (r"\.self_attn\.out_proj\.", ".self_attn.o_proj."),
    (r"(layers\.\d+)\.norm1\.", r"\1.self_attn_layer_norm."),
    (r"(layers\.\d+)\.norm2\.", r"\1.cross_attn_layer_norm."),
    (r"(layers\.\d+)\.norm3\.", r"\1.layer_norm."),
    (r"(layers\.\d+)\.linear1\.", r"\1.mlp.fc1."),
    (r"(layers\.\d+)\.linear2\.", r"\1.mlp.fc2."),
    (r"^transformer\.", "detr."),
    (r"^(class_embed|bbox_embed)\.", r"detr.\1."),
)


def rfdetr_key_to_sleap_nn(key: str) -> str:
    """Map one RF-DETR parameter name to its sleap-nn ``Model`` name.

    Args:
        key: An original ``rfdetr`` name or an HF ``RfDetrForObjectDetection``
            name. The packed decoder ``self_attn.in_proj_*`` is not renamed here
            (see :func:`convert_rfdetr_state_dict`).

    Returns:
        The name under ``Model(model_type="bbox_detr")``: ``backbone.*`` or
        ``detr.*``.
    """
    # HF format.
    if key.startswith("model.backbone.backbone."):
        return "backbone." + key[len("model.backbone.backbone.") :]
    if key.startswith("model.backbone."):
        return "detr." + key[len("model.backbone.") :]
    if key.startswith("model."):
        return "detr." + key[len("model.") :]
    if key.startswith(("class_embed.", "bbox_embed.")):
        return "detr." + key
    # Original rfdetr format.
    if key.startswith("backbone.0.encoder.encoder."):
        key = "backbone." + key[len("backbone.0.encoder.encoder.") :]
        for old, new in _BACKBONE_RENAMES:
            key = key.replace(old, new)
        return key
    for pattern, repl in _RFDETR_RENAMES:
        key = re.sub(pattern, repl, key)
    return key


def convert_rfdetr_state_dict(
    state_dict: Dict[str, torch.Tensor], pad_class_logit: bool = True
) -> Dict[str, torch.Tensor]:
    """Convert an RF-DETR state dict to a sleap-nn ``bbox_detr`` ``Model`` one.

    Args:
        state_dict: Tensors of an RF-DETR detection checkpoint (original
            ``rfdetr`` or HF names).
        pad_class_logit: Append the never-predicted last class logit (zero
            weights, bias :data:`PAD_CLASS_LOGIT`) to the decoder and encoder class
            heads, so every RF-DETR logit becomes a sleap-nn category.

    Returns:
        The converted state dict (keys without the Lightning ``model.`` prefix).

    Raises:
        ValueError: If the checkpoint has segmentation or keypoint parameters.
    """
    unsupported = [k for k in state_dict if "segmentation_head" in k or "keypoint" in k]
    if unsupported:
        raise ValueError(
            "Only RF-DETR detection checkpoints can be imported; this one has "
            f"segmentation/keypoint parameters (e.g. {unsupported[0]})."
        )
    out = {}
    for key, value in state_dict.items():
        match = re.match(r"^(.*self_attn\.)in_proj_(weight|bias)$", key)
        if match:
            prefix = rfdetr_key_to_sleap_nn(match.group(1) + "x").removesuffix("x")
            for name, part in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3)):
                out[f"{prefix}{name}.{match.group(2)}"] = part.clone()
            continue
        out[rfdetr_key_to_sleap_nn(key)] = value
    if pad_class_logit:
        for key in list(out):
            if not re.search(r"(^detr\.class_embed|enc_out_class_embed\.\d+)\.", key):
                continue
            value = out[key]
            if key.endswith(".weight"):
                pad = value.new_zeros(1, value.shape[1])
            else:
                pad = value.new_full((1,), PAD_CLASS_LOGIT)
            out[key] = torch.cat([value, pad], dim=0)
    return out


def rfdetr_categories(hf_config: dict) -> List[str]:
    """Category names by class index (= COCO category id for COCO checkpoints).

    Unused ``N/A`` slots become ``N/A_<index>`` so every name is unique.
    """
    id2label = {int(k): v for k, v in hf_config["id2label"].items()}
    names = []
    for i in range(len(id2label)):
        name = id2label.get(i, "N/A")
        names.append(f"N/A_{i}" if name == "N/A" else name)
    return names


def _check_supported(hf_config: dict) -> None:
    """Refuse RF-DETR configs the native modules cannot reproduce."""
    archs = hf_config.get("architectures") or ["RfDetrForObjectDetection"]
    backbone = hf_config["backbone_config"]
    problems = []
    if archs != ["RfDetrForObjectDetection"]:
        problems.append(f"architectures={archs} (detection only)")
    if hf_config.get("model_type", "rf_detr") != "rf_detr":
        problems.append(f"model_type={hf_config.get('model_type')!r}")
    if list(hf_config.get("projector_scale_factors", [1.0])) != [1.0]:
        problems.append(
            f"projector_scale_factors={hf_config['projector_scale_factors']}"
        )
    if hf_config.get("num_feature_levels", 1) != 1:
        problems.append(f"num_feature_levels={hf_config['num_feature_levels']}")
    if backbone.get("model_type", "rf_detr_dinov2") != "rf_detr_dinov2":
        problems.append(f"backbone model_type={backbone.get('model_type')!r}")
    if backbone.get("use_swiglu_ffn", False):
        problems.append("use_swiglu_ffn=True")
    if backbone.get("num_register_tokens", 0):
        problems.append(f"num_register_tokens={backbone['num_register_tokens']}")
    if not backbone.get("apply_layernorm", True):
        problems.append("apply_layernorm=False")
    if hf_config.get("activation_function", "silu") != "silu":
        problems.append(f"activation_function={hf_config['activation_function']!r}")
    if hf_config.get("decoder_activation_function", "relu") != "relu":
        problems.append(
            "decoder_activation_function="
            f"{hf_config['decoder_activation_function']!r}"
        )
    if problems:
        raise ValueError(
            "Unsupported RF-DETR checkpoint for bbox_detr: " + "; ".join(problems)
        )


def _as_int(value) -> int:
    """An int from an int or a square ``[h, w]`` size."""
    if isinstance(value, (list, tuple)):
        if len(set(value)) != 1:
            raise ValueError(f"Only square sizes are supported, got {value}.")
        value = value[0]
    return int(value)


def rfdetr_config_to_sleap_nn(hf_config: dict) -> Tuple[dict, dict, int]:
    """The sleap-nn backbone and head settings of an RF-DETR config.

    Args:
        hf_config: The checkpoint's ``config.json``.

    Returns:
        ``(backbone, detr, resolution)``: ``Dinov2WindowedConfig`` and
        ``DetrHeadConfig`` keyword arguments, and the square input resolution.

    Raises:
        ValueError: If the checkpoint is not a supported RF-DETR detector.
    """
    _check_supported(hf_config)
    bb = hf_config["backbone_config"]
    patch = _as_int(bb["patch_size"])
    resolution = _as_int(bb["image_size"])
    num_windows = int(bb.get("num_windows", 1))
    if resolution % (patch * num_windows):
        raise ValueError(
            f"RF-DETR resolution {resolution} is not a multiple of patch {patch} x "
            f"{num_windows} windows."
        )
    backbone = dict(
        in_channels=int(bb.get("num_channels", 3)),
        hidden_size=int(bb["hidden_size"]),
        num_layers=int(bb["num_hidden_layers"]),
        num_heads=int(bb["num_attention_heads"]),
        mlp_ratio=float(bb.get("mlp_ratio", 4)),
        patch_size=patch,
        pos_embed_grid=resolution // patch,
        out_indices=[int(i) for i in bb["out_indices"]],
        num_windows=num_windows,
        window_block_indexes=[int(i) for i in bb["window_block_indexes"]],
        output_stride=patch,
        max_stride=patch * num_windows,
    )
    detr = dict(
        categories=rfdetr_categories(hf_config),
        output_stride=patch,
        hidden_dim=int(hf_config["d_model"]),
        num_queries=int(hf_config["num_queries"]),
        group_detr=int(hf_config["group_detr"]),
        dec_layers=int(hf_config["decoder_layers"]),
        sa_nheads=int(hf_config["decoder_self_attention_heads"]),
        ca_nheads=int(hf_config["decoder_cross_attention_heads"]),
        dec_n_points=int(hf_config["decoder_n_points"]),
        dim_feedforward=int(hf_config["decoder_ffn_dim"]),
        c2f_num_blocks=int(hf_config["c2f_num_blocks"]),
        c2f_expansion=float(hf_config["hidden_expansion"]),
        num_select=int(hf_config["num_queries"]),
    )
    return backbone, detr, resolution


def rfdetr_training_config(
    hf_config: dict, name: str = "", description: str = ""
) -> "OmegaConf":
    """A sleap-nn training config describing an imported RF-DETR model.

    Args:
        hf_config: The checkpoint's ``config.json``.
        name: Config name.
        description: Config description (e.g. the source checkpoint).

    Returns:
        The ``TrainingJobConfig`` as an OmegaConf object.
    """
    from sleap_nn.config.data_config import DataConfig, PreprocessingConfig
    from sleap_nn.config.model_config import (
        BackboneConfig,
        BBoxDetrConfig,
        DetrHeadConfig,
        Dinov2WindowedConfig,
        HeadConfig,
        ModelConfig,
    )
    from sleap_nn.config.training_job_config import TrainingJobConfig

    backbone, detr, resolution = rfdetr_config_to_sleap_nn(hf_config)
    config = TrainingJobConfig(
        data_config=DataConfig(
            preprocessing=PreprocessingConfig(
                ensure_rgb=True,
                max_height=resolution,
                max_width=resolution,
                resize_mode="squash",
                scale=1.0,
            ),
        ),
        model_config=ModelConfig(
            init_weights="default",
            backbone_config=BackboneConfig(
                dinov2_windowed=Dinov2WindowedConfig(**backbone)
            ),
            head_configs=HeadConfig(
                bbox_detr=BBoxDetrConfig(detr=DetrHeadConfig(**detr))
            ),
        ),
        name=name,
        description=description,
    )
    return config.to_sleap_nn_cfg()


def _fetch(
    model: Union[str, Path], revision: Optional[str]
) -> Tuple[dict, Dict[str, torch.Tensor], str]:
    """Load ``config.json`` and the weights of a Hub id, short name or local dir."""
    from safetensors.torch import load_file

    model = RFDETR_MODELS.get(str(model), str(model))
    directory = Path(model)
    if directory.is_dir():
        config_path = directory / "config.json"
        weights_path = directory / "model.safetensors"
        source = directory.resolve().as_posix()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - depends on the env
            raise ImportError(
                "Downloading RF-DETR checkpoints needs huggingface_hub; install "
                "sleap-nn[backbones]."
            ) from e
        config_path = Path(hf_hub_download(model, "config.json", revision=revision))
        weights_path = hf_hub_download(model, "model.safetensors", revision=revision)
        # The snapshot directory is named by the resolved commit.
        source = f"{model}@{config_path.parent.name}"
    hf_config = json.loads(Path(config_path).read_text())
    return hf_config, load_file(str(weights_path)), source


def import_rfdetr(
    model: Union[str, Path],
    output_dir: Union[str, Path],
    revision: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Import an RF-DETR detection checkpoint as a sleap-nn model directory.

    Args:
        model: A short name (``nano``, ``small``, ``medium``, ``large``), an HF Hub
            id (``Roboflow/rf-detr-medium``) or a local directory with
            ``config.json`` and ``model.safetensors``.
        output_dir: Directory to write ``training_config.yaml`` and ``best.ckpt``
            to (created if needed).
        revision: Optional HF Hub revision.
        overwrite: Replace an existing ``best.ckpt`` in ``output_dir``.

    Returns:
        The output directory.

    Raises:
        FileExistsError: If ``output_dir`` already holds a model and
            ``overwrite`` is off.
        ValueError: If the checkpoint is not a supported RF-DETR detector, or its
            converted weights do not load strictly into the configured model.
    """
    import lightning

    from sleap_nn.architectures.model import Model

    output_dir = Path(output_dir)
    if (output_dir / "best.ckpt").exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already has a best.ckpt; pass overwrite=True (--overwrite) "
            "to replace it."
        )
    hf_config, state_dict, source = _fetch(model, revision)
    config = rfdetr_training_config(
        hf_config,
        name=f"rfdetr-{Path(str(model)).name}",
        description=f"Imported from RF-DETR checkpoint {source}.",
    )
    state_dict = convert_rfdetr_state_dict(state_dict)

    # Load strictly into the configured model, so a config/weights mismatch fails
    # here and not at predict time.
    net = Model(
        backbone_type="dinov2_windowed",
        backbone_config=config.model_config.backbone_config.dinov2_windowed,
        head_configs=config.model_config.head_configs.bbox_detr,
        model_type="bbox_detr",
    )
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise ValueError(f"RF-DETR weights do not fit the bbox_detr model: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, (output_dir / "training_config.yaml").as_posix())
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in net.state_dict().items()},
            "pytorch-lightning_version": lightning.__version__,
            "epoch": 0,
            "global_step": 0,
            "rfdetr_source": source,
        },
        output_dir / "best.ckpt",
    )
    logger.info(f"Imported {source} into {output_dir}.")
    return output_dir
