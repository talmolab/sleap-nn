"""Generate the HF transformers parity fixture for `sleap_nn.training.detr`.

HF's ``RfDetrForObjectDetection`` trains with the LW-DETR loss
(``transformers.loss.loss_lw_detr``: ``LwDetrHungarianMatcher`` + ``LwDetrImageLoss``),
so that is what this script runs. It builds seeded random outputs and targets, runs
HF's matcher and criterion on them, and saves inputs, match indices and loss values
to ``tests/assets/detr/hf_lw_detr_parity.pt``, which
``test_detr_parity.py`` checks against sleap-nn without needing transformers.

It does not import sleap_nn, so it runs in any env with torch, scipy, Pillow and
transformers >= 5.20 (which has ``rf_detr``). From the repo root, CPU only::

    CUDA_VISIBLE_DEVICES= /path/to/venv/bin/python tests/training/detr/make_hf_parity_fixture.py
"""

from pathlib import Path

import torch

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "detr" / "hf_lw_detr_parity.pt"
)

# Each case: (seed, target counts per image, queries per group, groups, aux layers,
# num_classes, training).
CASES = [
    # Group DETR in training, one image without targets, aux + encoder layers.
    dict(
        seed=0,
        sizes=[3, 0, 5],
        group_size=8,
        groups=3,
        aux_layers=2,
        num_classes=3,
        training=True,
    ),
    # Same shapes in eval: one group of all queries, normalizer without groups.
    dict(
        seed=1,
        sizes=[3, 0, 5],
        group_size=8,
        groups=3,
        aux_layers=2,
        num_classes=3,
        training=False,
    ),
    # More targets than queries per group (rectangular assignment).
    dict(
        seed=2,
        sizes=[6, 2],
        group_size=4,
        groups=2,
        aux_layers=1,
        num_classes=1,
        training=True,
    ),
]


def _random_boxes(n: int, generator: torch.Generator) -> torch.Tensor:
    """Normalized cxcywh boxes with positive size."""
    centers = 0.1 + 0.8 * torch.rand(n, 2, generator=generator)
    sizes = 0.05 + 0.45 * torch.rand(n, 2, generator=generator)
    return torch.cat([centers, sizes], dim=-1)


def _random_layer(batch_size, num_queries, num_logits, generator):
    """One layer's outputs in sleap-nn naming."""
    return {
        "pred_logits": 2.0
        * torch.randn(batch_size, num_queries, num_logits, generator=generator),
        "pred_boxes": _random_boxes(batch_size * num_queries, generator).view(
            batch_size, num_queries, 4
        ),
    }


def make_case(
    seed, sizes, group_size, groups, aux_layers, num_classes, training
) -> dict:
    """Build seeded random outputs and targets in sleap-nn naming.

    ``pred_logits`` has ``num_classes + 1`` columns (sleap-nn's class-head
    convention); labels are drawn from ``[0, num_classes)``.
    """
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(sizes)
    num_queries = group_size * groups
    num_logits = num_classes + 1
    outputs = _random_layer(batch_size, num_queries, num_logits, generator)
    outputs["aux_outputs"] = [
        _random_layer(batch_size, num_queries, num_logits, generator)
        for _ in range(aux_layers)
    ]
    outputs["enc_outputs"] = _random_layer(
        batch_size, num_queries, num_logits, generator
    )
    targets = [
        {
            "labels": torch.randint(0, num_classes, (n,), generator=generator),
            "boxes": _random_boxes(n, generator),
        }
        for n in sizes
    ]
    return {
        "outputs": outputs,
        "targets": targets,
        "group_detr": groups,
        "num_classes": num_classes,
        "training": training,
    }


def _to_hf_layer(layer: dict) -> dict:
    return {"logits": layer["pred_logits"], "pred_boxes": layer["pred_boxes"]}


def run_hf(case: dict) -> dict:
    """Run HF's LW-DETR matcher and criterion on a case.

    Returns:
        ``{"indices": {suffix: [(query_idx, target_idx), ...]}, "losses": {key: float}}``
        where suffix is ``""`` (final), ``"_<i>"`` (aux) or ``"_enc"``.
    """
    from transformers.loss.loss_lw_detr import (
        LwDetrHungarianMatcher,
        LwDetrImageLoss,
    )

    outputs, targets = case["outputs"], case["targets"]
    hf_targets = [{"class_labels": t["labels"], "boxes": t["boxes"]} for t in targets]
    hf_outputs = _to_hf_layer(outputs)
    hf_outputs["auxiliary_outputs"] = [_to_hf_layer(a) for a in outputs["aux_outputs"]]
    hf_outputs["enc_outputs"] = _to_hf_layer(outputs["enc_outputs"])

    matcher = LwDetrHungarianMatcher(class_cost=2.0, bbox_cost=5.0, giou_cost=2.0)
    criterion = LwDetrImageLoss(
        matcher=matcher,
        num_classes=case["num_classes"] + 1,
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
        group_detr=case["group_detr"],
    )
    criterion.train(case["training"])
    groups = case["group_detr"] if case["training"] else 1

    layers = {"": _to_hf_layer(outputs)}
    for i, aux in enumerate(hf_outputs["auxiliary_outputs"]):
        layers[f"_{i}"] = aux
    layers["_enc"] = hf_outputs["enc_outputs"]
    indices = {
        suffix: [(i.clone(), j.clone()) for i, j in matcher(layer, hf_targets, groups)]
        for suffix, layer in layers.items()
    }
    with torch.no_grad():
        losses = criterion(hf_outputs, hf_targets)
    return {
        "indices": indices,
        "losses": {k: float(v) for k, v in losses.items()},
    }


def main():
    """Write the fixture."""
    import transformers
    from transformers.loss.loss_lw_detr import LwDetrForObjectDetectionLoss
    from transformers.loss.loss_utils import LOSS_MAPPING

    import transformers.models.rf_detr  # noqa: F401  (needs a release with RF-DETR)

    # The fixture stands for RF-DETR's detection loss only while HF routes it here.
    assert LOSS_MAPPING["RfDetrForObjectDetection"] is LwDetrForObjectDetectionLoss

    cases = []
    for spec in CASES:
        case = make_case(**spec)
        case["expected"] = run_hf(case)
        cases.append(case)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"transformers_version": transformers.__version__, "cases": cases},
        FIXTURE_PATH,
    )
    print(f"Wrote {len(cases)} cases to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
