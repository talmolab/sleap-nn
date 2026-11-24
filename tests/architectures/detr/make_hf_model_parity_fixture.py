"""Generate the HF transformers parity fixture for `sleap_nn.architectures.detr`.

Builds tiny ``RfDetrForObjectDetection`` models with random weights (every
parameter redrawn, so zero-inits do not hide mistakes), feeds them fixed backbone
features in place of the DINOv2 backbone, and saves the non-backbone weights, the
features and every output (final, per decoder layer, encoder proposals) to
``tests/assets/detr/hf_rf_detr_model_parity.pt``. ``test_detr_hf_parity.py``
loads the weights into ``BBoxDetrHead`` by renaming and checks the outputs without
needing transformers.

It does not import sleap_nn, so it runs in any env with torch and
transformers >= 5.20 (which has ``rf_detr``). From the repo root, CPU only::

    CUDA_VISIBLE_DEVICES= /path/to/venv/bin/python \
        tests/architectures/detr/make_hf_model_parity_fixture.py
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "detr"
    / "hf_rf_detr_model_parity.pt"
)

# Tiny decoder dims shared by every case (HF config field names).
DIMS = dict(
    d_model=32,
    decoder_ffn_dim=64,
    decoder_layers=2,
    decoder_self_attention_heads=2,
    decoder_cross_attention_heads=4,
    decoder_n_points=2,
    num_queries=12,
    group_detr=3,
    c2f_num_blocks=3,
    hidden_expansion=0.5,
    dropout=0.0,
)
BACKBONE_WIDTH = 16
NUM_TAPS = 2
NUM_LABELS = 4  # = sleap-nn num_classes + 1

WEIGHT_SEED = 0

# Each case: (feature seed, token map (h, w), batch, training). All cases share one
# set of weights. A map wider than 50 tokens makes its edge proposals invalid
# (center outside (0.01, 0.99)).
CASES = [
    dict(seed=1, hw=(6, 8), batch=2, training=False),
    dict(seed=2, hw=(6, 8), batch=2, training=True),
    dict(seed=3, hw=(3, 60), batch=1, training=True),
]


class _FixedFeatures(nn.Module):
    """Stands in for the HF DINOv2 backbone: returns preset feature maps."""

    def __init__(self):
        super().__init__()
        self.features = None

    def forward(self, pixel_values):
        return SimpleNamespace(feature_maps=tuple(self.features))


def _build_model(seed: int):
    from transformers import RfDetrConfig, RfDetrForObjectDetection

    config = RfDetrConfig(
        backbone_config=dict(
            model_type="rf_detr_dinov2",
            hidden_size=BACKBONE_WIDTH,
            num_hidden_layers=NUM_TAPS,
            num_attention_heads=2,
            out_indices=list(range(1, NUM_TAPS + 1)),
            image_size=64,
            patch_size=16,
        ),
        num_labels=NUM_LABELS,
        **DIMS,
    )
    model = RfDetrForObjectDetection(config)
    model.model.backbone.backbone = _FixedFeatures()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    return model


def _run_case(model, case):
    model.train(case["training"])
    h, w = case["hw"]
    generator = torch.Generator().manual_seed(case["seed"])
    features = [
        torch.randn(case["batch"], BACKBONE_WIDTH, h, w, generator=generator)
        for _ in range(NUM_TAPS)
    ]
    model.model.backbone.backbone.features = features
    pixel_values = torch.zeros(case["batch"], 3, h * 16, w * 16)
    with torch.no_grad():
        out = model(pixel_values)
        layer_logits, layer_boxes = model.predict_class_and_boxes(
            out.intermediate_hidden_states, out.intermediate_reference_points[-1]
        )
    return dict(
        case=case,
        features=features,
        pred_logits=out.logits,
        pred_boxes=out.pred_boxes,
        layer_logits=layer_logits,
        layer_boxes=layer_boxes,
        enc_logits=out.enc_outputs_class,
        enc_boxes=out.enc_outputs_coord_logits,
    )


def main():
    import transformers

    model = _build_model(WEIGHT_SEED)
    state_dict = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if not k.startswith("model.backbone.backbone.")
    }
    fixture = dict(
        transformers_version=transformers.__version__,
        dims=DIMS,
        backbone_width=BACKBONE_WIDTH,
        num_taps=NUM_TAPS,
        num_labels=NUM_LABELS,
        state_dict=state_dict,
        cases=[_run_case(model, case) for case in CASES],
    )
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, FIXTURE_PATH)
    print(f"Wrote {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
