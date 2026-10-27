"""Generate the HF transformers fixtures for ``sleap_nn.convert.rfdetr``.

Writes to ``tests/assets/detr/``:

* ``rfdetr_tiny/``: a tiny ``RfDetrForObjectDetection`` (windowed DINOv2 backbone
  included) with every parameter redrawn at random, saved with
  ``save_pretrained`` (``config.json`` + ``model.safetensors``, which HF writes
  with the original ``rfdetr`` parameter names, as on the Hub), plus
  ``hf_outputs.pt``: inputs and HF's outputs (final, per decoder layer, encoder
  proposals) for a few cases. ``test_rfdetr.py`` imports the directory with
  ``import_rfdetr`` and checks the outputs without transformers.
* ``rfdetr_small_reference.pt``: HF's float64 outputs of the real
  ``Roboflow/rf-detr-small`` on a synthetic image (first ``KEEP`` queries), for
  the download test that runs with ``SLEAP_NN_TEST_HF_WEIGHTS=1``.

It does not import sleap_nn, so it runs in any env with torch and
transformers >= 5.20 (which has ``rf_detr``). From the repo root, CPU only::

    CUDA_VISIBLE_DEVICES= /path/to/venv/bin/python tests/convert/make_rfdetr_fixture.py
"""

import math
import shutil
from pathlib import Path

import torch

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "detr"
TINY_DIR = ASSETS / "rfdetr_tiny"
SMALL_REFERENCE = ASSETS / "rfdetr_small_reference.pt"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

TINY_CONFIG = dict(
    backbone_config=dict(
        model_type="rf_detr_dinov2",
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        out_indices=[2, 4],  # windowed blocks 0, 1, 3; block 2 global
        image_size=64,  # 4x4 position-embedding grid at patch 16
        patch_size=16,
        num_windows=2,
    ),
    num_labels=5,
    d_model=16,
    decoder_ffn_dim=32,
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
WEIGHT_SEED = 0

# Each case: (input seed, square side, batch, training). 96 px interpolates the
# position embedding (6x6 grid from 4x4).
CASES = [
    dict(seed=1, side=64, batch=2, training=False),
    dict(seed=2, side=96, batch=1, training=False),
    dict(seed=3, side=64, batch=2, training=True),
]

SMALL_SIDE = 512
#: Queries of each output kept in the real-weights reference.
KEEP = 32


def synthetic_image(side: int) -> torch.Tensor:
    """A deterministic smooth ``(1, 3, side, side)`` image in ``[0, 1]``."""
    t = torch.arange(side, dtype=torch.float64) / side
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    channels = [
        0.5 + 0.5 * torch.sin(2 * math.pi * (3 * xx + 2 * yy)),
        0.5 + 0.5 * torch.cos(2 * math.pi * (5 * xx * yy + yy)),
        (xx + yy) / 2,
    ]
    return torch.stack(channels)[None].float()


def _outputs(model, pixel_values):
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
        layer_logits, layer_boxes = model.predict_class_and_boxes(
            out.intermediate_hidden_states, out.intermediate_reference_points[-1]
        )
    return dict(
        pred_logits=out.logits,
        pred_boxes=out.pred_boxes,
        layer_logits=layer_logits,
        layer_boxes=layer_boxes,
        enc_logits=out.enc_outputs_class,
        enc_boxes=out.enc_outputs_coord_logits,
    )


def make_tiny():
    from transformers import RfDetrConfig, RfDetrForObjectDetection

    model = RfDetrForObjectDetection(RfDetrConfig(**TINY_CONFIG))
    generator = torch.Generator().manual_seed(WEIGHT_SEED)
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    if TINY_DIR.exists():
        shutil.rmtree(TINY_DIR)
    model.save_pretrained(TINY_DIR)

    cases = []
    for case in CASES:
        model.train(case["training"])
        generator = torch.Generator().manual_seed(case["seed"])
        side = case["side"]
        # Stored as uint8 (the test divides by 255) to keep the fixture small.
        image = torch.randint(
            0, 256, (case["batch"], 3, side, side), generator=generator
        ).to(torch.uint8)
        x = image.float() / 255
        cases.append(dict(case=case, image=image, **_outputs(model, (x - MEAN) / STD)))
    torch.save(dict(cases=cases), TINY_DIR / "hf_outputs.pt")
    size = sum(p.stat().st_size for p in TINY_DIR.iterdir()) / 1024
    print(f"Wrote {TINY_DIR} ({size:.0f} KiB)")


def make_small_reference():
    import transformers
    from transformers import RfDetrForObjectDetection

    model = RfDetrForObjectDetection.from_pretrained(
        "Roboflow/rf-detr-small", attn_implementation="eager"
    )
    model = model.eval().double()
    x = synthetic_image(SMALL_SIDE)
    out = _outputs(model, (x.double() - MEAN.double()) / STD.double())
    keep = {k: v[..., :KEEP, :].float().clone() for k, v in out.items()}
    torch.save(
        dict(
            transformers_version=transformers.__version__,
            side=SMALL_SIDE,
            keep=KEEP,
            **keep,
        ),
        SMALL_REFERENCE,
    )
    print(f"Wrote {SMALL_REFERENCE} ({SMALL_REFERENCE.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    make_tiny()
    make_small_reference()
