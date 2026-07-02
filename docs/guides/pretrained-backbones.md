# Pretrained backbones (HuggingFace)

Reuse an external **pretrained image backbone** — a ConvNeXtV2, ResNet, Swinv2,
or DINOv2 encoder from HuggingFace `transformers` — as the backbone of any
sleap-nn model, frozen or fine-tuned. This lets pose, centroid, and segmentation
models start from ImageNet / foundation-model features instead of random init,
which usually helps most on small labeled datasets.

It plugs into the existing `Model`/backbone path: a new `pretrained` member on
`backbone_config` selects the HuggingFace encoder, which is wrapped to emit the
same backbone contract as the native UNet/ConvNeXt/SwinT backbones. No head
changes, no new model type.

## Install

`transformers` is an optional dependency (the `backbones` extra), imported only
when a `pretrained` backbone is requested:

```bash
pip install "sleap-nn[backbones]"      # or: uv sync --extra backbones
```

Native HuggingFace backbones (ConvNeXt/Swin/ResNet/DINOv2) need `transformers`
core only — no `timm`, no extra packages.

## Two modes: decoder (Case A) vs. encoder (Case B)

A pretrained encoder is used one of two ways, auto-selected from the model
family (override with `mode`):

| Mode | `mode` | Backbone family | Emits | For |
|------|--------|-----------------|-------|-----|
| **Case A — hierarchical + decoder** | `decoder` | ConvNeXtV2, ResNet, Swinv2, DINOv3-ConvNeXt | multi-scale pyramid → sleap-nn decoder | spatial heads: pose, centroid, segmentation |
| **Case B — encoder-only pooled** | `encoder` | DINOv2, DINOv2-with-registers (isotropic ViT) | single pooled bottleneck | pooled heads: class-vectors / re-ID |

**Why the split?** Hierarchical backbones emit a stride-4/8/16/32 feature
pyramid that maps cleanly onto the decoder's skip connections. Plain ViTs are
*isotropic* — every stage is a single stride (14 or 16) — so they cannot supply
a U-Net-style pyramid; they are used encoder-only. A ViT feeding a spatial head
would need a ViTDet-style Simple Feature Pyramid, which is not yet implemented.

`mode: auto` picks `encoder` for isotropic ViTs and `decoder` for everything
else.

## Tested model families

Verified to build and train through the sleap-nn pipeline (`weights=false` builds
are exercised network-free in CI; the ✓ families were run end-to-end locally):

| Model | Example `model_name` | Family | Strides | Mode | License / gated | Notes |
|-------|----------------------|--------|---------|------|-----------------|-------|
| **ConvNeXtV2** | `facebook/convnextv2-nano-22k-224` | CNN | 4/8/16/32 | decoder | Apache-2.0 / no | ✓ recommended default for pose/seg |
| **ResNet** | `microsoft/resnet-50` | CNN | 4/8/16/32 | decoder | Apache-2.0 / no | ✓ legacy-SLEAP parity target |
| **Swinv2** | `microsoft/swinv2-tiny-patch4-window8-256` | hier. transformer | 4/8/16/32 | decoder | MIT / no | builds; ONNX export is fragile (see below) |
| **DINOv2** | `facebook/dinov2-base` | ViT (isotropic) | 14 | encoder | Apache-2.0 / no | pooled bottleneck; patch-14 (see gotchas) |
| **DINOv2-with-registers** | `facebook/dinov2-with-registers-base` | ViT (isotropic) | 14 | encoder | Apache-2.0 / no | ✓ recommended for pooled/re-ID heads |
| **DINOv3-ConvNeXt** | `facebook/dinov3-convnext-base-…` | CNN | 4/8/16/32 | decoder | DINOv3 custom / **gated** | foundation SSL + pyramid; opt-in |
| **DINOv3-ViT** | `facebook/dinov3-vit…16-…` | ViT (isotropic) | 16 | encoder | DINOv3 custom / **gated** | patch-16, RoPE (resolution-agnostic) |

Any other `AutoBackbone`-compatible model id should work; the wrapper probes
strides and channels at construction, so it is family-agnostic.

## Config

```yaml
model_config:
  init_weights: xavier
  backbone_config:
    unet: null
    convnext: null
    swint: null
    pretrained:
      source: hf
      model_name: facebook/convnextv2-nano-22k-224
      weights: true       # download & load pretrained weights
      mode: auto          # auto | decoder | encoder
      freeze: false       # true = freeze encoder (feature extraction)
      normalize: true     # apply the model's image_mean/std
      revision: null      # pin an HF commit sha/tag for reproducibility
      in_channels: 3      # HF stems are 3-channel (grayscale is replicated)
      output_stride: 2    # finest decoder output (decoder mode)
      max_stride: 32      # deepest encoder stride
  head_configs:
    bottomup_segmentation:
      segmentation: {output_stride: 2, loss_weight: 1.0}
      center:       {sigma: 4.0, output_stride: 2, loss_weight: 1.0}
      offsets:      {output_stride: 2, loss_weight: 0.005}
```

Train exactly as usual — the backbone is selected purely from config:

```bash
sleap-nn train config.yaml \
  data_config.train_labels_path="[train.pkg.slp]" \
  data_config.val_labels_path="[val.pkg.slp]"
```

A ready-to-edit sample lives at
[`config_bottomup_segmentation_pretrained.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bottomup_segmentation_pretrained.yaml).

### Freeze vs. fine-tune

- `freeze: false` (default) — the whole backbone fine-tunes end-to-end. Best
  accuracy; recommended when you have enough labeled data.
- `freeze: true` — the encoder is frozen (feature extraction) and only the
  decoder/head train. Faster, lower memory, and more robust on very small
  datasets. sleap-nn filters frozen parameters out of the optimizer
  automatically.

### Channels and normalization

Pretrained stems are **3-channel**. The trainer sets `in_channels=3` and
`ensure_rgb=true` automatically for a `pretrained` backbone, so grayscale videos
are replicated to 3 channels. Model-specific mean/std normalization (read from
the model's `AutoImageProcessor`, or set explicitly via `image_mean`/`image_std`)
is applied **inside the backbone** — the data pipeline still only rescales to
`[0, 1]`.

## Reproducibility, gating, and offline use

- **Pin `revision`.** HuggingFace `main` moves. Set `revision` to a commit sha
  or tag so a re-run loads identical weights.
- **Gated models (DINOv3, SAM3).** These live in gated HF repos with a custom
  license. Request access, then authenticate before training:

  ```bash
  huggingface-cli login          # or: export HF_TOKEN=hf_...
  ```

  DINOv3's license is non-OSI (commercial use permitted with attribution) —
  review it before bundling. Hiera / I-JEPA weights are `cc-by-nc-4.0`
  (non-commercial).
- **Offline / air-gapped.** Pre-seed the HuggingFace cache
  (`~/.cache/huggingface/hub`) on a networked machine, copy it over, and set
  `HF_HUB_OFFLINE=1`. With `weights: false` the architecture still needs the
  model's `config.json` (tiny), but no checkpoint download.

## Gotchas

- **DINOv2 is patch-14.** Its stride (14) does not divide the 16/32-padded crops
  sleap-nn produces. For pooled (encoder) use this is fine; if you need a
  ÷14-friendly crop, size crops to a multiple of 14, or use DINOv3-ViT
  (patch-16). ConvNeXt/ResNet/Swin avoid the issue entirely.
- **`output_stride=2` on a hierarchical HF backbone.** These backbones bottom out
  at stride 4 (no stride-2 feature), so reaching `output_stride=2` adds one
  learned-upsample decoder block with no skip connection — a small
  localization-quality nuance, not a blocker. `output_stride=4` uses skips at
  every level.
- **ONNX export.** ViT/DINOv2/ConvNeXt/ResNet export cleanly (opset-17). Swin is
  export-fragile: a graph can pass the structural `onnx.checker` yet be
  numerically wrong. The exporter's optional `numerical_check` runs a
  torch-vs-onnxruntime parity assert for exactly this case — enable it when
  exporting transformer backbones.
- **`dtype`.** `transformers` v5 defaults to `dtype="auto"`, which can silently
  load fp16/bf16. sleap-nn forces `float32` in the backbone factory.
