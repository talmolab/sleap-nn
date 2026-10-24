# Pretrained backbones (HuggingFace)

Reuse an external **pretrained image backbone** as the backbone of any sleap-nn
model, frozen or fine-tuned: a hierarchical CNN or transformer (ConvNeXtV2,
ResNet, Swinv2, DINOv3-ConvNeXt) or a plain vision transformer (DINOv2/v3, CLIP,
SigLIP, MAE, PE-Spatial, EVA02, ...), loaded from the HuggingFace Hub through
`transformers` (and `timm`, for `timm/...` checkpoints). This lets pose,
centroid, and segmentation models start from ImageNet / foundation-model
features instead of random init, which usually helps most on small labeled
datasets.

It plugs into the existing `Model`/backbone path: a `pretrained` member on
`backbone_config` selects the HuggingFace encoder, which is wrapped to emit the
same backbone contract as the native UNet/ConvNeXt/SwinT backbones. No head
changes, no new model type.

## Install

`transformers` and `timm` are optional dependencies (the `backbones` extra),
imported only when a `pretrained` backbone is requested:

```bash
pip install "sleap-nn[backbones]"      # or: uv sync --extra backbones
```

`transformers` loads every model id; `timm` is needed only for `timm/...` Hub
checkpoints (see [timm Hub checkpoints](#timm-hub-checkpoints)).

## Three cases

A pretrained encoder is wired in one of three ways. `mode: auto` (the default)
picks the case from the backbone family and the model's heads; `mode: decoder` /
`mode: encoder` force it.

| Case | Resolved `mode` | Backbone family | What the backbone emits | For |
|------|-----------------|-----------------|-------------------------|-----|
| **A — hierarchical + decoder** | `decoder` | ConvNeXtV2, ResNet, Swinv2, DINOv3-ConvNeXt | its own stride-4/8/16/32 stage pyramid → sleap-nn decoder | spatial heads: pose, centroid, PAFs, segmentation |
| **B — encoder-only pooled** | `encoder` | any; normally a plain ViT | a single bottleneck feature map, no decoder | pooled heads: `embedding` (re-ID) |
| **C — plain ViT + pyramid neck + decoder** | `decoder` | plain (isotropic) ViTs: DINOv2/v3, CLIP, SigLIP/SigLIP2, ViT, MAE, PE-Spatial, EVA02, ... | 4 tapped transformer layers → feature-pyramid neck → sleap-nn decoder | spatial heads: pose, centroid, PAFs, segmentation |

**Why a neck?** Hierarchical backbones already emit a stride-4/8/16/32 pyramid
that maps onto the decoder's skip connections. A plain ViT is *isotropic*: every
layer sits at one stride (its patch size), so on its own it cannot supply a
U-Net-style pyramid. Case C builds that pyramid from several of the ViT's layers,
in the style of DPT (Ranftl et al. 2021) and ViTDet's Simple Feature Pyramid
(Li et al. 2022):

```text
image [0,1] → normalize → resize so the token grid is (H/s, W/s)
  → ViT (output_hidden_states) → layers [3, 6, 9, 12]   (12-layer default)
  → strip prefix tokens (CLS, registers) → LayerNorm per tap → reshape to grid
  → ViTPyramidNeck → strides s/4, s/2, s, 2s  (4, 8, 16, 32 at s = 16)
  → sleap-nn decoder → heads
```

`s` is the **token stride** (`token_stride`, default 16 — see
[ViT knobs](#vit-knobs)). The input is bilinearly resampled so the token grid
always lands on that power-of-two stride (a patch-14 DINOv2 runs at stride 16),
so the rest of sleap-nn's stride bookkeeping is unchanged. Two-tower checkpoints
(CLIP, SigLIP) load only their vision tower. The tap LayerNorms and the neck are
new, randomly initialized layers that always train, like the decoder.

### How `mode: auto` resolves

| Backbone | Model has a spatial head | Pooled head only (`embedding`) |
|----------|--------------------------|--------------------------------|
| hierarchical (ConvNeXtV2, ResNet, Swinv2, DINOv3-ConvNeXt) | `decoder` → Case A | `encoder` → Case B |
| plain ViT (DINOv2/v3, CLIP, SigLIP, MAE, ...) | `decoder` → Case C | `encoder` → Case B |

A model "has a spatial head" if any head predicts a map (confidence maps, PAFs,
class maps, segmentation, centers/offsets); `multi_class_topdown` counts, since
its confidence-map head is spatial (its class-vectors head reads the neck's
coarsest level).

- **Explicit `mode: decoder` on a plain ViT** builds Case C (it used to raise a
  "single-scale feature map" error).
- **Explicit `mode: encoder` with a spatial head** is rejected at model build
  ("encoder-only backbone supports only pooled heads").
- **The `embedding` (re-ID) model type always resolves to `encoder`**, whatever
  the family. That head is a lone pooled head that reads the encoder bottleneck,
  so a decoder underneath it would be dead weight: on convnextv2-nano, 4.9 M of
  20.1 M parameters (24%) receive no gradient, which is also a DDP hazard.
  `mode: decoder` with an `embedding` head is rejected at setup for the same
  reason.

In Case B, a ViT that ships a `*Backbone` class in `transformers` (DINOv2,
DINOv2-with-registers) keeps the `AutoBackbone` path at its native stride (14),
so existing embedding checkpoints load unchanged. ViTs without one (CLIP,
SigLIP, every `timm/...` ViT) use the Case C token extractor instead, tapping the
last layer at the token stride, with no neck or decoder.

## Tested model families

The wrapper is family-agnostic: it probes strides and channels (Case A) or
reads the hidden states (Case C) rather than hard-coding per-model taps. The
families below were verified to build and run forward and backward at
non-square input sizes; ✓ marks families also trained end-to-end through
`sleap-nn train`. `weights: false` builds of representative families are
exercised in CI.

**Hierarchical (Case A):**

| Model | Example `model_name` | Loader | Strides | License / gated | Notes |
|-------|----------------------|--------|---------|-----------------|-------|
| **ConvNeXtV2** | `facebook/convnextv2-nano-22k-224` | transformers | 4/8/16/32 | Apache-2.0 / no | ✓ recommended default CNN for pose/seg |
| **ResNet** | `microsoft/resnet-50` | transformers | 4/8/16/32 | Apache-2.0 / no | ✓ legacy-SLEAP parity target |
| **Swinv2** | `microsoft/swinv2-tiny-patch4-window8-256` | transformers | 4/8/16/32 | MIT / no | builds; ONNX export is fragile (see below) |
| **DINOv3-ConvNeXt** | `timm/convnext_tiny.dinov3_lvd1689m` | timm | 4/8/16/32 | DINOv3 License / no | foundation SSL + pyramid; the `facebook/dinov3-convnext-*` repos are gated |

**Plain ViTs (Case C with a spatial head, Case B for `embedding`):**

| Model | `model_name` | Loader | Patch → token stride | License / gated | Notes |
|-------|--------------|--------|----------------------|-----------------|-------|
| **DINOv2** S/B/L | `facebook/dinov2-small`, `-base`, `-large` | transformers | 14 → 16 | Apache-2.0 / no | |
| **DINOv2-with-registers** S/B/L | `facebook/dinov2-with-registers-small`, `-base`, `-large` | transformers | 14 → 16 | Apache-2.0 / no | ✓ recommended for pooled/re-ID heads; 4 register tokens stripped |
| **DINOv3 ViT** S/B/L | `timm/vit_small_patch16_dinov3.lvd1689m`, `timm/vit_base_patch16_dinov3.lvd1689m`, `timm/vit_large_patch16_dinov3.lvd1689m` | timm | 16 → 16 | DINOv3 License / no | RoPE; the `facebook/dinov3-vit*` repos are gated |
| **CLIP** B/16 | `openai/clip-vit-base-patch16` | transformers | 16 → 16 | MIT / no | vision tower only |
| **SigLIP** B/16 | `google/siglip-base-patch16-224` | transformers | 16 → 16 | Apache-2.0 / no | vision tower only; no CLS token |
| **SigLIP2** B/16 | `google/siglip2-base-patch16-224` | transformers | 16 → 16 | Apache-2.0 / no | fixed-resolution variant (not NaFlex) |
| **ViT (ImageNet-21k)** B/16 | `google/vit-base-patch16-224-in21k` | transformers | 16 → 16 | Apache-2.0 / no | supervised baseline |
| **MAE** B/16 | `facebook/vit-mae-base` | transformers | 16 → 16 | CC-BY-NC-4.0 upstream / no | pretraining masking forced off (`mask_ratio: 0`, tokens kept in place) |
| **PE-Spatial** S/B | `timm/vit_pe_spatial_small_patch16_512.fb`, `timm/vit_pe_spatial_base_patch16_512.fb` | timm | 16 → 16 | Apache-2.0 / no | Perception Encoder, spatial-tuned |
| **EVA02** S | `timm/eva02_small_patch14_224.mim_in22k` | timm | 14 → 16 | MIT / no | |

Other checkpoints should work if they fit one of the paths:
`AutoBackbone`-compatible hierarchical models (e.g. ConvNeXt v1, BiT, FocalNet)
for Case A, and HuggingFace or `timm/...` ViTs that return hidden states and run
off their pretraining resolution (e.g. BEiT, DeiT, I-JEPA) for Case C. These are
not in the verified set. Case C runs a dummy forward and backward on a non-square
input at construction, so a model that cannot work fails at setup with a clear
error, not at the first training step. Mind the license column (Hiera and I-JEPA
weights are non-commercial).

**Known not to work:**

| Model | Why |
|-------|-----|
| SAM ViT image encoder (`facebook/sam-vit-*`) | fixed 1024×1024 positional embedding |
| Hiera (`facebook/hiera-*`) | fixed input size |
| AIMv2 224-px checkpoints (`apple/aimv2-*-224`) | fixed input size |
| SigLIP2 NaFlex (`google/siglip2-*-naflex`) | expects pre-patchified inputs |
| RADIO (`nvidia/*RADIO*`) | requires `trust_remote_code`, which sleap-nn does not enable |

## Benchmark

Every pretrained backbone beat a UNet trained from scratch with the same budget,
and three were consistently near the top: ConvNeXtV2-N, DINOv3 ConvNeXt-T, and
DINOv2-reg ViT-B. Numbers are test-split OKS mAP from a single seed per run, so
differences under about 0.02 should be read as ties.

**Protocol.** Adam at LR 1e-4, batch 8, 15k steps (50 epochs × 300), cosine LR
with a 2-epoch warmup, best-validation-loss checkpoint, seed 0. Datasets:
`fly32` (single instance, 192 × 192 gray, 1,200 training frames), `flies13`
(centered instance on a 192 px crop, 3,200 training instances) and `gerbils`
(centered instance, RGB, scale 0.5 and a 288 px crop, 340 training frames).
Centered-instance models are scored with ground-truth centroids. Timings are
fp16 forward passes on one RTX A6000 with random weights, batch of 8 crops.

| Backbone | fly32 | flies13 | gerbils | Mean gap to best | Params (M) | ms / batch of 8, 192 px | ms / batch of 8, 288 px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXtV2-N (IN-22k) | 0.876 | 0.779 | 0.299 | 0.015 | 19.9 | 6.0 | 10.6 |
| DINOv3 ConvNeXt-T | 0.889 | 0.729 | 0.331 | 0.017 | 34.9 | 5.7 | 8.5 |
| DINOv2-reg ViT-B/14 | 0.886 | 0.770 | 0.294 | 0.017 | 111.2 | 8.7 | 14.7 |
| PE-Spatial ViT-B/16 | 0.884 | 0.751 | 0.305 | 0.020 | 111.1 | 10.7 | 17.8 |
| DINOv2 ViT-L/14 | 0.883 | 0.771 | 0.281 | 0.022 | 348.1 | 17.4 | 37.3 |
| MAE ViT-B/16 | 0.877 | 0.755 | 0.291 | 0.025 | 110.4 | 8.0 | 15.9 |
| DINOv3 ViT-B/16 | 0.884 | 0.732 | 0.308 | 0.026 | 110.3 | 10.6 | 18.3 |
| SigLIP2 ViT-B/16 | 0.877 | 0.735 | 0.277 | 0.037 | 117.5 | 8.7 | 17.6 |
| DINOv3 ViT-S/16 | 0.856 | 0.736 | 0.289 | 0.039 | 27.7 | 10.2 | 10.7 |
| ViT-B/16 (IN-21k sup.) | 0.867 | 0.734 | 0.271 | 0.043 | 111.0 | 7.9 | 16.0 |
| DINOv2-reg ViT-S/14 | 0.883 | 0.730 | 0.255 | 0.044 | 28.2 | 8.3 | 8.2 |
| EVA02 ViT-S/14 (MIM) | 0.874 | 0.727 | 0.256 | 0.048 | 27.8 | 11.1 | 11.1 |
| CLIP ViT-B/16 | 0.863 | 0.737 | 0.227 | 0.057 | 110.4 | 8.5 | 17.3 |
| UNet (scratch) | 0.761 | 0.000 | 0.011 | 0.409 | 0.3 | 1.5 | 5.3 |

*Mean gap to best* is the backbone's average mAP shortfall from the best
backbone on each dataset. For reference, the long-trained UNets distributed with
these datasets reach 0.796 (fly32, trained on about as many samples) and 0.818
(flies13, about 13× as many).

**Ablations** (mAP, change from the base backbone in parentheses):

| Change | Base | fly32 | flies13 | gerbils |
| --- | --- | --- | --- | --- |
| Frozen encoder | DINOv2-reg ViT-B/14 |  | 0.758 (-0.011) | 0.244 (-0.050) |
| Frozen encoder | DINOv3 ViT-B/16 |  | 0.741 (+0.010) | 0.246 (-0.061) |
| Layer decay 0.75 | DINOv2-reg ViT-B/14 |  | 0.778 (+0.008) | 0.309 (+0.015) |
| SFP neck | DINOv2-reg ViT-S/14 |  | 0.759 (+0.029) | 0.281 (+0.026) |
| Token stride 8 | DINOv2-reg ViT-S/14 |  | 0.765 (+0.034) | 0.258 (+0.003) |
| Random init | DINOv2-reg ViT-S/14 |  | 0.555 (-0.175) | 0.107 (-0.148) |
| Detail stem | DINOv2-reg ViT-S/14 | 0.871 (-0.012) | 0.714 (-0.016) | 0.260 (+0.005) |
| Detail stem | DINOv3 ViT-S/16 | 0.875 (+0.018) | 0.672 (-0.064) | 0.274 (-0.016) |
| Detail stem | DINOv3 ViT-B/16 | 0.886 (+0.003) | 0.745 (+0.013) | 0.294 (-0.013) |
| 3x training steps | UNet (scratch) |  | 0.533 (+0.533) | 0.187 (+0.177) |

**What to use.**

- Start from a pretrained backbone. A scratch UNet needs far more training to
  catch up (0.533 on flies13 at 3× the steps, versus 0.73–0.78 for every
  pretrained backbone at 1×).
- ConvNeXtV2-N or DINOv3 ConvNeXt-T for the best accuracy per millisecond;
  DINOv2-reg ViT-B (with `layer_decay: 0.75`) among the plain ViTs.
- Bigger is not better here: DINOv2 ViT-L matches ViT-B at 2–2.5× the inference cost.
- Self-supervised (DINOv2/v3) and masked-image (MAE) pretraining beat
  language-supervised CLIP and SigLIP2 on these tasks.
- Fine-tune rather than freeze when labels are scarce (−0.05 to −0.06 mAP on
  gerbils when frozen).
- For ViT-S, `vit_neck: sfp` and `token_stride: 8` each gained about 0.03 on
  flies13; `detail_stem` did not help on average and stays off by default.

The configs, logs, per-run metrics and the scripts behind these tables are in
`scratch/2026-10-20-hf-foundation-backbones/` on the development machine; runs
were logged to the wandb project `sleap-nn-hf-backbones`.

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

Ready-to-edit samples:

- [`config_bottomup_segmentation_pretrained.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bottomup_segmentation_pretrained.yaml)
  — bottom-up segmentation on ConvNeXtV2 (Case A).
- [`config_topdown_centered_instance_pretrained_vit.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_centered_instance_pretrained_vit.yaml)
  — top-down centered-instance pose on DINOv3 ViT-S/16 (Case C).

### timm Hub checkpoints

A `model_name` under the `timm/` namespace (e.g.
`timm/vit_small_patch16_dinov3.lvd1689m`) is loaded through `transformers`'
`TimmWrapperModel`, and its features are read with timm's
`forward_intermediates`. This is how sleap-nn gets ungated DINOv3 weights,
PE-Spatial, EVA02, and DINOv3-ConvNeXt.

- **Needs `timm`** (`timm>=1.0.20`, included in the `backbones` extra). Without
  it, construction fails with an install hint.
- **Plain ViT or hierarchical is read off the built model**, not the model id:
  `timm/vit_*` checkpoints take Case C (or B), `timm/convnext_*` take Case A.
- **Any input size.** timm ViTs are built with `dynamic_img_size=True`, so
  positional embeddings are resampled per call.
- **Normalization** comes from the checkpoint's timm `pretrained_cfg`
  (`mean`/`std`) unless you set `image_mean`/`image_std`.
- `revision`, `weights: false`, and the offline cache behave as for any other
  model id.

### ViT knobs

`token_stride`, `layers`, `vit_neck`, `neck_channels`, and `detail_stem` shape
the plain-ViT path (Case C, and the token path of Case B); a hierarchical
backbone logs a warning that it ignores them, and so does the plain-ViT path for
the knobs it cannot use (`out_indices`; the neck knobs in encoder-only mode). `freeze`,
`gradient_checkpointing` and `drop_path_rate` apply to every encoder, as does
`trainer_config.optimizer.encoder_lr_scale`. `trainer_config.optimizer.layer_decay`
applies to plain ViTs (including DINOv2 in encoder-only mode); on a hierarchical
backbone it logs a warning and applies only `encoder_lr_scale`.

| Field | Default | What it does |
|-------|---------|--------------|
| `token_stride` | `null` → power of two nearest the patch size (16 for patch 14 or 16) | Stride of the token grid. The input is resized to `(H/s·p, W/s·p)` for patch size `p`, so the grid is `H/s × W/s`. Must be a power of two ≥ 4. `8` gives a 2× finer grid at 4× the tokens (roughly 4× the encoder cost or more, since attention is quadratic in token count). |
| `layers` | `null` → one evenly spaced layer per pyramid level, ending at the last (`dpt`): `[3, 6, 9, 12]` for 12 layers, `[6, 12, 18, 24]` for 24; the last layer for `sfp` and encoder mode | Transformer layers to tap, 1-indexed (`-1` = last). `dpt` needs one per built level (4 unless `output_stride` is coarser than `token_stride/4`), and a layer may repeat; `sfp` and encoder mode take exactly one. |
| `vit_neck` | `dpt` | `dpt`: one tapped layer per pyramid level, shallowest tap → finest level. `sfp`: ViTDet Simple Feature Pyramid; every level is built from the last tap. |
| `neck_channels` | `null` → `[h/8, h/4, h/2, h]` for hidden size `h` (ViT-S: `[48, 96, 192, 384]`; ViT-B: `[96, 192, 384, 768]`) | Channels of the 4 neck levels, finest first. |
| `detail_stem` | `false` | Add a light conv stream on the image (ViTMatte-style) whose stride-2/4/8 features are concatenated into the pyramid as decoder skips, so the decoder sees real image detail below the token stride. ~0.3 M parameters. |
| `detail_channels` | `null` → `[32, 64, 128]` | Detail-stem channels at strides 2, 4, 8 (exactly three values). |
| `freeze` | `false` | Freeze the ViT. The tap LayerNorms, neck, decoder, and heads still train. |
| `drop_path_rate` | `null` (checkpoint's own) | Stochastic depth, ramped linearly from `0` at the first block to this value at the last, as in timm and the ViTPose/MAE fine-tuning recipes (0.1–0.3 is typical). Passed to timm models and set on HF configs that have the field. HF DINOv2 and DINOv3-ViT apply the rate uniformly to every block; sleap-nn converts them to the linear ramp so one value means the same regularization across families. Families without stochastic depth (CLIP, SigLIP, ViT, MAE) log a warning and ignore it. Active in training only. |
| `encoder_lr_scale` | `1.0` | **Deprecated** here: set `trainer_config.optimizer.encoder_lr_scale` (see **Learning rates** below). |
| `layer_decay` | `null` (off) | **Deprecated** here: set `trainer_config.optimizer.layer_decay` (see **Learning rates** below). |
| `gradient_checkpointing` | `false` | Recompute encoder activations in the backward pass: less memory, slower steps. Only active in training (off for inference and export). Logs a warning and does nothing on a model that does not support it. |

**Strides.** The neck emits strides `s/4, s/2, s, 2s`: 4/8/16/32 at the default
`s = 16`, 2/4/8/16 at `token_stride: 8`. `max_stride` must be a multiple of
`2s` so padded inputs tile the token grid exactly: the default `max_stride: 32`
works for both, and anything else fails at construction with a message naming
the required multiple. `output_stride` is the finest decoder output, as for
Case A: at `s = 16`, `output_stride: 4` has a skip connection at every level and
`output_stride: 2` adds one upsampling block without a skip; at `s = 8` the neck
reaches stride 2 itself. `output_stride` can be at most `s`: the decoder needs at
least two pyramid levels.

**Unused parameters.** Only the tapped layers feed the neck, and the encoder
stops after the deepest tap (timm's `stop_early`; an HF encoder's block loop is
cut short, with every block kept in the state dict). On the plain-ViT path (Case C and the Case B token path),
encoder parameters that cannot receive a gradient — blocks after the deepest
tap, final/pooler norms, pooling heads (CLIP's `post_layernorm`, SigLIP's
attention-pool head), unused mask tokens — are found by a dummy backward pass at
construction and excluded from training, so DDP does not fail on unused
parameters. They stay in the state dict, so pretrained and trained weights load
unchanged. Likewise the neck builds only the levels the decoder reads: at
`output_stride: 8` (token stride 16) it builds strides 8/16/32 from three taps,
and the detail stem contributes only the strides at or above `output_stride`.

**Learning rates.** A fine-tuned ViT trains at the same LR as the randomly initialized neck and
decoder by default. `trainer_config.optimizer.encoder_lr_scale` (e.g. `0.1`) and
`trainer_config.optimizer.layer_decay` (values around 0.65–0.9 are common in ViT
fine-tuning recipes) let the pretrained layers move more slowly than the new
ones. Block `i` of `L` trains at `encoder_lr_scale · layer_decay^(L+1−i)` × base
LR. Everything registered before the first block (patch/position embeddings,
prefix tokens, CLIP's pre-norm) gets `layer_decay^(L+1)`; everything after the
last block (final norm) gets `encoder_lr_scale`. An LR schedule's absolute values
(`warmup_start_lr`, `eta_min`, `end_lr`, `min_lr`) are scaled by each group's
factor too, so the ratios hold for the whole run. Both are applied through
optimizer parameter groups, so they also apply to the `embedding` model type, and
they compose with `trainer_config.optimizer.weight_decay` (which exempts the
position embeddings and tokens). Before 0.4 these keys lived on
`backbone_config.pretrained`; they are still read there, with a deprecation
warning. See the [trainer config](../configuration/trainer.md#optimizer) for the
rest of the ViTPose-style recipe (per-step warmup, multi-step decay, weight EMA).

A top-down centered-instance pose model on DINOv3 ViT-S/16 (swap in
`facebook/dinov2-with-registers-small` for an Apache-2.0, `transformers`-only
alternative):

```yaml
data_config:
  preprocessing:
    ensure_rgb: true          # set automatically for a pretrained backbone
    crop_size: 256            # a multiple of max_stride avoids extra padding
model_config:
  init_weights: xavier        # pretrained encoder weights are re-applied after init
  backbone_config:
    unet: null
    convnext: null
    swint: null
    pretrained:
      source: hf
      model_name: timm/vit_small_patch16_dinov3.lvd1689m
      weights: true
      mode: auto              # resolves to decoder (Case C): confmaps are spatial
      freeze: false
      in_channels: 3
      output_stride: 2
      max_stride: 32          # multiple of 2 x token_stride
      token_stride: null      # 16 here; 8 = finer grid, ~4x encoder cost
      layers: null            # [3, 6, 9, 12] for a 12-layer ViT
      vit_neck: dpt           # dpt | sfp
      neck_channels: null     # [48, 96, 192, 384] for ViT-S (hidden size 384)
      drop_path_rate: null
      gradient_checkpointing: false
  head_configs:
    centered_instance:
      confmaps:
        part_names: null
        anchor_part: null
        sigma: 2.5
        output_stride: 2
```

The full config is
[`config_topdown_centered_instance_pretrained_vit.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_centered_instance_pretrained_vit.yaml).
Train the matching centroid model as usual (any backbone).

### Freeze vs. fine-tune

- `freeze: false` (default) — the whole backbone fine-tunes end-to-end. Best
  accuracy; recommended when you have enough labeled data.
  `trainer_config.optimizer.encoder_lr_scale` and `layer_decay` (above) are a
  middle ground between this and freezing.
- `freeze: true` — the encoder is frozen (feature extraction) and only the
  decoder/head train (for Case C also the tap LayerNorms and the neck). Faster,
  lower memory, and more robust on very small datasets. sleap-nn filters frozen
  parameters out of the optimizer automatically, and holds the encoder in eval
  mode for the whole run so a BatchNorm-bearing backbone (ResNet, BiT) does not
  drift its running statistics — "frozen" covers the statistics, not just the
  weights.
- For the `embedding` model type, `head_configs.embedding.embedding.freeze_backbone:
  true` does the same thing, and also works on the native `convnext` / `swint`
  (their ImageNet encoder; the randomly initialized middle blocks still train) and
  `unet` (stem + encoder blocks) backbones. Without pretrained weights it logs a
  warning, since the encoder would then stay at its random initialization.
- A frozen encoder also changes the `embedding` head's default pooling. With
  `head_configs.embedding.embedding.pool` unset, a frozen pretrained encoder pools
  with `avg` and everything else with `gem`. GeM pools only the positive part of
  each channel, and a pretrained encoder's LayerNorm output is about half negative.
  A fine-tuned encoder adapts to that; a frozen one cannot. On the gerbil re-ID
  set, `avg` beat `gem` by about 0.04 val rank-1 on frozen DINOv2 and ConvNeXtV2
  (8 seeds each) and tied when the encoder was fine-tuned. The resolved value is
  written into the saved config, and an explicit `pool` always wins.

### Channels and normalization

Pretrained stems are **3-channel**. The trainer sets `in_channels=3` and
`ensure_rgb=true` automatically for a `pretrained` backbone, so grayscale videos
are replicated to 3 channels. Model-specific mean/std normalization is applied
**inside the backbone** — the data pipeline still only rescales to `[0, 1]`. The
statistics come from `image_mean`/`image_std` if set, otherwise from the
checkpoint's timm `pretrained_cfg` (`timm/...` models) or its
`AutoImageProcessor`, falling back to ImageNet statistics.

!!! note "The `embedding` model type is the exception"

    The re-ID (`embedding`) pipeline per-crop standardizes to ~N(0, 1) before the
    backbone sees the crop, so it does **not** feed `[0, 1]`. Applying the ImageNet
    shift on top of that hands the stem mean −1.99 / std 4.43, which trains — badly.
    Measured on the gerbil re-ID set (DINOv2-with-registers, 3 epochs, seed 0,
    paired runs): val rank-1 **0.363** with normalization on vs **0.920** with it
    off. The trainer therefore forces `normalize: false` for `embedding` and logs
    that it did; set it back to `true` only if you also disable the per-crop
    standardize.

    For the same reason `embedding` also keeps its own data channels: unlike every
    other model type it is **not** flipped to `ensure_rgb`, because the 1-channel
    crop is repeated to 3 in `Model.forward` and grayscale is often a deliberate
    anti-confound choice (a recording setup mixing color and monochrome cameras
    makes color an identity cue that will not generalize).

## Reproducibility, gating, and offline use

- **Pin `revision`.** HuggingFace `main` moves. Set `revision` to a commit sha
  or tag so a re-run loads identical weights.
- **Gated models (`facebook/dinov3-*`, SAM3).** These live in gated HF repos
  with a custom license. Request access, then authenticate before training:

  ```bash
  huggingface-cli login          # or: export HF_TOKEN=hf_...
  ```

  The `timm/...dinov3...` checkpoints are not gated, but the DINOv3 License still
  applies to them. That license is non-OSI (commercial use permitted with
  attribution) — review it before bundling. Hiera / I-JEPA weights are
  `cc-by-nc-4.0` (non-commercial), and the upstream MAE weights are released
  under CC-BY-NC-4.0.
- **Offline / air-gapped.** Pre-seed the HuggingFace cache
  (`~/.cache/huggingface/hub`) on a networked machine, copy it over, and set
  `HF_HUB_OFFLINE=1`. `timm/...` checkpoints use the same cache. With
  `weights: false` the architecture still needs the model's `config.json` (tiny),
  but no checkpoint download.

  sleap-nn resolves the concrete model class from `AutoConfig` (the `*Backbone`
  class for Case A/B, the vision-model class for Case C) rather than going
  through `AutoBackbone.from_pretrained`, which probes the Hub for the repo's
  existence before resolving anything and so fails offline even for a
  fully-cached model. A backbone whose config type is not in `transformers`'
  backbone mapping (a custom or very new model) falls back to `AutoBackbone` and
  inherits that limitation.

## Gotchas

- **Patch-14 ViTs (DINOv2, EVA02) run at token stride 16.** The input is
  bilinearly resized by 14/16 before patch embedding so the token grid lands on
  stride 16, so 16/32-padded crops work without changes. Set `token_stride: 8`
  for a finer grid instead. The exception is Case B on DINOv2 (the `embedding`
  model type), which keeps its native stride 14; pooled heads are unaffected.
- **`max_stride` for a plain ViT** must be a multiple of `2 × token_stride` (32
  by default, 16 at `token_stride: 8`); the default 32 works for both. A larger
  multiple (e.g. 64) builds, but only adds padding: the backbone logs a warning
  and uses its real deepest stride.
- **`output_stride=2`.** Hierarchical HF backbones bottom out at stride 4 (no
  stride-2 feature), and so does the Case C neck at the default token stride, so
  reaching `output_stride=2` adds one learned-upsample decoder block with no skip
  connection — a small localization-quality nuance, not a blocker.
  `output_stride=4` uses skips at every level.
- **Fixed-resolution models fail at construction.** Case C runs a dummy forward
  at a non-square size off the pretraining resolution and raises a `ValueError`
  naming the model if it cannot run there (see
  [Known not to work](#tested-model-families)).
- **Memory.** A ViT-L at large crops, or any ViT at `token_stride: 8`, is much
  heavier than a ConvNeXtV2-nano. `gradient_checkpointing: true` and
  `freeze: true` are the main levers.
- **ONNX export.** ConvNeXt/ResNet encoders and plain ViTs (the Case C neck
  included) export at opset 17; DINOv2-with-registers resamples its positional
  embeddings with an antialiased bicubic resize, so the exporter retries it at
  opset 18. The export tests check Case C graphs (DINOv2-reg, CLIP) against
  PyTorch at batch and input sizes other than the traced ones. Swin is
  export-fragile: a graph can pass the structural `onnx.checker` yet be
  numerically wrong, so enable the exporter's `numerical_check` (a
  torch-vs-onnxruntime parity assert) when exporting transformer backbones.
- **RoPE models export at a fixed input size.** Rotary position embeddings
  (`timm/` DINOv3, EVA02, PE-Spatial) bake the traced token grid into the
  graph, so their ONNX/TensorRT models take only the traced height and width
  (the batch stays dynamic), and other sizes are rejected rather than silently
  wrong. Crop models (`centered_instance`, `multi_class_topdown`) are traced
  at their crop size, so this costs nothing there; a combined top-down export
  is fixed only when its centroid model uses RoPE. For a whole-frame RoPE
  model, set `--input-height`/`--input-width` to your frame size. TensorRT
  profiles are clamped to the fixed size, with a warning.
- **`dtype`.** `transformers` v5 defaults to `dtype="auto"`, which can silently
  load fp16/bf16. sleap-nn forces `float32` in the backbone factory.
- **Re-ID (`embedding`) has three trainer-applied overrides.** `mode` resolves to
  `encoder`, `normalize` is forced off (the pipeline already standardizes each
  crop), and the data channels are *not* flipped to RGB. Each is logged when it
  fires; see the notes under
  [Channels and normalization](#channels-and-normalization). Fine-tuned DINOv2
  reached val rank-1 0.92 in 3 epochs on the gerbil re-ID set once the
  normalization override was in place.
