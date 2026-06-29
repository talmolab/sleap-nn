# Supervised ID models

Supervised ID models (`multi_class_topdown` and `multi_class_bottomup`) predict
pose **and** a persistent identity for each animal — so identities come straight
from the model and you don't need a separate [tracking](tracking.md) step.

Use them when your animals have **distinct, consistent appearances** (visual
markers, fur color, ear-clips, dye spots). If the animals look too similar to
tell apart, or you don't need fixed identities, use a standard pose model plus
[tracking](tracking.md) instead. These models can be a bit finicky to train,
since you're optimizing for pose and identity at the same time — but for animals
with clearly distinct appearances they can essentially eliminate ID
proofreading.

## 1. Label identities as tracks

Each track *name* becomes an identity *class*. In the
[SLEAP GUI](https://github.com/talmolab/sleap), assign a track to every labeled
instance:

1. Select an instance (it should show **Track: none**).
2. **Tracks → Set Instance Track → New Track**.
3. In the **Instances** panel, double-click the track and rename it to the
   identity class — e.g. `male`, `female`, `dark_fur`. Any consistent scheme
   works (even `A`/`B`, or names based on the marker pattern or location).
4. For the rest of your frames, select each instance and assign it with the
   **Ctrl+1 – Ctrl+9** shortcuts (hold **Ctrl** to see which shortcut maps to
   which track).

Only user instances **with a track assigned** are used for training, so give
every labeled instance a track and keep the names consistent across the project.
The class list is inferred from those names (leave `classes: null` in the
config).

## 2. Configure the model

Take your existing top-down (`centered_instance`) or bottom-up (`bottomup`)
config and swap in the ID head, keeping the backbone and pose parameters the
same. Starter configs:
[`config_topdown_multi_class_centered_instance_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_multi_class_centered_instance_unet.yaml)
and
[`config_multi_class_bottomup_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_multi_class_bottomup_unet.yaml).

`multi_class_topdown` is a top-down model, so it still pairs with a centroid
model. `multi_class_bottomup` is a single model.

The one knob to tune is the **`loss_weight` on the classification head**
(`class_vectors` for top-down, `class_maps` for bottom-up). Start around `0.001`;
**decrease** toward `0.0001` if poses get worse, or **increase** toward `0.01`
if identities don't separate.

## 3. Train

```bash
# Bottom-up ID: one model
sleap-nn train --config-name config_multi_class_bottomup_unet.yaml

# Top-down ID: centroid model + multi-class centered-instance model
sleap-nn train --config-name config_centroid_unet.yaml
sleap-nn train --config-name config_topdown_multi_class_centered_instance_unet.yaml
```

## 4. Run inference

No tracking arguments needed — the model assigns identities:

```bash
# Bottom-up ID
sleap-nn predict -i video.mp4 -m models/multi_class_bottomup/ -o predictions.slp

# Top-down ID: centroid + multi-class centered-instance
sleap-nn predict -i video.mp4 -m models/centroid/ -m models/multi_class_topdown/ -o predictions.slp
```

Each predicted instance carries its identity as a track in the output `.slp`.

## See also

- [Choosing a Model](../reference/models.md#choosing-a-model)
- [Model config reference](../configuration/model.md) — full `multi_class_*` head schemas
- [SLEAP paper](https://www.nature.com/articles/s41592-022-01426-1) — supervised-identity background
