# Centroid-only models — cross-repo architecture plan

> First-class **centroid-only** training + inference in `sleap-nn`: train a model that
> predicts just the centroid (anchor point) of each instance, and run inference with
> **only** that model — no second-stage centered-instance model. Primary use case:
> **multi-instance single-node skeletons** (animals tracked as single points), where a
> top-down centered-instance second stage is pure redundancy.
>
> Status: **plan** (study + design complete; implementation not started).
> Produced from a cross-repo study of `sleap-nn`, `sleap-io`, and `sleap` (frontend PR #2724).

---

## 0. Binding decisions (from the maintainer)

| # | Decision |
|---|----------|
| Scope | `sleap-nn` for train/infer/eval/track/export/authoring/docs. `sleap-io` **only** for centroid file-format support. **Do NOT touch the `sleap` frontend this session** — add the entrypoint in `infer`; circle back to the frontend later. |
| Entry point | Centroid-only inference goes through the **new** flow (`sleap-nn infer` → `inference/run.py::predict` → `inference/predictor.py::Predictor.from_model_paths` → `CentroidLayer`). The **legacy** path (`track`, `sleap_nn.predict.run_inference`, `inference/predictors.py`) gets **no** new centroid features and **hard-errors** for a lone centroid model. |
| Depth | **Full first-class** (authoring UX, eval, tracking defaults, export, docs). |
| Output rep | Multi-node-trained centroid model → **collapse** to a single-node `Skeleton(['centroid'])`. **Also emit `sio.Centroid`** as an **opt-in** alongside the default single-node `PredictedInstance`. Default output stays a single-node `PredictedInstance` (loadable by today's Centroid-unaware frontend). |
| #586 | The centroid's *meaning* (anchor vs mean-of-visible fallback) is a **shared train+infer contract** via `data/instance_centroids.py::generate_centroids` (currently mean-of-visible, locked by tests). Collapse semantics, `sio.Centroid.source/method`, and eval GT-centroid computation must mirror it and must not diverge training from inference. |

---

## 1. Current state (verified)

### Training — already works ✅
A standalone centroid head trains end-to-end today. `get_model_type_from_cfg` (`config/utils.py:7-14`)
returns `"centroid"` purely from the head-config key; `CentroidConfmapsHead` emits 1 channel;
`CentroidLightningModule` hard-codes `node_names=["centroid"]`. The whole data/model/loss path is
**node-count-agnostic** (a 1-node skeleton just works; `tests/assets/generated_configs/centroid.yaml`
is a working standalone config).

**Gaps:** authoring (`config_generator`/TUI only emit centroid as a *paired* top-down bundle), and
post-training auto-eval (`train.py:12` imports the **legacy** `run_inference`, which is GT-dependent
for centroid).

### Inference — new flow works, legacy is broken 🟡
- **New** path (`sleap-nn infer`): centroid-only **works** — `_select_layer` builds a standalone
  `CentroidLayer`; `outputs.to_instances` produces real `PredictedInstance`s (verified empirically).
- **Legacy** path (`sleap-nn track` → deprecated `predict.py::run_inference` → `predictors.py`):
  routes a lone centroid to `FindInstancePeaksGroundTruth` (**requires GT**, useless on video) and
  **pops `--centroid_only`** (`cli.py:968`). **This is the PR #2724 blocker** — the frontend's
  `runners.py:541` hardcodes its inference subprocess to call `track`, not `infer`.

**Three correctness bugs in the new path** (harmless for pure single-node, wrong for multi-node anchors):
1. **Anchor dropped** — `loaders._build_topdown:500-513` builds the real `CentroidCrop` without
   `anchor_ind` (passed only in the GT branch `:493`). *Impact (corrected by review): this affects only
   `CentroidLayer.anchor_ind` → the packaging slot + `source` tag, **not** the predicted coordinate
   (the real-model forward derives centroids from `find_local_peaks` and ignores `anchor_ind`).*
2. **Writer ignores anchor/mode** — `writer.py:103` calls `to_labels` without the packaging params,
   so streaming/`predict_to_file` diverge from in-memory `predict()`.
3. **Filters no-op** — every filter early-returns when `pred_keypoints is None`
   (`filters.py:146,162,182,212`), so centroid-only output is unfiltered.

### Scoring degenerate for single points 🔴
Default OKS uses bbox area = 0 for a single node → a 1.0/0.0 step function. Breaks the tracker default
`scoring_method='oks'` (`tracker.py:81`) and `run_evaluation`'s OKS metrics. `euclidean_dist`/`iou` work
but aren't the defaults.

### `sio.Centroid` 🟢 (available) / frontend unaware 🔴
sleap-io 0.7.0 (already pinned `>=0.7.0,<0.8.0`) ships `Centroid`/`PredictedCentroid` +
`get_centroid_skeleton()` (`Skeleton(['centroid'])`), a separate `LabeledFrame.centroids` container, a
`/centroids` HDF5 group, and **lossless** `to_instance()`/`from_instance()`. **But**: the frontend is
Centroid-unaware (would drop `/centroids` on resave) and sleap-io doesn't bump `format_id` for centroids.
→ single-node `PredictedInstance` is the zero-friction default; `sio.Centroid` is opt-in now.

---

## 2. The keystone: Output Representation Contract

*(Consolidates the two overlapping design specs the reviewer flagged. One owner, one PR.)*

A single decision point on the `Predictor` resolves how centroid output is packaged, and the **same
struct** is threaded into in-memory `predict()`, streaming, and the writer so all three are identical.

```
Predictor._resolve_centroid_packaging() -> CentroidPackaging(
    collapse_skeleton,   # sio.get_centroid_skeleton() when centroid-only AND model skeleton has >1 node; else None
    anchor_ind,          # CentroidLayer.anchor_ind (for source tagging; packaging slot is node 0 on collapsed skel)
    emit_centroid,       # 'instance' (default) | 'centroid' | 'both'
    source_method,       # 'anchor:<node>' if anchor configured, else 'center_of_mass' (== generate_centroids mean-of-visible, #586)
)
```

**Rules**
- **Collapse** engages only when `isinstance(layer, (CentroidLayer, ExportedCentroidLayer))` **and** the
  model skeleton has `>1` node → output is a genuine 1-node `Skeleton(['centroid'])` instance (centroid
  at node 0). A **genuinely 1-node** model is emitted verbatim (structurally identical). The old
  multi-node-NaN-pad path is **not** used in the new flow.
- **Raw `Outputs.to_instances` stays backward-compatible** (keeps NaN-pad when no `collapse_skeleton` is
  passed) — collapse engages *only* via `Predictor`. This preserves the existing
  `tests/inference/test_centroid_only.py` packaging tests.
- **`emit_centroid`** (the opt-in): `'instance'` (default; single-node `PredictedInstance`, frontend-safe),
  `'centroid'` (`sio.PredictedCentroid` into `LabeledFrame.centroids`), `'both'`. A new module
  `sleap_nn/inference/centroid_convert.py` owns the sio mapping (built on lossless
  `Centroid.to_instance()/from_instance()`), incl. `source_method` derivation consistent with #586.
- **Scores**: single-node `PredictedInstance` carries both per-point (centroid value at node 0) and
  per-instance score; `PredictedCentroid` carries instance-level `.score`/`.tracking_score` only.

**Unified API names** (resolving the reviewer's collision): `emit_centroid: {instance,centroid,both}`
everywhere (CLI `--centroid-output`, `run.predict`, `Predictor.from_model_paths`, writer, export);
`source_method` for the #586 tag; adopt `centroid_convert.py` for the conversion helpers.

---

## 3. Staged implementation plan (PRs)

Ordering minimizes risk and lands a working slice early. PR 2 is the keystone everything else consumes.

### PR 1 — Foundation (low risk, independent)
- Thread `anchor_ind` into the real-model `CentroidCrop` in `_build_topdown` (`loaders.py:500-513`) and
  `_build_topdown_multiclass` (`:650-663`). *(Affects the packaging slot/source tag, not coordinates.)*
- **Legacy closure**: `predict.py::run_inference` raises `ValueError` for a lone-centroid `model_paths`
  (detect via `get_model_type_from_cfg`, config-parse only) pointing to `sleap-nn infer`; `cli.py track`
  stops silently popping `--centroid_only` and raises `click.UsageError`; update the deprecation docstring.
- **Regression guard**: a 2-model top-down run still emits the **full** multi-node skeleton (assert
  `_resolve_centroid_packaging` returns `collapse_skeleton=None` for a `TopDownLayer` even with
  `anchor_ind` now populated on the underlying `CentroidCrop`).

### PR 2 — Keystone: output representation (medium risk)
- `outputs.py`: collapse logic + `to_centroids()`; `to_labels` empty-frame guard
  `if not instances and not centroids: continue`; `Labels.skeletons=[collapse_skeleton]` when collapsing.
- `predictor.py`: `_resolve_centroid_packaging()`, `Predictor.emit_centroid` field,
  `from_model_paths(emit_centroid=...)`, `to_labels` + `predict_to_file` wiring.
- `centroid_convert.py` (new): `centroid_source_for_anchor`, `build_predicted_centroid`,
  `predicted_instance_to_centroid`, `centroid_to_instance`.
- `writer.py`: thread `anchor_ind`/`collapse_skeleton`/`emit_centroid`/`source_method` so streaming ==
  in-memory; `_finalize` uses the collapse skeleton.
- `filters.py`: centroid-aware branches — `_nan_out_where` clears `pred_centroids`/`pred_centroid_values`;
  `_filter_confidence` gates `min_instance_score` on centroid values; skip node-count/mean-node-score.
  **Overlap-NMS decision (see §4): add a small `FilterConfig` centroid-distance/radius knob** so
  overlap-NMS is a *real* dedup (point-bbox IoU is 0 for distinct points → otherwise a no-op).
- Update `tests/inference/test_centroid_only.py` to assert collapse semantics where the `Predictor`
  drives it; keep raw-`Outputs` NaN-pad tests for the un-collapsed API.

### PR 3 — CLI/run wiring (low risk; depends on PR 2)
- `run.predict(..., emit_centroid='instance')`; CLI `infer` gains `--centroid-output {instance,centroid,both}`;
  thread through `_run_in_memory_new_flow` + `_run_stream_to_file`. Fix the stale `--centroid_only` help text.
- Tests via `CliRunner` + mocked `run.predict` (avoid subprocess/torch-venv corruption — see §5).

### PR 4 — Tracking defaults (medium risk; depends on PR 2; parallel with PR 5)
- `TrackerConfig` gains `scoring_method_explicit`/`features_explicit` sentinels (default True for direct
  callers). `apply_tracking`: when `len(labels.skeletons)==1 and len(nodes)==1` *(guard promoted from the
  reviewer's note)* and the user didn't pin one, default `scoring_method='euclidean_dist'`,
  `features='centroids'`; log it. CLI `--scoring_method`/`--features` default `None`.
- `filters` OKS-NMS fallback-to-IoU guard widened to `<2` nodes (collapsed output has 1 *populated* node).
- Document the `> min_new_track_points` / `> min_match_points` off-by-one for single points (keep default 0).

### PR 5 — Evaluation (high risk; depends on PR 2; parallel with PR 4)
- `compute_gt_centroids(points, anchor_ind)` — numpy mirror of `generate_centroids` (mean-of-visible,
  **#586 parity test** against `find_points_mean`).
- Move `match_centroids` from `training/callbacks.py` into `evaluation.py` **byte-identical** and
  re-export from `callbacks.py` (keep callback tests green).
- `Evaluator(match_method='oks'|'centroid', anchor_ind=...)`; `detection_metrics()`
  (precision/recall/F1 + localization-error percentiles); `run_evaluation(match_method, anchor_part)`
  auto-detects centroid mode (pred skeleton == `get_centroid_skeleton()` or single visible node);
  default `match_threshold=50.0` in centroid mode; consume `trainer_config.eval.match_threshold`.
- **OKS-for-1-node decision (see §4): omit OKS as primary** for 1-node (report distance/detection);
  do not ship the arbitrary fixed-scale constant unless explicitly wanted.

### PR 6 — Authoring (medium risk; mostly independent)
- `recommender.py`: new `PipelineType` member `"centroid_only"`; `num_nodes==1` multi-instance branch
  (ordered **before** the animal-to-frame split) → recommend standalone centroid, `requires_second_model=False`.
- `generator.py`: `centroid_only` builds **one** centroid YAML (`is_topdown` stays False → no dual-emit);
  `_build_head_config` emits a canonical `centroid` head; full-res defaults (`scale=1.0`, `sigma=2.5`).
- TUI: standalone "Centroid (single point per animal)" card + state/selection wiring.
- CLI `config --pipeline`: `centroid` = standalone (1 config); `topdown` = paired (2 configs).
- New golden `tests/assets/generated_configs/centroid_only.yaml`.
- *(train.py post-eval is **deferred to PR 7** to avoid the two-spec collision.)*

### PR 7 — `train.py` post-eval (high risk; single merged edit; depends on PR 2 + PR 5)
- One owner for the ~30-line block: branch on `model_type=='centroid'` → call the **new**
  `sleap_nn.inference.run.predict` with `source=path` (positional), `centroid_only=True`,
  `output_path=...` — **drop `make_labels`** (the new `predict` has no such param; hardcodes True) and
  rename `data_path`→`source`. Route eval to `run_evaluation(match_method='centroid', anchor_part=...)`;
  log detection + distance metrics, **guard every metric-key access** (no `oks_voc.mAP` for centroid).
  Non-centroid models stay on the existing path. *(Verify the predicted `frame_idx` set matches the GT
  labeled-frame set for `labels_gt.*.slp`.)*

### PR 8 — Export + docs (low/medium risk; depends on PR 2)
- `from_export_dir`: resolve `skeleton = sio.get_centroid_skeleton()` when `metadata.model_type=='centroid'`
  (so the export Predictor self-describes collapse); `export predict` CLI stops force-passing the full
  training skeleton for centroid models; `--centroid-output` flag matching PR 3.
- Clearer guard when two centroid dirs are passed to `export` (point to single-dir standalone).
- Docs: rewrite `docs/guides/centroid-only-inference.md` to the collapse contract + end-to-end
  (train → `infer` → distance `eval` → export → exported infer); fix `docs/guides/export.md`
  ("No standalone centroid" is now false); add `config_centroid_unet_standalone.yaml` + single-node
  skeleton example; update `docs/configuration/samples.md` and a one-line `CLAUDE.md` pointer.

---

## 4. Open decisions resolved (recommendations — adjust if you disagree)

| Decision | Recommendation | Why |
|---|---|---|
| Centroid overlap-NMS | **Add a small `FilterConfig` centroid-distance (px-radius) knob** and do distance-NMS | Multi-instance single-point tracking specifically needs duplicate-centroid suppression; point-bbox IoU is structurally 0 → no-op. The only spot that adds a `FilterConfig` field. |
| OKS for 1-node | **Omit OKS as primary**; report detection (P/R/F1) + localization error. No magic OKS-scale constant. | bbox-area OKS is ill-defined for points; distance is the standard for point detection. Avoids shipping an arbitrary constant + an untested forced-OKS path. |
| `emit_centroid='centroid'` + tracking | **Auto-upgrade to `'both'` (or warn)** when a tracker is configured | Tracker/eval read `predicted_instances`; a centroids-only frame would silently track nothing. |
| Default `emit_centroid` | **`'instance'`** | Frontend-compatible today; `sio.Centroid` is opt-in per your decision. |
| 1-node bottom-up | **Do not offer** bottom-up for `num_nodes==1` (PAF grouping needs edges) | A single-node skeleton has 0 edges; guard/doc it in the recommender. |

---

## 5. Risks & test strategy

- **Top-down stage-1 reuse**: the centroid head/`CentroidCrop` is shared. Verified `anchor_ind` is inert
  in the real-model forward, so PR 1 doesn't change top-down crop centering — but add a regression test
  that a 2-model top-down run still emits the full skeleton (collapse gated on layer type).
- **#530/#582/#583/#584 parity**: the `to_labels` empty-frame guard loosening (PR 2) must not change
  non-centroid behavior; run the full `tests/inference` suite. The #582 strict video-index handling stays.
- **CI is CPU-only and CLI tests can corrupt the venv torch** (per project memory): prefer in-process
  `CliRunner` + mocked `run.predict` over subprocess; guard ONNX/export tests behind availability skips;
  assert flag propagation via monkeypatch, not real inference.
- **`match_centroids` move** (PR 5): keep byte-identical + re-export from `callbacks.py`; callback
  `match_threshold` tests are the regression gate.
- **Back-compat**: default `emit_centroid='instance'` → no `/centroids`, no `format_id` bump. Add a test
  that a centroid-free `Labels` keeps `format_id <= 2.2` and the default `.slp` has zero `/centroids` datasets.

---

## 6. `sio.Centroid` porting (Phase 2, in this initiative)

Per your "also emit `sio.Centroid` now" decision — folded into PR 2/PR 8 as an **opt-in**, plus minimal
sleap-io format support:

- **sleap-nn**: `centroid_convert.py` (lossless conversion via `Centroid.to_instance/from_instance`),
  `emit_centroid` opt-in, `source/method` tagging consistent with #586. Round-trip test
  (save → load preserves `x/y/score/source`).
- **sleap-io** (format only, allowed): bump `format_id` to `2.3` when `labels.centroids` is non-empty
  (`write_metadata`, `slp.py:1613-1648`) + history docstring; add `Centroid.numpy()` and
  `Labels.numpy_centroids()` array accessors for eval/tracking. Backward-compatible (reader has no
  max-version guard; older `.slp` without `/centroids` → empty list).
- **Version coordination**: the bump is backward-compatible and can ship in a `0.7.x` patch under the
  existing pin; only raise the floor if sleap-nn must *guarantee* a version that stamps `2.3`.

**Inter-conversion summary** (`Centroid` ↔ single-node `Instance`):
- `Centroid.to_instance()` → 1-node `Skeleton(['centroid'])` instance (drops z/category/name/source).
- `Centroid.from_instance(method=...)` ← maps to #586 fallback: `center_of_mass` ≈ mean-of-visible,
  `bbox_center` ≈ bbox-midpoint, `anchor` ≈ explicit node. Older `.slp` (pre-Centroid) need no migration.

---

## 7. Frontend contract (for the later `sleap` session — NOT this session)

When we circle back to PR #2724, the frontend should:
- Switch its inference subprocess from `sleap track` to **`sleap-nn infer`** (`runners.py:541`,
  `make_predict_cli_call`). For a lone centroid model dir, **`--centroid_only` is not needed**
  (auto-detected); it's only needed to force centroid output when a centered-instance model is also passed.
- Invocation: `sleap-nn infer --data_path <video|slp> --model_paths <centroid_dir> [--device ...]
  [--max_instances N] [--peak_threshold T] [--centroid-output instance]`.
- Default output is a single-node `Skeleton(['centroid'])` `PredictedInstance` — already mergeable via
  `Labels.merge` against the frontend's committed `sleap/skeletons/centroid.json`. The frontend can stay
  Centroid-unaware until it opts into `--centroid-output centroid`/`both`.
- Remove the `is_centroid_models_enabled` feature flag and add the inference-side pipeline option +
  `get_most_recent_pipeline_trained` extension to `mode=='inference'`.

---

## 8. Subsystem → file map (quick reference)

| Subsystem | Key files |
|---|---|
| Output rep (keystone) | `inference/outputs.py`, `inference/predictor.py`, `inference/loaders.py`, `inference/writer.py`, `inference/filters.py`, `inference/centroid_convert.py` (new), `inference/layers/centroid.py` |
| Entry points | `cli.py` (`infer`/`track`), `inference/run.py`, `predict.py` (legacy hard-error), `inference/predictors.py` |
| Authoring + train eval | `config_generator/{generator,recommender,analyzer}.py`, `config_generator/tui/*`, `cli.py config`, `train.py`, `training/model_trainer.py` (callback) |
| Evaluation | `evaluation.py`, `data/instance_centroids.py` (#586), `cli.py eval`, `training/callbacks.py` |
| Tracking | `tracking/tracker.py`, `tracking/utils.py`, `tracking/candidates/*`, `inference/tracking.py`, `inference/filters.py` |
| sio.Centroid | `inference/centroid_convert.py`, `inference/outputs.py`; sleap-io `model/centroid.py`, `model/labels.py`, `io/slp.py` |
| Export + docs | `export/cli.py`, `export/metadata.py`, `export/wrappers/centroid.py`, `docs/guides/*`, `docs/sample_configs/*` |

---

## 9. Suggested first slice

PR 1 (foundation/legacy-closure) + PR 2 (keystone) together produce a correct, frontend-loadable
centroid-only `infer` flow on a single-node skeleton — the minimum that unblocks the eventual PR #2724
frontend switch. PRs 3–8 layer on the full first-class experience.
