# SLEAP-NN Inference Performance Hotspots

This note summarizes the Nsight Systems capture (`sleap_inference.nsys-rep`) and highlights concrete code sites worth modifying to raise GPU utilization. Line references follow the current `main` branch state.

## Profile Highlights

- ~80.5% of CUDA API time is spent in `cudaStreamSynchronize` (1,024,127 calls) with another ~11.9% in `cudaLaunchKernel` (7.5M launches). The synchronizations keep the GPU idle after bursts of micro-kernels, pointing to downstream host work that forces per-batch waits.
- GPU memory activity is dominated by Host→Device copies (67.7% of time across 214k transfers, avg 0.74 MB) and memsets (28.5% across 139k calls), so we are repeatedly shuttling and zeroing small buffers instead of reusing device-resident tensors.
- NVTX ranges show each `ModelForward`/`BottomUpInferenceModel.forward` instance taking ~179 ms, with `PAFScorer.predict` and `Postprocess` still the largest subranges—these stages remain the main kernel launch sources once the model finishes.
- OS runtime is still 94% `pthread_cond_wait`, matching the heavy `cudaStreamSynchronize` usage. Queue waits (`frame_buffer.get`) remain negligible, so the ingest pipeline continues to keep up with inference.

## Hot Paths to Rework

### 1. Output Staging (Implemented)

`Predictor._stage_output_for_numpy` and the updated `_predict_generator` (`sleap_nn/inference/predictors.py:356`, `sleap_nn/inference/predictors.py:437`) now pin host buffers, enqueue async GPU→CPU copies on a dedicated stream, and only call `_convert_tensors_to_numpy` once transfers complete. The generator pipelines GPU work on the default stream while the copy stream drains pending batches, so we should see the ~1M `cudaStreamSynchronize` calls fall sharply.

**Next checks:**
- Re-profile to confirm the expected drop in `cudaStreamSynchronize` time and verify that Host→Device copy overlap improves overall GPU duty cycle.
- Watch for new hotspots in `_stage_output_for_numpy` (e.g., excessive pinned allocations); consider adding a small pinned-buffer cache if Nsight shows spikes in `cudaMallocHost`.

### 2. Batch Construction and Preprocessing Loop

`Predictor._predict_generator` (`sleap_nn/inference/predictors.py:386`) dequeues frames one at a time, performs normalization/resizing on the CPU, and immediately pushes the batch through the inference model. Every frame triggers multiple small CUDA ops downstream.

```python
# sleap_nn/inference/predictors.py:437
for _ in range(self.batch_size):
    frame = self.pipeline.frame_buffer.get()
    ...
    frame["image"] = apply_normalization(frame["image"])
    frame["image"], eff_scale = apply_sizematcher(...)
    ...
outputs_list = self.inference_model(ex)
```

**Ideas:**
- Move normalization/resizing into a GPU-side preprocessing module so the model sees larger contiguous batches and less host/device churn.
- Collect multiple batches (e.g., aggregate `n` batches before calling the model) to amortize post-processing costs.
- Prototype a `torch.compile`-wrapped inference path to see if PyTorch can fuse some of the small kernels.

### 3. Peak Finding & Instance Grouping

The top-down pipeline calls the peak-finding utilities in `sleap_nn/inference/peak_finding.py` (`find_global_peaks`, `crop_bboxes`, `integral_regression`). These functions rely on ATen elementwise ops that emit the CUB kernels dominating the profile.

```python
# sleap_nn/inference/peak_finding.py:154-200
rough_peaks, peak_vals = find_global_peaks_rough(cms, threshold=threshold)
...
valid_idx = torch.where(~torch.isnan(rough_peaks[:, 0]))[0]
...
cm_crops = crop_bboxes(cms, bboxes, valid_idx)
...
refined_peaks[valid_idx] += offsets
```

**Ideas:**
- Fuse peak detection and refinement into a single custom CUDA/Triton kernel to cut launch count.
- Cache intermediate tensors/allocation plans (e.g., reuse scratch buffers, avoid reshape/where each iteration).
- Push tracking logic (currently CPU heavy) to operate on entire batches instead of per-instance lists.

### 4. Tracker Integration

With `tracking=True`, `Tracker.track` runs after each batch (`sleap_nn/predict.py:334-374`). Tracking currently loops in Python and spawns additional GPU kernel bursts for similarity metrics.

```python
# sleap_nn/predict.py:334-373
if tracking and not isinstance(predictor, BottomUpMultiClassPredictor):
    predictor.tracker = Tracker.from_config(...)
...
output = predictor.predict(...)
```

**Ideas:**
- Profile `sleap_nn/tracking/tracker.py` to identify hotspots (e.g., per-frame similarity scoring). Consider batching track association across multiple frames or porting the heavy math to CUDA.
- Allow inference mode without tracking for throughput-critical runs, or delay tracking until after all frames are batched (streaming pass).

### 5. Memory Traffic

The `cudaMemcpyAsync`/mem-size summaries show 214k Host→Device copies (avg 0.74 MB) and 139k memsets totaling 3.8 TB of zero fill. We are repeatedly allocating and clearing work buffers across peak finding and tracking.

**Ideas:**
- Use pinned staging buffers and `torch.cuda.Stream` to overlap `cudaMemcpyAsync` with kernel execution (currently prevented by the frequent `cudaStreamSynchronize`).
- Reuse GPU tensors for crops and reduce host→device copying by generating crops directly on device tensors.

## Proposed Experiments

1. **Re-profile after async staging:** Capture a fresh Nsight Systems run to validate the reduced `cudaStreamSynchronize` footprint and measure new GPU utilization.
2. **Instrument batches:** Add NVTX ranges in `TopDownInferenceModel.forward` and tracker entry points to confirm kernel bursts per stage.
3. **Batch fusion prototype:** Modify `_predict_generator` to accumulate, say, 4 batches before calling `inference_model` and measure the impact on kernel counts.
4. **GPU preprocessing proof-of-concept:** Move normalization + `apply_sizematcher` into a scripted `torch.nn.Module` that runs on CUDA tensors.
5. **Custom peak finder:** Implement a fused CUDA/Triton kernel replacing `find_global_peaks`/`DeviceSelect` chain, compare launch counts.

## Validation Steps

- Re-run the Nsight profile after each change; expect reductions in `cudaStreamSynchronize` call count and average kernel runtime.
- Track wall-clock FPS via the existing progress bar and compare with the baseline (23.3 FPS in the sample run).
- Watch memory usage when increasing batch sizes or buffering more frames to avoid OOM.

## Notes

- Lightning is only used during training; inference here is plain PyTorch code, so local modifications are safe.
- The queue producer is already fast enough; effort is best spent on the GPU-bound post-processing stages described above.
