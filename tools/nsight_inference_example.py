#!/usr/bin/env python3
"""Run SLEAP-NN inference with NVTX markers for Nsight profiling."""

from __future__ import annotations

import argparse
import queue
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Iterable, List

import torch

# Optional NVTX integration (no-op on CPU-only hosts).


def _nvtx_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")


if _nvtx_available():

    @contextmanager
    def nvtx_range(name: str):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

else:

    @contextmanager
    def nvtx_range(name: str):  # type: ignore[override]
        yield


# Import after NVTX helpers so patched functions see them.
from sleap_nn import predict  # noqa: E402
from sleap_nn.data import normalization as normalization_mod  # noqa: E402
from sleap_nn.data import resizing as resizing_mod  # noqa: E402
from sleap_nn.inference import topdown as topdown_mod  # noqa: E402
from sleap_nn.inference import bottomup as bottomup_mod  # noqa: E402
from sleap_nn.inference import peak_finding as peak_finding_mod  # noqa: E402
from sleap_nn.inference import paf_grouping as paf_grouping_mod  # noqa: E402
from sleap_nn.inference import predictors as predictors_mod  # noqa: E402
from sleap_nn.tracking import tracker as tracker_mod  # noqa: E402

queue_wait_times: List[float] = []
normalize_times: List[float] = []
size_match_times: List[float] = []
pad_times: List[float] = []
centroid_times: List[float] = []
instance_times: List[float] = []
tracker_times: List[float] = []
bottomup_forward_times: List[float] = []
local_peak_times: List[float] = []
paf_predict_times: List[float] = []
prep_times: List[float] = []
assemble_times: List[float] = []
postprocess_times: List[float] = []


# ---- Monkey patches ----
_orig_queue_get = queue.Queue.get


def _instrumented_queue_get(self, *args, **kwargs):
    start = time.perf_counter()
    with nvtx_range("frame_buffer.get"):
        item = _orig_queue_get(self, *args, **kwargs)
    elapsed = time.perf_counter() - start
    if isinstance(item, dict) and item.get("image") is not None:
        queue_wait_times.append(elapsed)
    return item


queue.Queue.get = _instrumented_queue_get  # type: ignore[assignment]

_orig_normalize = normalization_mod.apply_normalization


def _instrumented_normalize(image: torch.Tensor) -> torch.Tensor:
    start = time.perf_counter()
    with nvtx_range("apply_normalization"):
        result = _orig_normalize(image)
    normalize_times.append(time.perf_counter() - start)
    return result


normalization_mod.apply_normalization = _instrumented_normalize  # type: ignore[assignment]

_orig_sizematcher = resizing_mod.apply_sizematcher


def _instrumented_sizematcher(
    image: torch.Tensor,
    max_height: int | None = None,
    max_width: int | None = None,
):
    start = time.perf_counter()
    with nvtx_range("apply_sizematcher"):
        result = _orig_sizematcher(image, max_height=max_height, max_width=max_width)
    size_match_times.append(time.perf_counter() - start)
    return result


resizing_mod.apply_sizematcher = _instrumented_sizematcher  # type: ignore[assignment]


def _instrument_function(fn, label: str, collector: List[float] | None = None):
    """Wrap a free function with NVTX and optional timing."""

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        with nvtx_range(label):
            result = fn(*args, **kwargs)
        if collector is not None:
            collector.append(time.perf_counter() - start)
        return result

    return wrapped


_orig_pad_to_stride = resizing_mod.apply_pad_to_stride


def _instrumented_pad_to_stride(image: torch.Tensor, max_stride: int) -> torch.Tensor:
    start = time.perf_counter()
    with nvtx_range("apply_pad_to_stride"):
        result = _orig_pad_to_stride(image, max_stride)
    pad_times.append(time.perf_counter() - start)
    return result


resizing_mod.apply_pad_to_stride = _instrumented_pad_to_stride  # type: ignore[assignment]


# Bottom-up peak finder
if hasattr(peak_finding_mod, "find_local_peaks"):
    peak_finding_mod.find_local_peaks = _instrument_function(  # type: ignore[assignment]
        peak_finding_mod.find_local_peaks,
        "find_local_peaks",
        local_peak_times,
    )


# ---- Module / method instrumentation ----


def _instrument_method(cls, method_name: str, label: str, collector: List[float] | None = None):
    """Wrap a class method with NVTX and optional timing."""

    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        start = time.perf_counter()
        with nvtx_range(label):
            result = original(self, *args, **kwargs)
        if collector is not None:
            collector.append(time.perf_counter() - start)
        return result

    setattr(cls, method_name, wrapped)


# Top-down inference stages
if hasattr(topdown_mod, "CentroidCrop"):
    _instrument_method(topdown_mod.CentroidCrop, "forward", "CentroidCrop.forward", centroid_times)

if hasattr(topdown_mod, "FindInstancePeaks"):
    _instrument_method(
        topdown_mod.FindInstancePeaks,
        "forward",
        "FindInstancePeaks.forward",
        instance_times,
    )

if hasattr(topdown_mod, "FindInstancePeaksGroundTruth"):
    _instrument_method(
        topdown_mod.FindInstancePeaksGroundTruth,
        "forward",
        "FindInstancePeaksGroundTruth.forward",
        instance_times,
    )

if hasattr(topdown_mod, "TopDownMultiClassFindInstancePeaks"):
    _instrument_method(
        topdown_mod.TopDownMultiClassFindInstancePeaks,
        "forward",
        "TopDownMultiClassFindInstancePeaks.forward",
        instance_times,
    )

if hasattr(topdown_mod, "TopDownInferenceModel"):
    _instrument_method(
        topdown_mod.TopDownInferenceModel,
        "forward",
        "TopDownInferenceModel.forward",
    )


# Bottom-up inference stages
if hasattr(bottomup_mod, "BottomUpInferenceModel"):
    _instrument_method(
        bottomup_mod.BottomUpInferenceModel,
        "forward",
        "BottomUpInferenceModel.forward",
        bottomup_forward_times,
    )

if hasattr(bottomup_mod, "BottomUpMultiClassInferenceModel"):
    _instrument_method(
        bottomup_mod.BottomUpMultiClassInferenceModel,
        "forward",
        "BottomUpMultiClassInferenceModel.forward",
        bottomup_forward_times,
    )

if hasattr(paf_grouping_mod, "PAFScorer"):
    _instrument_method(
        paf_grouping_mod.PAFScorer,
        "predict",
        "PAFScorer.predict",
        paf_predict_times,
    )


# Tracker
if hasattr(tracker_mod, "Tracker"):
    _instrument_method(tracker_mod.Tracker, "track", "Tracker.track", tracker_times)


# Predictor batch loop (CPU side)
Progress = predictors_mod.Progress
BarColumn = predictors_mod.BarColumn
TimeElapsedColumn = predictors_mod.TimeElapsedColumn
TimeRemainingColumn = predictors_mod.TimeRemainingColumn
MofNCompleteColumn = predictors_mod.MofNCompleteColumn
RateColumn = predictors_mod.RateColumn
apply_normalization_fn = predictors_mod.apply_normalization
apply_sizematcher_fn = predictors_mod.apply_sizematcher
apply_resizer_fn = predictors_mod.apply_resizer
resize_image_fn = predictors_mod.resize_image
apply_pad_to_stride_fn = predictors_mod.apply_pad_to_stride
F_mod = predictors_mod.F
torch_mod = predictors_mod.torch
time_fn = predictors_mod.time
logger_mod = predictors_mod.logger

_orig_predict_generator = predictors_mod.Predictor._predict_generator


def _instrumented_predict_generator(self):
    if self.inference_model is None:
        self._initialize_inference_model()

    self.pipeline.start()
    total_frames = self.pipeline.total_len()
    done = False

    try:
        with Progress(
            '{task.description}',
            BarColumn(),
            '[progress.percentage]{task.percentage:>3.0f}%',
            MofNCompleteColumn(),
            'ETA:',
            TimeRemainingColumn(),
            'Elapsed:',
            TimeElapsedColumn(),
            RateColumn(),
            auto_refresh=False,
            refresh_per_second=4,
            speed_estimate_period=5,
        ) as progress:

            task = progress.add_task('Predicting...', total=total_frames)
            last_report = time_fn()

            done = False
            while not done:
                imgs = []
                fidxs = []
                vidxs = []
                org_szs = []
                instances = []
                eff_scales = []

                batch_start = time.perf_counter()
                with nvtx_range('BatchPrepare'):
                    for _ in range(self.batch_size):
                        frame = self.pipeline.frame_buffer.get()
                        if frame['image'] is None:
                            done = True
                            break
                        frame['image'] = apply_normalization_fn(frame['image'])
                        frame['image'], eff_scale = apply_sizematcher_fn(
                            frame['image'],
                            self.preprocess_config['max_height'],
                            self.preprocess_config['max_width'],
                        )
                        if self.instances_key:
                            frame['instances'] = frame['instances'] * eff_scale
                        if (
                            self.preprocess_config['ensure_rgb']
                            and frame['image'].shape[-3] != 3
                        ):
                            frame['image'] = frame['image'].repeat(1, 3, 1, 1)
                        elif (
                            self.preprocess_config['ensure_grayscale']
                            and frame['image'].shape[-3] != 1
                        ):
                            frame['image'] = F_mod.rgb_to_grayscale(
                                frame['image'], num_output_channels=1
                            )

                        eff_scales.append(torch_mod.tensor(eff_scale))
                        imgs.append(frame['image'].unsqueeze(dim=0))
                        fidxs.append(frame['frame_idx'])
                        vidxs.append(frame['video_idx'])
                        org_szs.append(frame['orig_size'].unsqueeze(dim=0))
                        if self.instances_key:
                            instances.append(frame['instances'].unsqueeze(dim=0))
                prep_times.append(time.perf_counter() - batch_start)

                if imgs:
                    assemble_start = time.perf_counter()
                    with nvtx_range('BatchAssemble'):
                        imgs = torch_mod.concatenate(imgs, dim=0)
                        fidxs = torch_mod.tensor(fidxs, dtype=torch_mod.int32)
                        vidxs = torch_mod.tensor(vidxs, dtype=torch_mod.int32)
                        org_szs = torch_mod.concatenate(org_szs, dim=0)
                        eff_scales = torch_mod.tensor(eff_scales, dtype=torch_mod.float32)
                        if self.instances_key:
                            instances = torch_mod.concatenate(instances, dim=0)
                        ex = {
                            'image': imgs,
                            'frame_idx': fidxs,
                            'video_idx': vidxs,
                            'orig_size': org_szs,
                            'eff_scale': eff_scales,
                        }
                        if self.instances_key:
                            ex['instances'] = instances
                        if self.preprocess:
                            scale = self.preprocess_config['scale']
                            if scale != 1.0:
                                if self.instances_key:
                                    ex['image'], ex['instances'] = apply_resizer_fn(
                                        ex['image'], ex['instances']
                                    )
                                else:
                                    ex['image'] = resize_image_fn(ex['image'], scale)
                            ex['image'] = apply_pad_to_stride_fn(
                                ex['image'], self.max_stride
                            )
                    assemble_times.append(time.perf_counter() - assemble_start)

                    with nvtx_range('ModelForward'):
                        outputs_list = self.inference_model(ex)

                    if outputs_list is not None:
                        post_start = time.perf_counter()
                        with nvtx_range('Postprocess'):
                            for output in outputs_list:
                                output = self._convert_tensors_to_numpy(output)
                                yield output
                        postprocess_times.append(time.perf_counter() - post_start)

                    num_frames = (
                        len(ex['frame_idx'])
                        if 'frame_idx' in ex
                        else self.batch_size
                    )
                    progress.update(task, advance=num_frames)

                if time_fn() - last_report > 0.25:
                    progress.refresh()
                    last_report = time_fn()

    except KeyboardInterrupt:
        logger_mod.info('Inference interrupted by user')
        raise
    except Exception as exc:  # pragma: no cover
        message = f'Error in _predict_generator: {exc}'
        logger_mod.error(message)
        raise Exception(message)

    self.pipeline.join()

predictors_mod.Predictor._predict_generator = _instrumented_predict_generator  # type: ignore[assignment]



# ---- CLI helpers ----

def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile SLEAP-NN inference with Nsight",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        default="subset.mp4",
        help="Path to the video file to analyze (10-minute sample in this example).",
    )
    parser.add_argument(
        "--model",
        dest="model_paths",
        action="append",
        required=True,
        help="Path(s) to trained model directories (repeat for multi-part models).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional predictions output path. If omitted, defaults to the standard `.predictions.slp`.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--queue-maxsize", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device spec passed to run_inference (e.g. auto, cuda, cuda:0).",
    )
    parser.add_argument("--tracking", action="store_true", help="Enable tracking post-processing.")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--candidates-method",
        default="local_queues",
        choices=["fixed_window", "local_queues"],
    )
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="Optional frame indices to limit inference to.")
    return parser.parse_args(list(argv))


def _print_summary(label: str, samples: List[float]) -> None:
    if not samples:
        return
    avg_ms = mean(samples) * 1e3
    p95_ms = sorted(samples)[int(0.95 * len(samples)) - 1] * 1e3
    print(f"{label}: count={len(samples)} avg={avg_ms:.3f} ms p95={p95_ms:.3f} ms")


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    for model_dir in args.model_paths:
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

    with nvtx_range("run_inference"):
        labels = predict.run_inference(
            data_path=video_path.as_posix(),
            model_paths=args.model_paths,
            output_path=args.output,
            batch_size=args.batch_size,
            queue_maxsize=args.queue_maxsize,
            device=args.device,
            tracking=args.tracking,
            candidates_method=args.candidates_method,
            max_tracks=args.max_tracks,
            max_instances=args.max_instances,
            frames=args.frames,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\nProfiling summary")
    print("-----------------")
    _print_summary("frame_buffer.get", queue_wait_times)
    _print_summary("apply_normalization", normalize_times)
    _print_summary("apply_sizematcher", size_match_times)
    _print_summary("apply_pad_to_stride", pad_times)
    _print_summary("BatchPrepare", prep_times)
    _print_summary("BatchAssemble", assemble_times)
    _print_summary("Postprocess", postprocess_times)
    _print_summary("CentroidCrop.forward", centroid_times)
    _print_summary("InstancePeaks.forward", instance_times)
    _print_summary("BottomUpInferenceModel.forward", bottomup_forward_times)
    _print_summary("find_local_peaks", local_peak_times)
    _print_summary("PAFScorer.predict", paf_predict_times)
    _print_summary("Tracker.track", tracker_times)
    print(f"Predicted {len(labels)} labeled frames")


if __name__ == "__main__":
    main()
