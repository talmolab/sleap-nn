"""Inference-related modules.

Quick start::

    from sleap_nn.inference import predict, get_predictor_from_model_paths

    # One-liner: source + model paths → Labels
    labels = predict("video.mp4", model_paths=["/path/to/model"])

    # Two-step: build once, predict many times with different settings
    predictor = get_predictor_from_model_paths(["/path/to/model"], device="cuda")
    labels = predictor.predict("video.mp4", peak_threshold=0.3)
"""

from sleap_nn.inference.factory import (
    get_predictor_from_export_dir,
    get_predictor_from_model_paths,
)
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.provenance import (
    build_inference_provenance,
    build_tracking_only_provenance,
    merge_provenance,
)
from sleap_nn.inference.run import predict
