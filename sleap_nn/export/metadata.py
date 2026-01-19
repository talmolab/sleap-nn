"""Metadata helpers for exported models."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json

from sleap_nn import __version__


@dataclass
class ExportMetadata:
    """Metadata embedded or saved alongside exported models."""

    # Version info
    sleap_nn_version: str
    export_timestamp: str
    export_format: str  # "onnx" or "tensorrt"

    # Model info
    model_type: str  # "centroid", "centered_instance", "topdown", "bottomup"
    model_name: str
    checkpoint_path: str

    # Architecture
    backbone: str
    n_nodes: int
    n_edges: int
    node_names: List[str]
    edge_inds: List[Tuple[int, int]]

    # Input/output spec
    input_scale: float
    input_channels: int
    output_stride: int
    crop_size: Optional[Tuple[int, int]] = None

    # Export parameters
    max_instances: Optional[int] = None
    max_peaks_per_node: Optional[int] = None
    max_batch_size: int = 1
    precision: str = "fp32"

    # Preprocessing - input is uint8 [0,255], normalized internally to float32 [0,1]
    input_dtype: str = "uint8"
    normalization: str = "0_to_1"

    # Multiclass model fields (optional)
    n_classes: Optional[int] = None
    class_names: Optional[List[str]] = None

    # Training config reference
    training_config_embedded: bool = False
    training_config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        data["edge_inds"] = [list(pair) for pair in self.edge_inds]
        if self.crop_size is not None:
            data["crop_size"] = list(self.crop_size)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportMetadata":
        """Load from dict."""
        edge_inds = [tuple(pair) for pair in data.get("edge_inds", [])]
        crop_size = data.get("crop_size")
        if crop_size is not None:
            crop_size = tuple(crop_size)
        return cls(
            sleap_nn_version=data.get("sleap_nn_version", ""),
            export_timestamp=data.get("export_timestamp", ""),
            export_format=data.get("export_format", ""),
            model_type=data.get("model_type", ""),
            model_name=data.get("model_name", ""),
            checkpoint_path=data.get("checkpoint_path", ""),
            backbone=data.get("backbone", ""),
            n_nodes=int(data.get("n_nodes", 0)),
            n_edges=int(data.get("n_edges", 0)),
            node_names=list(data.get("node_names", [])),
            edge_inds=edge_inds,
            input_scale=float(data.get("input_scale", 1.0)),
            input_channels=int(data.get("input_channels", 1)),
            output_stride=int(data.get("output_stride", 1)),
            crop_size=crop_size,
            max_instances=data.get("max_instances"),
            max_peaks_per_node=data.get("max_peaks_per_node"),
            max_batch_size=int(data.get("max_batch_size", 1)),
            precision=data.get("precision", "fp32"),
            input_dtype=data.get("input_dtype", "uint8"),
            normalization=data.get("normalization", "0_to_1"),
            n_classes=data.get("n_classes"),
            class_names=data.get("class_names"),
            training_config_embedded=bool(data.get("training_config_embedded", False)),
            training_config_hash=data.get("training_config_hash", ""),
        )

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "ExportMetadata":
        """Load from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def default_timestamp(cls) -> str:
        """Return an ISO timestamp for export."""
        return datetime.now().isoformat()


def hash_file(path: str | Path) -> str:
    """Compute SHA256 hash for a file."""
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_base_metadata(
    *,
    export_format: str,
    model_type: str,
    model_name: str,
    checkpoint_path: str,
    backbone: str,
    n_nodes: int,
    n_edges: int,
    node_names: List[str],
    edge_inds: List[Tuple[int, int]],
    input_scale: float,
    input_channels: int,
    output_stride: int,
    crop_size: Optional[Tuple[int, int]] = None,
    max_instances: Optional[int] = None,
    max_peaks_per_node: Optional[int] = None,
    max_batch_size: int = 1,
    precision: str = "fp32",
    training_config_hash: str = "",
    training_config_embedded: bool = False,
    input_dtype: str = "uint8",
    normalization: str = "0_to_1",
    n_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> ExportMetadata:
    """Create an ExportMetadata instance with standard defaults."""
    return ExportMetadata(
        sleap_nn_version=__version__,
        export_timestamp=ExportMetadata.default_timestamp(),
        export_format=export_format,
        model_type=model_type,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_names=node_names,
        edge_inds=edge_inds,
        input_scale=input_scale,
        input_channels=input_channels,
        output_stride=output_stride,
        crop_size=crop_size,
        max_instances=max_instances,
        max_peaks_per_node=max_peaks_per_node,
        max_batch_size=max_batch_size,
        precision=precision,
        input_dtype=input_dtype,
        normalization=normalization,
        n_classes=n_classes,
        class_names=class_names,
        training_config_embedded=training_config_embedded,
        training_config_hash=training_config_hash,
    )


def embed_metadata_in_onnx(
    model_path: str | Path,
    metadata: ExportMetadata,
    training_config_text: Optional[str] = None,
) -> None:
    """Embed metadata into an ONNX model file.

    Raises ImportError if onnx is unavailable.
    """
    import onnx  # local import to keep dependency optional

    model_path = Path(model_path)
    model = onnx.load(model_path.as_posix())
    model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="sleap_nn_metadata", value=json.dumps(metadata.to_dict())
        )
    )
    if training_config_text:
        model.metadata_props.append(
            onnx.StringStringEntryProto(
                key="training_config", value=training_config_text
            )
        )
    onnx.save(model, model_path.as_posix())


def extract_metadata_from_onnx(model_path: str | Path) -> ExportMetadata:
    """Extract metadata from an ONNX model file.

    Raises ValueError if metadata is missing.
    """
    import onnx  # local import to keep dependency optional

    model = onnx.load(Path(model_path).as_posix())
    for prop in model.metadata_props:
        if prop.key == "sleap_nn_metadata":
            return ExportMetadata.from_dict(json.loads(prop.value))
    raise ValueError("No sleap_nn metadata found in ONNX model")
