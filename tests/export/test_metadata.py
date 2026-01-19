"""Tests for sleap_nn.export.metadata module."""

import json
import pytest
from pathlib import Path

from sleap_nn.export.metadata import (
    ExportMetadata,
    hash_file,
    build_base_metadata,
    embed_metadata_in_onnx,
    extract_metadata_from_onnx,
)
from sleap_nn import __version__

from .conftest import requires_onnx


class TestExportMetadata:
    """Tests for ExportMetadata dataclass."""

    def test_export_metadata_defaults(self):
        """Test that default values are correctly set for optional fields."""
        metadata = ExportMetadata(
            sleap_nn_version=__version__,
            export_timestamp="2026-01-18T12:00:00",
            export_format="onnx",
            model_type="single_instance",
            model_name="test_model",
            checkpoint_path="/path/to/ckpt",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
        )

        # Check defaults
        assert metadata.crop_size is None
        assert metadata.max_instances is None
        assert metadata.max_peaks_per_node is None
        assert metadata.max_batch_size == 1
        assert metadata.precision == "fp32"
        assert metadata.input_dtype == "uint8"
        assert metadata.normalization == "0_to_1"
        assert metadata.n_classes is None
        assert metadata.class_names is None
        assert metadata.training_config_embedded is False
        assert metadata.training_config_hash == ""

    def test_export_metadata_to_dict(self):
        """Test that to_dict() produces a serializable dict with all keys."""
        metadata = ExportMetadata(
            sleap_nn_version=__version__,
            export_timestamp="2026-01-18T12:00:00",
            export_format="onnx",
            model_type="bottomup",
            model_name="test_model",
            checkpoint_path="/path/to/ckpt",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
            crop_size=(192, 192),
            max_instances=10,
            max_peaks_per_node=20,
        )

        d = metadata.to_dict()

        # Check all expected keys are present
        expected_keys = {
            "sleap_nn_version",
            "export_timestamp",
            "export_format",
            "model_type",
            "model_name",
            "checkpoint_path",
            "backbone",
            "n_nodes",
            "n_edges",
            "node_names",
            "edge_inds",
            "input_scale",
            "input_channels",
            "output_stride",
            "crop_size",
            "max_instances",
            "max_peaks_per_node",
            "max_batch_size",
            "precision",
            "input_dtype",
            "normalization",
            "n_classes",
            "class_names",
            "training_config_embedded",
            "training_config_hash",
        }
        assert set(d.keys()) == expected_keys

        # Check that edge_inds is converted to lists (for JSON serialization)
        assert d["edge_inds"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
        assert d["crop_size"] == [192, 192]

        # Verify it's JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_export_metadata_from_dict_full(self):
        """Test that from_dict() can deserialize a complete dict."""
        data = {
            "sleap_nn_version": "1.0.0",
            "export_timestamp": "2026-01-18T12:00:00",
            "export_format": "tensorrt",
            "model_type": "topdown",
            "model_name": "my_model",
            "checkpoint_path": "/path/to/model",
            "backbone": "convnext",
            "n_nodes": 10,
            "n_edges": 9,
            "node_names": [f"node{i}" for i in range(10)],
            "edge_inds": [[i, i + 1] for i in range(9)],
            "input_scale": 0.5,
            "input_channels": 3,
            "output_stride": 2,
            "crop_size": [256, 256],
            "max_instances": 5,
            "max_peaks_per_node": 15,
            "max_batch_size": 8,
            "precision": "fp16",
            "input_dtype": "uint8",
            "normalization": "0_to_1",
            "n_classes": 2,
            "class_names": ["female", "male"],
            "training_config_embedded": True,
            "training_config_hash": "abc123",
        }

        metadata = ExportMetadata.from_dict(data)

        assert metadata.sleap_nn_version == "1.0.0"
        assert metadata.model_type == "topdown"
        assert metadata.n_nodes == 10
        assert metadata.edge_inds == [(i, i + 1) for i in range(9)]
        assert metadata.crop_size == (256, 256)
        assert metadata.precision == "fp16"
        assert metadata.n_classes == 2
        assert metadata.class_names == ["female", "male"]
        assert metadata.training_config_embedded is True

    def test_export_metadata_from_dict_minimal(self):
        """Test that from_dict() handles a dict with only required fields."""
        # Minimal data with defaults
        data = {
            "model_type": "single_instance",
            "n_nodes": 5,
        }

        metadata = ExportMetadata.from_dict(data)

        assert metadata.model_type == "single_instance"
        assert metadata.n_nodes == 5
        # Check defaults are applied
        assert metadata.sleap_nn_version == ""
        assert metadata.input_scale == 1.0
        assert metadata.input_dtype == "uint8"
        assert metadata.normalization == "0_to_1"
        assert metadata.edge_inds == []
        assert metadata.crop_size is None

    def test_export_metadata_from_dict_unknown_keys(self):
        """Test that unknown keys in dict are ignored gracefully."""
        data = {
            "sleap_nn_version": "1.0.0",
            "export_timestamp": "2026-01-18",
            "export_format": "onnx",
            "model_type": "centroid",
            "model_name": "test",
            "checkpoint_path": "/path",
            "backbone": "unet",
            "n_nodes": 1,
            "n_edges": 0,
            "node_names": ["centroid"],
            "edge_inds": [],
            "input_scale": 1.0,
            "input_channels": 1,
            "output_stride": 4,
            # Unknown keys that should be ignored
            "unknown_field": "should_be_ignored",
            "another_unknown": 12345,
        }

        # Should not raise an error
        metadata = ExportMetadata.from_dict(data)
        assert metadata.model_type == "centroid"
        assert metadata.n_nodes == 1

    def test_export_metadata_save_load_roundtrip(self, tmp_path):
        """Test that save() and load() produce identical metadata."""
        original = ExportMetadata(
            sleap_nn_version=__version__,
            export_timestamp="2026-01-18T12:00:00",
            export_format="onnx",
            model_type="bottomup",
            model_name="roundtrip_test",
            checkpoint_path="/path/to/ckpt",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=0.75,
            input_channels=1,
            output_stride=4,
            max_instances=10,
            max_peaks_per_node=20,
        )

        json_path = tmp_path / "metadata.json"
        original.save(json_path)

        # Verify file was created
        assert json_path.exists()

        # Load and compare
        loaded = ExportMetadata.load(json_path)

        assert loaded.sleap_nn_version == original.sleap_nn_version
        assert loaded.export_timestamp == original.export_timestamp
        assert loaded.model_type == original.model_type
        assert loaded.n_nodes == original.n_nodes
        assert loaded.edge_inds == original.edge_inds
        assert loaded.input_scale == original.input_scale
        assert loaded.max_instances == original.max_instances
        assert loaded.max_peaks_per_node == original.max_peaks_per_node

    def test_export_metadata_multiclass_fields(self):
        """Test that n_classes and class_names are optional and work correctly."""
        # Without multiclass fields
        metadata_no_class = ExportMetadata(
            sleap_nn_version=__version__,
            export_timestamp="2026-01-18",
            export_format="onnx",
            model_type="bottomup",
            model_name="test",
            checkpoint_path="/path",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
        )
        assert metadata_no_class.n_classes is None
        assert metadata_no_class.class_names is None

        # With multiclass fields
        metadata_with_class = ExportMetadata(
            sleap_nn_version=__version__,
            export_timestamp="2026-01-18",
            export_format="onnx",
            model_type="multi_class_bottomup",
            model_name="test",
            checkpoint_path="/path",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
            n_classes=2,
            class_names=["female", "male"],
        )
        assert metadata_with_class.n_classes == 2
        assert metadata_with_class.class_names == ["female", "male"]

        # Check roundtrip with multiclass
        d = metadata_with_class.to_dict()
        loaded = ExportMetadata.from_dict(d)
        assert loaded.n_classes == 2
        assert loaded.class_names == ["female", "male"]

    def test_default_timestamp(self):
        """Test that default_timestamp() returns a valid ISO format string."""
        ts = ExportMetadata.default_timestamp()
        assert isinstance(ts, str)
        # Should contain date/time separators
        assert "T" in ts or "-" in ts


class TestBuildBaseMetadata:
    """Tests for build_base_metadata helper function."""

    def test_build_base_metadata(self):
        """Test build_base_metadata with typical inputs."""
        metadata = build_base_metadata(
            export_format="onnx",
            model_type="single_instance",
            model_name="test_model",
            checkpoint_path="/path/to/ckpt",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
        )

        assert isinstance(metadata, ExportMetadata)
        assert metadata.sleap_nn_version == __version__
        assert metadata.model_type == "single_instance"
        assert metadata.n_nodes == 5
        # Should have a timestamp set
        assert metadata.export_timestamp != ""

    def test_build_base_metadata_with_classes(self):
        """Test build_base_metadata with multiclass fields."""
        metadata = build_base_metadata(
            export_format="onnx",
            model_type="multi_class_bottomup",
            model_name="multiclass_test",
            checkpoint_path="/path/to/ckpt",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
            n_classes=3,
            class_names=["class_a", "class_b", "class_c"],
        )

        assert metadata.n_classes == 3
        assert metadata.class_names == ["class_a", "class_b", "class_c"]


class TestHashFile:
    """Tests for hash_file utility function."""

    def test_hash_file(self, tmp_path):
        """Test that hash_file returns a consistent SHA256 hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = hash_file(test_file)
        hash2 = hash_file(test_file)

        # Hash should be consistent
        assert hash1 == hash2
        # Hash should be a 64-character hex string (SHA256)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_hash_file_different_content(self, tmp_path):
        """Test that different file contents produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        assert hash1 != hash2


class TestONNXMetadataEmbedding:
    """Tests for ONNX metadata embedding/extraction (requires onnx)."""

    @requires_onnx
    def test_embed_metadata_in_onnx(self, tmp_path, mock_single_instance_wrapper):
        """Test that metadata can be embedded in an ONNX model."""
        from sleap_nn.export.exporters import export_to_onnx

        onnx_path = tmp_path / "model.onnx"

        # Export a simple model
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            verify=False,
        )

        metadata = build_base_metadata(
            export_format="onnx",
            model_type="single_instance",
            model_name="test",
            checkpoint_path="/path",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
        )

        # Embed metadata
        embed_metadata_in_onnx(onnx_path, metadata)

        # Verify by loading the model
        import onnx

        model = onnx.load(str(onnx_path))
        found = False
        for prop in model.metadata_props:
            if prop.key == "sleap_nn_metadata":
                found = True
                data = json.loads(prop.value)
                assert data["model_type"] == "single_instance"
                break
        assert found, "Metadata not found in ONNX model"

    @requires_onnx
    def test_extract_metadata_from_onnx(self, tmp_path, mock_single_instance_wrapper):
        """Test that embedded metadata can be extracted from ONNX."""
        from sleap_nn.export.exporters import export_to_onnx

        onnx_path = tmp_path / "model.onnx"

        # Export and embed
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            verify=False,
        )

        original_metadata = build_base_metadata(
            export_format="onnx",
            model_type="single_instance",
            model_name="extraction_test",
            checkpoint_path="/path",
            backbone="unet",
            n_nodes=5,
            n_edges=4,
            node_names=["n0", "n1", "n2", "n3", "n4"],
            edge_inds=[(0, 1), (1, 2), (2, 3), (3, 4)],
            input_scale=1.0,
            input_channels=1,
            output_stride=4,
        )

        embed_metadata_in_onnx(onnx_path, original_metadata)

        # Extract and compare
        extracted = extract_metadata_from_onnx(onnx_path)

        assert extracted.model_type == original_metadata.model_type
        assert extracted.model_name == original_metadata.model_name
        assert extracted.n_nodes == original_metadata.n_nodes
        assert extracted.edge_inds == original_metadata.edge_inds

    @requires_onnx
    def test_extract_metadata_missing_raises(
        self, tmp_path, mock_single_instance_wrapper
    ):
        """Test that extract raises ValueError if metadata is missing."""
        from sleap_nn.export.exporters import export_to_onnx

        onnx_path = tmp_path / "model_no_meta.onnx"

        # Export without embedding metadata
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            verify=False,
        )

        with pytest.raises(ValueError, match="No sleap_nn metadata found"):
            extract_metadata_from_onnx(onnx_path)
