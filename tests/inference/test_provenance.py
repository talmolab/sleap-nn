"""Tests for inference provenance utilities."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sleap_io as sio

from sleap_nn.inference.provenance import (
    build_inference_provenance,
    build_tracking_only_provenance,
    merge_provenance,
)


class TestBuildInferenceProvenance:
    """Tests for build_inference_provenance function."""

    def test_minimal_provenance(self):
        """Test building provenance with minimal arguments."""
        provenance = build_inference_provenance()

        assert "sleap_nn_version" in provenance
        assert "sleap_io_version" in provenance
        assert "system_info" in provenance

    def test_timestamps(self):
        """Test timestamp handling."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 10, 5, 30)

        provenance = build_inference_provenance(
            start_time=start,
            end_time=end,
            include_system_info=False,
        )

        assert provenance["inference_start_timestamp"] == "2024-01-15T10:00:00"
        assert provenance["inference_end_timestamp"] == "2024-01-15T10:05:30"
        assert provenance["inference_runtime_seconds"] == 330.0

    def test_model_paths(self):
        """Test model path handling - paths are resolved to absolute."""
        model_paths = ["/path/to/model1.ckpt", Path("/path/to/model2.ckpt")]

        provenance = build_inference_provenance(
            model_paths=model_paths,
            model_type="top_down",
            include_system_info=False,
        )

        # Paths are resolved to absolute
        assert "/path/to/model1.ckpt" in provenance["model_paths"][0]
        assert "/path/to/model2.ckpt" in provenance["model_paths"][1]
        assert provenance["model_type"] == "top_down"

    def test_input_path(self):
        """Test input path handling - paths are resolved to absolute."""
        provenance = build_inference_provenance(
            input_path="/path/to/input.slp",
            include_system_info=False,
        )

        assert "/path/to/input.slp" in provenance["source_file"]

    def test_input_path_as_path_object(self):
        """Test input path as Path object - resolved to absolute."""
        provenance = build_inference_provenance(
            input_path=Path("/path/to/input.slp"),
            include_system_info=False,
        )

        assert "/path/to/input.slp" in provenance["source_file"]

    def test_input_labels_provenance_preservation(self):
        """Test that input labels provenance is preserved."""
        # Create mock input labels with provenance
        input_labels = MagicMock(spec=sio.Labels)
        input_labels.provenance = {
            "filename": "/path/to/original.slp",
            "sleap_version": "1.2.7",
            "custom_field": "custom_value",
        }

        provenance = build_inference_provenance(
            input_labels=input_labels,
            include_system_info=False,
        )

        assert "input_provenance" in provenance
        assert provenance["input_provenance"]["filename"] == "/path/to/original.slp"
        assert provenance["input_provenance"]["sleap_version"] == "1.2.7"
        assert provenance["source_labels"] == "/path/to/original.slp"

    def test_frame_selection(self):
        """Test frame selection information."""
        provenance = build_inference_provenance(
            frames_processed=100,
            frames_total=500,
            frame_selection_method="labeled",
            include_system_info=False,
        )

        assert provenance["frame_selection"]["method"] == "labeled"
        assert provenance["frame_selection"]["frames_processed"] == 100
        assert provenance["frame_selection"]["frames_total"] == 500

    def test_inference_params(self):
        """Test inference parameters."""
        inference_params = {
            "peak_threshold": 0.2,
            "integral_refinement": "integral",
            "batch_size": 4,
            "unused_param": None,  # Should be filtered out
        }

        provenance = build_inference_provenance(
            inference_params=inference_params,
            include_system_info=False,
        )

        assert provenance["inference_config"]["peak_threshold"] == 0.2
        assert provenance["inference_config"]["integral_refinement"] == "integral"
        assert provenance["inference_config"]["batch_size"] == 4
        assert "unused_param" not in provenance["inference_config"]

    def test_tracking_params(self):
        """Test tracking parameters."""
        tracking_params = {
            "window_size": 5,
            "candidates_method": "fixed_window",
            "scoring_method": "oks",
        }

        provenance = build_inference_provenance(
            tracking_params=tracking_params,
            include_system_info=False,
        )

        assert provenance["tracking_config"]["window_size"] == 5
        assert provenance["tracking_config"]["candidates_method"] == "fixed_window"

    def test_device(self):
        """Test device information."""
        provenance = build_inference_provenance(
            device="cuda:0",
            include_system_info=False,
        )

        assert provenance["device"] == "cuda:0"

    def test_cli_args(self):
        """Test CLI arguments."""
        cli_args = {
            "model_paths": ["/path/to/model.ckpt"],
            "batch_size": 4,
            "empty_arg": None,  # Should be filtered
        }

        provenance = build_inference_provenance(
            cli_args=cli_args,
            include_system_info=False,
        )

        assert "cli_args" in provenance
        assert "model_paths" in provenance["cli_args"]
        assert "empty_arg" not in provenance["cli_args"]

    def test_system_info_included(self):
        """Test that system info is included by default."""
        provenance = build_inference_provenance(include_system_info=True)

        assert "system_info" in provenance
        assert "python_version" in provenance["system_info"]
        assert "platform" in provenance["system_info"]
        assert "pytorch_version" in provenance["system_info"]

    def test_system_info_excluded(self):
        """Test that system info can be excluded."""
        provenance = build_inference_provenance(include_system_info=False)

        assert "system_info" not in provenance

    @patch("sleap_nn.inference.provenance.get_system_info_dict")
    def test_system_info_failure_handled(self, mock_system_info):
        """Test that system info collection failure is handled gracefully."""
        mock_system_info.side_effect = Exception("System info error")

        # Should not raise an exception
        provenance = build_inference_provenance(include_system_info=True)

        # Should have version info but no system_info
        assert "sleap_nn_version" in provenance


class TestBuildTrackingOnlyProvenance:
    """Tests for build_tracking_only_provenance function."""

    def test_minimal_tracking_provenance(self):
        """Test building tracking-only provenance with minimal arguments."""
        provenance = build_tracking_only_provenance()

        assert "sleap_nn_version" in provenance
        assert "sleap_io_version" in provenance
        assert provenance["pipeline_type"] == "tracking_only"

    def test_tracking_timestamps(self):
        """Test timestamp handling for tracking."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 10, 2, 0)

        provenance = build_tracking_only_provenance(
            start_time=start,
            end_time=end,
            include_system_info=False,
        )

        assert provenance["tracking_start_timestamp"] == "2024-01-15T10:00:00"
        assert provenance["tracking_end_timestamp"] == "2024-01-15T10:02:00"
        assert provenance["tracking_runtime_seconds"] == 120.0

    def test_tracking_params(self):
        """Test tracking parameters in tracking-only provenance."""
        tracking_params = {
            "window_size": 10,
            "candidates_method": "local_queues",
            "scoring_method": "iou",
        }

        provenance = build_tracking_only_provenance(
            tracking_params=tracking_params,
            frames_processed=500,
            include_system_info=False,
        )

        assert provenance["tracking_config"]["window_size"] == 10
        assert provenance["frames_processed"] == 500

    def test_input_provenance_preservation(self):
        """Test that input provenance is preserved in tracking-only mode."""
        input_labels = MagicMock(spec=sio.Labels)
        input_labels.provenance = {
            "filename": "/path/to/predictions.slp",
            "model_paths": ["/path/to/model.ckpt"],
        }

        provenance = build_tracking_only_provenance(
            input_labels=input_labels,
            input_path="/path/to/predictions.slp",
            include_system_info=False,
        )

        assert provenance["source_labels"] == "/path/to/predictions.slp"
        assert "input_provenance" in provenance


class TestMergeProvenance:
    """Tests for merge_provenance function."""

    def test_merge_overwrites(self):
        """Test that merge overwrites by default."""
        base = {"field1": "base_value", "field2": "keep"}
        additional = {"field1": "new_value", "field3": "added"}

        result = merge_provenance(base, additional, overwrite=True)

        assert result["field1"] == "new_value"
        assert result["field2"] == "keep"
        assert result["field3"] == "added"

    def test_merge_no_overwrite(self):
        """Test that merge preserves base when overwrite=False."""
        base = {"field1": "base_value", "field2": "keep"}
        additional = {"field1": "new_value", "field3": "added"}

        result = merge_provenance(base, additional, overwrite=False)

        assert result["field1"] == "base_value"
        assert result["field2"] == "keep"
        assert result["field3"] == "added"

    def test_merge_does_not_modify_originals(self):
        """Test that merge creates a new dictionary."""
        base = {"field1": "value1"}
        additional = {"field2": "value2"}

        result = merge_provenance(base, additional)

        assert result is not base
        assert "field2" not in base
