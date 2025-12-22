"""Tests for sleap_nn.system_info module."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestParseDriverVersion:
    """Tests for parse_driver_version function."""

    def test_parse_simple_version(self):
        from sleap_nn.system_info import parse_driver_version

        assert parse_driver_version("560.76") == (560, 76)

    def test_parse_three_part_version(self):
        from sleap_nn.system_info import parse_driver_version

        assert parse_driver_version("560.28.03") == (560, 28, 3)

    def test_parse_invalid_version(self):
        from sleap_nn.system_info import parse_driver_version

        assert parse_driver_version("invalid") == (0,)

    def test_parse_empty_version(self):
        from sleap_nn.system_info import parse_driver_version

        assert parse_driver_version("") == (0,)


class TestGetMinDriverForCuda:
    """Tests for get_min_driver_for_cuda function."""

    def test_known_cuda_version(self):
        from sleap_nn.system_info import get_min_driver_for_cuda

        result = get_min_driver_for_cuda("12.6")
        assert result == ("560.28.03", "560.76")

    def test_cuda_version_with_patch(self):
        from sleap_nn.system_info import get_min_driver_for_cuda

        # Should strip patch version and match major.minor
        result = get_min_driver_for_cuda("12.6.1")
        assert result == ("560.28.03", "560.76")

    def test_unknown_cuda_version(self):
        from sleap_nn.system_info import get_min_driver_for_cuda

        result = get_min_driver_for_cuda("11.0")
        assert result is None

    def test_empty_cuda_version(self):
        from sleap_nn.system_info import get_min_driver_for_cuda

        result = get_min_driver_for_cuda("")
        assert result is None

    def test_none_cuda_version(self):
        from sleap_nn.system_info import get_min_driver_for_cuda

        result = get_min_driver_for_cuda(None)
        assert result is None


class TestCheckDriverCompatibility:
    """Tests for check_driver_compatibility function."""

    def test_compatible_driver_linux(self):
        from sleap_nn.system_info import check_driver_compatibility

        with patch.object(sys, "platform", "linux"):
            is_ok, min_ver = check_driver_compatibility("570.00", "12.6")
            assert is_ok is True
            assert min_ver == "560.28.03"

    def test_incompatible_driver_linux(self):
        from sleap_nn.system_info import check_driver_compatibility

        with patch.object(sys, "platform", "linux"):
            is_ok, min_ver = check_driver_compatibility("550.00", "12.6")
            assert is_ok is False
            assert min_ver == "560.28.03"

    def test_compatible_driver_windows(self):
        from sleap_nn.system_info import check_driver_compatibility

        with patch.object(sys, "platform", "win32"):
            is_ok, min_ver = check_driver_compatibility("565.00", "12.6")
            assert is_ok is True
            assert min_ver == "560.76"

    def test_unknown_cuda_version_skips_check(self):
        from sleap_nn.system_info import check_driver_compatibility

        is_ok, min_ver = check_driver_compatibility("500.00", "11.0")
        assert is_ok is True
        assert min_ver is None


class TestGetPackageInfo:
    """Tests for get_package_info function."""

    def test_installed_package(self):
        from sleap_nn.system_info import get_package_info

        # pytest is definitely installed
        info = get_package_info("pytest")
        assert info["version"] != "not installed"
        assert isinstance(info["editable"], bool)
        assert "location" in info
        assert "source" in info

    def test_not_installed_package(self):
        from sleap_nn.system_info import get_package_info

        info = get_package_info("definitely-not-a-real-package-12345")
        assert info["version"] == "not installed"
        assert info["editable"] is False
        assert info["location"] == ""
        assert info["source"] == ""


class TestGetNvidiaDriverVersion:
    """Tests for get_nvidia_driver_version function."""

    def test_nvidia_smi_not_found(self):
        from sleap_nn.system_info import get_nvidia_driver_version

        with patch("shutil.which", return_value=None):
            result = get_nvidia_driver_version()
            assert result is None

    def test_nvidia_smi_success(self):
        from sleap_nn.system_info import get_nvidia_driver_version

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "560.76\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                result = get_nvidia_driver_version()
                assert result == "560.76"

    def test_nvidia_smi_failure(self):
        from sleap_nn.system_info import get_nvidia_driver_version

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                result = get_nvidia_driver_version()
                assert result is None


class TestGetSystemInfoDict:
    """Tests for get_system_info_dict function."""

    def test_returns_dict_with_expected_keys(self):
        from sleap_nn.system_info import get_system_info_dict

        info = get_system_info_dict()

        assert isinstance(info, dict)
        assert "python_version" in info
        assert "platform" in info
        assert "pytorch_version" in info
        assert "cuda_available" in info
        assert "accelerator" in info
        assert info["accelerator"] in ("cpu", "cuda", "mps")
        assert "packages" in info
        assert "gpus" in info

    def test_packages_includes_expected_packages(self):
        from sleap_nn.system_info import PACKAGES, get_system_info_dict

        info = get_system_info_dict()

        for pkg in PACKAGES:
            assert pkg in info["packages"]
            assert "version" in info["packages"][pkg]
            assert "editable" in info["packages"][pkg]
            assert "location" in info["packages"][pkg]
            assert "source" in info["packages"][pkg]


class TestGetStartupInfoString:
    """Tests for get_startup_info_string function."""

    def test_returns_string(self):
        from sleap_nn.system_info import get_startup_info_string

        result = get_startup_info_string()
        assert isinstance(result, str)

    def test_contains_version_info(self):
        from sleap_nn.system_info import get_startup_info_string

        result = get_startup_info_string()
        assert "sleap-nn" in result
        assert "Python" in result
        assert "PyTorch" in result


class TestPrintSystemInfo:
    """Tests for print_system_info function."""

    def test_runs_without_error(self, capsys):
        from sleap_nn.system_info import print_system_info

        # Should run without raising exceptions
        print_system_info()

        # Should produce some output
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestTestGpuOperations:
    """Tests for test_gpu_operations function."""

    def test_returns_tuple(self):
        from sleap_nn.system_info import test_gpu_operations

        result = test_gpu_operations()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
