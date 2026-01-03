"""Tests for sleap_nn.system_info module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestShortenPath:
    """Tests for _shorten_path function."""

    def test_short_path_unchanged(self):
        """Test that paths shorter than max_len are unchanged."""
        from sleap_nn.system_info import _shorten_path

        result = _shorten_path("short/path", 40)
        assert result == "short/path"

    def test_long_path_truncated(self):
        """Test that long paths are truncated with ... prefix."""
        from sleap_nn.system_info import _shorten_path

        long_path = "/very/long/path/that/exceeds/the/maximum/length/allowed/here"
        result = _shorten_path(long_path, 40)
        assert result.startswith("...")
        assert len(result) == 40
        assert result.endswith("here")

    def test_exact_length_unchanged(self):
        """Test that paths exactly at max_len are unchanged."""
        from sleap_nn.system_info import _shorten_path

        path = "x" * 40
        result = _shorten_path(path, 40)
        assert result == path


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

    def test_single_part_version(self):
        """Test that single-part version (e.g., '12') returns None."""
        from sleap_nn.system_info import get_min_driver_for_cuda

        result = get_min_driver_for_cuda("12")
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


class TestGetPackageLocation:
    """Tests for _get_package_location function."""

    def test_package_with_file_attribute(self):
        """Test getting location from package's __file__ attribute."""
        from sleap_nn.system_info import _get_package_location

        # Use a real package that has __file__
        import importlib.metadata

        dist = importlib.metadata.distribution("pytest")
        result = _get_package_location("pytest", dist)
        assert len(result) > 0
        assert "pytest" in result.lower() or "site-packages" in result.lower()

    def test_import_error_fallback(self):
        """Test fallback when import fails and dist._path is None."""
        from sleap_nn.system_info import _get_package_location

        mock_dist = MagicMock()
        mock_dist._path = None

        # Only mock the specific module import, not all imports
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "nonexistent_pkg":
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _get_package_location("nonexistent-pkg", mock_dist)
            assert result == ""

    def test_dist_path_fallback(self):
        """Test fallback to dist._path when import fails."""
        from sleap_nn.system_info import _get_package_location

        mock_dist = MagicMock()
        # Use an actual absolute Path for parent
        abs_path = Path("/some/site-packages").resolve()
        mock_dist._path.parent = abs_path

        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "some_pkg":
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _get_package_location("some-pkg", mock_dist)
            assert result == str(abs_path)

    def test_dist_path_relative(self):
        """Test that relative dist._path is made absolute."""
        from sleap_nn.system_info import _get_package_location

        mock_dist = MagicMock()
        # Use an actual relative Path for parent
        relative_path = Path("relative/path")
        mock_dist._path.parent = relative_path

        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "some_pkg":
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _get_package_location("some-pkg", mock_dist)
            # Should be made absolute
            assert Path(result).is_absolute()


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

    def test_editable_install_via_direct_url(self):
        """Test detection of editable install via direct_url.json."""
        from sleap_nn.system_info import get_package_info

        mock_dist = MagicMock()
        mock_dist.version = "1.0.0"
        mock_dist._path = None
        mock_dist.read_text.side_effect = lambda f: (
            '{"dir_info": {"editable": true}, "url": "file:///path/to/pkg"}'
            if f == "direct_url.json"
            else None
        )

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            with patch(
                "sleap_nn.system_info._get_package_location", return_value="/path/to"
            ):
                info = get_package_info("test-pkg")
                assert info["editable"] is True
                assert info["source"] == "editable"

    def test_git_install_via_direct_url(self):
        """Test detection of git install via direct_url.json."""
        from sleap_nn.system_info import get_package_info

        mock_dist = MagicMock()
        mock_dist.version = "1.0.0"
        mock_dist._path = None
        mock_dist.read_text.side_effect = lambda f: (
            '{"vcs_info": {"vcs": "git"}, "url": "https://github.com/org/repo"}'
            if f == "direct_url.json"
            else None
        )

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            with patch(
                "sleap_nn.system_info._get_package_location", return_value="/path/to"
            ):
                info = get_package_info("test-pkg")
                assert info["source"] == "git"

    def test_local_install_via_direct_url(self):
        """Test detection of local file install via direct_url.json."""
        from sleap_nn.system_info import get_package_info

        mock_dist = MagicMock()
        mock_dist.version = "1.0.0"
        mock_dist._path = None
        mock_dist.read_text.side_effect = lambda f: (
            '{"url": "file:///local/path/to/pkg"}' if f == "direct_url.json" else None
        )

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            with patch(
                "sleap_nn.system_info._get_package_location", return_value="/path/to"
            ):
                info = get_package_info("test-pkg")
                assert info["source"] == "local"

    def test_conda_install_via_installer(self):
        """Test detection of conda install via INSTALLER file."""
        from sleap_nn.system_info import get_package_info

        mock_dist = MagicMock()
        mock_dist.version = "1.0.0"
        mock_dist._path = None

        def read_text_side_effect(filename):
            if filename == "direct_url.json":
                raise FileNotFoundError()
            if filename == "INSTALLER":
                return "conda\n"
            return None

        mock_dist.read_text.side_effect = read_text_side_effect

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            with patch(
                "sleap_nn.system_info._get_package_location", return_value="/path/to"
            ):
                info = get_package_info("test-pkg")
                assert info["source"] == "conda"


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

    def test_nvidia_smi_exception(self):
        """Test that exceptions from subprocess are handled gracefully."""
        from sleap_nn.system_info import get_nvidia_driver_version

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", side_effect=Exception("Timeout")):
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

    def test_mps_accelerator_detection(self):
        """Test MPS accelerator detection on Apple Silicon (runs on Mac with MPS)."""
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available - skipping MPS test")

        from sleap_nn.system_info import get_system_info_dict

        info = get_system_info_dict()
        assert info["accelerator"] == "mps"
        assert info["mps_available"] is True
        assert info["gpu_count"] == 1


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

    def test_mps_accelerator_string(self):
        """Test startup string with MPS accelerator (runs on Mac with MPS)."""
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available - skipping MPS test")

        from sleap_nn.system_info import get_startup_info_string

        result = get_startup_info_string()
        assert "MPS" in result

    def test_cpu_only_string(self):
        """Test startup string with CPU only (runs when no GPU available)."""
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA available - skipping CPU-only test")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pytest.skip("MPS available - skipping CPU-only test")

        from sleap_nn.system_info import get_startup_info_string

        result = get_startup_info_string()
        assert "CPU only" in result


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

    def test_mps_gpu_operations(self):
        """Test GPU operations on MPS (runs on Mac with MPS)."""
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available - skipping MPS test")

        from sleap_nn.system_info import test_gpu_operations

        success, error = test_gpu_operations()
        assert success is True
        assert error is None

    def test_no_gpu_available(self):
        """Test GPU operations when no GPU available (runs on CPU-only systems)."""
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA available - skipping no-GPU test")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pytest.skip("MPS available - skipping no-GPU test")

        from sleap_nn.system_info import test_gpu_operations

        success, error = test_gpu_operations()
        assert success is False
        assert error == "No GPU available"

    def test_cuda_exception_handling(self):
        """Test that CUDA exceptions are caught and returned as error."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - skipping CUDA exception test")

        from sleap_nn.system_info import test_gpu_operations

        # Mock torch.randn to raise an exception
        with patch("torch.randn", side_effect=RuntimeError("CUDA out of memory")):
            success, error = test_gpu_operations()
            assert success is False
            assert "CUDA out of memory" in error
