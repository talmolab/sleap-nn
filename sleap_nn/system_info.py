"""System diagnostics and compatibility checking for sleap-nn."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from typing import Optional

# Key packages to check versions for
PACKAGES = [
    "sleap-nn",
    "sleap-io",
    "torch",
    "pytorch-lightning",
    "kornia",
    "wandb",
    "numpy",
    "h5py",
]

# CUDA version -> (min_driver_linux, min_driver_windows)
# Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
CUDA_DRIVER_REQUIREMENTS = {
    "12.6": ("560.28.03", "560.76"),
    "12.8": ("570.26", "570.65"),
    "13.0": ("580.65.06", "580.00"),
}


def parse_driver_version(version: str) -> tuple[int, ...]:
    """Parse driver version string into comparable tuple."""
    try:
        return tuple(int(x) for x in version.split("."))
    except ValueError:
        return (0,)


def get_min_driver_for_cuda(cuda_version: str) -> Optional[tuple[str, str]]:
    """Get minimum driver versions for a CUDA version.

    Args:
        cuda_version: CUDA version string (e.g., "12.6" or "12.6.1")

    Returns:
        Tuple of (min_linux, min_windows) or None if unknown version.
    """
    if not cuda_version:
        return None
    # Match major.minor (e.g., "12.6" from "12.6.1")
    parts = cuda_version.split(".")
    if len(parts) >= 2:
        major_minor = f"{parts[0]}.{parts[1]}"
        return CUDA_DRIVER_REQUIREMENTS.get(major_minor)
    return None


def check_driver_compatibility(
    driver_version: str, cuda_version: str
) -> tuple[bool, Optional[str]]:
    """Check if driver version is compatible with CUDA version.

    Args:
        driver_version: Installed driver version string
        cuda_version: CUDA version from PyTorch

    Returns:
        Tuple of (is_compatible, min_required_version).
        If CUDA version is unknown, returns (True, None).
    """
    min_versions = get_min_driver_for_cuda(cuda_version)
    if not min_versions:
        return True, None  # Unknown CUDA version, skip check

    if sys.platform == "win32":
        min_version = min_versions[1]
    else:
        min_version = min_versions[0]

    current = parse_driver_version(driver_version)
    required = parse_driver_version(min_version)

    # Pad tuples to same length for comparison
    max_len = max(len(current), len(required))
    current = current + (0,) * (max_len - len(current))
    required = required + (0,) * (max_len - len(required))

    return current >= required, min_version


def get_nvidia_driver_version() -> Optional[str]:
    """Get NVIDIA driver version from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _get_package_location(name: str, dist) -> str:
    """Get the actual installed location of a package.

    For editable installs, returns the source directory.
    For regular installs, returns the site-packages path.
    """
    # Try to import the package and get its __file__
    try:
        # Convert package name to module name (e.g., "sleap-nn" -> "sleap_nn")
        module_name = name.replace("-", "_")
        module = __import__(module_name)
        if hasattr(module, "__file__") and module.__file__:
            # Return parent of the package's __init__.py
            from pathlib import Path

            return str(Path(module.__file__).parent.parent)
    except (ImportError, AttributeError):
        pass

    # Fallback to dist._path
    if dist._path:
        path = dist._path.parent
        # Make absolute if relative
        if not path.is_absolute():
            from pathlib import Path

            path = Path.cwd() / path
        return str(path)

    return ""


def get_package_info(name: str) -> dict:
    """Get package version, location, and install source.

    Args:
        name: Package name (e.g., "sleap-nn")

    Returns:
        Dict with version, location, source, and editable fields.
    """
    try:
        dist = importlib.metadata.distribution(name)
        version = dist.version

        # Check for editable install and source via direct_url.json
        is_editable = False
        source = "pip"  # Default assumption
        try:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct_url = json.loads(direct_url_text)
                is_editable = direct_url.get("dir_info", {}).get("editable", False)
                if is_editable:
                    source = "editable"
                elif "vcs_info" in direct_url:
                    source = "git"
                elif direct_url.get("url", "").startswith("file://"):
                    source = "local"
        except FileNotFoundError:
            pass

        # Fallback: detect old-style editable installs (.egg-info not in site-packages)
        if not is_editable and dist._path:
            path_str = str(dist._path)
            # Old-style editable: .egg-info in source dir, not site-packages
            if ".egg-info" in path_str and "site-packages" not in path_str:
                is_editable = True
                source = "editable"

        # Check for conda install via INSTALLER file (only if not already known)
        if source == "pip":
            try:
                installer = dist.read_text("INSTALLER")
                if installer and installer.strip() == "conda":
                    source = "conda"
            except FileNotFoundError:
                pass

        # Get location (after determining if editable, so we can use the right method)
        location = _get_package_location(name, dist)

        return {
            "version": version,
            "location": location,
            "source": source,
            "editable": is_editable,
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "version": "not installed",
            "location": "",
            "source": "",
            "editable": False,
        }


def get_system_info_dict() -> dict:
    """Get system information as a dictionary.

    Returns:
        Dictionary with system info including Python version, platform,
        PyTorch version, CUDA availability, GPU details, and package versions.
    """
    import torch

    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "cudnn_version": None,
        "driver_version": None,
        "driver_compatible": None,
        "driver_min_required": None,
        "gpu_count": 0,
        "gpus": [],
        "mps_available": False,
        "accelerator": "cpu",  # cpu, cuda, or mps
        "packages": {},
    }

    # Driver version (check even if CUDA unavailable - old driver can cause this)
    driver = get_nvidia_driver_version()
    if driver:
        info["driver_version"] = driver

    # CUDA details
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
        info["gpu_count"] = torch.cuda.device_count()
        info["accelerator"] = "cuda"

        # Check driver compatibility
        if driver and info["cuda_version"]:
            is_compatible, min_required = check_driver_compatibility(
                driver, info["cuda_version"]
            )
            info["driver_compatible"] = is_compatible
            info["driver_min_required"] = min_required

        # GPU details
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "id": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_gb": round(props.total_memory / (1024**3), 1),
                }
            )

    # MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["accelerator"] = "mps"
        info["gpu_count"] = 1

    # Package versions
    for pkg in PACKAGES:
        info["packages"][pkg] = get_package_info(pkg)

    return info


def test_gpu_operations() -> tuple[bool, Optional[str]]:
    """Test that GPU tensor operations work.

    Returns:
        Tuple of (success, error_message).
    """
    import torch

    if torch.cuda.is_available():
        try:
            x = torch.randn(100, 100, device="cuda")
            _ = torch.mm(x, x)
            return True, None
        except Exception as e:
            return False, str(e)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            x = torch.randn(100, 100, device="mps")
            _ = torch.mm(x, x)
            return True, None
        except Exception as e:
            return False, str(e)
    return False, "No GPU available"


def _shorten_path(path: str, max_len: int = 40) -> str:
    """Shorten a path for display, keeping the end."""
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3) :]


def print_system_info() -> None:
    """Print comprehensive system diagnostics to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    info = get_system_info_dict()

    # System info table (with GPU details integrated)
    table = Table(title="System Information", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Python", info["python_version"])
    table.add_row("Platform", info["platform"])
    table.add_row("PyTorch", info["pytorch_version"])

    # GPU/Accelerator info
    if info["accelerator"] == "cuda":
        table.add_row("Accelerator", "CUDA")
        table.add_row("CUDA version", info["cuda_version"] or "N/A")
        table.add_row("cuDNN version", info["cudnn_version"] or "N/A")
        table.add_row("Driver version", info["driver_version"] or "N/A")
        table.add_row("GPU count", str(info["gpu_count"]))
        # GPU details inline
        for gpu in info["gpus"]:
            gpu_str = f"{gpu['name']} ({gpu['memory_gb']} GB, compute {gpu['compute_capability']})"
            table.add_row(f"GPU {gpu['id']}", gpu_str)
    elif info["accelerator"] == "mps":
        table.add_row("Accelerator", "MPS (Apple Silicon)")
    else:
        table.add_row("Accelerator", "CPU only")
        # Show driver if present but CUDA unavailable (helps diagnose issues)
        if info["driver_version"]:
            table.add_row("Driver version", info["driver_version"])

    console.print(table)

    # Package versions table
    console.print()
    pkg_table = Table(title="Package Versions")
    pkg_table.add_column("Package", style="cyan")
    pkg_table.add_column("Version", style="white")
    pkg_table.add_column("Source", style="yellow")
    pkg_table.add_column("Location", style="dim")

    for pkg, pkg_info in info["packages"].items():
        if pkg_info["version"] == "not installed":
            version_display = f"[dim]{pkg_info['version']}[/dim]"
            pkg_table.add_row(pkg, version_display, "", "")
        else:
            location_display = _shorten_path(pkg_info["location"])
            pkg_table.add_row(
                pkg, pkg_info["version"], pkg_info["source"], location_display
            )

    console.print(pkg_table)

    # Actionable diagnostics
    console.print()

    # Driver compatibility check (CUDA only)
    if info["accelerator"] == "cuda" and info["driver_version"]:
        if info["driver_min_required"]:
            if info["driver_compatible"]:
                console.print(
                    f"[green]OK[/green] Driver is compatible: "
                    f"{info['driver_version']} >= {info['driver_min_required']} "
                    f"(required for CUDA {info['cuda_version']})"
                )
            else:
                console.print(
                    f"[red]FAIL[/red] Driver is too old: "
                    f"{info['driver_version']} < {info['driver_min_required']} "
                    f"(required for CUDA {info['cuda_version']})"
                )
                console.print(
                    "  [yellow]Update your driver: https://www.nvidia.com/drivers[/yellow]"
                )
        else:
            # Unknown CUDA version, can't check compatibility
            console.print(
                f"[yellow]![/yellow] Driver version: {info['driver_version']} "
                f"(CUDA {info['cuda_version']} compatibility unknown)"
            )
    elif info["accelerator"] == "cpu" and info["driver_version"]:
        # Has driver but no CUDA - might be a problem
        console.print(
            f"[yellow]![/yellow] NVIDIA driver found ({info['driver_version']}) "
            "but CUDA is not available"
        )
        console.print(
            "  [dim]This may indicate a driver/PyTorch version mismatch[/dim]"
        )

    # GPU connection test
    success, error = test_gpu_operations()
    if info["accelerator"] == "cuda":
        if success:
            console.print("[green]OK[/green] PyTorch can use GPU")
        else:
            console.print(f"[red]FAIL[/red] PyTorch cannot use GPU: {error}")
    elif info["accelerator"] == "mps":
        if success:
            console.print("[green]OK[/green] PyTorch can use GPU")
        else:
            console.print(f"[red]FAIL[/red] PyTorch cannot use GPU: {error}")
    else:
        console.print("[dim]--[/dim] No GPU available (using CPU)")


def get_startup_info_string() -> str:
    """Get a concise system info string for startup logging.

    Returns:
        Single-line string with key system info.
    """
    import torch

    from sleap_nn import __version__

    parts = [f"sleap-nn {__version__}"]
    parts.append(f"Python {sys.version.split()[0]}")
    parts.append(f"PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        parts.append(f"CUDA {torch.version.cuda}")
        parts.append(f"{torch.cuda.device_count()} GPU(s)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        parts.append("MPS")
    else:
        parts.append("CPU only")

    return " | ".join(parts)
