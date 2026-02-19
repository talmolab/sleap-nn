"""Memory estimation gauge widget.

Displays GPU and CPU memory estimates with color-coded status indicators.
"""

from typing import Optional

from textual.widgets import Static

from sleap_nn.config_generator.memory import MemoryEstimate


class MemoryGauge(Static):
    """Widget displaying memory estimation with color-coded status.

    Shows estimated GPU and CPU memory usage with visual indicators
    for whether the configuration will fit on available hardware.
    """

    DEFAULT_CSS = """
    MemoryGauge {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    MemoryGauge .memory-title {
        text-style: bold;
    }

    MemoryGauge .status-green {
        color: $success;
    }

    MemoryGauge .status-yellow {
        color: $warning;
    }

    MemoryGauge .status-red {
        color: $error;
    }

    MemoryGauge .memory-breakdown {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        estimate: Optional[MemoryEstimate] = None,
        show_breakdown: bool = True,
        **kwargs,
    ):
        """Initialize with optional memory estimate.

        Args:
            estimate: Initial memory estimate to display.
            show_breakdown: Whether to show detailed breakdown.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._estimate = estimate
        self._show_breakdown = show_breakdown

    def update_estimate(self, estimate: MemoryEstimate) -> None:
        """Update the displayed memory estimate.

        Args:
            estimate: New memory estimate to display.
        """
        self._estimate = estimate
        self.refresh()

    def toggle_breakdown(self) -> None:
        """Toggle the detailed breakdown visibility."""
        self._show_breakdown = not self._show_breakdown
        self.refresh()

    @property
    def estimate(self) -> Optional[MemoryEstimate]:
        """Get the current memory estimate."""
        return self._estimate

    def render(self) -> str:
        """Render the memory gauge display."""
        if self._estimate is None:
            return "Memory Estimate\n" "──────────────\n" "Loading..."

        status_icons = {"green": "✓", "yellow": "⚠", "red": "✗"}
        icon = status_icons.get(self._estimate.gpu_status, "?")

        lines = [
            "Memory Estimate",
            "──────────────",
            f"GPU: {self._estimate.total_gpu_gb:.1f} GB {icon}",
            f"  {self._estimate.gpu_message}",
            "",
            f"CPU: {self._estimate.cache_memory_gb:.1f} GB",
            f"  {self._estimate.cpu_message}",
        ]

        if self._show_breakdown and hasattr(self._estimate, "breakdown"):
            lines.append("")
            lines.append("Breakdown:")
            breakdown = self._estimate.breakdown
            if breakdown:
                for key, value in breakdown.items():
                    lines.append(f"  {key}: {value:.1f} MB")

        return "\n".join(lines)


class MemoryBreakdownCard(Static):
    """Detailed memory breakdown card with component-level estimates.

    Shows individual memory consumption for:
    - Model parameters/weights
    - Batch images
    - Feature maps/activations
    - Confidence maps/PAFs
    - Gradients
    """

    DEFAULT_CSS = """
    MemoryBreakdownCard {
        height: auto;
        padding: 1;
        border: solid $surface-lighten-2;
        margin: 1 0;
    }

    MemoryBreakdownCard .card-title {
        text-style: bold;
        margin-bottom: 1;
    }

    MemoryBreakdownCard .memory-row {
        height: 1;
    }

    MemoryBreakdownCard .memory-label {
        width: 1fr;
        color: $text-muted;
    }

    MemoryBreakdownCard .memory-value {
        width: auto;
        text-align: right;
    }
    """

    def __init__(
        self,
        estimate: Optional[MemoryEstimate] = None,
        model_type: str = "Single Instance",
        **kwargs,
    ):
        """Initialize the breakdown card.

        Args:
            estimate: Memory estimate to display.
            model_type: Name of model type for header display.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._estimate = estimate
        self._model_type = model_type

    def update_estimate(
        self, estimate: MemoryEstimate, model_type: Optional[str] = None
    ) -> None:
        """Update the displayed estimate.

        Args:
            estimate: New memory estimate.
            model_type: Optional new model type name.
        """
        self._estimate = estimate
        if model_type:
            self._model_type = model_type
        self.refresh()

    def render(self) -> str:
        """Render the breakdown card."""
        if self._estimate is None:
            return "Memory Breakdown\n" "Calculating..."

        est = self._estimate
        lines = [
            f"GPU Memory ({self._model_type})",
            "─" * 30,
        ]

        # Format values
        params_mb = est.model_memory_gb * 1024
        batch_mb = (est.batch_size * est.image_bytes) / (1024 * 1024)
        activations_mb = est.activations_memory_gb * 1024
        gradients_mb = est.gradients_memory_gb * 1024

        rows = [
            ("Model Params", f"~{self._format_count(params_mb / 4 * 1e6)}"),
            ("Weights", f"~{params_mb:.0f} MB"),
            ("Batch Images", f"~{batch_mb:.0f} MB"),
            ("Feature Maps", f"~{activations_mb:.0f} MB"),
            ("Gradients", f"~{gradients_mb:.0f} MB"),
        ]

        for label, value in rows:
            lines.append(f"  {label:<16} {value:>10}")

        lines.append("")
        lines.append(f"  {'Total':<16} {est.total_gpu_gb:.1f} GB")

        # Status line
        status_tags = {
            "green": ("[green]", "[/green]"),
            "yellow": ("[yellow]", "[/yellow]"),
            "red": ("[red]", "[/red]"),
        }
        open_tag, close_tag = status_tags.get(est.gpu_status, ("", ""))
        lines.append("")
        lines.append(f"{open_tag}{est.gpu_message}{close_tag}")

        return "\n".join(lines)

    def _format_count(self, count: float) -> str:
        """Format parameter count with appropriate suffix."""
        if count >= 1e9:
            return f"{count/1e9:.1f}B"
        elif count >= 1e6:
            return f"{count/1e6:.1f}M"
        elif count >= 1e3:
            return f"{count/1e3:.1f}K"
        return f"{count:.0f}"
