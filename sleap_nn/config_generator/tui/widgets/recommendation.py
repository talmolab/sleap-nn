"""Recommendation panel widget.

Displays pipeline and backbone recommendations based on dataset analysis.
"""

from typing import Optional

from textual.widgets import Static

from sleap_nn.config_generator.recommender import ConfigRecommendation


class RecommendationPanel(Static):
    """Widget displaying pipeline recommendation.

    Shows the recommended pipeline type, backbone architecture,
    and any warnings or suggestions based on data analysis.
    """

    DEFAULT_CSS = """
    RecommendationPanel {
        height: auto;
        padding: 1;
        border: solid $secondary;
        margin-bottom: 1;
    }

    RecommendationPanel .rec-title {
        text-style: bold;
    }

    RecommendationPanel .rec-pipeline {
        color: $success;
        text-style: bold;
    }

    RecommendationPanel .rec-backbone {
        color: $primary;
    }

    RecommendationPanel .rec-warning {
        color: $warning;
    }
    """

    def __init__(self, recommendation: Optional[ConfigRecommendation] = None, **kwargs):
        """Initialize with optional recommendation.

        Args:
            recommendation: Initial recommendation to display.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._recommendation = recommendation

    def update_recommendation(self, rec: ConfigRecommendation) -> None:
        """Update the displayed recommendation.

        Args:
            rec: New recommendation to display.
        """
        self._recommendation = rec
        self.refresh()

    @property
    def recommendation(self) -> Optional[ConfigRecommendation]:
        """Get the current recommendation."""
        return self._recommendation

    def render(self) -> str:
        """Render the recommendation panel."""
        if self._recommendation is None:
            return "Recommendation\n" "──────────────\n" "Analyzing..."

        rec = self._recommendation
        lines = [
            "Recommendation",
            "──────────────",
            f"Pipeline: {rec.pipeline.recommended}",
            f"  {rec.pipeline.reason}",
            "",
            f"Backbone: {rec.backbone}",
            f"  {rec.backbone_reason}",
        ]

        if rec.pipeline.alternatives:
            lines.append("")
            lines.append("Alternatives:")
            for alt in rec.pipeline.alternatives[:2]:  # Show top 2 alternatives
                lines.append(f"  • {alt}")

        if rec.pipeline.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in rec.pipeline.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)


class DatasetStatsPanel(Static):
    """Widget displaying dataset statistics.

    Shows key statistics about the loaded SLP file including
    frame count, image dimensions, skeleton info, and instance counts.
    """

    DEFAULT_CSS = """
    DatasetStatsPanel {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    DatasetStatsPanel .stats-title {
        text-style: bold;
    }

    DatasetStatsPanel .stats-value {
        color: $primary;
        text-style: bold;
    }

    DatasetStatsPanel .stats-label {
        color: $text-muted;
    }
    """

    def __init__(self, stats=None, **kwargs):
        """Initialize with optional stats.

        Args:
            stats: DatasetStats object to display.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._stats = stats

    def update_stats(self, stats) -> None:
        """Update the displayed statistics.

        Args:
            stats: New DatasetStats object.
        """
        self._stats = stats
        self.refresh()

    def render(self) -> str:
        """Render the stats panel."""
        if self._stats is None:
            return "Dataset Statistics\n" "──────────────────\n" "Loading..."

        stats = self._stats

        # Format channels
        channels = "Grayscale" if stats.is_grayscale else "RGB"

        # Format instances
        if stats.is_single_instance:
            instances = "Single instance"
        else:
            instances = f"Multi-instance (max {stats.max_instances_per_frame})"

        lines = [
            "Dataset Statistics",
            "──────────────────",
            "",
            f"  Frames:     {stats.num_labeled_frames}",
            f"  Videos:     {stats.num_videos}",
            f"  Size:       {stats.max_width} × {stats.max_height}",
            f"  Channels:   {channels}",
            "",
            f"  Skeleton:   {stats.num_nodes} nodes, {stats.num_edges} edges",
            f"  Instances:  {instances}",
            f"  Avg bbox:   {stats.avg_bbox_size:.0f}px",
        ]

        if stats.has_tracks:
            lines.append(f"  Tracks:     {stats.num_tracks}")

        return "\n".join(lines)


class QuickSettingsPanel(Static):
    """Widget showing quick summary of current settings.

    Provides at-a-glance view of key configuration parameters.
    """

    DEFAULT_CSS = """
    QuickSettingsPanel {
        height: auto;
        padding: 1;
        border: solid $surface-lighten-2;
        margin-bottom: 1;
    }

    QuickSettingsPanel .setting-highlight {
        color: $primary;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the panel."""
        super().__init__(**kwargs)
        self._settings = {}

    def update_settings(
        self,
        pipeline: str = "",
        backbone: str = "",
        batch_size: int = 0,
        input_scale: float = 1.0,
        sigma: float = 5.0,
    ) -> None:
        """Update displayed settings.

        Args:
            pipeline: Pipeline type.
            backbone: Backbone architecture.
            batch_size: Batch size.
            input_scale: Input scaling factor.
            sigma: Confidence map sigma.
        """
        self._settings = {
            "pipeline": pipeline,
            "backbone": backbone,
            "batch_size": batch_size,
            "input_scale": input_scale,
            "sigma": sigma,
        }
        self.refresh()

    def render(self) -> str:
        """Render the quick settings panel."""
        if not self._settings:
            return "Current Settings\n" "────────────────\n" "Not configured"

        s = self._settings
        lines = [
            "Current Settings",
            "────────────────",
            f"  Pipeline:   {s['pipeline'] or '-'}",
            f"  Backbone:   {s['backbone'] or '-'}",
            f"  Batch size: {s['batch_size']}",
            f"  Scale:      {s['input_scale']:.2f}",
            f"  Sigma:      {s['sigma']:.1f}",
        ]

        return "\n".join(lines)
