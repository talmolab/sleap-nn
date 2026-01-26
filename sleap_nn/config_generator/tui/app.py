"""Main TUI application for config generation.

This module provides the main Textual application for interactively
generating sleap-nn training configurations.
"""

from pathlib import Path
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    ProgressBar,
    RadioButton,
    RadioSet,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType, analyze_slp
from sleap_nn.config_generator.generator import ConfigGenerator
from sleap_nn.config_generator.memory import MemoryEstimate, estimate_memory
from sleap_nn.config_generator.recommender import (
    ConfigRecommendation,
    PipelineType,
    recommend_config,
)


class MemoryGauge(Static):
    """Widget displaying memory estimation with color-coded status."""

    DEFAULT_CSS = """
    MemoryGauge {
        height: auto;
        padding: 1;
        border: solid $primary;
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
    """

    def __init__(self, estimate: Optional[MemoryEstimate] = None, **kwargs):
        """Initialize with optional memory estimate."""
        super().__init__(**kwargs)
        self._estimate = estimate

    def update_estimate(self, estimate: MemoryEstimate) -> None:
        """Update the displayed memory estimate."""
        self._estimate = estimate
        self.refresh()

    def render(self) -> str:
        """Render the memory gauge."""
        if self._estimate is None:
            return "Memory Estimate\n──────────────\nLoading..."

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
        return "\n".join(lines)


class RecommendationPanel(Static):
    """Widget displaying pipeline recommendation."""

    DEFAULT_CSS = """
    RecommendationPanel {
        height: auto;
        padding: 1;
        border: solid $secondary;
    }

    RecommendationPanel .rec-title {
        text-style: bold;
    }
    """

    def __init__(self, recommendation: Optional[ConfigRecommendation] = None, **kwargs):
        """Initialize with optional recommendation."""
        super().__init__(**kwargs)
        self._recommendation = recommendation

    def update_recommendation(self, rec: ConfigRecommendation) -> None:
        """Update the displayed recommendation."""
        self._recommendation = rec
        self.refresh()

    def render(self) -> str:
        """Render the recommendation panel."""
        if self._recommendation is None:
            return "Recommendation\n──────────────\nAnalyzing..."

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

        if rec.pipeline.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in rec.pipeline.warnings:
                lines.append(f"  * {w}")

        return "\n".join(lines)


class ConfigGeneratorApp(App):
    """Interactive TUI for generating sleap-nn training configurations."""

    TITLE = "SLEAP-NN Config Generator"

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 2fr 1fr;
    }

    #main-content {
        column-span: 1;
        height: 100%;
    }

    #sidebar {
        column-span: 1;
        height: 100%;
        padding: 1;
    }

    .section-title {
        text-style: bold;
        padding: 1 0;
    }

    .form-group {
        height: auto;
        padding: 1 0;
    }

    .form-label {
        padding: 0 1 0 0;
    }

    #stats-display {
        height: auto;
        padding: 1;
        border: solid $primary;
    }

    #yaml-preview {
        height: 100%;
    }

    .button-row {
        height: auto;
        padding: 1;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "save", "Save Config"),
        Binding("a", "auto_fill", "Auto-Fill"),
        Binding("tab", "next_tab", "Next Tab", show=False),
    ]

    def __init__(self, slp_path: str, **kwargs):
        """Initialize the config generator app.

        Args:
            slp_path: Path to the SLP file to analyze.
        """
        super().__init__(**kwargs)
        self.slp_path = Path(slp_path)
        self.generator = ConfigGenerator.from_slp(str(slp_path))
        self._stats: Optional[DatasetStats] = None
        self._recommendation: Optional[ConfigRecommendation] = None
        self._view_type: ViewType = ViewType.UNKNOWN

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        with Horizontal():
            with Container(id="main-content"):
                with TabbedContent(id="tabs"):
                    with TabPane("Data", id="data-tab"):
                        yield from self._compose_data_tab()

                    with TabPane("Model", id="model-tab"):
                        yield from self._compose_model_tab()

                    with TabPane("Training", id="training-tab"):
                        yield from self._compose_training_tab()

                    with TabPane("Export", id="export-tab"):
                        yield from self._compose_export_tab()

            with Vertical(id="sidebar"):
                yield MemoryGauge(id="memory-gauge")
                yield RecommendationPanel(id="recommendation-panel")

        yield Footer()

    def _compose_data_tab(self) -> ComposeResult:
        """Compose the Data tab content."""
        with VerticalScroll():
            yield Static("Dataset Analysis", classes="section-title")
            yield Static("Loading...", id="stats-display")

            yield Rule()

            yield Static("Camera View", classes="section-title")
            yield Static(
                "Select the camera view orientation for augmentation defaults:",
                classes="form-label",
            )
            with RadioSet(id="view-type"):
                yield RadioButton("Side View (±15° rotation)", id="view-side")
                yield RadioButton("Top View (±180° rotation)", id="view-top")
                yield RadioButton(
                    "Unknown (conservative defaults)", id="view-unknown", value=True
                )

            yield Rule()

            with Horizontal(classes="button-row"):
                yield Button("Auto-Fill All", id="auto-fill-btn", variant="primary")

    def _compose_model_tab(self) -> ComposeResult:
        """Compose the Model tab content."""
        with VerticalScroll():
            yield Static("Pipeline Type", classes="section-title")
            with RadioSet(id="pipeline-select"):
                yield RadioButton("Single Instance", id="pipe-single")
                yield RadioButton("Top-Down: Centroid", id="pipe-centroid")
                yield RadioButton("Top-Down: Centered Instance", id="pipe-centered")
                yield RadioButton("Bottom-Up", id="pipe-bottomup")
                yield RadioButton("Multi-Class Bottom-Up", id="pipe-mc-bottomup")
                yield RadioButton("Multi-Class Top-Down", id="pipe-mc-topdown")

            yield Rule()

            yield Static("Backbone Architecture", classes="section-title")
            with RadioSet(id="backbone-select"):
                yield RadioButton("UNet (Medium RF)", id="bb-unet-medium", value=True)
                yield RadioButton("UNet (Large RF)", id="bb-unet-large")
                yield RadioButton("ConvNeXt (Tiny)", id="bb-convnext-tiny")
                yield RadioButton("ConvNeXt (Small)", id="bb-convnext-small")
                yield RadioButton("SwinT (Tiny)", id="bb-swint-tiny")
                yield RadioButton("SwinT (Small)", id="bb-swint-small")

            yield Rule()

            yield Static("Confidence Map Settings", classes="section-title")

            with Horizontal(classes="form-group"):
                yield Static("Sigma:", classes="form-label")
                yield Input(value="5.0", id="sigma-input", type="number")

            with Horizontal(classes="form-group"):
                yield Static("Output Stride:", classes="form-label")
                yield Select(
                    [(str(s), s) for s in [1, 2, 4, 8]],
                    value=1,
                    id="output-stride-select",
                )

    def _compose_training_tab(self) -> ComposeResult:
        """Compose the Training tab content."""
        with VerticalScroll():
            yield Static("Training Parameters", classes="section-title")

            with Horizontal(classes="form-group"):
                yield Static("Batch Size:", classes="form-label")
                yield Input(value="4", id="batch-size-input", type="integer")

            with Horizontal(classes="form-group"):
                yield Static("Max Epochs:", classes="form-label")
                yield Input(value="200", id="max-epochs-input", type="integer")

            with Horizontal(classes="form-group"):
                yield Static("Learning Rate:", classes="form-label")
                yield Input(value="0.0001", id="learning-rate-input", type="number")

            with Horizontal(classes="form-group"):
                yield Static("Validation Fraction:", classes="form-label")
                yield Input(value="0.1", id="val-fraction-input", type="number")

            yield Rule()

            yield Static("Data Augmentation", classes="section-title")

            with Horizontal(classes="form-group"):
                yield Static("Enable Augmentation:", classes="form-label")
                yield Switch(value=True, id="augmentation-switch")

            with Horizontal(classes="form-group"):
                yield Static("Rotation Min (deg):", classes="form-label")
                yield Input(value="-15.0", id="rotation-min-input", type="number")

            with Horizontal(classes="form-group"):
                yield Static("Rotation Max (deg):", classes="form-label")
                yield Input(value="15.0", id="rotation-max-input", type="number")

            with Horizontal(classes="form-group"):
                yield Static("Scale Min:", classes="form-label")
                yield Input(value="0.9", id="scale-min-input", type="number")

            with Horizontal(classes="form-group"):
                yield Static("Scale Max:", classes="form-label")
                yield Input(value="1.1", id="scale-max-input", type="number")

            yield Rule()

            yield Static("Early Stopping", classes="section-title")

            with Horizontal(classes="form-group"):
                yield Static("Enable Early Stopping:", classes="form-label")
                yield Switch(value=True, id="early-stopping-switch")

            with Horizontal(classes="form-group"):
                yield Static("Patience:", classes="form-label")
                yield Input(value="10", id="patience-input", type="integer")

    def _compose_export_tab(self) -> ComposeResult:
        """Compose the Export tab content."""
        with VerticalScroll():
            yield Static("YAML Configuration Preview", classes="section-title")
            yield TextArea(id="yaml-preview", read_only=True, language="yaml")

            yield Rule()

            with Horizontal(classes="button-row"):
                yield Button("Refresh Preview", id="refresh-btn")
                yield Button("Save to File", id="save-btn", variant="primary")
                yield Button("Copy to Clipboard", id="copy-btn")

            yield Static("", id="save-status")

    async def on_mount(self) -> None:
        """Handle app mount - load data."""
        # Analyze the SLP file
        self._stats = analyze_slp(str(self.slp_path))
        self._recommendation = recommend_config(self._stats, self._view_type)

        # Update stats display
        stats_display = self.query_one("#stats-display", Static)
        stats_display.update(str(self._stats))

        # Update memory gauge
        mem = estimate_memory(self._stats)
        self.query_one("#memory-gauge", MemoryGauge).update_estimate(mem)

        # Update recommendation panel
        self.query_one(
            "#recommendation-panel", RecommendationPanel
        ).update_recommendation(self._recommendation)

        # Initial YAML preview
        self._update_yaml_preview()

    def _get_selected_pipeline(self) -> PipelineType:
        """Get the currently selected pipeline type."""
        pipeline_map = {
            "pipe-single": "single_instance",
            "pipe-centroid": "centroid",
            "pipe-centered": "centered_instance",
            "pipe-bottomup": "bottomup",
            "pipe-mc-bottomup": "multi_class_bottomup",
            "pipe-mc-topdown": "multi_class_topdown",
        }

        radio_set = self.query_one("#pipeline-select", RadioSet)
        if radio_set.pressed_button:
            return pipeline_map.get(radio_set.pressed_button.id, "single_instance")
        return "single_instance"

    def _get_selected_backbone(self) -> str:
        """Get the currently selected backbone."""
        backbone_map = {
            "bb-unet-medium": "unet_medium_rf",
            "bb-unet-large": "unet_large_rf",
            "bb-convnext-tiny": "convnext_tiny",
            "bb-convnext-small": "convnext_small",
            "bb-swint-tiny": "swint_tiny",
            "bb-swint-small": "swint_small",
        }

        radio_set = self.query_one("#backbone-select", RadioSet)
        if radio_set.pressed_button:
            return backbone_map.get(radio_set.pressed_button.id, "unet_medium_rf")
        return "unet_medium_rf"

    def _update_generator_from_ui(self) -> None:
        """Update the generator from current UI values."""
        # Pipeline
        self.generator._pipeline = self._get_selected_pipeline()

        # Backbone
        self.generator.backbone(self._get_selected_backbone())

        # Sigma
        try:
            sigma = float(self.query_one("#sigma-input", Input).value)
            self.generator._sigma = sigma
        except ValueError:
            pass

        # Output stride
        output_stride = self.query_one("#output-stride-select", Select).value
        if output_stride:
            self.generator._output_stride = int(output_stride)

        # Batch size
        try:
            batch_size = int(self.query_one("#batch-size-input", Input).value)
            self.generator._batch_size = batch_size
        except ValueError:
            pass

        # Max epochs
        try:
            max_epochs = int(self.query_one("#max-epochs-input", Input).value)
            self.generator._max_epochs = max_epochs
        except ValueError:
            pass

        # Learning rate
        try:
            lr = float(self.query_one("#learning-rate-input", Input).value)
            self.generator._learning_rate = lr
        except ValueError:
            pass

        # Validation fraction
        try:
            val_frac = float(self.query_one("#val-fraction-input", Input).value)
            self.generator._validation_fraction = val_frac
        except ValueError:
            pass

        # Augmentation
        self.generator._use_augmentations = self.query_one(
            "#augmentation-switch", Switch
        ).value

        # Rotation range
        try:
            rot_min = float(self.query_one("#rotation-min-input", Input).value)
            rot_max = float(self.query_one("#rotation-max-input", Input).value)
            self.generator._rotation_range = (rot_min, rot_max)
        except ValueError:
            pass

        # Scale range
        try:
            scale_min = float(self.query_one("#scale-min-input", Input).value)
            scale_max = float(self.query_one("#scale-max-input", Input).value)
            self.generator._scale_range = (scale_min, scale_max)
        except ValueError:
            pass

        # Early stopping
        self.generator._early_stopping = self.query_one(
            "#early-stopping-switch", Switch
        ).value

        try:
            patience = int(self.query_one("#patience-input", Input).value)
            self.generator._early_stopping_patience = patience
        except ValueError:
            pass

    def _update_yaml_preview(self) -> None:
        """Update the YAML preview."""
        self._update_generator_from_ui()

        try:
            yaml_str = self.generator.to_yaml()
            self.query_one("#yaml-preview", TextArea).text = yaml_str
        except Exception as e:
            self.query_one("#yaml-preview", TextArea).text = f"Error: {e}"

    def _update_ui_from_generator(self) -> None:
        """Update UI widgets from generator state."""
        # Pipeline
        pipeline_map = {
            "single_instance": "pipe-single",
            "centroid": "pipe-centroid",
            "centered_instance": "pipe-centered",
            "bottomup": "pipe-bottomup",
            "multi_class_bottomup": "pipe-mc-bottomup",
            "multi_class_topdown": "pipe-mc-topdown",
        }
        if self.generator._pipeline:
            btn_id = pipeline_map.get(self.generator._pipeline)
            if btn_id:
                try:
                    self.query_one(f"#{btn_id}", RadioButton).value = True
                except Exception:
                    pass

        # Backbone
        backbone_map = {
            "unet_medium_rf": "bb-unet-medium",
            "unet_large_rf": "bb-unet-large",
            "convnext_tiny": "bb-convnext-tiny",
            "convnext_small": "bb-convnext-small",
            "swint_tiny": "bb-swint-tiny",
            "swint_small": "bb-swint-small",
        }
        btn_id = backbone_map.get(self.generator._backbone)
        if btn_id:
            try:
                self.query_one(f"#{btn_id}", RadioButton).value = True
            except Exception:
                pass

        # Sigma
        self.query_one("#sigma-input", Input).value = str(self.generator._sigma)

        # Output stride
        self.query_one("#output-stride-select", Select).value = (
            self.generator._output_stride
        )

        # Batch size
        self.query_one("#batch-size-input", Input).value = str(
            self.generator._batch_size
        )

        # Max epochs
        self.query_one("#max-epochs-input", Input).value = str(
            self.generator._max_epochs
        )

        # Learning rate
        self.query_one("#learning-rate-input", Input).value = str(
            self.generator._learning_rate
        )

        # Validation fraction
        self.query_one("#val-fraction-input", Input).value = str(
            self.generator._validation_fraction
        )

        # Augmentation
        self.query_one("#augmentation-switch", Switch).value = (
            self.generator._use_augmentations
        )

        # Rotation range
        self.query_one("#rotation-min-input", Input).value = str(
            self.generator._rotation_range[0]
        )
        self.query_one("#rotation-max-input", Input).value = str(
            self.generator._rotation_range[1]
        )

        # Scale range
        self.query_one("#scale-min-input", Input).value = str(
            self.generator._scale_range[0]
        )
        self.query_one("#scale-max-input", Input).value = str(
            self.generator._scale_range[1]
        )

        # Early stopping
        self.query_one("#early-stopping-switch", Switch).value = (
            self.generator._early_stopping
        )
        self.query_one("#patience-input", Input).value = str(
            self.generator._early_stopping_patience
        )

    def _update_memory_estimate(self) -> None:
        """Update memory estimate based on current settings."""
        if self._stats is None:
            return

        self._update_generator_from_ui()

        mem = estimate_memory(
            self._stats,
            self.generator._backbone,
            self.generator._batch_size,
            self.generator._input_scale,
            self.generator._output_stride,
        )
        self.query_one("#memory-gauge", MemoryGauge).update_estimate(mem)

    @on(Button.Pressed, "#auto-fill-btn")
    def handle_auto_fill(self) -> None:
        """Handle auto-fill button press."""
        # Get view type from radio set
        view_radio = self.query_one("#view-type", RadioSet)
        if view_radio.pressed_button:
            view_map = {
                "view-side": "side",
                "view-top": "top",
                "view-unknown": None,
            }
            view = view_map.get(view_radio.pressed_button.id)
        else:
            view = None

        # Apply auto configuration
        self.generator.auto(view=view)

        # Update UI from generator
        self._update_ui_from_generator()

        # Update preview
        self._update_yaml_preview()

        # Update memory estimate
        self._update_memory_estimate()

    @on(Button.Pressed, "#refresh-btn")
    def handle_refresh(self) -> None:
        """Handle refresh preview button press."""
        self._update_yaml_preview()
        self._update_memory_estimate()

    @on(Button.Pressed, "#save-btn")
    def handle_save(self) -> None:
        """Handle save button press."""
        self._update_generator_from_ui()

        # Generate output filename
        output_path = self.slp_path.parent / f"{self.slp_path.stem}_config.yaml"

        try:
            self.generator.save(str(output_path))
            status = self.query_one("#save-status", Static)
            status.update(f"Saved to: {output_path}")
        except Exception as e:
            status = self.query_one("#save-status", Static)
            status.update(f"Error saving: {e}")

    @on(Button.Pressed, "#copy-btn")
    def handle_copy(self) -> None:
        """Handle copy to clipboard button press."""
        self._update_generator_from_ui()

        try:
            yaml_str = self.generator.to_yaml()
            # Use pyperclip if available, otherwise show message
            try:
                import pyperclip

                pyperclip.copy(yaml_str)
                status = self.query_one("#save-status", Static)
                status.update("Copied to clipboard!")
            except ImportError:
                status = self.query_one("#save-status", Static)
                status.update("Install pyperclip for clipboard support")
        except Exception as e:
            status = self.query_one("#save-status", Static)
            status.update(f"Error: {e}")

    @on(RadioSet.Changed)
    def handle_radio_change(self, event: RadioSet.Changed) -> None:
        """Handle radio button changes."""
        # Update view type if view radio changed
        if event.radio_set.id == "view-type":
            view_map = {
                "view-side": ViewType.SIDE,
                "view-top": ViewType.TOP,
                "view-unknown": ViewType.UNKNOWN,
            }
            if event.pressed.id in view_map:
                self._view_type = view_map[event.pressed.id]

                # Update rotation defaults
                if self._view_type == ViewType.TOP:
                    self.query_one("#rotation-min-input", Input).value = "-180.0"
                    self.query_one("#rotation-max-input", Input).value = "180.0"
                elif self._view_type == ViewType.SIDE:
                    self.query_one("#rotation-min-input", Input).value = "-15.0"
                    self.query_one("#rotation-max-input", Input).value = "15.0"

        # Update memory estimate when backbone changes
        if event.radio_set.id == "backbone-select":
            self._update_memory_estimate()

    @on(Input.Changed)
    def handle_input_change(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        # Update memory estimate for relevant fields
        if event.input.id in ["batch-size-input", "sigma-input"]:
            self._update_memory_estimate()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_save(self) -> None:
        """Save the configuration."""
        self.handle_save()

    def action_auto_fill(self) -> None:
        """Auto-fill configuration."""
        self.handle_auto_fill()


def launch_tui(slp_path: str) -> None:
    """Launch the TUI configuration generator.

    Args:
        slp_path: Path to the SLP file to configure.
    """
    app = ConfigGeneratorApp(slp_path)
    app.run()
