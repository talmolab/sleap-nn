"""Configuration screen for the config generator TUI.

Step 3: Configure training parameters (simplified view with advanced options).
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widget import Widget

from sleap_nn.config_generator.tui.state import ConfigState


class ParameterRow(Horizontal):
    """A row containing a parameter label and input."""

    DEFAULT_CSS = """
    ParameterRow {
        height: auto;
        padding: 0 0 1 0;
    }

    ParameterRow .param-label {
        width: 20;
        padding-right: 1;
    }

    ParameterRow .param-input {
        width: 20;
    }

    ParameterRow .param-hint {
        width: 1fr;
        padding-left: 2;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        label: str,
        param_id: str,
        hint: str = "",
        **kwargs
    ):
        """Initialize parameter row."""
        super().__init__(**kwargs)
        self.param_label = label
        self.param_id = param_id
        self.hint = hint


class ConfigureScreen(Widget):
    """Screen for configuring training parameters."""

    DEFAULT_CSS = """
    ConfigureScreen {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #config-container {
        width: 100%;
        height: 100%;
    }

    #config-scroll {
        width: 100%;
        height: 1fr;
    }

    .section-box {
        background: $panel;
        border: solid $primary;
        padding: 1;
        margin: 0 0 1 0;
    }

    .section-header {
        text-style: bold;
        padding-bottom: 1;
        border-bottom: solid $primary;
        margin-bottom: 1;
    }

    .param-row {
        height: auto;
        padding: 0 0 1 0;
    }

    .param-label {
        width: 22;
        padding-right: 1;
    }

    .param-input {
        width: 15;
    }

    .param-hint {
        width: 1fr;
        padding-left: 2;
        color: $text-muted;
        text-style: italic;
    }

    #memory-estimate {
        background: $panel;
        border: solid $warning;
        padding: 1;
        margin-top: 1;
    }

    .advanced-section {
        margin-top: 1;
    }

    #topdown-tabs {
        height: auto;
        min-height: 20;
    }
    """

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize the configure screen.

        Args:
            state: ConfigState with current configuration.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical(id="config-container"):
            yield Label("[bold]Step 3: Configure Training[/bold]", classes="section-title")

            if self._state and self._state.is_topdown:
                yield Label(
                    "[yellow]Top-down model selected - configuring 2 models[/yellow]",
                    classes="hint"
                )

            with VerticalScroll(id="config-scroll"):
                # For top-down, show tabbed interface
                if self._state and self._state.is_topdown:
                    yield from self._compose_topdown_config()
                else:
                    yield from self._compose_basic_config()

                # Memory estimate
                yield from self._compose_memory_estimate()

    def _compose_basic_config(self) -> ComposeResult:
        """Compose basic configuration sections."""
        # Model section
        with Container(classes="section-box"):
            yield Static("[bold]Model Architecture[/bold]", classes="section-header")
            yield from self._compose_model_params()

        # Data section
        with Container(classes="section-box"):
            yield Static("[bold]Data[/bold]", classes="section-header")
            yield from self._compose_data_params()

        # Training section
        with Container(classes="section-box"):
            yield Static("[bold]Training[/bold]", classes="section-header")
            yield from self._compose_training_params()

        # Advanced section (button to toggle)
        with Container(classes="section-box advanced-section", id="advanced-section"):
            yield Button("Show Advanced Settings", id="toggle-advanced-btn", variant="default")
            with Vertical(id="advanced-content", classes="hidden"):
                yield from self._compose_advanced_params()

    def _compose_topdown_config(self) -> ComposeResult:
        """Compose top-down specific configuration with tabs."""
        with TabbedContent(id="topdown-tabs"):
            with TabPane("Shared Settings", id="shared-tab"):
                yield from self._compose_shared_topdown_params()

            with TabPane("Centroid Model", id="centroid-tab"):
                yield from self._compose_centroid_params()

            with TabPane("Instance Model", id="instance-tab"):
                yield from self._compose_instance_params()

    def _compose_model_params(self) -> ComposeResult:
        """Compose model architecture parameters."""
        # Backbone selection
        with Horizontal(classes="param-row"):
            yield Label("Backbone:", classes="param-label")
            yield Select(
                [
                    ("UNet (recommended)", "unet_medium_rf"),
                    ("UNet Large RF", "unet_large_rf"),
                    ("ConvNeXt Tiny", "convnext_tiny"),
                    ("ConvNeXt Small", "convnext_small"),
                    ("SwinT Tiny", "swint_tiny"),
                    ("SwinT Small", "swint_small"),
                ],
                value=self._state._backbone if self._state else "unet_medium_rf",
                id="backbone-select",
                classes="param-input",
            )
            yield Label("Network architecture", classes="param-hint")

        # Max stride
        with Horizontal(classes="param-row"):
            yield Label("Max Stride:", classes="param-label")
            yield Select(
                [("8", "8"), ("16", "16"), ("32", "32"), ("64", "64")],
                value=str(self._state._max_stride if self._state else 16),
                id="max-stride-select",
                classes="param-input",
            )
            yield Label("Receptive field (larger = more context)", classes="param-hint")

        # Filters
        with Horizontal(classes="param-row"):
            yield Label("Filters:", classes="param-label")
            yield Input(
                value=str(self._state._filters if self._state else 32),
                id="filters-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Base filter count (model capacity)", classes="param-hint")

        # Sigma
        with Horizontal(classes="param-row"):
            yield Label("Sigma:", classes="param-label")
            yield Input(
                value=str(self._state._sigma if self._state else 5.0),
                id="sigma-input",
                classes="param-input",
                type="number",
            )
            yield Label("Confidence map spread (pixels)", classes="param-hint")

    def _compose_data_params(self) -> ComposeResult:
        """Compose data configuration parameters."""
        # Scale
        with Horizontal(classes="param-row"):
            yield Label("Scale:", classes="param-label")
            yield Input(
                value=str(self._state._input_scale if self._state else 1.0),
                id="scale-input",
                classes="param-input",
                type="number",
            )
            yield Label("Image resize factor (0.1-1.0)", classes="param-hint")

        # Augmentation
        with Horizontal(classes="param-row"):
            yield Label("Augmentation:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=self._state._augmentation.enabled if self._state else True,
                id="augmentation-checkbox",
            )
            yield Label("Data augmentation (recommended)", classes="param-hint")

        # Validation fraction
        with Horizontal(classes="param-row"):
            yield Label("Val Fraction:", classes="param-label")
            yield Input(
                value=str(self._state._validation_fraction if self._state else 0.1),
                id="val-fraction-input",
                classes="param-input",
                type="number",
            )
            yield Label("Fraction for validation (0.0-1.0)", classes="param-hint")

    def _compose_training_params(self) -> ComposeResult:
        """Compose training hyperparameters."""
        # Batch size
        with Horizontal(classes="param-row"):
            yield Label("Batch Size:", classes="param-label")
            yield Select(
                [("1", "1"), ("2", "2"), ("4", "4"), ("8", "8"), ("16", "16"), ("32", "32")],
                value=str(self._state._batch_size if self._state else 4),
                id="batch-size-select",
                classes="param-input",
            )
            yield Label("Samples per gradient update", classes="param-hint")

        # Max epochs
        with Horizontal(classes="param-row"):
            yield Label("Max Epochs:", classes="param-label")
            yield Input(
                value=str(self._state._max_epochs if self._state else 200),
                id="max-epochs-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Maximum training epochs", classes="param-hint")

        # Learning rate
        with Horizontal(classes="param-row"):
            yield Label("Learning Rate:", classes="param-label")
            yield Input(
                value=str(self._state._learning_rate if self._state else 1e-4),
                id="lr-input",
                classes="param-input",
                type="number",
            )
            yield Label("Optimizer step size", classes="param-hint")

        # Early stopping
        with Horizontal(classes="param-row"):
            yield Label("Early Stopping:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=self._state._early_stopping if self._state else True,
                id="early-stopping-checkbox",
            )
            yield Label("Stop if validation plateaus", classes="param-hint")

        # Save checkpoints
        with Horizontal(classes="param-row"):
            yield Label("Save Checkpoints:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=True,
                id="save-ckpt-checkbox",
            )
            yield Label("Save model checkpoints", classes="param-hint")

    def _compose_advanced_params(self) -> ComposeResult:
        """Compose advanced parameters section."""
        yield Static("[dim]Advanced settings for fine-tuning[/dim]")

        # Output stride
        with Horizontal(classes="param-row"):
            yield Label("Output Stride:", classes="param-label")
            yield Select(
                [("1", "1"), ("2", "2"), ("4", "4")],
                value=str(self._state._output_stride if self._state else 1),
                id="output-stride-select",
                classes="param-input",
            )
            yield Label("Output resolution (1=full)", classes="param-hint")

        # Filters rate
        with Horizontal(classes="param-row"):
            yield Label("Filters Rate:", classes="param-label")
            yield Input(
                value=str(self._state._filters_rate if self._state else 2.0),
                id="filters-rate-input",
                classes="param-input",
                type="number",
            )
            yield Label("Filter scaling per block", classes="param-hint")

        # Early stopping patience
        with Horizontal(classes="param-row"):
            yield Label("ES Patience:", classes="param-label")
            yield Input(
                value=str(self._state._early_stopping_patience if self._state else 10),
                id="es-patience-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Epochs without improvement", classes="param-hint")

        # Optimizer
        with Horizontal(classes="param-row"):
            yield Label("Optimizer:", classes="param-label")
            yield Select(
                [("Adam", "Adam"), ("AdamW", "AdamW")],
                value=self._state._optimizer if self._state else "Adam",
                id="optimizer-select",
                classes="param-input",
            )
            yield Label("Optimization algorithm", classes="param-hint")

    def _compose_shared_topdown_params(self) -> ComposeResult:
        """Compose shared parameters for top-down models."""
        # Anchor part selection
        with Container(classes="section-box"):
            yield Static("[bold]Anchor Point[/bold]", classes="section-header")

            nodes = self._state.skeleton_nodes if self._state else []
            # Add auto option at the start for when no specific anchor is selected
            options = [("(auto - centroid)", "")] + [(n, n) for n in nodes]

            # Use empty string for None to match the auto option
            current_value = self._state._anchor_part if self._state and self._state._anchor_part else ""

            with Horizontal(classes="param-row"):
                yield Label("Anchor Part:", classes="param-label")
                yield Select(
                    options,
                    value=current_value,
                    id="anchor-part-select",
                    classes="param-input",
                )
                yield Label("Reference keypoint for centering", classes="param-hint")

        # Shared training params
        with Container(classes="section-box"):
            yield Static("[bold]Shared Training Settings[/bold]", classes="section-header")
            yield from self._compose_training_params()

    def _compose_centroid_params(self) -> ComposeResult:
        """Compose centroid model specific parameters."""
        with Container(classes="section-box"):
            yield Static("[bold]Centroid Model (Stage 1)[/bold]", classes="section-header")
            yield Static("[dim]Detects animal centers in full frame[/dim]")

            # Scale for centroid (typically lower)
            with Horizontal(classes="param-row"):
                yield Label("Scale:", classes="param-label")
                yield Input(
                    value="0.5",
                    id="centroid-scale-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Lower scale OK for centroid (0.5 typical)", classes="param-hint")

    def _compose_instance_params(self) -> ComposeResult:
        """Compose centered instance model specific parameters."""
        with Container(classes="section-box"):
            yield Static("[bold]Centered Instance Model (Stage 2)[/bold]", classes="section-header")
            yield Static("[dim]Detects keypoints within cropped regions[/dim]")

            # Crop size
            crop_size = self._state._crop_size if self._state else None
            if crop_size is None and self._state:
                crop_size = self._state._compute_auto_crop_size()

            with Horizontal(classes="param-row"):
                yield Label("Crop Size:", classes="param-label")
                yield Input(
                    value=str(crop_size or 256),
                    id="crop-size-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Square crop size (auto-calculated)", classes="param-hint")

            # Instance scale
            with Horizontal(classes="param-row"):
                yield Label("Scale:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_input_scale if self._state else 1.0),
                    id="instance-scale-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Crop resize factor (1.0 typical)", classes="param-hint")

            # Instance sigma
            with Horizontal(classes="param-row"):
                yield Label("Sigma:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_sigma if self._state else 2.5),
                    id="instance-sigma-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Tighter sigma for crops (2.5 typical)", classes="param-hint")

    def _compose_memory_estimate(self) -> ComposeResult:
        """Compose memory estimate display."""
        if not self._state:
            return

        mem = self._state.memory_estimate()
        # gpu_status is already "green", "yellow", or "red"
        status_color = mem.gpu_status

        with Container(id="memory-estimate"):
            yield Static(
                f"[bold]Memory Estimate:[/bold] [{status_color}]{mem.total_gpu_gb:.1f} GB[/{status_color}]\n"
                f"[dim]{mem.gpu_message}[/dim]"
            )

    # Event handlers to update state
    @on(Select.Changed, "#backbone-select")
    def handle_backbone_change(self, event: Select.Changed) -> None:
        """Handle backbone selection change."""
        if self._state and event.value:
            self._state._backbone = event.value
            # Update max_stride for non-UNet backbones
            if "convnext" in event.value or "swint" in event.value:
                self._state._max_stride = 32
                # Update the select widget
                try:
                    select = self.query_one("#max-stride-select", Select)
                    select.value = "32"
                except Exception:
                    pass

    @on(Select.Changed, "#max-stride-select")
    def handle_stride_change(self, event: Select.Changed) -> None:
        """Handle max stride change."""
        if self._state and event.value:
            self._state._max_stride = int(event.value)

    @on(Select.Changed, "#batch-size-select")
    def handle_batch_change(self, event: Select.Changed) -> None:
        """Handle batch size change."""
        if self._state and event.value:
            self._state._batch_size = int(event.value)

    @on(Input.Changed, "#filters-input")
    def handle_filters_change(self, event: Input.Changed) -> None:
        """Handle filters change."""
        if self._state and event.value:
            try:
                self._state._filters = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#sigma-input")
    def handle_sigma_change(self, event: Input.Changed) -> None:
        """Handle sigma change."""
        if self._state and event.value:
            try:
                self._state._sigma = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scale-input")
    def handle_scale_change(self, event: Input.Changed) -> None:
        """Handle scale change."""
        if self._state and event.value:
            try:
                self._state._input_scale = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#val-fraction-input")
    def handle_val_fraction_change(self, event: Input.Changed) -> None:
        """Handle validation fraction change."""
        if self._state and event.value:
            try:
                self._state._validation_fraction = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#max-epochs-input")
    def handle_epochs_change(self, event: Input.Changed) -> None:
        """Handle max epochs change."""
        if self._state and event.value:
            try:
                self._state._max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#lr-input")
    def handle_lr_change(self, event: Input.Changed) -> None:
        """Handle learning rate change."""
        if self._state and event.value:
            try:
                self._state._learning_rate = float(event.value)
            except ValueError:
                pass

    @on(Checkbox.Changed, "#augmentation-checkbox")
    def handle_augmentation_change(self, event: Checkbox.Changed) -> None:
        """Handle augmentation toggle."""
        if self._state:
            self._state._augmentation.enabled = event.value

    @on(Checkbox.Changed, "#early-stopping-checkbox")
    def handle_early_stopping_change(self, event: Checkbox.Changed) -> None:
        """Handle early stopping toggle."""
        if self._state:
            self._state._early_stopping = event.value

    @on(Select.Changed, "#anchor-part-select")
    def handle_anchor_change(self, event: Select.Changed) -> None:
        """Handle anchor part selection."""
        if self._state:
            # Empty string means auto/centroid (no specific anchor)
            self._state._anchor_part = event.value if event.value else None

    @on(Input.Changed, "#crop-size-input")
    def handle_crop_size_change(self, event: Input.Changed) -> None:
        """Handle crop size change."""
        if self._state and event.value:
            try:
                self._state._crop_size = int(event.value)
            except ValueError:
                pass

    @on(Button.Pressed, "#toggle-advanced-btn")
    def handle_toggle_advanced(self, event: Button.Pressed) -> None:
        """Handle advanced settings toggle."""
        try:
            btn = self.query_one("#toggle-advanced-btn", Button)
            content = self.query_one("#advanced-content", Vertical)

            if content.has_class("hidden"):
                content.remove_class("hidden")
                btn.label = "Hide Advanced Settings"
            else:
                content.add_class("hidden")
                btn.label = "Show Advanced Settings"
        except Exception:
            pass
