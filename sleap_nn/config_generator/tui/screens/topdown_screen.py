"""Top-down dual configuration screen.

Provides UI for configuring both centroid and centered instance models
for top-down multi-instance pose estimation pipelines.
"""

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Input,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from sleap_nn.config_generator.tui.widgets import (
    InfoBox,
    LabeledSlider,
    RangeSlider,
    SigmaVisualization,
    TipBox,
)

if TYPE_CHECKING:
    from sleap_nn.config_generator.tui.state import ConfigState


class TopDownScreen(Container):
    """Top-down dual configuration screen component.

    Provides tabbed interface for configuring:
    - Centroid model (detects instance centers)
    - Centered instance model (detects keypoints in crops)

    Each model has independent settings for:
    - Backbone architecture
    - Head configuration
    - Training parameters
    - Augmentation
    """

    DEFAULT_CSS = """
    TopDownScreen {
        width: 100%;
        height: 100%;
        padding: 0;
    }

    #topdown-tabs {
        width: 100%;
        height: 1fr;
    }

    #topdown-tabs ContentSwitcher {
        width: 100%;
        height: 1fr;
    }

    #centroid-config-tab, #ci-config-tab {
        width: 100%;
        height: 100%;
    }

    #centroid-scroll, #ci-scroll {
        width: 100%;
        height: 100%;
    }

    .section-title {
        text-style: bold;
        padding: 1 0;
    }

    .subsection-title {
        text-style: bold;
        padding: 0 0 1 0;
        color: $text-muted;
    }

    .form-row {
        height: auto;
        padding: 0 0 1 0;
    }

    .form-label {
        width: 20;
        padding-right: 1;
    }

    .step-indicator {
        height: auto;
        padding: 1;
        background: $surface-lighten-1;
        text-align: center;
        border: solid $primary;
        margin-bottom: 1;
    }

    .step-number {
        color: $primary;
        text-style: bold;
    }
    """

    def __init__(self, state: "ConfigState", **kwargs):
        """Initialize the top-down screen.

        Args:
            state: Shared configuration state.
            **kwargs: Additional arguments for Container.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the top-down screen layout."""
        yield InfoBox(
            "Top-down pose estimation uses two models:\n"
            "1. Centroid model: Detects the center of each animal\n"
            "2. Centered Instance model: Detects keypoints within cropped regions\n\n"
            "Configure each model separately in the tabs below.",
        )

        with TabbedContent(id="topdown-tabs"):
            with TabPane("1. Centroid", id="centroid-config-tab"):
                yield from self._compose_centroid_tab()

            with TabPane("2. Centered Instance", id="ci-config-tab"):
                yield from self._compose_centered_instance_tab()

    def _compose_centroid_tab(self) -> ComposeResult:
        """Compose centroid model configuration tab."""
        with VerticalScroll(id="centroid-scroll"):
            yield Static(
                "[bold]Step 1:[/bold] Centroid Model",
                classes="step-indicator",
            )

            yield TipBox(
                "The centroid model detects the center point of each animal in the full image. "
                "This is typically a fast, low-resolution model.",
                title=None,
            )

            # Anchor Point Selection
            yield Static("Anchor Point", classes="section-title")

            node_options = [("Auto (centroid)", None)]
            for name in self._state.skeleton_nodes:
                node_options.append((name, name))

            with Horizontal(classes="form-row"):
                yield Static("Anchor Part:", classes="form-label")
                yield Select(
                    node_options,
                    value=self._state._anchor_part,
                    id="centroid-anchor-select",
                )

            yield InfoBox(
                "The anchor point is the body part used to define the center of each animal. "
                "Choose a stable, easily detectable part like the head or thorax.",
            )

            yield Rule()

            # Model Settings
            yield Static("Model Configuration", classes="section-title")

            # Backbone
            with Horizontal(classes="form-row"):
                yield Static("Backbone:", classes="form-label")
                yield Select(
                    [
                        ("UNet (Medium RF)", "unet_medium_rf"),
                        ("UNet (Large RF)", "unet_large_rf"),
                        ("ConvNeXt (Tiny)", "convnext_tiny"),
                    ],
                    value=self._state._backbone,
                    id="centroid-backbone-select",
                )

            # Max Stride
            with Horizontal(classes="form-row"):
                yield Static("Max Stride:", classes="form-label")
                yield Select(
                    [
                        ("16", 16),
                        ("32", 32),
                    ],
                    value=self._state._max_stride,
                    id="centroid-stride-select",
                )

            # Filters
            with Horizontal(classes="form-row"):
                yield Static("Base Filters:", classes="form-label")
                yield Select(
                    [
                        ("24", 24),
                        ("32", 32),
                    ],
                    value=self._state._filters,
                    id="centroid-filters-select",
                )

            yield Rule()

            # Head Settings
            yield Static("Confidence Map Settings", classes="section-title")

            yield LabeledSlider(
                label="Sigma",
                value=self._state._sigma,
                min_value=2.0,
                max_value=10.0,
                step=0.5,
                format_str="{:.1f}",
                unit="px",
                id="centroid-sigma-slider",
            )

            with Horizontal(classes="form-row"):
                yield Static("Output Stride:", classes="form-label")
                yield Select(
                    [
                        ("1 (full)", 1),
                        ("2 (half)", 2),
                        ("4 (quarter)", 4),
                    ],
                    value=self._state._output_stride,
                    id="centroid-output-stride-select",
                )

            yield Rule()

            # Training Settings
            yield Static("Training Settings", classes="section-title")

            with Horizontal(classes="form-row"):
                yield Static("Batch Size:", classes="form-label")
                yield Select(
                    [(str(i), i) for i in [1, 2, 4, 8]],
                    value=self._state._batch_size,
                    id="centroid-batch-select",
                )

            with Horizontal(classes="form-row"):
                yield Static("Max Epochs:", classes="form-label")
                yield Input(
                    value=str(self._state._max_epochs),
                    type="integer",
                    id="centroid-epochs-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Learning Rate:", classes="form-label")
                yield Select(
                    [
                        ("1e-4", 1e-4),
                        ("5e-4", 5e-4),
                        ("1e-3", 1e-3),
                    ],
                    value=self._state._learning_rate,
                    id="centroid-lr-select",
                )

    def _compose_centered_instance_tab(self) -> ComposeResult:
        """Compose centered instance model configuration tab."""
        with VerticalScroll(id="ci-scroll"):
            yield Static(
                "[bold]Step 2:[/bold] Centered Instance Model",
                classes="step-indicator",
            )

            yield TipBox(
                "The centered instance model detects all keypoints within cropped regions "
                "centered on each detected centroid. This typically uses higher resolution input.",
                title=None,
            )

            # Crop Configuration
            yield Static("Crop Configuration", classes="section-title")

            with Horizontal(classes="form-row"):
                yield Static("Crop Size:", classes="form-label")
                yield Input(
                    value=str(self._state._crop_size or ""),
                    placeholder=f"Auto (~{self._state._compute_auto_crop_size()}px)",
                    type="integer",
                    id="ci-crop-size-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Min Crop Size:", classes="form-label")
                yield Input(
                    value=str(self._state._ci_min_crop_size),
                    type="integer",
                    id="ci-min-crop-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Crop Padding:", classes="form-label")
                yield Input(
                    value=str(self._state._ci_crop_padding or ""),
                    placeholder="Auto",
                    type="integer",
                    id="ci-crop-padding-input",
                )

            yield LabeledSlider(
                label="Input Scale",
                value=self._state._ci_input_scale,
                min_value=0.5,
                max_value=1.0,
                step=0.125,
                format_str="{:.3f}",
                id="ci-scale-slider",
            )

            yield InfoBox(
                f"Based on your data, recommended crop size is ~{self._state._compute_auto_crop_size()}px. "
                "This should be large enough to contain the entire animal.",
            )

            yield Rule()

            # Model Settings
            yield Static("Model Configuration", classes="section-title")

            with Horizontal(classes="form-row"):
                yield Static("Backbone:", classes="form-label")
                yield Select(
                    [
                        ("UNet (Medium RF)", "unet_medium_rf"),
                        ("UNet (Large RF)", "unet_large_rf"),
                    ],
                    value=self._state._ci_backbone,
                    id="ci-backbone-select",
                )

            with Horizontal(classes="form-row"):
                yield Static("Max Stride:", classes="form-label")
                yield Select(
                    [
                        ("8", 8),
                        ("16", 16),
                        ("32", 32),
                    ],
                    value=self._state._ci_max_stride,
                    id="ci-stride-select",
                )

            with Horizontal(classes="form-row"):
                yield Static("Base Filters:", classes="form-label")
                yield Select(
                    [
                        ("24", 24),
                        ("32", 32),
                        ("48", 48),
                    ],
                    value=self._state._ci_filters,
                    id="ci-filters-select",
                )

            with Horizontal(classes="form-row"):
                yield Static("Filters Rate:", classes="form-label")
                yield Input(
                    value=str(self._state._ci_filters_rate),
                    type="number",
                    id="ci-filters-rate-input",
                )

            yield Rule()

            # Head Settings
            yield Static("Confidence Map Settings", classes="section-title")

            yield LabeledSlider(
                label="Sigma",
                value=self._state._ci_sigma,
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                format_str="{:.1f}",
                unit="px",
                id="ci-sigma-slider",
            )

            yield SigmaVisualization(
                sigma=self._state._ci_sigma,
                output_stride=self._state._ci_output_stride,
                id="ci-sigma-viz",
            )

            with Horizontal(classes="form-row"):
                yield Static("Output Stride:", classes="form-label")
                yield Select(
                    [
                        ("1 (full)", 1),
                        ("2 (half)", 2),
                    ],
                    value=self._state._ci_output_stride,
                    id="ci-output-stride-select",
                )

            yield Rule()

            # Training Settings
            yield Static("Training Settings", classes="section-title")

            with Horizontal(classes="form-row"):
                yield Static("Batch Size:", classes="form-label")
                yield Select(
                    [(str(i), i) for i in [1, 2, 4, 8, 16]],
                    value=self._state._ci_batch_size,
                    id="ci-batch-select",
                )

            with Horizontal(classes="form-row"):
                yield Static("Max Epochs:", classes="form-label")
                yield Input(
                    value=str(self._state._ci_max_epochs),
                    type="integer",
                    id="ci-epochs-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Learning Rate:", classes="form-label")
                yield Select(
                    [
                        ("1e-4", 1e-4),
                        ("5e-4", 5e-4),
                        ("1e-3", 1e-3),
                    ],
                    value=self._state._ci_learning_rate,
                    id="ci-lr-select",
                )

            yield Rule()

            # Augmentation (independent from centroid)
            yield Static("Data Augmentation", classes="section-title")

            with Horizontal(classes="form-row"):
                yield Static("Enable Augmentation:", classes="form-label")
                yield Switch(
                    value=self._state._ci_augmentation.enabled,
                    id="ci-aug-switch",
                )

            yield RangeSlider(
                label="Rotation",
                min_val=self._state._ci_augmentation.rotation_min,
                max_val=self._state._ci_augmentation.rotation_max,
                limit_min=-180.0,
                limit_max=180.0,
                step=5.0,
                format_str="{:.0f}",
                unit="°",
                id="ci-rotation-slider",
            )

            yield RangeSlider(
                label="Scale",
                min_val=self._state._ci_augmentation.scale_min,
                max_val=self._state._ci_augmentation.scale_max,
                limit_min=0.5,
                limit_max=2.0,
                step=0.05,
                format_str="{:.2f}",
                id="ci-scale-aug-slider",
            )

    # Event handlers for centroid tab

    @on(Select.Changed, "#centroid-anchor-select")
    def handle_anchor_change(self, event: Select.Changed) -> None:
        """Handle anchor selection changes."""
        self._state._anchor_part = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#centroid-backbone-select")
    def handle_centroid_backbone_change(self, event: Select.Changed) -> None:
        """Handle centroid backbone selection changes."""
        self._state._backbone = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#centroid-stride-select")
    def handle_centroid_stride_change(self, event: Select.Changed) -> None:
        """Handle centroid stride selection changes."""
        self._state._max_stride = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#centroid-filters-select")
    def handle_centroid_filters_change(self, event: Select.Changed) -> None:
        """Handle centroid filters selection changes."""
        self._state._filters = event.value
        self._state.notify_observers()

    @on(LabeledSlider.Changed, "#centroid-sigma-slider")
    def handle_centroid_sigma_change(self, event: LabeledSlider.Changed) -> None:
        """Handle centroid sigma slider changes."""
        self._state._sigma = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#centroid-output-stride-select")
    def handle_centroid_output_stride_change(self, event: Select.Changed) -> None:
        """Handle centroid output stride selection changes."""
        self._state._output_stride = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#centroid-batch-select")
    def handle_centroid_batch_change(self, event: Select.Changed) -> None:
        """Handle centroid batch size selection changes."""
        self._state._batch_size = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#centroid-epochs-input")
    def handle_centroid_epochs_change(self, event: Input.Changed) -> None:
        """Handle centroid epochs input changes."""
        try:
            self._state._max_epochs = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Select.Changed, "#centroid-lr-select")
    def handle_centroid_lr_change(self, event: Select.Changed) -> None:
        """Handle centroid learning rate selection changes."""
        self._state._learning_rate = event.value
        self._state.notify_observers()

    # Event handlers for centered instance tab

    @on(Input.Changed, "#ci-crop-size-input")
    def handle_ci_crop_size_change(self, event: Input.Changed) -> None:
        """Handle crop size input changes."""
        try:
            self._state._crop_size = int(event.value) if event.value else None
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#ci-min-crop-input")
    def handle_ci_min_crop_change(self, event: Input.Changed) -> None:
        """Handle min crop size input changes."""
        try:
            self._state._ci_min_crop_size = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#ci-crop-padding-input")
    def handle_ci_crop_padding_change(self, event: Input.Changed) -> None:
        """Handle crop padding input changes."""
        try:
            self._state._ci_crop_padding = int(event.value) if event.value else None
            self._state.notify_observers()
        except ValueError:
            pass

    @on(LabeledSlider.Changed, "#ci-scale-slider")
    def handle_ci_scale_change(self, event: LabeledSlider.Changed) -> None:
        """Handle CI input scale slider changes."""
        self._state._ci_input_scale = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#ci-backbone-select")
    def handle_ci_backbone_change(self, event: Select.Changed) -> None:
        """Handle CI backbone selection changes."""
        self._state._ci_backbone = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#ci-stride-select")
    def handle_ci_stride_change(self, event: Select.Changed) -> None:
        """Handle CI stride selection changes."""
        self._state._ci_max_stride = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#ci-filters-select")
    def handle_ci_filters_change(self, event: Select.Changed) -> None:
        """Handle CI filters selection changes."""
        self._state._ci_filters = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#ci-filters-rate-input")
    def handle_ci_filters_rate_change(self, event: Input.Changed) -> None:
        """Handle CI filters rate input changes."""
        try:
            self._state._ci_filters_rate = float(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(LabeledSlider.Changed, "#ci-sigma-slider")
    def handle_ci_sigma_change(self, event: LabeledSlider.Changed) -> None:
        """Handle CI sigma slider changes."""
        self._state._ci_sigma = event.value
        # Update visualization
        try:
            viz = self.query_one("#ci-sigma-viz", SigmaVisualization)
            viz.update_sigma(event.value, self._state._ci_output_stride)
        except Exception:
            pass
        self._state.notify_observers()

    @on(Select.Changed, "#ci-output-stride-select")
    def handle_ci_output_stride_change(self, event: Select.Changed) -> None:
        """Handle CI output stride selection changes."""
        self._state._ci_output_stride = event.value
        # Update visualization
        try:
            viz = self.query_one("#ci-sigma-viz", SigmaVisualization)
            viz.update_sigma(self._state._ci_sigma, event.value)
        except Exception:
            pass
        self._state.notify_observers()

    @on(Select.Changed, "#ci-batch-select")
    def handle_ci_batch_change(self, event: Select.Changed) -> None:
        """Handle CI batch size selection changes."""
        self._state._ci_batch_size = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#ci-epochs-input")
    def handle_ci_epochs_change(self, event: Input.Changed) -> None:
        """Handle CI epochs input changes."""
        try:
            self._state._ci_max_epochs = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Select.Changed, "#ci-lr-select")
    def handle_ci_lr_change(self, event: Select.Changed) -> None:
        """Handle CI learning rate selection changes."""
        self._state._ci_learning_rate = event.value
        self._state.notify_observers()

    @on(Switch.Changed, "#ci-aug-switch")
    def handle_ci_aug_toggle(self, event: Switch.Changed) -> None:
        """Handle CI augmentation toggle."""
        self._state._ci_augmentation.enabled = event.value
        self._state.notify_observers()

    @on(RangeSlider.Changed, "#ci-rotation-slider")
    def handle_ci_rotation_change(self, event: RangeSlider.Changed) -> None:
        """Handle CI rotation range changes."""
        self._state._ci_augmentation.rotation_min = event.min_val
        self._state._ci_augmentation.rotation_max = event.max_val
        self._state.notify_observers()

    @on(RangeSlider.Changed, "#ci-scale-aug-slider")
    def handle_ci_scale_aug_change(self, event: RangeSlider.Changed) -> None:
        """Handle CI scale augmentation range changes."""
        self._state._ci_augmentation.scale_min = event.min_val
        self._state._ci_augmentation.scale_max = event.max_val
        self._state.notify_observers()
