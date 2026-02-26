"""Data configuration screen.

Provides UI for configuring data preprocessing and augmentation settings.
"""

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Input,
    RadioButton,
    RadioSet,
    Rule,
    Select,
    Static,
    Switch,
)

from sleap_nn.config_generator.tui.widgets import (
    InfoBox,
    LabeledSlider,
    RangeSlider,
    SizeDisplay,
    TipBox,
)

if TYPE_CHECKING:
    from sleap_nn.config_generator.tui.state import ConfigState


class DataScreen(VerticalScroll):
    """Data configuration screen component.

    Provides controls for:
    - Dataset info display
    - Preprocessing settings (scale, channels, max dimensions)
    - Data augmentation configuration
    - View type selection for augmentation defaults
    """

    DEFAULT_CSS = """
    DataScreen {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    .section-title {
        text-style: bold;
        padding: 1 0;
    }

    .form-row {
        height: auto;
        padding: 0 0 1 0;
    }

    .form-label {
        width: 20;
        padding-right: 1;
    }

    .form-input {
        width: 1fr;
    }
    """

    def __init__(self, state: "ConfigState", **kwargs):
        """Initialize the data screen.

        Args:
            state: Shared configuration state.
            **kwargs: Additional arguments for Container.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the data screen layout."""
        # Dataset Info Section
        yield Static("Dataset Analysis", classes="section-title")
        yield Static(str(self._state.stats), id="stats-display")

        yield Rule()

        # Size Display
        yield SizeDisplay(
            original=(self._state.stats.max_width, self._state.stats.max_height),
            scale=self._state._input_scale,
            output_stride=self._state._output_stride,
            title="Effective Image Size",
            id="size-display",
        )

        yield Rule()

        # Preprocessing Section
        yield from self._compose_preprocessing_section()

        yield Rule()

        # Augmentation Section
        yield from self._compose_augmentation_section()

        yield Rule()

        # Auto-fill button
        with Horizontal(classes="button-row"):
            yield Button("Auto-Fill All", id="auto-fill-btn", variant="primary")

    def _compose_preprocessing_section(self) -> ComposeResult:
        """Compose the preprocessing settings section."""
        yield Static("Preprocessing", classes="section-title")

        # Input Scale
        yield LabeledSlider(
            label="Input Scale",
            value=self._state._input_scale,
            min_value=0.125,
            max_value=1.0,
            step=0.125,
            format_str="{:.3f}",
            id="input-scale-slider",
        )

        yield InfoBox(
            "Scale factor applied to input images. Lower values reduce memory "
            "usage but may lose fine details. Use 1.0 for full resolution.",
        )

        # Input Channels
        with Horizontal(classes="form-row"):
            yield Static("Input Channels:", classes="form-label")
            yield Select(
                [
                    ("Grayscale (1 channel)", "grayscale"),
                    ("RGB (3 channels)", "rgb"),
                ],
                value="grayscale" if self._state._ensure_grayscale else "rgb",
                id="channels-select",
            )

        # Max Dimensions
        with Horizontal(classes="form-row"):
            yield Static("Max Height:", classes="form-label")
            yield Input(
                value=str(self._state._max_height or ""),
                placeholder="No limit",
                type="integer",
                id="max-height-input",
            )

        with Horizontal(classes="form-row"):
            yield Static("Max Width:", classes="form-label")
            yield Input(
                value=str(self._state._max_width or ""),
                placeholder="No limit",
                type="integer",
                id="max-width-input",
            )

        yield TipBox(
            "Leave max dimensions empty to use original image size. "
            "Set limits if your images are very large to reduce memory usage.",
            title=None,
        )

        # Data Pipeline
        with Horizontal(classes="form-row"):
            yield Static("Data Pipeline:", classes="form-label")
            yield Select(
                [
                    ("Standard (torch_dataset)", "torch_dataset"),
                    ("Memory Cache (litdata)", "litdata"),
                    ("Disk Cache (litdata_disk)", "litdata_disk"),
                ],
                value=self._state._data_pipeline.value,
                id="pipeline-select",
            )

        # Num Workers
        with Horizontal(classes="form-row"):
            yield Static("Num Workers:", classes="form-label")
            yield Input(
                value=str(self._state._num_workers),
                type="integer",
                id="num-workers-input",
            )

        # Validation Fraction
        with Horizontal(classes="form-row"):
            yield Static("Validation Split:", classes="form-label")
            yield Input(
                value=str(self._state._validation_fraction),
                type="number",
                id="val-fraction-input",
            )

    def _compose_augmentation_section(self) -> ComposeResult:
        """Compose the augmentation settings section."""
        yield Static("Data Augmentation", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Static("Enable Augmentation:", classes="form-label")
            yield Switch(value=self._state._augmentation.enabled, id="aug-switch")

        # Geometric augmentations
        yield Static("Geometric", classes="subsection-title")

        yield RangeSlider(
            label="Rotation (degrees)",
            min_val=self._state._augmentation.rotation_min,
            max_val=self._state._augmentation.rotation_max,
            limit_min=-180.0,
            limit_max=180.0,
            step=5.0,
            format_str="{:.0f}",
            unit="°",
            id="rotation-slider",
        )

        yield RangeSlider(
            label="Scale",
            min_val=self._state._augmentation.scale_min,
            max_val=self._state._augmentation.scale_max,
            limit_min=0.5,
            limit_max=2.0,
            step=0.05,
            format_str="{:.2f}",
            id="scale-slider",
        )

        yield RangeSlider(
            label="Translate X",
            min_val=-self._state._augmentation.translate_x,
            max_val=self._state._augmentation.translate_x,
            limit_min=-0.5,
            limit_max=0.5,
            step=0.05,
            format_str="{:.2f}",
            symmetric=True,
            id="translate-x-slider",
        )

        yield RangeSlider(
            label="Translate Y",
            min_val=-self._state._augmentation.translate_y,
            max_val=self._state._augmentation.translate_y,
            limit_min=-0.5,
            limit_max=0.5,
            step=0.05,
            format_str="{:.2f}",
            symmetric=True,
            id="translate-y-slider",
        )

        # Intensity augmentations
        yield Static("Intensity", classes="subsection-title")

        yield LabeledSlider(
            label="Brightness",
            value=self._state._augmentation.brightness_limit * 100,
            min_value=0.0,
            max_value=50.0,
            step=5.0,
            format_str="{:.0f}",
            unit="%",
            id="brightness-slider",
        )

        yield LabeledSlider(
            label="Contrast",
            value=self._state._augmentation.contrast_limit * 100,
            min_value=0.0,
            max_value=50.0,
            step=5.0,
            format_str="{:.0f}",
            unit="%",
            id="contrast-slider",
        )

        yield TipBox(
            "Augmentation helps the model generalize better by showing "
            "varied versions of training images. Rotation range depends "
            "on your camera view (side vs top).",
            title=None,
        )

    def _update_size_display(self) -> None:
        """Update the size display with current settings."""
        try:
            size_display = self.query_one("#size-display", SizeDisplay)
            size_display.update_sizes(
                original=(self._state.stats.max_width, self._state.stats.max_height),
                scale=self._state._input_scale,
                max_size=(
                    (self._state._max_width, self._state._max_height)
                    if self._state._max_height or self._state._max_width
                    else None
                ),
                output_stride=self._state._output_stride,
            )
        except Exception:
            pass

    @on(LabeledSlider.Changed, "#input-scale-slider")
    def handle_scale_change(self, event: LabeledSlider.Changed) -> None:
        """Handle input scale slider changes."""
        self._state._input_scale = event.value
        self._update_size_display()
        self._state.notify_observers()

    @on(Select.Changed, "#channels-select")
    def handle_channels_change(self, event: Select.Changed) -> None:
        """Handle channel selection changes."""
        if event.value == "grayscale":
            self._state._ensure_grayscale = True
            self._state._ensure_rgb = False
        else:
            self._state._ensure_grayscale = False
            self._state._ensure_rgb = True
        self._state.notify_observers()

    @on(Input.Changed, "#max-height-input")
    def handle_max_height_change(self, event: Input.Changed) -> None:
        """Handle max height input changes."""
        try:
            self._state._max_height = int(event.value) if event.value else None
            self._update_size_display()
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#max-width-input")
    def handle_max_width_change(self, event: Input.Changed) -> None:
        """Handle max width input changes."""
        try:
            self._state._max_width = int(event.value) if event.value else None
            self._update_size_display()
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Select.Changed, "#pipeline-select")
    def handle_pipeline_change(self, event: Select.Changed) -> None:
        """Handle data pipeline selection changes."""
        from sleap_nn.config_generator.tui.state import DataPipelineType

        self._state._data_pipeline = DataPipelineType(event.value)
        self._state.notify_observers()

    @on(Input.Changed, "#num-workers-input")
    def handle_num_workers_change(self, event: Input.Changed) -> None:
        """Handle num workers input changes."""
        try:
            self._state._num_workers = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#val-fraction-input")
    def handle_val_fraction_change(self, event: Input.Changed) -> None:
        """Handle validation fraction input changes."""
        try:
            self._state._validation_fraction = float(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Switch.Changed, "#aug-switch")
    def handle_aug_toggle(self, event: Switch.Changed) -> None:
        """Handle augmentation toggle."""
        self._state._augmentation.enabled = event.value
        self._state.notify_observers()

    @on(RangeSlider.Changed, "#rotation-slider")
    def handle_rotation_change(self, event: RangeSlider.Changed) -> None:
        """Handle rotation range changes."""
        self._state._augmentation.rotation_min = event.min_val
        self._state._augmentation.rotation_max = event.max_val
        self._state.notify_observers()

    @on(RangeSlider.Changed, "#scale-slider")
    def handle_scale_aug_change(self, event: RangeSlider.Changed) -> None:
        """Handle scale augmentation range changes."""
        self._state._augmentation.scale_min = event.min_val
        self._state._augmentation.scale_max = event.max_val
        self._state.notify_observers()

    @on(RangeSlider.Changed, "#translate-x-slider")
    def handle_translate_x_change(self, event: RangeSlider.Changed) -> None:
        """Handle translate X range changes."""
        self._state._augmentation.translate_x = event.max_val
        self._state.notify_observers()

    @on(RangeSlider.Changed, "#translate-y-slider")
    def handle_translate_y_change(self, event: RangeSlider.Changed) -> None:
        """Handle translate Y range changes."""
        self._state._augmentation.translate_y = event.max_val
        self._state.notify_observers()

    @on(LabeledSlider.Changed, "#brightness-slider")
    def handle_brightness_change(self, event: LabeledSlider.Changed) -> None:
        """Handle brightness slider changes."""
        self._state._augmentation.brightness_limit = event.value / 100.0
        self._state.notify_observers()

    @on(LabeledSlider.Changed, "#contrast-slider")
    def handle_contrast_change(self, event: LabeledSlider.Changed) -> None:
        """Handle contrast slider changes."""
        self._state._augmentation.contrast_limit = event.value / 100.0
        self._state.notify_observers()

    @on(Button.Pressed, "#auto-fill-btn")
    def handle_auto_fill(self) -> None:
        """Handle auto-fill button press."""
        # Auto-configure with default settings
        self._state.auto_configure()

        # Update all UI elements
        self._update_ui_from_state()

    def _update_ui_from_state(self) -> None:
        """Update all UI elements from current state."""
        try:
            # Input scale
            scale_slider = self.query_one("#input-scale-slider", LabeledSlider)
            scale_slider.set_value(self._state._input_scale)

            # Channels
            channels_select = self.query_one("#channels-select", Select)
            channels_select.value = (
                "grayscale" if self._state._ensure_grayscale else "rgb"
            )

            # Max dimensions
            self.query_one("#max-height-input", Input).value = (
                str(self._state._max_height) if self._state._max_height else ""
            )
            self.query_one("#max-width-input", Input).value = (
                str(self._state._max_width) if self._state._max_width else ""
            )

            # Augmentation
            self.query_one("#aug-switch", Switch).value = (
                self._state._augmentation.enabled
            )

            rotation_slider = self.query_one("#rotation-slider", RangeSlider)
            rotation_slider.set_range(
                self._state._augmentation.rotation_min,
                self._state._augmentation.rotation_max,
            )

            scale_aug_slider = self.query_one("#scale-slider", RangeSlider)
            scale_aug_slider.set_range(
                self._state._augmentation.scale_min,
                self._state._augmentation.scale_max,
            )

            # Size display
            self._update_size_display()

        except Exception:
            pass
