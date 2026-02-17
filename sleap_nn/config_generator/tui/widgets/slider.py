"""Custom slider widget with numeric input.

Provides a slider with integrated numeric input for precise value entry.
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, ProgressBar, Static
from textual.widget import Widget


class LabeledSlider(Widget):
    """Slider widget with label, progress bar, and numeric input.

    Combines a visual progress bar with a numeric input field
    for both visual feedback and precise value entry.

    Attributes:
        value: Current slider value.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        step: Step increment for value changes.
        label: Display label for the slider.
    """

    DEFAULT_CSS = """
    LabeledSlider {
        height: auto;
        padding: 0;
        margin: 1 0;
    }

    LabeledSlider .slider-header {
        height: auto;
        width: 100%;
    }

    LabeledSlider .slider-label {
        width: 1fr;
    }

    LabeledSlider .slider-value-display {
        width: auto;
        color: $primary;
        text-style: bold;
        text-align: right;
        min-width: 8;
    }

    LabeledSlider .slider-controls {
        height: auto;
        width: 100%;
        margin-top: 1;
    }

    LabeledSlider ProgressBar {
        width: 1fr;
        margin-right: 1;
    }

    LabeledSlider Input {
        width: 10;
    }

    LabeledSlider .slider-range {
        height: auto;
        width: 100%;
        margin-top: 0;
    }

    LabeledSlider .range-min {
        width: 1fr;
        color: $text-muted;
    }

    LabeledSlider .range-max {
        width: auto;
        color: $text-muted;
        text-align: right;
    }
    """

    value: reactive[float] = reactive(0.0)
    min_value: reactive[float] = reactive(0.0)
    max_value: reactive[float] = reactive(100.0)
    step: reactive[float] = reactive(1.0)

    class Changed(Message):
        """Posted when the slider value changes."""

        def __init__(self, slider: "LabeledSlider", value: float) -> None:
            super().__init__()
            self.slider = slider
            self.value = value

        @property
        def control(self) -> "LabeledSlider":
            return self.slider

    def __init__(
        self,
        label: str = "Value",
        value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 100.0,
        step: float = 1.0,
        format_str: str = "{:.1f}",
        unit: str = "",
        show_range: bool = True,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the labeled slider.

        Args:
            label: Display label.
            value: Initial value.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            step: Step increment.
            format_str: Format string for value display.
            unit: Unit suffix for display (e.g., "px", "%").
            show_range: Whether to show min/max range labels.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._label = label
        self._format_str = format_str
        self._unit = unit
        self._show_range = show_range

        # Set initial values
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.value = max(min_value, min(max_value, value))

    def compose(self) -> ComposeResult:
        """Compose the slider layout."""
        with Horizontal(classes="slider-header"):
            yield Static(self._label, classes="slider-label")
            yield Static(
                self._format_value(self.value),
                id="value-display",
                classes="slider-value-display",
            )

        with Horizontal(classes="slider-controls"):
            yield ProgressBar(
                total=100,
                show_eta=False,
                show_percentage=False,
                id="progress",
            )
            yield Input(
                value=str(self.value),
                type="number",
                id="input",
            )

        if self._show_range:
            with Horizontal(classes="slider-range"):
                yield Static(
                    self._format_str.format(self.min_value),
                    classes="range-min",
                )
                yield Static(
                    self._format_str.format(self.max_value),
                    classes="range-max",
                )

    def on_mount(self) -> None:
        """Initialize progress bar on mount."""
        self._update_progress()

    def _format_value(self, value: float) -> str:
        """Format value for display."""
        formatted = self._format_str.format(value)
        if self._unit:
            formatted = f"{formatted}{self._unit}"
        return formatted

    def _update_progress(self) -> None:
        """Update the progress bar to match current value."""
        try:
            progress = self.query_one("#progress", ProgressBar)
            if self.max_value > self.min_value:
                percent = (
                    (self.value - self.min_value)
                    / (self.max_value - self.min_value)
                    * 100
                )
                progress.update(progress=percent)
        except Exception:
            pass

    def _update_display(self) -> None:
        """Update the value display."""
        try:
            display = self.query_one("#value-display", Static)
            display.update(self._format_value(self.value))
        except Exception:
            pass

    def watch_value(self, value: float) -> None:
        """React to value changes."""
        self._update_progress()
        self._update_display()

    @on(Input.Changed, "#input")
    def handle_input_change(self, event: Input.Changed) -> None:
        """Handle direct input changes."""
        try:
            new_value = float(event.value)
            # Clamp to range
            new_value = max(self.min_value, min(self.max_value, new_value))
            # Snap to step
            if self.step > 0:
                new_value = round(new_value / self.step) * self.step

            if new_value != self.value:
                self.value = new_value
                self.post_message(self.Changed(self, new_value))
        except ValueError:
            pass

    def increase(self) -> None:
        """Increase value by one step."""
        new_value = min(self.max_value, self.value + self.step)
        if new_value != self.value:
            self.value = new_value
            self._update_input()
            self.post_message(self.Changed(self, new_value))

    def decrease(self) -> None:
        """Decrease value by one step."""
        new_value = max(self.min_value, self.value - self.step)
        if new_value != self.value:
            self.value = new_value
            self._update_input()
            self.post_message(self.Changed(self, new_value))

    def _update_input(self) -> None:
        """Update the input field to match current value."""
        try:
            input_widget = self.query_one("#input", Input)
            input_widget.value = str(self.value)
        except Exception:
            pass

    def set_value(self, value: float) -> None:
        """Programmatically set the slider value.

        Args:
            value: New value to set.
        """
        clamped = max(self.min_value, min(self.max_value, value))
        if clamped != self.value:
            self.value = clamped
            self._update_input()


class RangeSlider(Widget):
    """Dual-handle range slider for min/max values.

    Provides visual representation of a range with two adjustable bounds.
    """

    DEFAULT_CSS = """
    RangeSlider {
        height: auto;
        padding: 0;
        margin: 1 0;
    }

    RangeSlider .range-header {
        height: auto;
        width: 100%;
    }

    RangeSlider .range-label {
        width: 1fr;
    }

    RangeSlider .range-value-display {
        width: auto;
        color: $primary;
        text-style: bold;
    }

    RangeSlider .range-inputs {
        height: auto;
        width: 100%;
        margin-top: 1;
    }

    RangeSlider Input {
        width: 1fr;
    }

    RangeSlider .range-separator {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    min_val: reactive[float] = reactive(0.0)
    max_val: reactive[float] = reactive(100.0)

    class Changed(Message):
        """Posted when the range values change."""

        def __init__(
            self, slider: "RangeSlider", min_val: float, max_val: float
        ) -> None:
            super().__init__()
            self.slider = slider
            self.min_val = min_val
            self.max_val = max_val

        @property
        def control(self) -> "RangeSlider":
            return self.slider

    def __init__(
        self,
        label: str = "Range",
        min_val: float = 0.0,
        max_val: float = 100.0,
        limit_min: float = -1000.0,
        limit_max: float = 1000.0,
        step: float = 1.0,
        format_str: str = "{:.1f}",
        unit: str = "",
        symmetric: bool = False,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the range slider.

        Args:
            label: Display label.
            min_val: Initial minimum value.
            max_val: Initial maximum value.
            limit_min: Absolute minimum allowed.
            limit_max: Absolute maximum allowed.
            step: Step increment.
            format_str: Format string for values.
            unit: Unit suffix for display.
            symmetric: If True, min = -max when adjusting.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._label = label
        self._limit_min = limit_min
        self._limit_max = limit_max
        self._step = step
        self._format_str = format_str
        self._unit = unit
        self._symmetric = symmetric

        self.min_val = min_val
        self.max_val = max_val

    def compose(self) -> ComposeResult:
        """Compose the range slider layout."""
        with Horizontal(classes="range-header"):
            yield Static(self._label, classes="range-label")
            yield Static(
                self._format_range(),
                id="range-display",
                classes="range-value-display",
            )

        with Horizontal(classes="range-inputs"):
            yield Input(
                value=str(self.min_val),
                type="number",
                id="min-input",
            )
            yield Static("to", classes="range-separator")
            yield Input(
                value=str(self.max_val),
                type="number",
                id="max-input",
            )

    def _format_range(self) -> str:
        """Format range for display."""
        min_str = self._format_str.format(self.min_val)
        max_str = self._format_str.format(self.max_val)
        if self._unit:
            return f"{min_str} to {max_str}{self._unit}"
        return f"{min_str} to {max_str}"

    def _update_display(self) -> None:
        """Update the range display."""
        try:
            display = self.query_one("#range-display", Static)
            display.update(self._format_range())
        except Exception:
            pass

    def watch_min_val(self, value: float) -> None:
        """React to min value changes."""
        self._update_display()

    def watch_max_val(self, value: float) -> None:
        """React to max value changes."""
        self._update_display()

    @on(Input.Changed, "#min-input")
    def handle_min_change(self, event: Input.Changed) -> None:
        """Handle min input changes."""
        try:
            new_min = float(event.value)
            new_min = max(self._limit_min, min(self.max_val, new_min))
            if self._step > 0:
                new_min = round(new_min / self._step) * self._step

            if self._symmetric:
                self.max_val = -new_min

            if new_min != self.min_val:
                self.min_val = new_min
                self.post_message(self.Changed(self, self.min_val, self.max_val))
        except ValueError:
            pass

    @on(Input.Changed, "#max-input")
    def handle_max_change(self, event: Input.Changed) -> None:
        """Handle max input changes."""
        try:
            new_max = float(event.value)
            new_max = max(self.min_val, min(self._limit_max, new_max))
            if self._step > 0:
                new_max = round(new_max / self._step) * self._step

            if self._symmetric:
                self.min_val = -new_max

            if new_max != self.max_val:
                self.max_val = new_max
                self.post_message(self.Changed(self, self.min_val, self.max_val))
        except ValueError:
            pass

    def set_range(self, min_val: float, max_val: float) -> None:
        """Programmatically set the range values.

        Args:
            min_val: New minimum value.
            max_val: New maximum value.
        """
        self.min_val = max(self._limit_min, min(max_val, min_val))
        self.max_val = max(min_val, min(self._limit_max, max_val))
        try:
            self.query_one("#min-input", Input).value = str(self.min_val)
            self.query_one("#max-input", Input).value = str(self.max_val)
        except Exception:
            pass
