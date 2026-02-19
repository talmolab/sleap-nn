"""Main TUI application for config generation.

This module provides the main Textual application for interactively
generating sleap-nn training configurations with a step-by-step wizard.
"""

from pathlib import Path
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Static,
)

from sleap_nn.config_generator.tui.state import ConfigState
from sleap_nn.config_generator.tui.screens.load_screen import LoadScreen
from sleap_nn.config_generator.tui.screens.model_select_screen import ModelSelectScreen
from sleap_nn.config_generator.tui.screens.configure_screen import ConfigureScreen
from sleap_nn.config_generator.tui.screens.export_screen import ExportScreen


class StepIndicator(Static):
    """Visual step indicator for the wizard."""

    current_step: reactive[int] = reactive(1)
    total_steps: int = 4

    STEP_LABELS = ["Load Data", "Select Model", "Configure", "Export"]

    def render(self) -> str:
        """Render the step indicator."""
        parts = []
        for i in range(1, self.total_steps + 1):
            if i < self.current_step:
                parts.append(f"[green][{i}][/green]")
            elif i == self.current_step:
                parts.append(f"[bold cyan]({i})[/bold cyan]")
            else:
                parts.append(f"[dim][{i}][/dim]")

        step_line = "──".join(parts)
        label = self.STEP_LABELS[self.current_step - 1] if self.current_step <= len(self.STEP_LABELS) else ""
        return f"{step_line}\n[bold]{label}[/bold]"


class ConfigGeneratorApp(App):
    """Interactive TUI for generating sleap-nn training configurations.

    This application provides a step-by-step wizard for:
    1. Loading and analyzing SLP files
    2. Selecting model type with smart recommendations
    3. Configuring training parameters (simplified)
    4. Previewing and exporting YAML configurations
    """

    TITLE = "SLEAP-NN Config Generator"

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    #step-indicator {
        dock: top;
        height: 3;
        padding: 0 2;
        text-align: center;
        background: $panel;
        border-bottom: solid $primary;
    }

    #content-area {
        width: 100%;
        height: 1fr;
        padding: 1 0;
    }

    #nav-buttons {
        dock: bottom;
        height: 3;
        padding: 0 2;
        align: center middle;
        background: $panel;
        border-top: solid $primary;
    }

    #nav-buttons Button {
        margin: 0 1;
        min-width: 16;
    }

    .nav-back {
        background: $surface-lighten-1;
    }

    .nav-next {
        background: $success;
    }

    .section-title {
        text-style: bold;
        padding: 1 0;
        color: $text;
    }

    .form-group {
        height: auto;
        padding: 1 0;
    }

    .form-label {
        padding: 0 1 0 0;
        min-width: 20;
    }

    .hint {
        color: $text-muted;
        text-style: italic;
        padding-left: 2;
    }

    .info-panel {
        background: $panel;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
    }

    .warning-panel {
        background: $warning-darken-3;
        border: solid $warning;
        padding: 1;
        margin: 1 0;
    }

    .success-panel {
        background: $success-darken-3;
        border: solid $success;
        padding: 1;
        margin: 1 0;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("ctrl+s", "save", "Save Config"),
        Binding("left", "prev_step", "Previous", show=False),
        Binding("right", "next_step", "Next", show=False),
        Binding("f1", "help", "Help"),
    ]

    current_step: reactive[int] = reactive(1)

    def __init__(self, slp_path: Optional[str] = None, **kwargs):
        """Initialize the config generator app.

        Args:
            slp_path: Optional path to the SLP file to analyze.
            **kwargs: Additional arguments passed to parent App class.
        """
        super().__init__(**kwargs)
        self.slp_path = Path(slp_path) if slp_path else None
        self._state: Optional[ConfigState] = None

        # Create screens (will be added to content area)
        self._screens = {}

    @property
    def state(self) -> Optional[ConfigState]:
        """Get the current config state."""
        return self._state

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        with Container(id="main-container"):
            yield StepIndicator(id="step-indicator")

            with Container(id="content-area"):
                # Screens are mounted dynamically
                pass

            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn", classes="nav-back", disabled=True)
                yield Button("Next", id="next-btn", classes="nav-next")

        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount - show initial screen."""
        # Initialize with slp_path if provided
        if self.slp_path and self.slp_path.exists():
            self._state = ConfigState(str(self.slp_path))

        await self._show_step(1)

    async def _show_step(self, step: int) -> None:
        """Show the specified step screen."""
        self.current_step = step

        # Update step indicator
        indicator = self.query_one("#step-indicator", StepIndicator)
        indicator.current_step = step

        # Update navigation buttons
        back_btn = self.query_one("#back-btn", Button)
        next_btn = self.query_one("#next-btn", Button)

        back_btn.disabled = step == 1
        next_btn.label = "Export" if step == 4 else "Next"
        next_btn.disabled = step == 1 and self._state is None  # Can't proceed without data

        # Clear and mount appropriate screen
        content_area = self.query_one("#content-area")
        await content_area.remove_children()

        if step == 1:
            screen = LoadScreen(self._state, id="load-screen")
        elif step == 2:
            screen = ModelSelectScreen(self._state, id="model-screen")
        elif step == 3:
            screen = ConfigureScreen(self._state, id="configure-screen")
        elif step == 4:
            screen = ExportScreen(self._state, id="export-screen")
        else:
            return

        await content_area.mount(screen)

    def watch_current_step(self, step: int) -> None:
        """React to step changes."""
        # Update step indicator
        try:
            indicator = self.query_one("#step-indicator", StepIndicator)
            indicator.current_step = step
        except Exception:
            pass

    @on(Button.Pressed, "#back-btn")
    async def handle_back(self) -> None:
        """Handle back button press."""
        if self.current_step > 1:
            await self._show_step(self.current_step - 1)

    @on(Button.Pressed, "#next-btn")
    async def handle_next(self) -> None:
        """Handle next button press."""
        if self.current_step == 1:
            # Validate data is loaded
            if self._state is None:
                self.notify("Please load an SLP file first", severity="error")
                return
        elif self.current_step == 2:
            # Validate model type is selected
            if self._state._pipeline is None:
                self.notify("Please select a model type", severity="error")
                return
        elif self.current_step == 4:
            # Export step - save configs
            self.action_save()
            return

        if self.current_step < 4:
            await self._show_step(self.current_step + 1)

    def on_load_screen_file_loaded(self, event) -> None:
        """Handle file loaded event from LoadScreen."""
        self._state = event.state
        # Enable next button
        next_btn = self.query_one("#next-btn", Button)
        next_btn.disabled = False
        self.notify(f"Loaded: {event.state.stats.slp_path}")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_save(self) -> None:
        """Save the configuration."""
        if self._state is None:
            self.notify("No configuration to save", severity="error")
            return

        try:
            if self._state.is_topdown:
                base_path = str(self.slp_path.parent / self.slp_path.stem) if self.slp_path else "config"
                centroid_path, ci_path = self._state.save_dual(base_path)
                self.notify(f"Saved: {centroid_path} and {ci_path}")
            else:
                output_path = self.slp_path.parent / f"{self.slp_path.stem}_config.yaml" if self.slp_path else Path("config.yaml")
                self._state.save(str(output_path))
                self.notify(f"Saved to: {output_path}")
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")

    async def action_prev_step(self) -> None:
        """Navigate to previous step."""
        if self.current_step > 1:
            await self._show_step(self.current_step - 1)

    async def action_next_step(self) -> None:
        """Navigate to next step."""
        await self.handle_next()

    def action_help(self) -> None:
        """Show help information."""
        help_text = """
[bold]SLEAP-NN Config Generator Help[/bold]

[cyan]Steps:[/cyan]
  1. Load Data - Select your .slp file
  2. Select Model - Choose model type with recommendations
  3. Configure - Set training parameters
  4. Export - Preview and save configuration

[cyan]Keyboard Shortcuts:[/cyan]
  q / Escape  - Quit
  Left/Right  - Navigate steps
  Ctrl+S      - Save configuration
  F1          - Show this help

[cyan]Tips:[/cyan]
  - Follow the recommended model type for best results
  - Basic parameters are shown by default
  - Use Advanced Settings for fine-tuning
  - For top-down, two configs will be generated
"""
        self.notify(help_text, title="Help", timeout=15)


def launch_tui(slp_path: Optional[str] = None) -> None:
    """Launch the TUI configuration generator.

    Args:
        slp_path: Optional path to the SLP file to configure.
    """
    app = ConfigGeneratorApp(slp_path)
    app.run()
