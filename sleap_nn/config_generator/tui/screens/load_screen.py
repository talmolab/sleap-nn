"""Load data screen for the config generator TUI.

Step 1: Load and analyze an SLP file.
"""

from pathlib import Path
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.message import Message
from textual.widgets import Button, Input, Label, Static
from textual.widget import Widget

from sleap_nn.config_generator.tui.state import ConfigState


class DatasetSummary(Static):
    """Widget to display dataset statistics summary."""

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize with optional state."""
        super().__init__(**kwargs)
        self._state = state

    def update_state(self, state: ConfigState) -> None:
        """Update with new state."""
        self._state = state
        self.refresh()

    def render(self) -> str:
        """Render the dataset summary."""
        if self._state is None:
            return "[dim]No data loaded[/dim]"

        stats = self._state.stats
        lines = [
            "[bold cyan]Dataset Summary[/bold cyan]",
            "",
            f"[bold]File:[/bold] {Path(stats.slp_path).name}",
            f"[bold]Labeled frames:[/bold] {stats.num_labeled_frames:,}",
            f"[bold]Videos:[/bold] {stats.num_videos}",
            "",
            f"[bold]Image size:[/bold] {stats.max_width}x{stats.max_height} "
            f"({'grayscale' if stats.is_grayscale else 'RGB'})",
            "",
            f"[bold]Max instances/frame:[/bold] {stats.max_instances_per_frame}",
            f"[bold]Avg instances/frame:[/bold] {stats.avg_instances_per_frame:.1f}",
            f"[bold]Max bbox size:[/bold] {stats.max_bbox_size:.0f}px",
            f"[bold]Avg bbox size:[/bold] {stats.avg_bbox_size:.0f}px",
            "",
            f"[bold]Skeleton:[/bold] {stats.num_nodes} nodes, {stats.num_edges} edges",
            f"[bold]Tracks:[/bold] {stats.num_tracks if stats.has_tracks else 'none'}",
        ]

        # Add data-based recommendation hint
        if stats.is_single_instance:
            lines.append("")
            lines.append("[green]Single animal detected[/green]")
        else:
            lines.append("")
            animal_pct = stats.animal_to_frame_ratio * 100
            overlap_pct = stats.overlap_frequency * 100
            if animal_pct < 20:
                lines.append(f"[yellow]Small animals (~{animal_pct:.0f}% of frame)[/yellow]")
            else:
                lines.append(f"[yellow]Large animals (~{animal_pct:.0f}% of frame)[/yellow]")
            lines.append(f"[yellow]Overlap frequency: {overlap_pct:.1f}%[/yellow]")

        return "\n".join(lines)


class LoadScreen(Widget):
    """Screen for loading and analyzing SLP files."""

    DEFAULT_CSS = """
    LoadScreen {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #load-container {
        width: 100%;
        height: auto;
    }

    #file-input-group {
        width: 100%;
        height: auto;
        padding: 1;
    }

    #file-input {
        width: 1fr;
    }

    #browse-btn {
        min-width: 12;
        margin-left: 1;
    }

    #load-btn {
        min-width: 12;
        margin-left: 1;
    }

    #summary-container {
        width: 100%;
        height: auto;
        margin-top: 2;
    }

    #dataset-summary {
        background: $panel;
        border: solid $primary;
        padding: 1 2;
        min-height: 15;
    }
    """

    class FileLoaded(Message):
        """Message sent when a file is loaded."""

        def __init__(self, state: ConfigState):
            """Initialize with the loaded state."""
            super().__init__()
            self.state = state

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize the load screen.

        Args:
            state: Optional existing ConfigState.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical(id="load-container"):
            yield Label("[bold]Step 1: Load Your Data[/bold]", classes="section-title")
            yield Label(
                "Enter the path to your .slp file (SLEAP labels file)",
                classes="hint"
            )

            with Horizontal(id="file-input-group"):
                yield Input(
                    placeholder="Path to .slp file...",
                    id="file-input",
                    value=str(self._state.slp_path) if self._state else "",
                )
                yield Button("Load", id="load-btn", variant="primary")

            with Container(id="summary-container"):
                yield DatasetSummary(self._state, id="dataset-summary")

    def on_mount(self) -> None:
        """Handle mount - update summary if state exists."""
        if self._state is not None:
            summary = self.query_one("#dataset-summary", DatasetSummary)
            summary.update_state(self._state)

    @on(Button.Pressed, "#load-btn")
    async def handle_load(self) -> None:
        """Handle load button press."""
        file_input = self.query_one("#file-input", Input)
        path = file_input.value.strip()

        if not path:
            self.app.notify("Please enter a file path", severity="error")
            return

        path_obj = Path(path).expanduser().resolve()

        if not path_obj.exists():
            self.app.notify(f"File not found: {path}", severity="error")
            return

        if not path_obj.suffix.lower() in [".slp", ".h5"]:
            self.app.notify("Please select a .slp file", severity="error")
            return

        try:
            self.app.notify("Loading and analyzing file...")
            self._state = ConfigState(str(path_obj))

            # Trigger lazy loading of stats
            _ = self._state.stats

            # Update summary display
            summary = self.query_one("#dataset-summary", DatasetSummary)
            summary.update_state(self._state)

            # Auto-configure with defaults
            self._state.auto_configure()

            # Post message to parent app
            self.post_message(self.FileLoaded(self._state))

        except Exception as e:
            self.app.notify(f"Error loading file: {e}", severity="error")

    @on(Input.Submitted, "#file-input")
    async def handle_input_submit(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        await self.handle_load()
