"""Main TUI application for config generation.

This module provides the main Textual application for interactively
generating sleap-nn training configurations.
"""

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    TabbedContent,
    TabPane,
)

from sleap_nn.config_generator.tui.screens import (
    DataScreen,
    ExportScreen,
    ModelScreen,
    TrainingScreen,
)
from sleap_nn.config_generator.tui.state import ConfigState
from sleap_nn.config_generator.tui.widgets import (
    MemoryGauge,
    RecommendationPanel,
    DatasetStatsPanel,
    QuickSettingsPanel,
)


class ConfigGeneratorApp(App):
    """Interactive TUI for generating sleap-nn training configurations.

    This application provides a full-featured interface for:
    - Analyzing SLP files and viewing dataset statistics
    - Configuring data preprocessing and augmentation
    - Selecting model architecture and head settings
    - Setting training hyperparameters
    - Previewing and exporting YAML configurations
    - Special support for top-down dual-model pipelines
    """

    TITLE = "SLEAP-NN Config Generator"

    CSS_PATH = "styles/app.tcss"

    # Fallback CSS if file not found
    CSS = """
    #sidebar {
        dock: right;
        width: 40;
        height: 100%;
        padding: 1;
        overflow-y: auto;
    }

    #tabs {
        width: 100%;
        height: 100%;
    }

    ContentSwitcher {
        width: 100%;
        height: 1fr;
    }

    TabPane {
        width: 100%;
        height: 100%;
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

    .button-row {
        height: auto;
        padding: 1;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "save", "Save Config"),
        Binding("a", "auto_fill", "Auto-Fill"),
        Binding("r", "refresh", "Refresh"),
        Binding("tab", "next_tab", "Next Tab", show=False),
        Binding("shift+tab", "prev_tab", "Prev Tab", show=False),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, slp_path: str, **kwargs):
        """Initialize the config generator app.

        Args:
            slp_path: Path to the SLP file to analyze.
            **kwargs: Additional arguments passed to parent App class.
        """
        super().__init__(**kwargs)
        self.slp_path = Path(slp_path)

        # Initialize state
        self._state = ConfigState(str(slp_path))

        # Register as observer for state changes
        self._state.add_observer(self._on_state_change)

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        # Sidebar (docked to right)
        with Vertical(id="sidebar"):
            yield DatasetStatsPanel(self._state.stats, id="stats-panel")
            yield MemoryGauge(id="memory-gauge")
            yield RecommendationPanel(
                self._state.recommendation, id="recommendation-panel"
            )
            yield QuickSettingsPanel(id="quick-settings")

        # Main content area with tabs
        with TabbedContent(id="tabs"):
            with TabPane("Data", id="data-tab"):
                yield DataScreen(self._state, id="data-screen")

            with TabPane("Model", id="model-tab"):
                yield ModelScreen(self._state, id="model-screen")

            with TabPane("Training", id="training-tab"):
                yield TrainingScreen(self._state, id="training-screen")

            with TabPane("Export", id="export-tab"):
                yield ExportScreen(self._state, id="export-screen")

        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount - initialize displays."""
        # Update sidebar displays
        self._update_sidebar()

    def _on_state_change(self) -> None:
        """Handle state change notifications."""
        self._update_sidebar()

    def _update_sidebar(self) -> None:
        """Update sidebar displays."""
        try:
            # Update stats panel
            stats_panel = self.query_one("#stats-panel", DatasetStatsPanel)
            stats_panel.update_stats(self._state.stats)

            # Update memory gauge
            mem = self._state.memory_estimate()
            self.query_one("#memory-gauge", MemoryGauge).update_estimate(mem)

            # Update recommendation panel
            self.query_one(
                "#recommendation-panel", RecommendationPanel
            ).update_recommendation(self._state.recommendation)

            # Update quick settings
            quick_settings = self.query_one("#quick-settings", QuickSettingsPanel)
            quick_settings.update_settings(
                pipeline=self._state._pipeline or "",
                backbone=self._state._backbone,
                batch_size=self._state._batch_size,
                input_scale=self._state._input_scale,
                sigma=self._state._sigma,
            )
        except Exception:
            pass

    @on(TabbedContent.TabActivated)
    def handle_tab_change(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab activation."""
        # Refresh export preview when Export tab is activated
        if event.tab.id == "export-tab":
            try:
                export_screen = self.query_one("#export-screen", ExportScreen)
                export_screen.update_preview()
            except Exception:
                pass

    @on(Button.Pressed, "#auto-fill-btn")
    def handle_auto_fill(self) -> None:
        """Handle auto-fill button press."""
        self.action_auto_fill()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_save(self) -> None:
        """Save the configuration."""
        try:
            if self._state.is_topdown:
                base_path = str(self.slp_path.parent / self.slp_path.stem)
                centroid_path, ci_path = self._state.save_dual(base_path)
                self.notify(f"Saved: {centroid_path} and {ci_path}")
            else:
                output_path = self.slp_path.parent / f"{self.slp_path.stem}_config.yaml"
                self._state.save(str(output_path))
                self.notify(f"Saved to: {output_path}")
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")

    def action_auto_fill(self) -> None:
        """Auto-fill configuration."""
        self._state.auto_configure()

        # Update all screens
        self._update_screens_from_state()

        self.notify("Configuration auto-filled based on data analysis")

    def action_refresh(self) -> None:
        """Refresh all displays."""
        self._update_sidebar()

        # Refresh export preview if on export tab
        try:
            export_screen = self.query_one("#export-screen", ExportScreen)
            export_screen.update_preview()
        except Exception:
            pass

        self.notify("Refreshed")

    def action_next_tab(self) -> None:
        """Navigate to next tab."""
        try:
            tabs = self.query_one("#tabs", TabbedContent)
            tabs.action_next_tab()
        except Exception:
            pass

    def action_prev_tab(self) -> None:
        """Navigate to previous tab."""
        try:
            tabs = self.query_one("#tabs", TabbedContent)
            tabs.action_previous_tab()
        except Exception:
            pass

    def action_help(self) -> None:
        """Show help information."""
        help_text = """
SLEAP-NN Config Generator Help

Keyboard Shortcuts:
  q       - Quit
  s       - Save configuration
  a       - Auto-fill from data analysis
  r       - Refresh displays
  Tab     - Next tab
  Shift+Tab - Previous tab
  F1      - Show this help

Tabs:
  Data    - Configure preprocessing and augmentation
  Model   - Select pipeline and backbone architecture
  Training - Set training hyperparameters
  Top-Down - Configure dual models (if top-down pipeline)
  Export  - Preview and save configuration

Tips:
  - Use 'Auto-Fill' to get recommended settings
  - For top-down pipelines, configure both models
  - Check memory estimates in the sidebar
"""
        self.notify(help_text, title="Help", timeout=10)

    def _update_screens_from_state(self) -> None:
        """Update all screen UIs from current state."""
        try:
            data_screen = self.query_one("#data-screen", DataScreen)
            data_screen._update_ui_from_state()
        except Exception:
            pass

        try:
            model_screen = self.query_one("#model-screen", ModelScreen)
            model_screen._update_ui_from_state()
        except Exception:
            pass

        try:
            training_screen = self.query_one("#training-screen", TrainingScreen)
            training_screen._update_ui_from_state()
        except Exception:
            pass


def launch_tui(slp_path: str) -> None:
    """Launch the TUI configuration generator.

    Args:
        slp_path: Path to the SLP file to configure.
    """
    app = ConfigGeneratorApp(slp_path)
    app.run()
