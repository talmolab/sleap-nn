"""Export configuration screen.

Provides UI for previewing and exporting generated configurations.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Input,
    Rule,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from sleap_nn.config_generator.tui.widgets import (
    InfoBox,
)

if TYPE_CHECKING:
    from sleap_nn.config_generator.tui.state import ConfigState


class ExportScreen(VerticalScroll):
    """Export configuration screen component.

    Provides:
    - YAML configuration preview
    - Dual config preview for top-down pipelines
    - Save to file functionality
    - Copy to clipboard
    - CLI command suggestions
    """

    DEFAULT_CSS = """
    ExportScreen {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    .section-title {
        text-style: bold;
        padding: 1 0;
    }

    .yaml-preview {
        height: 1fr;
        min-height: 20;
    }

    .button-row {
        height: auto;
        padding: 1;
        align: center middle;
    }

    .cli-box {
        height: auto;
        padding: 1;
        margin: 1 0;
        border: solid $surface-lighten-2;
        background: $surface;
    }

    .cli-command {
        color: $success;
    }

    .status-message {
        height: auto;
        padding: 1 0;
    }
    """

    def __init__(self, state: "ConfigState", **kwargs):
        """Initialize the export screen.

        Args:
            state: Shared configuration state.
            **kwargs: Additional arguments for Container.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the export screen layout."""
        yield Static("Configuration Preview", classes="section-title")

        # Show different preview based on pipeline type
        if self._state.is_topdown:
            yield from self._compose_dual_preview()
        else:
            yield from self._compose_single_preview()

        yield Rule()

        # Export Options
        yield from self._compose_export_options()

        yield Rule()

        # CLI Commands
        yield from self._compose_cli_section()

        # Status message
        yield Static("", id="export-status", classes="status-message")

    def _compose_single_preview(self) -> ComposeResult:
        """Compose single config preview."""
        yield TextArea(
            id="yaml-preview",
            read_only=True,
            language="yaml",
            classes="yaml-preview",
        )

    def _compose_dual_preview(self) -> ComposeResult:
        """Compose dual config preview for top-down pipelines."""
        yield InfoBox(
            "Top-down pipelines require two separate configs: one for the centroid model "
            "and one for the centered instance model.",
        )

        with TabbedContent(id="dual-preview-tabs"):
            with TabPane("Centroid Config", id="centroid-tab"):
                yield TextArea(
                    id="centroid-yaml-preview",
                    read_only=True,
                    language="yaml",
                    classes="yaml-preview",
                )

            with TabPane("Centered Instance Config", id="ci-tab"):
                yield TextArea(
                    id="ci-yaml-preview",
                    read_only=True,
                    language="yaml",
                    classes="yaml-preview",
                )

    def _compose_export_options(self) -> ComposeResult:
        """Compose export options section."""
        yield Static("Export Options", classes="section-title")

        # Output path
        with Horizontal(classes="form-row"):
            yield Static("Output Path:", classes="form-label")
            yield Input(
                value=str(self._get_default_output_path()),
                id="output-path-input",
            )

        # Buttons
        with Horizontal(classes="button-row"):
            yield Button("Refresh Preview", id="refresh-btn")
            yield Button("Save to File", id="save-btn", variant="primary")
            yield Button("Copy to Clipboard", id="copy-btn")

    def _compose_cli_section(self) -> ComposeResult:
        """Compose CLI command suggestions section."""
        yield Static("CLI Commands", classes="section-title")

        yield InfoBox(
            "After saving your config, use these commands to train and run inference:",
        )

        # Training command
        output_path = self._get_default_output_path()
        with Container(classes="cli-box"):
            yield Static("Train the model:", classes="subsection-title")
            if self._state.is_topdown:
                yield Static(
                    f"# Step 1: Train centroid model\n"
                    f"sleap-nn train --config {output_path.stem}_centroid.yaml\n\n"
                    f"# Step 2: Train centered instance model\n"
                    f"sleap-nn train --config {output_path.stem}_centered_instance.yaml",
                    id="train-command",
                    classes="cli-command",
                )
            else:
                yield Static(
                    f"sleap-nn train --config {output_path}",
                    id="train-command",
                    classes="cli-command",
                )

        # Inference command
        with Container(classes="cli-box"):
            yield Static("Run inference:", classes="subsection-title")
            if self._state.is_topdown:
                yield Static(
                    "sleap-nn predict \\\n"
                    "  --model-paths centroid_model.ckpt centered_instance_model.ckpt \\\n"
                    "  --video-path your_video.mp4 \\\n"
                    "  --output-path predictions.slp",
                    id="predict-command",
                    classes="cli-command",
                )
            else:
                yield Static(
                    "sleap-nn predict \\\n"
                    "  --model-paths your_model.ckpt \\\n"
                    "  --video-path your_video.mp4 \\\n"
                    "  --output-path predictions.slp",
                    id="predict-command",
                    classes="cli-command",
                )

    def _get_default_output_path(self) -> Path:
        """Get the default output path for the config file."""
        return self._state.slp_path.parent / f"{self._state.slp_path.stem}_config.yaml"

    def update_preview(self) -> None:
        """Update the YAML preview displays."""
        try:
            if self._state.is_topdown:
                # Update dual previews
                try:
                    centroid_preview = self.query_one(
                        "#centroid-yaml-preview", TextArea
                    )
                    centroid_preview.text = self._state.to_centroid_yaml()
                except Exception:
                    pass

                try:
                    ci_preview = self.query_one("#ci-yaml-preview", TextArea)
                    ci_preview.text = self._state.to_centered_instance_yaml()
                except Exception:
                    pass
            else:
                # Update single preview
                try:
                    preview = self.query_one("#yaml-preview", TextArea)
                    preview.text = self._state.to_yaml()
                except Exception:
                    pass
        except Exception as e:
            self._show_status(f"Error generating preview: {e}", error=True)

    def _show_status(
        self, message: str, error: bool = False, success: bool = False
    ) -> None:
        """Show a status message.

        Args:
            message: Message to display.
            error: Whether this is an error message.
            success: Whether this is a success message.
        """
        try:
            status = self.query_one("#export-status", Static)
            if error:
                status.update(f"[red]Error: {message}[/red]")
            elif success:
                status.update(f"[green]✓ {message}[/green]")
            else:
                status.update(message)
        except Exception:
            pass

    @on(Button.Pressed, "#refresh-btn")
    def handle_refresh(self) -> None:
        """Handle refresh button press."""
        self.update_preview()
        self._show_status("Preview refreshed", success=True)

    @on(Button.Pressed, "#save-btn")
    def handle_save(self) -> None:
        """Handle save button press."""
        try:
            output_path = self.query_one("#output-path-input", Input).value

            if self._state.is_topdown:
                # Save dual configs
                base_path = output_path.replace(".yaml", "").replace("_config", "")
                centroid_path, ci_path = self._state.save_dual(base_path)
                self._show_status(
                    f"Saved: {centroid_path} and {ci_path}",
                    success=True,
                )
            else:
                # Save single config
                self._state.save(output_path)
                self._show_status(f"Saved to: {output_path}", success=True)

        except Exception as e:
            self._show_status(str(e), error=True)

    @on(Button.Pressed, "#copy-btn")
    def handle_copy(self) -> None:
        """Handle copy to clipboard button press."""
        try:
            if self._state.is_topdown:
                # Copy both configs
                yaml_str = (
                    "# ============== CENTROID CONFIG ==============\n"
                    + self._state.to_centroid_yaml()
                    + "\n\n# ============== CENTERED INSTANCE CONFIG ==============\n"
                    + self._state.to_centered_instance_yaml()
                )
            else:
                yaml_str = self._state.to_yaml()

            # Try to copy to clipboard
            try:
                import pyperclip

                pyperclip.copy(yaml_str)
                self._show_status("Copied to clipboard!", success=True)
            except ImportError:
                self._show_status(
                    "Install pyperclip for clipboard support: pip install pyperclip",
                    error=True,
                )
        except Exception as e:
            self._show_status(str(e), error=True)

    def on_mount(self) -> None:
        """Initialize preview when mounted."""
        self.update_preview()


class TopDownExportScreen(Container):
    """Specialized export screen for top-down pipelines.

    Shows side-by-side configuration for centroid and centered instance models.
    """

    DEFAULT_CSS = """
    TopDownExportScreen {
        height: 100%;
        padding: 0;
    }

    TopDownExportScreen .dual-preview {
        height: 1fr;
        layout: horizontal;
    }

    TopDownExportScreen .preview-pane {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }

    TopDownExportScreen .preview-title {
        text-style: bold;
        padding: 1 0;
        text-align: center;
    }
    """

    def __init__(self, state: "ConfigState", **kwargs):
        """Initialize the top-down export screen.

        Args:
            state: Shared configuration state.
            **kwargs: Additional arguments for Container.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the top-down export screen layout."""
        with Container(classes="dual-preview"):
            # Centroid config
            with Container(classes="preview-pane"):
                yield Static("Centroid Model Config", classes="preview-title")
                yield TextArea(
                    id="td-centroid-preview",
                    read_only=True,
                    language="yaml",
                )

            # Centered instance config
            with Container(classes="preview-pane"):
                yield Static("Centered Instance Model Config", classes="preview-title")
                yield TextArea(
                    id="td-ci-preview",
                    read_only=True,
                    language="yaml",
                )

        with Horizontal(classes="button-row"):
            yield Button("Refresh", id="td-refresh-btn")
            yield Button("Save Both", id="td-save-btn", variant="primary")

        yield Static("", id="td-export-status")

    def update_preview(self) -> None:
        """Update both preview panes."""
        try:
            self.query_one("#td-centroid-preview", TextArea).text = (
                self._state.to_centroid_yaml()
            )
            self.query_one("#td-ci-preview", TextArea).text = (
                self._state.to_centered_instance_yaml()
            )
        except Exception:
            pass

    @on(Button.Pressed, "#td-refresh-btn")
    def handle_refresh(self) -> None:
        """Handle refresh button press."""
        self.update_preview()

    @on(Button.Pressed, "#td-save-btn")
    def handle_save(self) -> None:
        """Handle save button press."""
        try:
            base_path = str(self._state.slp_path.parent / self._state.slp_path.stem)
            centroid_path, ci_path = self._state.save_dual(base_path)

            status = self.query_one("#td-export-status", Static)
            status.update(f"[green]✓ Saved: {centroid_path}, {ci_path}[/green]")
        except Exception as e:
            status = self.query_one("#td-export-status", Static)
            status.update(f"[red]Error: {e}[/red]")

    def on_mount(self) -> None:
        """Initialize previews on mount."""
        self.update_preview()
