"""Model selection screen for the config generator TUI.

Step 2: Select model type with smart recommendations.
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Button, Checkbox, Label, RadioButton, RadioSet, Static
from textual.widget import Widget

from sleap_nn.config_generator.tui.state import ConfigState
from sleap_nn.config_generator.recommender import PipelineType


class ModelTypeOption(Static):
    """Widget representing a model type option."""

    def __init__(
        self,
        pipeline_type: PipelineType,
        title: str,
        description: str,
        is_recommended: bool = False,
        is_disabled: bool = False,
        disabled_reason: str = "",
        is_two_stage: bool = False,
        **kwargs
    ):
        """Initialize the model type option."""
        super().__init__(**kwargs)
        self.pipeline_type = pipeline_type
        self.title = title
        self.description = description
        self.is_recommended = is_recommended
        self.is_disabled = is_disabled
        self.disabled_reason = disabled_reason
        self.is_two_stage = is_two_stage
        self._selected = False

    @property
    def selected(self) -> bool:
        """Check if this option is selected."""
        return self._selected

    @selected.setter
    def selected(self, value: bool) -> None:
        """Set selected state."""
        self._selected = value
        self.refresh()

    def render(self) -> str:
        """Render the model type option."""
        if self.is_disabled:
            indicator = "[dim][ ][/dim]"
            title_style = "dim strike"
            desc_style = "dim"
        elif self._selected:
            indicator = "[bold green][\u2713][/bold green]"
            title_style = "bold green"
            desc_style = ""
        else:
            indicator = "[ ]"
            title_style = "bold"
            desc_style = ""

        lines = [f"{indicator} [{title_style}]{self.title}[/{title_style}]"]

        # Add badges
        badges = []
        if self.is_recommended:
            badges.append("[green]Recommended[/green]")
        if self.is_two_stage:
            badges.append("[yellow]2-Stage[/yellow]")

        if badges:
            lines[0] += "  " + " ".join(badges)

        lines.append(f"    [{desc_style}]{self.description}[/{desc_style}]")

        if self.is_disabled and self.disabled_reason:
            lines.append(f"    [dim italic]{self.disabled_reason}[/dim italic]")

        return "\n".join(lines)


class ModelSelectScreen(Widget):
    """Screen for selecting model type."""

    DEFAULT_CSS = """
    ModelSelectScreen {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #model-container {
        width: 100%;
        height: auto;
    }

    #recommendation-box {
        background: $success-darken-3;
        border: solid $success;
        padding: 1;
        margin: 1 0;
    }

    #model-options {
        width: 100%;
        height: auto;
        margin: 1 0;
    }

    .model-option {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $panel;
        border: solid $primary;
    }

    .model-option:hover {
        background: $panel-lighten-1;
    }

    .model-option.selected {
        border: solid $success;
        background: $success-darken-3;
    }

    .model-option.disabled {
        background: $surface-darken-1;
    }

    #identity-option {
        margin-top: 2;
        padding: 1;
    }
    """

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize the model select screen.

        Args:
            state: ConfigState with dataset statistics.
        """
        super().__init__(**kwargs)
        self._state = state
        self._selected_type: Optional[PipelineType] = None
        self._use_identity = False

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical(id="model-container"):
            yield Label("[bold]Step 2: Select Model Type[/bold]", classes="section-title")

            # Recommendation box
            if self._state:
                rec = self._state.recommendation
                with Container(id="recommendation-box"):
                    yield Static(
                        f"[bold]Recommendation:[/bold] {rec.pipeline.recommended.upper()}\n"
                        f"[dim]{rec.pipeline.reason}[/dim]"
                    )

            yield Label("Choose the model architecture:", classes="hint")

            # Model type options
            with Vertical(id="model-options"):
                yield from self._create_model_options()

            # Identity tracking option
            with Horizontal(id="identity-option"):
                yield Checkbox(
                    "Enable identity tracking",
                    id="identity-checkbox",
                    value=False,
                )
                yield Label(
                    "(requires track annotations in your data)",
                    classes="hint"
                )

    def _create_model_options(self):
        """Create model type option widgets."""
        if not self._state:
            return

        stats = self._state.stats
        rec = self._state.recommendation

        # Single Instance
        is_disabled = stats.max_instances_per_frame > 1
        yield ModelTypeOption(
            pipeline_type="single_instance",
            title="Single Instance",
            description="One animal per frame - simplest and fastest",
            is_recommended=rec.pipeline.recommended == "single_instance",
            is_disabled=is_disabled,
            disabled_reason="Multiple instances detected in data" if is_disabled else "",
            classes="model-option",
            id="opt-single",
        )

        # Top-Down
        is_topdown_rec = rec.pipeline.recommended in ["centroid", "centered_instance"]
        yield ModelTypeOption(
            pipeline_type="centroid",
            title="Top-Down",
            description="Best for small, well-separated animals. Crops around each animal.",
            is_recommended=is_topdown_rec,
            is_two_stage=True,
            classes="model-option",
            id="opt-topdown",
        )

        # Bottom-Up
        no_edges = stats.num_edges == 0
        is_bottomup_rec = rec.pipeline.recommended == "bottomup"
        yield ModelTypeOption(
            pipeline_type="bottomup",
            title="Bottom-Up",
            description="Best for overlapping animals. Detects all keypoints and links them.",
            is_recommended=is_bottomup_rec,
            is_disabled=no_edges,
            disabled_reason="Requires skeleton edges for Part Affinity Fields" if no_edges else "",
            classes="model-option",
            id="opt-bottomup",
        )

    def on_mount(self) -> None:
        """Handle mount - pre-select recommended option."""
        if self._state and self._state._pipeline:
            self._select_type(self._state._pipeline)
        elif self._state:
            rec = self._state.recommendation
            # Map to base type (not multi-class variant)
            base_type = rec.pipeline.recommended
            if base_type in ["multi_class_bottomup", "multi_class_topdown"]:
                base_type = "bottomup" if "bottomup" in base_type else "centroid"
            self._select_type(base_type)

        # Disable identity checkbox if no tracks
        if self._state and not self._state.stats.has_tracks:
            checkbox = self.query_one("#identity-checkbox", Checkbox)
            checkbox.disabled = True

    def _select_type(self, pipeline_type: PipelineType) -> None:
        """Select a model type."""
        self._selected_type = pipeline_type

        # Update visual selection
        for opt in self.query(".model-option"):
            if isinstance(opt, ModelTypeOption):
                opt.selected = opt.pipeline_type == pipeline_type
                if opt.selected:
                    opt.add_class("selected")
                else:
                    opt.remove_class("selected")

        # Update state
        if self._state:
            # Check if identity is enabled
            checkbox = self.query_one("#identity-checkbox", Checkbox)
            if checkbox.value:
                if pipeline_type == "centroid":
                    self._state._pipeline = "multi_class_topdown"
                elif pipeline_type == "bottomup":
                    self._state._pipeline = "multi_class_bottomup"
                else:
                    self._state._pipeline = pipeline_type
            else:
                self._state._pipeline = pipeline_type

    @on(Checkbox.Changed, "#identity-checkbox")
    def handle_identity_change(self, event: Checkbox.Changed) -> None:
        """Handle identity checkbox change."""
        self._use_identity = event.value
        # Re-apply selection with identity consideration
        if self._selected_type:
            self._select_type(self._selected_type)

    def on_click(self, event) -> None:
        """Handle click on model options."""
        # Find which option was clicked
        for opt in self.query(".model-option"):
            if isinstance(opt, ModelTypeOption):
                # Check if click was within this widget
                if opt.region.contains(event.x, event.y):
                    if not opt.is_disabled:
                        self._select_type(opt.pipeline_type)
                    break
