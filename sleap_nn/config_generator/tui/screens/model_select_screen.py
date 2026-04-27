"""Model selection screen for the config generator TUI.

Step 2: Select model type with smart recommendations.
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, Label, RadioButton, RadioSet, Static
from textual.widget import Widget

from sleap_nn.config_generator.tui.state import ConfigState
from sleap_nn.config_generator.recommender import PipelineType


class ModelTypeOption(Static):
    """Widget representing a model type option."""

    class Selected(Message):
        """Message sent when this option is selected."""

        def __init__(self, pipeline_type: PipelineType):
            """Initialize with selected pipeline type."""
            super().__init__()
            self.pipeline_type = pipeline_type

    def __init__(
        self,
        pipeline_type: PipelineType,
        title: str,
        description: str,
        details: str = "",
        is_recommended: bool = False,
        option_disabled: bool = False,
        disabled_reason: str = "",
        is_two_stage: bool = False,
        **kwargs,
    ):
        """Initialize the model type option."""
        super().__init__(**kwargs)
        self.pipeline_type = pipeline_type
        self.title = title
        self.description = description
        self.details = details
        self.is_recommended = is_recommended
        self.option_disabled = option_disabled
        self.disabled_reason = disabled_reason
        self.is_two_stage = is_two_stage
        self._selected = False

    def on_click(self) -> None:
        """Handle click on this option."""
        if not self.option_disabled:
            self.post_message(self.Selected(self.pipeline_type))

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
        if self.option_disabled:
            indicator = "[dim][ ][/dim]"
            title_style = "dim strike"
            desc_style = "dim"
            details_style = "dim"
        elif self._selected:
            indicator = "[bold green][✓][/bold green]"
            title_style = "bold green"
            desc_style = ""
            details_style = "dim cyan"
        else:
            indicator = "[ ]"
            title_style = "bold"
            desc_style = ""
            details_style = "dim cyan"

        lines = [f"{indicator} [{title_style}]{self.title}[/{title_style}]"]

        # Add badges
        badges = []
        if self.is_recommended:
            badges.append("[green]Recommended[/green]")
        if self.is_two_stage:
            badges.append("[yellow]2-Stage[/yellow]")

        if badges:
            lines[0] += "  " + " ".join(badges)

        # Description
        if desc_style:
            lines.append(f"    [{desc_style}]{self.description}[/{desc_style}]")
        else:
            lines.append(f"    {self.description}")

        # Details (best for / when to use)
        if self.details:
            lines.append(f"    [{details_style}]{self.details}[/{details_style}]")

        if self.option_disabled and self.disabled_reason:
            lines.append(f"    [red italic]{self.disabled_reason}[/red italic]")

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

    #pipeline-info-box {
        background: $primary-darken-3;
        border: solid $primary;
        padding: 1;
        margin-top: 1;
        min-height: 3;
    }
    """

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize the model select screen.

        Args:
            state: ConfigState with dataset statistics.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._state = state
        self._selected_type: Optional[PipelineType] = None
        # Restore identity state from pipeline type
        self._use_identity = False
        if state and state._pipeline:
            self._use_identity = "multi_class" in str(state._pipeline)

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical(id="model-container"):
            yield Label(
                "[bold]Step 2: Select Model Type[/bold]", classes="section-title"
            )

            yield Label("Choose the model architecture:", classes="hint")

            # Model type options
            with Vertical(id="model-options"):
                yield from self._create_model_options()

            # Pipeline info box (shows details about selected pipeline)
            with Container(id="pipeline-info-box"):
                yield Static(id="pipeline-info")

            # Identity tracking option - restore from state
            identity_enabled = False
            if self._state and self._state._pipeline:
                identity_enabled = "multi_class" in str(self._state._pipeline)

            with Horizontal(id="identity-option"):
                yield Checkbox(
                    "Enable identity tracking",
                    id="identity-checkbox",
                    value=identity_enabled,
                )
                yield Label("(requires track annotations in your data)", classes="hint")

    def _create_model_options(self):
        """Create model type option widgets."""
        if not self._state:
            return

        stats = self._state.stats
        rec = self._state.recommendation

        # Single Instance
        single_disabled = stats.max_instances_per_frame > 1
        yield ModelTypeOption(
            pipeline_type="single_instance",
            title="Single Instance",
            description="For videos with exactly one animal per frame.",
            details="Best for: Single-animal experiments, isolated subjects.",
            is_recommended=rec.pipeline.recommended == "single_instance",
            option_disabled=single_disabled,
            disabled_reason=(
                "Multiple instances detected in data" if single_disabled else ""
            ),
            classes="model-option",
            id="opt-single",
        )

        # Top-Down
        is_topdown_rec = rec.pipeline.recommended in [
            "centroid",
            "centered_instance",
            "multi_class_topdown",
        ]
        yield ModelTypeOption(
            pipeline_type="centroid",
            title="Top-Down",
            description="Detect centroids first, then find keypoints in cropped regions.",
            details="Best for: Multiple well-separated animals that are small relative to frame.",
            is_recommended=is_topdown_rec,
            is_two_stage=True,
            classes="model-option",
            id="opt-topdown",
        )

        # Bottom-Up
        bottomup_disabled = stats.num_edges == 0
        is_bottomup_rec = rec.pipeline.recommended in [
            "bottomup",
            "multi_class_bottomup",
        ]
        yield ModelTypeOption(
            pipeline_type="bottomup",
            title="Bottom-Up",
            description="Multi-animal detection using Part Affinity Fields (PAFs).",
            details="Best for: Crowded/overlapping animals. Detects all parts, then groups into instances.",
            is_recommended=is_bottomup_rec,
            option_disabled=bottomup_disabled,
            disabled_reason=(
                "Requires skeleton edges for Part Affinity Fields"
                if bottomup_disabled
                else ""
            ),
            classes="model-option",
            id="opt-bottomup",
        )

    def on_mount(self) -> None:
        """Handle mount - preselect recommended option."""
        selected_type = None

        if self._state and self._state._pipeline:
            selected_type = self._state._pipeline
        elif self._state:
            rec = self._state.recommendation
            # Map to base type (not multi-class variant)
            base_type = rec.pipeline.recommended
            if base_type in ["multi_class_bottomup", "multi_class_topdown"]:
                base_type = "bottomup" if "bottomup" in base_type else "centroid"
            selected_type = base_type

        if selected_type:
            self._select_type(selected_type)

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

        # Update pipeline info box
        self._update_pipeline_info(pipeline_type)

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

            # Update input scale based on pipeline type (web app behavior)
            # Top-down centroid uses lower scale (0.5), others use full scale (1.0)
            is_topdown = self._state._pipeline in [
                "centroid",
                "centered_instance",
                "multi_class_topdown",
            ]
            if is_topdown:
                self._state._input_scale = 0.5  # Lower scale for centroid
                self._state._sigma = 5.0  # Larger sigma for centroid
                self._state._output_stride = 2
            else:
                self._state._input_scale = 1.0  # Full scale for other models

            # Floor max_stride by RF coverage at the new scale, but DON'T
            # downgrade below what the bucket recommendation gave on SLP load
            # (which uses scale=1.0 — matches web app's setDefaultParameters).
            base_stride = 32 if "large_rf" in self._state._backbone else 16
            scaled_max_animal_size = (
                self._state.stats.max_bbox_size * self._state._input_scale
            )
            coverage_stride = self._state._compute_max_stride_for_animal_size(
                scaled_max_animal_size
            )
            self._state._max_stride = max(
                base_stride, coverage_stride, self._state._max_stride
            )

    def _update_pipeline_info(self, pipeline_type: PipelineType) -> None:
        """Update the pipeline info box based on selected type."""
        info_widget = self.query_one("#pipeline-info", Static)

        if pipeline_type == "single_instance":
            info_widget.update(
                "[bold cyan]Single Instance Pipeline[/bold cyan]\n"
                "Trains one model that detects all keypoints in full-frame images.\n"
                "You'll get [bold]1 YAML config[/bold] file."
            )
        elif pipeline_type == "centroid":
            info_widget.update(
                "[bold cyan]Top-Down Pipeline[/bold cyan]\n"
                "This requires training [bold]two models[/bold] sequentially:\n"
                "  [yellow]1. Centroid Model[/yellow] - Detects animal centers in full images\n"
                "  [yellow]2. Centered Instance Model[/yellow] - Detects keypoints in cropped regions\n"
                "You'll configure both models and get [bold]2 YAML configs[/bold]."
            )
        elif pipeline_type == "bottomup":
            info_widget.update(
                "[bold cyan]Bottom-Up Pipeline[/bold cyan]\n"
                "Trains one model that detects all keypoints and uses Part Affinity Fields\n"
                "to group them into animal instances. You'll get [bold]1 YAML config[/bold] file."
            )

    @on(Checkbox.Changed, "#identity-checkbox")
    def handle_identity_change(self, event: Checkbox.Changed) -> None:
        """Handle identity checkbox change."""
        self._use_identity = event.value
        # Re-apply selection with identity consideration
        if self._selected_type:
            self._select_type(self._selected_type)

    @on(ModelTypeOption.Selected)
    def handle_model_selected(self, event: ModelTypeOption.Selected) -> None:
        """Handle model type selection."""
        self._select_type(event.pipeline_type)
