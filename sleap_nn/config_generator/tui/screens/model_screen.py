"""Model configuration screen.

Provides UI for configuring model architecture and head settings.
"""

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Input,
    RadioButton,
    RadioSet,
    Rule,
    Select,
    Static,
)

from sleap_nn.config_generator.tui.widgets import (
    Collapsible,
    GuideBox,
    InfoBox,
    LabeledSlider,
    ModelInfoDisplay,
    SigmaVisualization,
    TipBox,
)

if TYPE_CHECKING:
    from sleap_nn.config_generator.tui.state import ConfigState


class ModelScreen(VerticalScroll):
    """Model configuration screen component.

    Provides controls for:
    - Pipeline type selection
    - Backbone architecture
    - Model hyperparameters (max_stride, filters, etc.)
    - Head configuration (sigma, output_stride)
    - Anchor point selection (for top-down)
    - PAF settings (for bottom-up)
    - Multi-class settings
    - Pretrained weights
    """

    DEFAULT_CSS = """
    ModelScreen {
        width: 100%;
        height: 100%;
        padding: 1;
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

    .form-input {
        width: 1fr;
    }

    .pipeline-card {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $surface-lighten-2;
    }

    .pipeline-card.selected {
        border: solid $primary;
        background: $surface-lighten-1;
    }

    .pipeline-card.recommended {
        border: solid $success;
    }
    """

    def __init__(self, state: "ConfigState", **kwargs):
        """Initialize the model screen.

        Args:
            state: Shared configuration state.
            **kwargs: Additional arguments for Container.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the model screen layout."""
        # Pipeline Selection
        yield from self._compose_pipeline_section()

        yield Rule()

        # Backbone Architecture
        yield from self._compose_backbone_section()

        yield Rule()

        # Model Architecture Settings
        yield from self._compose_architecture_section()

        yield Rule()

        # Head Configuration
        yield from self._compose_head_section()

        # PAF Settings (for bottom-up)
        yield from self._compose_paf_section()

        # Multi-class Settings
        yield from self._compose_multiclass_section()

        # Anchor Point (for top-down)
        yield from self._compose_anchor_section()

        # Centered Instance Config (for top-down)
        yield from self._compose_centered_instance_section()

        yield Rule()

        # Pretrained Weights
        yield from self._compose_pretrained_section()

    def _compose_pipeline_section(self) -> ComposeResult:
        """Compose pipeline type selection section."""
        yield Static("Pipeline Type", classes="section-title")

        recommended = self._state.recommendation.pipeline.recommended

        with RadioSet(id="pipeline-select"):
            # Single Instance
            is_rec = recommended == "single_instance"
            label = "Single Instance" + (" (Recommended)" if is_rec else "")
            yield RadioButton(
                label,
                id="pipe-single",
                value=self._state._pipeline == "single_instance",
            )

            # Top-Down: Centroid
            is_rec = recommended == "centroid"
            label = "Top-Down: Centroid" + (" (Recommended)" if is_rec else "")
            yield RadioButton(
                label,
                id="pipe-centroid",
                value=self._state._pipeline == "centroid",
            )

            # Top-Down: Centered Instance
            is_rec = recommended == "centered_instance"
            label = "Top-Down: Centered Instance" + (" (Recommended)" if is_rec else "")
            yield RadioButton(
                label,
                id="pipe-centered",
                value=self._state._pipeline == "centered_instance",
            )

            # Bottom-Up
            is_rec = recommended == "bottomup"
            label = "Bottom-Up" + (" (Recommended)" if is_rec else "")
            yield RadioButton(
                label,
                id="pipe-bottomup",
                value=self._state._pipeline == "bottomup",
            )

            # Multi-Class Bottom-Up
            is_rec = recommended == "multi_class_bottomup"
            label = "Multi-Class Bottom-Up" + (" (Recommended)" if is_rec else "")
            yield RadioButton(
                label,
                id="pipe-mc-bottomup",
                value=self._state._pipeline == "multi_class_bottomup",
            )

            # Multi-Class Top-Down
            is_rec = recommended == "multi_class_topdown"
            label = "Multi-Class Top-Down" + (" (Recommended)" if is_rec else "")
            yield RadioButton(
                label,
                id="pipe-mc-topdown",
                value=self._state._pipeline == "multi_class_topdown",
            )

        yield InfoBox(
            f"{self._state.recommendation.pipeline.reason}",
            id="pipeline-reason",
        )

    def _compose_backbone_section(self) -> ComposeResult:
        """Compose backbone architecture selection section."""
        yield Static("Backbone Architecture", classes="section-title")

        with RadioSet(id="backbone-select"):
            yield RadioButton(
                "UNet (Medium RF)",
                id="bb-unet-medium",
                value=self._state._backbone == "unet_medium_rf",
            )
            yield RadioButton(
                "UNet (Large RF)",
                id="bb-unet-large",
                value=self._state._backbone == "unet_large_rf",
            )
            yield RadioButton(
                "ConvNeXt (Tiny)",
                id="bb-convnext-tiny",
                value=self._state._backbone == "convnext_tiny",
            )
            yield RadioButton(
                "ConvNeXt (Small)",
                id="bb-convnext-small",
                value=self._state._backbone == "convnext_small",
            )
            yield RadioButton(
                "SwinT (Tiny)",
                id="bb-swint-tiny",
                value=self._state._backbone == "swint_tiny",
            )
            yield RadioButton(
                "SwinT (Small)",
                id="bb-swint-small",
                value=self._state._backbone == "swint_small",
            )

        yield TipBox(
            "UNet: Fast, lightweight, good for most cases.\n"
            "ConvNeXt/SwinT: Pretrained on ImageNet, better for complex scenes but requires RGB input.",
            title=None,
        )

    def _compose_architecture_section(self) -> ComposeResult:
        """Compose model architecture settings section."""
        yield Static("Architecture Settings", classes="section-title")

        # Max Stride
        with Horizontal(classes="form-row"):
            yield Static("Max Stride:", classes="form-label")
            with RadioSet(id="max-stride-select"):
                yield RadioButton(
                    "8", id="stride-8", value=self._state._max_stride == 8
                )
                yield RadioButton(
                    "16", id="stride-16", value=self._state._max_stride == 16
                )
                yield RadioButton(
                    "32", id="stride-32", value=self._state._max_stride == 32
                )
                yield RadioButton(
                    "64", id="stride-64", value=self._state._max_stride == 64
                )

        # Base Filters
        with Horizontal(classes="form-row"):
            yield Static("Base Filters:", classes="form-label")
            yield Select(
                [
                    ("16", 16),
                    ("24", 24),
                    ("32", 32),
                    ("48", 48),
                    ("64", 64),
                ],
                value=self._state._filters,
                id="filters-select",
            )

        # Filters Rate
        with Horizontal(classes="form-row"):
            yield Static("Filters Rate:", classes="form-label")
            yield Input(
                value=str(self._state._filters_rate),
                type="number",
                id="filters-rate-input",
            )

        # Model Info Display
        yield ModelInfoDisplay(
            params=self._state.model_params_estimate,
            receptive_field=self._state.receptive_field,
            encoder_blocks=self._state.encoder_blocks,
            decoder_blocks=self._state.encoder_blocks,
            id="model-info",
        )

        yield GuideBox(
            title="Stride & Receptive Field Guide",
            sections={
                "Max Stride 8": "RF ~36px - Good for small features, more parameters",
                "Max Stride 16": "RF ~76px - Balanced choice for most use cases",
                "Max Stride 32": "RF ~156px - Captures large context, fewer parameters",
                "Max Stride 64": "RF ~316px - Maximum context, smallest model",
            },
        )

    def _compose_head_section(self) -> ComposeResult:
        """Compose head configuration section."""
        yield Static("Confidence Map Settings", classes="section-title")

        # Sigma
        yield LabeledSlider(
            label="Sigma",
            value=self._state._sigma,
            min_value=1.0,
            max_value=15.0,
            step=0.5,
            format_str="{:.1f}",
            unit="px",
            id="sigma-slider",
        )

        # Sigma Visualization
        yield SigmaVisualization(
            sigma=self._state._sigma,
            output_stride=self._state._output_stride,
            id="sigma-viz",
        )

        # Output Stride
        with Horizontal(classes="form-row"):
            yield Static("Output Stride:", classes="form-label")
            yield Select(
                [
                    ("1 (full resolution)", 1),
                    ("2 (half resolution)", 2),
                    ("4 (quarter resolution)", 4),
                    ("8 (1/8 resolution)", 8),
                ],
                value=self._state._output_stride,
                id="output-stride-select",
            )

        yield TipBox(
            "Sigma: Gaussian spread for confidence maps. Larger = easier to learn but less precise.\n"
            "Output Stride: Lower values give more precise predictions but use more memory.",
            title=None,
        )

    def _compose_paf_section(self) -> ComposeResult:
        """Compose PAF settings section (for bottom-up models)."""
        # Create a collapsible section that's shown/hidden based on pipeline
        with Collapsible(
            title="Part Affinity Fields (PAFs)",
            expanded=False,
            id="paf-section",
        ):
            yield Static(
                "PAF settings are used for bottom-up multi-instance pose estimation.",
                classes="description-text",
            )

            # PAF Sigma
            yield LabeledSlider(
                label="PAF Sigma",
                value=self._state._paf_config.sigma,
                min_value=5.0,
                max_value=30.0,
                step=1.0,
                format_str="{:.0f}",
                unit="px",
                id="paf-sigma-slider",
            )

            # PAF Output Stride
            with Horizontal(classes="form-row"):
                yield Static("PAF Output Stride:", classes="form-label")
                yield Select(
                    [
                        ("2", 2),
                        ("4", 4),
                        ("8", 8),
                    ],
                    value=self._state._paf_config.output_stride,
                    id="paf-stride-select",
                )

            # Loss Weights
            with Horizontal(classes="form-row"):
                yield Static("Confmap Loss Weight:", classes="form-label")
                yield Input(
                    value="1.0",
                    type="number",
                    id="confmap-loss-weight",
                )

            with Horizontal(classes="form-row"):
                yield Static("PAF Loss Weight:", classes="form-label")
                yield Input(
                    value=str(self._state._paf_config.loss_weight),
                    type="number",
                    id="paf-loss-weight",
                )

    def _compose_multiclass_section(self) -> ComposeResult:
        """Compose multi-class settings section."""
        with Collapsible(
            title="Multi-Class Settings",
            expanded=False,
            id="multiclass-section",
        ):
            yield Static(
                "Settings for multi-class (identity) models.",
                classes="description-text",
            )

            # FC Layers
            with Horizontal(classes="form-row"):
                yield Static("FC Layers:", classes="form-label")
                yield Input(
                    value=str(self._state._class_vector_config.num_fc_layers),
                    type="integer",
                    id="fc-layers-input",
                )

            # FC Units
            with Horizontal(classes="form-row"):
                yield Static("FC Units:", classes="form-label")
                yield Input(
                    value=str(self._state._class_vector_config.num_fc_units),
                    type="integer",
                    id="fc-units-input",
                )

            # Class Loss Weight
            with Horizontal(classes="form-row"):
                yield Static("Class Loss Weight:", classes="form-label")
                yield Input(
                    value=str(self._state._class_vector_config.loss_weight),
                    type="number",
                    id="class-loss-weight",
                )

    def _compose_anchor_section(self) -> ComposeResult:
        """Compose anchor point selection section (for top-down)."""
        with Collapsible(
            title="Anchor Point (Top-Down)",
            expanded=False,
            id="anchor-section",
        ):
            yield Static(
                "Select the body part to use as the anchor for centroid detection.",
                classes="description-text",
            )

            # Get node names for dropdown
            node_options = [("Auto (centroid)", None)]
            for name in self._state.skeleton_nodes:
                node_options.append((name, name))

            with Horizontal(classes="form-row"):
                yield Static("Anchor Part:", classes="form-label")
                yield Select(
                    node_options,
                    value=self._state._anchor_part,
                    id="anchor-select",
                )

            # Crop Size (for centered instance)
            with Horizontal(classes="form-row"):
                yield Static("Crop Size:", classes="form-label")
                yield Input(
                    value=str(self._state._crop_size or ""),
                    placeholder="Auto",
                    type="integer",
                    id="crop-size-input",
                )

            yield TipBox(
                "Anchor Point: The body part used to center crops. Choose a stable, "
                "easily detectable point like the head or thorax.\n"
                "Crop Size: Size of the square crop around each instance. Leave empty for auto.",
                title=None,
            )

    def _compose_centered_instance_section(self) -> ComposeResult:
        """Compose centered instance model configuration (for top-down)."""
        with Collapsible(
            title="Centered Instance Model (Top-Down)",
            expanded=False,
            id="ci-section",
        ):
            yield InfoBox(
                "Top-down pipelines require a second model to detect keypoints "
                "within cropped regions centered on each detected centroid.",
            )

            with Horizontal(classes="form-row"):
                yield Static("CI Backbone:", classes="form-label")
                yield Select(
                    [
                        ("UNet (Medium RF)", "unet_medium_rf"),
                        ("UNet (Large RF)", "unet_large_rf"),
                    ],
                    value=self._state._ci_backbone,
                    id="ci-backbone-select",
                )

            yield LabeledSlider(
                label="CI Sigma",
                value=self._state._ci_sigma,
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                format_str="{:.1f}",
                id="ci-sigma-slider",
            )

            with Horizontal(classes="form-row"):
                yield Static("CI Output Stride:", classes="form-label")
                yield Select(
                    [
                        ("1 (full resolution)", 1),
                        ("2 (half resolution)", 2),
                        ("4 (quarter resolution)", 4),
                    ],
                    value=self._state._ci_output_stride,
                    id="ci-output-stride-select",
                )

    def _compose_pretrained_section(self) -> ComposeResult:
        """Compose pretrained weights section."""
        with Collapsible(
            title="Pretrained Weights",
            expanded=False,
            id="pretrained-section",
        ):
            yield Static(
                "Load weights from a previously trained model.",
                classes="description-text",
            )

            with Horizontal(classes="form-row"):
                yield Static("Backbone Weights:", classes="form-label")
                yield Input(
                    value=self._state._pretrained_backbone,
                    placeholder="Path to .ckpt file",
                    id="pretrained-backbone-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Head Weights:", classes="form-label")
                yield Input(
                    value=self._state._pretrained_head,
                    placeholder="Path to .ckpt file",
                    id="pretrained-head-input",
                )

    def _update_model_info(self) -> None:
        """Update the model info display."""
        try:
            model_info = self.query_one("#model-info", ModelInfoDisplay)
            model_info.update_info(
                params=self._state.model_params_estimate,
                receptive_field=self._state.receptive_field,
                encoder_blocks=self._state.encoder_blocks,
                decoder_blocks=self._state.encoder_blocks,
            )
        except Exception:
            pass

    def _update_sigma_viz(self) -> None:
        """Update the sigma visualization."""
        try:
            sigma_viz = self.query_one("#sigma-viz", SigmaVisualization)
            sigma_viz.update_sigma(self._state._sigma, self._state._output_stride)
        except Exception:
            pass

    def _update_section_visibility(self) -> None:
        """Update section visibility based on pipeline type."""
        try:
            # PAF section - visible for bottom-up
            paf_section = self.query_one("#paf-section", Collapsible)
            if self._state.is_bottomup:
                paf_section.expand()
            else:
                paf_section.collapse()

            # Multi-class section - visible for multi-class
            mc_section = self.query_one("#multiclass-section", Collapsible)
            if self._state.is_multiclass:
                mc_section.expand()
            else:
                mc_section.collapse()

            # Anchor section - visible for top-down
            anchor_section = self.query_one("#anchor-section", Collapsible)
            if self._state.is_topdown:
                anchor_section.expand()
            else:
                anchor_section.collapse()
        except Exception:
            pass

    @on(RadioSet.Changed, "#pipeline-select")
    def handle_pipeline_change(self, event: RadioSet.Changed) -> None:
        """Handle pipeline selection changes."""
        pipeline_map = {
            "pipe-single": "single_instance",
            "pipe-centroid": "centroid",
            "pipe-centered": "centered_instance",
            "pipe-bottomup": "bottomup",
            "pipe-mc-bottomup": "multi_class_bottomup",
            "pipe-mc-topdown": "multi_class_topdown",
        }

        if event.pressed.id in pipeline_map:
            self._state._pipeline = pipeline_map[event.pressed.id]
            self._update_section_visibility()
            self._state.notify_observers()

    @on(RadioSet.Changed, "#backbone-select")
    def handle_backbone_change(self, event: RadioSet.Changed) -> None:
        """Handle backbone selection changes."""
        backbone_map = {
            "bb-unet-medium": "unet_medium_rf",
            "bb-unet-large": "unet_large_rf",
            "bb-convnext-tiny": "convnext_tiny",
            "bb-convnext-small": "convnext_small",
            "bb-swint-tiny": "swint_tiny",
            "bb-swint-small": "swint_small",
        }

        if event.pressed.id in backbone_map:
            self._state._backbone = backbone_map[event.pressed.id]

            # Update related parameters based on backbone
            if "large_rf" in self._state._backbone:
                self._state._max_stride = 32
                self._state._filters = 24
                self._state._filters_rate = 1.5
            elif "unet" in self._state._backbone:
                self._state._max_stride = 16
                self._state._filters = 32
                self._state._filters_rate = 2.0

            self._update_model_info()
            self._state.notify_observers()

    @on(RadioSet.Changed, "#max-stride-select")
    def handle_max_stride_change(self, event: RadioSet.Changed) -> None:
        """Handle max stride selection changes."""
        stride_map = {
            "stride-8": 8,
            "stride-16": 16,
            "stride-32": 32,
            "stride-64": 64,
        }

        if event.pressed.id in stride_map:
            self._state._max_stride = stride_map[event.pressed.id]
            self._update_model_info()
            self._state.notify_observers()

    @on(Select.Changed, "#filters-select")
    def handle_filters_change(self, event: Select.Changed) -> None:
        """Handle filters selection changes."""
        self._state._filters = event.value
        self._update_model_info()
        self._state.notify_observers()

    @on(Input.Changed, "#filters-rate-input")
    def handle_filters_rate_change(self, event: Input.Changed) -> None:
        """Handle filters rate input changes."""
        try:
            self._state._filters_rate = float(event.value)
            self._update_model_info()
            self._state.notify_observers()
        except ValueError:
            pass

    @on(LabeledSlider.Changed, "#sigma-slider")
    def handle_sigma_change(self, event: LabeledSlider.Changed) -> None:
        """Handle sigma slider changes."""
        self._state._sigma = event.value
        self._update_sigma_viz()
        self._state.notify_observers()

    @on(Select.Changed, "#output-stride-select")
    def handle_output_stride_change(self, event: Select.Changed) -> None:
        """Handle output stride selection changes."""
        self._state._output_stride = event.value
        self._update_sigma_viz()
        self._state.notify_observers()

    @on(LabeledSlider.Changed, "#paf-sigma-slider")
    def handle_paf_sigma_change(self, event: LabeledSlider.Changed) -> None:
        """Handle PAF sigma slider changes."""
        self._state._paf_config.sigma = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#paf-stride-select")
    def handle_paf_stride_change(self, event: Select.Changed) -> None:
        """Handle PAF stride selection changes."""
        self._state._paf_config.output_stride = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#paf-loss-weight")
    def handle_paf_loss_weight_change(self, event: Input.Changed) -> None:
        """Handle PAF loss weight input changes."""
        try:
            self._state._paf_config.loss_weight = float(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#fc-layers-input")
    def handle_fc_layers_change(self, event: Input.Changed) -> None:
        """Handle FC layers input changes."""
        try:
            self._state._class_vector_config.num_fc_layers = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#fc-units-input")
    def handle_fc_units_change(self, event: Input.Changed) -> None:
        """Handle FC units input changes."""
        try:
            self._state._class_vector_config.num_fc_units = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Select.Changed, "#anchor-select")
    def handle_anchor_change(self, event: Select.Changed) -> None:
        """Handle anchor point selection changes."""
        self._state._anchor_part = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#crop-size-input")
    def handle_crop_size_change(self, event: Input.Changed) -> None:
        """Handle crop size input changes."""
        try:
            self._state._crop_size = int(event.value) if event.value else None
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#pretrained-backbone-input")
    def handle_pretrained_backbone_change(self, event: Input.Changed) -> None:
        """Handle pretrained backbone path changes."""
        self._state._pretrained_backbone = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#pretrained-head-input")
    def handle_pretrained_head_change(self, event: Input.Changed) -> None:
        """Handle pretrained head path changes."""
        self._state._pretrained_head = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#ci-backbone-select")
    def handle_ci_backbone_change(self, event: Select.Changed) -> None:
        """Handle centered instance backbone changes."""
        self._state._ci_backbone = event.value
        self._state.notify_observers()

    @on(LabeledSlider.Changed, "#ci-sigma-slider")
    def handle_ci_sigma_change(self, event: LabeledSlider.Changed) -> None:
        """Handle centered instance sigma changes."""
        self._state._ci_sigma = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#ci-output-stride-select")
    def handle_ci_output_stride_change(self, event: Select.Changed) -> None:
        """Handle centered instance output stride changes."""
        self._state._ci_output_stride = event.value
        self._state.notify_observers()

    def _update_ui_from_state(self) -> None:
        """Update all UI elements from current state."""
        try:
            # Backbone
            backbone_map = {
                "unet_medium_rf": "bb-unet-medium",
                "unet_large_rf": "bb-unet-large",
                "convnext_tiny": "bb-convnext-tiny",
                "convnext_small": "bb-convnext-small",
                "swint_tiny": "bb-swint-tiny",
                "swint_small": "bb-swint-small",
            }
            btn_id = backbone_map.get(self._state._backbone)
            if btn_id:
                self.query_one(f"#{btn_id}", RadioButton).value = True

            # Max stride
            stride_map = {
                8: "stride-8",
                16: "stride-16",
                32: "stride-32",
                64: "stride-64",
            }
            btn_id = stride_map.get(self._state._max_stride)
            if btn_id:
                self.query_one(f"#{btn_id}", RadioButton).value = True

            # Filters
            self.query_one("#filters-select", Select).value = self._state._filters

            # Sigma
            self.query_one("#sigma-slider", LabeledSlider).set_value(self._state._sigma)

            # Output stride
            self.query_one("#output-stride-select", Select).value = (
                self._state._output_stride
            )

            # Update displays
            self._update_model_info()
            self._update_sigma_viz()
            self._update_section_visibility()

        except Exception:
            pass
