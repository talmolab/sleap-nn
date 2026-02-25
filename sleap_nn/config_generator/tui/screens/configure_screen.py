"""Configuration screen for the config generator TUI.

Step 3: Configure training parameters with comprehensive parameter support.
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widget import Widget

from sleap_nn.config_generator.tui.state import (
    ConfigState,
    DataPipelineType,
    SchedulerType,
)


class ConfigureScreen(Widget):
    """Screen for configuring training parameters."""

    DEFAULT_CSS = """
    ConfigureScreen {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #config-container {
        width: 100%;
        height: 100%;
    }

    #config-scroll {
        width: 100%;
        height: 1fr;
        padding: 0 1;
    }

    /* Section boxes with consistent spacing */
    .section-box {
        background: $panel;
        border: solid $primary;
        padding: 1 2;
        margin-bottom: 1;
        width: 100%;
        height: auto;
        layout: vertical;
    }

    .section-header {
        text-style: bold;
        padding-bottom: 1;
        border-bottom: solid $primary;
        margin-bottom: 1;
        width: 100%;
    }

    .subsection-header {
        text-style: bold;
        color: $text-muted;
        height: 2;
        margin-top: 1;
    }

    /* Parameter rows */
    .param-row {
        height: 3;
        margin-bottom: 0;
        width: 100%;
    }

    .param-label {
        width: 18;
        text-align: right;
        padding-right: 1;
    }

    .param-input {
        width: 12;
    }

    .param-input-wide {
        width: 28;
    }

    .param-hint {
        width: 1fr;
        padding-left: 1;
        color: $text-muted;
        text-style: italic;
    }

    /* Memory estimate */
    #memory-estimate {
        background: $panel;
        border: solid $warning;
        padding: 1 2;
        margin-top: 1;
    }

    /* Collapsible sections */
    .collapsible-section {
        margin-bottom: 1;
        width: 100%;
    }

    Collapsible {
        padding: 0;
        margin-bottom: 1;
    }

    Collapsible > Contents {
        padding: 1 2;
    }

    /* Tabs styling */
    #topdown-tabs {
        width: 100%;
        height: auto;
    }

    TabPane {
        padding: 1;
    }

    /* Model-specific sections */
    .model-specific-section {
        background: $warning-darken-3;
        border: solid $warning;
        padding: 1 2;
        margin-bottom: 1;
    }

    .hidden {
        display: none;
    }

    /* Input and Select styling */
    Input {
        width: 12;
    }

    Select {
        width: 18;
    }

    Checkbox {
        width: auto;
        padding-right: 1;
    }
    """

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize the configure screen.

        Args:
            state: ConfigState with current configuration.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical(id="config-container"):
            yield Label("[bold]Step 3: Configure Training[/bold]", classes="section-title")

            if self._state and self._state.is_topdown:
                yield Label(
                    "[yellow]Top-down model selected - configuring 2 models[/yellow]",
                    classes="hint"
                )

            with VerticalScroll(id="config-scroll"):
                # For top-down, show tabbed interface
                if self._state and self._state.is_topdown:
                    yield from self._compose_topdown_config()
                else:
                    yield from self._compose_standard_config()

                # Memory estimate
                yield from self._compose_memory_estimate()

    def _compose_standard_config(self) -> ComposeResult:
        """Compose standard (non-topdown) configuration sections."""
        # Data section - main params
        with Container(classes="section-box"):
            yield Static("[bold]Data Configuration[/bold]", classes="section-header")
            yield from self._compose_data_params()

        # Data Augmentation section
        with Container(classes="section-box"):
            yield Static("[bold]Data Augmentation[/bold]", classes="section-header")
            yield from self._compose_data_augmentation_params()

        # Additional Data Parameters (collapsible)
        with Collapsible(title="Additional Data Parameters", classes="collapsible-section"):
            yield from self._compose_additional_data_params()

        # Model section
        with Container(classes="section-box"):
            yield Static("[bold]Model Architecture[/bold]", classes="section-header")
            yield from self._compose_model_params()

        # Model-specific sections (PAF for bottom-up, class vector for multi-class)
        if self._state and self._state.is_bottomup:
            with Container(classes="model-specific-section"):
                yield Static("[bold]Part Affinity Fields (PAF)[/bold]", classes="section-header")
                yield from self._compose_paf_params()

        if self._state and self._state.is_multiclass:
            with Container(classes="model-specific-section"):
                yield Static("[bold]Identity Classification[/bold]", classes="section-header")
                yield from self._compose_class_vector_params()

        # Training section
        with Container(classes="section-box"):
            yield Static("[bold]Training[/bold]", classes="section-header")
            yield from self._compose_training_params()

        # Checkpoint section (collapsible)
        with Collapsible(title="Checkpoints & Logging", classes="collapsible-section"):
            yield from self._compose_checkpoint_params()

        # Advanced section (collapsible)
        with Collapsible(title="Advanced Settings", classes="collapsible-section"):
            yield from self._compose_advanced_params()

    def _compose_topdown_config(self) -> ComposeResult:
        """Compose top-down specific configuration with tabs."""
        with TabbedContent(id="topdown-tabs"):
            with TabPane("Shared Settings", id="shared-tab"):
                yield from self._compose_shared_topdown_params()

            with TabPane("Centroid Model", id="centroid-tab"):
                yield from self._compose_centroid_params()

            with TabPane("Instance Model", id="instance-tab"):
                yield from self._compose_instance_params()

    # ==================== DATA PARAMETERS ====================

    def _compose_data_params(self) -> ComposeResult:
        """Compose data configuration parameters (main section only)."""
        # Input Scale (prominent - main parameter)
        with Horizontal(classes="param-row"):
            yield Label("Input Scale:", classes="param-label")
            yield Input(
                value=str(self._state._input_scale if self._state else 1.0),
                id="scale-input",
                classes="param-input",
                type="number",
            )
            yield Label("Resize factor (0.125-1.0) - lower = faster training", classes="param-hint")

        # Show effective size info
        if self._state:
            orig_h, orig_w = self._state.stats.max_height, self._state.stats.max_width
            eff_h, eff_w = self._state.effective_height, self._state.effective_width
            yield Static(
                f"[dim]Original: {orig_w}×{orig_h} → Input: {eff_w}×{eff_h}[/dim]",
                classes="param-hint"
            )

    def _compose_data_augmentation_params(self) -> ComposeResult:
        """Compose data augmentation parameters section."""
        aug = self._state._augmentation if self._state else None

        # Geometric augmentations
        yield Static("[dim]Geometric[/dim]", classes="subsection-header")

        # Rotation
        with Horizontal(classes="param-row"):
            yield Checkbox(
                "Rotation",
                value=aug.rotation_enabled if aug else True,
                id="rotation-checkbox",
            )
            yield Label("±", classes="param-label")
            yield Input(
                value=str(aug.rotation_max if aug else 15.0),
                id="rotation-input",
                classes="param-input",
                type="number",
            )
            yield Label("degrees", classes="param-hint")

        # Scale
        with Horizontal(classes="param-row"):
            yield Checkbox(
                "Scale",
                value=aug.scale_enabled if aug else True,
                id="scale-checkbox",
            )
            yield Input(
                value=str(aug.scale_min if aug else 0.9),
                id="scale-min-input",
                classes="param-input",
                type="number",
            )
            yield Label("to", classes="param-hint")
            yield Input(
                value=str(aug.scale_max if aug else 1.1),
                id="scale-max-input",
                classes="param-input",
                type="number",
            )

        # Intensity augmentations
        yield Static("[dim]Intensity[/dim]", classes="subsection-header")

        # Brightness
        with Horizontal(classes="param-row"):
            yield Checkbox(
                "Brightness",
                value=aug.brightness_enabled if aug else False,
                id="brightness-checkbox",
            )
            yield Label("±", classes="param-label")
            yield Input(
                value=str(aug.brightness_limit if aug else 0.2),
                id="brightness-limit-input",
                classes="param-input",
                type="number",
            )
            yield Label("(0.2 = ±20%)", classes="param-hint")

        # Contrast
        with Horizontal(classes="param-row"):
            yield Checkbox(
                "Contrast",
                value=aug.contrast_enabled if aug else False,
                id="contrast-checkbox",
            )
            yield Label("±", classes="param-label")
            yield Input(
                value=str(aug.contrast_limit if aug else 0.2),
                id="contrast-limit-input",
                classes="param-input",
                type="number",
            )
            yield Label("(0.2 = ±20%)", classes="param-hint")

    def _compose_additional_data_params(self) -> ComposeResult:
        """Compose additional data parameters (collapsible section)."""
        # Validation Fraction
        with Horizontal(classes="param-row"):
            yield Label("Val Fraction:", classes="param-label")
            yield Input(
                value=str(self._state._validation_fraction if self._state else 0.1),
                id="val-fraction-input",
                classes="param-input",
                type="number",
            )
            yield Label("Fraction held out for validation (0.05-0.3)", classes="param-hint")

        # Input Channels
        current_channels = "grayscale"
        if self._state:
            if self._state._ensure_rgb:
                current_channels = "rgb"
            elif self._state._ensure_grayscale:
                current_channels = "grayscale"

        with Horizontal(classes="param-row"):
            yield Label("Input Channels:", classes="param-label")
            yield Select(
                [
                    ("Grayscale (1 channel)", "grayscale"),
                    ("RGB (3 channels)", "rgb"),
                ],
                value=current_channels,
                id="channels-select",
                classes="param-input",
            )
            yield Label("Image color mode", classes="param-hint")

        # Max Height / Max Width on same row conceptually
        with Horizontal(classes="param-row"):
            yield Label("Max Height:", classes="param-label")
            yield Input(
                value=str(self._state._max_height) if self._state and self._state._max_height else "",
                id="max-height-input",
                classes="param-input",
                type="integer",
                placeholder="auto",
            )
            yield Label("Max Width:", classes="param-label")
            yield Input(
                value=str(self._state._max_width) if self._state and self._state._max_width else "",
                id="max-width-input",
                classes="param-input",
                type="integer",
                placeholder="auto",
            )

        # Data Pipeline
        current_pipeline = self._state._data_pipeline.value if self._state else "torch_dataset"
        with Horizontal(classes="param-row"):
            yield Label("Data Pipeline:", classes="param-label")
            yield Select(
                [
                    ("Video (default)", "torch_dataset"),
                    ("Cache in Memory", "litdata"),
                    ("Cache to Disk", "litdata_disk"),
                ],
                value=current_pipeline,
                id="data-pipeline-select",
                classes="param-input",
            )
            yield Label("Data loading method", classes="param-hint")

    # ==================== MODEL PARAMETERS ====================

    def _compose_model_params(self) -> ComposeResult:
        """Compose model architecture parameters."""
        # Backbone selection
        with Horizontal(classes="param-row"):
            yield Label("Backbone:", classes="param-label")
            yield Select(
                [
                    ("UNet (recommended)", "unet_medium_rf"),
                    ("UNet Large RF", "unet_large_rf"),
                    ("ConvNeXt Tiny", "convnext_tiny"),
                    ("ConvNeXt Small", "convnext_small"),
                    ("SwinT Tiny", "swint_tiny"),
                    ("SwinT Small", "swint_small"),
                ],
                value=self._state._backbone if self._state else "unet_medium_rf",
                id="backbone-select",
                classes="param-input",
            )
            yield Label("Network architecture", classes="param-hint")

        # Max stride (UNet only)
        with Horizontal(classes="param-row", id="max-stride-row"):
            yield Label("Max Stride:", classes="param-label")
            yield Select(
                [("8", "8"), ("16", "16"), ("32", "32"), ("64", "64")],
                value=str(self._state._max_stride if self._state else 16),
                id="max-stride-select",
                classes="param-input",
            )
            yield Label("Receptive field (larger = more context)", classes="param-hint")

        # Base Filters (UNet only)
        with Horizontal(classes="param-row", id="filters-row"):
            yield Label("Base Filters:", classes="param-label")
            yield Input(
                value=str(self._state._filters if self._state else 32),
                id="filters-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Initial filter count (model capacity)", classes="param-hint")

        # Filters Rate (UNet only)
        with Horizontal(classes="param-row", id="filters-rate-row"):
            yield Label("Filters Rate:", classes="param-label")
            yield Input(
                value=str(self._state._filters_rate if self._state else 2.0),
                id="filters-rate-input",
                classes="param-input",
                type="number",
            )
            yield Label("Filter scaling per block", classes="param-hint")

        # Sigma (confidence map spread)
        with Horizontal(classes="param-row"):
            yield Label("Sigma:", classes="param-label")
            yield Input(
                value=str(self._state._sigma if self._state else 5.0),
                id="sigma-input",
                classes="param-input",
                type="number",
            )
            yield Label("Confidence map spread (pixels)", classes="param-hint")

        # Output Stride
        with Horizontal(classes="param-row"):
            yield Label("Output Stride:", classes="param-label")
            yield Select(
                [("1 (full res)", "1"), ("2 (half res)", "2"), ("4 (quarter)", "4")],
                value=str(self._state._output_stride if self._state else 1),
                id="output-stride-select",
                classes="param-input",
            )
            yield Label("Output resolution", classes="param-hint")

        # Pretrained Backbone Path
        with Horizontal(classes="param-row"):
            yield Label("Pretrained Backbone:", classes="param-label")
            yield Input(
                value=self._state._pretrained_backbone if self._state else "",
                id="pretrained-backbone-input",
                classes="param-input-wide",
                placeholder="Path to .ckpt (optional)",
            )

        # Pretrained Head Path
        with Horizontal(classes="param-row"):
            yield Label("Pretrained Head:", classes="param-label")
            yield Input(
                value=self._state._pretrained_head if self._state else "",
                id="pretrained-head-input",
                classes="param-input-wide",
                placeholder="Path to .ckpt (optional)",
            )

    # ==================== PAF PARAMETERS (Bottom-Up) ====================

    def _compose_paf_params(self) -> ComposeResult:
        """Compose Part Affinity Field parameters for bottom-up models."""
        paf = self._state._paf_config if self._state else None

        # PAF Sigma
        with Horizontal(classes="param-row"):
            yield Label("PAF Sigma:", classes="param-label")
            yield Input(
                value=str(paf.sigma if paf else 15.0),
                id="paf-sigma-input",
                classes="param-input",
                type="number",
            )
            yield Label("Part affinity field spread (typically 15)", classes="param-hint")

        # PAF Output Stride
        with Horizontal(classes="param-row"):
            yield Label("PAF Output Stride:", classes="param-label")
            yield Select(
                [("1", "1"), ("2", "2"), ("4", "4")],
                value=str(paf.output_stride if paf else 4),
                id="paf-output-stride-select",
                classes="param-input",
            )
            yield Label("PAF output resolution", classes="param-hint")

        # PAF Loss Weight
        with Horizontal(classes="param-row"):
            yield Label("PAF Loss Weight:", classes="param-label")
            yield Input(
                value=str(paf.loss_weight if paf else 1.0),
                id="paf-loss-weight-input",
                classes="param-input",
                type="number",
            )
            yield Label("Relative weight in total loss", classes="param-hint")

    # ==================== CLASS VECTOR PARAMETERS (Multi-Class) ====================

    def _compose_class_vector_params(self) -> ComposeResult:
        """Compose class vector parameters for multi-class models."""
        cv = self._state._class_vector_config if self._state else None

        # Number of FC Layers
        with Horizontal(classes="param-row"):
            yield Label("FC Layers:", classes="param-label")
            yield Select(
                [("1", "1"), ("2", "2"), ("3", "3")],
                value=str(cv.num_fc_layers if cv else 1),
                id="fc-layers-select",
                classes="param-input",
            )
            yield Label("Fully-connected layers for classification", classes="param-hint")

        # FC Units
        with Horizontal(classes="param-row"):
            yield Label("FC Units:", classes="param-label")
            yield Input(
                value=str(cv.num_fc_units if cv else 64),
                id="fc-units-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Units per FC layer", classes="param-hint")

        # Class Loss Weight
        with Horizontal(classes="param-row"):
            yield Label("Class Loss Weight:", classes="param-label")
            yield Input(
                value=str(cv.loss_weight if cv else 1.0),
                id="class-loss-weight-input",
                classes="param-input",
                type="number",
            )
            yield Label("Relative weight for classification loss", classes="param-hint")

    # ==================== TRAINING PARAMETERS ====================

    def _compose_training_params(self) -> ComposeResult:
        """Compose training hyperparameters."""
        # Batch size
        with Horizontal(classes="param-row"):
            yield Label("Batch Size:", classes="param-label")
            yield Select(
                [("1", "1"), ("2", "2"), ("4", "4"), ("8", "8"), ("16", "16"), ("32", "32")],
                value=str(self._state._batch_size if self._state else 4),
                id="batch-size-select",
                classes="param-input",
            )
            yield Label("Samples per gradient update", classes="param-hint")

        # Max epochs
        with Horizontal(classes="param-row"):
            yield Label("Max Epochs:", classes="param-label")
            yield Input(
                value=str(self._state._max_epochs if self._state else 200),
                id="max-epochs-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Maximum training epochs", classes="param-hint")

        # Learning rate
        with Horizontal(classes="param-row"):
            yield Label("Learning Rate:", classes="param-label")
            yield Input(
                value=str(self._state._learning_rate if self._state else 1e-4),
                id="lr-input",
                classes="param-input",
                type="number",
            )
            yield Label("Optimizer step size (e.g., 1e-4)", classes="param-hint")

        # Optimizer
        with Horizontal(classes="param-row"):
            yield Label("Optimizer:", classes="param-label")
            yield Select(
                [("Adam", "Adam"), ("AdamW", "AdamW")],
                value=self._state._optimizer if self._state else "Adam",
                id="optimizer-select",
                classes="param-input",
            )
            yield Label("Optimization algorithm", classes="param-hint")

        # Early stopping
        with Horizontal(classes="param-row"):
            yield Label("Early Stopping:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=self._state._early_stopping if self._state else True,
                id="early-stopping-checkbox",
            )
            yield Label("Stop if validation plateaus", classes="param-hint")

        # Early stopping patience (shown when early stopping enabled)
        with Horizontal(classes="param-row", id="es-patience-row"):
            yield Label("ES Patience:", classes="param-label")
            yield Input(
                value=str(self._state._early_stopping_patience if self._state else 10),
                id="es-patience-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Epochs without improvement", classes="param-hint")

        # LR Scheduler
        scheduler_type = self._state._scheduler.type.value if self._state else "none"
        with Horizontal(classes="param-row"):
            yield Label("LR Scheduler:", classes="param-label")
            yield Select(
                [
                    ("None", "none"),
                    ("Reduce on Plateau", "ReduceLROnPlateau"),
                    ("Step LR", "StepLR"),
                ],
                value=scheduler_type,
                id="scheduler-select",
                classes="param-input",
            )
            yield Label("Learning rate decay strategy", classes="param-hint")

    # ==================== CHECKPOINT PARAMETERS ====================

    def _compose_checkpoint_params(self) -> ComposeResult:
        """Compose checkpoint and logging parameters."""
        ckpt = self._state._checkpoint if self._state else None

        # Checkpoint Directory
        with Horizontal(classes="param-row"):
            yield Label("Checkpoint Dir:", classes="param-label")
            yield Input(
                value=ckpt.checkpoint_dir if ckpt else "",
                id="ckpt-dir-input",
                classes="param-input-wide",
                placeholder="./models",
            )

        # Run Name
        with Horizontal(classes="param-row"):
            yield Label("Run Name:", classes="param-label")
            yield Input(
                value=ckpt.run_name if ckpt else "",
                id="run-name-input",
                classes="param-input-wide",
                placeholder="auto-generated",
            )

        # Resume from checkpoint
        with Horizontal(classes="param-row"):
            yield Label("Resume From:", classes="param-label")
            yield Input(
                value=ckpt.resume_from if ckpt else "",
                id="resume-ckpt-input",
                classes="param-input-wide",
                placeholder="Path to checkpoint.ckpt",
            )

        # Save Top K
        with Horizontal(classes="param-row"):
            yield Label("Save Top K:", classes="param-label")
            yield Input(
                value=str(ckpt.save_top_k if ckpt else 1),
                id="save-top-k-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Number of best checkpoints to keep", classes="param-hint")

        # Save Last
        with Horizontal(classes="param-row"):
            yield Label("Save Last:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=ckpt.save_last if ckpt else True,
                id="save-last-checkbox",
            )
            yield Label("Keep last.ckpt symlink", classes="param-hint")

        yield Static("[dim]Weights & Biases[/dim]", classes="subsection-header")

        # WandB Enable
        wandb = self._state._wandb if self._state else None
        with Horizontal(classes="param-row"):
            yield Label("Enable WandB:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=wandb.enabled if wandb else False,
                id="wandb-checkbox",
            )
            yield Label("Log to Weights & Biases", classes="param-hint")

        # WandB Project
        with Horizontal(classes="param-row"):
            yield Label("WandB Project:", classes="param-label")
            yield Input(
                value=wandb.project if wandb else "sleap-nn",
                id="wandb-project-input",
                classes="param-input-wide",
            )

        # WandB Entity
        with Horizontal(classes="param-row"):
            yield Label("WandB Entity:", classes="param-label")
            yield Input(
                value=wandb.entity if wandb else "",
                id="wandb-entity-input",
                classes="param-input-wide",
                placeholder="username or team",
            )

    # ==================== ADVANCED PARAMETERS ====================

    def _compose_advanced_params(self) -> ComposeResult:
        """Compose advanced training parameters."""
        # Accelerator
        with Horizontal(classes="param-row"):
            yield Label("Accelerator:", classes="param-label")
            yield Select(
                [
                    ("Auto", "auto"),
                    ("CPU", "cpu"),
                    ("GPU (CUDA)", "gpu"),
                    ("MPS (Apple)", "mps"),
                ],
                value=self._state._accelerator if self._state else "auto",
                id="accelerator-select",
                classes="param-input",
            )
            yield Label("Hardware for training", classes="param-hint")

        # Num Workers
        with Horizontal(classes="param-row"):
            yield Label("Num Workers:", classes="param-label")
            yield Input(
                value=str(self._state._num_workers if self._state else 0),
                id="num-workers-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Data loader workers (0 for video)", classes="param-hint")

        # Min Steps per Epoch
        with Horizontal(classes="param-row"):
            yield Label("Min Steps/Epoch:", classes="param-label")
            yield Input(
                value=str(self._state._min_steps_per_epoch if self._state else 50),
                id="min-steps-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Minimum training steps per epoch", classes="param-hint")

        # Random Seed
        with Horizontal(classes="param-row"):
            yield Label("Random Seed:", classes="param-label")
            yield Input(
                value=str(self._state._random_seed) if self._state and self._state._random_seed else "",
                id="random-seed-input",
                classes="param-input",
                type="integer",
                placeholder="none",
            )
            yield Label("For reproducibility (empty = random)", classes="param-hint")

        yield Static("[dim]Online Hard Keypoint Mining (OHKM)[/dim]", classes="subsection-header")

        ohkm = self._state._ohkm if self._state else None
        with Horizontal(classes="param-row"):
            yield Label("Enable OHKM:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=ohkm.enabled if ohkm else False,
                id="ohkm-checkbox",
            )
            yield Label("Focus training on difficult keypoints", classes="param-hint")

        with Horizontal(classes="param-row"):
            yield Label("Hard-to-Easy Ratio:", classes="param-label")
            yield Input(
                value=str(ohkm.hard_to_easy_ratio if ohkm else 2.0),
                id="ohkm-ratio-input",
                classes="param-input",
                type="number",
            )
            yield Label("Ratio to classify as hard", classes="param-hint")

        with Horizontal(classes="param-row"):
            yield Label("OHKM Loss Scale:", classes="param-label")
            yield Input(
                value=str(ohkm.loss_scale if ohkm else 1.0),
                id="ohkm-scale-input",
                classes="param-input",
                type="number",
            )
            yield Label("Scale factor for hard keypoint losses", classes="param-hint")

    # ==================== TOP-DOWN SPECIFIC ====================

    def _compose_shared_topdown_params(self) -> ComposeResult:
        """Compose shared parameters for top-down models (anchor point only)."""
        # Anchor part selection with visibility percentages
        with Container(classes="section-box"):
            yield Static("[bold]Anchor Point[/bold]", classes="section-header")
            yield Static(
                "[dim]Select a keypoint to use as the reference for centering crops. "
                "The percentage shows how often each keypoint is labeled.[/dim]"
            )

            # Build options with visibility percentages
            nodes = self._state.skeleton_nodes if self._state else []
            visibility = self._state.stats.node_visibility if self._state else {}

            options = [("(auto - centroid)", "")]
            for node in nodes:
                pct = visibility.get(node, 0) if visibility else 0
                # Color code: green >80%, orange >50%, red <=50%
                if pct > 80:
                    color = "green"
                elif pct > 50:
                    color = "yellow"
                else:
                    color = "red"
                # Format: "node_name (95%)"
                label = f"{node} ([{color}]{pct:.0f}%[/{color}])"
                options.append((label, node))

            current_value = self._state._anchor_part if self._state and self._state._anchor_part else ""

            with Horizontal(classes="param-row"):
                yield Label("Anchor Part:", classes="param-label")
                yield Select(
                    options,
                    value=current_value,
                    id="anchor-part-select",
                    classes="param-input-wide",
                )

    def _compose_centroid_params(self) -> ComposeResult:
        """Compose centroid model specific parameters."""
        # Data configuration - main params
        with Container(classes="section-box"):
            yield Static("[bold]Data Configuration[/bold]", classes="section-header")

            # Scale for centroid (typically lower)
            with Horizontal(classes="param-row"):
                yield Label("Input Scale:", classes="param-label")
                yield Input(
                    value=str(self._state._input_scale if self._state else 0.5),
                    id="centroid-scale-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Lower scale OK for centroids (0.5 typical)", classes="param-hint")

            # Show effective size
            if self._state:
                orig_h, orig_w = self._state.stats.max_height, self._state.stats.max_width
                eff_h = int(orig_h * self._state._input_scale)
                eff_w = int(orig_w * self._state._input_scale)
                yield Static(
                    f"[dim]Original: {orig_w}×{orig_h} → Input: {eff_w}×{eff_h}[/dim]",
                    classes="param-hint"
                )

        # Data Augmentation section
        with Container(classes="section-box"):
            yield Static("[bold]Data Augmentation[/bold]", classes="section-header")
            aug = self._state._augmentation if self._state else None

            yield Static("[dim]Geometric[/dim]", classes="subsection-header")

            # Rotation
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Rotation",
                    value=aug.rotation_enabled if aug else True,
                    id="centroid-rotation-checkbox",
                )
                yield Label("±", classes="param-label")
                yield Input(
                    value=str(aug.rotation_max if aug else 15.0),
                    id="centroid-rotation-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("degrees", classes="param-hint")

            # Scale
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Scale",
                    value=aug.scale_enabled if aug else True,
                    id="centroid-scale-checkbox",
                )
                yield Input(
                    value=str(aug.scale_min if aug else 0.9),
                    id="centroid-scale-min-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("to", classes="param-hint")
                yield Input(
                    value=str(aug.scale_max if aug else 1.1),
                    id="centroid-scale-max-input",
                    classes="param-input",
                    type="number",
                )

            yield Static("[dim]Intensity[/dim]", classes="subsection-header")

            # Brightness
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Brightness",
                    value=aug.brightness_enabled if aug else False,
                    id="centroid-brightness-checkbox",
                )
                yield Label("±", classes="param-label")
                yield Input(
                    value=str(aug.brightness_limit if aug else 0.2),
                    id="centroid-brightness-input",
                    classes="param-input",
                    type="number",
                )

            # Contrast
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Contrast",
                    value=aug.contrast_enabled if aug else False,
                    id="centroid-contrast-checkbox",
                )
                yield Label("±", classes="param-label")
                yield Input(
                    value=str(aug.contrast_limit if aug else 0.2),
                    id="centroid-contrast-input",
                    classes="param-input",
                    type="number",
                )

        # Additional Data Parameters (collapsible)
        with Collapsible(title="Additional Data Parameters", classes="collapsible-section"):
            with Horizontal(classes="param-row"):
                yield Label("Max Height:", classes="param-label")
                yield Input(
                    value=str(self._state._max_height) if self._state and self._state._max_height else "",
                    id="centroid-max-height-input",
                    classes="param-input",
                    type="integer",
                    placeholder="auto",
                )
                yield Label("Max Width:", classes="param-label")
                yield Input(
                    value=str(self._state._max_width) if self._state and self._state._max_width else "",
                    id="centroid-max-width-input",
                    classes="param-input",
                    type="integer",
                    placeholder="auto",
                )

        # Model configuration
        with Container(classes="section-box"):
            yield Static("[bold]Model Architecture[/bold]", classes="section-header")
            yield Static("[dim]Detects animal centers in full frame[/dim]")

            # Backbone for centroid
            with Horizontal(classes="param-row"):
                yield Label("Backbone:", classes="param-label")
                yield Select(
                    [
                        ("UNet (recommended)", "unet_medium_rf"),
                        ("UNet Large RF", "unet_large_rf"),
                        ("ConvNeXt Tiny", "convnext_tiny"),
                        ("ConvNeXt Small", "convnext_small"),
                    ],
                    value=self._state._backbone if self._state else "unet_medium_rf",
                    id="centroid-backbone-select",
                    classes="param-input",
                )
                yield Label("Network architecture", classes="param-hint")

            # Max stride
            with Horizontal(classes="param-row"):
                yield Label("Max Stride:", classes="param-label")
                yield Select(
                    [("8", "8"), ("16", "16"), ("32", "32")],
                    value=str(self._state._max_stride if self._state else 16),
                    id="centroid-max-stride-select",
                    classes="param-input",
                )
                yield Label("Receptive field", classes="param-hint")

            # Sigma for centroid
            with Horizontal(classes="param-row"):
                yield Label("Sigma:", classes="param-label")
                yield Input(
                    value=str(self._state._sigma if self._state else 5.0),
                    id="centroid-sigma-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Centroid confidence map spread", classes="param-hint")

            # Output stride
            with Horizontal(classes="param-row"):
                yield Label("Output Stride:", classes="param-label")
                yield Select(
                    [("1", "1"), ("2", "2"), ("4", "4")],
                    value=str(self._state._output_stride if self._state else 1),
                    id="centroid-output-stride-select",
                    classes="param-input",
                )
                yield Label("Output resolution", classes="param-hint")

        # Training configuration
        with Container(classes="section-box"):
            yield Static("[bold]Training[/bold]", classes="section-header")

            # Batch size
            with Horizontal(classes="param-row"):
                yield Label("Batch Size:", classes="param-label")
                yield Select(
                    [("1", "1"), ("2", "2"), ("4", "4"), ("8", "8"), ("16", "16")],
                    value=str(self._state._batch_size if self._state else 4),
                    id="centroid-batch-size-select",
                    classes="param-input",
                )
                yield Label("Samples per batch", classes="param-hint")

            # Max epochs
            with Horizontal(classes="param-row"):
                yield Label("Max Epochs:", classes="param-label")
                yield Input(
                    value=str(self._state._max_epochs if self._state else 200),
                    id="centroid-max-epochs-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Maximum training epochs", classes="param-hint")

            # Learning rate
            with Horizontal(classes="param-row"):
                yield Label("Learning Rate:", classes="param-label")
                yield Input(
                    value=str(self._state._learning_rate if self._state else 1e-4),
                    id="centroid-lr-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Optimizer step size", classes="param-hint")

        # Checkpoint & Logging
        with Collapsible(title="Checkpoints & Logging", classes="collapsible-section"):
            ckpt = self._state._checkpoint if self._state else None

            with Horizontal(classes="param-row"):
                yield Label("Checkpoint Dir:", classes="param-label")
                yield Input(
                    value=ckpt.checkpoint_dir if ckpt else "",
                    id="centroid-ckpt-dir-input",
                    classes="param-input-wide",
                    placeholder="./models",
                )

            with Horizontal(classes="param-row"):
                yield Label("Run Name:", classes="param-label")
                yield Input(
                    value=ckpt.run_name if ckpt else "",
                    id="centroid-run-name-input",
                    classes="param-input-wide",
                    placeholder="centroid_model",
                )

    def _compose_instance_params(self) -> ComposeResult:
        """Compose centered instance model specific parameters."""
        # Data configuration - main params
        with Container(classes="section-box"):
            yield Static("[bold]Data Configuration[/bold]", classes="section-header")

            # Crop size (main param for instance model)
            crop_size = self._state._crop_size if self._state else None
            if crop_size is None and self._state:
                crop_size = self._state._compute_auto_crop_size()

            with Horizontal(classes="param-row"):
                yield Label("Crop Size:", classes="param-label")
                yield Input(
                    value=str(crop_size or 256),
                    id="crop-size-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Square crop around each instance", classes="param-hint")

            # Instance scale
            with Horizontal(classes="param-row"):
                yield Label("Input Scale:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_input_scale if self._state else 1.0),
                    id="instance-scale-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Crop resize factor (1.0 = no resize)", classes="param-hint")

        # Data Augmentation section
        with Container(classes="section-box"):
            yield Static("[bold]Data Augmentation[/bold]", classes="section-header")
            aug = self._state._ci_augmentation if self._state else None

            yield Static("[dim]Geometric[/dim]", classes="subsection-header")

            # Rotation
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Rotation",
                    value=aug.rotation_enabled if aug else True,
                    id="instance-rotation-checkbox",
                )
                yield Label("±", classes="param-label")
                yield Input(
                    value=str(aug.rotation_max if aug else 15.0),
                    id="instance-rotation-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("degrees", classes="param-hint")

            # Scale
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Scale",
                    value=aug.scale_enabled if aug else True,
                    id="instance-scale-checkbox",
                )
                yield Input(
                    value=str(aug.scale_min if aug else 0.9),
                    id="instance-scale-min-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("to", classes="param-hint")
                yield Input(
                    value=str(aug.scale_max if aug else 1.1),
                    id="instance-scale-max-input",
                    classes="param-input",
                    type="number",
                )

            yield Static("[dim]Intensity[/dim]", classes="subsection-header")

            # Brightness
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Brightness",
                    value=aug.brightness_enabled if aug else False,
                    id="instance-brightness-checkbox",
                )
                yield Label("±", classes="param-label")
                yield Input(
                    value=str(aug.brightness_limit if aug else 0.2),
                    id="instance-brightness-input",
                    classes="param-input",
                    type="number",
                )

            # Contrast
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Contrast",
                    value=aug.contrast_enabled if aug else False,
                    id="instance-contrast-checkbox",
                )
                yield Label("±", classes="param-label")
                yield Input(
                    value=str(aug.contrast_limit if aug else 0.2),
                    id="instance-contrast-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("0=off, 0.2=±20%", classes="param-hint")

        # Additional Data Parameters (collapsible)
        with Collapsible(title="Additional Data Parameters", classes="collapsible-section"):
            with Horizontal(classes="param-row"):
                yield Label("Min Crop Size:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_min_crop_size if self._state else 100),
                    id="min-crop-size-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Minimum allowed crop size", classes="param-hint")

        # Model configuration
        with Container(classes="section-box"):
            yield Static("[bold]Model Architecture[/bold]", classes="section-header")
            yield Static("[dim]Detects keypoints within cropped regions[/dim]")

            # Instance backbone
            with Horizontal(classes="param-row"):
                yield Label("Backbone:", classes="param-label")
                yield Select(
                    [
                        ("UNet (recommended)", "unet_medium_rf"),
                        ("UNet Large RF", "unet_large_rf"),
                        ("ConvNeXt Tiny", "convnext_tiny"),
                        ("ConvNeXt Small", "convnext_small"),
                    ],
                    value=self._state._ci_backbone if self._state else "unet_medium_rf",
                    id="instance-backbone-select",
                    classes="param-input",
                )
                yield Label("Network architecture", classes="param-hint")

            # Max stride
            with Horizontal(classes="param-row"):
                yield Label("Max Stride:", classes="param-label")
                yield Select(
                    [("8", "8"), ("16", "16"), ("32", "32")],
                    value=str(self._state._ci_max_stride if self._state else 16),
                    id="instance-max-stride-select",
                    classes="param-input",
                )
                yield Label("Receptive field", classes="param-hint")

            # Instance sigma
            with Horizontal(classes="param-row"):
                yield Label("Sigma:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_sigma if self._state else 2.5),
                    id="instance-sigma-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Tighter sigma for crops (2.5 typical)", classes="param-hint")

            # Instance output stride
            with Horizontal(classes="param-row"):
                yield Label("Output Stride:", classes="param-label")
                yield Select(
                    [("1", "1"), ("2", "2"), ("4", "4")],
                    value=str(self._state._ci_output_stride if self._state else 2),
                    id="instance-output-stride-select",
                    classes="param-input",
                )
                yield Label("Output resolution", classes="param-hint")

        # Training configuration
        with Container(classes="section-box"):
            yield Static("[bold]Training[/bold]", classes="section-header")

            # Batch size
            with Horizontal(classes="param-row"):
                yield Label("Batch Size:", classes="param-label")
                yield Select(
                    [("1", "1"), ("2", "2"), ("4", "4"), ("8", "8"), ("16", "16")],
                    value=str(self._state._ci_batch_size if self._state else 4),
                    id="instance-batch-size-select",
                    classes="param-input",
                )
                yield Label("Samples per batch", classes="param-hint")

            # Max epochs
            with Horizontal(classes="param-row"):
                yield Label("Max Epochs:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_max_epochs if self._state else 200),
                    id="instance-max-epochs-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Maximum training epochs", classes="param-hint")

            # Learning rate
            with Horizontal(classes="param-row"):
                yield Label("Learning Rate:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_learning_rate if self._state else 1e-4),
                    id="instance-lr-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Optimizer step size", classes="param-hint")

        # Checkpoint & Logging
        with Collapsible(title="Checkpoints & Logging", classes="collapsible-section"):
            ckpt = self._state._checkpoint if self._state else None

            with Horizontal(classes="param-row"):
                yield Label("Checkpoint Dir:", classes="param-label")
                yield Input(
                    value=ckpt.checkpoint_dir if ckpt else "",
                    id="instance-ckpt-dir-input",
                    classes="param-input-wide",
                    placeholder="./models",
                )

            with Horizontal(classes="param-row"):
                yield Label("Run Name:", classes="param-label")
                yield Input(
                    value=ckpt.run_name if ckpt else "",
                    id="instance-run-name-input",
                    classes="param-input-wide",
                    placeholder="centered_instance_model",
                )

    # ==================== MEMORY ESTIMATE ====================

    def _compose_memory_estimate(self) -> ComposeResult:
        """Compose memory estimate display."""
        if not self._state:
            return

        mem = self._state.memory_estimate()
        status_color = mem.gpu_status

        with Container(id="memory-estimate"):
            yield Static(
                f"[bold]Memory Estimate:[/bold] [{status_color}]{mem.total_gpu_gb:.1f} GB[/{status_color}]\n"
                f"[dim]{mem.gpu_message}[/dim]"
            )

    # ==================== EVENT HANDLERS ====================

    # Data handlers
    @on(Input.Changed, "#scale-input")
    def handle_scale_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._input_scale = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#max-height-input")
    def handle_max_height_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._max_height = int(event.value) if event.value else None

    @on(Input.Changed, "#max-width-input")
    def handle_max_width_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._max_width = int(event.value) if event.value else None

    @on(Select.Changed, "#channels-select")
    def handle_channels_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._ensure_rgb = event.value == "rgb"
            self._state._ensure_grayscale = event.value == "grayscale"

    @on(Input.Changed, "#val-fraction-input")
    def handle_val_fraction_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._validation_fraction = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#data-pipeline-select")
    def handle_data_pipeline_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._data_pipeline = DataPipelineType(event.value)

    # Model handlers
    @on(Select.Changed, "#backbone-select")
    def handle_backbone_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._backbone = event.value
            if "convnext" in event.value or "swint" in event.value:
                self._state._max_stride = 32
                try:
                    select = self.query_one("#max-stride-select", Select)
                    select.value = "32"
                except Exception:
                    pass

    @on(Select.Changed, "#max-stride-select")
    def handle_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._max_stride = int(event.value)

    @on(Input.Changed, "#filters-input")
    def handle_filters_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._filters = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#filters-rate-input")
    def handle_filters_rate_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._filters_rate = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#sigma-input")
    def handle_sigma_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._sigma = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#output-stride-select")
    def handle_output_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._output_stride = int(event.value)

    @on(Input.Changed, "#pretrained-backbone-input")
    def handle_pretrained_backbone_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._pretrained_backbone = event.value or ""

    @on(Input.Changed, "#pretrained-head-input")
    def handle_pretrained_head_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._pretrained_head = event.value or ""

    # PAF handlers
    @on(Input.Changed, "#paf-sigma-input")
    def handle_paf_sigma_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._paf_config.sigma = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#paf-output-stride-select")
    def handle_paf_output_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._paf_config.output_stride = int(event.value)

    @on(Input.Changed, "#paf-loss-weight-input")
    def handle_paf_loss_weight_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._paf_config.loss_weight = float(event.value)
            except ValueError:
                pass

    # Class vector handlers
    @on(Select.Changed, "#fc-layers-select")
    def handle_fc_layers_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._class_vector_config.num_fc_layers = int(event.value)

    @on(Input.Changed, "#fc-units-input")
    def handle_fc_units_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._class_vector_config.num_fc_units = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#class-loss-weight-input")
    def handle_class_loss_weight_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._class_vector_config.loss_weight = float(event.value)
            except ValueError:
                pass

    # Training handlers
    @on(Select.Changed, "#batch-size-select")
    def handle_batch_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._batch_size = int(event.value)

    @on(Input.Changed, "#max-epochs-input")
    def handle_epochs_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#lr-input")
    def handle_lr_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._learning_rate = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#optimizer-select")
    def handle_optimizer_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._optimizer = event.value

    @on(Checkbox.Changed, "#early-stopping-checkbox")
    def handle_early_stopping_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._early_stopping = event.value

    @on(Input.Changed, "#es-patience-input")
    def handle_es_patience_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._early_stopping_patience = int(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#scheduler-select")
    def handle_scheduler_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._scheduler.type = SchedulerType(event.value)

    # Augmentation checkbox handlers (standard config)
    @on(Checkbox.Changed, "#rotation-checkbox")
    def handle_rotation_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.rotation_enabled = event.value

    @on(Checkbox.Changed, "#scale-checkbox")
    def handle_scale_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.scale_enabled = event.value

    @on(Checkbox.Changed, "#brightness-checkbox")
    def handle_brightness_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.brightness_enabled = event.value

    @on(Checkbox.Changed, "#contrast-checkbox")
    def handle_contrast_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.contrast_enabled = event.value

    @on(Input.Changed, "#rotation-input")
    def handle_rotation_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                val = float(event.value)
                self._state._augmentation.rotation_min = -abs(val)
                self._state._augmentation.rotation_max = abs(val)
            except ValueError:
                pass

    @on(Input.Changed, "#scale-min-input")
    def handle_scale_min_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.scale_min = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scale-max-input")
    def handle_scale_max_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.scale_max = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#brightness-limit-input")
    def handle_brightness_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.brightness_limit = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#contrast-limit-input")
    def handle_contrast_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.contrast_limit = float(event.value)
            except ValueError:
                pass

    # Checkpoint handlers
    @on(Input.Changed, "#ckpt-dir-input")
    def handle_ckpt_dir_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._checkpoint.checkpoint_dir = event.value or ""

    @on(Input.Changed, "#run-name-input")
    def handle_run_name_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._checkpoint.run_name = event.value or ""

    @on(Input.Changed, "#resume-ckpt-input")
    def handle_resume_ckpt_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._checkpoint.resume_from = event.value or ""

    @on(Input.Changed, "#save-top-k-input")
    def handle_save_top_k_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._checkpoint.save_top_k = int(event.value)
            except ValueError:
                pass

    @on(Checkbox.Changed, "#save-last-checkbox")
    def handle_save_last_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._checkpoint.save_last = event.value

    @on(Checkbox.Changed, "#wandb-checkbox")
    def handle_wandb_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._wandb.enabled = event.value

    @on(Input.Changed, "#wandb-project-input")
    def handle_wandb_project_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._wandb.project = event.value or "sleap-nn"

    @on(Input.Changed, "#wandb-entity-input")
    def handle_wandb_entity_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._wandb.entity = event.value or ""

    # Advanced handlers
    @on(Select.Changed, "#accelerator-select")
    def handle_accelerator_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._accelerator = event.value

    @on(Input.Changed, "#num-workers-input")
    def handle_num_workers_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._num_workers = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#min-steps-input")
    def handle_min_steps_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._min_steps_per_epoch = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#random-seed-input")
    def handle_random_seed_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._random_seed = int(event.value) if event.value else None

    @on(Checkbox.Changed, "#ohkm-checkbox")
    def handle_ohkm_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._ohkm.enabled = event.value

    @on(Input.Changed, "#ohkm-ratio-input")
    def handle_ohkm_ratio_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ohkm.hard_to_easy_ratio = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#ohkm-scale-input")
    def handle_ohkm_scale_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ohkm.loss_scale = float(event.value)
            except ValueError:
                pass

    # Top-down specific handlers
    @on(Select.Changed, "#anchor-part-select")
    def handle_anchor_change(self, event: Select.Changed) -> None:
        if self._state:
            self._state._anchor_part = event.value if event.value else None

    @on(Input.Changed, "#centroid-scale-input")
    def handle_centroid_scale_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._input_scale = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-sigma-input")
    def handle_centroid_sigma_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._sigma = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#centroid-backbone-select")
    def handle_centroid_backbone_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._backbone = event.value

    @on(Input.Changed, "#crop-size-input")
    def handle_crop_size_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._crop_size = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#min-crop-size-input")
    def handle_min_crop_size_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_min_crop_size = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-scale-input")
    def handle_instance_scale_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_input_scale = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-sigma-input")
    def handle_instance_sigma_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_sigma = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#instance-backbone-select")
    def handle_instance_backbone_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._ci_backbone = event.value

    @on(Select.Changed, "#instance-output-stride-select")
    def handle_instance_output_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._ci_output_stride = int(event.value)

    # Additional top-down handlers
    @on(Input.Changed, "#centroid-max-height-input")
    def handle_centroid_max_height_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._max_height = int(event.value) if event.value else None

    @on(Input.Changed, "#centroid-max-width-input")
    def handle_centroid_max_width_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._max_width = int(event.value) if event.value else None

    @on(Select.Changed, "#centroid-max-stride-select")
    def handle_centroid_max_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._max_stride = int(event.value)

    @on(Select.Changed, "#centroid-output-stride-select")
    def handle_centroid_output_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._output_stride = int(event.value)

    @on(Select.Changed, "#centroid-batch-size-select")
    def handle_centroid_batch_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._batch_size = int(event.value)

    @on(Input.Changed, "#centroid-max-epochs-input")
    def handle_centroid_epochs_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-lr-input")
    def handle_centroid_lr_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._learning_rate = float(event.value)
            except ValueError:
                pass

    # Centroid augmentation checkbox handlers
    @on(Checkbox.Changed, "#centroid-rotation-checkbox")
    def handle_centroid_rotation_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.rotation_enabled = event.value

    @on(Checkbox.Changed, "#centroid-scale-checkbox")
    def handle_centroid_scale_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.scale_enabled = event.value

    @on(Checkbox.Changed, "#centroid-brightness-checkbox")
    def handle_centroid_brightness_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.brightness_enabled = event.value

    @on(Checkbox.Changed, "#centroid-contrast-checkbox")
    def handle_centroid_contrast_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._augmentation.contrast_enabled = event.value

    @on(Input.Changed, "#centroid-rotation-input")
    def handle_centroid_rotation_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                val = float(event.value)
                self._state._augmentation.rotation_max = val
                self._state._augmentation.rotation_min = -val
            except ValueError:
                pass

    @on(Select.Changed, "#instance-max-stride-select")
    def handle_instance_max_stride_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._ci_max_stride = int(event.value)

    @on(Select.Changed, "#instance-batch-size-select")
    def handle_instance_batch_change(self, event: Select.Changed) -> None:
        if self._state and event.value:
            self._state._ci_batch_size = int(event.value)

    @on(Input.Changed, "#instance-max-epochs-input")
    def handle_instance_epochs_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-lr-input")
    def handle_instance_lr_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_learning_rate = float(event.value)
            except ValueError:
                pass

    # Instance augmentation checkbox handlers
    @on(Checkbox.Changed, "#instance-rotation-checkbox")
    def handle_instance_rotation_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._ci_augmentation.rotation_enabled = event.value

    @on(Checkbox.Changed, "#instance-scale-checkbox")
    def handle_instance_scale_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._ci_augmentation.scale_enabled = event.value

    @on(Checkbox.Changed, "#instance-brightness-checkbox")
    def handle_instance_brightness_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._ci_augmentation.brightness_enabled = event.value

    @on(Checkbox.Changed, "#instance-contrast-checkbox")
    def handle_instance_contrast_checkbox_change(self, event: Checkbox.Changed) -> None:
        if self._state:
            self._state._ci_augmentation.contrast_enabled = event.value

    @on(Input.Changed, "#instance-rotation-input")
    def handle_instance_rotation_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                val = float(event.value)
                self._state._ci_augmentation.rotation_max = val
                self._state._ci_augmentation.rotation_min = -val
            except ValueError:
                pass

    # Checkpoint handlers for top-down models
    @on(Input.Changed, "#centroid-ckpt-dir-input")
    def handle_centroid_ckpt_dir_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._checkpoint.checkpoint_dir = event.value or ""

    @on(Input.Changed, "#centroid-run-name-input")
    def handle_centroid_run_name_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._checkpoint.run_name = event.value or ""

    @on(Input.Changed, "#instance-ckpt-dir-input")
    def handle_instance_ckpt_dir_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._ci_checkpoint_dir = event.value or ""

    @on(Input.Changed, "#instance-run-name-input")
    def handle_instance_run_name_change(self, event: Input.Changed) -> None:
        if self._state:
            self._state._ci_run_name = event.value or ""

    # Centroid augmentation handlers
    @on(Input.Changed, "#centroid-scale-min-input")
    def handle_centroid_scale_min_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.scale_min = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-scale-max-input")
    def handle_centroid_scale_max_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.scale_max = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-brightness-input")
    def handle_centroid_brightness_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.brightness_limit = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-contrast-input")
    def handle_centroid_contrast_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._augmentation.contrast_limit = float(event.value)
            except ValueError:
                pass

    # Instance augmentation handlers
    @on(Input.Changed, "#instance-scale-min-input")
    def handle_instance_scale_min_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_augmentation.scale_min = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-scale-max-input")
    def handle_instance_scale_max_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_augmentation.scale_max = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-brightness-input")
    def handle_instance_brightness_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_augmentation.brightness_limit = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-contrast-input")
    def handle_instance_contrast_change(self, event: Input.Changed) -> None:
        if self._state and event.value:
            try:
                self._state._ci_augmentation.contrast_limit = float(event.value)
            except ValueError:
                pass
