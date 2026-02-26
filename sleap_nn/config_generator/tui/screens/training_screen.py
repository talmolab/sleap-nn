"""Training configuration screen.

Provides UI for configuring training hyperparameters, schedulers, and logging.
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
    Switch,
)

from sleap_nn.config_generator.tui.widgets import (
    Collapsible,
    InfoBox,
    TipBox,
)

if TYPE_CHECKING:
    from sleap_nn.config_generator.tui.state import ConfigState


class TrainingScreen(VerticalScroll):
    """Training configuration screen component.

    Provides controls for:
    - Basic training parameters (batch size, epochs, learning rate)
    - Optimizer selection
    - Learning rate schedulers
    - Early stopping
    - Checkpoint configuration
    - OHKM (Online Hard Keypoint Mining)
    - Weights & Biases logging
    - Evaluation settings
    """

    DEFAULT_CSS = """
    TrainingScreen {
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
    """

    def __init__(self, state: "ConfigState", **kwargs):
        """Initialize the training screen.

        Args:
            state: Shared configuration state.
            **kwargs: Additional arguments for Container.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the training screen layout."""
        # Basic Training Parameters
        yield from self._compose_basic_section()

        yield Rule()

        # Optimizer & Scheduler
        yield from self._compose_optimizer_section()

        yield Rule()

        # Early Stopping
        yield from self._compose_early_stopping_section()

        yield Rule()

        # Checkpoint Configuration
        yield from self._compose_checkpoint_section()

        yield Rule()

        # OHKM
        yield from self._compose_ohkm_section()

        yield Rule()

        # Weights & Biases
        yield from self._compose_wandb_section()

        yield Rule()

        # Evaluation
        yield from self._compose_evaluation_section()

    def _compose_basic_section(self) -> ComposeResult:
        """Compose basic training parameters section."""
        yield Static("Training Parameters", classes="section-title")

        # Batch Size
        with Horizontal(classes="form-row"):
            yield Static("Batch Size:", classes="form-label")
            yield Select(
                [
                    ("1", 1),
                    ("2", 2),
                    ("4", 4),
                    ("8", 8),
                    ("16", 16),
                    ("32", 32),
                ],
                value=self._state._batch_size,
                id="batch-size-select",
            )

        # Max Epochs
        with Horizontal(classes="form-row"):
            yield Static("Max Epochs:", classes="form-label")
            yield Input(
                value=str(self._state._max_epochs),
                type="integer",
                id="max-epochs-input",
            )

        # Learning Rate
        with Horizontal(classes="form-row"):
            yield Static("Learning Rate:", classes="form-label")
            yield Select(
                [
                    ("1e-5", 1e-5),
                    ("5e-5", 5e-5),
                    ("1e-4 (Default)", 1e-4),
                    ("5e-4", 5e-4),
                    ("1e-3", 1e-3),
                ],
                value=self._state._learning_rate,
                id="learning-rate-select",
            )

        # Min Steps Per Epoch
        with Horizontal(classes="form-row"):
            yield Static("Min Steps/Epoch:", classes="form-label")
            yield Input(
                value=str(self._state._min_steps_per_epoch),
                type="integer",
                id="min-steps-input",
            )

        # Random Seed
        with Horizontal(classes="form-row"):
            yield Static("Random Seed:", classes="form-label")
            yield Input(
                value=str(self._state._random_seed) if self._state._random_seed else "",
                placeholder="Random (no seed)",
                type="integer",
                id="seed-input",
            )

        yield TipBox(
            "Batch Size: Larger = faster training but more memory. Start with 4.\n"
            "Learning Rate: 1e-4 works well for most cases.",
            title=None,
        )

    def _compose_optimizer_section(self) -> ComposeResult:
        """Compose optimizer and scheduler section."""
        yield Static("Optimizer & Scheduler", classes="section-title")

        # Optimizer
        with Horizontal(classes="form-row"):
            yield Static("Optimizer:", classes="form-label")
            with RadioSet(id="optimizer-select"):
                yield RadioButton(
                    "Adam",
                    id="opt-adam",
                    value=self._state._optimizer == "Adam",
                )
                yield RadioButton(
                    "AdamW (with weight decay)",
                    id="opt-adamw",
                    value=self._state._optimizer == "AdamW",
                )

        # Accelerator
        with Horizontal(classes="form-row"):
            yield Static("Accelerator:", classes="form-label")
            yield Select(
                [
                    ("Auto (detect)", "auto"),
                    ("GPU (CUDA)", "gpu"),
                    ("MPS (Apple Silicon)", "mps"),
                    ("CPU", "cpu"),
                ],
                value=self._state._accelerator,
                id="accelerator-select",
            )

        yield Rule()

        # LR Scheduler
        yield Static("Learning Rate Scheduler", classes="subsection-title")

        with Horizontal(classes="form-row"):
            yield Static("Scheduler Type:", classes="form-label")
            yield Select(
                [
                    ("None", "none"),
                    ("ReduceLROnPlateau", "ReduceLROnPlateau"),
                    ("StepLR", "StepLR"),
                    ("CosineAnnealingWarmup", "CosineAnnealingWarmup"),
                    ("LinearWarmupLinearDecay", "LinearWarmupLinearDecay"),
                ],
                value=self._state._scheduler.type.value,
                id="scheduler-type-select",
            )

        # ReduceLROnPlateau settings
        with Container(id="scheduler-plateau-settings", classes="scheduler-settings"):
            with Horizontal(classes="form-row"):
                yield Static("Factor:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.factor),
                    type="number",
                    id="scheduler-factor-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Patience:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.plateau_patience),
                    type="integer",
                    id="scheduler-patience-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Min LR:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.min_lr),
                    type="number",
                    id="scheduler-min-lr-input",
                )

        # StepLR settings
        with Container(
            id="scheduler-step-settings", classes="scheduler-settings hidden"
        ):
            with Horizontal(classes="form-row"):
                yield Static("Step Size:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.step_size),
                    type="integer",
                    id="scheduler-step-size-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Gamma:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.gamma),
                    type="number",
                    id="scheduler-gamma-input",
                )

        # CosineAnnealingWarmup settings
        with Container(
            id="scheduler-cosine-settings", classes="scheduler-settings hidden"
        ):
            with Horizontal(classes="form-row"):
                yield Static("Warmup Epochs:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.warmup_epochs),
                    type="integer",
                    id="scheduler-warmup-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Min LR Ratio:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.min_lr_ratio),
                    type="number",
                    id="scheduler-min-lr-ratio-input",
                )

        # LinearWarmupLinearDecay settings
        with Container(
            id="scheduler-linear-settings", classes="scheduler-settings hidden"
        ):
            with Horizontal(classes="form-row"):
                yield Static("Warmup Ratio:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.warmup_ratio),
                    type="number",
                    id="scheduler-warmup-ratio-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Decay Ratio:", classes="form-label")
                yield Input(
                    value=str(self._state._scheduler.decay_ratio),
                    type="number",
                    id="scheduler-decay-ratio-input",
                )

    def _compose_early_stopping_section(self) -> ComposeResult:
        """Compose early stopping section."""
        yield Static("Early Stopping", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Static("Enable Early Stopping:", classes="form-label")
            yield Switch(value=self._state._early_stopping, id="early-stopping-switch")

        with Container(id="early-stopping-settings"):
            with Horizontal(classes="form-row"):
                yield Static("Patience:", classes="form-label")
                yield Input(
                    value=str(self._state._early_stopping_patience),
                    type="integer",
                    id="early-stopping-patience-input",
                )

            with Horizontal(classes="form-row"):
                yield Static("Min Delta:", classes="form-label")
                yield Input(
                    value=str(self._state._early_stopping_min_delta),
                    type="number",
                    id="early-stopping-min-delta-input",
                )

        yield TipBox(
            "Early stopping prevents overfitting by stopping training when "
            "validation loss stops improving. Patience is the number of epochs "
            "to wait before stopping.",
            title=None,
        )

    def _compose_checkpoint_section(self) -> ComposeResult:
        """Compose checkpoint configuration section."""
        with Collapsible(
            title="Checkpoint Settings", expanded=False, id="checkpoint-section"
        ):
            # Run Name
            with Horizontal(classes="form-row"):
                yield Static("Run Name:", classes="form-label")
                yield Input(
                    value=self._state._checkpoint.run_name,
                    placeholder="Auto-generated",
                    id="run-name-input",
                )

            # Checkpoint Directory
            with Horizontal(classes="form-row"):
                yield Static("Checkpoint Dir:", classes="form-label")
                yield Input(
                    value=self._state._checkpoint.checkpoint_dir,
                    placeholder="./models",
                    id="ckpt-dir-input",
                )

            # Save Top K
            with Horizontal(classes="form-row"):
                yield Static("Save Top K:", classes="form-label")
                yield Input(
                    value=str(self._state._checkpoint.save_top_k),
                    type="integer",
                    id="save-top-k-input",
                )

            # Save Last
            with Horizontal(classes="form-row"):
                yield Static("Save Last:", classes="form-label")
                yield Switch(
                    value=self._state._checkpoint.save_last, id="save-last-switch"
                )

            # Resume From
            with Horizontal(classes="form-row"):
                yield Static("Resume From:", classes="form-label")
                yield Input(
                    value=self._state._checkpoint.resume_from,
                    placeholder="Path to checkpoint",
                    id="resume-from-input",
                )

    def _compose_ohkm_section(self) -> ComposeResult:
        """Compose OHKM (Online Hard Keypoint Mining) section."""
        with Collapsible(
            title="Online Hard Keypoint Mining (OHKM)",
            expanded=False,
            id="ohkm-section",
        ):
            yield InfoBox(
                "OHKM focuses training on hard-to-predict keypoints to improve accuracy.",
            )

            with Horizontal(classes="form-row"):
                yield Static("Enable OHKM:", classes="form-label")
                yield Switch(value=self._state._ohkm.enabled, id="ohkm-switch")

            with Container(id="ohkm-settings"):
                with Horizontal(classes="form-row"):
                    yield Static("Hard/Easy Ratio:", classes="form-label")
                    yield Input(
                        value=str(self._state._ohkm.hard_to_easy_ratio),
                        type="number",
                        id="ohkm-ratio-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Loss Scale:", classes="form-label")
                    yield Input(
                        value=str(self._state._ohkm.loss_scale),
                        type="number",
                        id="ohkm-scale-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Min Hard Keypoints:", classes="form-label")
                    yield Input(
                        value=str(self._state._ohkm.min_hard_keypoints),
                        type="integer",
                        id="ohkm-min-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Max Hard Keypoints:", classes="form-label")
                    yield Input(
                        value=str(self._state._ohkm.max_hard_keypoints),
                        type="integer",
                        id="ohkm-max-input",
                    )

    def _compose_wandb_section(self) -> ComposeResult:
        """Compose Weights & Biases logging section."""
        with Collapsible(
            title="Weights & Biases Logging",
            expanded=False,
            id="wandb-section",
        ):
            yield InfoBox(
                "Log training metrics and visualizations to Weights & Biases for experiment tracking.",
            )

            with Horizontal(classes="form-row"):
                yield Static("Enable W&B:", classes="form-label")
                yield Switch(value=self._state._wandb.enabled, id="wandb-switch")

            with Container(id="wandb-settings"):
                with Horizontal(classes="form-row"):
                    yield Static("Entity:", classes="form-label")
                    yield Input(
                        value=self._state._wandb.entity,
                        placeholder="Username or team name",
                        id="wandb-entity-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Project:", classes="form-label")
                    yield Input(
                        value=self._state._wandb.project,
                        id="wandb-project-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Run Name:", classes="form-label")
                    yield Input(
                        value=self._state._wandb.name,
                        placeholder="Auto-generated",
                        id="wandb-name-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("API Key:", classes="form-label")
                    yield Input(
                        value=self._state._wandb.api_key,
                        placeholder="Leave empty if logged in",
                        password=True,
                        id="wandb-api-key-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Mode:", classes="form-label")
                    yield Select(
                        [
                            ("Online", "online"),
                            ("Offline", "offline"),
                            ("Disabled", "disabled"),
                        ],
                        value=self._state._wandb.mode,
                        id="wandb-mode-select",
                    )

    def _compose_evaluation_section(self) -> ComposeResult:
        """Compose OKS evaluation section."""
        with Collapsible(
            title="OKS Evaluation During Training",
            expanded=False,
            id="eval-section",
        ):
            yield InfoBox(
                "Run Object Keypoint Similarity (OKS) evaluation during training "
                "to track pose estimation accuracy.",
            )

            with Horizontal(classes="form-row"):
                yield Static("Enable Evaluation:", classes="form-label")
                yield Switch(value=self._state._evaluation.enabled, id="eval-switch")

            with Container(id="eval-settings"):
                with Horizontal(classes="form-row"):
                    yield Static("Frequency (epochs):", classes="form-label")
                    yield Input(
                        value=str(self._state._evaluation.frequency),
                        type="integer",
                        id="eval-frequency-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("OKS Stddev:", classes="form-label")
                    yield Input(
                        value=str(self._state._evaluation.oks_stddev),
                        type="number",
                        id="eval-oks-stddev-input",
                    )

    def _update_scheduler_visibility(self) -> None:
        """Update scheduler settings visibility based on selected type."""
        from sleap_nn.config_generator.tui.state import SchedulerType

        try:
            # Hide all scheduler settings
            for settings_id in [
                "scheduler-plateau-settings",
                "scheduler-step-settings",
                "scheduler-cosine-settings",
                "scheduler-linear-settings",
            ]:
                try:
                    container = self.query_one(f"#{settings_id}", Container)
                    container.add_class("hidden")
                except Exception:
                    pass

            # Show the relevant one
            type_to_settings = {
                SchedulerType.REDUCE_ON_PLATEAU: "scheduler-plateau-settings",
                SchedulerType.STEP_LR: "scheduler-step-settings",
                SchedulerType.COSINE_ANNEALING_WARMUP: "scheduler-cosine-settings",
                SchedulerType.LINEAR_WARMUP_LINEAR_DECAY: "scheduler-linear-settings",
            }

            settings_id = type_to_settings.get(self._state._scheduler.type)
            if settings_id:
                container = self.query_one(f"#{settings_id}", Container)
                container.remove_class("hidden")
        except Exception:
            pass

    @on(Select.Changed, "#batch-size-select")
    def handle_batch_size_change(self, event: Select.Changed) -> None:
        """Handle batch size selection changes."""
        self._state._batch_size = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#max-epochs-input")
    def handle_max_epochs_change(self, event: Input.Changed) -> None:
        """Handle max epochs input changes."""
        try:
            self._state._max_epochs = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Select.Changed, "#learning-rate-select")
    def handle_learning_rate_change(self, event: Select.Changed) -> None:
        """Handle learning rate selection changes."""
        self._state._learning_rate = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#min-steps-input")
    def handle_min_steps_change(self, event: Input.Changed) -> None:
        """Handle min steps input changes."""
        try:
            self._state._min_steps_per_epoch = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#seed-input")
    def handle_seed_change(self, event: Input.Changed) -> None:
        """Handle random seed input changes."""
        try:
            self._state._random_seed = int(event.value) if event.value else None
            self._state.notify_observers()
        except ValueError:
            pass

    @on(RadioSet.Changed, "#optimizer-select")
    def handle_optimizer_change(self, event: RadioSet.Changed) -> None:
        """Handle optimizer selection changes."""
        optimizer_map = {
            "opt-adam": "Adam",
            "opt-adamw": "AdamW",
        }
        if event.pressed.id in optimizer_map:
            self._state._optimizer = optimizer_map[event.pressed.id]
            self._state.notify_observers()

    @on(Select.Changed, "#accelerator-select")
    def handle_accelerator_change(self, event: Select.Changed) -> None:
        """Handle accelerator selection changes."""
        self._state._accelerator = event.value
        self._state.notify_observers()

    @on(Select.Changed, "#scheduler-type-select")
    def handle_scheduler_type_change(self, event: Select.Changed) -> None:
        """Handle scheduler type selection changes."""
        from sleap_nn.config_generator.tui.state import SchedulerType

        self._state._scheduler.type = SchedulerType(event.value)
        self._update_scheduler_visibility()
        self._state.notify_observers()

    @on(Input.Changed, "#scheduler-factor-input")
    def handle_scheduler_factor_change(self, event: Input.Changed) -> None:
        """Handle scheduler factor input changes."""
        try:
            self._state._scheduler.factor = float(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#scheduler-patience-input")
    def handle_scheduler_patience_change(self, event: Input.Changed) -> None:
        """Handle scheduler patience input changes."""
        try:
            self._state._scheduler.plateau_patience = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Switch.Changed, "#early-stopping-switch")
    def handle_early_stopping_toggle(self, event: Switch.Changed) -> None:
        """Handle early stopping toggle."""
        self._state._early_stopping = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#early-stopping-patience-input")
    def handle_es_patience_change(self, event: Input.Changed) -> None:
        """Handle early stopping patience input changes."""
        try:
            self._state._early_stopping_patience = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Input.Changed, "#run-name-input")
    def handle_run_name_change(self, event: Input.Changed) -> None:
        """Handle run name input changes."""
        self._state._checkpoint.run_name = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#ckpt-dir-input")
    def handle_ckpt_dir_change(self, event: Input.Changed) -> None:
        """Handle checkpoint directory input changes."""
        self._state._checkpoint.checkpoint_dir = event.value
        self._state.notify_observers()

    @on(Switch.Changed, "#ohkm-switch")
    def handle_ohkm_toggle(self, event: Switch.Changed) -> None:
        """Handle OHKM toggle."""
        self._state._ohkm.enabled = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#ohkm-ratio-input")
    def handle_ohkm_ratio_change(self, event: Input.Changed) -> None:
        """Handle OHKM ratio input changes."""
        try:
            self._state._ohkm.hard_to_easy_ratio = float(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    @on(Switch.Changed, "#wandb-switch")
    def handle_wandb_toggle(self, event: Switch.Changed) -> None:
        """Handle W&B toggle."""
        self._state._wandb.enabled = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#wandb-entity-input")
    def handle_wandb_entity_change(self, event: Input.Changed) -> None:
        """Handle W&B entity input changes."""
        self._state._wandb.entity = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#wandb-project-input")
    def handle_wandb_project_change(self, event: Input.Changed) -> None:
        """Handle W&B project input changes."""
        self._state._wandb.project = event.value
        self._state.notify_observers()

    @on(Switch.Changed, "#eval-switch")
    def handle_eval_toggle(self, event: Switch.Changed) -> None:
        """Handle evaluation toggle."""
        self._state._evaluation.enabled = event.value
        self._state.notify_observers()

    @on(Input.Changed, "#eval-frequency-input")
    def handle_eval_frequency_change(self, event: Input.Changed) -> None:
        """Handle evaluation frequency input changes."""
        try:
            self._state._evaluation.frequency = int(event.value)
            self._state.notify_observers()
        except ValueError:
            pass

    def _update_ui_from_state(self) -> None:
        """Update all UI elements from current state."""
        try:
            # Batch size
            self.query_one("#batch-size-select", Select).value = self._state._batch_size

            # Max epochs
            self.query_one("#max-epochs-input", Input).value = str(
                self._state._max_epochs
            )

            # Learning rate
            self.query_one("#learning-rate-select", Select).value = (
                self._state._learning_rate
            )

            # Scheduler visibility
            self._update_scheduler_visibility()

        except Exception:
            pass
