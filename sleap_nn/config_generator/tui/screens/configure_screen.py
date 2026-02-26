"""Configuration screen for the config generator TUI.

Step 3: Configure training parameters with comprehensive parameter support.
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
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

    BINDINGS = [
        Binding("bracketleft", "prev_tab", "Previous Tab", show=False),
        Binding("bracketright", "next_tab", "Next Tab", show=False),
        Binding("1", "goto_tab_1", "Shared Settings", show=False),
        Binding("2", "goto_tab_2", "Centroid Model", show=False),
        Binding("3", "goto_tab_3", "Instance Model", show=False),
    ]

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

    /* Shape info display */
    .shape-info {
        background: $surface;
        border: solid $primary-darken-2;
        padding: 1 2;
        margin: 1 0;
        width: 100%;
    }

    /* Memory estimate display */
    .memory-info {
        background: $surface;
        border: solid $success-darken-2;
        padding: 1 2;
        margin: 1 0;
        width: 100%;
    }

    .memory-warning {
        border: solid $warning;
    }

    .memory-danger {
        border: solid $error;
    }

    /* Info boxes (for tips, warnings) */
    .info-box {
        background: rgba(16, 185, 129, 0.1);
        border: solid $success;
        padding: 1 2;
        margin-bottom: 1;
        width: 100%;
    }

    .speed-tip {
        background: rgba(16, 185, 129, 0.15);
        border: solid $success;
    }
    """

    def __init__(self, state: Optional[ConfigState] = None, **kwargs):
        """Initialize the configure screen.

        Args:
            state: ConfigState with current configuration.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical(id="config-container"):
            yield Label(
                "[bold]Step 3: Configure Training[/bold]", classes="section-title"
            )

            if self._state and self._state.is_topdown:
                yield Label(
                    "[yellow]Top-down model selected - configuring 2 models[/yellow]",
                    classes="hint",
                )

            with VerticalScroll(id="config-scroll"):
                # For top-down, show tabbed interface
                if self._state and self._state.is_topdown:
                    yield from self._compose_topdown_config()
                else:
                    yield from self._compose_standard_config()
                    # Memory estimate only for non-topdown (topdown has per-model estimates)
                    yield from self._compose_memory_estimate()

    def on_mount(self) -> None:
        """Initialize displays when widget is mounted."""
        self._update_shape_display()
        self._update_memory_display()
        self._update_model_info_display()
        self._update_cache_memory_display()
        self._update_cache_options_visibility()
        self._update_imagenet_visibility()
        self._update_wandb_options_visibility()
        self._update_viz_options_visibility()
        self._update_eval_options_visibility()
        # For top-down models, also update centroid and instance-specific displays
        if self._state and self._state.is_topdown:
            self._update_centroid_shape_display()
            self._update_centroid_model_info_display()
            self._update_centroid_cache_memory_display()
            self._update_centroid_cache_options_visibility()
            self._update_instance_cache_memory_display()
            self._update_instance_cache_options_visibility()

    # ==================== TAB NAVIGATION ====================

    def action_prev_tab(self) -> None:
        """Switch to the previous tab (for top-down models)."""
        if not self._state or not self._state.is_topdown:
            return
        try:
            tabs = self.query_one("#topdown-tabs", TabbedContent)
            # Get current tab index and switch to previous
            tab_ids = ["shared-tab", "centroid-tab", "instance-tab"]
            current = tabs.active
            if current in tab_ids:
                idx = tab_ids.index(current)
                new_idx = (idx - 1) % len(tab_ids)
                tabs.active = tab_ids[new_idx]
        except Exception:
            pass

    def action_next_tab(self) -> None:
        """Switch to the next tab (for top-down models)."""
        if not self._state or not self._state.is_topdown:
            return
        try:
            tabs = self.query_one("#topdown-tabs", TabbedContent)
            # Get current tab index and switch to next
            tab_ids = ["shared-tab", "centroid-tab", "instance-tab"]
            current = tabs.active
            if current in tab_ids:
                idx = tab_ids.index(current)
                new_idx = (idx + 1) % len(tab_ids)
                tabs.active = tab_ids[new_idx]
        except Exception:
            pass

    def action_goto_tab_1(self) -> None:
        """Switch to the Shared Settings tab."""
        if not self._state or not self._state.is_topdown:
            return
        try:
            tabs = self.query_one("#topdown-tabs", TabbedContent)
            tabs.active = "shared-tab"
        except Exception:
            pass

    def action_goto_tab_2(self) -> None:
        """Switch to the Centroid Model tab."""
        if not self._state or not self._state.is_topdown:
            return
        try:
            tabs = self.query_one("#topdown-tabs", TabbedContent)
            tabs.active = "centroid-tab"
        except Exception:
            pass

    def action_goto_tab_3(self) -> None:
        """Switch to the Instance Model tab."""
        if not self._state or not self._state.is_topdown:
            return
        try:
            tabs = self.query_one("#topdown-tabs", TabbedContent)
            tabs.active = "instance-tab"
        except Exception:
            pass

    def _update_centroid_shape_display(self) -> None:
        """Update the centroid shape info display."""
        try:
            shape_widget = self.query_one("#centroid-shape-info-display", Static)
        except Exception:
            return

        if not self._state:
            shape_widget.update("[dim]Load SLP file to see shape info[/dim]")
            return

        import math

        # Get dimensions
        orig_h, orig_w = self._state.stats.max_height, self._state.stats.max_width
        eff_h = int(orig_h * self._state._input_scale)
        eff_w = int(orig_w * self._state._input_scale)
        out_h = eff_h // self._state._output_stride
        out_w = eff_w // self._state._output_stride

        # Get channels
        in_channels = 3 if self._state._ensure_rgb else 1

        # Build display text
        text = (
            f"[bold]Shape Pipeline:[/bold]\n"
            f"  Original: [cyan]{orig_w}×{orig_h}[/cyan]\n"
            f"  → Input:  [green]{eff_w}×{eff_h}×{in_channels}[/green] (H×W×C)\n"
            f"  → Output: [yellow]{out_w}×{out_h}×1[/yellow] (centroid confmap)"
        )

        shape_widget.update(text)

    def _update_centroid_model_info_display(self) -> None:
        """Update the centroid model info display (RF, params, encoder/decoder, sigma)."""
        try:
            model_widget = self.query_one("#centroid-model-info-display", Static)
        except Exception:
            return

        if not self._state:
            model_widget.update("[dim]Load SLP file to see model info[/dim]")
            return

        import math

        # Get max animal size from stats
        max_animal_size = self._state.stats.max_bbox_size

        # Compute receptive field based on max_stride
        rf_table = {8: 36, 16: 76, 32: 156, 64: 316, 128: 636}
        rf_pixels = rf_table.get(self._state._max_stride, 76)

        # Scale RF by input scale to get effective RF in original image space
        effective_rf = (
            rf_pixels / self._state._input_scale
            if self._state._input_scale > 0
            else rf_pixels
        )

        # Get parameter estimate
        params = self._state.model_params_estimate
        params_str = f"{params / 1e6:.1f}M" if params >= 1e6 else f"{params / 1e3:.0f}K"

        # Get encoder/decoder blocks
        encoder_blocks = int(math.log2(self._state._max_stride))
        decoder_blocks = (
            int(math.log2(self._state._max_stride / self._state._output_stride))
            if self._state._output_stride > 0
            else encoder_blocks
        )

        # Compute effective Gaussian radius: sigma * output_stride * 2
        effective_sigma_radius = self._state._sigma * self._state._output_stride * 2

        # Check if RF covers the animal
        rf_status = "green" if effective_rf >= max_animal_size else "yellow"
        rf_note = "✓" if effective_rf >= max_animal_size else "⚠ RF < animal"

        # Build display text
        text = (
            f"[bold]Model Architecture:[/bold]\n"
            f"  Parameters: [cyan]~{params_str}[/cyan] | "
            f"Encoder: [yellow]{encoder_blocks} blocks[/yellow] | "
            f"Decoder: [yellow]{decoder_blocks} blocks[/yellow]\n"
            f"  Max Animal Size: [magenta]~{max_animal_size:.0f}px[/magenta] | "
            f"Receptive Field: [{rf_status}]{rf_pixels}px ({effective_rf:.0f}px scaled) {rf_note}[/{rf_status}]\n"
            f"  Gaussian Radius: [blue]{effective_sigma_radius:.0f}px[/blue] (2σ covers 95%)"
        )

        model_widget.update(text)

    def _update_centroid_cache_memory_display(self) -> None:
        """Update the centroid cache memory display (shown when caching is enabled)."""
        try:
            cache_widget = self.query_one("#centroid-cache-memory-display", Static)
        except Exception:
            return

        if not self._state:
            cache_widget.update("")
            return

        from sleap_nn.config_generator.tui.state import DataPipelineType

        # Hide if not caching
        if self._state._data_pipeline == DataPipelineType.TORCH_DATASET:
            cache_widget.update("")
            return

        # Calculate cache memory
        bytes_per_frame = (
            self._state.stats.max_height
            * self._state.stats.max_width
            * self._state.stats.num_channels
        )
        raw_cache_mb = (bytes_per_frame * self._state.stats.num_labeled_frames) / 1e6
        cache_with_overhead = raw_cache_mb * 1.2

        # Check if memory caching with workers > 0
        num_workers = self._state._num_workers
        is_memory_cache = self._state._data_pipeline == DataPipelineType.MEMORY_CACHE
        total_cache_mb = cache_with_overhead

        if is_memory_cache and num_workers > 0:
            # Each worker gets a copy
            total_cache_mb = cache_with_overhead * (1 + num_workers)

        # Format display
        cache_type = "Memory" if is_memory_cache else "Disk"

        # Status color
        if total_cache_mb < 4000:
            status_color = "green"
            status = "OK"
        elif total_cache_mb < 8000:
            status_color = "yellow"
            status = "Large"
        else:
            status_color = "red"
            status = "Very Large"

        text = (
            f"[bold]💾 Image Cache ({cache_type}):[/bold] [{status_color}]{status}[/{status_color}]\n"
            f"  Raw data: {raw_cache_mb:.1f} MB\n"
            f"  With overhead (1.2×): {cache_with_overhead:.1f} MB"
        )

        if is_memory_cache and num_workers > 0:
            text += (
                f"\n  [yellow]⚠ With {num_workers} workers: {total_cache_mb:.1f} MB (×{1 + num_workers})[/yellow]\n"
                f"  [dim]Each worker gets a copy of the cache[/dim]"
            )

        cache_widget.update(text)

    def _update_centroid_cache_options_visibility(self) -> None:
        """Show/hide centroid cache options based on data pipeline selection."""
        if not self._state:
            return

        from sleap_nn.config_generator.tui.state import DataPipelineType

        is_caching = self._state._data_pipeline != DataPipelineType.TORCH_DATASET
        is_disk_cache = self._state._data_pipeline == DataPipelineType.DISK_CACHE

        # Cache options (shown when any caching is enabled)
        cache_option_ids = [
            "centroid-parallel-caching-row",
            "centroid-cache-workers-row",
        ]
        for widget_id in cache_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if is_caching:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

        # Disk-specific options (shown only for disk cache)
        disk_option_ids = [
            "centroid-disk-cache-header",
            "centroid-cache-path-row",
            "centroid-use-existing-row",
            "centroid-delete-cache-row",
        ]
        for widget_id in disk_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if is_disk_cache:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _update_shape_display(self) -> None:
        """Update the shape info display with current configuration."""
        try:
            shape_widget = self.query_one("#shape-info-display", Static)
        except Exception:
            return

        if not self._state:
            shape_widget.update("[dim]Load SLP file to see shape info[/dim]")
            return

        # Get dimensions
        orig_h, orig_w = self._state.stats.max_height, self._state.stats.max_width
        eff_h, eff_w = self._state.effective_height, self._state.effective_width
        out_h, out_w = self._state.output_height, self._state.output_width

        # Get channels
        in_channels = 3 if self._state._ensure_rgb else 1
        num_keypoints = len(self._state.skeleton_nodes)

        # Determine output channels based on model type
        if self._state.is_topdown:
            out_channels = 1  # Centroid outputs 1 channel
        elif self._state.is_bottomup:
            num_edges = (
                len(self._state.stats.edges)
                if hasattr(self._state.stats, "edges")
                else 0
            )
            out_channels = f"{num_keypoints} + {num_edges * 2}"  # confmaps + PAFs
        else:
            out_channels = num_keypoints

        # Build display text
        text = (
            f"[bold]Shape Pipeline:[/bold]\n"
            f"  Original: [cyan]{orig_w}×{orig_h}[/cyan]\n"
            f"  → Input:  [green]{eff_w}×{eff_h}×{in_channels}[/green] (H×W×C)\n"
            f"  → Output: [yellow]{out_w}×{out_h}×{out_channels}[/yellow] (confmaps)"
        )

        shape_widget.update(text)

    def _update_memory_display(self) -> None:
        """Update the memory estimate display(s)."""
        if not self._state:
            return

        if self._state.is_topdown:
            # Update centroid memory display
            self._update_centroid_memory_display()
            # Update instance memory display
            self._update_instance_memory_display()
        else:
            # Update single memory display
            self._update_single_memory_display()

    def _update_single_memory_display(self) -> None:
        """Update the single model memory display."""
        try:
            mem_widget = self.query_one("#memory-info-display", Static)
        except Exception:
            return

        if not self._state:
            mem_widget.update("[dim]Load SLP file to see memory estimate[/dim]")
            return

        try:
            mem = self._state.memory_estimate()
        except Exception:
            mem_widget.update("[dim]Unable to estimate memory[/dim]")
            return

        # Build display text
        status_color = mem.gpu_status
        text = (
            f"[bold]GPU Memory Estimate:[/bold] [{status_color}]~{mem.total_gpu_gb:.1f} GB[/{status_color}]\n"
            f"  Model weights: {mem.model_weights_mb:.0f} MB\n"
            f"  Batch images: {mem.batch_images_mb:.0f} MB\n"
            f"  Activations: {mem.activations_mb:.0f} MB\n"
            f"  Gradients: {mem.gradients_mb:.0f} MB\n"
            f"  [dim]{mem.gpu_message}[/dim]"
        )

        # Add cache memory estimate if caching is enabled
        from sleap_nn.config_generator.tui.state import DataPipelineType

        if self._state._data_pipeline != DataPipelineType.TORCH_DATASET:
            cache_gb = mem.cache_memory_gb
            cache_type = (
                "disk"
                if self._state._data_pipeline == DataPipelineType.DISK_CACHE
                else "RAM"
            )
            # Status color for cache
            if cache_gb < 4:
                cache_color = "green"
            elif cache_gb < 8:
                cache_color = "yellow"
            else:
                cache_color = "red"
            text += (
                f"\n\n[bold]Image Cache ({cache_type}):[/bold] [{cache_color}]~{cache_gb:.1f} GB[/{cache_color}]\n"
                f"  [dim]{mem.cpu_message}[/dim]"
            )

        mem_widget.update(text)

        # Update CSS class for border color
        mem_widget.remove_class("memory-warning", "memory-danger")
        if status_color == "yellow":
            mem_widget.add_class("memory-warning")
        elif status_color == "red":
            mem_widget.add_class("memory-danger")

    def _update_centroid_memory_display(self) -> None:
        """Update the centroid model memory display."""
        try:
            mem_widget = self.query_one("#centroid-memory-display", Static)
        except Exception:
            return

        if not self._state:
            mem_widget.update("[dim]Load SLP file to see memory estimate[/dim]")
            return

        try:
            # Estimate for centroid model (uses main model params with centroid-specific settings)
            from sleap_nn.config_generator.memory import estimate_memory

            mem = estimate_memory(
                self._state.stats,
                self._state._backbone,  # Centroid uses main backbone
                self._state._batch_size,
                self._state._input_scale,
                self._state._output_stride,
                filters=self._state._filters,
                filters_rate=self._state._filters_rate,
                max_stride=self._state._max_stride,
                num_keypoints=len(self._state.skeleton_nodes),
            )
        except Exception:
            mem_widget.update("[dim]Unable to estimate memory[/dim]")
            return

        status_color = mem.gpu_status
        text = (
            f"[bold]Centroid GPU Memory:[/bold] [{status_color}]~{mem.total_gpu_gb:.1f} GB[/{status_color}]\n"
            f"  [dim]{mem.gpu_message}[/dim]"
        )

        # Add cache memory estimate if caching is enabled
        from sleap_nn.config_generator.tui.state import DataPipelineType

        if self._state._data_pipeline != DataPipelineType.TORCH_DATASET:
            cache_gb = mem.cache_memory_gb
            cache_type = (
                "disk"
                if self._state._data_pipeline == DataPipelineType.DISK_CACHE
                else "RAM"
            )
            if cache_gb < 4:
                cache_color = "green"
            elif cache_gb < 8:
                cache_color = "yellow"
            else:
                cache_color = "red"
            text += f"\n[bold]Cache ({cache_type}):[/bold] [{cache_color}]~{cache_gb:.1f} GB[/{cache_color}]"

        mem_widget.update(text)
        mem_widget.remove_class("memory-warning", "memory-danger")
        if status_color == "yellow":
            mem_widget.add_class("memory-warning")
        elif status_color == "red":
            mem_widget.add_class("memory-danger")

    def _update_instance_memory_display(self) -> None:
        """Update the centered instance model memory display."""
        try:
            mem_widget = self.query_one("#instance-memory-display", Static)
        except Exception:
            return

        if not self._state:
            mem_widget.update("[dim]Load SLP file to see memory estimate[/dim]")
            return

        try:
            # Estimate for instance model (uses CI-specific params)
            from sleap_nn.config_generator.memory import estimate_memory

            # Get crop size with fallback
            try:
                crop_size = self._state._crop_size
                if crop_size is None:
                    crop_size = self._state._compute_auto_crop_size()
            except Exception:
                crop_size = 256  # Fallback default

            # Get state attributes with defaults
            ci_backbone = getattr(self._state, "_ci_backbone", "unet_medium_rf")
            ci_batch_size = getattr(self._state, "_ci_batch_size", 4)
            ci_input_scale = getattr(self._state, "_ci_input_scale", 1.0)
            ci_output_stride = getattr(self._state, "_ci_output_stride", 2)
            ci_filters = getattr(self._state, "_ci_filters", 32)
            ci_filters_rate = getattr(self._state, "_ci_filters_rate", 1.5)
            ci_max_stride = getattr(self._state, "_ci_max_stride", 16)

            # Get stats with validation
            stats = self._state.stats
            if stats is None:
                mem_widget.update("[dim]Loading stats...[/dim]")
                return

            # Create modified stats for instance model with all required attributes
            class CropStats:
                def __init__(self, orig_stats, crop_size):
                    self.max_width = crop_size
                    self.max_height = crop_size
                    self.num_frames = getattr(
                        orig_stats, "num_frames", orig_stats.num_labeled_frames
                    )
                    self.num_labeled_frames = orig_stats.num_labeled_frames
                    self.is_rgb = getattr(
                        orig_stats, "is_rgb", orig_stats.num_channels == 3
                    )
                    self.num_channels = orig_stats.num_channels
                    self.num_nodes = getattr(
                        orig_stats,
                        "num_nodes",
                        len(getattr(orig_stats, "node_names", [])),
                    )
                    self.node_names = getattr(orig_stats, "node_names", [])
                    self.nodes = getattr(orig_stats, "nodes", self.node_names)

            crop_stats = CropStats(stats, crop_size)

            # Get num_keypoints safely
            skeleton_nodes = getattr(self._state, "skeleton_nodes", [])
            num_keypoints = (
                len(skeleton_nodes) if skeleton_nodes else crop_stats.num_nodes
            )
            if num_keypoints == 0:
                num_keypoints = 13  # Fallback default

            mem = estimate_memory(
                crop_stats,
                ci_backbone,
                ci_batch_size,
                ci_input_scale,
                ci_output_stride,
                filters=ci_filters,
                filters_rate=ci_filters_rate,
                max_stride=ci_max_stride,
                num_keypoints=num_keypoints,
            )
        except Exception as e:
            # Show more informative error for debugging
            err_msg = str(e) if str(e) else type(e).__name__
            mem_widget.update(f"[dim]Unable to estimate memory: {err_msg}[/dim]")
            return

        status_color = mem.gpu_status
        text = (
            f"[bold]Instance GPU Memory:[/bold] [{status_color}]~{mem.total_gpu_gb:.1f} GB[/{status_color}]\n"
            f"  [dim]{mem.gpu_message}[/dim]"
        )

        # Add cache memory estimate if caching is enabled
        # Note: Instance model caches full-frame images, same as centroid
        from sleap_nn.config_generator.tui.state import DataPipelineType

        if self._state._data_pipeline != DataPipelineType.TORCH_DATASET:
            # Calculate cache based on original image size, not crop
            full_mem = self._state.memory_estimate()
            cache_gb = full_mem.cache_memory_gb
            cache_type = (
                "disk"
                if self._state._data_pipeline == DataPipelineType.DISK_CACHE
                else "RAM"
            )
            if cache_gb < 4:
                cache_color = "green"
            elif cache_gb < 8:
                cache_color = "yellow"
            else:
                cache_color = "red"
            text += f"\n[bold]Cache ({cache_type}):[/bold] [{cache_color}]~{cache_gb:.1f} GB[/{cache_color}]"

        mem_widget.update(text)
        mem_widget.remove_class("memory-warning", "memory-danger")
        if status_color == "yellow":
            mem_widget.add_class("memory-warning")
        elif status_color == "red":
            mem_widget.add_class("memory-danger")

    def _update_cache_memory_display(self) -> None:
        """Update the cache memory display (shown when caching is enabled)."""
        try:
            cache_widget = self.query_one("#cache-memory-display", Static)
        except Exception:
            return

        if not self._state:
            cache_widget.update("")
            return

        from sleap_nn.config_generator.tui.state import DataPipelineType

        # Hide if not caching
        if self._state._data_pipeline == DataPipelineType.TORCH_DATASET:
            cache_widget.update("")
            return

        # Calculate cache memory
        bytes_per_frame = (
            self._state.stats.max_height
            * self._state.stats.max_width
            * self._state.stats.num_channels
        )
        raw_cache_mb = (bytes_per_frame * self._state.stats.num_labeled_frames) / 1e6
        cache_with_overhead = raw_cache_mb * 1.2

        # Check if memory caching with workers > 0
        num_workers = self._state._num_workers
        is_memory_cache = self._state._data_pipeline == DataPipelineType.MEMORY_CACHE
        total_cache_mb = cache_with_overhead

        if is_memory_cache and num_workers > 0:
            # Each worker gets a copy
            total_cache_mb = cache_with_overhead * (1 + num_workers)

        # Format display
        cache_type = "Memory" if is_memory_cache else "Disk"

        # Status color
        if total_cache_mb < 4000:
            status_color = "green"
            status = "OK"
        elif total_cache_mb < 8000:
            status_color = "yellow"
            status = "Large"
        else:
            status_color = "red"
            status = "Very Large"

        text = (
            f"[bold]💾 Image Cache ({cache_type}):[/bold] [{status_color}]{status}[/{status_color}]\n"
            f"  Raw data: {raw_cache_mb:.1f} MB\n"
            f"  With overhead (1.2×): {cache_with_overhead:.1f} MB"
        )

        if is_memory_cache and num_workers > 0:
            text += (
                f"\n  [yellow]⚠ With {num_workers} workers: {total_cache_mb:.1f} MB (×{1 + num_workers})[/yellow]\n"
                f"  [dim]Each worker gets a copy of the cache[/dim]"
            )

        cache_widget.update(text)

    def _update_cache_options_visibility(self) -> None:
        """Show/hide cache options based on data pipeline selection."""
        if not self._state:
            return

        from sleap_nn.config_generator.tui.state import DataPipelineType

        is_caching = self._state._data_pipeline != DataPipelineType.TORCH_DATASET
        is_disk_cache = self._state._data_pipeline == DataPipelineType.DISK_CACHE

        # Cache options (shown when any caching is enabled)
        cache_option_ids = ["parallel-caching-row", "cache-workers-row"]
        for widget_id in cache_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if is_caching:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

        # Disk-specific options (shown only for disk cache)
        disk_option_ids = [
            "disk-cache-header",
            "cache-path-row",
            "use-existing-row",
            "delete-cache-row",
        ]
        for widget_id in disk_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if is_disk_cache:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _update_imagenet_visibility(self) -> None:
        """Show/hide ImageNet weights option based on backbone selection.

        ImageNet pretrained weights are only available for ConvNeXt and SwinT,
        not for UNet backbones.
        """
        if not self._state:
            return

        # Check if backbone supports ImageNet pretrained weights
        backbone = self._state._backbone
        is_pretrained_backbone = backbone.startswith("convnext") or backbone.startswith(
            "swint"
        )

        # Standard model ImageNet row
        try:
            widget = self.query_one("#imagenet-pretrained-row")
            if is_pretrained_backbone:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")
        except Exception:
            pass

        # Centroid model ImageNet row
        try:
            widget = self.query_one("#centroid-imagenet-row")
            if is_pretrained_backbone:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")
        except Exception:
            pass

        # Instance model ImageNet row
        try:
            widget = self.query_one("#instance-imagenet-row")
            if is_pretrained_backbone:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")
        except Exception:
            pass

    def _update_instance_cache_options_visibility(self) -> None:
        """Show/hide instance cache options based on data pipeline selection."""
        if not self._state:
            return

        from sleap_nn.config_generator.tui.state import DataPipelineType

        is_caching = self._state._data_pipeline != DataPipelineType.TORCH_DATASET
        is_disk_cache = self._state._data_pipeline == DataPipelineType.DISK_CACHE

        # Cache options (shown when any caching is enabled)
        cache_option_ids = [
            "instance-parallel-caching-row",
            "instance-cache-workers-row",
        ]
        for widget_id in cache_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if is_caching:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

        # Disk-specific options (shown only for disk cache)
        disk_option_ids = [
            "instance-disk-cache-header",
            "instance-cache-path-row",
            "instance-use-existing-row",
            "instance-delete-cache-row",
        ]
        for widget_id in disk_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if is_disk_cache:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _update_instance_cache_memory_display(self) -> None:
        """Update the instance cache memory display.

        Note: Both centroid and instance models cache the same full-frame images.
        Cropping happens after loading from cache, so the cache size is identical.
        """
        try:
            cache_widget = self.query_one("#instance-cache-memory-display", Static)
        except Exception:
            return

        if not self._state:
            cache_widget.update("")
            return

        from sleap_nn.config_generator.tui.state import DataPipelineType

        # Hide if not caching
        if self._state._data_pipeline == DataPipelineType.TORCH_DATASET:
            cache_widget.update("")
            return

        # Cache is based on FULL FRAME size (same as centroid) - cropping happens after loading
        bytes_per_frame = (
            self._state.stats.max_height
            * self._state.stats.max_width
            * self._state.stats.num_channels
        )
        raw_cache_mb = (bytes_per_frame * self._state.stats.num_labeled_frames) / 1e6
        cache_with_overhead = raw_cache_mb * 1.2

        num_workers = self._state._num_workers
        is_memory_cache = self._state._data_pipeline == DataPipelineType.MEMORY_CACHE
        total_cache_mb = cache_with_overhead

        if is_memory_cache and num_workers > 0:
            # Each worker gets a copy
            total_cache_mb = cache_with_overhead * (1 + num_workers)

        cache_type = "Memory" if is_memory_cache else "Disk"

        if total_cache_mb < 4000:
            status_color = "green"
            status = "OK"
        elif total_cache_mb < 8000:
            status_color = "yellow"
            status = "Large"
        else:
            status_color = "red"
            status = "Very Large"

        text = (
            f"[bold]💾 Image Cache ({cache_type}):[/bold] [{status_color}]{status}[/{status_color}]\n"
            f"  Raw data: {raw_cache_mb:.1f} MB\n"
            f"  With overhead (1.2×): {cache_with_overhead:.1f} MB"
        )

        if is_memory_cache and num_workers > 0:
            text += (
                f"\n  [yellow]⚠ With {num_workers} workers: {total_cache_mb:.1f} MB (×{1 + num_workers})[/yellow]\n"
                f"  [dim]Each worker gets a copy of the cache[/dim]"
            )

        cache_widget.update(text)

    def _update_wandb_options_visibility(self) -> None:
        """Show/hide WandB options based on Enable WandB checkbox.

        WandB configuration options are only shown when WandB logging is enabled.
        """
        if not self._state:
            return

        wandb_enabled = self._state._wandb.enabled if self._state._wandb else False

        # Standard model WandB options
        wandb_option_ids = [
            "wandb-project-row",
            "wandb-entity-row",
            "wandb-name-row",
            "wandb-api-key-row",
            "wandb-mode-row",
            "wandb-viz-row",
            "wandb-save-viz-row",
        ]
        for widget_id in wandb_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if wandb_enabled:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

        # Centroid model WandB options
        centroid_wandb_ids = [
            "centroid-wandb-project-row",
            "centroid-wandb-entity-row",
            "centroid-wandb-name-row",
            "centroid-wandb-api-key-row",
            "centroid-wandb-mode-row",
            "centroid-wandb-viz-row",
            "centroid-wandb-save-viz-row",
        ]
        for widget_id in centroid_wandb_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if wandb_enabled:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

        # Instance model WandB options
        instance_wandb_ids = ["instance-wandb-project-row", "instance-wandb-entity-row"]
        for widget_id in instance_wandb_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if wandb_enabled:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _update_viz_options_visibility(self) -> None:
        """Show/hide viz options based on Save Visualizations checkbox."""
        if not self._state:
            return

        viz_enabled = self._state._visualize_preds

        # All model viz options (standard, centroid, instance)
        viz_option_ids = [
            "viz-keep-folder-row",
            "centroid-viz-keep-folder-row",
            "instance-viz-keep-folder-row",
        ]
        for widget_id in viz_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if viz_enabled:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _update_eval_options_visibility(self) -> None:
        """Show/hide evaluation options based on Enable Evaluation checkbox."""
        if not self._state:
            return

        eval_enabled = (
            self._state._evaluation.enabled if self._state._evaluation else False
        )

        # All model eval options (standard, centroid, instance)
        eval_option_ids = [
            "eval-frequency-row",
            "eval-oks-row",
            "centroid-eval-frequency-row",
            "centroid-eval-oks-row",
            "instance-eval-frequency-row",
            "instance-eval-oks-row",
        ]
        for widget_id in eval_option_ids:
            try:
                widget = self.query_one(f"#{widget_id}")
                if eval_enabled:
                    widget.remove_class("hidden")
                else:
                    widget.add_class("hidden")
            except Exception:
                pass

    def _update_model_info_display(self) -> None:
        """Update the model info display (RF, params, animal size, effective sigma)."""
        try:
            model_widget = self.query_one("#model-info-display", Static)
        except Exception:
            return

        if not self._state:
            model_widget.update("[dim]Load SLP file to see model info[/dim]")
            return

        import math

        # Get max animal size from stats
        max_animal_size = self._state.stats.max_bbox_size

        # Compute receptive field based on max_stride
        # RF = 1 + sum((kernel[l] - 1) * prod(strides[:l]))
        # Pre-computed values for UNet: stride 8->36, 16->76, 32->156, 64->316
        rf_table = {8: 36, 16: 76, 32: 156, 64: 316, 128: 636}
        rf_pixels = rf_table.get(self._state._max_stride, 76)

        # Scale RF by input scale to get effective RF in original image space
        effective_rf = (
            rf_pixels / self._state._input_scale
            if self._state._input_scale > 0
            else rf_pixels
        )

        # Get parameter estimate
        params = self._state.model_params_estimate
        params_str = f"{params / 1e6:.1f}M" if params >= 1e6 else f"{params / 1e3:.0f}K"

        # Get encoder/decoder blocks
        encoder_blocks = int(math.log2(self._state._max_stride))
        decoder_blocks = (
            int(math.log2(self._state._max_stride / self._state._output_stride))
            if self._state._output_stride > 0
            else encoder_blocks
        )

        # Compute effective Gaussian radius: sigma * output_stride * 2
        effective_sigma_radius = self._state._sigma * self._state._output_stride * 2

        # Check if RF covers the animal
        rf_status = "green" if effective_rf >= max_animal_size else "yellow"
        rf_note = "✓" if effective_rf >= max_animal_size else "⚠ RF < animal"

        # Build display text
        text = (
            f"[bold]Model Architecture:[/bold]\n"
            f"  Parameters: [cyan]~{params_str}[/cyan] | "
            f"Encoder: [yellow]{encoder_blocks} blocks[/yellow] | "
            f"Decoder: [yellow]{decoder_blocks} blocks[/yellow]\n"
            f"  Max Animal Size: [magenta]~{max_animal_size:.0f}px[/magenta] | "
            f"Receptive Field: [{rf_status}]{rf_pixels}px ({effective_rf:.0f}px scaled) {rf_note}[/{rf_status}]\n"
            f"  Gaussian Radius: [blue]{effective_sigma_radius:.0f}px[/blue] (2σ covers 95%)"
        )

        model_widget.update(text)

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

        # Data Pipeline section (visible - NOT collapsible per webapp design)
        with Container(classes="section-box"):
            yield Static("[bold]Data Pipeline[/bold]", classes="section-header")
            yield from self._compose_data_pipeline_section()

        # Additional Data Parameters (collapsible)
        with Collapsible(
            title="Additional Data Parameters", classes="collapsible-section"
        ):
            yield from self._compose_additional_data_params()

        # Model section
        with Container(classes="section-box"):
            yield Static("[bold]Model Architecture[/bold]", classes="section-header")
            yield from self._compose_model_params()

        # Model-specific sections (PAF for bottom-up, class vector for multi-class)
        if self._state and self._state.is_bottomup:
            with Container(classes="model-specific-section"):
                yield Static(
                    "[bold]Part Affinity Fields (PAF)[/bold]", classes="section-header"
                )
                yield from self._compose_paf_params()

        if self._state and self._state.is_multiclass:
            with Container(classes="model-specific-section"):
                yield Static(
                    "[bold]Identity Classification[/bold]", classes="section-header"
                )
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
            yield Label(
                "Resize factor (0.125-1.0) - lower = faster training",
                classes="param-hint",
            )

        # Show shape pipeline info
        if self._state:
            yield Static(id="shape-info-display", classes="shape-info")
            # Initial content will be set by _update_shape_display()

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

        # Translate
        with Horizontal(classes="param-row"):
            yield Checkbox(
                "Translate",
                value=aug.translate_enabled if aug else False,
                id="translate-checkbox",
            )
            yield Label("±", classes="param-label")
            yield Input(
                value=str(aug.translate if aug else 0),
                id="translate-input",
                classes="param-input",
                type="number",
            )
            yield Label("% (shift in x/y)", classes="param-hint")

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

    def _compose_data_pipeline_section(self) -> ComposeResult:
        """Compose data pipeline section (visible, not collapsible - matches webapp)."""
        # Speed tip info box (matching webapp)
        yield Static(
            "[green bold]💡 Speed Tip:[/green bold] Set [bold]Num Workers[/bold] to 2+ and "
            "use [bold]Cache to Memory[/bold] or [bold]Cache to Disk[/bold] for faster training.",
            classes="info-box speed-tip",
        )

        # Num Workers + Data Pipeline side by side (matching webapp grid-2 layout)
        with Horizontal(classes="param-row"):
            yield Label("Num Workers:", classes="param-label")
            yield Select(
                [
                    ("0 (Default)", "0"),
                    ("2", "2"),
                    ("4", "4"),
                    ("8", "8"),
                ],
                value=str(self._state._num_workers if self._state else 0),
                id="num-workers-select",
                classes="param-input",
            )

        current_pipeline = (
            self._state._data_pipeline.value if self._state else "torch_dataset"
        )
        with Horizontal(classes="param-row"):
            yield Label("Data Pipeline:", classes="param-label")
            yield Select(
                [
                    ("Video (default)", "torch_dataset"),
                    ("Cache to Memory", "litdata"),
                    ("Cache to Disk", "litdata_disk"),
                ],
                value=current_pipeline,
                id="data-pipeline-select",
                classes="param-input",
            )
            yield Label("Data loading method", classes="param-hint")

        # Cache Memory Estimate (shown when caching is enabled)
        yield Static(id="cache-memory-display", classes="memory-info")

        # Cache config options (shown when caching is enabled)
        cache = self._state._cache_config if self._state else None

        # Parallel Caching
        with Horizontal(classes="param-row", id="parallel-caching-row"):
            yield Label("Parallel Caching:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=cache.parallel_caching if cache else True,
                id="parallel-caching-checkbox",
            )
            yield Label("Cache images in parallel", classes="param-hint")

        # Cache Workers (for caching process)
        with Horizontal(classes="param-row", id="cache-workers-row"):
            yield Label("Cache Workers:", classes="param-label")
            yield Input(
                value=str(cache.cache_workers if cache else 0),
                id="cache-workers-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Workers for caching (0 = main process)", classes="param-hint")

        # Disk cache specific options
        yield Static(
            "[dim]Disk Cache Options[/dim]",
            classes="subsection-header",
            id="disk-cache-header",
        )

        # Cache Image Path
        with Horizontal(classes="param-row", id="cache-path-row"):
            yield Label("Cache Path:", classes="param-label")
            yield Input(
                value=cache.cache_img_path if cache else "",
                id="cache-path-input",
                classes="param-input-wide",
                placeholder="./cache (auto if empty)",
            )

        # Use Existing Images
        with Horizontal(classes="param-row", id="use-existing-row"):
            yield Label("Use Existing:", classes="param-label")
            yield Checkbox(
                "Reuse cached images",
                value=cache.use_existing_imgs if cache else False,
                id="use-existing-checkbox",
            )
            yield Label("Skip caching if images exist", classes="param-hint")

        # Delete Cache After Training
        with Horizontal(classes="param-row", id="delete-cache-row"):
            yield Label("Delete After:", classes="param-label")
            yield Checkbox(
                "Delete cache after training",
                value=cache.delete_cache_after_training if cache else True,
                id="delete-cache-checkbox",
            )
            yield Label("Clean up disk space", classes="param-hint")

    def _compose_additional_data_params(self) -> ComposeResult:
        """Compose additional data parameters (collapsible section)."""
        # User Instances Only
        with Horizontal(classes="param-row"):
            yield Label("User Instances:", classes="param-label")
            yield Checkbox(
                "Only user-labeled instances",
                value=self._state._user_instances_only if self._state else True,
                id="user-instances-checkbox",
            )
            yield Label("Filter out predicted instances", classes="param-hint")

        # Validation Fraction
        with Horizontal(classes="param-row"):
            yield Label("Val Fraction:", classes="param-label")
            yield Input(
                value=str(self._state._validation_fraction if self._state else 0.1),
                id="val-fraction-input",
                classes="param-input",
                type="number",
            )
            yield Label(
                "Fraction held out for validation (0.05-0.3)", classes="param-hint"
            )

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
                value=(
                    str(self._state._max_height)
                    if self._state and self._state._max_height
                    else ""
                ),
                id="max-height-input",
                classes="param-input",
                type="integer",
                placeholder="auto",
            )
            yield Label("Max Width:", classes="param-label")
            yield Input(
                value=(
                    str(self._state._max_width)
                    if self._state and self._state._max_width
                    else ""
                ),
                id="max-width-input",
                classes="param-input",
                type="integer",
                placeholder="auto",
            )

    # ==================== MODEL PARAMETERS ====================

    def _compose_model_params(self) -> ComposeResult:
        """Compose model architecture parameters."""
        # Model info display (RF and params)
        if self._state:
            yield Static(id="model-info-display", classes="shape-info")

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
                [
                    ("1 (full res)", "1"),
                    ("2 (half res)", "2"),
                    ("4 (quarter)", "4"),
                    ("8 (1/8 res)", "8"),
                    ("16 (1/16 res)", "16"),
                ],
                value=str(self._state._output_stride if self._state else 2),
                id="output-stride-select",
                classes="param-input",
            )
            yield Label("Higher = faster but less precise", classes="param-hint")

        # ImageNet Pretrained (for ConvNeXt/SwinT)
        with Horizontal(classes="param-row", id="imagenet-pretrained-row"):
            yield Label("ImageNet Weights:", classes="param-label")
            yield Checkbox(
                "Use ImageNet pretrained",
                value=self._state._use_imagenet_pretrained if self._state else True,
                id="imagenet-pretrained-checkbox",
            )
            yield Label("For ConvNeXt/SwinT backbones", classes="param-hint")

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
            yield Label(
                "Vector field width (10-20 precise, 15 default)", classes="param-hint"
            )

        # PAF Output Stride
        with Horizontal(classes="param-row"):
            yield Label("PAF Output Stride:", classes="param-label")
            yield Select(
                [
                    ("1 (full res)", "1"),
                    ("2 (half res)", "2"),
                    ("4 (quarter)", "4"),
                    ("8 (1/8 res)", "8"),
                    ("16 (1/16 res)", "16"),
                ],
                value=str(paf.output_stride if paf else 4),
                id="paf-output-stride-select",
                classes="param-input",
            )
            yield Label("Usually 2× confmaps stride", classes="param-hint")

        # Confmaps Loss Weight
        with Horizontal(classes="param-row"):
            yield Label("Confmaps Loss:", classes="param-label")
            yield Input(
                value=str(paf.confmaps_loss_weight if paf else 1.0),
                id="confmaps-loss-weight-input",
                classes="param-input",
                type="number",
            )
            yield Label(
                "Loss weight for keypoint heatmaps (1.0 default)", classes="param-hint"
            )

        # PAF Loss Weight
        with Horizontal(classes="param-row"):
            yield Label("PAF Loss Weight:", classes="param-label")
            yield Input(
                value=str(paf.loss_weight if paf else 1.0),
                id="paf-loss-weight-input",
                classes="param-input",
                type="number",
            )
            yield Label(
                "Increase to emphasize limb grouping (1.0 default)",
                classes="param-hint",
            )

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
            yield Label(
                "Fully-connected layers for classification", classes="param-hint"
            )

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
        """Compose training hyperparameters (matching centroid layout)."""
        # Basic params visible (like centroid)
        # Batch size
        with Horizontal(classes="param-row"):
            yield Label("Batch Size:", classes="param-label")
            yield Select(
                [
                    ("1", "1"),
                    ("2", "2"),
                    ("4", "4"),
                    ("8", "8"),
                    ("16", "16"),
                    ("32", "32"),
                ],
                value=str(self._state._batch_size if self._state else 4),
                id="batch-size-select",
                classes="param-input",
            )
            yield Label("Samples per batch", classes="param-hint")

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
            yield Label(
                "Initial optimizer step size (1e-4 is good default)",
                classes="param-hint",
            )

        # Advanced training params in collapsible
        with Collapsible(title="Advanced Training", classes="collapsible-section"):
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

            # Accelerator
            with Horizontal(classes="param-row"):
                yield Label("Accelerator:", classes="param-label")
                yield Select(
                    [
                        ("Auto", "auto"),
                        ("GPU", "gpu"),
                        ("MPS (Apple Silicon)", "mps"),
                        ("CPU", "cpu"),
                    ],
                    value=self._state._accelerator if self._state else "auto",
                    id="accelerator-select",
                    classes="param-input",
                )
                yield Label("Hardware for training", classes="param-hint")

            yield Static("[dim]Early Stopping[/dim]", classes="subsection-header")

            # Early stopping
            with Horizontal(classes="param-row"):
                yield Label("Enable:", classes="param-label")
                yield Checkbox(
                    "Stop if validation plateaus",
                    value=self._state._early_stopping if self._state else True,
                    id="early-stopping-checkbox",
                )

            # Early stopping patience
            with Horizontal(classes="param-row", id="es-patience-row"):
                yield Label("Patience:", classes="param-label")
                yield Input(
                    value=str(
                        self._state._early_stopping_patience if self._state else 10
                    ),
                    id="es-patience-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs without improvement", classes="param-hint")

            # Early stopping min delta
            with Horizontal(classes="param-row", id="es-min-delta-row"):
                yield Label("Min Delta:", classes="param-label")
                yield Input(
                    value=str(
                        self._state._early_stopping_min_delta if self._state else 1e-8
                    ),
                    id="es-min-delta-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Minimum improvement threshold", classes="param-hint")

            # Min Steps per Epoch
            with Horizontal(classes="param-row"):
                yield Label("Min Steps/Epoch:", classes="param-label")
                yield Input(
                    value=str(self._state._min_steps_per_epoch if self._state else 200),
                    id="min-steps-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Minimum training steps per epoch", classes="param-hint")

            # Random Seed
            with Horizontal(classes="param-row"):
                yield Label("Random Seed:", classes="param-label")
                yield Input(
                    value=(
                        str(self._state._random_seed)
                        if self._state and self._state._random_seed
                        else ""
                    ),
                    id="random-seed-input",
                    classes="param-input",
                    type="integer",
                    placeholder="None (random)",
                )
                yield Label("For reproducibility", classes="param-hint")

            yield Static(
                "[dim]Learning Rate Scheduler[/dim]", classes="subsection-header"
            )

            # LR Scheduler
            scheduler_type = (
                self._state._scheduler.type.value if self._state else "none"
            )
            with Horizontal(classes="param-row"):
                yield Label("Scheduler:", classes="param-label")
                yield Select(
                    [
                        ("Reduce on Plateau", "ReduceLROnPlateau"),
                        ("Cosine Annealing + Warmup", "CosineAnnealingWarmup"),
                        ("Linear Warmup + Decay", "LinearWarmupLinearDecay"),
                        ("Step LR", "StepLR"),
                        ("None (Constant)", "none"),
                    ],
                    value=(
                        scheduler_type
                        if scheduler_type != "none"
                        else "ReduceLROnPlateau"
                    ),
                    id="scheduler-select",
                    classes="param-input",
                )
                yield Label("LR decay strategy", classes="param-hint")

            # Scheduler-specific parameters
            yield from self._compose_scheduler_params()

    def _compose_scheduler_params(self) -> ComposeResult:
        """Compose scheduler-specific parameters based on selected scheduler type."""
        scheduler = self._state._scheduler if self._state else None
        scheduler_type = scheduler.type if scheduler else SchedulerType.NONE

        # ReduceLROnPlateau params
        with Container(
            id="scheduler-reduce-params",
            classes=(
                "hidden" if scheduler_type != SchedulerType.REDUCE_ON_PLATEAU else ""
            ),
        ):
            with Horizontal(classes="param-row"):
                yield Label("Factor:", classes="param-label")
                yield Input(
                    value=str(scheduler.factor if scheduler else 0.5),
                    id="scheduler-factor-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("LR multiply factor on plateau", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Patience:", classes="param-label")
                yield Input(
                    value=str(scheduler.plateau_patience if scheduler else 5),
                    id="scheduler-patience-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs to wait before reducing", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Min LR:", classes="param-label")
                yield Input(
                    value=str(scheduler.min_lr if scheduler else 1e-8),
                    id="scheduler-min-lr-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Minimum learning rate", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Cooldown:", classes="param-label")
                yield Input(
                    value=str(scheduler.cooldown if scheduler else 3),
                    id="scheduler-cooldown-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs to wait after LR reduction", classes="param-hint")

        # StepLR params
        with Container(
            id="scheduler-step-params",
            classes="hidden" if scheduler_type != SchedulerType.STEP_LR else "",
        ):
            with Horizontal(classes="param-row"):
                yield Label("Step Size:", classes="param-label")
                yield Input(
                    value=str(scheduler.step_size if scheduler else 10),
                    id="scheduler-step-size-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs between LR decay", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Gamma:", classes="param-label")
                yield Input(
                    value=str(scheduler.gamma if scheduler else 0.1),
                    id="scheduler-gamma-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("LR decay multiplier", classes="param-hint")

        # CosineAnnealingWarmup params
        with Container(
            id="scheduler-cosine-params",
            classes=(
                "hidden"
                if scheduler_type != SchedulerType.COSINE_ANNEALING_WARMUP
                else ""
            ),
        ):
            with Horizontal(classes="param-row"):
                yield Label("Warmup Epochs:", classes="param-label")
                yield Input(
                    value=str(scheduler.warmup_epochs if scheduler else 5),
                    id="scheduler-warmup-epochs-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs for warmup phase", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Warmup Start LR:", classes="param-label")
                yield Input(
                    value=str(scheduler.warmup_start_lr if scheduler else 0),
                    id="scheduler-warmup-start-lr-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Initial LR during warmup", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Eta Min:", classes="param-label")
                yield Input(
                    value=str(scheduler.eta_min if scheduler else 0),
                    id="scheduler-eta-min-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Minimum LR after cosine decay", classes="param-hint")

        # LinearWarmupLinearDecay params
        with Container(
            id="scheduler-linear-params",
            classes=(
                "hidden"
                if scheduler_type != SchedulerType.LINEAR_WARMUP_LINEAR_DECAY
                else ""
            ),
        ):
            with Horizontal(classes="param-row"):
                yield Label("Warmup Epochs:", classes="param-label")
                yield Input(
                    value=str(scheduler.linear_warmup_epochs if scheduler else 5),
                    id="scheduler-linear-warmup-epochs-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs for warmup phase", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Warmup Start LR:", classes="param-label")
                yield Input(
                    value=str(scheduler.linear_warmup_start_lr if scheduler else 0),
                    id="scheduler-linear-warmup-start-lr-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Initial LR during warmup", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("End LR:", classes="param-label")
                yield Input(
                    value=str(scheduler.end_lr if scheduler else 0),
                    id="scheduler-end-lr-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Final LR after linear decay", classes="param-hint")

    # ==================== CHECKPOINT PARAMETERS ====================

    def _compose_checkpoint_params(self) -> ComposeResult:
        """Compose checkpoint and logging parameters."""
        ckpt = self._state._checkpoint if self._state else None

        # Save Checkpoints (master toggle)
        with Horizontal(classes="param-row"):
            yield Label("Save Checkpoints:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=ckpt.enabled if ckpt else True,
                id="save-ckpt-checkbox",
            )
            yield Label("Enable checkpoint saving", classes="param-hint")

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
                value=ckpt.save_last if ckpt else False,
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

        # WandB Project (hidden until WandB enabled)
        with Horizontal(classes="param-row hidden", id="wandb-project-row"):
            yield Label("WandB Project:", classes="param-label")
            yield Input(
                value=wandb.project if wandb else "sleap-training",
                id="wandb-project-input",
                classes="param-input-wide",
            )

        # WandB Entity (hidden until WandB enabled)
        with Horizontal(classes="param-row hidden", id="wandb-entity-row"):
            yield Label("WandB Entity:", classes="param-label")
            yield Input(
                value=wandb.entity if wandb else "",
                id="wandb-entity-input",
                classes="param-input-wide",
                placeholder="username or team",
            )

        # WandB Run Name (hidden until WandB enabled)
        with Horizontal(classes="param-row hidden", id="wandb-name-row"):
            yield Label("WandB Name:", classes="param-label")
            yield Input(
                value=wandb.name if wandb else "",
                id="wandb-name-input",
                classes="param-input-wide",
                placeholder="auto-generated",
            )

        # WandB API Key (hidden until WandB enabled)
        with Horizontal(classes="param-row hidden", id="wandb-api-key-row"):
            yield Label("WandB API Key:", classes="param-label")
            yield Input(
                value=wandb.api_key if wandb else "",
                id="wandb-api-key-input",
                classes="param-input-wide",
                placeholder="(uses env var if empty)",
                password=True,
            )

        # WandB Mode (hidden until WandB enabled)
        with Horizontal(classes="param-row hidden", id="wandb-mode-row"):
            yield Label("WandB Mode:", classes="param-label")
            yield Select(
                [
                    ("Online", "online"),
                    ("Offline", "offline"),
                    ("Disabled", "disabled"),
                ],
                value=wandb.mode if wandb else "online",
                id="wandb-mode-select",
                classes="param-input",
            )
            yield Label("Logging mode", classes="param-hint")

        # WandB Visualization options (hidden until WandB enabled)
        with Horizontal(classes="param-row hidden", id="wandb-viz-row"):
            yield Label("Log Visualizations:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=wandb.viz_enabled if wandb else True,
                id="wandb-viz-checkbox",
            )
            yield Label("Log prediction visualizations to WandB", classes="param-hint")

        with Horizontal(classes="param-row hidden", id="wandb-save-viz-row"):
            yield Label("Upload Local Viz:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=wandb.save_viz_imgs if wandb else False,
                id="wandb-save-viz-checkbox",
            )
            yield Label("Upload local visualization images", classes="param-hint")

        yield Static("[dim]Save Visualizations[/dim]", classes="subsection-header")

        # Save Visualizations toggle (matches webapp enable-viz)
        with Horizontal(classes="param-row"):
            yield Label("Save Visualizations:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=self._state._visualize_preds if self._state else False,
                id="save-viz-checkbox",
            )
            yield Label("Save prediction visualizations locally", classes="param-hint")

        # Keep Viz Folder (hidden until Save Viz enabled - matches webapp)
        with Horizontal(classes="param-row hidden", id="viz-keep-folder-row"):
            yield Label("Keep Viz Folder:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=self._state._keep_viz if self._state else False,
                id="viz-keep-folder-checkbox",
            )
            yield Label(
                "Keep visualization folder after training", classes="param-hint"
            )

        yield Static("[dim]Evaluation[/dim]", classes="subsection-header")

        # Enable Evaluation toggle (matches webapp enable-eval)
        evaluation = self._state._evaluation if self._state else None
        with Horizontal(classes="param-row"):
            yield Label("Enable Evaluation:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=evaluation.enabled if evaluation else False,
                id="eval-checkbox",
            )
            yield Label("Run OKS evaluation during training", classes="param-hint")

        # Eval options (hidden until Enable Eval is checked)
        with Horizontal(classes="param-row hidden", id="eval-frequency-row"):
            yield Label("Eval Frequency:", classes="param-label")
            yield Input(
                value=str(evaluation.frequency if evaluation else 1),
                id="eval-frequency-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Run eval every N epochs", classes="param-hint")

        with Horizontal(classes="param-row hidden", id="eval-oks-row"):
            yield Label("OKS Stddev:", classes="param-label")
            yield Input(
                value=str(evaluation.oks_stddev if evaluation else 0.025),
                id="eval-oks-input",
                classes="param-input",
                type="number",
            )
            yield Label("Standard deviation for OKS scoring", classes="param-hint")

    # ==================== ADVANCED PARAMETERS ====================

    def _compose_advanced_params(self) -> ComposeResult:
        """Compose advanced training parameters (less commonly changed)."""
        # Devices (number of GPUs)
        with Horizontal(classes="param-row"):
            yield Label("Devices:", classes="param-label")
            yield Select(
                [
                    ("Auto", "auto"),
                    ("1", "1"),
                    ("2", "2"),
                    ("4", "4"),
                    ("8", "8"),
                ],
                value=self._state._devices if self._state else "auto",
                id="devices-select",
                classes="param-input",
            )
            yield Label("Number of GPUs to use", classes="param-hint")

        # Progress Bar (moved from Logging to Advanced)
        with Horizontal(classes="param-row"):
            yield Label("Progress Bar:", classes="param-label")
            yield Checkbox(
                "Enabled",
                value=self._state._enable_progress_bar if self._state else True,
                id="progress-bar-checkbox",
            )
            yield Label("Show progress during training", classes="param-hint")

        yield Static(
            "[dim]Online Hard Keypoint Mining (OHKM)[/dim]", classes="subsection-header"
        )

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
                value=str(ohkm.loss_scale if ohkm else 5.0),
                id="ohkm-scale-input",
                classes="param-input",
                type="number",
            )
            yield Label("Scale factor for hard keypoint losses", classes="param-hint")

        with Horizontal(classes="param-row"):
            yield Label("Min Hard Keypoints:", classes="param-label")
            yield Input(
                value=str(ohkm.min_hard_keypoints if ohkm else 2),
                id="ohkm-min-hard-input",
                classes="param-input",
                type="integer",
            )
            yield Label("Minimum keypoints to classify as hard", classes="param-hint")

        with Horizontal(classes="param-row"):
            yield Label("Max Hard Keypoints:", classes="param-label")
            yield Input(
                value=(
                    str(ohkm.max_hard_keypoints)
                    if ohkm and ohkm.max_hard_keypoints
                    else ""
                ),
                id="ohkm-max-hard-input",
                classes="param-input",
                type="integer",
                placeholder="none",
            )
            yield Label("Maximum (empty = unlimited)", classes="param-hint")

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

            current_value = (
                self._state._anchor_part
                if self._state and self._state._anchor_part
                else ""
            )

            with Horizontal(classes="param-row"):
                yield Label("Anchor Part:", classes="param-label")
                yield Select(
                    options,
                    value=current_value,
                    id="anchor-part-select",
                    classes="param-input-wide",
                )

    def _compose_centroid_params(self) -> ComposeResult:
        """Compose centroid model specific parameters (full configuration like web app)."""
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
                yield Label(
                    "Lower scale OK for centroids (0.5 typical)", classes="param-hint"
                )

            # Shape info display
            yield Static(id="centroid-shape-info-display", classes="shape-info")

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

            # Translate
            with Horizontal(classes="param-row"):
                yield Checkbox(
                    "Translate",
                    value=aug.translate_enabled if aug else False,
                    id="centroid-translate-checkbox",
                )
                yield Input(
                    value=str(aug.translate if aug else 0.0),
                    id="centroid-translate-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("fraction (0-0.3)", classes="param-hint")

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

        # Data Pipeline section (visible - matches webapp layout)
        with Container(classes="section-box"):
            yield Static("[bold]Data Pipeline[/bold]", classes="section-header")

            # Speed tip info box (matching webapp)
            yield Static(
                "[green bold]💡 Speed Tip:[/green bold] Set [bold]Num Workers[/bold] to 2+ and "
                "use [bold]Cache to Memory[/bold] or [bold]Cache to Disk[/bold] for faster training.",
                classes="info-box speed-tip",
            )

            # Num Workers
            with Horizontal(classes="param-row"):
                yield Label("Num Workers:", classes="param-label")
                yield Select(
                    [
                        ("0 (Default)", "0"),
                        ("2", "2"),
                        ("4", "4"),
                        ("8", "8"),
                    ],
                    value=str(self._state._num_workers if self._state else 0),
                    id="centroid-num-workers-select",
                    classes="param-input",
                )
                yield Label("Data loader workers (0 for video)", classes="param-hint")

            # Data Pipeline
            current_pipeline = (
                self._state._data_pipeline.value if self._state else "torch_dataset"
            )
            with Horizontal(classes="param-row"):
                yield Label("Data Pipeline:", classes="param-label")
                yield Select(
                    [
                        ("Video (default)", "torch_dataset"),
                        ("Cache to Memory", "litdata"),
                        ("Cache to Disk", "litdata_disk"),
                    ],
                    value=current_pipeline,
                    id="centroid-data-pipeline-select",
                    classes="param-input",
                )
                yield Label("Data loading method", classes="param-hint")

            # Cache Memory Estimate
            yield Static(id="centroid-cache-memory-display", classes="memory-info")

            # Cache config options
            cache = self._state._cache_config if self._state else None

            # Parallel Caching
            with Horizontal(classes="param-row", id="centroid-parallel-caching-row"):
                yield Label("Parallel Caching:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=cache.parallel_caching if cache else True,
                    id="centroid-parallel-caching-checkbox",
                )
                yield Label("Cache images in parallel", classes="param-hint")

            # Cache Workers
            with Horizontal(classes="param-row", id="centroid-cache-workers-row"):
                yield Label("Cache Workers:", classes="param-label")
                yield Input(
                    value=str(cache.cache_workers if cache else 0),
                    id="centroid-cache-workers-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label(
                    "Workers for caching (0 = main process)", classes="param-hint"
                )

            # Disk cache specific options
            yield Static(
                "[dim]Disk Cache Options[/dim]",
                classes="subsection-header",
                id="centroid-disk-cache-header",
            )

            # Cache Image Path
            with Horizontal(classes="param-row", id="centroid-cache-path-row"):
                yield Label("Cache Path:", classes="param-label")
                yield Input(
                    value=cache.cache_img_path if cache else "",
                    id="centroid-cache-path-input",
                    classes="param-input-wide",
                    placeholder="./cache (auto if empty)",
                )

            # Use Existing Images
            with Horizontal(classes="param-row", id="centroid-use-existing-row"):
                yield Label("Use Existing:", classes="param-label")
                yield Checkbox(
                    "Reuse cached images",
                    value=cache.use_existing_imgs if cache else False,
                    id="centroid-use-existing-checkbox",
                )
                yield Label("Skip caching if images exist", classes="param-hint")

            # Delete Cache After Training
            with Horizontal(classes="param-row", id="centroid-delete-cache-row"):
                yield Label("Delete After:", classes="param-label")
                yield Checkbox(
                    "Delete cache after training",
                    value=cache.delete_cache_after_training if cache else True,
                    id="centroid-delete-cache-checkbox",
                )
                yield Label("Clean up disk space", classes="param-hint")

        # Additional Data Parameters (collapsible)
        with Collapsible(
            title="Additional Data Parameters", classes="collapsible-section"
        ):
            # Validation Fraction
            with Horizontal(classes="param-row"):
                yield Label("Val Fraction:", classes="param-label")
                yield Input(
                    value=str(self._state._validation_fraction if self._state else 0.1),
                    id="centroid-val-fraction-input",
                    classes="param-input",
                    type="number",
                )
                yield Label(
                    "Fraction held out for validation (0.05-0.3)", classes="param-hint"
                )

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
                    id="centroid-channels-select",
                    classes="param-input",
                )
                yield Label("Image color mode", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Max Height:", classes="param-label")
                yield Input(
                    value=(
                        str(self._state._max_height)
                        if self._state and self._state._max_height
                        else ""
                    ),
                    id="centroid-max-height-input",
                    classes="param-input",
                    type="integer",
                    placeholder="auto",
                )
                yield Label("Max Width:", classes="param-label")
                yield Input(
                    value=(
                        str(self._state._max_width)
                        if self._state and self._state._max_width
                        else ""
                    ),
                    id="centroid-max-width-input",
                    classes="param-input",
                    type="integer",
                    placeholder="auto",
                )

        # Model configuration
        with Container(classes="section-box"):
            yield Static("[bold]Model Architecture[/bold]", classes="section-header")
            yield Static("[dim]Detects animal centers in full frame[/dim]")

            # Model info display
            yield Static(id="centroid-model-info-display", classes="shape-info")

            # Backbone for centroid
            with Horizontal(classes="param-row"):
                yield Label("Backbone:", classes="param-label")
                yield Select(
                    [
                        ("UNet (recommended)", "unet_medium_rf"),
                        ("UNet Large RF", "unet_large_rf"),
                        ("ConvNeXt Tiny", "convnext_tiny"),
                        ("ConvNeXt Small", "convnext_small"),
                        ("SwinT Tiny", "swint_tiny"),
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
                    [("8", "8"), ("16", "16"), ("32", "32"), ("64", "64")],
                    value=str(self._state._max_stride if self._state else 16),
                    id="centroid-max-stride-select",
                    classes="param-input",
                )
                yield Label(
                    "Receptive field (larger = more context)", classes="param-hint"
                )

            # Base Filters
            with Horizontal(classes="param-row"):
                yield Label("Base Filters:", classes="param-label")
                yield Input(
                    value=str(self._state._filters if self._state else 32),
                    id="centroid-filters-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("First encoder block filters", classes="param-hint")

            # Filters Rate
            with Horizontal(classes="param-row"):
                yield Label("Filters Rate:", classes="param-label")
                yield Input(
                    value=str(self._state._filters_rate if self._state else 1.5),
                    id="centroid-filters-rate-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Multiplier per block (1.5 typical)", classes="param-hint")

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
                    [
                        ("1 (full res)", "1"),
                        ("2 (half res)", "2"),
                        ("4 (quarter)", "4"),
                        ("8 (1/8 res)", "8"),
                        ("16 (1/16 res)", "16"),
                    ],
                    value=str(self._state._output_stride if self._state else 2),
                    id="centroid-output-stride-select",
                    classes="param-input",
                )
                yield Label("Higher = faster but less precise", classes="param-hint")

            # ImageNet Pretrained (for ConvNeXt/SwinT)
            with Horizontal(classes="param-row", id="centroid-imagenet-row"):
                yield Label("ImageNet Weights:", classes="param-label")
                yield Checkbox(
                    "Use ImageNet pretrained",
                    value=self._state._use_imagenet_pretrained if self._state else True,
                    id="centroid-imagenet-checkbox",
                )
                yield Label("For ConvNeXt/SwinT backbones", classes="param-hint")

            # Pretrained Backbone Path
            with Horizontal(classes="param-row"):
                yield Label("Pretrained Backbone:", classes="param-label")
                yield Input(
                    value=self._state._pretrained_backbone if self._state else "",
                    id="centroid-pretrained-backbone-input",
                    classes="param-input-wide",
                    placeholder="Path to .ckpt (optional)",
                )

            # Pretrained Head Path
            with Horizontal(classes="param-row"):
                yield Label("Pretrained Head:", classes="param-label")
                yield Input(
                    value=self._state._pretrained_head if self._state else "",
                    id="centroid-pretrained-head-input",
                    classes="param-input-wide",
                    placeholder="Path to .ckpt (optional)",
                )

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
                yield Label(
                    "Initial optimizer step size (1e-4 is good default)",
                    classes="param-hint",
                )

        # Checkpoint & Logging
        with Collapsible(title="Checkpoints & Logging", classes="collapsible-section"):
            ckpt = self._state._checkpoint if self._state else None

            # Save Checkpoints toggle
            with Horizontal(classes="param-row"):
                yield Label("Save Checkpoints:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=ckpt.enabled if ckpt else True,
                    id="centroid-save-ckpt-checkbox",
                )

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

            # Save Top K
            with Horizontal(classes="param-row"):
                yield Label("Save Top K:", classes="param-label")
                yield Input(
                    value=str(ckpt.save_top_k if ckpt else 1),
                    id="centroid-save-top-k-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Best checkpoints to keep", classes="param-hint")

            # Resume from checkpoint
            with Horizontal(classes="param-row"):
                yield Label("Resume From:", classes="param-label")
                yield Input(
                    value=ckpt.resume_from if ckpt else "",
                    id="centroid-resume-ckpt-input",
                    classes="param-input-wide",
                    placeholder="Path to checkpoint.ckpt",
                )

            yield Static("[dim]Weights & Biases[/dim]", classes="subsection-header")

            # WandB Enable
            wandb = self._state._wandb if self._state else None
            with Horizontal(classes="param-row"):
                yield Label("Enable WandB:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=wandb.enabled if wandb else False,
                    id="centroid-wandb-checkbox",
                )
                yield Label("Log to Weights & Biases", classes="param-hint")

            # WandB Project (hidden until WandB enabled)
            with Horizontal(
                classes="param-row hidden", id="centroid-wandb-project-row"
            ):
                yield Label("WandB Project:", classes="param-label")
                yield Input(
                    value=wandb.project if wandb else "sleap-training",
                    id="centroid-wandb-project-input",
                    classes="param-input-wide",
                )

            # WandB Entity (hidden until WandB enabled)
            with Horizontal(classes="param-row hidden", id="centroid-wandb-entity-row"):
                yield Label("WandB Entity:", classes="param-label")
                yield Input(
                    value=wandb.entity if wandb else "",
                    id="centroid-wandb-entity-input",
                    classes="param-input-wide",
                    placeholder="username or team",
                )

            # WandB Run Name (hidden until WandB enabled)
            with Horizontal(classes="param-row hidden", id="centroid-wandb-name-row"):
                yield Label("WandB Name:", classes="param-label")
                yield Input(
                    value=wandb.name if wandb else "",
                    id="centroid-wandb-name-input",
                    classes="param-input-wide",
                    placeholder="auto-generated",
                )

            # WandB API Key (hidden until WandB enabled)
            with Horizontal(
                classes="param-row hidden", id="centroid-wandb-api-key-row"
            ):
                yield Label("WandB API Key:", classes="param-label")
                yield Input(
                    value=wandb.api_key if wandb else "",
                    id="centroid-wandb-api-key-input",
                    classes="param-input-wide",
                    placeholder="(uses env var if empty)",
                    password=True,
                )

            # WandB Mode (hidden until WandB enabled)
            with Horizontal(classes="param-row hidden", id="centroid-wandb-mode-row"):
                yield Label("WandB Mode:", classes="param-label")
                yield Select(
                    [
                        ("Online", "online"),
                        ("Offline", "offline"),
                        ("Disabled", "disabled"),
                    ],
                    value=wandb.mode if wandb else "online",
                    id="centroid-wandb-mode-select",
                    classes="param-input",
                )
                yield Label("Logging mode", classes="param-hint")

            # WandB Visualization options (hidden until WandB enabled)
            with Horizontal(classes="param-row hidden", id="centroid-wandb-viz-row"):
                yield Label("Log Visualizations:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=wandb.viz_enabled if wandb else True,
                    id="centroid-wandb-viz-checkbox",
                )
                yield Label(
                    "Log prediction visualizations to WandB", classes="param-hint"
                )

            with Horizontal(
                classes="param-row hidden", id="centroid-wandb-save-viz-row"
            ):
                yield Label("Upload Local Viz:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=wandb.save_viz_imgs if wandb else False,
                    id="centroid-wandb-save-viz-checkbox",
                )
                yield Label("Upload local visualization images", classes="param-hint")

            yield Static("[dim]Save Visualizations[/dim]", classes="subsection-header")

            # Save Visualizations toggle (matches webapp enable-viz)
            with Horizontal(classes="param-row"):
                yield Label("Save Visualizations:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=self._state._visualize_preds if self._state else False,
                    id="centroid-save-viz-checkbox",
                )
                yield Label(
                    "Save prediction visualizations locally", classes="param-hint"
                )

            # Keep Viz Folder (hidden until Save Viz enabled)
            with Horizontal(
                classes="param-row hidden", id="centroid-viz-keep-folder-row"
            ):
                yield Label("Keep Viz Folder:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=self._state._keep_viz if self._state else False,
                    id="centroid-viz-keep-folder-checkbox",
                )
                yield Label(
                    "Keep visualization folder after training", classes="param-hint"
                )

            yield Static("[dim]Evaluation[/dim]", classes="subsection-header")

            # Enable Evaluation toggle
            evaluation = self._state._evaluation if self._state else None
            with Horizontal(classes="param-row"):
                yield Label("Enable Evaluation:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=evaluation.enabled if evaluation else False,
                    id="centroid-eval-checkbox",
                )
                yield Label("Run OKS evaluation during training", classes="param-hint")

            # Eval options (hidden until Enable Eval is checked)
            with Horizontal(
                classes="param-row hidden", id="centroid-eval-frequency-row"
            ):
                yield Label("Eval Frequency:", classes="param-label")
                yield Input(
                    value=str(evaluation.frequency if evaluation else 1),
                    id="centroid-eval-frequency-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Run eval every N epochs", classes="param-hint")

            with Horizontal(classes="param-row hidden", id="centroid-eval-oks-row"):
                yield Label("OKS Stddev:", classes="param-label")
                yield Input(
                    value=str(evaluation.oks_stddev if evaluation else 0.025),
                    id="centroid-eval-oks-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Standard deviation for OKS scoring", classes="param-hint")

        # Advanced Parameters (collapsible)
        with Collapsible(title="Advanced Parameters", classes="collapsible-section"):
            # Optimizer
            with Horizontal(classes="param-row"):
                yield Label("Optimizer:", classes="param-label")
                yield Select(
                    [("Adam", "Adam"), ("AdamW", "AdamW")],
                    value=self._state._optimizer if self._state else "Adam",
                    id="centroid-optimizer-select",
                    classes="param-input",
                )
                yield Label("Adam is recommended for most cases", classes="param-hint")

            # Early stopping
            with Horizontal(classes="param-row"):
                yield Label("Early Stopping:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=self._state._early_stopping if self._state else True,
                    id="centroid-early-stopping-checkbox",
                )
                yield Label(
                    "Stop when validation loss stops improving", classes="param-hint"
                )

            # ES Patience
            with Horizontal(classes="param-row"):
                yield Label("ES Patience:", classes="param-label")
                yield Input(
                    value=str(
                        self._state._early_stopping_patience if self._state else 10
                    ),
                    id="centroid-es-patience-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs to wait before stopping", classes="param-hint")

            # ES Min Delta
            with Horizontal(classes="param-row"):
                yield Label("ES Min Delta:", classes="param-label")
                yield Input(
                    value=str(
                        self._state._early_stopping_min_delta if self._state else 1e-6
                    ),
                    id="centroid-es-min-delta-input",
                    classes="param-input",
                    type="number",
                )
                yield Label(
                    "Minimum change to qualify as improvement", classes="param-hint"
                )

            yield Static(
                "[dim]Learning Rate Scheduler[/dim]", classes="subsection-header"
            )

            # LR Scheduler
            scheduler = self._state._scheduler if self._state else None
            with Horizontal(classes="param-row"):
                yield Label("Scheduler:", classes="param-label")
                yield Select(
                    [
                        ("ReduceLROnPlateau", "ReduceLROnPlateau"),
                        ("Cosine Annealing + Warmup", "CosineAnnealingWarmup"),
                        ("Step LR", "StepLR"),
                        ("None", "none"),
                    ],
                    value=scheduler.type.value if scheduler else "ReduceLROnPlateau",
                    id="centroid-scheduler-select",
                    classes="param-input",
                )
                yield Label("Reduces LR when loss plateaus", classes="param-hint")

            # Scheduler-specific params (ReduceLROnPlateau)
            with Horizontal(classes="param-row"):
                yield Label("LR Patience:", classes="param-label")
                yield Input(
                    value=str(scheduler.plateau_patience if scheduler else 5),
                    id="centroid-lr-patience-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs to wait before reducing LR", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("LR Factor:", classes="param-label")
                yield Input(
                    value=str(scheduler.factor if scheduler else 0.5),
                    id="centroid-lr-factor-input",
                    classes="param-input",
                    type="number",
                )
                yield Label(
                    "Factor to reduce LR by (0.5 = halve)", classes="param-hint"
                )

            yield Static("[dim]Hardware[/dim]", classes="subsection-header")

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
                    id="centroid-accelerator-select",
                    classes="param-input",
                )
                yield Label("Hardware for training", classes="param-hint")

            # Devices
            with Horizontal(classes="param-row"):
                yield Label("Devices:", classes="param-label")
                yield Select(
                    [("Auto", "auto"), ("1", "1"), ("2", "2"), ("4", "4")],
                    value=self._state._devices if self._state else "auto",
                    id="centroid-devices-select",
                    classes="param-input",
                )
                yield Label("Number of GPUs", classes="param-hint")

            # Min Steps per Epoch
            with Horizontal(classes="param-row"):
                yield Label("Min Steps/Epoch:", classes="param-label")
                yield Input(
                    value=str(self._state._min_steps_per_epoch if self._state else 200),
                    id="centroid-min-steps-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Minimum training steps per epoch", classes="param-hint")

            # Random Seed
            with Horizontal(classes="param-row"):
                yield Label("Random Seed:", classes="param-label")
                yield Input(
                    value=(
                        str(self._state._random_seed)
                        if self._state and self._state._random_seed
                        else ""
                    ),
                    id="centroid-random-seed-input",
                    classes="param-input",
                    type="integer",
                    placeholder="none",
                )
                yield Label("For reproducibility", classes="param-hint")

        # Centroid memory estimate
        yield Static(id="centroid-memory-display", classes="memory-info")

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
                yield Label(
                    "Crop resize factor (1.0 = no resize)", classes="param-hint"
                )

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

        # Data Pipeline section (visible - matches webapp layout)
        with Container(classes="section-box"):
            yield Static("[bold]Data Pipeline[/bold]", classes="section-header")

            # Speed tip info box
            yield Static(
                "[green bold]💡 Speed Tip:[/green bold] Set [bold]Num Workers[/bold] to 2+ and "
                "use [bold]Cache to Memory[/bold] or [bold]Cache to Disk[/bold] for faster training.",
                classes="info-box speed-tip",
            )

            # Num Workers
            with Horizontal(classes="param-row"):
                yield Label("Num Workers:", classes="param-label")
                yield Select(
                    [
                        ("0 (Default)", "0"),
                        ("2", "2"),
                        ("4", "4"),
                        ("8", "8"),
                    ],
                    value=str(self._state._num_workers if self._state else 0),
                    id="instance-num-workers-select",
                    classes="param-input",
                )
                yield Label("Data loader workers (0 for video)", classes="param-hint")

            # Data Pipeline
            current_pipeline = (
                self._state._data_pipeline.value if self._state else "torch_dataset"
            )
            with Horizontal(classes="param-row"):
                yield Label("Data Pipeline:", classes="param-label")
                yield Select(
                    [
                        ("Video (default)", "torch_dataset"),
                        ("Cache to Memory", "litdata"),
                        ("Cache to Disk", "litdata_disk"),
                    ],
                    value=current_pipeline,
                    id="instance-data-pipeline-select",
                    classes="param-input",
                )
                yield Label("Data loading method", classes="param-hint")

            # Cache Memory Estimate
            yield Static(id="instance-cache-memory-display", classes="memory-info")

            # Cache config options
            cache = self._state._cache_config if self._state else None

            # Parallel Caching
            with Horizontal(classes="param-row", id="instance-parallel-caching-row"):
                yield Label("Parallel Caching:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=cache.parallel_caching if cache else True,
                    id="instance-parallel-caching-checkbox",
                )
                yield Label("Cache images in parallel", classes="param-hint")

            # Cache Workers
            with Horizontal(classes="param-row", id="instance-cache-workers-row"):
                yield Label("Cache Workers:", classes="param-label")
                yield Input(
                    value=str(cache.cache_workers if cache else 0),
                    id="instance-cache-workers-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label(
                    "Workers for caching (0 = main process)", classes="param-hint"
                )

            # Disk cache options
            yield Static(
                "[dim]Disk Cache Options[/dim]",
                classes="subsection-header",
                id="instance-disk-cache-header",
            )

            with Horizontal(classes="param-row", id="instance-cache-path-row"):
                yield Label("Cache Path:", classes="param-label")
                yield Input(
                    value=cache.cache_img_path if cache else "",
                    id="instance-cache-path-input",
                    classes="param-input-wide",
                    placeholder="./cache (auto if empty)",
                )

            with Horizontal(classes="param-row", id="instance-use-existing-row"):
                yield Label("Use Existing:", classes="param-label")
                yield Checkbox(
                    "Reuse cached images",
                    value=cache.use_existing_imgs if cache else False,
                    id="instance-use-existing-checkbox",
                )
                yield Label("Skip caching if images exist", classes="param-hint")

            with Horizontal(classes="param-row", id="instance-delete-cache-row"):
                yield Label("Delete After:", classes="param-label")
                yield Checkbox(
                    "Delete cache after training",
                    value=cache.delete_cache_after_training if cache else True,
                    id="instance-delete-cache-checkbox",
                )
                yield Label("Clean up disk space", classes="param-hint")

        # Additional Data Parameters (collapsible)
        with Collapsible(
            title="Additional Data Parameters", classes="collapsible-section"
        ):
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
                    id="instance-channels-select",
                    classes="param-input",
                )
                yield Label("Image color mode", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Min Crop Size:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_min_crop_size if self._state else 100),
                    id="min-crop-size-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Minimum allowed crop size", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Crop Padding:", classes="param-label")
                yield Input(
                    value=(
                        str(self._state._ci_crop_padding)
                        if self._state and self._state._ci_crop_padding is not None
                        else ""
                    ),
                    id="crop-padding-input",
                    classes="param-input",
                    type="integer",
                    placeholder="auto",
                )
                yield Label(
                    "Extra padding around crops (empty = auto)", classes="param-hint"
                )

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
                yield Label(
                    "Tighter sigma for crops (2.5 typical)", classes="param-hint"
                )

            # Instance output stride
            with Horizontal(classes="param-row"):
                yield Label("Output Stride:", classes="param-label")
                yield Select(
                    [
                        ("1 (full res)", "1"),
                        ("2 (half res)", "2"),
                        ("4 (quarter)", "4"),
                        ("8 (1/8 res)", "8"),
                        ("16 (1/16 res)", "16"),
                    ],
                    value=str(self._state._ci_output_stride if self._state else 2),
                    id="instance-output-stride-select",
                    classes="param-input",
                )
                yield Label("Higher = faster but less precise", classes="param-hint")

            # Base Filters
            with Horizontal(classes="param-row"):
                yield Label("Base Filters:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_filters if self._state else 32),
                    id="instance-filters-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label(
                    "First encoder block filters (32 typical)", classes="param-hint"
                )

            # Filters Rate
            with Horizontal(classes="param-row"):
                yield Label("Filters Rate:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_filters_rate if self._state else 1.5),
                    id="instance-filters-rate-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Multiplier per block (1.5 typical)", classes="param-hint")

            # ImageNet Pretrained (for ConvNeXt/SwinT)
            with Horizontal(classes="param-row", id="instance-imagenet-row"):
                yield Label("ImageNet Weights:", classes="param-label")
                yield Checkbox(
                    "Use ImageNet pretrained",
                    value=self._state._use_imagenet_pretrained if self._state else True,
                    id="instance-imagenet-checkbox",
                )
                yield Label("For ConvNeXt/SwinT backbones", classes="param-hint")

            # Pretrained Backbone Path
            with Horizontal(classes="param-row"):
                yield Label("Pretrained Backbone:", classes="param-label")
                yield Input(
                    value=self._state._ci_pretrained_backbone if self._state else "",
                    id="instance-pretrained-backbone-input",
                    classes="param-input-wide",
                    placeholder="Path to .ckpt (optional)",
                )

            # Pretrained Head Path
            with Horizontal(classes="param-row"):
                yield Label("Pretrained Head:", classes="param-label")
                yield Input(
                    value=self._state._ci_pretrained_head if self._state else "",
                    id="instance-pretrained-head-input",
                    classes="param-input-wide",
                    placeholder="Path to .ckpt (optional)",
                )

            # Model Info Display
            yield Static(id="instance-model-info-display", classes="model-info")

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
                yield Label(
                    "Crops per batch (more = faster, more memory)", classes="param-hint"
                )

            # Max epochs
            with Horizontal(classes="param-row"):
                yield Label("Max Epochs:", classes="param-label")
                yield Input(
                    value=str(self._state._ci_max_epochs if self._state else 200),
                    id="instance-max-epochs-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label(
                    "Training stops after this many epochs", classes="param-hint"
                )

            # Learning rate
            with Horizontal(classes="param-row"):
                yield Label("Learning Rate:", classes="param-label")
                yield Select(
                    [
                        ("1e-4 (Default)", "0.0001"),
                        ("5e-4", "0.0005"),
                        ("1e-3", "0.001"),
                        ("5e-5", "0.00005"),
                    ],
                    value=str(
                        self._state._ci_learning_rate if self._state else "0.0001"
                    ),
                    id="instance-lr-select",
                    classes="param-input",
                )
                yield Label("Initial optimizer step size", classes="param-hint")

        # Checkpoint & Logging section
        with Collapsible(title="Checkpoints & Logging", classes="collapsible-section"):
            ckpt = self._state._checkpoint if self._state else None

            yield Static("[dim]Checkpoints[/dim]", classes="subsection-header")

            # Save Checkpoint
            with Horizontal(classes="param-row"):
                yield Label("Save Checkpoints:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=ckpt.enabled if ckpt else True,
                    id="instance-save-ckpt-checkbox",
                )
                yield Label(
                    "Save model checkpoints during training", classes="param-hint"
                )

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

            with Horizontal(classes="param-row"):
                yield Label("Save Top K:", classes="param-label")
                yield Input(
                    value=str(ckpt.save_top_k if ckpt else 1),
                    id="instance-save-top-k-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Number of best checkpoints to keep", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("Save Last:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=ckpt.save_last if ckpt else False,
                    id="instance-save-last-checkbox",
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
                    id="instance-wandb-checkbox",
                )
                yield Label("Log to Weights & Biases", classes="param-hint")

            # WandB Project (hidden until WandB enabled)
            with Horizontal(
                classes="param-row hidden", id="instance-wandb-project-row"
            ):
                yield Label("WandB Project:", classes="param-label")
                yield Input(
                    value=wandb.project if wandb else "sleap-training",
                    id="instance-wandb-project-input",
                    classes="param-input-wide",
                )

            # WandB Entity (hidden until WandB enabled)
            with Horizontal(classes="param-row hidden", id="instance-wandb-entity-row"):
                yield Label("WandB Entity:", classes="param-label")
                yield Input(
                    value=wandb.entity if wandb else "",
                    id="instance-wandb-entity-input",
                    classes="param-input-wide",
                    placeholder="username or team",
                )

            yield Static("[dim]Save Visualizations[/dim]", classes="subsection-header")

            # Save Visualizations toggle
            with Horizontal(classes="param-row"):
                yield Label("Save Visualizations:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=self._state._visualize_preds if self._state else False,
                    id="instance-save-viz-checkbox",
                )
                yield Label(
                    "Save prediction visualizations locally", classes="param-hint"
                )

            # Keep Viz Folder (hidden until Save Viz enabled)
            with Horizontal(
                classes="param-row hidden", id="instance-viz-keep-folder-row"
            ):
                yield Label("Keep Viz Folder:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=self._state._keep_viz if self._state else False,
                    id="instance-viz-keep-folder-checkbox",
                )
                yield Label(
                    "Keep visualization folder after training", classes="param-hint"
                )

            yield Static("[dim]Evaluation[/dim]", classes="subsection-header")

            # Enable Evaluation toggle
            evaluation = self._state._evaluation if self._state else None
            with Horizontal(classes="param-row"):
                yield Label("Enable Evaluation:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=evaluation.enabled if evaluation else False,
                    id="instance-eval-checkbox",
                )
                yield Label("Run OKS evaluation during training", classes="param-hint")

            # Eval options (hidden until Enable Eval is checked)
            with Horizontal(
                classes="param-row hidden", id="instance-eval-frequency-row"
            ):
                yield Label("Eval Frequency:", classes="param-label")
                yield Input(
                    value=str(evaluation.frequency if evaluation else 1),
                    id="instance-eval-frequency-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Run eval every N epochs", classes="param-hint")

            with Horizontal(classes="param-row hidden", id="instance-eval-oks-row"):
                yield Label("OKS Stddev:", classes="param-label")
                yield Input(
                    value=str(evaluation.oks_stddev if evaluation else 0.025),
                    id="instance-eval-oks-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Standard deviation for OKS scoring", classes="param-hint")

        # Advanced Parameters (collapsible)
        with Collapsible(title="Advanced Parameters", classes="collapsible-section"):
            # Optimizer
            with Horizontal(classes="param-row"):
                yield Label("Optimizer:", classes="param-label")
                yield Select(
                    [("Adam", "Adam"), ("AdamW", "AdamW")],
                    value=self._state._ci_optimizer if self._state else "Adam",
                    id="instance-optimizer-select",
                    classes="param-input",
                )
                yield Label("Adam is recommended for most cases", classes="param-hint")

            # Early stopping
            with Horizontal(classes="param-row"):
                yield Label("Early Stopping:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=self._state._ci_early_stopping if self._state else True,
                    id="instance-early-stopping-checkbox",
                )
                yield Label(
                    "Stop when validation loss stops improving", classes="param-hint"
                )

            # ES Patience
            with Horizontal(classes="param-row"):
                yield Label("ES Patience:", classes="param-label")
                yield Input(
                    value=str(
                        self._state._ci_early_stopping_patience if self._state else 5
                    ),
                    id="instance-es-patience-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs to wait before stopping", classes="param-hint")

            # ES Min Delta
            with Horizontal(classes="param-row"):
                yield Label("ES Min Delta:", classes="param-label")
                yield Input(
                    value=str(
                        self._state._ci_early_stopping_min_delta
                        if self._state
                        else 1e-6
                    ),
                    id="instance-es-min-delta-input",
                    classes="param-input",
                    type="number",
                )
                yield Label(
                    "Minimum change to qualify as improvement", classes="param-hint"
                )

            yield Static(
                "[dim]Learning Rate Scheduler[/dim]", classes="subsection-header"
            )

            # LR Scheduler
            scheduler = self._state._ci_scheduler if self._state else None
            with Horizontal(classes="param-row"):
                yield Label("Scheduler:", classes="param-label")
                yield Select(
                    [
                        ("ReduceLROnPlateau", "ReduceLROnPlateau"),
                        ("Cosine Annealing + Warmup", "CosineAnnealingWarmup"),
                        ("Step LR", "StepLR"),
                        ("None", "none"),
                    ],
                    value=scheduler.type.value if scheduler else "ReduceLROnPlateau",
                    id="instance-scheduler-select",
                    classes="param-input",
                )
                yield Label("Reduces LR when loss plateaus", classes="param-hint")

            # Scheduler-specific params (ReduceLROnPlateau)
            with Horizontal(classes="param-row"):
                yield Label("LR Patience:", classes="param-label")
                yield Input(
                    value=str(scheduler.plateau_patience if scheduler else 5),
                    id="instance-lr-patience-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Epochs to wait before reducing LR", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("LR Factor:", classes="param-label")
                yield Input(
                    value=str(scheduler.factor if scheduler else 0.5),
                    id="instance-lr-factor-input",
                    classes="param-input",
                    type="number",
                )
                yield Label(
                    "Factor to reduce LR by (0.5 = halve)", classes="param-hint"
                )

            yield Static(
                "[dim]Online Hard Keypoint Mining (OHKM)[/dim]",
                classes="subsection-header",
            )

            ohkm = self._state._ohkm if self._state else None
            with Horizontal(classes="param-row"):
                yield Label("Enable OHKM:", classes="param-label")
                yield Checkbox(
                    "Enabled",
                    value=ohkm.enabled if ohkm else False,
                    id="instance-ohkm-checkbox",
                )
                yield Label(
                    "Focus training on difficult keypoints", classes="param-hint"
                )

            with Horizontal(classes="param-row"):
                yield Label("Hard-to-Easy Ratio:", classes="param-label")
                yield Input(
                    value=str(ohkm.hard_to_easy_ratio if ohkm else 2.0),
                    id="instance-ohkm-ratio-input",
                    classes="param-input",
                    type="number",
                )
                yield Label("Ratio to classify keypoints as hard", classes="param-hint")

            with Horizontal(classes="param-row"):
                yield Label("OHKM Loss Scale:", classes="param-label")
                yield Input(
                    value=str(ohkm.loss_scale if ohkm else 5.0),
                    id="instance-ohkm-scale-input",
                    classes="param-input",
                    type="number",
                )
                yield Label(
                    "Scale factor for hard keypoint losses", classes="param-hint"
                )

            with Horizontal(classes="param-row"):
                yield Label("Min Hard Keypoints:", classes="param-label")
                yield Input(
                    value=str(ohkm.min_hard_keypoints if ohkm else 2),
                    id="instance-ohkm-min-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label(
                    "Minimum keypoints to classify as hard", classes="param-hint"
                )

            with Horizontal(classes="param-row"):
                yield Label("Max Hard Keypoints:", classes="param-label")
                yield Input(
                    value=(
                        str(ohkm.max_hard_keypoints)
                        if ohkm and ohkm.max_hard_keypoints
                        else ""
                    ),
                    id="instance-ohkm-max-input",
                    classes="param-input",
                    type="integer",
                    placeholder="none",
                )
                yield Label("Maximum (empty = unlimited)", classes="param-hint")

            yield Static("[dim]Hardware[/dim]", classes="subsection-header")

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
                    id="instance-accelerator-select",
                    classes="param-input",
                )
                yield Label("Hardware for training", classes="param-hint")

            # Devices
            with Horizontal(classes="param-row"):
                yield Label("Devices:", classes="param-label")
                yield Select(
                    [("Auto", "auto"), ("1", "1"), ("2", "2"), ("4", "4")],
                    value=self._state._devices if self._state else "auto",
                    id="instance-devices-select",
                    classes="param-input",
                )
                yield Label("Number of GPUs", classes="param-hint")

            # Min Steps per Epoch
            with Horizontal(classes="param-row"):
                yield Label("Min Steps/Epoch:", classes="param-label")
                yield Input(
                    value=str(self._state._min_steps_per_epoch if self._state else 200),
                    id="instance-min-steps-input",
                    classes="param-input",
                    type="integer",
                )
                yield Label("Minimum training steps per epoch", classes="param-hint")

            # Random Seed
            with Horizontal(classes="param-row"):
                yield Label("Random Seed:", classes="param-label")
                yield Input(
                    value=(
                        str(self._state._random_seed)
                        if self._state and self._state._random_seed
                        else ""
                    ),
                    id="instance-random-seed-input",
                    classes="param-input",
                    type="integer",
                    placeholder="none",
                )
                yield Label("For reproducibility", classes="param-hint")

        # Instance memory estimate
        yield Static(id="instance-memory-display", classes="memory-info")

    # ==================== MEMORY ESTIMATE ====================

    def _compose_memory_estimate(self) -> ComposeResult:
        """Compose memory estimate display."""
        if not self._state:
            return

        yield Static(id="memory-info-display", classes="memory-info")

    # ==================== EVENT HANDLERS ====================

    # Data handlers
    @on(Input.Changed, "#scale-input")
    def handle_scale_change(self, event: Input.Changed) -> None:
        """Handle scale changes."""
        if self._state and event.value:
            try:
                self._state._input_scale = float(event.value)
                self._update_shape_display()
                self._update_memory_display()
            except ValueError:
                pass

    @on(Input.Changed, "#max-height-input")
    def handle_max_height_change(self, event: Input.Changed) -> None:
        """Handle max height changes."""
        if self._state:
            self._state._max_height = int(event.value) if event.value else None
            self._update_shape_display()
            self._update_memory_display()

    @on(Input.Changed, "#max-width-input")
    def handle_max_width_change(self, event: Input.Changed) -> None:
        """Handle max width changes."""
        if self._state:
            self._state._max_width = int(event.value) if event.value else None
            self._update_shape_display()
            self._update_memory_display()

    @on(Select.Changed, "#channels-select")
    def handle_channels_change(self, event: Select.Changed) -> None:
        """Handle channels changes."""
        if self._state and event.value:
            self._state._ensure_rgb = event.value == "rgb"
            self._state._ensure_grayscale = event.value == "grayscale"
            self._update_shape_display()
            self._update_memory_display()
            self._update_model_info_display()

    @on(Input.Changed, "#val-fraction-input")
    def handle_val_fraction_change(self, event: Input.Changed) -> None:
        """Handle val fraction changes."""
        if self._state and event.value:
            try:
                self._state._validation_fraction = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#num-workers-select")
    def handle_num_workers_select_change(self, event: Select.Changed) -> None:
        """Handle num workers changes."""
        if self._state and event.value:
            self._state._num_workers = int(event.value)
            self._update_cache_memory_display()

    @on(Select.Changed, "#data-pipeline-select")
    def handle_data_pipeline_change(self, event: Select.Changed) -> None:
        """Handle data pipeline changes."""
        if self._state and event.value:
            self._state._data_pipeline = DataPipelineType(event.value)
            self._update_memory_display()
            self._update_cache_memory_display()
            self._update_cache_options_visibility()

    # Model handlers
    @on(Select.Changed, "#backbone-select")
    def handle_backbone_change(self, event: Select.Changed) -> None:
        """Handle backbone changes."""
        if self._state and event.value:
            self._state._backbone = event.value
            if "convnext" in event.value or "swint" in event.value:
                self._state._max_stride = 32
                try:
                    select = self.query_one("#max-stride-select", Select)
                    select.value = "32"
                except Exception:
                    pass
            self._update_model_info_display()
            self._update_memory_display()
            self._update_imagenet_visibility()

    @on(Select.Changed, "#max-stride-select")
    def handle_stride_change(self, event: Select.Changed) -> None:
        """Handle stride changes."""
        if self._state and event.value:
            self._state._max_stride = int(event.value)
            self._update_model_info_display()
            self._update_memory_display()

    @on(Input.Changed, "#filters-input")
    def handle_filters_change(self, event: Input.Changed) -> None:
        """Handle filters changes."""
        if self._state and event.value:
            try:
                self._state._filters = int(event.value)
                self._update_model_info_display()
                self._update_memory_display()
            except ValueError:
                pass

    @on(Input.Changed, "#filters-rate-input")
    def handle_filters_rate_change(self, event: Input.Changed) -> None:
        """Handle filters rate changes."""
        if self._state and event.value:
            try:
                self._state._filters_rate = float(event.value)
                self._update_model_info_display()
                self._update_memory_display()
            except ValueError:
                pass

    @on(Input.Changed, "#sigma-input")
    def handle_sigma_change(self, event: Input.Changed) -> None:
        """Handle sigma changes."""
        if self._state and event.value:
            try:
                self._state._sigma = float(event.value)
                self._update_model_info_display()  # Update Gaussian radius display
            except ValueError:
                pass

    @on(Select.Changed, "#output-stride-select")
    def handle_output_stride_change(self, event: Select.Changed) -> None:
        """Handle output stride changes."""
        if self._state and event.value:
            self._state._output_stride = int(event.value)
            self._update_shape_display()
            self._update_memory_display()
            self._update_model_info_display()  # Update decoder blocks and Gaussian radius

    @on(Input.Changed, "#pretrained-backbone-input")
    def handle_pretrained_backbone_change(self, event: Input.Changed) -> None:
        """Handle pretrained backbone changes."""
        if self._state:
            self._state._pretrained_backbone = event.value or ""

    @on(Input.Changed, "#pretrained-head-input")
    def handle_pretrained_head_change(self, event: Input.Changed) -> None:
        """Handle pretrained head changes."""
        if self._state:
            self._state._pretrained_head = event.value or ""

    # PAF handlers
    @on(Input.Changed, "#paf-sigma-input")
    def handle_paf_sigma_change(self, event: Input.Changed) -> None:
        """Handle paf sigma changes."""
        if self._state and event.value:
            try:
                self._state._paf_config.sigma = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#paf-output-stride-select")
    def handle_paf_output_stride_change(self, event: Select.Changed) -> None:
        """Handle paf output stride changes."""
        if self._state and event.value:
            self._state._paf_config.output_stride = int(event.value)

    @on(Input.Changed, "#paf-loss-weight-input")
    def handle_paf_loss_weight_change(self, event: Input.Changed) -> None:
        """Handle paf loss weight changes."""
        if self._state and event.value:
            try:
                self._state._paf_config.loss_weight = float(event.value)
            except ValueError:
                pass

    # Class vector handlers
    @on(Select.Changed, "#fc-layers-select")
    def handle_fc_layers_change(self, event: Select.Changed) -> None:
        """Handle fc layers changes."""
        if self._state and event.value:
            self._state._class_vector_config.num_fc_layers = int(event.value)

    @on(Input.Changed, "#fc-units-input")
    def handle_fc_units_change(self, event: Input.Changed) -> None:
        """Handle fc units changes."""
        if self._state and event.value:
            try:
                self._state._class_vector_config.num_fc_units = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#class-loss-weight-input")
    def handle_class_loss_weight_change(self, event: Input.Changed) -> None:
        """Handle class loss weight changes."""
        if self._state and event.value:
            try:
                self._state._class_vector_config.loss_weight = float(event.value)
            except ValueError:
                pass

    # Training handlers
    @on(Select.Changed, "#batch-size-select")
    def handle_batch_change(self, event: Select.Changed) -> None:
        """Handle batch changes."""
        if self._state and event.value:
            self._state._batch_size = int(event.value)
            self._update_memory_display()

    @on(Input.Changed, "#max-epochs-input")
    def handle_epochs_change(self, event: Input.Changed) -> None:
        """Handle epochs changes."""
        if self._state and event.value:
            try:
                self._state._max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#lr-input")
    def handle_lr_change(self, event: Input.Changed) -> None:
        """Handle lr changes."""
        if self._state and event.value:
            try:
                self._state._learning_rate = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#optimizer-select")
    def handle_optimizer_change(self, event: Select.Changed) -> None:
        """Handle optimizer changes."""
        if self._state and event.value:
            self._state._optimizer = event.value

    @on(Checkbox.Changed, "#early-stopping-checkbox")
    def handle_early_stopping_change(self, event: Checkbox.Changed) -> None:
        """Handle early stopping changes."""
        if self._state:
            self._state._early_stopping = event.value

    @on(Input.Changed, "#es-patience-input")
    def handle_es_patience_change(self, event: Input.Changed) -> None:
        """Handle es patience changes."""
        if self._state and event.value:
            try:
                self._state._early_stopping_patience = int(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#scheduler-select")
    def handle_scheduler_change(self, event: Select.Changed) -> None:
        """Handle scheduler changes."""
        if self._state and event.value:
            self._state._scheduler.type = SchedulerType(event.value)
            # Show/hide scheduler param containers
            self._update_scheduler_params_visibility(event.value)

    # Augmentation checkbox handlers (standard config)
    @on(Checkbox.Changed, "#rotation-checkbox")
    def handle_rotation_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle rotation changes."""
        if self._state:
            self._state._augmentation.rotation_enabled = event.value

    @on(Checkbox.Changed, "#scale-checkbox")
    def handle_scale_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle scale changes."""
        if self._state:
            self._state._augmentation.scale_enabled = event.value

    @on(Checkbox.Changed, "#brightness-checkbox")
    def handle_brightness_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle brightness changes."""
        if self._state:
            self._state._augmentation.brightness_enabled = event.value

    @on(Checkbox.Changed, "#contrast-checkbox")
    def handle_contrast_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle contrast changes."""
        if self._state:
            self._state._augmentation.contrast_enabled = event.value

    @on(Input.Changed, "#rotation-input")
    def handle_rotation_change(self, event: Input.Changed) -> None:
        """Handle rotation changes."""
        if self._state and event.value:
            try:
                val = float(event.value)
                self._state._augmentation.rotation_min = -abs(val)
                self._state._augmentation.rotation_max = abs(val)
            except ValueError:
                pass

    @on(Input.Changed, "#scale-min-input")
    def handle_scale_min_change(self, event: Input.Changed) -> None:
        """Handle scale min changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.scale_min = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scale-max-input")
    def handle_scale_max_change(self, event: Input.Changed) -> None:
        """Handle scale max changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.scale_max = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#brightness-limit-input")
    def handle_brightness_change(self, event: Input.Changed) -> None:
        """Handle brightness changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.brightness_limit = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#contrast-limit-input")
    def handle_contrast_change(self, event: Input.Changed) -> None:
        """Handle contrast changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.contrast_limit = float(event.value)
            except ValueError:
                pass

    # Checkpoint handlers
    @on(Input.Changed, "#ckpt-dir-input")
    def handle_ckpt_dir_change(self, event: Input.Changed) -> None:
        """Handle ckpt dir changes."""
        if self._state:
            self._state._checkpoint.checkpoint_dir = event.value or ""

    @on(Input.Changed, "#run-name-input")
    def handle_run_name_change(self, event: Input.Changed) -> None:
        """Handle run name changes."""
        if self._state:
            self._state._checkpoint.run_name = event.value or ""

    @on(Input.Changed, "#resume-ckpt-input")
    def handle_resume_ckpt_change(self, event: Input.Changed) -> None:
        """Handle resume ckpt changes."""
        if self._state:
            self._state._checkpoint.resume_from = event.value or ""

    @on(Input.Changed, "#save-top-k-input")
    def handle_save_top_k_change(self, event: Input.Changed) -> None:
        """Handle save top k changes."""
        if self._state and event.value:
            try:
                self._state._checkpoint.save_top_k = int(event.value)
            except ValueError:
                pass

    @on(Checkbox.Changed, "#save-last-checkbox")
    def handle_save_last_change(self, event: Checkbox.Changed) -> None:
        """Handle save last changes."""
        if self._state:
            self._state._checkpoint.save_last = event.value

    @on(Checkbox.Changed, "#wandb-checkbox")
    def handle_wandb_change(self, event: Checkbox.Changed) -> None:
        """Handle wandb changes."""
        if self._state:
            self._state._wandb.enabled = event.value
            self._update_wandb_options_visibility()

    @on(Input.Changed, "#wandb-project-input")
    def handle_wandb_project_change(self, event: Input.Changed) -> None:
        """Handle wandb project changes."""
        if self._state:
            self._state._wandb.project = event.value or "sleap-nn"

    @on(Input.Changed, "#wandb-entity-input")
    def handle_wandb_entity_change(self, event: Input.Changed) -> None:
        """Handle wandb entity changes."""
        if self._state:
            self._state._wandb.entity = event.value or ""

    # Advanced handlers
    @on(Select.Changed, "#accelerator-select")
    def handle_accelerator_change(self, event: Select.Changed) -> None:
        """Handle accelerator changes."""
        if self._state and event.value:
            self._state._accelerator = event.value

    @on(Input.Changed, "#min-steps-input")
    def handle_min_steps_change(self, event: Input.Changed) -> None:
        """Handle min steps changes."""
        if self._state and event.value:
            try:
                self._state._min_steps_per_epoch = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#random-seed-input")
    def handle_random_seed_change(self, event: Input.Changed) -> None:
        """Handle random seed changes."""
        if self._state:
            self._state._random_seed = int(event.value) if event.value else None

    # Visualization handlers
    @on(Checkbox.Changed, "#progress-bar-checkbox")
    def handle_progress_bar_change(self, event: Checkbox.Changed) -> None:
        """Handle progress bar changes."""
        if self._state:
            self._state._enable_progress_bar = event.value

    @on(Checkbox.Changed, "#save-viz-checkbox")
    def handle_save_viz_change(self, event: Checkbox.Changed) -> None:
        """Handle save viz changes."""
        if self._state:
            self._state._visualize_preds = event.value
            self._update_viz_options_visibility()

    @on(Checkbox.Changed, "#viz-keep-folder-checkbox")
    def handle_viz_keep_folder_change(self, event: Checkbox.Changed) -> None:
        """Handle viz keep folder changes."""
        if self._state:
            self._state._keep_viz = event.value

    @on(Checkbox.Changed, "#eval-checkbox")
    def handle_eval_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle eval changes."""
        if self._state:
            self._state._evaluation.enabled = event.value
            self._update_eval_options_visibility()

    # Legacy handlers for backwards compatibility
    @on(Checkbox.Changed, "#visualize-preds-checkbox")
    def handle_visualize_preds_change(self, event: Checkbox.Changed) -> None:
        """Handle visualize preds changes."""
        if self._state:
            self._state._visualize_preds = event.value
            self._update_viz_options_visibility()

    @on(Checkbox.Changed, "#keep-viz-checkbox")
    def handle_keep_viz_change(self, event: Checkbox.Changed) -> None:
        """Handle keep viz changes."""
        if self._state:
            self._state._keep_viz = event.value

    # Cache config handlers
    @on(Checkbox.Changed, "#parallel-caching-checkbox")
    def handle_parallel_caching_change(self, event: Checkbox.Changed) -> None:
        """Handle parallel caching changes."""
        if self._state:
            self._state._cache_config.parallel_caching = event.value

    @on(Input.Changed, "#cache-workers-input")
    def handle_cache_workers_change(self, event: Input.Changed) -> None:
        """Handle cache workers changes."""
        if self._state and event.value:
            try:
                self._state._cache_config.cache_workers = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#cache-path-input")
    def handle_cache_path_change(self, event: Input.Changed) -> None:
        """Handle cache path changes."""
        if self._state:
            self._state._cache_config.cache_img_path = event.value or ""

    @on(Checkbox.Changed, "#use-existing-checkbox")
    def handle_use_existing_change(self, event: Checkbox.Changed) -> None:
        """Handle use existing changes."""
        if self._state:
            self._state._cache_config.use_existing_imgs = event.value

    @on(Checkbox.Changed, "#delete-cache-checkbox")
    def handle_delete_cache_change(self, event: Checkbox.Changed) -> None:
        """Handle delete cache changes."""
        if self._state:
            self._state._cache_config.delete_cache_after_training = event.value

    @on(Checkbox.Changed, "#ohkm-checkbox")
    def handle_ohkm_change(self, event: Checkbox.Changed) -> None:
        """Handle ohkm changes."""
        if self._state:
            self._state._ohkm.enabled = event.value

    @on(Input.Changed, "#ohkm-ratio-input")
    def handle_ohkm_ratio_change(self, event: Input.Changed) -> None:
        """Handle ohkm ratio changes."""
        if self._state and event.value:
            try:
                self._state._ohkm.hard_to_easy_ratio = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#ohkm-scale-input")
    def handle_ohkm_scale_change(self, event: Input.Changed) -> None:
        """Handle ohkm scale changes."""
        if self._state and event.value:
            try:
                self._state._ohkm.loss_scale = float(event.value)
            except ValueError:
                pass

    # Top-down specific handlers
    @on(Select.Changed, "#anchor-part-select")
    def handle_anchor_change(self, event: Select.Changed) -> None:
        """Handle anchor changes."""
        if self._state:
            self._state._anchor_part = event.value if event.value else None

    @on(Input.Changed, "#centroid-scale-input")
    def handle_centroid_scale_change(self, event: Input.Changed) -> None:
        """Handle centroid scale changes."""
        if self._state and event.value:
            try:
                self._state._input_scale = float(event.value)
                self._update_centroid_shape_display()
                self._update_centroid_model_info_display()
                self._update_centroid_memory_display()
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-sigma-input")
    def handle_centroid_sigma_change(self, event: Input.Changed) -> None:
        """Handle centroid sigma changes."""
        if self._state and event.value:
            try:
                self._state._sigma = float(event.value)
                self._update_centroid_model_info_display()
            except ValueError:
                pass

    @on(Select.Changed, "#centroid-backbone-select")
    def handle_centroid_backbone_change(self, event: Select.Changed) -> None:
        """Handle centroid backbone changes."""
        if self._state and event.value:
            self._state._backbone = event.value
            self._update_centroid_model_info_display()
            self._update_centroid_memory_display()
            self._update_imagenet_visibility()

    @on(Input.Changed, "#crop-size-input")
    def handle_crop_size_change(self, event: Input.Changed) -> None:
        """Handle crop size changes."""
        if self._state and event.value:
            try:
                self._state._crop_size = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#min-crop-size-input")
    def handle_min_crop_size_change(self, event: Input.Changed) -> None:
        """Handle min crop size changes."""
        if self._state and event.value:
            try:
                self._state._ci_min_crop_size = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-scale-input")
    def handle_instance_scale_change(self, event: Input.Changed) -> None:
        """Handle instance scale changes."""
        if self._state and event.value:
            try:
                self._state._ci_input_scale = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-sigma-input")
    def handle_instance_sigma_change(self, event: Input.Changed) -> None:
        """Handle instance sigma changes."""
        if self._state and event.value:
            try:
                self._state._ci_sigma = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#instance-backbone-select")
    def handle_instance_backbone_change(self, event: Select.Changed) -> None:
        """Handle instance backbone changes."""
        if self._state and event.value:
            self._state._ci_backbone = event.value

    @on(Select.Changed, "#instance-output-stride-select")
    def handle_instance_output_stride_change(self, event: Select.Changed) -> None:
        """Handle instance output stride changes."""
        if self._state and event.value:
            self._state._ci_output_stride = int(event.value)

    # Additional top-down handlers
    @on(Input.Changed, "#centroid-max-height-input")
    def handle_centroid_max_height_change(self, event: Input.Changed) -> None:
        """Handle centroid max height changes."""
        if self._state:
            self._state._max_height = int(event.value) if event.value else None

    @on(Input.Changed, "#centroid-max-width-input")
    def handle_centroid_max_width_change(self, event: Input.Changed) -> None:
        """Handle centroid max width changes."""
        if self._state:
            self._state._max_width = int(event.value) if event.value else None

    @on(Select.Changed, "#centroid-max-stride-select")
    def handle_centroid_max_stride_change(self, event: Select.Changed) -> None:
        """Handle centroid max stride changes."""
        if self._state and event.value:
            self._state._max_stride = int(event.value)
            self._update_centroid_model_info_display()

    @on(Select.Changed, "#centroid-output-stride-select")
    def handle_centroid_output_stride_change(self, event: Select.Changed) -> None:
        """Handle centroid output stride changes."""
        if self._state and event.value:
            self._state._output_stride = int(event.value)
            self._update_centroid_shape_display()
            self._update_centroid_model_info_display()

    @on(Select.Changed, "#centroid-batch-size-select")
    def handle_centroid_batch_change(self, event: Select.Changed) -> None:
        """Handle centroid batch changes."""
        if self._state and event.value:
            self._state._batch_size = int(event.value)
            self._update_centroid_memory_display()

    @on(Input.Changed, "#centroid-max-epochs-input")
    def handle_centroid_epochs_change(self, event: Input.Changed) -> None:
        """Handle centroid epochs changes."""
        if self._state and event.value:
            try:
                self._state._max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-lr-input")
    def handle_centroid_lr_change(self, event: Input.Changed) -> None:
        """Handle centroid lr changes."""
        if self._state and event.value:
            try:
                self._state._learning_rate = float(event.value)
            except ValueError:
                pass

    # Centroid augmentation checkbox handlers
    @on(Checkbox.Changed, "#centroid-rotation-checkbox")
    def handle_centroid_rotation_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid rotation changes."""
        if self._state:
            self._state._augmentation.rotation_enabled = event.value

    @on(Checkbox.Changed, "#centroid-scale-checkbox")
    def handle_centroid_scale_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid scale changes."""
        if self._state:
            self._state._augmentation.scale_enabled = event.value

    @on(Checkbox.Changed, "#centroid-brightness-checkbox")
    def handle_centroid_brightness_checkbox_change(
        self, event: Checkbox.Changed
    ) -> None:
        """Handle centroid brightness changes."""
        if self._state:
            self._state._augmentation.brightness_enabled = event.value

    @on(Checkbox.Changed, "#centroid-contrast-checkbox")
    def handle_centroid_contrast_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid contrast changes."""
        if self._state:
            self._state._augmentation.contrast_enabled = event.value

    @on(Input.Changed, "#centroid-rotation-input")
    def handle_centroid_rotation_change(self, event: Input.Changed) -> None:
        """Handle centroid rotation changes."""
        if self._state and event.value:
            try:
                val = float(event.value)
                self._state._augmentation.rotation_max = val
                self._state._augmentation.rotation_min = -val
            except ValueError:
                pass

    @on(Select.Changed, "#instance-max-stride-select")
    def handle_instance_max_stride_change(self, event: Select.Changed) -> None:
        """Handle instance max stride changes."""
        if self._state and event.value:
            self._state._ci_max_stride = int(event.value)

    @on(Select.Changed, "#instance-batch-size-select")
    def handle_instance_batch_change(self, event: Select.Changed) -> None:
        """Handle instance batch changes."""
        if self._state and event.value:
            self._state._ci_batch_size = int(event.value)

    @on(Input.Changed, "#instance-max-epochs-input")
    def handle_instance_epochs_change(self, event: Input.Changed) -> None:
        """Handle instance epochs changes."""
        if self._state and event.value:
            try:
                self._state._ci_max_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-lr-input")
    def handle_instance_lr_change(self, event: Input.Changed) -> None:
        """Handle instance lr changes."""
        if self._state and event.value:
            try:
                self._state._ci_learning_rate = float(event.value)
            except ValueError:
                pass

    # Instance augmentation checkbox handlers
    @on(Checkbox.Changed, "#instance-rotation-checkbox")
    def handle_instance_rotation_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle instance rotation changes."""
        if self._state:
            self._state._ci_augmentation.rotation_enabled = event.value

    @on(Checkbox.Changed, "#instance-scale-checkbox")
    def handle_instance_scale_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle instance scale changes."""
        if self._state:
            self._state._ci_augmentation.scale_enabled = event.value

    @on(Checkbox.Changed, "#instance-brightness-checkbox")
    def handle_instance_brightness_checkbox_change(
        self, event: Checkbox.Changed
    ) -> None:
        """Handle instance brightness changes."""
        if self._state:
            self._state._ci_augmentation.brightness_enabled = event.value

    @on(Checkbox.Changed, "#instance-contrast-checkbox")
    def handle_instance_contrast_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle instance contrast changes."""
        if self._state:
            self._state._ci_augmentation.contrast_enabled = event.value

    @on(Input.Changed, "#instance-rotation-input")
    def handle_instance_rotation_change(self, event: Input.Changed) -> None:
        """Handle instance rotation changes."""
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
        """Handle centroid ckpt dir changes."""
        if self._state:
            self._state._checkpoint.checkpoint_dir = event.value or ""

    @on(Input.Changed, "#centroid-run-name-input")
    def handle_centroid_run_name_change(self, event: Input.Changed) -> None:
        """Handle centroid run name changes."""
        if self._state:
            self._state._checkpoint.run_name = event.value or ""

    @on(Input.Changed, "#instance-ckpt-dir-input")
    def handle_instance_ckpt_dir_change(self, event: Input.Changed) -> None:
        """Handle instance ckpt dir changes."""
        if self._state:
            self._state._ci_checkpoint_dir = event.value or ""

    @on(Input.Changed, "#instance-run-name-input")
    def handle_instance_run_name_change(self, event: Input.Changed) -> None:
        """Handle instance run name changes."""
        if self._state:
            self._state._ci_run_name = event.value or ""

    # Centroid augmentation handlers
    @on(Input.Changed, "#centroid-scale-min-input")
    def handle_centroid_scale_min_change(self, event: Input.Changed) -> None:
        """Handle centroid scale min changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.scale_min = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-scale-max-input")
    def handle_centroid_scale_max_change(self, event: Input.Changed) -> None:
        """Handle centroid scale max changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.scale_max = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-brightness-input")
    def handle_centroid_brightness_change(self, event: Input.Changed) -> None:
        """Handle centroid brightness changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.brightness_limit = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-contrast-input")
    def handle_centroid_contrast_change(self, event: Input.Changed) -> None:
        """Handle centroid contrast changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.contrast_limit = float(event.value)
            except ValueError:
                pass

    # Instance augmentation handlers
    @on(Input.Changed, "#instance-scale-min-input")
    def handle_instance_scale_min_change(self, event: Input.Changed) -> None:
        """Handle instance scale min changes."""
        if self._state and event.value:
            try:
                self._state._ci_augmentation.scale_min = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-scale-max-input")
    def handle_instance_scale_max_change(self, event: Input.Changed) -> None:
        """Handle instance scale max changes."""
        if self._state and event.value:
            try:
                self._state._ci_augmentation.scale_max = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-brightness-input")
    def handle_instance_brightness_change(self, event: Input.Changed) -> None:
        """Handle instance brightness changes."""
        if self._state and event.value:
            try:
                self._state._ci_augmentation.brightness_limit = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-contrast-input")
    def handle_instance_contrast_change(self, event: Input.Changed) -> None:
        """Handle instance contrast changes."""
        if self._state and event.value:
            try:
                self._state._ci_augmentation.contrast_limit = float(event.value)
            except ValueError:
                pass

    # Instance Checkpoint & Logging handlers
    @on(Checkbox.Changed, "#instance-save-ckpt-checkbox")
    def handle_instance_save_ckpt_change(self, event: Checkbox.Changed) -> None:
        """Handle instance save ckpt changes."""
        if self._state:
            self._state._checkpoint.enabled = event.value

    @on(Input.Changed, "#instance-save-top-k-input")
    def handle_instance_save_top_k_change(self, event: Input.Changed) -> None:
        """Handle instance save top k changes."""
        if self._state and event.value:
            try:
                self._state._checkpoint.save_top_k = int(event.value)
            except ValueError:
                pass

    @on(Checkbox.Changed, "#instance-save-last-checkbox")
    def handle_instance_save_last_change(self, event: Checkbox.Changed) -> None:
        """Handle instance save last changes."""
        if self._state:
            self._state._checkpoint.save_last = event.value

    @on(Checkbox.Changed, "#instance-wandb-checkbox")
    def handle_instance_wandb_change(self, event: Checkbox.Changed) -> None:
        """Handle instance wandb changes."""
        if self._state:
            self._state._wandb.enabled = event.value
            self._update_wandb_options_visibility()

    @on(Input.Changed, "#instance-wandb-project-input")
    def handle_instance_wandb_project_change(self, event: Input.Changed) -> None:
        """Handle instance wandb project changes."""
        if self._state:
            self._state._wandb.project = event.value or "sleap-training"

    @on(Input.Changed, "#instance-wandb-entity-input")
    def handle_instance_wandb_entity_change(self, event: Input.Changed) -> None:
        """Handle instance wandb entity changes."""
        if self._state:
            self._state._wandb.entity = event.value or ""

    @on(Checkbox.Changed, "#instance-save-viz-checkbox")
    def handle_instance_save_viz_change(self, event: Checkbox.Changed) -> None:
        """Handle instance save viz changes."""
        if self._state:
            self._state._visualize_preds = event.value
            self._update_viz_options_visibility()

    @on(Checkbox.Changed, "#instance-viz-keep-folder-checkbox")
    def handle_instance_viz_keep_folder_change(self, event: Checkbox.Changed) -> None:
        """Handle instance viz keep folder changes."""
        if self._state:
            self._state._keep_viz = event.value

    @on(Checkbox.Changed, "#instance-eval-checkbox")
    def handle_instance_eval_change(self, event: Checkbox.Changed) -> None:
        """Handle instance eval changes."""
        if self._state:
            self._state._evaluation.enabled = event.value
            self._update_eval_options_visibility()

    @on(Input.Changed, "#instance-eval-frequency-input")
    def handle_instance_eval_frequency_change(self, event: Input.Changed) -> None:
        """Handle instance eval frequency changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.frequency = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-eval-oks-input")
    def handle_instance_eval_oks_change(self, event: Input.Changed) -> None:
        """Handle instance eval oks changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.oks_stddev = float(event.value)
            except ValueError:
                pass

    # Instance Advanced params handlers
    @on(Select.Changed, "#instance-optimizer-select")
    def handle_instance_optimizer_change(self, event: Select.Changed) -> None:
        """Handle instance optimizer changes."""
        if self._state and event.value:
            self._state._ci_optimizer = event.value

    @on(Checkbox.Changed, "#instance-early-stopping-checkbox")
    def handle_instance_early_stopping_change(self, event: Checkbox.Changed) -> None:
        """Handle instance early stopping changes."""
        if self._state:
            self._state._ci_early_stopping = event.value

    @on(Input.Changed, "#instance-es-patience-input")
    def handle_instance_es_patience_change(self, event: Input.Changed) -> None:
        """Handle instance es patience changes."""
        if self._state and event.value:
            try:
                self._state._ci_early_stopping_patience = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-es-min-delta-input")
    def handle_instance_es_min_delta_change(self, event: Input.Changed) -> None:
        """Handle instance es min delta changes."""
        if self._state and event.value:
            try:
                self._state._ci_early_stopping_min_delta = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#instance-scheduler-select")
    def handle_instance_scheduler_change(self, event: Select.Changed) -> None:
        """Handle instance scheduler changes."""
        if self._state and event.value:
            from sleap_nn.config_generator.tui.state import SchedulerType

            self._state._ci_scheduler.type = SchedulerType(event.value)

    @on(Input.Changed, "#instance-lr-patience-input")
    def handle_instance_lr_patience_change(self, event: Input.Changed) -> None:
        """Handle instance lr patience changes."""
        if self._state and event.value:
            try:
                self._state._ci_scheduler.plateau_patience = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-lr-factor-input")
    def handle_instance_lr_factor_change(self, event: Input.Changed) -> None:
        """Handle instance lr factor changes."""
        if self._state and event.value:
            try:
                self._state._ci_scheduler.factor = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#instance-accelerator-select")
    def handle_instance_accelerator_change(self, event: Select.Changed) -> None:
        """Handle instance accelerator changes."""
        if self._state and event.value:
            self._state._accelerator = event.value

    @on(Select.Changed, "#instance-devices-select")
    def handle_instance_devices_change(self, event: Select.Changed) -> None:
        """Handle instance devices changes."""
        if self._state and event.value:
            self._state._devices = event.value

    @on(Input.Changed, "#instance-min-steps-input")
    def handle_instance_min_steps_change(self, event: Input.Changed) -> None:
        """Handle instance min steps changes."""
        if self._state and event.value:
            try:
                self._state._min_steps_per_epoch = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-random-seed-input")
    def handle_instance_random_seed_change(self, event: Input.Changed) -> None:
        """Handle instance random seed changes."""
        if self._state:
            self._state._random_seed = int(event.value) if event.value else None

    @on(Select.Changed, "#instance-lr-select")
    def handle_instance_lr_select_change(self, event: Select.Changed) -> None:
        """Handle instance lr changes."""
        if self._state and event.value:
            self._state._ci_learning_rate = float(event.value)

    # Instance Data Pipeline handlers
    @on(Select.Changed, "#instance-num-workers-select")
    def handle_instance_num_workers_select_change(self, event: Select.Changed) -> None:
        """Handle instance num workers changes."""
        if self._state and event.value:
            self._state._num_workers = int(event.value)
            self._update_instance_cache_memory_display()

    @on(Select.Changed, "#instance-data-pipeline-select")
    def handle_instance_data_pipeline_change(self, event: Select.Changed) -> None:
        """Handle instance data pipeline changes."""
        if self._state and event.value:
            from sleap_nn.config_generator.tui.state import DataPipelineType

            self._state._data_pipeline = DataPipelineType(event.value)
            self._update_instance_cache_memory_display()
            self._update_instance_cache_options_visibility()

    @on(Checkbox.Changed, "#instance-parallel-caching-checkbox")
    def handle_instance_parallel_caching_change(self, event: Checkbox.Changed) -> None:
        """Handle instance parallel caching changes."""
        if self._state:
            self._state._cache_config.parallel_caching = event.value

    @on(Input.Changed, "#instance-cache-workers-input")
    def handle_instance_cache_workers_change(self, event: Input.Changed) -> None:
        """Handle instance cache workers changes."""
        if self._state and event.value:
            try:
                self._state._cache_config.cache_workers = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-cache-path-input")
    def handle_instance_cache_path_change(self, event: Input.Changed) -> None:
        """Handle instance cache path changes."""
        if self._state:
            self._state._cache_config.cache_img_path = event.value or ""

    @on(Checkbox.Changed, "#instance-use-existing-checkbox")
    def handle_instance_use_existing_change(self, event: Checkbox.Changed) -> None:
        """Handle instance use existing changes."""
        if self._state:
            self._state._cache_config.use_existing_imgs = event.value

    @on(Checkbox.Changed, "#instance-delete-cache-checkbox")
    def handle_instance_delete_cache_change(self, event: Checkbox.Changed) -> None:
        """Handle instance delete cache changes."""
        if self._state:
            self._state._cache_config.delete_cache_after_training = event.value

    @on(Select.Changed, "#instance-channels-select")
    def handle_instance_channels_change(self, event: Select.Changed) -> None:
        """Handle instance channels changes."""
        if self._state and event.value:
            self._state._ensure_rgb = event.value == "rgb"
            self._state._ensure_grayscale = event.value == "grayscale"

    # Instance Model handlers
    @on(Input.Changed, "#instance-filters-input")
    def handle_instance_filters_change(self, event: Input.Changed) -> None:
        """Handle instance filters changes."""
        if self._state and event.value:
            try:
                self._state._ci_filters = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-filters-rate-input")
    def handle_instance_filters_rate_change(self, event: Input.Changed) -> None:
        """Handle instance filters rate changes."""
        if self._state and event.value:
            try:
                self._state._ci_filters_rate = float(event.value)
            except ValueError:
                pass

    @on(Checkbox.Changed, "#instance-imagenet-checkbox")
    def handle_instance_imagenet_change(self, event: Checkbox.Changed) -> None:
        """Handle instance imagenet changes."""
        if self._state:
            self._state._use_imagenet_pretrained = event.value

    @on(Input.Changed, "#instance-pretrained-backbone-input")
    def handle_instance_pretrained_backbone_change(self, event: Input.Changed) -> None:
        """Handle instance pretrained backbone changes."""
        if self._state:
            self._state._ci_pretrained_backbone = event.value or ""

    @on(Input.Changed, "#instance-pretrained-head-input")
    def handle_instance_pretrained_head_change(self, event: Input.Changed) -> None:
        """Handle instance pretrained head changes."""
        if self._state:
            self._state._ci_pretrained_head = event.value or ""

    # Instance OHKM handlers
    @on(Checkbox.Changed, "#instance-ohkm-checkbox")
    def handle_instance_ohkm_change(self, event: Checkbox.Changed) -> None:
        """Handle instance ohkm changes."""
        if self._state:
            self._state._ohkm.enabled = event.value

    @on(Input.Changed, "#instance-ohkm-ratio-input")
    def handle_instance_ohkm_ratio_change(self, event: Input.Changed) -> None:
        """Handle instance ohkm ratio changes."""
        if self._state and event.value:
            try:
                self._state._ohkm.hard_to_easy_ratio = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-ohkm-scale-input")
    def handle_instance_ohkm_scale_change(self, event: Input.Changed) -> None:
        """Handle instance ohkm scale changes."""
        if self._state and event.value:
            try:
                self._state._ohkm.loss_scale = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-ohkm-min-input")
    def handle_instance_ohkm_min_change(self, event: Input.Changed) -> None:
        """Handle instance ohkm min changes."""
        if self._state and event.value:
            try:
                self._state._ohkm.min_hard_keypoints = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#instance-ohkm-max-input")
    def handle_instance_ohkm_max_change(self, event: Input.Changed) -> None:
        """Handle instance ohkm max changes."""
        if self._state:
            self._state._ohkm.max_hard_keypoints = (
                int(event.value) if event.value else None
            )

    # Evaluation handlers (legacy)
    @on(Checkbox.Changed, "#evaluation-checkbox")
    def handle_evaluation_change(self, event: Checkbox.Changed) -> None:
        """Handle evaluation changes."""
        if self._state:
            self._state._evaluation.enabled = event.value
            self._update_eval_options_visibility()

    @on(Input.Changed, "#eval-frequency-input")
    def handle_eval_frequency_change(self, event: Input.Changed) -> None:
        """Handle eval frequency changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.frequency = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#eval-oks-input")
    def handle_eval_oks_change(self, event: Input.Changed) -> None:
        """Handle eval oks changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.oks_stddev = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#oks-stddev-input")
    def handle_oks_stddev_change(self, event: Input.Changed) -> None:
        """Handle oks stddev changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.oks_stddev = float(event.value)
            except ValueError:
                pass

    # Scheduler visibility helper
    def _update_scheduler_params_visibility(self, scheduler_type: str) -> None:
        """Show/hide scheduler parameter containers based on selected type."""
        containers = {
            "scheduler-reduce-params": SchedulerType.REDUCE_ON_PLATEAU,
            "scheduler-step-params": SchedulerType.STEP_LR,
            "scheduler-cosine-params": SchedulerType.COSINE_ANNEALING_WARMUP,
            "scheduler-linear-params": SchedulerType.LINEAR_WARMUP_LINEAR_DECAY,
        }
        for container_id, sched_type in containers.items():
            try:
                container = self.query_one(f"#{container_id}", Container)
                if scheduler_type == sched_type.value:
                    container.remove_class("hidden")
                else:
                    container.add_class("hidden")
            except Exception:
                pass

    # Scheduler parameter handlers - ReduceLROnPlateau
    @on(Input.Changed, "#scheduler-factor-input")
    def handle_scheduler_factor_change(self, event: Input.Changed) -> None:
        """Handle scheduler factor changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.factor = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-patience-input")
    def handle_scheduler_patience_change(self, event: Input.Changed) -> None:
        """Handle scheduler patience changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.plateau_patience = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-min-lr-input")
    def handle_scheduler_min_lr_change(self, event: Input.Changed) -> None:
        """Handle scheduler min lr changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.min_lr = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-cooldown-input")
    def handle_scheduler_cooldown_change(self, event: Input.Changed) -> None:
        """Handle scheduler cooldown changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.cooldown = int(event.value)
            except ValueError:
                pass

    # Scheduler parameter handlers - StepLR
    @on(Input.Changed, "#scheduler-step-size-input")
    def handle_scheduler_step_size_change(self, event: Input.Changed) -> None:
        """Handle scheduler step size changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.step_size = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-gamma-input")
    def handle_scheduler_gamma_change(self, event: Input.Changed) -> None:
        """Handle scheduler gamma changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.gamma = float(event.value)
            except ValueError:
                pass

    # Scheduler parameter handlers - CosineAnnealingWarmup
    @on(Input.Changed, "#scheduler-warmup-epochs-input")
    def handle_scheduler_warmup_epochs_change(self, event: Input.Changed) -> None:
        """Handle scheduler warmup epochs changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.warmup_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-warmup-start-lr-input")
    def handle_scheduler_warmup_start_lr_change(self, event: Input.Changed) -> None:
        """Handle scheduler warmup start lr changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.warmup_start_lr = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-eta-min-input")
    def handle_scheduler_eta_min_change(self, event: Input.Changed) -> None:
        """Handle scheduler eta min changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.eta_min = float(event.value)
            except ValueError:
                pass

    # Scheduler parameter handlers - LinearWarmupLinearDecay
    @on(Input.Changed, "#scheduler-linear-warmup-epochs-input")
    def handle_scheduler_linear_warmup_epochs_change(
        self, event: Input.Changed
    ) -> None:
        """Handle scheduler linear warmup epochs changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.linear_warmup_epochs = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-linear-warmup-start-lr-input")
    def handle_scheduler_linear_warmup_start_lr_change(
        self, event: Input.Changed
    ) -> None:
        """Handle scheduler linear warmup start lr changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.linear_warmup_start_lr = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#scheduler-end-lr-input")
    def handle_scheduler_end_lr_change(self, event: Input.Changed) -> None:
        """Handle scheduler end lr changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.end_lr = float(event.value)
            except ValueError:
                pass

    # WandB visualization handlers
    @on(Checkbox.Changed, "#wandb-viz-checkbox")
    def handle_wandb_viz_change(self, event: Checkbox.Changed) -> None:
        """Handle wandb viz changes."""
        if self._state:
            self._state._wandb.viz_enabled = event.value

    @on(Checkbox.Changed, "#wandb-save-viz-checkbox")
    def handle_wandb_save_viz_change(self, event: Checkbox.Changed) -> None:
        """Handle wandb save viz changes."""
        if self._state:
            self._state._wandb.save_viz_imgs = event.value

    # Devices handler
    @on(Select.Changed, "#devices-select")
    def handle_devices_change(self, event: Select.Changed) -> None:
        """Handle devices changes."""
        if self._state and event.value:
            self._state._devices = event.value

    # ==================== CENTROID-SPECIFIC HANDLERS ====================
    # These handle inputs specific to the centroid tab that aren't covered by standard handlers

    @on(Checkbox.Changed, "#centroid-translate-checkbox")
    def handle_centroid_translate_checkbox_change(
        self, event: Checkbox.Changed
    ) -> None:
        """Handle centroid translate changes."""
        if self._state:
            self._state._augmentation.translate_enabled = event.value

    @on(Input.Changed, "#centroid-translate-input")
    def handle_centroid_translate_change(self, event: Input.Changed) -> None:
        """Handle centroid translate changes."""
        if self._state and event.value:
            try:
                self._state._augmentation.translate = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-val-fraction-input")
    def handle_centroid_val_fraction_change(self, event: Input.Changed) -> None:
        """Handle centroid val fraction changes."""
        if self._state and event.value:
            try:
                self._state._validation_fraction = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#centroid-channels-select")
    def handle_centroid_channels_change(self, event: Select.Changed) -> None:
        """Handle centroid channels changes."""
        if self._state and event.value:
            self._state._ensure_rgb = event.value == "rgb"
            self._state._ensure_grayscale = event.value == "grayscale"
            self._update_centroid_shape_display()
            self._update_centroid_memory_display()
            self._update_centroid_model_info_display()

    @on(Select.Changed, "#centroid-data-pipeline-select")
    def handle_centroid_data_pipeline_change(self, event: Select.Changed) -> None:
        """Handle centroid data pipeline changes."""
        if self._state and event.value:
            self._state._data_pipeline = DataPipelineType(event.value)
            self._update_cache_memory_display()
            self._update_centroid_cache_memory_display()
            self._update_centroid_cache_options_visibility()

    @on(Input.Changed, "#centroid-filters-input")
    def handle_centroid_filters_change(self, event: Input.Changed) -> None:
        """Handle centroid filters changes."""
        if self._state and event.value:
            try:
                self._state._filters = int(event.value)
                self._update_centroid_model_info_display()
                self._update_centroid_memory_display()
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-filters-rate-input")
    def handle_centroid_filters_rate_change(self, event: Input.Changed) -> None:
        """Handle centroid filters rate changes."""
        if self._state and event.value:
            try:
                self._state._filters_rate = float(event.value)
                self._update_centroid_model_info_display()
                self._update_centroid_memory_display()
            except ValueError:
                pass

    @on(Checkbox.Changed, "#centroid-imagenet-checkbox")
    def handle_centroid_imagenet_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid imagenet changes."""
        if self._state:
            self._state._use_imagenet_pretrained = event.value

    @on(Input.Changed, "#centroid-pretrained-backbone-input")
    def handle_centroid_pretrained_backbone_change(self, event: Input.Changed) -> None:
        """Handle centroid pretrained backbone changes."""
        if self._state:
            self._state._pretrained_backbone = event.value or ""

    @on(Input.Changed, "#centroid-pretrained-head-input")
    def handle_centroid_pretrained_head_change(self, event: Input.Changed) -> None:
        """Handle centroid pretrained head changes."""
        if self._state:
            self._state._pretrained_head = event.value or ""

    @on(Select.Changed, "#centroid-optimizer-select")
    def handle_centroid_optimizer_change(self, event: Select.Changed) -> None:
        """Handle centroid optimizer changes."""
        if self._state and event.value:
            self._state._optimizer = event.value

    @on(Checkbox.Changed, "#centroid-early-stopping-checkbox")
    def handle_centroid_early_stopping_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid early stopping changes."""
        if self._state:
            self._state._early_stopping = event.value

    @on(Input.Changed, "#centroid-es-patience-input")
    def handle_centroid_es_patience_change(self, event: Input.Changed) -> None:
        """Handle centroid es patience changes."""
        if self._state and event.value:
            try:
                self._state._early_stopping_patience = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-es-min-delta-input")
    def handle_centroid_es_min_delta_change(self, event: Input.Changed) -> None:
        """Handle centroid es min delta changes."""
        if self._state and event.value:
            try:
                self._state._early_stopping_min_delta = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-lr-patience-input")
    def handle_centroid_lr_patience_change(self, event: Input.Changed) -> None:
        """Handle centroid lr patience changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.plateau_patience = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-lr-factor-input")
    def handle_centroid_lr_factor_change(self, event: Input.Changed) -> None:
        """Handle centroid lr factor changes."""
        if self._state and event.value:
            try:
                self._state._scheduler.factor = float(event.value)
            except ValueError:
                pass

    @on(Select.Changed, "#centroid-scheduler-select")
    def handle_centroid_scheduler_change(self, event: Select.Changed) -> None:
        """Handle centroid scheduler changes."""
        if self._state and event.value:
            self._state._scheduler.type = SchedulerType(event.value)

    @on(Checkbox.Changed, "#centroid-save-ckpt-checkbox")
    def handle_centroid_save_ckpt_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid save ckpt changes."""
        if self._state:
            self._state._checkpoint.enabled = event.value

    @on(Input.Changed, "#centroid-save-top-k-input")
    def handle_centroid_save_top_k_change(self, event: Input.Changed) -> None:
        """Handle centroid save top k changes."""
        if self._state and event.value:
            try:
                self._state._checkpoint.save_top_k = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-resume-ckpt-input")
    def handle_centroid_resume_ckpt_change(self, event: Input.Changed) -> None:
        """Handle centroid resume ckpt changes."""
        if self._state:
            self._state._checkpoint.resume_from = event.value or ""

    @on(Checkbox.Changed, "#centroid-wandb-checkbox")
    def handle_centroid_wandb_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid wandb changes."""
        if self._state:
            self._state._wandb.enabled = event.value
            self._update_wandb_options_visibility()

    @on(Checkbox.Changed, "#centroid-save-viz-checkbox")
    def handle_centroid_save_viz_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid save viz changes."""
        if self._state:
            self._state._visualize_preds = event.value
            self._update_viz_options_visibility()

    @on(Checkbox.Changed, "#centroid-viz-keep-folder-checkbox")
    def handle_centroid_viz_keep_folder_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid viz keep folder changes."""
        if self._state:
            self._state._keep_viz = event.value

    @on(Checkbox.Changed, "#centroid-eval-checkbox")
    def handle_centroid_eval_change(self, event: Checkbox.Changed) -> None:
        """Handle centroid eval changes."""
        if self._state:
            self._state._evaluation.enabled = event.value
            self._update_eval_options_visibility()

    @on(Input.Changed, "#centroid-eval-frequency-input")
    def handle_centroid_eval_frequency_change(self, event: Input.Changed) -> None:
        """Handle centroid eval frequency changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.frequency = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-eval-oks-input")
    def handle_centroid_eval_oks_change(self, event: Input.Changed) -> None:
        """Handle centroid eval oks changes."""
        if self._state and event.value:
            try:
                self._state._evaluation.oks_stddev = float(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-wandb-project-input")
    def handle_centroid_wandb_project_change(self, event: Input.Changed) -> None:
        """Handle centroid wandb project changes."""
        if self._state:
            self._state._wandb.project = event.value or "sleap-training"

    @on(Select.Changed, "#centroid-accelerator-select")
    def handle_centroid_accelerator_change(self, event: Select.Changed) -> None:
        """Handle centroid accelerator changes."""
        if self._state and event.value:
            self._state._accelerator = event.value

    @on(Select.Changed, "#centroid-devices-select")
    def handle_centroid_devices_change(self, event: Select.Changed) -> None:
        """Handle centroid devices changes."""
        if self._state and event.value:
            self._state._devices = event.value

    @on(Select.Changed, "#centroid-num-workers-select")
    def handle_centroid_num_workers_change(self, event: Select.Changed) -> None:
        """Handle centroid num workers changes."""
        if self._state and event.value:
            self._state._num_workers = int(event.value)
            self._update_centroid_cache_memory_display()

    # Legacy handler kept for compatibility (can be removed later)
    @on(Input.Changed, "#centroid-num-workers-input")
    def handle_centroid_num_workers_input_change(self, event: Input.Changed) -> None:
        """Handle centroid num workers changes."""
        if self._state and event.value:
            try:
                self._state._num_workers = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-min-steps-input")
    def handle_centroid_min_steps_change(self, event: Input.Changed) -> None:
        """Handle centroid min steps changes."""
        if self._state and event.value:
            try:
                self._state._min_steps_per_epoch = int(event.value)
            except ValueError:
                pass

    @on(Input.Changed, "#centroid-random-seed-input")
    def handle_centroid_random_seed_change(self, event: Input.Changed) -> None:
        """Handle centroid random seed changes."""
        if self._state:
            self._state._random_seed = int(event.value) if event.value else None
