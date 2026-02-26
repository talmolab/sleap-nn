"""Size display widget.

Shows image size transformations through the pipeline.
"""

from typing import Optional, Tuple

from textual.widgets import Static


class SizeDisplay(Static):
    """Widget displaying image size transformation pipeline.

    Shows how image dimensions change through preprocessing steps:
    Original → Scaled → Cropped → Output

    Useful for visualizing the effect of scale and output stride settings.
    """

    DEFAULT_CSS = """
    SizeDisplay {
        height: auto;
        padding: 1;
        margin: 1 0;
        border: solid $surface-lighten-2;
    }

    SizeDisplay .size-title {
        text-style: bold;
        margin-bottom: 1;
    }

    SizeDisplay .size-flow {
        height: auto;
    }

    SizeDisplay .size-step {
        color: $text;
    }

    SizeDisplay .size-value {
        color: $primary;
        text-style: bold;
    }

    SizeDisplay .size-arrow {
        color: $text-muted;
    }

    SizeDisplay .size-label {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        original: Tuple[int, int] = (0, 0),
        scale: float = 1.0,
        max_size: Optional[Tuple[int, int]] = None,
        crop_size: Optional[int] = None,
        output_stride: int = 1,
        title: str = "Image Size Pipeline",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the size display.

        Args:
            original: Original image dimensions (width, height).
            scale: Input scaling factor.
            max_size: Optional maximum size constraint (width, height).
            crop_size: Optional crop size (for centered instance).
            output_stride: Output stride for final dimensions.
            title: Display title.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._original = original
        self._scale = scale
        self._max_size = max_size
        self._crop_size = crop_size
        self._output_stride = output_stride
        self._title = title

    def update_sizes(
        self,
        original: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        max_size: Optional[Tuple[int, int]] = None,
        crop_size: Optional[int] = None,
        output_stride: Optional[int] = None,
    ) -> None:
        """Update size parameters.

        Args:
            original: New original dimensions.
            scale: New scale factor.
            max_size: New maximum size constraint.
            crop_size: New crop size.
            output_stride: New output stride.
        """
        if original is not None:
            self._original = original
        if scale is not None:
            self._scale = scale
        if max_size is not None:
            self._max_size = max_size
        if crop_size is not None:
            self._crop_size = crop_size
        if output_stride is not None:
            self._output_stride = output_stride
        self.refresh()

    def _compute_scaled(self) -> Tuple[int, int]:
        """Compute scaled dimensions."""
        w, h = self._original
        scaled_w = int(w * self._scale)
        scaled_h = int(h * self._scale)

        if self._max_size:
            max_w, max_h = self._max_size
            scaled_w = min(scaled_w, max_w)
            scaled_h = min(scaled_h, max_h)

        return scaled_w, scaled_h

    def _compute_model_input(self) -> Tuple[int, int]:
        """Compute model input dimensions."""
        if self._crop_size:
            return self._crop_size, self._crop_size
        return self._compute_scaled()

    def _compute_output(self) -> Tuple[int, int]:
        """Compute output dimensions."""
        input_w, input_h = self._compute_model_input()
        return input_w // self._output_stride, input_h // self._output_stride

    def render(self) -> str:
        """Render the size display."""
        orig_w, orig_h = self._original
        scaled_w, scaled_h = self._compute_scaled()
        input_w, input_h = self._compute_model_input()
        out_w, out_h = self._compute_output()

        lines = [
            self._title,
            "─" * len(self._title),
            "",
        ]

        # Original
        lines.append(f"Original:     {orig_w} × {orig_h}")

        # Scaled (if different)
        if self._scale != 1.0 or self._max_size:
            scale_text = f"×{self._scale:.2f}" if self._scale != 1.0 else ""
            max_text = ""
            if self._max_size:
                max_text = f" (max {self._max_size[0]}×{self._max_size[1]})"
            lines.append(f"    ↓ scale{scale_text}{max_text}")
            lines.append(f"Scaled:       {scaled_w} × {scaled_h}")

        # Cropped (if applicable)
        if self._crop_size:
            lines.append(f"    ↓ crop to {self._crop_size}px")
            lines.append(f"Model Input:  {input_w} × {input_h}")
        else:
            lines.append(f"Model Input:  {input_w} × {input_h}")

        # Output (if stride > 1)
        if self._output_stride > 1:
            lines.append(f"    ↓ stride {self._output_stride}")
        lines.append(f"Output:       {out_w} × {out_h}")

        # Summary
        lines.append("")
        total_reduction = (
            (orig_w * orig_h) / (out_w * out_h) if out_w * out_h > 0 else 0
        )
        lines.append(f"Reduction: {total_reduction:.1f}× fewer pixels")

        return "\n".join(lines)


class EffectiveSizeDisplay(Static):
    """Compact effective size display showing key dimensions.

    Shows a one-line summary of input and output sizes.
    """

    DEFAULT_CSS = """
    EffectiveSizeDisplay {
        height: auto;
        padding: 0;
    }

    EffectiveSizeDisplay .effective-label {
        color: $text-muted;
    }

    EffectiveSizeDisplay .effective-value {
        color: $primary;
        text-style: bold;
    }
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (0, 0),
        output_size: Tuple[int, int] = (0, 0),
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the effective size display.

        Args:
            input_size: Model input dimensions (width, height).
            output_size: Model output dimensions (width, height).
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._input_size = input_size
        self._output_size = output_size

    def update(
        self,
        input_size: Optional[Tuple[int, int]] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update size values.

        Args:
            input_size: New input dimensions.
            output_size: New output dimensions.
        """
        if input_size is not None:
            self._input_size = input_size
        if output_size is not None:
            self._output_size = output_size
        self.refresh()

    def render(self) -> str:
        """Render the effective size display."""
        in_w, in_h = self._input_size
        out_w, out_h = self._output_size
        return f"Input: {in_w}×{in_h}  →  Output: {out_w}×{out_h}"


class ModelInfoDisplay(Static):
    """Display for model architecture information.

    Shows key model metrics like parameter count, receptive field,
    and encoder/decoder block counts.
    """

    DEFAULT_CSS = """
    ModelInfoDisplay {
        height: auto;
        padding: 1;
        margin: 1 0;
        border: solid $surface-lighten-2;
    }

    ModelInfoDisplay .model-info-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ModelInfoDisplay .model-info-grid {
        height: auto;
    }

    ModelInfoDisplay .info-label {
        color: $text-muted;
    }

    ModelInfoDisplay .info-value {
        color: $primary;
        text-style: bold;
    }
    """

    def __init__(
        self,
        params: int = 0,
        receptive_field: int = 0,
        encoder_blocks: int = 0,
        decoder_blocks: int = 0,
        title: str = "Model Architecture",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the model info display.

        Args:
            params: Total parameter count.
            receptive_field: Receptive field size in pixels.
            encoder_blocks: Number of encoder blocks.
            decoder_blocks: Number of decoder blocks.
            title: Display title.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._params = params
        self._rf = receptive_field
        self._enc = encoder_blocks
        self._dec = decoder_blocks
        self._title = title

    def update_info(
        self,
        params: Optional[int] = None,
        receptive_field: Optional[int] = None,
        encoder_blocks: Optional[int] = None,
        decoder_blocks: Optional[int] = None,
    ) -> None:
        """Update model info values.

        Args:
            params: New parameter count.
            receptive_field: New receptive field.
            encoder_blocks: New encoder block count.
            decoder_blocks: New decoder block count.
        """
        if params is not None:
            self._params = params
        if receptive_field is not None:
            self._rf = receptive_field
        if encoder_blocks is not None:
            self._enc = encoder_blocks
        if decoder_blocks is not None:
            self._dec = decoder_blocks
        self.refresh()

    def _format_params(self, count: int) -> str:
        """Format parameter count with suffix."""
        if count >= 1e9:
            return f"{count/1e9:.1f}B"
        elif count >= 1e6:
            return f"{count/1e6:.1f}M"
        elif count >= 1e3:
            return f"{count/1e3:.1f}K"
        return str(count)

    def render(self) -> str:
        """Render the model info display."""
        lines = [
            self._title,
            "─" * len(self._title),
            "",
            f"  Parameters:     {self._format_params(self._params)}",
            f"  Receptive Field: {self._rf}px",
            f"  Encoder Blocks:  {self._enc}",
            f"  Decoder Blocks:  {self._dec}",
        ]
        return "\n".join(lines)


class SigmaVisualization(Static):
    """Visual representation of confidence map sigma.

    Shows a text-based representation of the Gaussian spread
    for confidence maps at the current sigma setting.
    """

    DEFAULT_CSS = """
    SigmaVisualization {
        height: auto;
        padding: 1;
        margin: 1 0;
        border: solid $surface-lighten-2;
    }

    SigmaVisualization .sigma-title {
        margin-bottom: 1;
    }

    SigmaVisualization .sigma-value {
        color: $primary;
        text-style: bold;
    }

    SigmaVisualization .sigma-viz {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        sigma: float = 5.0,
        output_stride: int = 1,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the sigma visualization.

        Args:
            sigma: Sigma value in pixels.
            output_stride: Output stride for scaling.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._sigma = sigma
        self._output_stride = output_stride

    def update_sigma(self, sigma: float, output_stride: Optional[int] = None) -> None:
        """Update sigma value.

        Args:
            sigma: New sigma value.
            output_stride: New output stride.
        """
        self._sigma = sigma
        if output_stride is not None:
            self._output_stride = output_stride
        self.refresh()

    def render(self) -> str:
        """Render the sigma visualization."""
        # 2 sigma covers ~95% of the Gaussian
        spread = int(self._sigma * 2)

        # Create a simple text visualization
        lines = [
            f"Sigma: {self._sigma:.1f}px",
            f"2σ spread: {spread}px (covers 95%)",
            "",
        ]

        # Visual representation using characters
        # Create a simple 1D Gaussian profile
        width = min(31, spread * 2 + 1)
        center = width // 2

        # Build visual rows using shading characters
        profile = []
        for i in range(width):
            dist = abs(i - center)
            if dist == 0:
                profile.append("█")
            elif dist <= self._sigma * 0.5:
                profile.append("▓")
            elif dist <= self._sigma:
                profile.append("▒")
            elif dist <= self._sigma * 2:
                profile.append("░")
            else:
                profile.append(" ")

        lines.append("  " + "".join(profile))
        lines.append(f"  {'─' * width}")
        lines.append(f"  {' ' * (center - 1)}↑")
        lines.append(f"  {' ' * (center - 3)}peak")

        return "\n".join(lines)
