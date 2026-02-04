"""TUI widgets for config generator.

This module exports all custom widgets used in the config generator TUI.
"""

from sleap_nn.config_generator.tui.widgets.collapsible import (
    Collapsible,
    CollapsibleGroup,
    ToggleSection,
)
from sleap_nn.config_generator.tui.widgets.info_box import (
    ErrorBox,
    GuideBox,
    InfoBox,
    InfoBoxType,
    SuccessBox,
    TipBox,
    WarningBox,
)
from sleap_nn.config_generator.tui.widgets.memory_gauge import (
    MemoryBreakdownCard,
    MemoryGauge,
)
from sleap_nn.config_generator.tui.widgets.recommendation import (
    DatasetStatsPanel,
    QuickSettingsPanel,
    RecommendationPanel,
)
from sleap_nn.config_generator.tui.widgets.size_display import (
    EffectiveSizeDisplay,
    ModelInfoDisplay,
    SigmaVisualization,
    SizeDisplay,
)
from sleap_nn.config_generator.tui.widgets.slider import (
    LabeledSlider,
    RangeSlider,
)

__all__ = [
    # Collapsible widgets
    "Collapsible",
    "CollapsibleGroup",
    "ToggleSection",
    # Info boxes
    "InfoBox",
    "InfoBoxType",
    "WarningBox",
    "SuccessBox",
    "ErrorBox",
    "TipBox",
    "GuideBox",
    # Memory display
    "MemoryGauge",
    "MemoryBreakdownCard",
    # Recommendation panels
    "RecommendationPanel",
    "DatasetStatsPanel",
    "QuickSettingsPanel",
    # Size displays
    "SizeDisplay",
    "EffectiveSizeDisplay",
    "ModelInfoDisplay",
    "SigmaVisualization",
    # Sliders
    "LabeledSlider",
    "RangeSlider",
]
