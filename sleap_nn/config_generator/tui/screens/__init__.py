"""TUI screens for config generator.

This module exports all screen components used in the config generator TUI.
"""

from sleap_nn.config_generator.tui.screens.load_screen import LoadScreen
from sleap_nn.config_generator.tui.screens.model_select_screen import ModelSelectScreen
from sleap_nn.config_generator.tui.screens.configure_screen import ConfigureScreen
from sleap_nn.config_generator.tui.screens.export_screen import (
    ExportScreen,
    TopDownExportScreen,
)

# Legacy screens (for compatibility with PR #438 code)
from sleap_nn.config_generator.tui.screens.data_screen import DataScreen
from sleap_nn.config_generator.tui.screens.model_screen import ModelScreen
from sleap_nn.config_generator.tui.screens.topdown_screen import TopDownScreen
from sleap_nn.config_generator.tui.screens.training_screen import TrainingScreen

__all__ = [
    # New wizard screens
    "LoadScreen",
    "ModelSelectScreen",
    "ConfigureScreen",
    "ExportScreen",
    # Legacy screens
    "DataScreen",
    "ModelScreen",
    "TrainingScreen",
    "TopDownScreen",
    "TopDownExportScreen",
]
