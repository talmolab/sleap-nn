"""TUI screens for config generator.

This module exports all screen components used in the config generator TUI.
"""

from sleap_nn.config_generator.tui.screens.data_screen import DataScreen
from sleap_nn.config_generator.tui.screens.export_screen import (
    ExportScreen,
    TopDownExportScreen,
)
from sleap_nn.config_generator.tui.screens.model_screen import ModelScreen
from sleap_nn.config_generator.tui.screens.topdown_screen import TopDownScreen
from sleap_nn.config_generator.tui.screens.training_screen import TrainingScreen

__all__ = [
    "DataScreen",
    "ModelScreen",
    "TrainingScreen",
    "ExportScreen",
    "TopDownScreen",
    "TopDownExportScreen",
]
