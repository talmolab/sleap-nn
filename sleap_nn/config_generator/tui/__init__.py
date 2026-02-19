"""TUI module for interactive configuration generation.

This module provides an interactive terminal user interface for
generating sleap-nn training configurations.
"""

from sleap_nn.config_generator.tui.app import ConfigGeneratorApp, launch_tui
from sleap_nn.config_generator.tui.state import ConfigState

__all__ = ["ConfigGeneratorApp", "ConfigState", "launch_tui"]
