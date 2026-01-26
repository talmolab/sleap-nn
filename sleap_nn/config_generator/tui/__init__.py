"""TUI module for interactive configuration generation.

This module provides an interactive terminal user interface for
generating sleap-nn training configurations.
"""

from sleap_nn.config_generator.tui.app import ConfigGeneratorApp, launch_tui

__all__ = ["ConfigGeneratorApp", "launch_tui"]
