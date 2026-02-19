"""Info box widget variants.

Provides styled information boxes for displaying hints, warnings, and errors.
"""

from enum import Enum
from typing import Optional

from textual.widgets import Static


class InfoBoxType(str, Enum):
    """Types of info boxes with different styling."""

    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"
    TIP = "tip"


class InfoBox(Static):
    """Styled information box widget.

    Displays informational content with visual styling to indicate
    the type of information (info, warning, success, error, tip).

    Attributes:
        box_type: The type of info box (affects styling).
        title: Optional title text.
        message: Main content text.
    """

    DEFAULT_CSS = """
    InfoBox {
        height: auto;
        padding: 1;
        margin: 1 0;
        border-left: thick $accent;
        background: $surface;
    }

    InfoBox.info {
        border-left: thick #3b82f6;
    }

    InfoBox.warning {
        border-left: thick $warning;
    }

    InfoBox.success {
        border-left: thick $success;
    }

    InfoBox.error {
        border-left: thick $error;
    }

    InfoBox.tip {
        border-left: thick #a855f7;
    }
    """

    def __init__(
        self,
        message: str,
        box_type: InfoBoxType = InfoBoxType.INFO,
        title: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the info box.

        Args:
            message: Content text.
            box_type: Type of info box.
            title: Optional title.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._message = message
        self._box_type = box_type
        self._title = title

        # Add type class
        self.add_class(box_type.value)

    @property
    def message(self) -> str:
        """Get the message content."""
        return self._message

    @message.setter
    def message(self, value: str) -> None:
        """Set the message content."""
        self._message = value
        self.refresh()

    def render(self) -> str:
        """Render the info box content."""
        lines = []

        # Icons for different types
        icons = {
            InfoBoxType.INFO: "ℹ",
            InfoBoxType.WARNING: "⚠",
            InfoBoxType.SUCCESS: "✓",
            InfoBoxType.ERROR: "✗",
            InfoBoxType.TIP: "💡",
        }
        icon = icons.get(self._box_type, "")

        if self._title:
            lines.append(f"{icon} {self._title}")
            lines.append("")

        lines.append(self._message)

        return "\n".join(lines)

    def update_message(self, message: str) -> None:
        """Update the message content.

        Args:
            message: New message text.
        """
        self._message = message
        self.refresh()

    def update_type(self, box_type: InfoBoxType) -> None:
        """Update the box type.

        Args:
            box_type: New box type.
        """
        # Remove old type class
        self.remove_class(self._box_type.value)
        # Add new type class
        self._box_type = box_type
        self.add_class(box_type.value)
        self.refresh()


class WarningBox(InfoBox):
    """Convenience class for warning-styled info box."""

    def __init__(
        self,
        message: str,
        title: Optional[str] = "Warning",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            box_type=InfoBoxType.WARNING,
            title=title,
            id=id,
            classes=classes,
        )


class SuccessBox(InfoBox):
    """Convenience class for success-styled info box."""

    def __init__(
        self,
        message: str,
        title: Optional[str] = "Success",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            box_type=InfoBoxType.SUCCESS,
            title=title,
            id=id,
            classes=classes,
        )


class ErrorBox(InfoBox):
    """Convenience class for error-styled info box."""

    def __init__(
        self,
        message: str,
        title: Optional[str] = "Error",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            box_type=InfoBoxType.ERROR,
            title=title,
            id=id,
            classes=classes,
        )


class TipBox(InfoBox):
    """Convenience class for tip-styled info box."""

    def __init__(
        self,
        message: str,
        title: Optional[str] = "Tip",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            box_type=InfoBoxType.TIP,
            title=title,
            id=id,
            classes=classes,
        )


class GuideBox(Static):
    """Multi-section guide box for parameter explanations.

    Displays structured guidance with multiple sections for
    explaining configuration parameters.
    """

    DEFAULT_CSS = """
    GuideBox {
        height: auto;
        padding: 1;
        margin: 1 0;
        border: solid $surface-lighten-2;
        background: $surface;
    }
    """

    def __init__(
        self,
        title: str = "Guide",
        sections: Optional[dict] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the guide box.

        Args:
            title: Main title.
            sections: Dict mapping section titles to content.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._title = title
        self._sections = sections or {}

    def render(self) -> str:
        """Render the guide box content."""
        lines = [self._title, "─" * len(self._title)]

        for section_title, content in self._sections.items():
            lines.append("")
            lines.append(f"▸ {section_title}")
            # Indent content
            for line in content.split("\n"):
                lines.append(f"  {line}")

        return "\n".join(lines)

    def add_section(self, title: str, content: str) -> None:
        """Add a section to the guide.

        Args:
            title: Section title.
            content: Section content.
        """
        self._sections[title] = content
        self.refresh()

    def update_sections(self, sections: dict) -> None:
        """Replace all sections.

        Args:
            sections: Dict mapping section titles to content.
        """
        self._sections = sections
        self.refresh()
