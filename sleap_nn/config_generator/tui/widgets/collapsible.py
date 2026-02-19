"""Collapsible section widget.

Provides expandable/collapsible content sections with header toggle.
"""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static
from textual.widget import Widget


class Collapsible(Widget):
    """Collapsible section with expandable content.

    A container that can be expanded or collapsed by clicking its header.
    Useful for organizing complex forms into logical sections.

    Attributes:
        expanded: Whether the content is currently visible.
        title: The header text.
    """

    DEFAULT_CSS = """
    Collapsible {
        height: auto;
        margin: 1 0;
    }

    Collapsible .collapsible-header {
        height: auto;
        padding: 1;
        background: $surface-lighten-1;
        border: solid $surface-lighten-2;
    }

    Collapsible .collapsible-header:hover {
        background: $surface-lighten-2;
    }

    Collapsible .collapsible-header.expanded {
        border-bottom: none;
    }

    Collapsible .header-row {
        height: auto;
        width: 100%;
    }

    Collapsible .header-title {
        width: 1fr;
        text-style: bold;
    }

    Collapsible .header-indicator {
        width: auto;
        color: $text-muted;
    }

    Collapsible .collapsible-content {
        padding: 1;
        border: solid $surface-lighten-2;
        border-top: none;
    }

    Collapsible .collapsible-content.collapsed {
        display: none;
    }
    """

    expanded: reactive[bool] = reactive(True)

    class Toggled(Message):
        """Posted when the collapsible is expanded or collapsed."""

        def __init__(self, collapsible: "Collapsible", expanded: bool) -> None:
            super().__init__()
            self.collapsible = collapsible
            self.expanded = expanded

        @property
        def control(self) -> "Collapsible":
            return self.collapsible

    def __init__(
        self,
        title: str = "Section",
        collapsed: bool = False,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the collapsible section.

        Args:
            title: Header text.
            collapsed: Initial collapsed state (expanded if False).
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._title = title
        self.expanded = not collapsed
        self._content_widgets = []

    def compose_add_child(self, widget: Widget) -> None:
        """Compose a child into this widget."""
        self._content_widgets.append(widget)

    def compose(self) -> ComposeResult:
        """Compose the collapsible layout."""
        header_classes = (
            "collapsible-header expanded" if self.expanded else "collapsible-header"
        )
        with Container(classes=header_classes, id="header"):
            with Container(classes="header-row"):
                yield Static(self._title, classes="header-title")
                yield Static(
                    "▼" if self.expanded else "▶",
                    id="indicator",
                    classes="header-indicator",
                )

        content_classes = (
            "collapsible-content" if self.expanded else "collapsible-content collapsed"
        )
        yield Container(*self._content_widgets, classes=content_classes, id="content")

    def add_content(self, *widgets: Widget) -> None:
        """Add widgets to the collapsible content.

        Args:
            widgets: Widgets to add as content.
        """
        self._content_widgets.extend(widgets)

    async def on_click(self, event) -> None:
        """Handle clicks on the header to toggle."""
        # Check if click was on header
        try:
            header = self.query_one("#header", Container)
            # Simple check - if the widget containing the click is the header or child of header
            widget = event.widget
            while widget is not None:
                if widget is header:
                    self.toggle()
                    break
                if widget is self:
                    break
                widget = widget.parent
        except Exception:
            pass

    def toggle(self) -> None:
        """Toggle the expanded state."""
        self.expanded = not self.expanded
        self._update_display()
        self.post_message(self.Toggled(self, self.expanded))

    def expand(self) -> None:
        """Expand the content."""
        if not self.expanded:
            self.expanded = True
            self._update_display()
            self.post_message(self.Toggled(self, True))

    def collapse(self) -> None:
        """Collapse the content."""
        if self.expanded:
            self.expanded = False
            self._update_display()
            self.post_message(self.Toggled(self, False))

    def _update_display(self) -> None:
        """Update the visual display based on expanded state."""
        try:
            header = self.query_one("#header", Container)
            content = self.query_one("#content", Container)
            indicator = self.query_one("#indicator", Static)

            if self.expanded:
                header.add_class("expanded")
                content.remove_class("collapsed")
                indicator.update("▼")
            else:
                header.remove_class("expanded")
                content.add_class("collapsed")
                indicator.update("▶")
        except Exception:
            pass


class CollapsibleGroup(Widget):
    """Group of collapsible sections with optional accordion behavior.

    Can be configured so that only one section is expanded at a time
    (accordion mode) or allow multiple sections to be open.

    Attributes:
        accordion: If True, only one section can be expanded at a time.
    """

    DEFAULT_CSS = """
    CollapsibleGroup {
        height: auto;
    }

    CollapsibleGroup Collapsible {
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        accordion: bool = False,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the collapsible group.

        Args:
            accordion: If True, only one section can be open at a time.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._accordion = accordion

    @on(Collapsible.Toggled)
    def handle_section_toggle(self, event: Collapsible.Toggled) -> None:
        """Handle section toggle events for accordion behavior."""
        if self._accordion and event.expanded:
            # Collapse all other sections
            for collapsible in self.query(Collapsible):
                if collapsible is not event.collapsible and collapsible.expanded:
                    collapsible.collapse()


class ToggleSection(Widget):
    """Toggle-enabled section with switch control.

    A section that can be enabled/disabled with a switch,
    with the content hidden when disabled.
    """

    DEFAULT_CSS = """
    ToggleSection {
        height: auto;
        margin: 1 0;
    }

    ToggleSection .toggle-header {
        height: auto;
        padding: 1;
        background: $surface-lighten-1;
        border: solid $surface-lighten-2;
    }

    ToggleSection .toggle-row {
        height: auto;
        width: 100%;
    }

    ToggleSection .toggle-title {
        width: 1fr;
    }

    ToggleSection .toggle-content {
        padding: 1;
        border: solid $surface-lighten-2;
        border-top: none;
        background: $surface;
    }

    ToggleSection .toggle-content.disabled {
        display: none;
    }
    """

    enabled: reactive[bool] = reactive(False)

    class Toggled(Message):
        """Posted when the section is enabled/disabled."""

        def __init__(self, section: "ToggleSection", enabled: bool) -> None:
            super().__init__()
            self.section = section
            self.enabled = enabled

        @property
        def control(self) -> "ToggleSection":
            return self.section

    def __init__(
        self,
        title: str = "Section",
        enabled: bool = False,
        description: str = "",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize the toggle section.

        Args:
            title: Header text.
            enabled: Initial enabled state.
            description: Optional description text.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._title = title
        self._description = description
        self.enabled = enabled
        self._content_widgets = []

    def compose(self) -> ComposeResult:
        """Compose the toggle section layout."""
        from textual.widgets import Switch

        with Container(classes="toggle-header"):
            with Container(classes="toggle-row"):
                yield Static(self._title, classes="toggle-title")
                yield Switch(value=self.enabled, id="toggle-switch")

            if self._description:
                yield Static(
                    self._description,
                    classes="description-text",
                )

        content_classes = (
            "toggle-content" if self.enabled else "toggle-content disabled"
        )
        yield Container(*self._content_widgets, classes=content_classes, id="content")

    def add_content(self, *widgets: Widget) -> None:
        """Add widgets to the section content.

        Args:
            widgets: Widgets to add.
        """
        self._content_widgets.extend(widgets)

    @on(Message)
    def handle_switch_change(self, event) -> None:
        """Handle switch toggle."""
        from textual.widgets import Switch

        if hasattr(event, "switch") and isinstance(event, Switch.Changed):
            self.enabled = event.value
            self._update_content()
            self.post_message(self.Toggled(self, self.enabled))

    def _update_content(self) -> None:
        """Update content visibility based on enabled state."""
        try:
            content = self.query_one("#content", Container)
            if self.enabled:
                content.remove_class("disabled")
            else:
                content.add_class("disabled")
        except Exception:
            pass

    def set_enabled(self, enabled: bool) -> None:
        """Programmatically set enabled state.

        Args:
            enabled: New enabled state.
        """
        from textual.widgets import Switch

        if enabled != self.enabled:
            self.enabled = enabled
            try:
                switch = self.query_one("#toggle-switch", Switch)
                switch.value = enabled
            except Exception:
                pass
            self._update_content()
