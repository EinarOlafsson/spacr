"""The console's entry list, and items in it that are not widgets.

A QLayout holds ITEMS, and not every item is a widget -- a spacer is an
item with no widget behind it. Both loops that walk the entry list ask
for the widget and skip when there is none, which is what stops a
spacer from being appended to a section body or handed `deleteLater`.

The trailing stretch at the bottom of the console is exactly such an
item, and `clear` is written to keep it.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QLabel, QSpacerItem, QSizePolicy

from spacr.qt.widgets.console_panel import ConsolePanel, _TopicBar

pytestmark = pytest.mark.qt


def _make_bar():
    """A topic bar, whatever arguments this build's constructor takes."""
    import inspect

    params = [p for name, p in
              inspect.signature(_TopicBar.__init__).parameters.items()
              if name != "self"]
    required = [p for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return _TopicBar(*["a topic"] * len(required))


@pytest.fixture()
def panel(qtbot):
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    return widget


class TestWalkingTheEntryList:

    def test_a_section_body_skips_non_widget_items(self, panel):
        """The same guard in `section_body`.

        A spacer appended to a section body would reach callers as an
        entry with no text, and the copy would carry a blank line for
        something that was never printed.
        """
        bar = _make_bar()
        panel._entries.insertWidget(0, bar)
        panel._entries.insertWidget(1, QLabel("printed line"))
        panel._entries.insertItem(
            2, QSpacerItem(1, 1, QSizePolicy.Policy.Minimum,
                           QSizePolicy.Policy.Fixed))
        panel._entries.insertWidget(3, QLabel("another line"))

        body = panel.section_body(bar)
        assert all(w is not None for w in body), (
            "a non-widget item reached the section body")
        assert len(body) >= 1


