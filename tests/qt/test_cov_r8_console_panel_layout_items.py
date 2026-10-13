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

    def test_clearing_removes_widgets_and_keeps_the_stretch(self, panel,
                                                            qtbot):
        panel._entries.insertWidget(0, QLabel("one"))
        panel._entries.insertWidget(1, QLabel("two"))
        before = panel._entries.count()
        assert before > 1

        panel.clear()
        assert panel._entries.count() == 1, (
            "clear did not leave exactly the trailing stretch")

    def test_a_spacer_among_the_entries_is_skipped_not_deleted(self, panel):
        """THE UNCOVERED ARC.

        `item.widget()` is None for a spacer. Calling `setParent` or
        `deleteLater` on that None would raise inside the teardown loop,
        so the guard skips it -- and the loop goes round to the next
        item, which is the arc.
        """
        panel._entries.insertWidget(0, QLabel("one"))
        panel._entries.insertItem(
            1, QSpacerItem(1, 1, QSizePolicy.Policy.Minimum,
                           QSizePolicy.Policy.Fixed))
        panel._entries.insertWidget(2, QLabel("two"))

        panel.clear()                      # must not raise
        assert panel._entries.count() == 1

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


class TestTheBoundedWalkToThePanel:
    """`_copy_section` climbs at most six parents looking for the panel.

    The bound matters: a widget re-parented into an unexpected tree
    would otherwise walk to the top of the application on every copy.
    """

    @pytest.mark.parametrize("intermediate_parents", [5, 6])
    def test_copy_checks_at_most_six_parents(
            self, panel, monkeypatch, intermediate_parents):
        """A panel at the sixth parent is reachable; the seventh is not."""
        from PySide6.QtWidgets import QApplication, QWidget

        requested = []

        def section_text(bar):
            requested.append(bar)
            return "this section"

        monkeypatch.setattr(panel, "section_text", section_text)
        parent = panel
        for _ in range(intermediate_parents):
            parent = QWidget(parent)
        bar = _make_bar()
        bar.setParent(parent)
        QApplication.clipboard().setText("previous contents")

        bar._copy_section()

        reachable = intermediate_parents == 5
        assert requested == ([bar] if reachable else [])
        assert QApplication.clipboard().text() == (
            "this section" if reachable else "previous contents")

    def test_a_bar_with_no_panel_above_it_copies_nothing(self, qtbot):
        """The live refusal: nothing to ask for a span.

        Asserted on the CLIPBOARD, not merely survived: "it did not
        raise" passes just as well against a version that copies the
        wrong thing, and copying the wrong thing silently is the failure
        worth catching here.
        """
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText("what was there before")

        bar = _make_bar()
        qtbot.addWidget(bar)

        bar._copy_section()                # must not raise

        assert QApplication.clipboard().text() == "what was there before", (
            "a bar with no panel above it put something on the clipboard")
