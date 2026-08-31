"""Two small widgets, and the tails neither had.

`InfoLink` is the teal documentation dot. Its `url()` and `set_url()`
were the only two statements in the file nothing reached -- the class was
always built and clicked, never asked what it points at or told to point
somewhere else.

`Flash` is the brief highlight a widget shows after a copy. Its
uncovered pair is the `RuntimeError` guard in `_end`: a screen torn down
between the trigger and the timeout is ordinary, not an error, and
shiboken raises on a deleted C++ object. That branch is exactly the kind
that is only ever exercised by the failure it exists for.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.flash import Flash
from spacr.qt.widgets.info_link import InfoLink

pytestmark = pytest.mark.qt


class TestTheDocumentationDot:

    def test_it_reports_the_url_it_was_built_with(self, qtbot):
        dot = InfoLink("https://spacr.invalid/api/mask")
        qtbot.addWidget(dot)
        assert dot.url() == "https://spacr.invalid/api/mask"

    def test_the_url_is_stored_as_text_whatever_it_arrives_as(self, qtbot):
        """`str(url)` -- a QUrl or a Path must not be kept as an object.

        `open_documentation` wraps it in `QUrl(...)`, which wants a string.
        """
        from pathlib import Path

        dot = InfoLink(Path("/docs/api/mask.html"))
        qtbot.addWidget(dot)
        assert dot.url() == "/docs/api/mask.html"
        assert isinstance(dot.url(), str)

    def test_the_destination_can_be_changed_without_rebuilding(self, qtbot):
        """Which is the whole reason `set_url` exists rather than a rebuild."""
        dot = InfoLink("https://spacr.invalid/api/mask")
        qtbot.addWidget(dot)
        before = dot.objectName()
        dot.set_url("https://spacr.invalid/api/measure")
        assert dot.url() == "https://spacr.invalid/api/measure"
        assert dot.objectName() == before == "InfoLink"

    def test_a_changed_url_is_the_one_that_gets_opened(self, qtbot,
                                                       monkeypatch):
        """Storing it and opening it must not drift apart."""
        from PySide6.QtGui import QDesktopServices

        opened = []
        monkeypatch.setattr(QDesktopServices, "openUrl",
                            staticmethod(lambda u: opened.append(u.toString())))
        dot = InfoLink("https://spacr.invalid/api/mask")
        qtbot.addWidget(dot)
        dot.set_url("https://spacr.invalid/api/measure")
        dot.open_documentation()
        assert opened == ["https://spacr.invalid/api/measure"]


class TestTheFlashOutlivingItsWidget:

    def test_a_flash_ends_by_repainting_a_live_widget(self, qtbot):
        """The ordinary path, so the guard below is visibly a guard."""
        from PySide6.QtWidgets import QWidget

        widget = QWidget()
        qtbot.addWidget(widget)
        flash = Flash(widget, duration_ms=1)
        flash.trigger()
        assert flash.active is True
        flash._end()
        assert flash.active is False

    def test_a_widget_destroyed_mid_flash_is_not_an_error(self, qtbot):
        """THE UNCOVERED PAIR.

        A screen torn down between the trigger and the timeout is
        ordinary. shiboken raises RuntimeError on a deleted C++ object,
        and there is nothing to repaint by then either way -- so `_end`
        has to finish rather than raise out of a singleShot, where
        nobody can catch it.
        """
        from PySide6.QtWidgets import QWidget

        widget = QWidget()
        qtbot.addWidget(widget)
        flash = Flash(widget, duration_ms=1)
        flash.trigger()

        class _Gone:
            def update(self):
                raise RuntimeError(
                    "Internal C++ object (QWidget) already deleted.")

        flash._widget = _Gone()
        flash._end()                       # must not raise
        assert flash.active is False, (
            "the flash stayed on after the widget it marks had gone")

    def test_only_a_runtime_error_is_swallowed(self, qtbot):
        """A guard that caught everything would hide real paint bugs."""
        from PySide6.QtWidgets import QWidget

        widget = QWidget()
        qtbot.addWidget(widget)
        flash = Flash(widget, duration_ms=1)
        flash.trigger()

        class _Broken:
            def update(self):
                raise ValueError("a real bug in a paint handler")

        flash._widget = _Broken()
        with pytest.raises(ValueError):
            flash._end()
        assert flash.active is False, (
            "the active flag must be cleared before the repaint is attempted")
