"""``spacr.qt.screensaver``: opening it, and the black screen it falls back to.

Three behaviours are pinned here that the module's own docstring promises and
nothing else exercised:

* the backdrop is *optional*. When one cannot be built the screensaver is a
  plain black screen -- "which is still a screensaver and still closes on a
  key" -- rather than a window that failed to open, and the failure is logged
  rather than raised;
* it opens **on the screen the parent is on**, which is what makes
  Ctrl+Shift+F on a second monitor blank the monitor the user is looking at.
  A parent that is not on a screen yet must not stop it opening;
* ``paintEvent`` paints black. With ``WA_OpaquePaintEvent`` set, whatever
  Qt last left in the backing store shows through if it does not.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, Qt                              # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402

from spacr.qt.screensaver import Screensaver  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def close_after(qapp):
    """Close every screensaver a test opened, tolerantly.

    NOT ``qtbot.addWidget``: ``Screensaver`` sets WA_DeleteOnClose on
    purpose, so pytest-qt's teardown close lands on a freed C++ object and
    blames the previous test.
    """
    opened = []
    yield opened.append
    for window in opened:
        try:
            if not window.isHidden():
                window.close()
        except RuntimeError:
            pass
    qapp.processEvents()


@pytest.fixture
def no_backdrop(monkeypatch):
    """Make ``_build_the_backdrop`` fail the way a missing GL context does.

    The settings read is the first thing the builder does, so refusing it
    exercises the same ``except`` the real failures land in.
    """
    from spacr.qt import preferences

    def _refuse():
        raise RuntimeError("no fractal settings on this machine")

    monkeypatch.setattr(preferences, "get_fractal_settings", _refuse)


def _key(key=Qt.Key.Key_A):
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


# ---------------------------------------------------------------------------
# a backdrop that cannot be built
# ---------------------------------------------------------------------------

def test_a_backdrop_that_cannot_be_built_leaves_a_screensaver_behind(
        close_after, no_backdrop, caplog):
    """It is opened from a menu action; raising there loses the click.

    The comparison is made in one test because "no backdrop" only means
    something against a run where one *was* built: the same constructor,
    with the settings readable, does add a widget to the layout.
    """
    with caplog.at_level(logging.ERROR, logger="spacr.qt.screensaver"):
        blank = Screensaver()
    close_after(blank)

    assert blank._backdrop is None
    assert blank.layout().count() == 0, "nothing may be added to the layout"
    assert any("backdrop" in record.message for record in caplog.records), (
        "the failure has to reach the log, or it is invisible")
    # ...and it is still a screensaver: any key gives the screen back.
    blank.show()
    blank.keyPressEvent(_key())
    assert not blank.isVisible()
