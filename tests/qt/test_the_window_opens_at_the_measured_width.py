"""A fresh window opens wide enough for the modules it holds (359).

The report: "every module must initially open wide enough that the right
side of its settings is not cut off". The window has always opened at its
own 1200 px minimum, and the measured matrix says most modules need more
than that before their settings column stops holding more than it can
show.

`open_at_the_measured_width` is the wire. These tests are about the two
ways it could do harm -- shrinking a window, or opening one off the edge
of the display -- and about the case where there is no artifact at all,
which is every wheel built before the generator ran.
"""

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QWidget

from spacr.qt import app as qt_app


class _Screen:
    """A display of a stated size."""

    def __init__(self, width, height):
        self._rect = QRect(0, 0, width, height)

    def availableGeometry(self):                     # noqa: N802 (Qt name)
        return self._rect


@pytest.fixture
def window(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.resize(1200, 720)
    return widget


def _policy(monkeypatch, width, screen, window, scale=1.0):
    """Point the wire at a stated policy, screen and font scale."""
    monkeypatch.setattr(qt_app, "QApplication", type(
        "App", (), {"primaryScreen": staticmethod(lambda: screen)}))
    monkeypatch.setattr(window, "screen", lambda: screen)
    monkeypatch.setattr("spacr.qt.layout_policy.recommended_window_size",
                        lambda available, font_scale: (width, 850))
    monkeypatch.setattr("spacr.qt.preferences.get_font_scale", lambda: scale)


class TestItGrowsAndNeverShrinks:

    def test_a_wider_requirement_widens_the_window(self, window, monkeypatch):
        _policy(monkeypatch, 2100, _Screen(2560, 1440), window)
        assert qt_app.open_at_the_measured_width(window) is True
        assert window.width() == 2100

    def test_a_narrower_requirement_is_ignored(self, window, monkeypatch):
        """Shrinking on a bad number is not bounded by anything.

        The window does not maximise precisely because a remote session's
        "available geometry" is often a stub, and a policy allowed to
        shrink would take that claim seriously.
        """
        _policy(monkeypatch, 800, _Screen(2560, 1440), window)
        assert qt_app.open_at_the_measured_width(window) is False
        assert window.width() == 1200

    def test_the_same_width_is_not_a_resize(self, window, monkeypatch):
        _policy(monkeypatch, 1200, _Screen(2560, 1440), window)
        assert qt_app.open_at_the_measured_width(window) is False
        assert window.width() == 1200

    def test_the_height_is_left_alone(self, window, monkeypatch):
        """This item is about the RIGHT side being cut off."""
        _policy(monkeypatch, 2100, _Screen(2560, 1440), window)
        qt_app.open_at_the_measured_width(window)
        assert window.height() == 720


class TestItNeverOpensPastTheEdge:

    def test_it_is_clamped_to_the_display(self, window, monkeypatch):
        _policy(monkeypatch, 2100, _Screen(1600, 900), window)
        assert qt_app.open_at_the_measured_width(window) is True
        assert window.width() == 1600

    def test_a_display_narrower_than_the_window_changes_nothing(
            self, window, monkeypatch):
        _policy(monkeypatch, 2100, _Screen(1024, 768), window)
        qt_app.open_at_the_measured_width(window)
        assert window.width() <= 1200, (
            "the window was widened past a display that cannot hold it")


class TestItIsNeverWorthFailingALaunchOver:

    def test_no_screen_is_not_an_error(self, window, monkeypatch):
        monkeypatch.setattr(qt_app, "QApplication", type(
            "App", (), {"primaryScreen": staticmethod(lambda: None)}))
        monkeypatch.setattr(window, "screen", lambda: None)
        assert qt_app.open_at_the_measured_width(window) is False

    def test_a_policy_that_throws_is_swallowed(self, window, monkeypatch):
        def _explode(*_a, **_k):
            raise RuntimeError("no policy here")

        monkeypatch.setattr(qt_app, "QApplication", type(
            "App", (), {"primaryScreen": staticmethod(
                lambda: _Screen(2560, 1440))}))
        monkeypatch.setattr("spacr.qt.layout_policy.recommended_window_size",
                            _explode)
        assert qt_app.open_at_the_measured_width(window) is False
        assert window.width() == 1200

    def test_without_an_artifact_the_window_is_untouched(self, window,
                                                         monkeypatch):
        """Every wheel built before the generator ran is this case."""
        monkeypatch.setattr("spacr.qt.layout_policy.read_policy",
                            lambda refresh=False: {})
        monkeypatch.setattr(qt_app, "QApplication", type(
            "App", (), {"primaryScreen": staticmethod(
                lambda: _Screen(2560, 1440))}))
        monkeypatch.setattr(window, "screen", lambda: _Screen(2560, 1440))
        assert qt_app.open_at_the_measured_width(window) is False
        assert window.width() == 1200
