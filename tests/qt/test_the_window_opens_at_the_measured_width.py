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
from PySide6.QtWidgets import QMainWindow

from spacr.qt import app as qt_app


class _Screen:
    """A display of a stated size."""

    def __init__(self, width, height):
        self._rect = QRect(0, 0, width, height)

    def availableGeometry(self):                     # noqa: N802 (Qt name)
        return self._rect


@pytest.fixture
def window(qtbot):
    widget = QMainWindow()
    qtbot.addWidget(widget)
    widget.resize(1200, 720)
    return widget


def _policy(monkeypatch, width, screen, window, scale=1.0):
    """Point the wire at a stated policy, screen and font scale."""
    monkeypatch.setattr("spacr.qt.hidpi.screen_for_widget",
                        lambda widget=None: screen)
    monkeypatch.setattr("spacr.qt._layout_policy.recommended_window_size",
                        lambda available, font_scale: (width, 850))
    monkeypatch.setattr("spacr.qt.preferences.get_font_scale", lambda: scale)


@pytest.mark.parametrize(
    "width, display_width, display_height, changed, expected_width", [
        pytest.param(2100, 2560, 1440, True, 2100,
                     id="test_a_wider_requirement_widens_the_window"),
        pytest.param(800, 2560, 1440, False, 1200,
                     id="test_a_narrower_requirement_is_ignored"),
        pytest.param(1200, 2560, 1440, False, 1200,
                     id="test_the_same_width_is_not_a_resize"),
        pytest.param(2100, 1600, 900, True, 1600,
                     id="test_it_is_clamped_to_the_display"),
    ],
)
def test_measured_width_boundaries(window, monkeypatch, width, display_width,
                                   display_height, changed, expected_width):
    """Preserve the wider, narrower, unchanged, and display-clamped cases."""
    _policy(monkeypatch, width, _Screen(display_width, display_height), window)
    assert qt_app._open_at_the_measured_width(window) is changed
    assert window.width() == expected_width


class TestItGrowsAndNeverShrinks:

    def test_the_height_is_left_alone(self, window, monkeypatch):
        """This item is about the RIGHT side being cut off."""
        _policy(monkeypatch, 2100, _Screen(2560, 1440), window)
        qt_app._open_at_the_measured_width(window)
        assert window.height() == 720


class TestItNeverOpensPastTheEdge:

    def test_a_display_narrower_than_the_window_changes_nothing(
            self, window, monkeypatch):
        _policy(monkeypatch, 2100, _Screen(1024, 768), window)
        qt_app._open_at_the_measured_width(window)
        assert window.width() <= 1200, (
            "the window was widened past a display that cannot hold it")


class TestItIsNeverWorthFailingALaunchOver:

    def test_no_screen_is_not_an_error(self, window, monkeypatch):
        monkeypatch.setattr("spacr.qt.hidpi.screen_for_widget",
                            lambda widget=None: None)
        assert qt_app._open_at_the_measured_width(window) is False

    def test_a_policy_that_throws_is_swallowed(self, window, monkeypatch):
        def _explode(*_a, **_k):
            raise RuntimeError("no policy here")

        monkeypatch.setattr("spacr.qt.hidpi.screen_for_widget",
                            lambda widget=None: _Screen(2560, 1440))
        monkeypatch.setattr("spacr.qt._layout_policy.recommended_window_size",
                            _explode)
        assert qt_app._open_at_the_measured_width(window) is False
        assert window.width() == 1200

    def test_without_an_artifact_the_window_is_untouched(self, window,
                                                         monkeypatch):
        """Every wheel built before the generator ran is this case."""
        monkeypatch.setattr("spacr.qt._layout_policy.read_policy",
                            lambda refresh=False: {})
        monkeypatch.setattr("spacr.qt.hidpi.screen_for_widget",
                            lambda widget=None: _Screen(2560, 1440))
        assert qt_app._open_at_the_measured_width(window) is False
        assert window.width() == 1200
