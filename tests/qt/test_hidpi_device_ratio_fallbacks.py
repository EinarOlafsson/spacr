"""``spacr.qt.hidpi.device_ratio`` when the display will not answer properly.

The ratio is asked of the target, then its screen, then its window, then the
primary screen, and is 1.0 when nothing answers. One table walks every way a
source can fail to answer (junk numbers, raising accessors, no screen yet, no
primary screen, no application) and records which accessors were consulted,
because the order of the search is part of the contract.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import hidpi  # noqa: E402
from spacr.qt.hidpi import MAX_RATIO, device_ratio  # noqa: E402

pytestmark = pytest.mark.qt


class _Ratios:
    """Answers both ratio accessors with fixed values and records the asks."""

    def __init__(self, ratio_f, ratio=None, *, raise_f=False):
        self._ratio_f = ratio_f
        self._ratio = ratio if ratio is not None else ratio_f
        self._raise_f = raise_f
        self.asked = []

    def devicePixelRatioF(self):  # noqa: N802 - Qt spelling
        self.asked.append("devicePixelRatioF")
        if self._raise_f:
            raise RuntimeError("Internal C++ object already deleted")
        return self._ratio_f

    def devicePixelRatio(self):  # noqa: N802 - Qt spelling
        self.asked.append("devicePixelRatio")
        return self._ratio


class _Placed:
    """A non-Qt widget stand-in with ``screen()`` and ``window()``."""

    def __init__(self, screen, window):
        self._screen = screen
        self._window = window

    def screen(self):
        if isinstance(self._screen, Exception):
            raise self._screen
        return self._screen

    def window(self):
        return self._window


class _App:
    def __init__(self, screen):
        self._screen = screen

    def primaryScreen(self):  # noqa: N802 - Qt spelling
        if isinstance(self._screen, Exception):
            raise self._screen
        return self._screen


class _Gui:
    """Stands in for ``QGuiApplication`` with a chosen instance."""

    def __init__(self, app):
        self._app = app

    def instance(self):
        return self._app


def test_device_ratio_searches_target_screen_window_primary_then_one(
        monkeypatch, subtests, qapp):
    """Exact ratio and the accessors consulted, for every way a source can fail."""
    assert MAX_RATIO == 8.0

    cases = []
    for junk in (0.0, -1.0, float("nan"), float("inf")):
        cases.append((f"a first answer of {junk!r} is not believed",
                      lambda junk=junk: _Ratios(junk, 2.0), 2.0,
                      ["devicePixelRatioF", "devicePixelRatio"]))
    cases += [
        ("a real first answer ends the search", lambda: _Ratios(1.5, 4.0),
         1.5, ["devicePixelRatioF"]),
        ("an impossible ratio is clamped", lambda: _Ratios(40.0), MAX_RATIO,
         ["devicePixelRatioF"]),
        ("a raising first accessor is stepped over",
         lambda: _Ratios(0.0, 3.0, raise_f=True), 3.0,
         ["devicePixelRatioF", "devicePixelRatio"]),
    ]
    for case, make, expected, asked in cases:
        with subtests.test(msg=case):
            target = make()
            assert device_ratio(target) == expected
            assert target.asked == asked

    placed = [
        ("no screen yet: the window answers", None, _Ratios(2.0), 2.0,
         ["devicePixelRatioF"]),
        ("a screen without a ratio: the window answers", _Ratios(0.0, 0.0),
         _Ratios(2.0), 2.0, ["devicePixelRatioF"]),
        ("a raising screen falls through to the window",
         RuntimeError("Internal C++ object already deleted"), _Ratios(2.5),
         2.5, ["devicePixelRatioF"]),
        ("a screen that answers keeps the window out", _Ratios(3.0),
         _Ratios(2.0), 3.0, []),
    ]
    for case, screen, window, expected, window_asked in placed:
        with subtests.test(msg=case):
            assert device_ratio(_Placed(screen, window)) == expected
            assert window.asked == window_asked

    class _NoRatioAtAll:
        """A target that has never heard of a device pixel ratio."""

    primary = [
        ("no primary screen", _Gui(_App(None)), None, 1.0),
        ("no primary screen, a ratio-less target", _Gui(_App(None)),
         _NoRatioAtAll(), 1.0),
        ("a primary screen with a ratio", _Gui(_App(_Ratios(2.0))), None,
         2.0),
        ("a primary screen with no usable ratio",
         _Gui(_App(_Ratios(0.0, 0.0))), _NoRatioAtAll(), 1.0),
        ("a primary screen that raises",
         _Gui(_App(RuntimeError("platform plugin not ready"))), None, 1.0),
        ("no application at all", _Gui(None), None, 1.0),
    ]
    for case, gui, target, expected in primary:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(hidpi, "QGuiApplication", gui)
            assert device_ratio(target) == expected
