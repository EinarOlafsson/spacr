"""The theme's held window sheet, its late widget blocks, and its colour arithmetic.

* ``_one_sheet_for`` holds a window's sheet for a decorator's block and
  releases it once, at the outer exit, even when the block raised, the
  window died meanwhile, or the thing held is not a widget at all;
* ``ensure_widget_qss_applied`` puts the late-registered blocks on a screen
  root as one replaceable suffix, and is a no-op with no root, no
  application or no spaCR sheet in force;
* ``_channels`` parses strictly unless a paint-time caller offers a fallback;
* ``_hue_rgb`` is the HSV V=1 plane, and ``_hue_ink`` returns the most
  coloured candidate its luminance band admits, or the plain hue shift when
  the band admits none.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt import theme  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# The held window sheet
# ---------------------------------------------------------------------------

class _DiesDuringTheHold:
    """A window deleted while its decorator held its sheet."""

    def __init__(self):
        self.properties = {}

    def property(self, name):
        return self.properties.get(name)

    def setProperty(self, name, value):          # noqa: N802 - Qt spelling
        if value is None:
            raise RuntimeError("Internal C++ object already deleted.")
        self.properties[name] = value


def test_one_sheet_for_holds_a_window_once_and_always_releases_it(
        subtests, qtbot, monkeypatch):
    """Nested holds release at the outer exit; raising blocks, dead windows and non-widgets are safe."""
    with subtests.test(msg="a nested hold releases only at the outer exit"):
        window = QWidget()
        qtbot.addWidget(window)
        with theme._one_sheet_for(window) as held:
            assert held is window
            assert window.property(theme._SHEET_HELD) is True
            with theme._one_sheet_for(window) as inner:
                assert inner is window
            assert window.property(theme._SHEET_HELD) is True
        assert window.property(theme._SHEET_HELD) is None

    with subtests.test(msg="a block that raises still releases the window"):
        window = QWidget()
        qtbot.addWidget(window)
        with pytest.raises(RuntimeError, match="decorator failed"):
            with theme._one_sheet_for(window):
                raise RuntimeError("decorator failed")
        assert window.property(theme._SHEET_HELD) is None

    with subtests.test(msg="an owed polish with no window sheet repolishes the tagged"), \
            monkeypatch.context() as patch:
        window = QWidget()
        qtbot.addWidget(window)
        repolished = []
        patch.setattr(theme, "window_stylesheet", lambda *a: None)
        patch.setattr(theme, "_repolish_the_tagged", repolished.append)
        with theme._one_sheet_for(window):
            window.setProperty(theme._SHEET_OWES_A_POLISH, True)
        assert repolished == [window]
        assert window.property(theme._SHEET_OWES_A_POLISH) is None
        with theme._one_sheet_for(window):
            pass
        assert repolished == [window]

    with subtests.test(msg="something that is not a widget passes through"):
        thing = object()
        with theme._one_sheet_for(thing) as held:
            assert held is thing

    with subtests.test(msg="a window deleted during its hold does not raise"):
        window = _DiesDuringTheHold()
        with theme._one_sheet_for(window) as held:
            assert held is window
            assert window.properties[theme._SHEET_HELD] is True
        assert window.properties == {theme._SHEET_HELD: True}


# ---------------------------------------------------------------------------
# Late widget blocks on a screen root
# ---------------------------------------------------------------------------

def test_ensure_widget_qss_applied_installs_one_replaceable_suffix(
        subtests, qtbot, monkeypatch):
    """No root/app/sheet is a no-op; a root gets one suffix, replaced rather than stacked."""
    start, end = theme._LOCAL_WIDGET_QSS_START, theme._LOCAL_WIDGET_QSS_END
    fake_app = SimpleNamespace(styleSheet=lambda: "QWidget { color: red; }")
    setattr(fake_app, theme._WIDGET_QSS_CONTEXT_ATTRIBUTE, ("dark", 1.0, None))

    with subtests.test(msg="no root is nothing to restyle"):
        assert theme.ensure_widget_qss_applied("SomeBlock") is False
        assert theme.ensure_widget_qss_applied("SomeBlock", root=None) is False

    root = QWidget()
    qtbot.addWidget(root)

    with subtests.test(msg="no application is nothing to restyle"), \
            monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", staticmethod(lambda: None))
        assert theme.ensure_widget_qss_applied("SomeBlock", root=root) is False
        assert root.styleSheet() == ""

    with subtests.test(msg="no spaCR sheet in force leaves the root alone"), \
            monkeypatch.context() as patch:
        bare = SimpleNamespace(styleSheet=lambda: "")
        patch.setattr(QApplication, "instance", staticmethod(lambda: bare))
        patch.setattr(theme, "window_stylesheet", lambda *a: None)
        assert theme.ensure_widget_qss_applied(root=root) is False
        assert root.styleSheet() == ""

    with subtests.test(msg="a root gets one suffix, and a second call changes nothing"), \
            monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", staticmethod(lambda: fake_app))
        root.setStyleSheet("QLabel { margin: 1px; }")
        assert theme.ensure_widget_qss_applied(root=root) is True
        sheet = root.styleSheet()
        suffix = getattr(root, theme._LOCAL_WIDGET_QSS_ATTRIBUTE)
        assert sheet == "QLabel { margin: 1px; }" + suffix
        assert suffix.startswith(f"\n{start}\n") and suffix.endswith(end)
        assert theme._WIDGET_QSS_MARKER.format(name="ImageScatter") in suffix
        assert theme.ensure_widget_qss_applied(root=root) is False
        assert root.styleSheet() == sheet

    with subtests.test(msg="a changed suffix replaces the old one"), \
            monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", staticmethod(lambda: fake_app))
        setattr(root, theme._LOCAL_WIDGET_QSS_ATTRIBUTE, "\n/* stale */")
        root.setStyleSheet("QLabel { margin: 1px; }\n/* stale */")
        assert theme.ensure_widget_qss_applied(root=root) is True
        assert root.styleSheet() == sheet
        assert root.styleSheet().count(start) == 1

    with subtests.test(msg="a root whose C++ half is gone is reported, not raised"), \
            monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", staticmethod(lambda: fake_app))

        class _Gone:
            def styleSheet(self):                 # noqa: N802 - Qt spelling
                raise RuntimeError("Internal C++ object already deleted.")

        assert theme.ensure_widget_qss_applied(root=_Gone()) is False


# ---------------------------------------------------------------------------
# Colour arithmetic
# ---------------------------------------------------------------------------

def test_channels_parses_strictly_unless_given_a_fallback(subtests):
    """#rgb and #rrggbb parse; anything else raises, or answers the fallback."""
    for text, expected in [("#123456", (0x12, 0x34, 0x56)),
                           ("#abc", (0xaa, 0xbb, 0xcc)),
                           (" #ABCDEF ", (0xab, 0xcd, 0xef)),
                           ("000000", (0, 0, 0))]:
        with subtests.test(msg=f"parses {text!r}"):
            assert theme._channels(text) == expected
            assert theme._channels(text, (1, 2, 3)) == expected
    for text in ("not-a-colour", "#12345", "#ggg", ""):
        with subtests.test(msg=f"refuses {text!r}"):
            with pytest.raises(ValueError,
                               match=f"^not a #rrggbb colour: {text!r}$"):
                theme._channels(text)
            assert theme._channels(text, (1, 2, 3)) == (1, 2, 3)


def _value_ramp(hue, saturation=1.0):
    """The hue's own colour, darkened: the first ramp ``_hue_ink`` scans."""
    base = theme._hue_rgb(hue, saturation)
    return [tuple(int(round(channel * level)) for channel in base)
            for level in range(256)]


def _tint_ramp(hue):
    """The hue mixed toward white: the second ramp ``_hue_ink`` scans."""
    base = theme._hue_rgb(hue)
    return [tuple(int(round(255.0 * (1.0 - step / 255.0
                                     + step / 255.0 * channel)))
                  for channel in base)
            for step in range(256)]


def _chroma(rgb):
    return max(rgb) - min(rgb)


def _reachable_hues():
    """Every hue the dressing can ask an ink role for."""
    return sorted({(seat + drift) % 360.0
                   for seat in theme.SPACEOUT_HUES.values()
                   for drift in theme._drift_grid()})


def test_hue_rgb_is_the_hsv_value_one_plane(subtests):
    """Exact corners and mixes; at full saturation the value ramp's chroma is its level."""
    for hue, saturation, expected in [
            (0.0, 1.0, (1.0, 0.0, 0.0)), (30.0, 1.0, (1.0, 0.5, 0.0)),
            (60.0, 1.0, (1.0, 1.0, 0.0)), (120.0, 1.0, (0.0, 1.0, 0.0)),
            (180.0, 1.0, (0.0, 1.0, 1.0)), (240.0, 1.0, (0.0, 0.0, 1.0)),
            (300.0, 1.0, (1.0, 0.0, 1.0)), (360.0, 1.0, (1.0, 0.0, 0.0)),
            (-60.0, 1.0, (1.0, 0.0, 1.0)), (0.0, 0.0, (1.0, 1.0, 1.0)),
            (0.0, 0.5, (1.0, 0.5, 0.5)), (0.0, 2.0, (1.0, 0.0, 0.0)),
            (0.0, -1.0, (1.0, 1.0, 1.0))]:
        with subtests.test(msg=f"hue {hue} saturation {saturation}"):
            assert theme._hue_rgb(hue, saturation) == pytest.approx(expected)

    with subtests.test(msg="every reachable hue's value ramp rises one per level"):
        for hue in _reachable_hues():
            base = theme._hue_rgb(hue)
            assert (min(base), max(base)) == (0.0, 1.0), hue
            assert [_chroma(rgb) for rgb in _value_ramp(hue)] == list(
                range(256)), hue

    with subtests.test(msg="the softened surface plane plateaus on every hue"):
        softened = theme.SPACEOUT_SATURATION["bg"]
        assert [hue for hue in _reachable_hues()
                if [_chroma(rgb) for rgb in _value_ramp(hue, softened)]
                != list(range(256))] == _reachable_hues()


def test_hue_ink_spends_its_band_on_colour(subtests):
    """The ink is the argmax of chroma in its band; an empty band gives the plain hue shift."""
    bands = {role: theme._ink_band("dark", role)
             for role in theme.SPACEOUT_INK_ROLES}
    assert None not in bands.values(), bands

    for role, (low, high) in bands.items():
        fallback = theme.DARK_PALETTE[role]
        for hue in (30.0, 210.0, 300.0):
            with subtests.test(msg=f"{role} at hue {hue}"):
                ink = theme._hue_ink(hue, low, high, fallback)
                plain = theme._hue_shift(fallback, hue)
                admitted = [rgb for rgb in _value_ramp(hue) + _tint_ramp(hue)
                            if low <= theme._rgb_luminance(rgb) <= high]
                assert admitted
                assert low <= theme.relative_luminance(ink) <= high
                assert _chroma(theme._channels(ink)) == max(
                    _chroma(rgb) for rgb in admitted)
                assert _chroma(theme._channels(ink)) > _chroma(
                    theme._channels(plain))

    with subtests.test(msg="a band no 8-bit colour lands in keeps the plain shift"):
        hue, fallback = 210.0, theme.DARK_PALETTE["fg_muted"]
        lumas = sorted(theme._rgb_luminance(rgb)
                       for rgb in _value_ramp(hue) + _tint_ramp(hue))
        gap, below = max((lumas[i + 1] - lumas[i], lumas[i])
                         for i in range(len(lumas) - 1))
        low, high = below + gap * 0.25, below + gap * 0.75
        assert [luma for luma in lumas if low <= luma <= high] == []
        assert theme._hue_ink(hue, low, high, fallback) == \
            theme._hue_shift(fallback, hue)
        wide_low, wide_high = bands["fg_muted"]
        coloured = theme._hue_ink(hue, wide_low, wide_high, fallback)
        assert coloured != theme._hue_shift(fallback, hue)
