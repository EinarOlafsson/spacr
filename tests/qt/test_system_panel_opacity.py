"""The System panel's four resource bars, measured against the backdrop.

A module screen's System card carries a CPU, a RAM, a GPU and a VRAM bar.
They must all be *equally* see-through, because they are all the same widget
in the same card and the page-opacity preference is one slider, not four.

Three of them were not. The application sheet gives ``QProgressBar#UsageBar``
a ``surface_alt`` fill at page opacity, and the card the bars sit in is
already ``surface_alt`` at page opacity — so each track laid a second copy of
the same translucent grey over the first and read as a band the slider could
not thin. The CPU bar escaped it by accident: it is wrapped in a widget whose
unqualified ``background: transparent`` reaches the bar, because in Qt a style
sheet set on an ANCESTOR beats the application sheet irrespective of selector
specificity. RAM, GPU and VRAM go straight into the card body, whose sheet is
qualified (``QWidget#CardBody``), so nothing cancelled their fill.

**The method matters more than the numbers.** Sampling a colour cannot tell
"opaque black" from "a dark part of the animation", so everything here is
measured by rendering the page twice over different backdrops and diffing the
pixels. Two backdrops are used, for two different jobs:

``a black render and a white render``
    solves ``P = a·B + (1-a)·F`` for the transmitted alpha at every pixel, so
    a region that is *half* as see-through as the card around it is caught.
    This is the one that fails on the bug: the tracks measured 0.49 against
    the card's 0.70 at a requested 30 %.

``the real ambient animation, on and then off``
    the plain form of the same idea, and the one that catches a region that
    has gone fully opaque. It cannot see a doubled alpha — a half-transparent
    slab still changes when the animation behind it changes — which is
    exactly why the first measurement exists as well.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                               QWidget)

from spacr.qt import preferences as prefs


#: The four bars, and the attribute each is held on.
BARS = (("CPU", "_usage_cpu"), ("RAM", "_usage_ram"),
        ("GPU", "_usage_gpu"), ("VRAM", "_usage_vram"))

#: A page opacity well below 100 %, so there is something to see through.
OPACITY = 0.3

#: How far a bar may sit from the CPU bar before it is a different surface.
#: The bug's gap is 0.21 — two orders of magnitude outside this.
TOLERANCE = 0.02


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write to the developer's real preferences.

    ``preferences._settings()`` builds ``QSettings(_ORG, _APP)``, which
    resolves to the NATIVE location whatever ``setPath`` says. Replacing the
    accessor is the only isolation that holds; the assertion refuses to run
    if it ever stops working.
    """
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def app_theme_restored(qt_theme_applied):
    """Undo what this file does to the session-scoped QApplication.

    ``apply_preferences_to_app`` re-palettes and re-stylesheets the whole
    application. Leaving it at 30 % opacity would take out every later test
    that measures a pixel.
    """
    yield
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


def _screen(qtbot, opacity, *, ambient=True):
    """Build a module screen at ``opacity``, in the window it really lives in.

    The screen goes inside a QMainWindow's stack because that is where the
    application stylesheet and palette actually reach it — a bare, parentless
    screen is styled differently, and a pixel test on one proves nothing.
    """
    from spacr.qt.screens.app_screen import AppScreen

    prefs.set_theme("dark")
    prefs.set_ambient_enabled(ambient)
    prefs.set_pane_opacity(opacity)
    prefs.apply_preferences_to_app(QApplication.instance())

    screen = AppScreen("mask")
    window = QMainWindow()
    stack = QStackedWidget()
    window.setCentralWidget(stack)
    stack.addWidget(screen)
    qtbot.addWidget(window)
    window.resize(1500, 1000)
    window.show()
    QApplication.processEvents()

    # The bars poll psutil / GPUtil on a timer. Freeze them at 0 so every
    # track is entirely unfilled: the chunk is opaque on purpose (it is the
    # reading), and a live value would move the boundary between renders.
    timer = getattr(screen, "_usage_timer", None)
    if timer is not None:
        timer.stop()
    for _, attr in BARS:
        getattr(screen, attr).set_value(0)
    QApplication.processEvents()
    # Keep the window alive: qtbot only weak-references it, and a collected
    # window deletes the C++ half of the screen under the test's feet.
    return window, screen


def _rect(screen, widget) -> QRect:
    top_left = widget.mapTo(screen, QPoint(0, 0))
    return QRect(top_left.x(), top_left.y(), widget.width(), widget.height())


def _regions(screen) -> dict:
    """The four bar tracks, plus a strip of bare card surface as reference."""
    out = {name: _rect(screen, getattr(screen, attr)._bar)
           for name, attr in BARS}
    body = getattr(screen, "_usage_ram").parentWidget()
    body_rect = _rect(screen, body)
    ram = out["RAM"]
    # A band of the card body between the title divider and the first row:
    # card surface with no child widget on it.
    out["card"] = QRect(ram.left(), body_rect.top() + 1, ram.width(), 3)
    return out


def _transmission(screen):
    """Per-pixel ``alpha`` of everything painted over the backdrop.

    Renders the screen over solid black and over solid white and solves
    ``P = a·B + (1-a)·F``. ``1.0`` means the backdrop reaches the eye
    untouched; ``0.0`` means something fully opaque is in front of it.
    """
    ambient = getattr(screen, "_ambient", None)
    if ambient is not None:
        # The sweep that makes the containers see-through has already run as
        # part of installing it; swap the animation itself for a flat colour
        # so the two renders differ by the backdrop and nothing else.
        ambient.hide()
    else:
        # No animation available in this environment. Put the screen in the
        # state having one would have put it in, or the containers are opaque
        # for a reason that has nothing to do with the bars.
        screen._clear_page_surfaces()

    backdrop = QWidget(screen)
    backdrop.setObjectName("BackdropProbe")
    backdrop.setGeometry(0, 0, screen.width(), screen.height())
    backdrop.lower()
    backdrop.show()

    def render(colour):
        backdrop.setStyleSheet(
            f"QWidget#BackdropProbe {{ background: {colour}; }}")
        backdrop.lower()
        QApplication.processEvents()
        return screen.grab().toImage()

    dark, light = render("#000000"), render("#ffffff")

    def alpha(x, y):
        a, b = QColor(dark.pixel(x, y)), QColor(light.pixel(x, y))
        return ((b.red() - a.red()) + (b.green() - a.green())
                + (b.blue() - a.blue())) / 765.0

    return alpha


def _mean(alpha, rect) -> float:
    values = [alpha(x, y)
              for y in range(rect.top(), rect.bottom() + 1)
              for x in range(rect.left(), rect.right() + 1)]
    assert values, f"empty measurement region {rect}"
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# The measurement itself has to be able to fail
# ---------------------------------------------------------------------------

def test_the_probe_can_tell_an_opaque_card_from_a_thinned_one(
        qtbot, app_theme_restored):
    """Guards the guard.

    A backdrop diff that reads the same number whatever the slider says is
    measuring nothing, and every assertion built on it is decoration. At
    100 % the card must swallow the backdrop; at 30 % most of it must arrive.
    """
    _win, screen = _screen(qtbot, 1.0)
    opaque = _mean(_transmission(screen), _regions(screen)["card"])

    _win2, screen2 = _screen(qtbot, OPACITY)
    thinned = _mean(_transmission(screen2), _regions(screen2)["card"])

    assert opaque < 0.05, (
        f"the System card still passes {opaque:.2f} of the backdrop at 100 % "
        "page opacity, so this file's probe is not measuring opacity")
    assert thinned > 0.5, (
        f"the System card passes only {thinned:.2f} of the backdrop at "
        f"{OPACITY:.0%}, so the preference is not reaching it at all")


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

def test_every_bar_track_is_as_see_through_as_the_cpu_one(
        qtbot, app_theme_restored):
    """RAM, GPU and VRAM must pass the backdrop exactly like CPU does.

    Measured before the fix, at a requested 30 %::

        card 0.702   CPU 0.702   RAM 0.494   GPU 0.494   VRAM 0.494

    — the three tracks were painting a second translucent grey over a card
    that already had one, so they sat at 1-(1-a)² instead of 1-a and no
    position of the slider could line them up with the CPU bar beside them.
    """
    _win, screen = _screen(qtbot, OPACITY)
    alpha = _transmission(screen)
    regions = _regions(screen)
    measured = {name: _mean(alpha, rect) for name, rect in regions.items()}

    cpu = measured["CPU"]
    off = {name: value for name, value in measured.items()
           if name in ("RAM", "GPU", "VRAM")
           and abs(value - cpu) > TOLERANCE}
    assert not off, (
        f"at {OPACITY:.0%} page opacity the CPU track passes {cpu:.3f} of the "
        f"backdrop but " + ", ".join(f"{k} passes {v:.3f}"
                                     for k, v in off.items()) +
        f" — measured over the whole card: {measured}")


def test_the_bars_do_not_dim_the_card_they_sit_in(qtbot, app_theme_restored):
    """And the shared number is the card's own, not some third value.

    "All four agree" would also be satisfied by all four being wrong
    together. The track is meant to blend into the box it sits in so that
    only the filled chunk stands out, which means the empty part of a bar has
    to be indistinguishable from the card surface next to it.
    """
    _win, screen = _screen(qtbot, OPACITY)
    alpha = _transmission(screen)
    regions = _regions(screen)
    card = _mean(alpha, regions["card"])

    off = {name: _mean(alpha, regions[name])
           for name, _ in BARS
           if abs(_mean(alpha, regions[name]) - card) > TOLERANCE}
    assert not off, (
        f"the card surface passes {card:.3f} of the backdrop but " +
        ", ".join(f"the {k} track passes {v:.3f}" for k, v in off.items()))


# ---------------------------------------------------------------------------
# The same claim against the real animation
# ---------------------------------------------------------------------------

def test_the_ambient_animation_reaches_all_four_bars_alike(
        qtbot, app_theme_restored):
    """Render the page with the animation on and again with it off, and diff.

    The comparison is made *inside each row*: how much the animation moves the
    bar's track, against how much it moves the bare card surface a few pixels
    above it, same columns. That controls for the thing a plain count cannot —
    the backdrop is an animation, its contrast against the flat page differs
    from one band of the screen to the next, and a blob happening to sit
    behind the CPU row would otherwise read as "the CPU bar is more
    transparent". The blobs are laid out from an unseeded RNG, so that is not
    hypothetical; measured across repeated builds the per-row ratio stays in
    0.92-1.04 when the bars are right and 0.68-0.71 for RAM, GPU and VRAM when
    they are not, while the CPU bar sits at 1.0 either way.

    Three animation times are accumulated so an unlucky flat patch behind one
    row cannot leave the ratio to be decided by rounding noise.
    """
    win_on, screen_on = _screen(qtbot, OPACITY, ambient=True)
    if getattr(screen_on, "_ambient", None) is None:
        pytest.skip("no ambient backdrop available in this environment")

    # Grab every animated frame BEFORE the second screen exists: building it
    # runs `apply_preferences_to_app`, and a screen that is still open when
    # the preference goes off takes the backdrop away — correctly, but it
    # would leave this test grabbing an un-animated page and calling it one.
    frames = []
    for moment in (2.0, 6.0, 11.0):
        screen_on._ambient.set_time(moment)
        QApplication.processEvents()
        frames.append(screen_on.grab().toImage())

    win_off, screen_off = _screen(qtbot, OPACITY, ambient=False)
    off = screen_off.grab().toImage()
    assert all(frame.size() == off.size() for frame in frames)

    def movement(rect):
        """Mean per-channel change the animation makes over ``rect``."""
        total = 0.0
        count = 0
        for frame in frames:
            for y in range(rect.top(), rect.bottom() + 1):
                for x in range(rect.left(), rect.right() + 1):
                    a, b = QColor(frame.pixel(x, y)), QColor(off.pixel(x, y))
                    total += (abs(a.red() - b.red())
                              + abs(a.green() - b.green())
                              + abs(a.blue() - b.blue())) / 3.0
                    count += 1
        return total / count

    ratios = {}
    for name, attr in BARS:
        row = getattr(screen_on, attr)
        track = _rect(screen_on, row._bar)
        # Bare card surface in the same row, directly above the track.
        row_rect = _rect(screen_on, row)
        surface = QRect(track.left(), row_rect.top() + 1, track.width(),
                        max(1, track.top() - row_rect.top() - 2))
        reference = movement(surface)
        assert reference > 1.0, (
            f"the animation barely moves the card behind the {name} row "
            f"({reference:.2f}/255), so nothing here can be concluded")
        ratios[name] = movement(track) / reference

    dull = {name: value for name, value in ratios.items() if value < 0.85}
    assert not dull, (
        "the animation reaches the card surface in these rows but not the "
        f"bar track on it: {dull} (all four ratios: {ratios})")

    cpu = ratios["CPU"]
    off_by = {name: value for name, value in ratios.items()
              if name != "CPU" and abs(value - cpu) > 0.15}
    assert not off_by, (
        f"the CPU track is influenced at {cpu:.2f} of its own row's card "
        f"surface but {off_by} are not — the bug is a bar painting a second "
        "translucent surface over the one already there")


def test_at_full_opacity_the_bars_match_the_card_as_well(
        qtbot, app_theme_restored):
    """The fix must be invisible where the user asked for an opaque page.

    At 100 % the card swallows the backdrop, and a bar that had gone
    transparent *relative to the window* rather than to the card would show
    up here as a stripe letting light through a page that should not.
    """
    _win, screen = _screen(qtbot, 1.0)
    alpha = _transmission(screen)
    regions = _regions(screen)
    card = _mean(alpha, regions["card"])
    for name, _ in BARS:
        assert abs(_mean(alpha, regions[name]) - card) <= TOLERANCE, (
            f"the {name} track does not match the card at 100 % page opacity")
