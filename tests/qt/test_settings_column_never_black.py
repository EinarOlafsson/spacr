"""The settings column must never be a black box. Measured, not asserted.

This regressed three times. Each time the report was the same sentence --
"there is a black box behind the settings categories" -- and each time the
fix swept one more container transparent, which made it worse, because the
thing *behind* the containers had no colour of its own. With the ambient
animation switched off there was nothing painting the page at all, so what
showed through was the blanket ``QWidget { background-color: bg }`` from
``_window_block``, and in the dark theme ``bg`` is literally ``#000000``.

So this file does not check a property of the code. It renders a real
:class:`AppScreen` onto a magenta page and counts pixels, because every
earlier fix passed the code-shaped checks that existed at the time.

Two traps are baked in, both of which produced a confidently wrong answer
before:

1. ``theme.stylesheet()`` MUST be applied to the QApplication first.
   Without it, every widget renders in Qt's default palette -- a uniform
   (239, 239, 239) -- and the probe reports a clean page for a screen that
   is black in the real app.

2. The page must start as a colour nothing else uses, not as transparent.
   Filling with 0 and counting "black" pixels scores an *unpainted* region
   as clean, and unpainted is exactly the failure: in the real window the
   thing underneath is ``bg``, which is ``#000000``.

Hence magenta. A magenta pixel means nobody painted it, which in the real
window is the black box. A (0, 0, 0) pixel means something painted the
window colour. Both are failures and both are counted.
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QScrollArea, QWidget

#: Magenta: not in any palette, so it can only be the untouched page.
UNPAINTED = QColor(255, 0, 255)

#: Every module the black box was reported on, plus the ones that share
#: the settings-column construction with them. `mask` and `timelapse` are
#: the widest forms (190 and 204 keys); `cellpose_masks` is among the
#: narrowest, and a short form leaves more page showing, which is where
#: the hole was most visible.
MODULES = (
    "mask", "timelapse", "motility", "measure", "ml_analyze", "classify",
    "map_barcodes", "regression", "external_masks", "illumination",
    "train_cellpose", "cellpose_masks", "umap", "activation",
    "barcode_qc", "model_compare", "model_zoo", "control_chart",
)

#: Below this the column is sound. Not zero: a handful of samples land on
#: antialiased glyph edges and on the focus ring, which are legitimately
#: dark. The failures this guards against were 44.4% and 45.9%, so the
#: gap between passing and failing is three orders of magnitude and the
#: exact threshold is not load-bearing.
TOLERANCE_PCT = 0.5


def _sample_settings_column(app_key: str, ambient: bool) -> tuple[float, float]:
    """Render the screen and return (unpainted %, pure-black %).

    :param ambient: the backdrop animation. ``False`` is the case that
        actually broke -- with it on, the ambient widget paints the page
        and hides the hole, which is why this was reported by users and
        not caught here.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    # trap 1 -- the palette + QSS come from the `qt_theme_applied` fixture
    # the tests below depend on. Without it every widget renders in Qt's
    # default (239, 239, 239) and this probe reports a clean page.
    app = QApplication.instance()

    was = preferences.get_ambient_enabled()
    preferences.set_ambient_enabled(ambient)
    try:
        screen = AppScreen(app_key)
        screen.resize(1600, 1000)
        screen.show()
        for _ in range(10):
            app.processEvents()

        box = next((w for w in screen.findChildren(QScrollArea)
                    if w.objectName() == "SettingsBox"), None)
        if box is None:
            pytest.skip(f"{app_key} has no SettingsBox")

        page = QImage(screen.size(), QImage.Format_ARGB32)
        page.fill(UNPAINTED)                       # trap 2 -- see module docstring
        screen.render(page, QPoint(), screen.rect(),
                      QWidget.RenderFlag.DrawChildren)

        top_left = box.mapTo(screen, QPoint(0, 0))
        unpainted = black = total = 0
        for y in range(top_left.y() + 4, top_left.y() + box.height() - 4, 4):
            for x in range(top_left.x() + 4, top_left.x() + box.width() - 4, 4):
                colour = page.pixelColor(x, y)
                total += 1
                if colour == UNPAINTED:
                    unpainted += 1
                elif colour.red() == colour.green() == colour.blue() == 0:
                    black += 1
        screen.deleteLater()
        return 100.0 * unpainted / total, 100.0 * black / total
    finally:
        preferences.set_ambient_enabled(was)


@pytest.mark.parametrize("app_key", MODULES)
def test_the_settings_column_is_not_a_black_box(qt_theme_applied, app_key):
    """With the backdrop off, the settings column still has a colour.

    This is the exact configuration that was reported. At the commit
    before the fix, `mask` scored 44.4% unpainted here.
    """
    unpainted, black = _sample_settings_column(app_key, ambient=False)
    assert unpainted + black < TOLERANCE_PCT, (
        f"{app_key}: {unpainted:.1f}% of the settings column is unpainted "
        f"and {black:.1f}% is the window colour. Unpainted is the bug -- in "
        f"the real window `bg` (#000000) shows through, which is the black "
        f"box. Do not fix this by making one more container transparent; "
        f"that is what caused it. The page itself needs a colour."
    )


@pytest.mark.parametrize("app_key", MODULES[:6])
def test_the_settings_column_is_not_a_black_box_with_the_backdrop_on(
        qt_theme_applied, app_key):
    """And with the backdrop on, which is the default.

    Fewer modules: the ambient widget paints the page in this mode, so
    this direction has never been the one that broke. It is here so that
    a future change to the page colour cannot fix the off case by
    breaking the on case.
    """
    unpainted, black = _sample_settings_column(app_key, ambient=True)
    assert unpainted + black < TOLERANCE_PCT, (
        f"{app_key}: {unpainted:.1f}% unpainted, {black:.1f}% window colour "
        f"with the ambient backdrop enabled"
    )


# ---------------------------------------------------------------------------
# A FOURTH TIME, AND FROM A DIRECTION A BARE `AppScreen` CANNOT SEE
# ---------------------------------------------------------------------------
# Everything above renders an `AppScreen` on its own. That is the right
# instrument for the three regressions it records, and it is blind to the
# one below, because the switch that produced this one lives on the WINDOW.
#
# `MainWindow` keeps exactly one animated backdrop, on the central widget,
# behind the dock and the screen stack both. A module screen that had built
# one of its own gives it up -- and from that moment it must paint no page,
# because whatever it painted would land straight on top of the window's
# animation. `AppScreen.page_fill` reads `_uses_window_backdrop` to decide,
# and that flag is a claim about the window AS IT STANDS NOW: switching the
# animation off in Preferences clears it on every screen, which is right,
# because there is then nothing to defer to and the page has to paint itself
# again.
#
# Switching it back on was the half that was missing. The reconcile at the
# end of `refresh_theme` only ever cleared the flag, and the branch that
# could have set it again asked "has this screen a backdrop of its own" --
# which is false both for a screen that never had one (HomePage, which must
# keep painting its page as the floor beneath the animation) and for a
# screen that gave one away (which must not paint anything). So every module
# already built went on painting its flat page colour over the restored
# animation for the rest of the session, while a module opened after the
# toggle was correct.
#
# MEASURED FROM THE SEAT THE BACKDROP OCCUPIES. A probe is put at the bottom
# of the central widget's children, exactly where the `AmbientWidget` sits,
# and the whole window is rendered over it once black and once white; the
# difference per pixel is how much of the backdrop reaches the eye. Dark
# theme, 30 % page opacity, the settings column of one module that stays
# open throughout:
#
#     the module is opened        1.00 page, 0.70 panels
#     the animation is off        0.00 everywhere     (correct: no backdrop)
#     the animation is on again   0.00 everywhere     (the defect)
#     with the fix                1.00 / 0.70 again
#
# The animation is HIDDEN for the measurement, so nothing here depends on
# `QWidget.grab` being able to see a GL-backed widget -- it cannot, and a
# grab of a working backdrop comes back black, which has cost this area an
# hour and a wrong conclusion before. What is measured is the transmission
# of everything stacked above the probe, which is exactly what the defect
# breaks.

#: A page opacity well below 100 %, so there is something to see through.
OPACITY = 0.30

#: The module the window-level measurement uses. Any plain `AppScreen` would
#: do: the flag and the page fill live on the class, not on the module.
MODULE = "mask"

#: How much of the settings column has to be passing at least half of the
#: backdrop for the column to be a column rather than a slab. A correct
#: screen measures about 0.81 of its samples above this line -- the clear
#: page between the cards, plus the cards themselves at 0.70 -- and a
#: slabbed one measures 0.00, because every sample is one opaque page
#: colour. The threshold sits between them with room on both sides, and
#: deliberately well under the good reading: input fields are opaque on
#: purpose and their share moves with the module, the window size and the
#: font.
CLEAR_ENOUGH = 0.50


@pytest.fixture
def preferences_restored():
    """Put the animation preference and the application sheet back."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt import preferences as prefs
    from spacr.qt.theme import apply_qpalette, stylesheet

    before = (prefs.get_theme(), prefs.get_ambient_enabled(),
              prefs.get_pane_opacity())
    yield
    prefs.set_theme(before[0])
    prefs.set_ambient_enabled(before[1])
    prefs.set_pane_opacity(before[2])
    app = QApplication.instance()
    if app is not None:
        apply_qpalette(app)
        app.setStyleSheet(stylesheet())


def _transmission(window):
    """Per-pixel ``alpha`` of everything painted over the window's backdrop.

    The probe goes where the backdrop goes -- the bottom of the central
    widget's children -- so what it measures is what the animation would
    have shown through. The animation itself is hidden first, because the
    two renders have to differ by the probe and nothing else.
    """
    from PySide6.QtWidgets import QApplication

    central = window.centralWidget()
    backdrop = window.window_backdrop()
    if backdrop is not None:
        backdrop.hide()
    # The first-run tour dims the whole window on purpose, which is a scrim
    # over the measurement rather than a defect in what is being measured.
    for child in window.findChildren(QWidget):
        if type(child).__name__ == "_TourOverlay":
            child.hide()

    probe = central.findChild(QWidget, "BackdropProbe")
    if probe is None:
        probe = QWidget(central)
        probe.setObjectName("BackdropProbe")
    probe.setGeometry(0, 0, central.width(), central.height())
    probe.lower()
    probe.show()

    def render(colour):
        probe.setStyleSheet(
            f"QWidget#BackdropProbe {{ background: {colour}; }}")
        probe.lower()
        QApplication.processEvents()
        return window.grab().toImage()

    dark, light = render("#000000"), render("#ffffff")
    probe.hide()
    if backdrop is not None:
        backdrop.show()

    def alpha(x, y):
        a, b = QColor(dark.pixel(x, y)), QColor(light.pixel(x, y))
        return ((b.red() - a.red()) + (b.green() - a.green())
                + (b.blue() - a.blue())) / 765.0

    return alpha


def _column_rect(window, screen):
    """The part of the settings column that is on screen, in window coords."""
    from PySide6.QtCore import QRect

    view = screen._settings_scroll.viewport()
    top_left = view.mapTo(window, QPoint(0, 0))
    return QRect(top_left.x(), top_left.y(), view.width(), view.height())


def _clear_share(alpha, rect, step: int = 4) -> float:
    """The share of ``rect`` that passes at least half of the backdrop.

    A share rather than a modal value, because the column is not one
    surface: it is clear page between the cards, one translucent panel at
    each card, and a scattering of opaque input fields. A slab replaces all
    three with a single opaque colour, so the share above the halfway line
    is the statistic that separates them -- 0.00 against about 0.81.
    """
    from collections import Counter

    counts = Counter()
    for y in range(rect.top() + 1, rect.bottom(), step):
        for x in range(rect.left() + 1, rect.right(), step):
            counts[alpha(x, y) >= 0.5] += 1
    total = sum(counts.values())
    assert total, f"empty measurement region {rect}"
    return counts[True] / total


def _window_with_a_module_open(qtbot):
    """A window with one module open on the dark theme, animation on."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow

    prefs.set_theme("dark")
    prefs.set_ambient_enabled(True)
    prefs.set_pane_opacity(OPACITY)
    prefs.apply_preferences_to_app(QApplication.instance())

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 950)
    window.show()
    QApplication.processEvents()
    if window.window_backdrop() is None:
        pytest.skip("the ambient backdrop is off in this configuration")
    window._on_nav_selected(MODULE)
    qtbot.wait(50)
    QApplication.processEvents()
    return window, window._screens[MODULE]


def _set_the_animation(window, on: bool) -> None:
    """Turn the animation on or off the way the Preferences dialog does.

    Through ``apply_preferences_to_app`` and then the owner window's
    ``refresh_theme``, which is the pair the Save button runs. A guard that
    reached past them would be setting up a world of its own choosing
    rather than the one the application builds.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt import preferences as prefs

    prefs.set_ambient_enabled(on)
    prefs.apply_preferences_to_app(QApplication.instance())
    window.refresh_theme()
    QApplication.processEvents()


def test_the_settings_column_comes_back_when_the_animation_does(
        qtbot, qt_theme_applied, preferences_restored):
    """Off and on again leaves the column exactly as it found it.

    Three readings from one window, so the comparison is of one screen
    against itself rather than of two builds that might differ for reasons
    of their own. The middle reading is the calibration this file's own
    history asks for: with the animation off the column MUST measure as a
    slab, because that is the state the probe has to be able to see.
    """
    window, screen = _window_with_a_module_open(qtbot)
    fresh = _clear_share(_transmission(window), _column_rect(window, screen))
    assert fresh > CLEAR_ENOUGH, (
        f"the settings column passes only {fresh:.1%} of the backdrop on a "
        f"freshly opened module, so this test cannot tell a slab from a "
        f"column and nothing below it means anything")

    _set_the_animation(window, False)
    dimmed = _clear_share(_transmission(window), _column_rect(window, screen))
    assert dimmed < CLEAR_ENOUGH, (
        f"with the animation off the settings column still passes "
        f"{dimmed:.1%} of what is behind it -- the page is not painting "
        f"itself, so the probe cannot see the slab this guards against")

    _set_the_animation(window, True)
    restored = _clear_share(_transmission(window),
                            _column_rect(window, screen))
    assert restored > CLEAR_ENOUGH, (
        f"the settings column passes {restored:.1%} of the restored "
        f"backdrop against {fresh:.1%} when the module was opened -- the "
        f"screen is painting a flat page over the animation, which is the "
        f"black box behind the settings")


def test_a_screen_that_gave_its_backdrop_up_defers_to_the_window_again(
        qtbot, qt_theme_applied, preferences_restored):
    """And the switch that decides it, named so a failure says which one.

    The pixel test above is the one that matters. This one reports which
    position the switch was left in when it fails.
    """
    window, screen = _window_with_a_module_open(qtbot)
    assert screen.page_fill() is None, (
        "the module screen is painting a page of its own while the window "
        "is running a backdrop")
    assert getattr(screen, "_surrendered_its_backdrop", False), (
        "the screen has no record that it gave its own backdrop away, so "
        "nothing can tell it apart from a screen that never had one")

    _set_the_animation(window, False)
    assert screen.page_fill() is not None, (
        "with no backdrop anywhere the screen must paint its own page "
        "again, or there is nothing behind it but the window colour")

    _set_the_animation(window, True)
    assert screen.page_fill() is None, (
        "the screen went on painting a flat page after the animation came "
        "back -- a slab over a running backdrop, on every module that was "
        "already open")
