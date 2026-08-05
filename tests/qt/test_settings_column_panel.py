"""The settings column paints nothing; its categories are the panels.

Every module screen is the same two-column page: settings on the left,
console on the right. The right-hand column has been a dark-grey rounded
box carrying the user's page opacity since the console was built —
``QFrame#ConsoleBox``. The left-hand one took three attempts.

``an opaque slab``
    A ``QScrollArea``'s viewport auto-fills with the palette's **Window**
    brush. That is not a surface, so no page-opacity setting could reach
    it and the column sat as a black rectangle over the animated
    backdrop, whatever the slider said.
``a panel of its own``
    Turning the auto-fill off left the column with nothing, so it was
    given the console box's treatment. That put a box round a column of
    boxes, and every category then composited two translucent greys: 0.51
    at a requested 30 %, a shade no position of the slider can produce.
``nothing at all``
    Which is what shipped. The categories float directly on the theme as
    separate rounded panels, with the backdrop visible in the gaps
    between them. A list of categories needs no box round it.

So the surface under test is ``QFrame#SectionCard``, one per category,
and the property is that each of them is exactly **one** panel thick and
moves when the page-opacity preference moves. The column between them
must transmit the backdrop untouched.

One thing had to be measured with the window's hooks installed rather
than a bare ``AppScreen``: :func:`spacr.qt.settings_search.install` wraps
the strip *and* the scroll area in a container of its own, and that
container was a named ``QWidget`` with no rule — an opaque black
rectangle spanning the whole column, behind everything that had just been
made translucent. Every panel in front of it measured 0.000 at every
position of the slider while a bare ``AppScreen`` measured perfect. That
is why :func:`_show` takes ``with_strip``.

**Measured, not read.** A colour sample cannot tell an opaque dark panel
from a dark part of the backdrop, so every number here comes from
rendering the screen over solid black and again over solid white and
solving ``P = a·B + (1-a)·F`` per pixel, at two different page opacities
so that a fill which ignores the preference cannot pass by luck.

And the trap, recorded because it cost an hour: render an ``AppScreen``
*without* applying the application stylesheet and every sample comes back
``(239, 239, 239)`` in both directions — Qt's default palette. That reads
as "still opaque" when in fact nothing was themed at all. The fixtures
below apply :func:`spacr.qt.preferences.apply_preferences_to_app` first,
and the two ``test_the_probe_can_see_…`` guards fail if the probe ever
stops discriminating.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                               QWidget)

from spacr.qt import preferences as prefs

#: A page opacity well below 100 %, so there is something to see through.
OPACITY = 0.30

#: What one translucent panel over a clear page transmits at ``OPACITY``.
EXPECTED = 1.0 - OPACITY

#: How far the column may sit from the console box and still count as the
#: same panel. Generous enough for the hairline border and the antialiased
#: corners, far tighter than the gap between a panel (0.70), a slab (0.00)
#: and a bare page (1.00).
TOLERANCE = 0.05

#: The fifteen modules named in the report, by registry key. Every one of
#: them is a plain ``AppScreen``, which is why one fix covers all fifteen —
#: and why a regression in ``AppScreen`` would take all fifteen with it.
MODULE_KEYS = (
    "mask", "timelapse", "motility", "measure", "ml_analyze", "classify",
    "map_barcodes", "regression", "external_masks", "illumination",
    "train_cellpose", "cellpose_masks", "umap", "activation", "barcode_qc",
)


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write to the developer's real preferences.

    ``preferences._settings()`` builds ``QSettings(_ORG, _APP)``, which
    resolves to the NATIVE location whatever ``setPath`` says. Replacing
    the accessor is the only isolation that holds. The settings-search
    strip keeps its Essentials/All choice in a store of its own, built the
    same way, so it needs the same treatment.
    """
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    from spacr.qt import settings_search
    monkeypatch.setattr(settings_search, "_settings", lambda: store,
                        raising=False)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def app_theme_restored(qt_theme_applied):
    """Undo what this file does to the session-scoped QApplication."""
    yield
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


def _show(qtbot, app_key: str, opacity: float = OPACITY,
          with_strip: bool = False, ambient: bool = True):
    """Build one module screen the way ``MainWindow`` puts it on screen.

    ``with_strip`` runs the two window hooks the real app runs after a
    screen is shown — the settings-search strip and the Recipes button.
    They are *not* part of ``AppScreen``, which is why the column measured
    correct without them and was black with them: the strip's installer
    wraps the whole column in a container of its own.

    ``ambient`` is the Preferences toggle for the animated backdrop, and it
    is a parameter because turning it off used to change the whole page —
    see :func:`test_the_page_shows_through_with_the_backdrop_turned_off`.
    """
    from spacr.qt.screens.app_screen import AppScreen

    prefs.set_theme("dark")
    prefs.set_ambient_enabled(ambient)
    prefs.set_pane_opacity(opacity)
    prefs.apply_preferences_to_app(QApplication.instance())

    screen = AppScreen(app_key=app_key)
    window = QMainWindow()
    stack = QStackedWidget()
    window.setCentralWidget(stack)
    stack.addWidget(screen)
    qtbot.addWidget(window)
    window.resize(1400, 950)
    window.show()
    QApplication.processEvents()
    if with_strip:
        from spacr.qt import recipes, settings_search
        settings_search.install(screen)
        recipes.install(screen)
        QApplication.processEvents()
    # Keep the window alive: qtbot only weak-references it, and a collected
    # window deletes the C++ half of the screen under the test.
    return window, screen


def _transmission(screen):
    """Per-pixel ``alpha`` of everything painted over the backdrop."""
    ambient = getattr(screen, "_ambient", None)
    if ambient is not None:
        # Swap the animation for a flat colour so the two renders differ by
        # the backdrop and nothing else.
        ambient.hide()

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


def _rect(screen, widget) -> QRect:
    top_left = widget.mapTo(screen, QPoint(0, 0))
    return QRect(top_left.x(), top_left.y(), widget.width(), widget.height())


def _surface(alpha, rect: QRect, step: int = 3) -> float:
    """The transmission of the *surface* filling ``rect`` — its modal pixel.

    A panel is never one number over its whole rect, and the two obvious
    summaries both lie about it:

    *mean* mixes the panel with everything sitting on it. A category card
    holding a form is roughly half input fields, and those are opaque on
    purpose, so the average of a correct panel lands in the same place as
    a uniformly half-opaque slab.

    *the clearest rows* was the first attempt here and it lies the other
    way. ``QFrame#SectionCard`` carries ``margin-bottom`` in the shared
    stylesheet, and a QSS margin shrinks the painted box **inside** the
    widget's geometry — so the last few rows of every category are page,
    not panel. On a collapsed category that band is a large share of the
    rows and the statistic reported 0.787 for a card measuring a perfectly
    correct 0.702.

    The mode has neither problem. The panel is the largest single thing in
    its own rect by a wide margin, so the most common pixel value *is* the
    surface; fields, text, borders, rounded corners and margin bands are
    each a minority and none of them can outvote it.
    """
    from collections import Counter

    counts = Counter()
    for y in range(rect.top() + 1, rect.bottom(), step):
        for x in range(rect.left() + 1, rect.right(), step):
            counts[round(alpha(x, y), 2)] += 1
    assert counts, f"empty measurement region {rect}"
    return counts.most_common(1)[0][0]


def _mean(alpha, rect: QRect, step: int = 1) -> float:
    """Flat mean over ``rect`` — for a band that should be nothing but page."""
    values = [alpha(x, y)
              for y in range(rect.top(), rect.bottom() + 1, step)
              for x in range(rect.left(), rect.right() + 1, step)]
    assert values, f"empty measurement region {rect}"
    return sum(values) / len(values)


def _viewport(screen) -> QRect:
    """The part of the settings column that is actually on screen.

    Everything below is clipped to this, and it is not a detail. A
    category scrolled past the bottom of the column still *has* a
    geometry, and mapping it to the screen puts it somewhere off the end
    of the viewport — over the console, as it happens, which is a panel.
    Measuring there reads a mix of two screens and reports a defect in a
    region the user cannot see.
    """
    view = screen._settings_scroll.viewport()
    top_left = view.mapTo(screen, QPoint(0, 0))
    return QRect(top_left.x(), top_left.y(), view.width(), view.height())


def _inside(view: QRect, rect: QRect) -> bool:
    """Is ``rect`` scrolled fully into ``view``, vertically?

    Vertically only, and the horizontal half is handled by :func:`_clip`
    instead, because a category is as wide as the *content* widget and the
    content widget is routinely wider than the viewport — measured on
    Measure, the categories are 624 px across a 408 px viewport. Requiring
    full containment would reject every category on a screen where nothing
    is wrong.
    """
    return rect.top() >= view.top() and rect.bottom() <= view.bottom()


def _clip(view: QRect, rect: QRect) -> QRect:
    """``rect`` cropped to the part of the column that is on screen.

    The half of the clipping that costs a wrong answer rather than a
    missing one. A category 624 px wide in a 408 px viewport has 216 px of
    geometry hanging off the right of the column — and what is rendered
    there is the console, which is also a panel. Measuring the unclipped
    rect silently averages two screens: the gaps between categories came
    back at 0.872 instead of 1.000, which reads as "something is painting
    a container behind them" when the something is the console.
    """
    clipped = rect.intersected(view)
    assert clipped.width() > 2 and clipped.height() > 2, (
        f"{rect} has almost nothing inside the column {view}")
    return clipped


def _sections(screen):
    """The settings categories that are actually on screen.

    Visible *and* inside the viewport: ``isVisible()`` is true for a
    category scrolled out of the column, and its mapped rect lands on the
    console. Fully inside, so a card clipped by the bottom edge is not
    measured half-and-half either.
    """
    view = _viewport(screen)
    found = [s for s in getattr(screen, "_settings_sections", [])
             if s.isVisible() and _inside(view, _rect(screen, s))]
    assert found, (
        "no settings categories are on screen — every one of them is "
        "scrolled out of the column, so there is nothing to measure")
    return found


def _console(screen):
    """``QFrame#ConsoleBox``, the reference panel across the splitter."""
    box = screen.findChild(QWidget, "ConsoleBox")
    assert box is not None, (
        "no QFrame#ConsoleBox on this screen — it is the reference this file "
        "compares against; renamed?")
    return box


def _gaps(screen):
    """The strip of column between one category and the next.

    Where the theme has to show. It is ``SPACING["sm"]`` tall, so the rect
    is grown a little into neither panel: the first row below one card and
    the last above the next.
    """
    sections = _sections(screen)
    view = _viewport(screen)
    out = []
    for first, second in zip(sections, sections[1:]):
        above = _rect(screen, first)
        top = above.bottom()
        bottom = _rect(screen, second).top()
        if bottom - top < 3:
            continue
        band = QRect(above.left() + 4, top + 1,
                     above.width() - 8, bottom - top - 1)
        if _inside(view, band):
            out.append(_clip(view, band))
    assert out, "the categories are flush against each other — no gaps to see"
    return out


@pytest.mark.parametrize("app_key", MODULE_KEYS)
def test_every_category_is_one_panel_at_the_page_opacity(app_key, qtbot,
                                                         app_theme_restored):
    """A category is a floating panel, and it is exactly one panel thick.

    Measured at 30 % page opacity, clearest quarter of the rows, against
    the console box on the other side of the splitter — the panel the
    whole page is calibrated to. Comparing rather than pinning an absolute
    number is what lets Map Barcodes into the list: the sequencing screen
    paints the DNA rain across the page, which is opaque and sits behind
    both columns, so both read ~0.00 and still match.

    Before, with the column carrying a panel of its own, a category read
    0.519 here — two translucent greys composited, which is a shade no
    position of the slider can produce.
    """
    _window, screen = _show(qtbot, app_key)
    alpha = _transmission(screen)
    reference = _surface(alpha, _rect(screen, _console(screen)))
    for section in _sections(screen)[:3]:
        measured = _surface(
            alpha, _clip(_viewport(screen), _rect(screen, section)))
        assert abs(measured - reference) < TOLERANCE, (
            f"{app_key}: a settings category passes {measured:.3f} of the "
            f"backdrop and the console box passes {reference:.3f} — "
            + ("two surfaces stacked" if measured < reference
               else "no surface at all"))


def test_the_column_between_the_categories_is_the_page(qtbot,
                                                       app_theme_restored):
    """And there is no box round them.

    The gaps are the whole design: floating elements read as floating
    because the theme is visible between them. A container would show up
    here as a gap that transmits a panel's worth rather than all of it.

        gap between categories   0.702 (with the container) -> 1.000
    """
    _window, screen = _show(qtbot, "measure")
    alpha = _transmission(screen)
    for gap in _gaps(screen)[:4]:
        measured = _mean(alpha, gap)
        assert measured > 0.95, (
            f"the column between two categories passes only {measured:.3f} "
            "of the backdrop — something is painting a container behind them")


@pytest.mark.parametrize("with_strip", [False, True])
def test_the_page_shows_through_with_the_backdrop_turned_off(
        with_strip, qtbot, app_theme_restored):
    """The whole page, for the user who turned the animation off.

    This is the one that made the settings half a solid black rectangle,
    and it went unnoticed for as long as it did because every test in
    this area — including the rest of this file — left the ambient
    backdrop enabled.

    ``AppScreen`` ran ``_clear_page_surfaces`` only as a *side effect* of
    installing an animation: the DNA rain calls it before its install,
    ``_install_ambient`` after. Turn the backdrop off in Preferences and
    ``_install_ambient`` returns early, so the sweep never ran, so every
    layout container kept the blanket ``QWidget {{ background-color: bg }}``
    — the WINDOW colour. Measured, with the preference off::

                              before   after
        settings column       0.000    0.916
        a settings category   0.000    0.720
        the gap between two   0.000    1.000
        console box           0.000    0.702

    Every number, including the console's, which is how you can tell it
    was the page and not the column. The "after" column is identical to
    what the same screen measures with the backdrop on, which is the
    property: the preference chooses whether an animation plays, not
    whether the theme is visible.
    """
    _window, screen = _show(qtbot, "illumination", OPACITY,
                            with_strip=with_strip, ambient=False)
    alpha = _transmission(screen)

    console = _surface(alpha, _rect(screen, _console(screen)))
    assert abs(console - EXPECTED) < 0.06, (
        f"with the backdrop off the console box passes {console:.3f} where "
        f"one panel passes {EXPECTED:.2f} — the page is opaque")

    for section in _sections(screen)[:3]:
        measured = _surface(
            alpha, _clip(_viewport(screen), _rect(screen, section)))
        assert abs(measured - console) < TOLERANCE, (
            f"with the backdrop off a settings category passes "
            f"{measured:.3f} against the console's {console:.3f}")

    for gap in _gaps(screen)[:3]:
        measured = _mean(alpha, gap)
        assert measured > 0.95, (
            f"with the backdrop off the column between two categories "
            f"passes only {measured:.3f} of the page — it is the black "
            "rectangle that filled the settings half")


def test_the_probe_can_see_a_container_behind_the_categories(
        qtbot, app_theme_restored, monkeypatch):
    """Guards the guard.

    Put a panel back on the column — the arrangement that was tried and
    rejected — and both properties above must fail: the gaps stop being
    the page, and a category becomes two greys deep.
    """
    from spacr.qt import theme as theme_mod
    from spacr.qt.screens.app_screen import SETTINGS_PANEL_NAME

    def container(palette, opacity):
        from spacr.qt.theme import pane_surface
        return (f"QScrollArea#{SETTINGS_PANEL_NAME} {{ background-color: "
                f"{pane_surface('surface_alt', palette['theme'], opacity)}; }}")

    monkeypatch.setitem(theme_mod._WIDGET_QSS, SETTINGS_PANEL_NAME, container)
    _window, screen = _show(qtbot, "measure")
    alpha = _transmission(screen)
    gap = _mean(alpha, _gaps(screen)[0])
    assert gap < 0.95, (
        f"with a container painted behind them the gaps still pass "
        f"{gap:.3f} of the backdrop, so this file cannot see a container")
    section = _surface(
        alpha, _clip(_viewport(screen), _rect(screen, _sections(screen)[0])))
    reference = _surface(alpha, _rect(screen, _console(screen)))
    assert section < reference - TOLERANCE, (
        f"with a container behind it a category still passes {section:.3f} "
        f"against the console's {reference:.3f}, so this file cannot see two "
        "surfaces stacked")


def test_the_column_block_is_subtractive():
    """The rule that names the column says *paint nothing*.

    A pixel probe cannot tell "the column has no rule" from "the column
    has a rule that clears it", and the difference matters: without a
    rule the scroll area inherits the blanket
    ``QWidget {{ background-color: bg }}``, the WINDOW colour, which is
    the opaque slab this all started as.
    """
    from spacr.qt.theme import palette_for, _WIDGET_QSS
    from spacr.qt.screens.app_screen import (EMPTY_STATE_NAME,
                                             SETTINGS_PANEL_NAME)

    assert SETTINGS_PANEL_NAME in _WIDGET_QSS, (
        "app_screen did not register a QSS block for the settings column")
    palette = dict(palette_for("dark"), theme="dark")
    block = _WIDGET_QSS[SETTINGS_PANEL_NAME](palette, OPACITY)

    assert f"QScrollArea#{SETTINGS_PANEL_NAME}" in block
    assert f"QWidget#{EMPTY_STATE_NAME}" in block
    assert "background: transparent" in block
    assert "rgba(" not in block and "background-color: #" not in block, (
        "the column must not paint a surface of its own — the categories "
        f"are the panels; got: {block}")


def test_the_category_surface_carries_the_page_opacity(qt_theme_applied):
    """And the categories' own rule goes through the page-opacity roles.

    ``QFrame#SectionCard`` is in the shared stylesheet rather than in a
    registered block, so this reads the generated sheet. ``rgba(`` is the
    tell: the surface roles are plain hex at 100 % and ``rgba()`` below it,
    so a sheet generated at 30 % that still has hex on that rule is one
    the slider cannot reach.
    """
    from spacr.qt.theme import stylesheet

    sheet = stylesheet("dark", surface_opacity=OPACITY)
    rule = sheet.split("QFrame#SectionCard {")[-1].split("}")[0]
    assert "rgba(" in rule, (
        "the settings categories must take a page surface, so the opacity "
        f"preference reaches them; got: {rule.strip()}")


# ---------------------------------------------------------------------------
# The strip above the column, and everything the strip was hiding
# ---------------------------------------------------------------------------
# `settings_search.install` wraps the strip AND the settings scroll area in a
# container of its own so the two share one splitter slot. That container is a
# NAMED QWidget, and `clear_container_surfaces` only tags anonymous ones — so
# it fell through to the blanket `QWidget { background-color: bg }`, and `bg`
# is the window colour. One opaque black rectangle spanning the whole settings
# column, behind everything that had just been made translucent.
#
# That is why the categories and the empty-state banner both read "not subject
# to the opacity setting": every one of them was correct, and every one of them
# was composited onto black. The tests above build the screen the way
# `AppScreen` does, which is without the strip, and they all passed while the
# running app was black. These build it the way the WINDOW does.

#: Everything on the strip, plus the empty-state line under it. None of these
#: is a card: they are type and controls on the page, so each must paint
#: nothing at all and let the theme through.
TRANSPARENT_NAMES = (
    "SettingsSearchBar",
    "SettingsSearchCount",
    "SettingsSearchModifiedLabel",
    "SettingsSearchDisclosure",
    "SettingsRecipeButton",
    "EmptyStateBanner",
)

#: Where the slider is put for the two-point comparison. Far enough apart
#: that a fill which does not follow the preference cannot fake it.
LOW, HIGH = 0.30, 0.80

#: Below this a region is passing so little of the backdrop that it is the
#: black rectangle these tests exist to catch.
OPAQUE_ENOUGH = 0.20


def _named(screen, name):
    widget = screen.findChild(QWidget, name)
    assert widget is not None and widget.isVisible(), (
        f"no visible widget named {name!r} on this screen — renamed? the QSS "
        "rule keys off the same name")
    return widget


@pytest.mark.parametrize("name", TRANSPARENT_NAMES)
def test_the_strip_and_the_banner_paint_nothing(name, qtbot,
                                                app_theme_restored):
    """Type on the page, not boxes.

    Measured with the window's hooks installed, at 30 % page opacity.
    Before, every one of these read 0.000 — the black container behind
    them. After::

        SettingsSearchBar            0.000 -> 0.995
        SettingsSearchCount          0.000 -> 0.997
        SettingsSearchModifiedLabel  0.000 -> 1.000
        SettingsSearchDisclosure     0.000 -> 1.000
        SettingsRecipeButton         0.000 -> 1.000
        EmptyStateBanner             0.000 -> 1.000

    1.000 is the backdrop arriving untouched, which is what "no background
    between the theme and the text" means as a number. The bar and the
    count line stop a hair short of it because their own text is opaque
    and some of it lands in even the clearest rows.
    """
    _window, screen = _show(qtbot, "illumination", LOW, with_strip=True)
    measured = _surface(_transmission(screen),
                         _rect(screen, _named(screen, name)))
    assert measured > 0.95, (
        f"{name} passes only {measured:.3f} of the backdrop — it is painting "
        "a background, and it must not")


def test_the_categories_follow_the_slider_under_the_strip(qtbot,
                                                          app_theme_restored):
    """The main event: drag the slider and the categories move.

    The two-point measurement is the point. A surface that does not go
    through :func:`spacr.qt.theme.pane_surface`, or one composited onto an
    opaque rectangle, reads the *same* at both ends. These move by the
    difference between the two settings, with the strip installed::

                            30 %    80 %
        SectionCard         0.702   0.200
        ConsoleBox          0.702   0.200   (the reference, unchanged)

    Before, both categories read 0.000 at both settings.
    """
    for opacity in (LOW, HIGH):
        _window, screen = _show(qtbot, "illumination", opacity,
                                with_strip=True)
        alpha = _transmission(screen)
        reference = _surface(alpha, _rect(screen, _console(screen)))
        assert abs(reference - (1.0 - opacity)) < 0.06, (
            f"the console box passes {reference:.3f} at {opacity:.0%}; the "
            "reference itself is wrong and nothing below means anything")
        for section in _sections(screen)[:3]:
            measured = _surface(
            alpha, _clip(_viewport(screen), _rect(screen, section)))
            assert abs(measured - reference) < TOLERANCE, (
                f"at {opacity:.0%} a settings category passes "
                f"{measured:.3f} against the console's {reference:.3f} — "
                + ("something opaque behind it" if measured < reference
                   else "no surface at all"))


def test_the_probe_can_see_the_black_pane(qtbot, app_theme_restored,
                                          monkeypatch):
    """Guards the guard, for the strip.

    Put the unstyled container back — drop the block that tells the pane
    and the strip to paint nothing — and the categories must go black
    again, which is the state that was reported.
    """
    from spacr.qt import theme as theme_mod
    from spacr.qt.settings_search import BAR_NAME

    monkeypatch.setitem(theme_mod._WIDGET_QSS, BAR_NAME,
                        lambda palette, opacity: "")
    _window, screen = _show(qtbot, "illumination", LOW, with_strip=True)
    first = _sections(screen)[0]
    measured = _surface(_transmission(screen),
                        _clip(_viewport(screen), _rect(screen, first)))
    assert measured < OPAQUE_ENOUGH, (
        f"with the strip's block removed a settings category still passes "
        f"{measured:.3f} of the backdrop, so this file is not measuring what "
        "made the running app black")


def test_a_late_registered_block_reaches_a_styled_application(qt_theme_applied):
    """The seam that made the fix invisible in the running app.

    ``app.py`` imports a screen's module inside the branch that builds it,
    which is long after the launch stylesheet was generated — so a block
    registered at module import is not in the sheet that is live, and the
    screen opens unstyled however correct its rule is.
    :func:`spacr.qt.theme.ensure_widget_qss_applied` is what closes it, and
    this is the case it exists for.
    """
    from spacr.qt.theme import ensure_widget_qss_applied, register_widget_qss

    name = "_TestLateBlock"
    register_widget_qss(name, lambda palette, opacity:
                        "QLabel#_TestLateBlock { color: #ff00ff; }",
                        replace=True)
    try:
        assert name not in qt_theme_applied.styleSheet(), (
            "the fixture's stylesheet predates this registration; that is "
            "the whole premise")
        assert ensure_widget_qss_applied(name) is True
        assert name in qt_theme_applied.styleSheet()
        # Idempotent: a second call has nothing to fix.
        assert ensure_widget_qss_applied(name) is False
    finally:
        from spacr.qt.theme import stylesheet, unregister_widget_qss
        unregister_widget_qss(name)
        qt_theme_applied.setStyleSheet(stylesheet())
