"""The page behind the settings categories has a colour of its own.

Reported three times, in the user's words: "there is STILL a black box
behind the settings categories, and the rest of the left side of many of
the modules including mask generation, measure, map bar codes,
regression, etc."

Two earlier fixes were real and neither one touched the cause. Both swept
more layout containers transparent so the backdrop would show between the
cards — correct, and it made the hole bigger, because with the ambient
animation switched off there is no backdrop and the thing behind the
containers had no colour at all. What showed through was the blanket
``QWidget {{ background-color: bg }}``, and ``DARK_PALETTE["bg"]`` is
literally ``#000000``.

Both earlier verifications passed because both ran with the ambient
backdrop **on**, which is the default and is not how the reporting user
runs. The measurement, on one real ``AppScreen``, sampling down the
settings column::

    ambient_enabled=True     2 / 74 samples pure black
    ambient_enabled=False   40 / 74 samples pure black

Histogrammed over the whole screen with it off, ``(0,0,0)`` was the
single most common colour on the page: 17,621 samples against 11,420 for
``surface``. It was black everywhere; it reads as "the left side" only
because the right half is under the console panel.

So this file fixes the verification as much as the code, and every test
in it runs the way the *user* runs:

* with ``ambient_enabled=False``, and separately with the Animation
  preference set to ``none`` — two different keys that answer the same
  question, and a fix that only satisfies one of them is half a fix;
* at 100 % page opacity **and** at 60 %, because a page that only
  separates from its panels while they are fully opaque is a page that
  breaks for anyone who moved the slider;
* through a real ``AppScreen``, for the four modules named in the
  report.

**Measured, not read.** A stylesheet string cannot tell you what reached
the screen, and neither can the backdrop-transmission probe the
neighbouring files use — that probe inserts an opaque widget *as the
page*, which is precisely the thing under test here. These assertions
render the screen and sample its pixels.

The fix is :data:`spacr.qt.theme.PAGE_SURFACES`' ``page`` role, added
alongside ``bg`` rather than replacing it: ``bg`` has three dozen uses
including ``QPalette.Window`` and ``QPalette.HighlightedText``, so
repointing it would change selected-text rendering and the ink on every
filled button. :func:`test_bg_is_still_the_window_colour` is what stops
the next person "simplifying" the two roles back into one.
"""
from __future__ import annotations

import collections

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from spacr.qt import preferences as prefs
from spacr.qt.theme import (PAGE_MIN_LSTAR, PAGE_MIN_RATIO, THEMES,
                            contrast_failures, contrast_ratio, lightness,
                            page_colour, page_separation_failures,
                            page_separation_report, palette_for)

#: Black. The colour this whole file exists to keep off the page.
BLACK = (0, 0, 0)

#: The four modules the user named, by registry key. Every one is a plain
#: ``AppScreen``, which is why one fix covers all four — and why a
#: regression in ``AppScreen`` would take all four with it.
NAMED_MODULES = ("mask", "measure", "map_barcodes", "regression")

#: The two preference paths that mean "no animation". ``ambient_enabled``
#: is the programmatic switch; ``animation = none`` is what the dropdown
#: writes. They are separate keys and both have to work — see
#: :func:`spacr.qt.preferences.get_ambient_enabled`.
NO_ANIMATION_MODES = ("ambient_enabled=False", "animation=none")

#: 100 % is the default and 60 % is a slider position a user actually
#: picks. A panel at 60 % composites ``0.6*panel + 0.4*page``, so it moves
#: most of the way toward the page and the separation is what has to
#: survive.
OPACITIES = (1.0, 0.6)

#: Every theme, plus the retired one. "space" is no longer offered but
#: `palette_for` is still called with whatever is persisted, so a settings
#: file written by an older spaCR can still resolve to it — and it would
#: be exactly as black.
ALL_THEMES = tuple(THEMES) + ("space",)


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write to the developer's real preferences.

    ``preferences._settings()`` builds ``QSettings(_ORG, _APP)``, which
    resolves to the NATIVE location whatever ``setPath`` says. Replacing
    the accessor is the only isolation that holds; the assertion refuses
    to run if it ever stops working.
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
    """Undo what this file does to the session-scoped QApplication.

    ``apply_preferences_to_app`` re-palettes and re-stylesheets the whole
    application. Leaving it at 60 % opacity would take out every later
    test that measures a pixel.
    """
    yield
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


# ---------------------------------------------------------------------------
# The palettes, without a screen in the way
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", ALL_THEMES)
def test_no_theme_resolves_its_page_to_black(theme):
    """The deliverable, in one line.

    Whatever else changes about the palettes, no theme may answer the
    question "what is behind the panels" with ``#000000``. The dark theme
    answered exactly that for as long as ``bg`` was doing this role's job.
    """
    resolved = page_colour(theme)
    assert QColor(resolved).getRgb()[:3] != BLACK, (
        f"the {theme} theme's page resolves to pure black ({resolved}) — "
        "with no ambient widget installed that IS the settings column, and "
        "it is the defect reported three times")


@pytest.mark.parametrize("theme", ALL_THEMES)
def test_every_theme_separates_its_page_from_its_panels(theme):
    """"Not black" is necessary and nowhere near sufficient.

    ``#010101`` would pass the test above and look identical to the user.
    What matters is that a settings category reads as a panel *on* the
    page, at full opacity and at 60 %, which is what
    :func:`spacr.qt.theme.page_separation_report` measures — in CIE L* as
    well as in contrast ratio, because down at the black end the ratio
    saturates and every pair of near-blacks is "about 1.1:1".

    Measured on the shipped values::

        dark   surface     100%  1.259:1   10.72 L*
        dark   surface      60%  1.169:1    6.92 L*
        dark   surface_alt 100%  1.170:1    6.96 L*
        dark   surface_alt  60%  1.102:1    3.99 L*
        light  surface     100%  1.277:1    9.58 L*
        light  surface_alt 100%  1.159:1    5.70 L*

    Against 1.087:1 / 3.95 L* for ``#000000`` on ``surface``, which is
    what the dark theme had and is why the panels did not read as panels.
    """
    failures = page_separation_failures(theme)
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("theme", ALL_THEMES)
def test_text_still_lands_on_the_page(theme):
    """The other half of the solve, and the ceiling on it.

    ``page`` is in :data:`spacr.qt.theme.PAGE_SURFACES`, so every rule in
    ``CONTRAST_RULES`` is enforced against it: text really does sit
    straight on the page between the cards — a section blurb, a hint under
    a field, an empty-state line.

    This is not a formality. ``fg_dim`` at 3.0:1 is what caps how far the
    dark page may travel from the panels, and ``accent`` at 4.5:1 caps the
    light one; between those ceilings and the separation floor above, each
    of the two lands in a band about two values wide. A change that
    brightens the dark page "a bit more" fails here.
    """
    failures = contrast_failures(theme)
    assert not failures, "; ".join(failures)


def test_bg_is_still_the_window_colour():
    """``page`` was added *beside* ``bg``, and must stay beside it.

    Repointing ``bg`` would have been the one-line version of this fix and
    it would have been wrong: ``bg`` is ``QPalette.Window``, it is
    ``QPalette.HighlightedText`` — the ink on a selected row and on every
    filled accent button — and it is the blanket ``QWidget`` fill. Three
    dozen uses, most of which are not "the page".

    So the dark theme's ``bg`` is still pure black on purpose, and the
    assertion that matters is that the two roles are different.
    """
    dark = palette_for("dark")
    assert dark["bg"] == "#000000", (
        "the dark window colour moved. That may well be right, but it is a "
        "change to selected-text rendering and to the ink on filled buttons, "
        "not a backdrop change — see the `page` block in theme.py")
    assert dark["page"] != dark["bg"], (
        "`page` and `bg` have been collapsed back into one role, which is "
        "the state that made the settings column a black box")


def test_the_ambient_flat_fill_is_the_page_not_the_window():
    """The animation's own backdrop had the same bug.

    ``_theme_background`` is the flat colour painted *under* the
    animation, and its docstring said "page colour" while the code read
    ``bg``. On the frames and in the gaps where the animation is thin,
    that black was what reached the eye.
    """
    from spacr.qt.widgets import ambient

    prefs.set_theme("dark")
    resolved = ambient._theme_background()
    assert resolved.getRgb()[:3] != BLACK
    assert resolved == QColor(page_colour("dark"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _show(qtbot, app_key: str, mode: str, opacity: float):
    """Build one module screen the way ``MainWindow`` puts it on screen.

    ``mode`` is one of :data:`NO_ANIMATION_MODES`. Both have to leave the
    screen with no ambient widget, and the assertion below says so rather
    than trusting it: ``set_ambient_animation("none")`` writes two keys
    and ``set_ambient_enabled(False)`` writes one, and a screen that
    quietly built an animation anyway would make every pixel assertion in
    this file meaningless.
    """
    from spacr.qt.screens.app_screen import AppScreen

    prefs.set_theme("dark")
    if mode == "ambient_enabled=False":
        prefs.set_ambient_enabled(False)
    else:
        prefs.set_ambient_animation("none")
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

    assert screen._ambient is None, (
        f"{app_key} built an ambient backdrop with {mode}; this file "
        "measures the page with nothing animating over it and an animation "
        "would hide every fault it looks for")
    # Keep the window alive: qtbot only weak-references it, and a collected
    # window deletes the C++ half of the screen under the test.
    return window, screen


def _column_histogram(screen, step: int = 4):
    """Sampled colours down the settings column, most common first.

    The column rather than the whole screen because that is the region the
    user pointed at, and because the right half is covered by the console
    panel — averaging the two together is part of how this was missed.
    """
    image = screen.grab().toImage()
    column = screen._settings_scroll
    top_left = column.mapTo(screen, QPoint(0, 0))
    counts: collections.Counter = collections.Counter()
    for y in range(top_left.y() + 2, top_left.y() + column.height() - 2, step):
        for x in range(top_left.x() + 2,
                       top_left.x() + column.width() - 2, step):
            colour = QColor(image.pixel(x, y))
            counts[colour.getRgb()[:3]] += 1
    assert counts, "sampled no pixels; the settings column has no area"
    return counts


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % rgb


# ---------------------------------------------------------------------------
# The four modules the user named
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opacity", OPACITIES)
@pytest.mark.parametrize("mode", NO_ANIMATION_MODES)
@pytest.mark.parametrize("app_key", NAMED_MODULES)
def test_the_settings_column_is_not_black(app_key, mode, opacity, qtbot,
                                          app_theme_restored):
    """No pure black anywhere down the settings column.

    The report, module by module. Before, at ``ambient_enabled=False``,
    sampling the same region: roughly half of every column was exactly
    ``(0,0,0)`` — 8,825 of 20,200 on mask, 9,472 of 19,594 on measure,
    11,013 of 21,000 on map_barcodes, 11,310 of 21,146 on regression.
    After, all four read zero.

    Two things had to change for that, and the second one is why "the page
    now has a colour" was not enough on its own: the scrollbar trough also
    took ``bg``, and a trough runs the full height of the column, so it
    stayed as a black stripe down the page — the last 168 black samples on
    mask, and the only ones left.
    """
    _window, screen = _show(qtbot, app_key, mode, opacity)
    counts = _column_histogram(screen)
    black = counts.get(BLACK, 0)
    total = sum(counts.values())
    assert black == 0, (
        f"{app_key} at {mode}, {opacity:.0%} page opacity: {black} of "
        f"{total} samples down the settings column are pure black. "
        f"Most common colours: "
        + ", ".join(f"{_hex(c)}x{n}" for c, n in counts.most_common(4)))


@pytest.mark.parametrize("opacity", OPACITIES)
@pytest.mark.parametrize("mode", NO_ANIMATION_MODES)
@pytest.mark.parametrize("app_key", NAMED_MODULES)
def test_the_page_is_the_page_colour(app_key, mode, opacity, qtbot,
                                     app_theme_restored):
    """And what is there instead is the theme's page, not something else.

    "Not black" is satisfied by any accident — a stray panel that happens
    to span the column, a scroll viewport left opaque. The positive
    assertion is that the *most common* colour down the column is exactly
    :func:`spacr.qt.theme.page_colour`, which is only true if the page is
    genuinely showing between the cards.
    """
    _window, screen = _show(qtbot, app_key, mode, opacity)
    counts = _column_histogram(screen)
    dominant, seen = counts.most_common(1)[0]
    assert _hex(dominant) == page_colour("dark"), (
        f"{app_key} at {mode}, {opacity:.0%}: the commonest colour down the "
        f"settings column is {_hex(dominant)} ({seen} samples), not the "
        f"page colour {page_colour('dark')}")


@pytest.mark.parametrize("opacity", OPACITIES)
@pytest.mark.parametrize("app_key", NAMED_MODULES)
def test_the_categories_read_as_panels_on_the_page(app_key, opacity, qtbot,
                                                   app_theme_restored):
    """The settings categories are distinguishable from what is behind them.

    This is the property the user was actually reporting. "A black box
    behind the settings categories" is what you say when the page and the
    panels are two near-blacks 1.087:1 apart — the categories stop reading
    as objects and the whole column reads as one slab.

    So: find the page colour and the commonest colour that is *not* it in
    the column, and require them to clear the same bars
    :func:`spacr.qt.theme.page_separation_report` sets on the palette.
    Measured from real pixels rather than from the palette, so a panel
    that ignores the opacity preference — or a container that quietly
    paints something else — fails here even though the palette is fine.
    """
    _window, screen = _show(qtbot, app_key, "ambient_enabled=False", opacity)
    counts = _column_histogram(screen)

    page = page_colour("dark")
    panels = [(rgb, n) for rgb, n in counts.most_common() if _hex(rgb) != page]
    assert panels, f"{app_key}: the column is a single flat colour"
    panel, _n = panels[0]

    ratio = contrast_ratio(page, _hex(panel))
    delta = abs(lightness(page) - lightness(_hex(panel)))
    # At 60 % the panel composites toward the page, so the bar is the
    # faded one — the same numbers `page_separation_report` uses.
    faded = opacity < 1.0
    min_ratio = 1.08 if faded else PAGE_MIN_RATIO
    min_delta = 2.0 if faded else PAGE_MIN_LSTAR
    assert ratio >= min_ratio and delta >= min_delta, (
        f"{app_key} at {opacity:.0%}: the biggest panel in the column is "
        f"{_hex(panel)} against a page of {page} — {ratio:.3f}:1 and "
        f"{delta:.2f} L*, where a panel must clear {min_ratio:.2f}:1 and "
        f"{min_delta:.2f} L* to read as a panel rather than as more page")


# ---------------------------------------------------------------------------
# Home takes the same treatment
# ---------------------------------------------------------------------------

def test_home_is_not_black_with_no_animation(qtbot, app_theme_restored):
    """Home had the fault twice over.

    Its ``_clear_page_surfaces`` sweep ran *only* inside the successful
    ambient install, so with no animation Home kept every layout container
    painting the blanket ``bg`` as well as having no page colour behind
    them. The module screens had already been corrected to sweep
    unconditionally; Home had not.
    """
    from spacr.qt.widgets.home import HomePage

    prefs.set_theme("dark")
    prefs.set_ambient_animation("none")
    prefs.set_pane_opacity(1.0)
    prefs.apply_preferences_to_app(QApplication.instance())

    page = HomePage([("mask", "Mask", "Generate masks", "Core")],
                    lambda key: None)
    window = QMainWindow()
    window.setCentralWidget(page)
    qtbot.addWidget(window)
    window.resize(1400, 950)
    window.show()
    QApplication.processEvents()
    assert page._ambient is None

    image = page.grab().toImage()
    counts: collections.Counter = collections.Counter()
    for y in range(0, image.height(), 4):
        for x in range(0, image.width(), 4):
            counts[QColor(image.pixel(x, y)).getRgb()[:3]] += 1
    black = counts.get(BLACK, 0)
    assert black == 0, (
        f"{black} of {sum(counts.values())} samples on Home are pure black "
        "with the animation set to none. Most common: "
        + ", ".join(f"{_hex(c)}x{n}" for c, n in counts.most_common(4)))


# ---------------------------------------------------------------------------
# The measurement has to be able to fail
# ---------------------------------------------------------------------------

def test_the_probe_still_sees_black_when_the_page_has_none(
        qtbot, app_theme_restored, monkeypatch):
    """Guards the guard, and it is the whole reason this file exists.

    Two earlier passes verified this fix and both verified it wrongly, so
    an assertion that "the column is not black" is worth nothing until it
    has been shown to go red for the original defect. Put the old
    behaviour back — a page that resolves to the window colour — and the
    same measurement, on the same screen, must report the black slab.
    """
    from spacr.qt import theme as theme_mod
    from spacr.qt.screens import app_screen as app_screen_mod

    monkeypatch.setattr(theme_mod, "page_colour",
                        lambda theme="dark": palette_for(theme)["bg"])
    monkeypatch.setattr(app_screen_mod, "AppScreen", app_screen_mod.AppScreen)

    _window, screen = _show(qtbot, "measure", "ambient_enabled=False", 1.0)
    counts = _column_histogram(screen)
    black = counts.get(BLACK, 0)
    assert black > 0, (
        "with the page pointed back at the window colour the settings column "
        "measured no black at all, so this file is not measuring the thing "
        "the user reported")


def test_the_report_names_the_roles_it_measured():
    """The separation report is data, not a boolean.

    A failure has to say which role, at which opacity, by how much — the
    three earlier rounds of this bug were all "it is still black", and a
    number that says *how* black is what shortens the next round.
    """
    rows = page_separation_report("dark")
    assert {r["role"] for r in rows} == {"surface", "surface_alt"}
    assert {r["opacity"] for r in rows} == {1.0, 0.6}
    for row in rows:
        assert row["page"] == page_colour("dark")
        assert row["ratio"] > 1.0 and row["delta_lstar"] > 0.0
