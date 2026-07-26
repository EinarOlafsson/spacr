"""Home page: category sizes, and labels that fit the widget drawing them.

Two invariants live here, both of which were broken before item #16c:

1. **No category holds more than nine apps.** "Tools" had grown to
   sixteen — a horizontal row nobody could read to the end of.
2. **No app name is silently clipped.** ``HTile`` paints its name in a
   child ``QLabel``, but ``QPushButton.sizeHint()`` only measures the
   button's *own* text and icon, so the label's width requirement never
   reached the layout: every tile was pinned to the 210 px floor and any
   name wider than the 118 px that left over ("Annotator Agreement",
   "Database Browser", "Format Converter", "Model Compare", "Cellpose
   Masks") lost its tail with no ellipsis and no warning.

The label-fit test is parametrised over every entry in ``APPS`` and runs
under both themes, so a new app with a long name fails here rather than
in a screenshot months later.
"""
from __future__ import annotations

import pytest

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QPushButton

from spacr.qt.app import (
    APPS,
    MAX_APPS_PER_SECTION,
    SECTIONS,
    Sidebar,
    _icon_for_app,
)
from spacr.qt.screens.startup import StartupPage
from spacr.qt.widgets.tile import HTile


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sections_in_order() -> list[str]:
    """Section names in the order APPS declares them, de-duplicated."""
    seen: list[str] = []
    for *_rest, section in APPS:
        if section not in seen:
            seen.append(section)
    return seen


def _counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for *_rest, section in APPS:
        counts[section] = counts.get(section, 0) + 1
    return counts


@pytest.fixture(scope="module", params=["dark", "light"])
def home(request, qapp, qt_theme_applied):
    """A laid-out Home page + sidebar under one theme.

    Module-scoped so the (fairly expensive) page build happens twice for
    the whole file rather than once per parametrised app.

    The theme is applied to these two widgets rather than to the
    QApplication: a global ``setStyleSheet`` re-polishes every widget any
    other test left behind, which is both slow and a good way to crash
    the interpreter on a stale one.
    """
    from spacr.qt.theme import stylesheet

    theme = request.param
    qss = stylesheet(theme)

    page = StartupPage(APPS, _icon_for_app)
    page.setStyleSheet(qss)
    page.resize(1400, 900)
    page.show()

    bar = Sidebar()
    bar.setStyleSheet(qss)
    bar.resize(bar.width(), 1400)
    bar.show()

    qapp.processEvents()
    yield theme, page, bar

    page.hide()
    bar.hide()
    qapp.processEvents()


def _tiles_by_name(page: StartupPage) -> dict:
    return {t.text_label: t for t in page.findChildren(HTile)}


def _nav_buttons_by_name(bar: Sidebar) -> dict:
    """App navigation buttons keyed by app name (Home excluded)."""
    return {b.accessibleName(): b for b in bar.findChildren(QPushButton)
            if b.property("navKey") not in (None, "__home__")}


# ---------------------------------------------------------------------------
# Part B — categories
# ---------------------------------------------------------------------------

def test_no_category_holds_more_than_nine_apps():
    """The invariant the whole regrouping exists to satisfy."""
    oversized = {s: n for s, n in _counts().items()
                 if n > MAX_APPS_PER_SECTION}
    assert not oversized, (
        f"sections over the {MAX_APPS_PER_SECTION}-app cap: {oversized}. "
        "Split one out with a name that means something rather than "
        "letting a row grow past what anyone reads.")


def test_every_category_is_big_enough_to_deserve_a_heading():
    """A one- or two-entry section is noise, not navigation (cf. #12b)."""
    tiny = {s: n for s, n in _counts().items() if n < 3}
    assert not tiny, (
        f"sections too small to justify their own heading: {tiny}")


def test_sections_appear_in_the_declared_workflow_order():
    """APPS order drives both Home and the sidebar, so it is the order."""
    assert _sections_in_order() == list(SECTIONS)


def test_every_app_is_in_a_declared_section():
    stray = sorted({s for *_r, s in APPS} - set(SECTIONS))
    assert not stray, f"apps in undeclared sections: {stray}"


def test_the_first_nine_apps_are_the_core_pipeline():
    """Ctrl+1..9 map to APPS[0..8]; keep that the end-to-end pipeline."""
    from spacr.qt.app import SECTION_CORE
    assert [a[3] for a in APPS[:9]] == [SECTION_CORE] * 9


def test_app_keys_are_unique():
    keys = [a[0] for a in APPS]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate app keys: {dupes}"


def test_app_display_names_are_unique():
    """Two apps with the same name are indistinguishable in the palette."""
    names = [a[1] for a in APPS]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate app names: {dupes}"


def test_every_app_has_a_title_and_an_intro():
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    missing_title = [k for k, *_ in APPS if not APP_TITLES.get(k, "").strip()]
    missing_intro = [k for k, *_ in APPS if not APP_INTROS.get(k, "").strip()]
    assert not missing_title, f"apps with no title: {missing_title}"
    assert not missing_intro, f"apps with no intro: {missing_intro}"


def test_every_app_resolves_to_a_screen(qtbot, qt_theme_applied):
    """An app in the registry that cannot be opened is a dead tile."""
    from PySide6.QtWidgets import QWidget
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    for key, name, *_rest in APPS:
        screen = win._build_screen(key)
        assert isinstance(screen, QWidget), (
            f"{key} ({name}) did not build a screen")
        screen.deleteLater()


# ---------------------------------------------------------------------------
# Part A — nothing is clipped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,name,desc,section", APPS,
                         ids=[a[0] for a in APPS])
def test_home_tile_shows_the_whole_name(home, key, name, desc, section):
    """The name fits the label drawing it — or elides WITH a tooltip.

    Silently truncated text is the bug; an ellipsis plus a tooltip is an
    acceptable fallback for a name no sane tile width could hold.
    """
    _theme, page, _bar = home
    tile = _tiles_by_name(page).get(name)
    assert tile is not None, f"{name} has no tile on Home"

    label = tile.name_label
    needed = QFontMetrics(label.font()).horizontalAdvance(name)
    available = label.available_text_width()
    if needed <= available:
        assert not label.is_elided()
        return
    assert label.is_elided(), (
        f"{name!r} needs {needed} px but has {available} px and was "
        "neither widened nor elided — it is being clipped")
    assert label.toolTip() == name, (
        f"{name!r} is elided but its tooltip does not carry the full name")


@pytest.mark.parametrize("key,name,desc,section", APPS,
                         ids=[a[0] for a in APPS])
def test_sidebar_item_shows_the_whole_name(home, key, name, desc, section):
    """Same contract for the left navigation column."""
    _theme, _page, bar = home
    btn = _nav_buttons_by_name(bar).get(name)
    assert btn is not None, f"{name} has no sidebar entry"

    needed = QFontMetrics(btn.font()).horizontalAdvance(btn.full_text())
    available = btn.available_text_width()
    if needed <= available:
        assert not btn.is_elided()
    else:
        assert btn.is_elided()
    # Elided or not, hovering must reveal which app this is.
    assert name in btn.toolTip()


def test_no_sidebar_item_needs_eliding_at_the_default_font(home):
    """The column widens to the longest name, so nothing should elide."""
    _theme, _page, bar = home
    clipped = [b.full_text().strip() for b in bar.clipped_items()]
    assert not clipped, (
        f"sidebar had to shorten {clipped} — widen Sidebar.WIDTH_MAX or "
        "shorten the name")


def test_no_home_tile_needs_eliding_at_the_default_font(home):
    """Same for Home: today's names all fit a tile that sizes to them."""
    _theme, page, _bar = home
    clipped = [t.text_label for t in page.findChildren(HTile)
               if t.is_name_elided()]
    assert not clipped, f"Home had to shorten {clipped}"


def test_a_pathologically_long_name_elides_instead_of_clipping(
        qtbot, qt_theme_applied):
    """The safety net: a name no tile could ever fit still stays readable."""
    from spacr.qt.app import SECTION_CORE

    long_name = "Extremely Long Hypothetical Module Name For Testing"
    page = StartupPage([("mask", long_name, "desc", SECTION_CORE)],
                       _icon_for_app)
    qtbot.addWidget(page)
    page.resize(1400, 600)
    page.show()
    qtbot.waitExposed(page)

    tile = page.findChildren(HTile)[0]
    label = tile.name_label
    assert label.is_elided(), "expected the label to elide"
    assert "…" in label.text()
    assert label.text() != long_name
    assert label.full_text() == long_name
    assert label.toolTip() == long_name
    # And the tile did not stretch across the whole page to fit it.
    from spacr.qt.preferences import scaled_px
    assert tile.width() <= scaled_px(StartupPage.TILE_CAP_W)


def test_nothing_clips_at_a_150_percent_font_scale(qtbot, qapp, monkeypatch):
    """The accessibility setting most likely to break a fixed width.

    Both the stylesheet and the px constants scale, so the tiles have to
    grow with the text rather than keep a 100 %-sized box.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.theme import stylesheet

    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.5)
    qss = stylesheet("dark", 1.5)

    page = StartupPage(APPS, _icon_for_app)
    qtbot.addWidget(page)
    page.setStyleSheet(qss)
    page.resize(1800, 1200)
    page.show()

    bar = Sidebar()
    qtbot.addWidget(bar)
    bar.setStyleSheet(qss)
    bar.resize(bar.width(), 1600)
    bar.show()
    qapp.processEvents()

    clipped = []
    for tile in page.findChildren(HTile):
        label = tile.name_label
        needed = QFontMetrics(label.font()).horizontalAdvance(tile.text_label)
        if needed > label.available_text_width() and not label.is_elided():
            clipped.append(tile.text_label)
    assert not clipped, f"clipped at 150 % font scale: {clipped}"
    assert not bar.clipped_items()
    # The column itself grew with the font rather than staying at 220.
    assert bar.width() == prefs.scaled_px(Sidebar.WIDTH_MIN)


def test_tile_width_tracks_the_name_it_has_to_draw(home):
    """A longer name gets a wider tile — that is the actual fix."""
    _theme, page, _bar = home
    tiles = _tiles_by_name(page)
    short, long = tiles["Mask"], tiles["Annotator Agreement"]
    assert long.width() > short.width()
    assert long.required_width() > short.required_width()


# ---------------------------------------------------------------------------
# Everything that iterates APPS still sees every app
# ---------------------------------------------------------------------------

def test_home_renders_every_app_under_every_section_heading(home):
    _theme, page, _bar = home
    rendered = set(_tiles_by_name(page))
    assert rendered == {a[1] for a in APPS}

    headings = {lbl.text() for lbl in page.findChildren(QLabel)}
    for section in SECTIONS:
        assert section.upper() in headings, (
            f"no '{section}' heading on Home")


def test_sidebar_renders_every_app_under_every_section_heading(home):
    _theme, _page, bar = home
    assert set(_nav_buttons_by_name(bar)) == {a[1] for a in APPS}

    headings = {lbl.text() for lbl in bar.findChildren(QLabel)
                if lbl.objectName() == "SidebarSection"}
    assert headings == set(SECTIONS)


def test_sidebar_still_has_a_home_button(home):
    _theme, _page, bar = home
    texts = {b.full_text().strip() for b in bar.findChildren(QPushButton)}
    assert "Home" in texts


def test_command_palette_finds_every_app_after_the_regrouping(
        qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.command_palette import CommandPalette

    win = MainWindow()
    qtbot.addWidget(win)
    palette = CommandPalette(win)
    qtbot.addWidget(palette)

    labels = [c.label for c in palette._commands]
    for _key, name, _desc, _section in APPS:
        assert f"Go to  {name}" in labels, f"{name} missing from the palette"

    # The section badge follows the app table, so the new names show up.
    badges = {c.section for c in palette._commands}
    for section in SECTIONS:
        assert f"Apps · {section}" in badges


def test_command_palette_filters_by_the_new_section_names(
        qtbot, qt_theme_applied):
    """Section names are searchable keywords — typing "batch" finds them."""
    from spacr.qt.app import MainWindow, SECTION_DATA
    from spacr.qt.command_palette import CommandPalette

    win = MainWindow()
    qtbot.addWidget(win)
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    palette._on_filter(SECTION_DATA)
    visible = [palette._list.item(i).text()
               for i in range(palette._list.count())]
    expected = {a[1] for a in APPS if a[3] == SECTION_DATA}
    for name in expected:
        assert any(name in v for v in visible), (
            f"{name} not found when filtering on its own section")


def test_the_first_run_tour_names_the_real_sections():
    """The tour used to advertise sections that no longer existed."""
    from spacr.qt.first_run import DEFAULT_TOUR

    step = next(s for s in DEFAULT_TOUR if "Sidebar" in s.title)
    for section in SECTIONS:
        assert section in step.body, (
            f"the sidebar tour step does not mention {section!r}")


def test_ctrl_number_shortcuts_still_reach_the_first_nine_apps(
        qtbot, qt_theme_applied):
    from spacr.qt import shortcuts
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    for idx in range(9):
        shortcuts._nav_by_index(win, idx)
        expected = APPS[idx][1]
        assert win._status_app_label.text() == expected, (
            f"Ctrl+{idx + 1} should open {expected}")


def test_menu_bar_lists_every_app(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    # Never hold a QMenu reference across statements — Qt owns it and
    # PySide will report it deleted (see _menu_labels in test_batch7).
    labels: set = set()
    for top in win.menuBar().actions():
        if top.text().replace("&", "") != "spaCR":
            continue
        for act in top.menu().actions():
            if not act.isSeparator():
                labels.add(act.text())
        break
    for _key, name, *_rest in APPS:
        assert name in labels, f"{name} missing from the spaCR menu"
