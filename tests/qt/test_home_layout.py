"""Home page: category sizes, and labels that fit the widget drawing them.

Two invariants live here, both of which were broken before item #16c:

1. **No category holds more than ``MAX_APPS_PER_SECTION`` apps.**
   "Tools" had grown to sixteen — a horizontal row nobody could read to
   the end of. The cap was nine until #16i raised it to thirteen; #16j
   deleted the staging sections that needed thirteen but kept the
   number, because a cap that is exactly the size of the biggest section
   fires on the next app added rather than when a row stops being
   readable.
2. **No app name is silently clipped.** ``AppTile`` paints its name in a
   child ``ElidingLabel`` of fixed width. A plain ``QLabel`` there does
   not elide — it stops painting — which is how "Annotator Agreement",
   "Database Browser", "Format Converter", "Model Compare" and "Cellpose
   Masks" lost their tails with no ellipsis and no warning.

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
    SECTION_CORE,
    SECTIONS,
    Sidebar,
    _icon_for_app,
    home_bands,
    make_home_page,
    section_members,
)
from spacr.qt.widgets.home import AppTile, HomePage


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
    """Apps per category *tab*, which is not apps per ``APPS`` section.

    Derived from ``section_members`` rather than by counting the fourth
    column. The two are the same thing again now that staging is a
    colour rather than a section, but the tab is what the cap is about,
    so the tab is what is counted.
    """
    return {s: len(section_members(s)) for s in SECTIONS}


@pytest.fixture(scope="module", params=["dark", "light"])
def home(request, qapp, qt_theme_applied):
    """A laid-out Home page + sidebar under one theme.

    Module-scoped so the (fairly expensive) page build happens twice for
    the whole file rather than once per parametrised app.

    The theme is applied to these two widgets rather than to the
    QApplication: a global ``setStyleSheet`` re-polishes every widget any
    other test left behind, which is both slow and a good way to crash
    the interpreter on a stale one.

    Module-scoped means it is BUILT before any function-scoped fixture of
    the test that first asks for it, and ``tests/conftest.py``'s autouse
    ``_isolated_qsettings_store`` is one of those: it swaps QSettings for
    an EMPTY per-test directory, and an empty store reads
    ``DEFAULT_FONT_SCALE`` (1.5), not the 1.0 the session fixture wrote.
    So the page is laid out at one zoom and measured at another unless a
    test says otherwise — the scale it was actually built at is recorded
    on the page as ``testBuildFontScale`` so a test that measures with
    ``scaled_px`` can put it back first.
    """
    from spacr.qt.preferences import get_font_scale
    from spacr.qt.theme import stylesheet

    theme = request.param
    qss = stylesheet(theme)

    page = make_home_page()  # the page MainWindow ships
    page.setProperty("testBuildFontScale", get_font_scale())
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


def _tiles_by_name(page: HomePage) -> dict:
    return {t.text_label: t for t in page.findChildren(AppTile)}


def _nav_buttons_by_name(bar: Sidebar) -> dict:
    """App navigation buttons keyed by app name (Home excluded)."""
    return {b.accessibleName(): b for b in bar.findChildren(QPushButton)
            if b.property("navKey") not in (None, "__home__")}


# ---------------------------------------------------------------------------
# Part B — categories
# ---------------------------------------------------------------------------

def test_no_category_holds_more_than_the_cap():
    """The invariant the whole regrouping exists to satisfy.

    Named for nine until #16i raised the cap to thirteen. The number
    lives in ``MAX_APPS_PER_SECTION``; the name no longer repeats it, so
    changing it again does not leave a test called
    ``..._more_than_nine_apps`` asserting something else."""
    oversized = {s: n for s, n in _counts().items()
                 if n > MAX_APPS_PER_SECTION}
    assert not oversized, (
        f"sections over the {MAX_APPS_PER_SECTION}-app cap: {oversized}. "
        "Split one out with a name that means something rather than "
        "letting a row grow past what anyone reads.")


def test_no_category_tab_is_empty():
    """A section with no members is a tab that opens on an empty pane.

    This used to be ``test_every_category_is_big_enough_to_deserve_a_
    heading``, which required three (a one- or two-entry section is
    noise, not navigation — cf. #12b). The floor that survives is the
    one that actually matters: a tab is its members, and zero members is
    nothing to open.
    """
    empty = sorted(s for s, n in _counts().items() if not n)
    assert not empty, (
        f"declared sections with no members: {empty}. Either file an app "
        "under them or drop the constant — an empty tab is worse than "
        "no tab.")


def test_every_app_is_on_exactly_one_subject_tab_and_one_home_band():
    """One tile per app on Home, one tab per app after it.

    While staging was a section this had to allow an app on two tabs —
    its subject and its stage. There is one grouping again, so both
    halves are back to a straight partition of the registry, and how
    finished an app is is asserted separately over ``APP_STAGE``.

    Registration is forced first, so the registry read here is the one a
    launched GUI has. Nine apps join it at ``spacr.qt.run()`` time rather than
    from ``app.py``, and every number below is a claim about the list the user
    is looking at. ``tests/qt/conftest.py::_restore_app_registry`` puts it back
    afterwards.
    """
    import spacr.qt
    from spacr.qt.app import app_stage

    spacr.qt.register_self_registering_modules()
    keys = [a[0] for a in APPS]

    subject = [k for s in SECTIONS for k, *_ in section_members(s)]
    assert sorted(subject) == sorted(keys), (
        "an app is on no category tab, or on two")

    banded = [k for _s, rows in home_bands() for k, *_ in rows]
    assert sorted(banded) == sorted(keys), (
        "an app is missing from Home, or drawn on it twice")

    staged = [k for k in keys if app_stage(k) != "stable"]
    # Forty-three, and it moved for two reasons at once, which is why it is
    # worth writing down. Apps kept arriving alpha — Pipeline Graph, Hit
    # List, Prediction Profiler and Methods & Results, then Control Charts,
    # Dose Response, Trellis, Gate Editor, Feature Explorer, Outliers,
    # Project Browser and the rest of the self-registering set. And the count
    # is now taken over the REGISTERED registry rather than the module-level
    # one, which is the list the user actually sees: the same expression read
    # 45 of 53 before registration and 42 of 62 after it, so the old number
    # was answering a question nobody asks. The count is the user's list — how
    # many of the apps in front of them carry a "not signed off" colour — and
    # it drops by one every time an app is signed off.
    #
    # Forty-three since 2026-08-06 (2d4da7df): the merged Classify module
    # registered itself STAGE_ALPHA on purpose, because "stable" is the
    # absence of a line in APP_STAGE and the merged screen has not been run
    # on real data. It is the only one that has ever moved this number UP
    # by arriving rather than by the registry being read differently.
    assert len(staged) == 43, (
        f"{len(staged)} apps staged, not 43 — if that is intended, say so "
        "here; the count is the user\'s list")


def test_sections_appear_in_the_declared_workflow_order():
    """APPS order drives both Home and the sidebar, so it is the order.

    Compared as a subsequence rather than as equality: a section that
    lost its last app would drop out of APPS and keep its constant, and
    that is not this test\'s complaint. What must not happen is APPS
    running the sections out of order, which would draw a heading twice.
    """
    order = _sections_in_order()
    assert order == [s for s in SECTIONS if s in set(order)]


def test_every_app_is_in_a_declared_section():
    stray = sorted({s for *_r, s in APPS} - set(SECTIONS))
    assert not stray, f"apps in undeclared sections: {stray}"


def test_the_core_pipeline_comes_first_and_is_unbroken():
    """Ctrl+1..9 map to APPS[0..8], so Core has to lead the table.

    Core is nine apps and they are APPS[0..8], so the nine Ctrl slots
    are exactly the Core pipeline again — #16i staged Timelapse and
    Motility Assay out of it and #16j put them back. The assertion is
    written to survive either: Core first, and contiguous. A Core app at
    APPS[20] would be unreachable by keyboard number and would also draw
    a second "Core" heading in the sidebar.
    """
    core = [a for a in APPS if a[3] == SECTION_CORE]
    assert core, "Core lost all its apps"
    assert [a[3] for a in APPS[:len(core)]] == [SECTION_CORE] * len(core)
    assert SECTIONS[0] == SECTION_CORE


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
    clipped = [t.text_label for t in page.findChildren(AppTile)
               if t.is_name_elided()]
    assert not clipped, f"Home had to shorten {clipped}"


def test_a_pathologically_long_name_elides_instead_of_clipping(
        qtbot, qt_theme_applied):
    """The safety net: a name no tile could ever fit still stays readable."""
    from spacr.qt.app import SECTION_CORE

    long_name = "Extremely Long Hypothetical Module Name For Testing"
    page = HomePage([("mask", long_name, "desc", SECTION_CORE)],
                    _icon_for_app)
    qtbot.addWidget(page)
    page.resize(1400, 600)
    page.show()
    qtbot.waitExposed(page)

    # The tile on the CURRENT tab. Every app is drawn twice — once on
    # Home and once on its category tab — and the one on the tab that is
    # not showing has never been laid out, so it has nothing to elide
    # against.
    tile = next(t for t in page.findChildren(AppTile) if t.isVisible())
    label = tile.name_label
    assert label.is_elided(), "expected the label to elide"
    assert "…" in label.text()
    assert label.text() != long_name
    assert label.full_text() == long_name
    assert label.toolTip() == long_name
    # And the tile did not stretch across the whole page to fit it.
    # ``TILE_MAX_W`` is the cap that stops a band of one drawing one tile
    # the width of the pane.
    from spacr.qt.preferences import scaled_px
    assert tile.width() <= scaled_px(HomePage.TILE_MAX_W)


def test_nothing_clips_at_a_150_percent_font_scale(qtbot, qapp, monkeypatch):
    """The accessibility setting most likely to break a fixed width.

    Both the stylesheet and the px constants scale, so the tiles have to
    grow with the text rather than keep a 100 %-sized box.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.theme import stylesheet

    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.5)
    qss = stylesheet("dark", 1.5)

    page = make_home_page()  # the page MainWindow ships
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
    for tile in page.findChildren(AppTile):
        label = tile.name_label
        needed = QFontMetrics(label.font()).horizontalAdvance(tile.text_label)
        if needed > label.available_text_width() and not label.is_elided():
            clipped.append(tile.text_label)
    assert not clipped, f"clipped at 150 % font scale: {clipped}"
    assert not bar.clipped_items()
    # The column itself grew with the font rather than staying at 220.
    assert bar.width() == prefs.scaled_px(Sidebar.WIDTH_MIN)


def test_the_widest_name_fits_the_tile_the_grid_gives_it(home):
    """Was ``test_tile_width_tracks_the_name_it_has_to_draw``.

    That version asserted ``HTile.required_width()``, a per-tile
    measurement that made sense when each section was a horizontal
    scroller and every tile sized itself. Every tile is now the same
    size, laid out in a uniform grid, so "wider name, wider tile" is not
    the contract any more — what is, is that the *widest* name still
    fits the one size they all share.

    Measured at the zoom the page was BUILT at. It used to measure at
    whatever ``get_font_scale()`` happened to answer inside the test, and
    once ``DEFAULT_FONT_SCALE`` became 1.5 that stopped being the same
    number: the module-scoped ``home`` fixture builds before the autouse
    per-test QSettings store arrives, so the tiles were laid out at 1.0
    and the ``floor``/``cap`` around them computed at 1.5. Restoring the
    build scale here — the autouse ``_restore_font_scale`` puts it back —
    makes the window and the thing in it agree, which is the only way
    this assertion measures the product rather than the fixture order.
    """
    from spacr.qt import preferences as prefs
    _theme, page, _bar = home
    prefs.set_font_scale(page.property("testBuildFontScale"))
    scaled_px = prefs.scaled_px
    tiles = _tiles_by_name(page)
    floor, cap = scaled_px(HomePage.TILE_MIN_W), scaled_px(HomePage.TILE_MAX_W)
    visible = [n for n, t in tiles.items() if t.isVisible()]
    # A guard on the loop below, not decoration: every tile being hidden
    # would pass it without measuring anything at all.
    assert visible, "no tile on the current tab was laid out"
    for name, tile in tiles.items():
        if not tile.isVisible():
            continue        # a tile on a tab that is not the current one
        assert floor <= tile.width() <= cap, (
            f"{name} is {tile.width()} px wide, outside {floor}..{cap}")
    for name in ("Mask", "Annotator Agreement", "Format Converter"):
        label = tiles[name].name_label
        needed = QFontMetrics(label.font()).horizontalAdvance(name)
        assert needed <= label.available_text_width(), (
            f"{name!r} needs {needed} px and the tile gives it "
            f"{label.available_text_width()} px")


# ---------------------------------------------------------------------------
# Everything that iterates APPS still sees every app
# ---------------------------------------------------------------------------

def test_home_renders_every_app_under_every_band_heading(home):
    """Every app once on Home, under a heading for every non-empty band.

    Was ``..._under_every_section_heading``, asserting a heading for
    every entry in SECTIONS. Three of those now have no apps filed under
    them, so Home draws no band for them — deliberately: a heading with
    nothing under it is worse than no heading. They keep their tabs,
    which is what ``test_no_category_tab_is_empty`` covers.
    """
    _theme, page, _bar = home
    rendered = set(_tiles_by_name(page))
    assert rendered == {a[1] for a in APPS}

    headings = {lbl.text() for lbl in page.findChildren(QLabel)}
    for section, _rows in home_bands():
        assert section.upper() in headings, (
            f"no '{section}' band heading on Home")


def test_sidebar_renders_every_app_under_every_section_heading(home):
    """Was ``headings == set(SECTIONS)``.

    The sidebar walks ``APPS`` and heads each run of filed sections, so
    it shows the four that apps are actually filed under — not the three
    that only exist as subject tabs.
    """
    _theme, _page, bar = home
    assert set(_nav_buttons_by_name(bar)) == {a[1] for a in APPS}

    headings = {lbl.text() for lbl in bar.findChildren(QLabel)
                if lbl.objectName() == "SidebarSection"}
    assert headings == set(_sections_in_order())


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
    # Only the sections apps are FILED under get a badge; the three that
    # exist as subject tabs alone are reachable by search, below.
    badges = {c.section for c in palette._commands}
    for section in _sections_in_order():
        assert f"Apps · {section}" in badges


@pytest.mark.parametrize("section", SECTIONS)
def test_command_palette_filters_by_the_section_names(
        qtbot, qt_theme_applied, section):
    """Section names are searchable keywords — typing "Data" finds them.

    Was pinned to SECTION_DATA and to ``a[3] == section``. #16i files
    the six Data apps under "Alpha modules", so matching on the fourth
    column alone made this a search for a name nothing carries — it
    would have passed vacuously while typing "Data" found nothing at
    all. Parametrised over SECTIONS and asserting the expectation is
    non-empty, so a category that stops being findable fails here.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt.command_palette import CommandPalette

    win = MainWindow()
    qtbot.addWidget(win)
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    palette._on_filter(section)
    visible = [palette._list.item(i).text()
               for i in range(palette._list.count())]
    expected = {row[1] for row in section_members(section)}
    assert expected, f"{section} has no apps to find"
    for name in expected:
        assert any(name in v for v in visible), (
            f"{name} not found when filtering on {section!r}")


def test_the_first_run_tour_names_the_real_sections():
    """The tour used to advertise sections that no longer existed.

    Was ``for section in SECTIONS``. The step describes the SIDEBAR, and
    the sidebar heads the sections apps are filed under — naming the
    three subject-only tabs there would send a first-time user looking
    down the sidebar for a "Data" heading that is not in it.
    """
    from spacr.qt.first_run import DEFAULT_TOUR

    step = next(s for s in DEFAULT_TOUR if "Sidebar" in s.title)
    for section in _sections_in_order():
        assert section in step.body, (
            f"the sidebar tour step does not mention {section!r}")
    for absent in set(SECTIONS) - set(_sections_in_order()):
        assert absent not in step.body, (
            f"the sidebar tour names {absent!r}, which has no sidebar "
            "heading")


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
