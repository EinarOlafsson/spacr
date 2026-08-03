"""The registration seams: `register_app` and `register_widget_qss`.

Item 0.1. Seven new modules and a pile of new widgets are being built in
parallel, and every one of them used to have to edit `spacr/qt/app.py`
(a row in the `APPS` literal, a branch in `_build_screen`) and
`spacr/qt/theme.py` (a block inside one 1100-line f-string). Six
workstreams editing three files is six merge conflicts in code nobody
can review line by line.

These tests are the contract those modules will be written against, so
they exercise the real objects the running app builds -- the shipped
`HomePage`, the shipped `Sidebar`, the shipped `MainWindow._build_screen`
and the QSS a `QApplication` is actually given -- rather than asserting
that a dict got a key.

Two properties are load-bearing beyond "the seam works":

* **Nothing registered means nothing changed.** The stylesheet with an
  empty widget registry must be the exact string it was before the seam
  existed, and `APPS` must be exactly the built-in table.
* **Order.** `APPS` is walked to draw the sidebar, and a heading is
  emitted every time the section changes, so a row filed at the end
  rather than beside its own section draws that section's heading twice.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QWidget

from spacr.qt import app as app_mod
from spacr.qt import theme as theme_mod


# ---------------------------------------------------------------------------
# Fixtures — a registration must never outlive the test that made it
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    Ctrl+N binding for every test that runs afterwards, so this restores
    the list object in place rather than trusting `unregister_app`.
    """
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    yield
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod._refresh_sections()


@pytest.fixture
def qss_sandbox():
    """Run with an EMPTY widget-QSS registry, and restore it afterwards.

    Emptied on the way in, not just restored on the way out: real widgets
    register blocks at import time now (``spacr.qt.widgets.field_fade``
    is the first), so whether a given block is present here depends on
    which test module ran before this one. These tests are about the seam
    itself and have to start from nothing to mean anything.
    """
    saved = dict(theme_mod._WIDGET_QSS)
    theme_mod._WIDGET_QSS.clear()
    yield
    theme_mod._WIDGET_QSS.clear()
    theme_mod._WIDGET_QSS.update(saved)


# ---------------------------------------------------------------------------
# Part A — the app registry
# ---------------------------------------------------------------------------

def test_the_built_in_table_is_registered_through_the_same_door():
    """`APPS` is built by `register_app`, not written down.

    The 34 built-ins go through the public function on every import, so
    an ordering or validation bug in it fails at import rather than the
    first time somebody adds the 35th app. This asserts the result is
    exactly the declared table, in the declared order -- i.e. that
    routing them through the seam changed nothing.
    """
    declared = [tuple(row) for row in app_mod._BUILTIN_APPS]
    # Plugins may add rows in a contributor's checkout; the built-ins
    # must still be present, in order, and be the whole list here.
    registered = [row for row in app_mod.APPS if row in set(declared)]
    assert registered == declared


def test_sections_are_the_ones_that_have_apps_not_the_ones_declared():
    """`SECTION_ORDER` declares seven; `SECTIONS` publishes the live five.

    An empty section is a tab that opens on an empty pane, which
    ``test_no_category_tab_is_empty`` forbids -- so a new section is
    named today and appears the day its first app registers. That is
    what lets a module claim one without editing app.py.
    """
    assert app_mod.SECTION_EXPLORE in app_mod.SECTION_ORDER
    assert app_mod.SECTION_DESIGN in app_mod.SECTION_ORDER
    assert app_mod.SECTION_EXPLORE not in app_mod.SECTIONS
    assert app_mod.SECTION_DESIGN not in app_mod.SECTIONS
    assert set(app_mod.SECTIONS) == {row[3] for row in app_mod.APPS}
    # Every declared section has its note written now, so the first app
    # to claim one gets a described tab rather than a bare heading.
    assert set(app_mod._SECTION_NOTE_LIBRARY) == set(app_mod.SECTION_ORDER)
    assert all(app_mod._SECTION_NOTE_LIBRARY.values())
    # ...and the published notes track the published sections exactly.
    assert set(app_mod.SECTION_NOTES) == set(app_mod.SECTIONS)


def test_a_registered_app_reaches_every_reader_of_the_registry(
        registry_sandbox):
    """One call, and the app is in all six derived views."""
    row = app_mod.register_app(
        "seam_probe", "Seam Probe", "A registered app, for the test",
        app_mod.SECTION_RESULTS, stage=app_mod.STAGE_ALPHA)

    assert row == ("seam_probe", "Seam Probe",
                   "A registered app, for the test", app_mod.SECTION_RESULTS)
    assert row in app_mod.APPS
    assert row in app_mod.visible_apps()
    assert app_mod.app_stage("seam_probe") == app_mod.STAGE_ALPHA
    assert app_mod.home_stages()["seam_probe"] == app_mod.STAGE_ALPHA
    assert row in app_mod.section_members(app_mod.SECTION_RESULTS)
    categories = dict(app_mod.home_categories())
    assert "seam_probe" in categories[app_mod.SECTION_RESULTS]
    bands = {s: [r[0] for r in rows] for s, rows in app_mod.home_bands()}
    assert "seam_probe" in bands[app_mod.SECTION_RESULTS]


def test_a_registered_app_is_drawn_on_home_and_in_the_sidebar(
        qtbot, qt_theme_applied, registry_sandbox):
    """The shipped widgets, not a stand-in: a tile and a nav row appear."""
    from spacr.qt.widgets.home import AppTile

    before = app_mod.make_home_page()
    qtbot.addWidget(before)
    assert "Seam Probe" not in {t.text_label
                                for t in before.findChildren(AppTile)}

    app_mod.register_app("seam_probe", "Seam Probe",
                         "A registered app, for the test",
                         app_mod.SECTION_RESULTS)

    page = app_mod.make_home_page()
    qtbot.addWidget(page)
    tiles = {t.text_label: t for t in page.findChildren(AppTile)}
    assert "Seam Probe" in tiles
    assert len(tiles) == len(app_mod.APPS)

    bar = app_mod.Sidebar()
    qtbot.addWidget(bar)
    keys = {b.property("navKey")
            for b in bar.findChildren(QPushButton)}
    assert "seam_probe" in keys
    # One heading per section, still: the row was filed beside its own
    # section rather than appended after everything.
    headings = [lbl.text() for lbl in bar.findChildren(QLabel)
                if lbl.objectName() == "SidebarSection"]
    assert headings == list(app_mod.SECTIONS)
    assert len(headings) == len(set(headings))


def test_claiming_an_empty_section_makes_it_appear_with_its_note(
        qtbot, qt_theme_applied, registry_sandbox):
    """The first Explore app is what gives Explore a tab.

    This is the whole point of the two new sections: seven new modules
    would take Results & QC past `MAX_APPS_PER_SECTION`, and the fix is
    a section with a name that means something -- registered into, not
    edited in.
    """
    assert app_mod.SECTION_EXPLORE not in app_mod.SECTIONS

    app_mod.register_app("graph_builder_probe", "Graph Builder",
                         "Drag columns onto x / y / colour / facet",
                         app_mod.SECTION_EXPLORE)

    assert app_mod.SECTION_EXPLORE in app_mod.SECTIONS
    assert (app_mod.SECTION_NOTES[app_mod.SECTION_EXPLORE]
            == app_mod._SECTION_NOTE_LIBRARY[app_mod.SECTION_EXPLORE])
    assert set(app_mod.SECTION_NOTES) == set(app_mod.SECTIONS)

    page = app_mod.make_home_page()
    qtbot.addWidget(page)
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert any(lbl.startswith(app_mod.SECTION_EXPLORE) for lbl in labels)
    # The tab carries its own note, which is why the note is written when
    # the section is named rather than when it fills up.
    assert any(app_mod._SECTION_NOTE_LIBRARY[app_mod.SECTION_EXPLORE]
               in lbl.text()
               for lbl in page.findChildren(QLabel))


def test_a_new_row_is_filed_beside_its_own_section_not_appended(
        registry_sandbox):
    """`APPS` order is section order, because the sidebar assumes it is.

    Explore sits between Results & QC and Toxoplasma in
    `SECTION_ORDER`, so an Explore app registered last still lands
    there. Appending would draw the Results heading, the Toxo heading,
    and then an Explore heading after the Toxo apps.
    """
    app_mod.register_app("explore_probe", "Explore Probe", "…",
                         app_mod.SECTION_EXPLORE)
    app_mod.register_app("core_probe", "Core Probe", "…",
                         app_mod.SECTION_CORE)

    order = []
    for _key, _name, _desc, section in app_mod.APPS:
        if not order or order[-1] != section:
            order.append(section)
    assert order == [s for s in app_mod.SECTION_ORDER if s in set(order)]
    assert len(order) == len(set(order)), "a section is split in two"

    keys = [row[0] for row in app_mod.APPS]
    # Core rows stay first — Ctrl+1..9 index straight into APPS[0..8].
    assert keys.index("core_probe") < keys.index("align")
    assert keys.index("explore_probe") > keys.index("report")
    assert keys.index("explore_probe") < keys.index("analyze_plaques")


@pytest.mark.parametrize("kwargs, exc, needle", [
    (dict(key="mask", name="X", desc="Y", section=app_mod.SECTION_CORE),
     ValueError, "already registered"),
    (dict(key="probe", name="X", desc="Y", section="Nowhere"),
     ValueError, "unknown section"),
    (dict(key="", name="X", desc="Y", section=app_mod.SECTION_CORE),
     ValueError, "needs a key"),
    (dict(key="probe", name="  ", desc="Y", section=app_mod.SECTION_CORE),
     ValueError, "needs a name"),
    (dict(key="probe", name="X", desc="", section=app_mod.SECTION_CORE),
     ValueError, "needs a name"),
])
def test_a_bad_registration_is_refused_and_changes_nothing(
        kwargs, exc, needle, registry_sandbox):
    before = list(app_mod.APPS)
    with pytest.raises(exc, match=needle):
        app_mod.register_app(**kwargs)
    assert app_mod.APPS == before


def test_an_unknown_stage_or_uncallable_factory_is_refused(registry_sandbox):
    with pytest.raises(ValueError, match="unknown stage"):
        app_mod.register_app("probe", "X", "Y", app_mod.SECTION_CORE,
                             stage="nearly-done")
    with pytest.raises(TypeError, match="not callable"):
        app_mod.register_app("probe", "X", "Y", app_mod.SECTION_CORE,
                             factory="spacr.qt.screens.mine:Screen")
    assert not any(row[0] == "probe" for row in app_mod.APPS)


def test_going_over_the_cap_warns_but_still_starts(registry_sandbox, caplog):
    """The cap is a design rule. Refusing to start would not help fix it."""
    room = app_mod.MAX_APPS_PER_SECTION - len(
        app_mod.section_members(app_mod.SECTION_MODELS))
    for i in range(room):
        app_mod.register_app(f"filler_{i}", f"Filler {i}", "…",
                             app_mod.SECTION_MODELS)
    with caplog.at_level(logging.WARNING, logger=app_mod.LOG.name):
        app_mod.register_app("one_too_many", "One Too Many", "…",
                             app_mod.SECTION_MODELS)
    assert any("over the 13 cap" in rec.getMessage()
               for rec in caplog.records)
    assert any(row[0] == "one_too_many" for row in app_mod.APPS)


def test_unregister_puts_everything_back(registry_sandbox):
    before_apps = list(app_mod.APPS)
    before_sections = app_mod.SECTIONS
    before_notes = dict(app_mod.SECTION_NOTES)

    app_mod.register_app("design_probe", "Design Probe", "…",
                         app_mod.SECTION_DESIGN,
                         factory=QWidget, stage=app_mod.STAGE_BETA)
    assert app_mod.SECTION_DESIGN in app_mod.SECTIONS

    assert app_mod.unregister_app("design_probe") is True
    assert app_mod.unregister_app("design_probe") is False
    assert app_mod.APPS == before_apps
    assert app_mod.SECTIONS == before_sections
    assert app_mod.SECTION_NOTES == before_notes
    assert app_mod.registered_factory("design_probe") is None
    assert "design_probe" not in app_mod.APP_STAGE


# ---------------------------------------------------------------------------
# Part B — the screen factory
# ---------------------------------------------------------------------------

class _Host:
    """Stand-in `self` for the unbound `MainWindow._build_screen`.

    Same trick as `test_all_module_smoke._FactoryHost`: anything
    `MainWindow` itself defines is stubbed as a no-op (the built-in
    chain hands `self._on_*` to `connect()`), and anything it does not
    define still raises -- that is a `connect()` to a slot that does not
    exist, which the shipped window would hit just as hard.
    """

    def __getattr__(self, name):
        from spacr.qt.app import MainWindow
        if callable(vars(MainWindow).get(name)):
            return lambda *_a, **_kw: None
        raise AttributeError(
            f"MainWindow._build_screen wired {name!r}, but MainWindow does "
            f"not define it")


def test_a_registered_factory_builds_the_screen(qtbot, registry_sandbox):
    """`_build_screen` consults the registry before its built-in chain."""
    from spacr.qt.app import MainWindow

    made = []

    def factory():
        widget = QFrame()
        widget.setObjectName("SeamScreen")
        made.append(widget)
        return widget

    app_mod.register_app("factory_probe", "Factory Probe", "…",
                         app_mod.SECTION_RESULTS, factory=factory)
    assert app_mod.registered_factory("factory_probe") is factory

    screen = MainWindow._build_screen(_Host(), "factory_probe")
    qtbot.addWidget(screen)
    assert screen is made[0]
    assert screen.objectName() == "SeamScreen"


def test_a_factory_is_given_the_arguments_it_declares(qtbot,
                                                      registry_sandbox):
    """Take-what-you-need, resolved by signature, never by retrying."""
    seen = {}

    def wants_both(app_key, host):
        seen["app_key"] = app_key
        seen["host"] = host
        return QWidget()

    def wants_kwargs(**kw):
        seen.update(kw)
        return QWidget()

    from spacr.qt.app import MainWindow

    host = _Host()
    app_mod.register_app("both_probe", "Both", "…", app_mod.SECTION_RESULTS,
                         factory=wants_both)
    qtbot.addWidget(MainWindow._build_screen(host, "both_probe"))
    assert seen == {"app_key": "both_probe", "host": host}

    seen.clear()
    app_mod.register_app("kwargs_probe", "Kwargs", "…",
                         app_mod.SECTION_RESULTS, factory=wants_kwargs)
    qtbot.addWidget(MainWindow._build_screen(host, "kwargs_probe"))
    assert seen == {"app_key": "kwargs_probe", "host": host}


def test_a_factory_is_called_once_even_when_it_raises_a_type_error(
        registry_sandbox):
    """The reason the arguments are resolved by inspection.

    "Call it, and retry with no arguments on TypeError" cannot tell a
    wrong call from a TypeError raised *inside* a factory that was
    called correctly -- and would build half the screen twice.
    """
    from spacr.qt.app import MainWindow

    calls = []

    def explodes():
        calls.append(1)
        raise TypeError("something inside the screen went wrong")

    app_mod.register_app("boom_probe", "Boom", "…", app_mod.SECTION_RESULTS,
                         factory=explodes)
    with pytest.raises(TypeError, match="inside the screen"):
        MainWindow._build_screen(_Host(), "boom_probe")
    assert calls == [1]


def test_a_factory_that_returns_a_non_widget_is_refused(registry_sandbox):
    from spacr.qt.app import MainWindow

    app_mod.register_app("bad_probe", "Bad", "…", app_mod.SECTION_RESULTS,
                         factory=lambda: {"not": "a widget"})
    with pytest.raises(TypeError, match="expected QWidget"):
        MainWindow._build_screen(_Host(), "bad_probe")


def test_an_app_without_a_factory_still_gets_the_generic_screen(
        qtbot, qt_theme_applied, registry_sandbox):
    """Omitting `factory` is the normal case, not a broken one."""
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.app_screen import AppScreen

    app_mod.register_app("plain_probe", "Plain Probe", "…",
                         app_mod.SECTION_RESULTS)
    screen = MainWindow._build_screen(_Host(), "plain_probe")
    qtbot.addWidget(screen)
    assert isinstance(screen, AppScreen)
    assert screen.app_key == "plain_probe"


# ---------------------------------------------------------------------------
# Part C — the widget QSS seam
# ---------------------------------------------------------------------------

def test_nothing_registered_is_byte_identical_to_no_seam_at_all(qss_sandbox):
    """The property that makes this change safe to ship."""
    baseline = {t: theme_mod.stylesheet(t, 1.25, None, 0.4)
                for t in theme_mod.THEMES}
    theme_mod.register_widget_qss(
        "SeamProbe", lambda palette, opacity: "QFrame#SeamProbe { border: 0; }")
    assert theme_mod.unregister_widget_qss("SeamProbe") is True
    assert {t: theme_mod.stylesheet(t, 1.25, None, 0.4)
            for t in theme_mod.THEMES} == baseline
    assert theme_mod.unregister_widget_qss("SeamProbe") is False


def test_a_registered_block_lands_at_the_end_of_the_stylesheet(qss_sandbox):
    """Last, so a widget's own rule wins a specificity tie."""
    rule = "QFrame#SeamProbe { background: #123456; }"
    theme_mod.register_widget_qss("SeamProbe", lambda p, o: rule)

    qss = theme_mod.stylesheet("glass")
    assert rule in qss
    assert "SeamProbe" in theme_mod.widget_qss_names()
    # After the glass material layer, which is itself the last built-in.
    assert qss.index("Glass material layer") < qss.index(rule)
    assert qss.index("QStatusBar") < qss.index(rule)


def test_the_callback_gets_the_palette_the_built_in_rules_use(qss_sandbox):
    """Surfaces already carry page opacity; theme + font scale ride along."""
    captured = {}

    def block(palette, opacity):
        captured["palette"] = dict(palette)
        captured["opacity"] = opacity
        return ""

    theme_mod.register_widget_qss("SeamProbe", block)
    theme_mod.stylesheet("cell", 1.5, None, 0.3)

    palette = captured["palette"]
    assert captured["opacity"] == 0.3
    assert palette["theme"] == "cell"
    assert palette["font_scale"] == 1.5
    # Every colour role is still there, and the three user-dimmable
    # surfaces arrive already rendered through the opacity preference —
    # the same values the built-in rules interpolate.
    base = theme_mod.palette_for("cell")
    assert set(base) <= set(palette)
    for role in ("surface", "surface_alt", "surface_hi"):
        assert palette[role] == theme_mod.css_color(
            base[role], theme_mod.panel_alpha("cell", role, 0.3))
    assert palette["fg"] == base["fg"]


def test_none_opacity_reaches_the_callback_as_none(qss_sandbox):
    """`None` is "use the theme's designed scrim", which is not 1.0."""
    seen = []
    theme_mod.register_widget_qss(
        "SeamProbe", lambda p, o: seen.append(o) or "")
    theme_mod.stylesheet("cell")
    assert seen == [None]
    assert theme_mod.pane_alpha("cell", None) != theme_mod.pane_alpha(
        "cell", 0.0)


def test_a_block_that_uses_pane_surface_follows_the_opacity_slider(
        qss_sandbox):
    """The seam is only useful if a contributed widget can be dialled."""
    def block(palette, opacity):
        colour = theme_mod.pane_surface("surface_alt", palette["theme"],
                                        opacity)
        return f"QFrame#SeamProbe {{ background: {colour}; }}"

    theme_mod.register_widget_qss("SeamProbe", block)

    opaque = theme_mod.stylesheet("cell", 1.0, None, 1.0)
    sheer = theme_mod.stylesheet("cell", 1.0, None, 0.15)
    assert (f"background: {theme_mod.pane_surface('surface_alt', 'cell', 1.0)}"
            in opaque)
    assert (f"background: {theme_mod.pane_surface('surface_alt', 'cell', 0.15)}"
            in sheer)
    assert opaque != sheer
    # And the floor is still the theme's to apply, not the widget's.
    floored = theme_mod.stylesheet("cell", 1.0, None, 0.0)
    assert (f"background: {theme_mod.pane_surface('surface_alt', 'cell', 0.0)}"
            in floored)


def test_a_broken_block_costs_its_own_widget_and_nothing_else(
        qss_sandbox, caplog):
    """An exception here would leave the whole app unstyled."""
    def explodes(palette, opacity):
        raise RuntimeError("typo in the contributed QSS")

    theme_mod.register_widget_qss("Broken", explodes)
    theme_mod.register_widget_qss(
        "Fine", lambda p, o: "QFrame#Fine { border: 1px solid red; }")

    with caplog.at_level(logging.ERROR, logger=theme_mod.LOG.name):
        qss = theme_mod.stylesheet("dark")
    assert "QFrame#Fine" in qss
    assert "QMenuBar" in qss and "QStatusBar" in qss
    assert any("Broken" in rec.getMessage() for rec in caplog.records)


def test_a_block_that_is_not_a_string_is_dropped_and_logged(qss_sandbox,
                                                            caplog):
    theme_mod.register_widget_qss("NotAString", lambda p, o: {"a": 1})
    with caplog.at_level(logging.ERROR, logger=theme_mod.LOG.name):
        qss = theme_mod.stylesheet("dark")
    assert "NotAString" not in qss
    assert any("expected str" in rec.getMessage() for rec in caplog.records)


def test_two_widgets_cannot_quietly_claim_one_name(qss_sandbox):
    theme_mod.register_widget_qss("SeamProbe", lambda p, o: "")
    with pytest.raises(ValueError, match="already registered"):
        theme_mod.register_widget_qss("SeamProbe", lambda p, o: "")
    theme_mod.register_widget_qss("SeamProbe", lambda p, o: "QFrame {}",
                                  replace=True)
    assert theme_mod.widget_qss_names() == ("SeamProbe",)
    with pytest.raises(TypeError, match="not callable"):
        theme_mod.register_widget_qss("NotCallable", "QFrame {}")


def test_a_registered_block_actually_paints_the_widget(qtbot, qapp,
                                                       qss_sandbox):
    """End to end: register, apply, and read the pixel back.

    Everything above checks the string. This checks that the string is
    QSS Qt honours, on a real widget, through the real
    `QApplication.setStyleSheet` the app calls.
    """
    theme_mod.register_widget_qss(
        "SeamPaint",
        lambda p, o: "QFrame#SeamPaint { background-color: #ff00ff; }")
    previous = qapp.styleSheet()
    try:
        qapp.setStyleSheet(theme_mod.stylesheet("dark"))
        frame = QFrame()
        frame.setObjectName("SeamPaint")
        qtbot.addWidget(frame)
        frame.resize(40, 40)
        frame.show()
        qapp.processEvents()
        pixel = frame.grab().toImage().pixelColor(20, 20)
        assert (pixel.red(), pixel.green(), pixel.blue()) == (255, 0, 255)
    finally:
        qapp.setStyleSheet(previous)
