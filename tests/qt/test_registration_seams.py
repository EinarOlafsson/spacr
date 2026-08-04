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

#: A section that exists only while a `registry_sandbox` test is running.
#:
#: Two of the tests below have to make a section APPEAR FROM NOTHING --
#: that is the property, not a detail of how they are written -- and for
#: that they need a declared section no app has claimed. They used Design,
#: which was the only one left, and Power / Design has now claimed it.
#:
#: Borrowing the last unclaimed real section was always the bug: it made
#: the proof's subject something another workstream could switch on, and
#: it did. This one cannot be claimed, because it is not in the shipped
#: `SECTION_ORDER` at all -- the fixture puts it there for the length of
#: one test and takes it out again. The property it pins is the
#: `SECTIONS`-rebinding bug fixed in `ea43a35c`, which is worth keeping a
#: subject for.
SANDBOX_SECTION = "Sandbox"
SANDBOX_NOTE = ("Declared by the test fixture, claimed by nothing until a "
                "test claims it.")


@pytest.fixture
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    Ctrl+N binding for every test that runs afterwards, so this restores
    the list object in place rather than trusting `unregister_app`.

    It also DECLARES :data:`SANDBOX_SECTION` for the length of the test,
    with a note, so a test that needs an unclaimed section has one that no
    shipped app can take away. `SECTION_ORDER` is rebound rather than
    mutated because it is a tuple; every function that reads it
    (`register_app`, `_insert_position`, `_refresh_sections`) reads it as a
    module global at call time, so the rebinding is visible where it has to
    be. `_SECTION_NOTE_LIBRARY` is a dict and is mutated in place, because
    `_refresh_sections` indexes it and a rebinding would strand the copy
    app.py holds.
    """
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    order = app_mod.SECTION_ORDER
    notes = dict(app_mod._SECTION_NOTE_LIBRARY)
    app_mod.SECTION_ORDER = tuple(order) + (SANDBOX_SECTION,)
    app_mod._SECTION_NOTE_LIBRARY[SANDBOX_SECTION] = SANDBOX_NOTE
    yield
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.SECTION_ORDER = order
    app_mod._SECTION_NOTE_LIBRARY.clear()
    app_mod._SECTION_NOTE_LIBRARY.update(notes)
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
    """`SECTIONS` is DERIVED from `APPS`, never declared beside it.

    An empty section is a tab that opens on an empty pane, which
    ``test_no_category_tab_is_empty`` forbids -- so a new section is
    named today and appears the day its first app registers. That is
    what lets a module claim one without editing app.py.

    Both of the sections that were declared and empty when this was
    written have since been claimed from their own modules, which is the
    mechanism working rather than the property lapsing: Explore by Layer
    Viewer and Graph Builder, Design by Power / Design. All seven are
    live, so the "appears from nothing" half of the property no longer
    has a shipped subject and is pinned instead by
    ``test_an_importer_of_sections_cannot_hold_a_stale_snapshot`` and
    ``test_claiming_an_empty_section_makes_it_appear_with_its_note`` on
    :data:`SANDBOX_SECTION`, which the fixture declares and no app can
    ever claim. What is asserted here is the steady state: the published
    list is exactly the sections `APPS` uses, in declared order.
    """
    assert app_mod.SECTION_EXPLORE in app_mod.SECTION_ORDER
    assert app_mod.SECTION_DESIGN in app_mod.SECTION_ORDER
    assert app_mod.SECTION_EXPLORE in app_mod.SECTIONS
    assert app_mod.SECTION_DESIGN in app_mod.SECTIONS
    assert set(app_mod.SECTIONS) == {row[3] for row in app_mod.APPS}
    # Derived, not a copy of the declaration that happens to match: the
    # order is SECTION_ORDER's and the membership is APPS'.
    assert list(app_mod.SECTIONS) == [
        section for section in app_mod.SECTION_ORDER
        if any(row[3] == section for row in app_mod.APPS)]
    # Every declared section has its note written now, so the first app
    # to claim one gets a described tab rather than a bare heading.
    assert set(app_mod._SECTION_NOTE_LIBRARY) == set(app_mod.SECTION_ORDER)
    assert all(app_mod._SECTION_NOTE_LIBRARY.values())
    # ...and the published notes track the published sections exactly.
    assert set(app_mod.SECTION_NOTES) == set(app_mod.SECTIONS)
    # Published as a tuple for this name's whole life, so it still has to
    # read as one — the container changed, the meaning did not.
    assert app_mod.SECTIONS == tuple(app_mod.SECTIONS)
    assert app_mod.SECTIONS == list(app_mod.SECTIONS)
    assert not (app_mod.SECTIONS != tuple(app_mod.SECTIONS))


def test_an_importer_of_sections_cannot_hold_a_stale_snapshot(
        registry_sandbox):
    """`from spacr.qt.app import SECTIONS` binds the object, not a copy.

    A module that registers the first app of a new section does so
    AFTER app.py has finished importing -- by definition, since it has
    to import app.py to reach `register_app`. While SECTIONS was a tuple
    that `_refresh_sections` rebound, every module that had already read
    the name kept a snapshot from before that registration and never saw
    the new section appear. Graph Builder hit exactly this.

    The subject is :data:`SANDBOX_SECTION`, declared by the fixture for
    the length of this test. It used to be Design, on the grounds that
    Design was the one section nothing had claimed -- which made this
    proof's subject something any workstream could take away by shipping
    an app, and Power / Design duly did. A section the fixture invents
    cannot be claimed out from under it.
    """
    from spacr.qt.app import SECTIONS as imported_earlier

    assert imported_earlier is app_mod.SECTIONS
    assert SANDBOX_SECTION in app_mod.SECTION_ORDER
    assert SANDBOX_SECTION not in imported_earlier

    app_mod.register_app("stale_probe", "Stale Probe", "…",
                         SANDBOX_SECTION)

    # The name that was imported BEFORE the registration sees it.
    assert SANDBOX_SECTION in imported_earlier
    assert list(imported_earlier) == list(app_mod.SECTIONS)

    app_mod.unregister_app("stale_probe")
    assert SANDBOX_SECTION not in imported_earlier


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
    """The first app of an empty section is what gives it a tab.

    This is the whole point of the two extra sections: seven new modules
    would take Results & QC past `MAX_APPS_PER_SECTION`, and the fix is
    a section with a name that means something -- registered into, not
    edited in. Explore was claimed that way by Layer Viewer and Graph
    Builder, and Design by Power / Design, which is the mechanism working
    twice over.

    It leaves this test without a shipped empty section to demonstrate on,
    so it demonstrates on :data:`SANDBOX_SECTION` -- declared by the
    fixture, noted by the fixture, claimed by nothing. Written against a
    section the fixture owns, the test says what it always meant to say
    ("a section appears when its first app registers") instead of
    depending on which real section happened not to have shipped yet.
    """
    empty = SANDBOX_SECTION
    assert empty not in app_mod.SECTIONS

    app_mod.register_app("power_probe", "Power Calculator",
                         "How many wells this effect size needs", empty)

    assert empty in app_mod.SECTIONS
    assert (app_mod.SECTION_NOTES[empty]
            == app_mod._SECTION_NOTE_LIBRARY[empty])
    assert set(app_mod.SECTION_NOTES) == set(app_mod.SECTIONS)

    page = app_mod.make_home_page()
    qtbot.addWidget(page)
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert any(lbl.startswith(empty) for lbl in labels)
    # The tab carries its own note, which is why the note is written when
    # the section is named rather than when it fills up.
    assert any(app_mod._SECTION_NOTE_LIBRARY[empty] in lbl.text()
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


def test_going_over_the_cap_still_starts_and_no_longer_says_so(
        registry_sandbox, caplog):
    """The cap is a design rule. Refusing to start would not help fix it.

    Inverted 2026-08-04. This used to require a "over the 13 cap" warning on
    the registration that crosses the line, and two things had moved under it:
    the cap is ``MAX_APPS_PER_SECTION`` and is 20 now, and the warning itself
    was removed on purpose (``spacr/qt/app.py:596``) because it fired once per
    app past the cap — a full section produced a stream of identical lines at
    launch — and said nothing the suite does not already assert.

    So the half that is still a claim about behaviour is asserted, and the
    silence is asserted as the deliberate thing it is: the registration is
    accepted, the app is reachable, and nothing is logged. The cap is enforced
    by ``tests/qt/test_cov_qt_app.py``, which is where a violation should be
    read about.
    """
    room = app_mod.MAX_APPS_PER_SECTION - len(
        app_mod.section_members(app_mod.SECTION_MODELS))
    for i in range(room):
        app_mod.register_app(f"filler_{i}", f"Filler {i}", "…",
                             app_mod.SECTION_MODELS)
    with caplog.at_level(logging.WARNING, logger=app_mod.LOG.name):
        app_mod.register_app("one_too_many", "One Too Many", "…",
                             app_mod.SECTION_MODELS)

    assert any(row[0] == "one_too_many" for row in app_mod.APPS), (
        "a registration past the cap must still be accepted; refusing it "
        "would take the app away without helping anyone split the section")
    assert "one_too_many" in {
        key for key, *_ in app_mod.section_members(app_mod.SECTION_MODELS)}
    assert not [rec.getMessage() for rec in caplog.records
                if "cap" in rec.getMessage()], (
        "the per-registration cap warning is back; it fires once per app past "
        "the cap, so a full section logs a stream of identical lines at launch")


def test_unregister_puts_everything_back(registry_sandbox):
    # Copied, not aliased. APPS, SECTIONS and SECTION_NOTES are all
    # published by mutation, so holding the object rather than its
    # contents would make every assertion below compare a thing to
    # itself and pass whatever unregister_app did.
    before_apps = list(app_mod.APPS)
    before_sections = list(app_mod.SECTIONS)
    before_notes = dict(app_mod.SECTION_NOTES)

    # Into SANDBOX_SECTION, so unregistering has to take a whole section
    # and its note back out again -- the harder half of "everything back",
    # and the half that reads as a no-op if the probe joins a section that
    # already had apps in it.
    app_mod.register_app("design_probe", "Design Probe", "…",
                         SANDBOX_SECTION,
                         factory=QWidget, stage=app_mod.STAGE_BETA)
    assert SANDBOX_SECTION in app_mod.SECTIONS
    assert SANDBOX_SECTION in app_mod.SECTION_NOTES

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


# ---------------------------------------------------------------------------
# Part C — the side tables a registration has to reach
# ---------------------------------------------------------------------------
# A row in `APPS` draws a tile. It does not give the tile a header, a
# blurb, an API link, a translated name, an answer to `spacr-run <key>`
# or a Run button that runs anything: those lived in six tables in five
# other files, and four finished, tested features sat unreachable for
# weeks because their authors could not edit those files. `register_app`
# now takes those strings once and fans them out. These are the tests
# that say so — against the shipped tables, not against APP_META.

#: The features item 0.7 was written to make reachable, with what each one
#: is: `entry` for the ones with a headless pipeline function, `factory`
#: for the ones that are their own screen. Every app must be one or the
#: other -- an app with neither has a Run button that says "Not runnable"
#: and no screen of its own to explain why.
WIRED_IN = {
    "illumination": "entry",
    "barcode_qc": "entry",
    "layer_viewer": "factory",
    "graph_builder": "factory",
    # The three that landed just after the seam did and were unreachable
    # for exactly the same reason. AnnData Export is the first app here
    # with an `entry` and NO screen of its own on purpose: its settings
    # are already typed and tooltipped, so the generic AppScreen draws the
    # export form and the Run button runs the export.
    "power": "factory",
    "run_compare": "factory",
    "anndata_export": "entry",
}


@pytest.mark.parametrize("key", sorted(WIRED_IN))
def test_the_waiting_feature_is_in_the_registry_under_a_live_section(key):
    """It has a row, and that row is filed somewhere with a tab."""
    rows = [row for row in app_mod.APPS if row[0] == key]
    assert len(rows) == 1, f"{key} is registered {len(rows)} times"
    _key, name, desc, section = rows[0]
    assert name.strip() and desc.strip()
    assert section in app_mod.SECTIONS, (
        f"{key} is filed under {section!r}, which has no tab")
    assert app_mod.section_members(section), f"{section} draws an empty tab"
    # ...and it is reachable from the two derived views the UI draws from.
    assert key in dict(app_mod.home_categories())[section]
    bands = {s: [r[0] for r in rows_] for s, rows_ in app_mod.home_bands()}
    assert key in bands[section]


@pytest.mark.parametrize("key", sorted(WIRED_IN))
def test_the_waiting_feature_has_a_header_and_an_intro(key):
    """The screen tables, which used to need a hand-edit per app."""
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES

    assert APP_TITLES.get(key, "").strip(), (
        f"{key} has no header in app_screen.APP_TITLES, so its screen is "
        f"titled with its raw app key")
    intro = APP_INTROS.get(key, "")
    assert len(intro) > 40, (
        f"{key}'s intro is too short to tell anyone what the module does: "
        f"{intro!r}")


@pytest.mark.parametrize("key", sorted(WIRED_IN))
def test_the_waiting_feature_is_translated_into_every_ui_language(key):
    """`test_i18n` walks APPS and requires this; here it is per app."""
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES

    name = {row[0]: row[1] for row in app_mod.APPS}[key]
    for code in VALID_LANGUAGE_CODES:
        if code == "en":
            continue
        assert CATALOGS[code].get(name, "").strip(), (
            f"{key} ({name!r}) has no {code} translation, so its sidebar "
            f"row is English in a Korean window")


@pytest.mark.parametrize("key", sorted(WIRED_IN))
def test_the_waiting_feature_links_to_its_own_api_page(key):
    """An unknown key lands on the generated API index instead."""
    from spacr.qt.screens.settings_model import _APP_API_MODULE, api_docs_url

    assert key in _APP_API_MODULE, f"{key} has no API doc module"
    url = api_docs_url(key)
    assert url.endswith("index.html") and "/spacr/" in url
    assert _APP_API_MODULE[key] in url


@pytest.mark.parametrize("key", sorted(WIRED_IN))
def test_the_waiting_feature_answers_spacr_run(key):
    """Headless-runnable or declared GUI-only, with a sentence saying so.

    `tests/test_app_registry_parity.py` asserts this across the whole
    registry; the point of repeating it here is the failure message, and
    that these four are the ones the item was about.
    """
    from spacr import cli

    if WIRED_IN[key] == "entry":
        assert key in cli.MODULES, (
            f"{key} has a pipeline entry point, so `spacr-run {key}` has to "
            f"work; it answers 'unknown module'")
        assert key not in cli.INTERACTIVE_ONLY
    else:
        assert key in cli.INTERACTIVE_ONLY, (
            f"{key} has no headless path and does not say so, so "
            f"`spacr-run {key}` answers 'unknown module' and the user "
            f"concludes they typed it wrong")
        assert len(cli.INTERACTIVE_ONLY[key]) >= 40


@pytest.mark.parametrize("key", sorted(k for k, v in WIRED_IN.items()
                                       if v == "entry"))
def test_the_run_button_resolves_to_something_runnable(key):
    """The measured symptom: Run said "Not runnable" for all of these.

    `AppScreen._on_run` calls `resolve_pipeline_entry` and shows the
    "Not runnable" box when it returns None, so this is the assertion
    that the button does something -- and it resolves the real callable,
    through the real bridge, not a stand-in.
    """
    from spacr.qt.bridge import resolve_pipeline_entry

    entry = resolve_pipeline_entry(key)
    assert entry is not None and callable(entry), (
        f"the {key} Run button reports 'Not runnable'")
    # The registries have to name the SAME function, or the app is
    # validated against one callable and runs another.
    from spacr import cli
    from spacr.validate import APP_FUNCTIONS

    inner = getattr(entry, "__wrapped__", entry)
    assert inner.__name__ == cli.MODULES[key].func_name
    assert APP_FUNCTIONS[key].rsplit(".", 1)[-1] == inner.__name__


@pytest.mark.parametrize("key", sorted(k for k, v in WIRED_IN.items()
                                       if v == "factory"))
def test_the_screen_owning_app_builds_its_own_screen(qtbot, qt_theme_applied,
                                                     key):
    """The other half: an app with no Run button has a screen instead."""
    from PySide6.QtWidgets import QWidget

    factory = app_mod.registered_factory(key)
    assert factory is not None, f"{key} registered no screen factory"
    widget = app_mod._call_screen_factory(factory, key)
    qtbot.addWidget(widget)
    assert isinstance(widget, QWidget)


@pytest.mark.parametrize("key", sorted(WIRED_IN))
def test_the_waiting_feature_has_a_settings_panel_or_a_screen(key):
    """A generic AppScreen with no defaults opens on an empty form.

    Illumination and Barcode QC register their settings at their own
    module's import, which the process drawing the panel has no reason
    to have done -- `register_app(..., defaults_module=...)` is what
    closes that, and this is the assertion that it did.
    """
    from spacr.qt.screens.settings_model import resolve_default_settings

    if WIRED_IN[key] == "factory":
        pytest.skip(f"{key} builds its own screen, not a settings form")
    settings = resolve_default_settings(key)
    assert isinstance(settings, dict) and settings, (
        f"the {key} screen would open on an empty settings form")


def test_a_registration_reaches_every_side_table_in_one_call(registry_sandbox):
    """The seam itself: one call, six tables, no hand-edits.

    This is the property item 0.7 exists for. Everything above asserts
    it for the four apps that were waiting; this asserts it for an app
    that did not exist when any of those tables were written, which is
    the only way to know the seam works rather than that somebody typed
    six entries correctly.
    """
    from spacr import cli
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    from spacr.qt.screens.settings_model import _APP_API_MODULE

    key = "fanout_probe"
    assert key not in APP_TITLES and key not in cli.INTERACTIVE_ONLY

    app_mod.register_app(
        key, "Fanout Probe", "One call, every table",
        app_mod.SECTION_RESULTS, stage=app_mod.STAGE_ALPHA,
        title="Fanout Probe (header)", intro="What the module does, at length.",
        cli_note="Fanout Probe is interactive; run it in the GUI (spacr-qt).",
        api_module="qt/fanout_probe",
        translations=("a", "b", "c", "d", "e", "f", "g", "h", "i"))
    try:
        assert APP_TITLES[key] == "Fanout Probe (header)"
        assert APP_INTROS[key] == "What the module does, at length."
        assert cli.INTERACTIVE_ONLY[key].startswith("Fanout Probe")
        assert _APP_API_MODULE[key] == "qt/fanout_probe"
        for code in VALID_LANGUAGE_CODES:
            if code != "en":
                assert CATALOGS[code]["Fanout Probe"]
    finally:
        app_mod.unregister_app(key)

    # ...and unregistering takes them back out again, or a plugin that
    # unloads leaves a GUI-only excuse behind for an app that is gone.
    assert key not in APP_TITLES
    assert key not in APP_INTROS
    assert key not in cli.INTERACTIVE_ONLY
    assert key not in _APP_API_MODULE


def test_title_and_intro_fall_back_to_the_name_and_the_description(
        registry_sandbox):
    """The minimum registration still yields a titled, described screen.

    Four strings are the price of a working app, not eight: an app that
    says nothing more than its row gets a header and a blurb anyway.
    """
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES

    app_mod.register_app("minimal_probe", "Minimal Probe",
                         "The one-line description", app_mod.SECTION_RESULTS)
    try:
        assert APP_TITLES["minimal_probe"] == "Minimal Probe"
        assert APP_INTROS["minimal_probe"] == "The one-line description"
    finally:
        app_mod.unregister_app("minimal_probe")


def test_a_registered_entry_is_spelled_module_colon_function(registry_sandbox):
    """A typo in the entry has to name itself, not resolve to None.

    `resolve_pipeline_entry` swallows exceptions and returns None, which
    is the right behaviour for a broken pipeline import and exactly the
    wrong one for a malformed registration: the app would silently be
    "Not runnable" forever.
    """
    app_mod.register_app("entry_probe", "Entry Probe", "…",
                         app_mod.SECTION_RESULTS,
                         entry="spacr.illumination.illumination_settings")
    with pytest.raises(ValueError, match="module:function"):
        app_mod.registered_entry("entry_probe")

    app_mod.unregister_app("entry_probe")
    app_mod.register_app("entry_probe", "Entry Probe", "…",
                         app_mod.SECTION_RESULTS,
                         entry="spacr.illumination:illumination_settings")
    from spacr.illumination import illumination_settings
    assert app_mod.registered_entry("entry_probe") is illumination_settings
    # No entry at all is not an error -- it is what an app with its own
    # screen and no headless path looks like.
    assert app_mod.registered_entry("annotate") is None


def test_an_earlier_import_of_sections_sees_the_four_new_apps():
    """`from spacr.qt.app import SECTIONS` cannot go stale.

    Graph Builder and Layer Viewer are the first apps of the Explore
    section, and they register from their own modules, after `app.py`
    has finished importing. While `SECTIONS` was a tuple that
    `_refresh_sections` rebound, every module holding the name -- and
    `tests/qt/test_home_layout.py`, which takes it at collection --
    kept a snapshot from before that, and Explore never appeared.
    """
    from spacr.qt.app import SECTIONS as imported_here

    assert imported_here is app_mod.SECTIONS
    assert app_mod.SECTION_EXPLORE in imported_here
    assert {row[3] for row in app_mod.APPS} <= set(imported_here)


def test_the_empty_state_names_this_screens_own_demo():
    """It said "use Demos → Mask demo…" on every screen.

    Measure, Timelapse, Classify and Sequencing each pointed the user at
    a dataset that opens a DIFFERENT module, so following the hint left
    the screen the user was trying to fill exactly as empty.
    """
    assert app_mod.demo_label_for_app("mask") == "Mask demo…"
    assert app_mod.demo_label_for_app("measure") == "Measure demo…"
    assert app_mod.demo_label_for_app("timelapse") == "Timelapse demo…"
    # The classify demo lands on Annotate (it generates crops to label),
    # so that is where its hint belongs -- not on the Classify screen.
    assert app_mod.demo_label_for_app("annotate") == "Classify demo…"
    assert app_mod.demo_label_for_app("map_barcodes") == "Sequencing demo…"
    # ...and an app with no demo says nothing rather than naming one.
    assert app_mod.demo_label_for_app("regression") is None
    assert app_mod.demo_label_for_app("barcode_qc") is None
    # Every demo the menu offers reaches an app that exists.
    keys = {row[0] for row in app_mod.APPS}
    for demo_key in app_mod.DEMO_LABELS:
        target = app_mod.MainWindow.DEMO_TARGETS[demo_key][0]
        assert target in keys, f"the {demo_key} demo opens a missing app"


@pytest.mark.parametrize("app_key,expected", [
    ("measure", "Measure demo…"),
    ("timelapse", "Timelapse demo…"),
    # No demo lands on Image UMAP, so its banner must not name one --
    # this is the case that used to read "use Demos → Mask demo…".
    ("umap", None),
])
def test_the_empty_state_banner_offers_the_right_demo(
        qtbot, qt_theme_applied, app_key, expected):
    """Through the shipped banner, not the lookup it calls."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    banner = screen._build_empty_state_banner()
    if banner is None:
        pytest.skip(f"{app_key} has no src field to leave empty")
    text = " ".join(lbl.text() for lbl in banner.findChildren(QLabel))
    if expected is None:
        assert "demo…" not in text, (
            f"{app_key} has no demo but the banner names one: {text!r}")
        assert "Demos menu" in text
    else:
        assert expected in text, (
            f"the {app_key} banner offers the wrong demo: {text!r}")
