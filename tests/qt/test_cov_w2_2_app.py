"""The window's registry, its background worker, and the marks it paints.

`spacr.qt.app` is mostly a MainWindow, and the parts of it that can be
reasoned about on their own are the ones that decide what the window will
contain before it exists: which apps are registered, in what order, with
which metadata pushed into which side table; what the update worker does when
the operation it is running raises; what the preloader does when a module
will not import; and what the frameless chrome draws when nobody has hovered
it.

Those are what this file drives. Every one of them is exercised through the
real registry -- `tests/qt/conftest.py` puts it back afterwards -- rather than
through a copy, because "two modules quietly claiming one key" is a property
of the real `APPS` list and of nothing else.
"""

import logging
import os
import sys
import types

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import QWidget

from spacr.qt import app as app_mod
from spacr.qt.app import (APPS, SECTION_ORDER, SECTIONS,
                          MainWindow, Sidebar, _ChromeButton, _LiveSections,
                          _PipelinePreloader, _UpdateWorker,
                          _call_screen_factory, _insert_position,
                          app_is_visible, app_stage, demo_label_for_app,
                          register_app, registered_entry, registered_metadata,
                          unregister_app, visible_apps)


# ---------------------------------------------------------------------------
# the background updater
# ---------------------------------------------------------------------------

def test_an_update_operation_hands_back_what_it_produced(qapp, qtbot):
    """The ordinary path: the worker runs off the GUI thread and reports."""
    worker = _UpdateWorker("check", lambda: {"version": "1.2.3"})
    try:
        with qtbot.waitSignal(worker.succeeded, timeout=10000) as caught:
            worker.start()
        assert caught.args == [{"version": "1.2.3"}]
        assert worker.operation == "check"
    finally:
        worker.wait(5000)


def test_an_update_operation_that_raises_surfaces_its_traceback(qapp, qtbot):
    """An exception out of a QThread::run override aborts the process.

    So the failure is caught, named and emitted instead -- with the operation
    it belongs to, because the message the user sees has to say which one.
    """
    def explode():
        raise RuntimeError("the index could not be reached")

    worker = _UpdateWorker("upgrade", explode)
    try:
        with qtbot.waitSignal(worker.failed, timeout=10000) as caught:
            worker.start()
        operation, details = caught.args
        assert operation == "upgrade"
        assert "RuntimeError: the index could not be reached" in details
        assert "Traceback" in details
    finally:
        worker.wait(5000)


# ---------------------------------------------------------------------------
# the pipeline preloader
# ---------------------------------------------------------------------------

def test_the_preloader_knows_its_denominator_before_it_starts():
    """A loading bar with an unknown total is a spinner with a number on it."""
    preloader = _PipelinePreloader()
    assert preloader.total() == len(_PipelinePreloader._MODULES)
    assert preloader.total() > 0


def test_the_preloader_reports_every_step_and_finishes_once(qapp, qtbot,
                                                            monkeypatch):
    """One module per event-loop tick, so Qt can repaint between them."""
    steps = []
    done = []
    monkeypatch.setattr(_PipelinePreloader, "_MODULES",
                        ("json", "os.path", "no.such.module.at.all"))

    preloader = _PipelinePreloader(on_step=lambda i, n: steps.append((i, n)),
                                   on_done=lambda: done.append(1))
    preloader.start()
    preloader.start()                      # begun already: a no-op
    qtbot.waitUntil(lambda: bool(done), timeout=10000)

    assert steps == [(1, 3), (2, 3), (3, 3)]
    assert done == [1], "the completion callback ran more than once"


def test_a_module_that_will_not_import_does_not_stop_the_preload(qapp, qtbot,
                                                                 monkeypatch):
    """Preloading is optional; the window still opens."""
    monkeypatch.setattr(_PipelinePreloader, "_MODULES",
                        ("no.such.module.at.all", "json"))
    done = []
    preloader = _PipelinePreloader(on_done=lambda: done.append(1))
    preloader.start()
    qtbot.waitUntil(lambda: bool(done), timeout=10000)
    assert done == [1]


def test_a_progress_callback_that_raises_does_not_stop_the_preload(
        qapp, qtbot, monkeypatch):
    """The loading screen can be gone by the time a step reports."""
    monkeypatch.setattr(_PipelinePreloader, "_MODULES", ("json", "os.path"))

    def explode(_done, _total):
        raise RuntimeError("the loading screen has closed")

    done = []
    preloader = _PipelinePreloader(on_step=explode,
                                   on_done=lambda: done.append(1))
    preloader.start()
    qtbot.waitUntil(lambda: bool(done), timeout=10000)
    assert done == [1]


def test_a_completion_callback_that_raises_is_swallowed(qapp, qtbot,
                                                        monkeypatch):
    """Same reason: the thing being told may already be gone."""
    monkeypatch.setattr(_PipelinePreloader, "_MODULES", ("json",))
    reached = []

    def explode():
        reached.append(1)
        raise RuntimeError("the loading screen has closed")

    preloader = _PipelinePreloader(on_done=explode)
    preloader.start()
    qtbot.waitUntil(lambda: bool(reached), timeout=10000)

    # SWALLOWED, NOT SKIPPED. The callback ran and raised, and the
    # preloader still finished: a guard that avoided calling it at all
    # would also "not raise" and would lose the completion.
    assert reached == [1]


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_the_section_list_compares_equal_to_the_tuple_it_used_to_be():
    """`from spacr.qt.app import SECTIONS` binds the OBJECT.

    A tuple that `_refresh_sections` rebound left every importer holding a
    snapshot; a list mutated in place is what makes a late registration
    visible everywhere. Changing the container must not change what the
    suite's `SECTIONS == (...)` assertions mean.
    """
    live = _LiveSections(["Core", "Explore"])
    assert live == ("Core", "Explore")
    assert live == ["Core", "Explore"]
    assert not (live != ("Core", "Explore"))
    assert live != ("Core",)
    assert live != 7
    assert _LiveSections.__hash__ is None

    with pytest.raises(TypeError):
        {live: 1}


def test_a_row_is_filed_after_its_own_section(qapp):
    """The sidebar starts a heading every time the section changes.

    A row filed out of order draws its section's heading a second time.
    """
    section = SECTIONS[0]
    register_app("w2_2_probe", "W2 2 Probe", "a test app", section)
    try:
        keys = [row[0] for row in APPS]
        index = keys.index("w2_2_probe")
        sections = [row[3] for row in APPS]
        assert sections[index] == section
        # every row of this section is contiguous
        first = sections.index(section)
        last = len(sections) - 1 - sections[::-1].index(section)
        assert sections[first:last + 1] == [section] * (last - first + 1)
    finally:
        assert unregister_app("w2_2_probe") is True


def test_a_row_in_an_unknown_section_does_not_break_the_ordering(qapp,
                                                                 monkeypatch):
    """A plugin can file a row this build's SECTION_ORDER does not name."""
    APPS.append(("w2_2_orphan", "Orphan", "from elsewhere", "No Such Section"))
    try:
        position = _insert_position(SECTION_ORDER[0])
        assert 0 <= position <= len(APPS)
    finally:
        APPS[:] = [row for row in APPS if row[0] != "w2_2_orphan"]


def test_registering_the_same_key_twice_is_refused(qapp):
    """Two modules quietly claiming one key is what a registry prevents."""
    register_app("w2_2_probe", "W2 2 Probe", "a test app", SECTIONS[0])
    try:
        with pytest.raises(ValueError) as raised:
            register_app("w2_2_probe", "Other", "another", SECTIONS[0])
        assert "already registered" in str(raised.value)
    finally:
        unregister_app("w2_2_probe")

    assert unregister_app("w2_2_probe") is False


def test_a_registration_that_cannot_mean_anything_is_refused(qapp):
    """The five ways a row can be malformed, each named where it was written."""
    with pytest.raises(ValueError):
        register_app("", "A Name", "a description", SECTIONS[0])
    with pytest.raises(ValueError):
        register_app("w2_2_probe", "  ", "a description", SECTIONS[0])
    with pytest.raises(ValueError):
        register_app("w2_2_probe", "A Name", "  ", SECTIONS[0])
    with pytest.raises(ValueError) as raised:
        register_app("w2_2_probe", "A Name", "a description", "No Section")
    assert "SECTION_ORDER" in str(raised.value)
    with pytest.raises(ValueError):
        register_app("w2_2_probe", "A Name", "a description", SECTIONS[0],
                     stage="not a stage")
    with pytest.raises(TypeError):
        register_app("w2_2_probe", "A Name", "a description", SECTIONS[0],
                     factory="not callable")

    assert "w2_2_probe" not in {row[0] for row in APPS}


def test_metadata_is_published_and_empty_fields_are_dropped(qapp):
    """The pull half of the seam: a side table absorbs what registered
    earlier."""
    register_app("w2_2_probe", "W2 2 Probe", "a test app", SECTIONS[0],
                 title="A Probe", intro="what it does")
    try:
        titles = registered_metadata("title")
        assert titles["w2_2_probe"] == "A Probe"
        # a field this app did not give is absent rather than empty
        assert "w2_2_probe" not in registered_metadata("cli_note")
    finally:
        unregister_app("w2_2_probe")


def test_a_translation_that_cannot_be_catalogued_costs_the_name_not_the_app(
        qapp, monkeypatch, caplog):
    """A side table that cannot take an entry must not take the tile down."""
    import spacr.qt.i18n as i18n

    def explode(_source, _values):
        raise ValueError("that row has a blank entry")

    monkeypatch.setattr(i18n, "add_translation", explode)

    # NAMED LOGGER, AND PROPAGATION FORCED. caplog reads the root handler,
    # so a sibling test that silences spacr.qt.app or clears propagation --
    # several do, to keep their own output readable -- leaves this one
    # asserting an empty string and failing only when run after them.
    logger = logging.getLogger("spacr.qt.app")
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "disabled", False)
    with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
        row = register_app("w2_2_probe", "W2 2 Probe", "a test app",
                           SECTIONS[0], translations=["x"] * 9)
        assert row[0] == "w2_2_probe"
    try:
        assert "w2_2_probe" in {row[0] for row in APPS}
        assert "Could not translate app name" in caplog.text
    finally:
        unregister_app("w2_2_probe")


def test_an_interactive_only_app_has_no_pipeline_entry(qapp):
    """`None` is what an app with its own screen and no Run button gives."""
    register_app("w2_2_probe", "W2 2 Probe", "a test app", SECTIONS[0])
    try:
        assert registered_entry("w2_2_probe") is None
    finally:
        unregister_app("w2_2_probe")

    assert registered_entry("no_such_app_key_at_all") is None


def test_an_entry_is_imported_on_demand_not_at_registration(qapp):
    """Registering an app must not drag numpy into a process drawing a
    sidebar."""
    register_app("w2_2_probe", "W2 2 Probe", "a test app", SECTIONS[0],
                 entry="json:dumps")
    try:
        import json

        assert registered_entry("w2_2_probe") is json.dumps
    finally:
        unregister_app("w2_2_probe")


def test_a_malformed_entry_says_how_it_should_be_spelled(qapp):
    """`module:function`, and anything else is refused where it was written."""
    register_app("w2_2_probe", "W2 2 Probe", "a test app", SECTIONS[0],
                 entry="justamodule")
    try:
        with pytest.raises(ValueError) as raised:
            registered_entry("w2_2_probe")
        assert "module:function" in str(raised.value)
    finally:
        unregister_app("w2_2_probe")


# ---------------------------------------------------------------------------
# building a screen
# ---------------------------------------------------------------------------

def test_a_factory_is_given_only_the_arguments_it_declares():
    """`lambda: MyScreen()` is a complete factory.

    Resolved by inspecting the signature rather than by calling and retrying
    on TypeError: a retry cannot tell a wrong call from a TypeError raised
    inside a factory that was called correctly, and would build the screen
    twice.
    """
    host = object()

    assert _call_screen_factory(lambda: "plain", "mask") == "plain"

    def wants_key(app_key):
        return f"key={app_key}"

    assert _call_screen_factory(wants_key, "mask") == "key=mask"

    def wants_both(app_key, host):
        return (app_key, host)

    assert _call_screen_factory(wants_both, "mask", host) == ("mask", host)

    def wants_anything(**kwargs):
        return sorted(kwargs)

    assert _call_screen_factory(wants_anything, "mask", host) == \
        ["app_key", "host"]


def test_a_factory_with_no_introspectable_signature_is_called_bare():
    """Builtins and C callables have no signature to read."""
    assert _call_screen_factory(dict, "mask") == {}


# ---------------------------------------------------------------------------
# what the sidebar is allowed to show
# ---------------------------------------------------------------------------

def test_an_unreadable_preference_shows_every_module(qapp, monkeypatch):
    """The historical all-modules-visible behaviour is the safe fallback.

    Hiding modules because a preference could not be read would look exactly
    like the modules not existing.
    """
    import spacr.qt.preferences as preferences

    def explode(_stage):
        raise RuntimeError("the preference store is unreadable")

    monkeypatch.setattr(preferences, "maturity_is_visible", explode)

    assert app_is_visible(APPS[0][0]) is True
    assert len(visible_apps()) == len(APPS)


def test_an_unregistered_key_still_has_a_stage(qapp):
    """`app_stage` answers for anything, so a caller need not guard."""
    assert isinstance(app_stage("no_such_app_key_at_all"), str)


# ---------------------------------------------------------------------------
# the Demos menu's hint
# ---------------------------------------------------------------------------

def test_a_module_with_no_demo_gets_no_demo_name(qapp):
    """The caller says something generic rather than naming a demo that would
    take the user somewhere else."""
    assert demo_label_for_app("no_such_app_key_at_all") is None

    targets = MainWindow.DEMO_TARGETS
    assert targets, "the demo table is empty"
    for demo_key, (target, _generator) in targets.items():
        label = demo_label_for_app(target)
        if label is not None:
            assert isinstance(label, str) and label
            break
    else:
        pytest.fail("no demo resolved to a label")


# ---------------------------------------------------------------------------
# the sidebar
# ---------------------------------------------------------------------------

@pytest.fixture
def sidebar(qapp):
    made = Sidebar()
    yield made
    made.deleteLater()


def test_a_section_header_click_folds_its_modules_away(sidebar, qapp):
    """The header is the control; the tiles under it are what it shows."""
    section = SECTIONS[0]
    assert sidebar.section_is_open(section) is True

    assert sidebar.toggle_section(section) is False
    assert sidebar.section_is_open(section) is False

    assert sidebar.toggle_section(section) is True
    assert sidebar.section_is_open(section) is True


def test_an_event_on_something_that_is_not_a_header_is_passed_on(sidebar,
                                                                 qapp):
    """The filter is installed on the whole sidebar, headers included."""
    plain = QWidget(sidebar)
    event = QEvent(QEvent.Type.Enter)
    assert sidebar.eventFilter(plain, event) is False


def test_hovering_a_section_header_lights_it(sidebar, qapp):
    """The hover state is a property the stylesheet reads."""
    headers = [child for child in sidebar.findChildren(QWidget)
               if child.property("sectionName")]
    assert headers, "the sidebar drew no section headers"
    header = headers[0]

    sidebar.eventFilter(header, QEvent(QEvent.Type.Enter))
    assert header.property("hovered") is True

    sidebar.eventFilter(header, QEvent(QEvent.Type.Leave))
    assert header.property("hovered") is False


def test_a_left_release_on_a_header_toggles_its_section(sidebar, qapp):
    """The gesture, not the method: this is what a mouse actually sends."""
    headers = [child for child in sidebar.findChildren(QWidget)
               if child.property("sectionName")]
    header = headers[0]
    section = str(header.property("sectionName"))
    before = sidebar.section_is_open(section)

    event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPoint(2, 2),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    assert sidebar.eventFilter(header, event) is True
    assert sidebar.section_is_open(section) is not before

    sidebar.toggle_section(section)


def test_the_column_is_wide_enough_for_its_longest_label(sidebar, qapp):
    """Names longer than the maximum elide with a tooltip rather than pushing
    the column across the window."""
    from spacr.qt.preferences import scaled_px

    width = sidebar.fitting_width()
    assert scaled_px(Sidebar.WIDTH_MIN) <= width <= scaled_px(
        Sidebar.WIDTH_MAX)
    assert sidebar.clipped_items() == []


def test_re_inking_the_icons_reaches_every_row(sidebar, qapp):
    """A QIcon bakes its pixmap when it is built, so a restyle does not
    recolour icons that already exist."""
    sidebar.refresh_icons()                # must not raise
    keyed = [b for b in sidebar._items if b.property("navKey")]
    assert keyed, "no sidebar row carried its app key"
    assert all(not b.icon().isNull() for b in keyed)


def test_the_visible_rows_follow_the_maturity_preference(sidebar, qapp,
                                                         monkeypatch):
    """A module hidden by preference is hidden, not removed."""
    import spacr.qt.preferences as preferences

    monkeypatch.setattr(preferences, "maturity_is_visible",
                        lambda _stage: False)
    sidebar.refresh_visibility()
    keyed = [b for b in sidebar._items if b.property("navKey")
             and b.property("navKey") != "__home__"]
    assert all(b.isHidden() for b in keyed)

    monkeypatch.setattr(preferences, "maturity_is_visible",
                        lambda _stage: True)
    sidebar.refresh_visibility()
    assert any(not b.isHidden() for b in keyed)


# ---------------------------------------------------------------------------
# the frameless window's own three marks
# ---------------------------------------------------------------------------

def test_each_chrome_mark_is_drawn_rather_than_shipped(qapp):
    """Painted at the size asked for, and in the colour asked for."""
    for painter in (MainWindow._close_icon, MainWindow._minimise_icon,
                    MainWindow._fullscreen_icon):
        plain = painter()
        assert isinstance(plain, QIcon)
        assert not plain.isNull()

        lit = painter(size=24, colour="#DC3C3C")
        assert not lit.isNull()
        assert lit.availableSizes()[0].width() == 24


def test_a_chrome_button_lights_its_mark_on_hover_and_on_press(qapp):
    """QSS can colour a background on :hover but not the contents of a QIcon.

    So the hover state is a second painting of the same glyph, and what the
    test can see is which painting was asked for.
    """
    asked = []

    def painter(colour=None):
        asked.append(colour)
        return MainWindow._close_icon(colour=colour)

    parent = QWidget()
    button = _ChromeButton(parent, painter, "#DC3C3C")
    try:
        assert asked == [None], "the mark started out lit"

        from PySide6.QtGui import QEnterEvent

        button.enterEvent(QEnterEvent(QPoint(2, 2), QPoint(2, 2),
                                      QPoint(2, 2)))
        assert asked[-1] == "#DC3C3C"

        button.leaveEvent(QEvent(QEvent.Type.Leave))
        assert asked[-1] is None

        press = QMouseEvent(QEvent.Type.MouseButtonPress, QPoint(2, 2),
                            Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        button.mousePressEvent(press)
        assert asked[-1] == "#DC3C3C"

        release = QMouseEvent(QEvent.Type.MouseButtonRelease, QPoint(2, 2),
                              Qt.MouseButton.LeftButton,
                              Qt.MouseButton.NoButton,
                              Qt.KeyboardModifier.NoModifier)
        button.mouseReleaseEvent(release)
        assert asked[-1] is None
    finally:
        parent.deleteLater()


def test_every_chrome_mark_has_a_hover_colour():
    """Red ends the session; blue is the accent every live control uses."""
    assert set(app_mod.CHROME_HOVER) == {"CloseWindow", "FullScreenToggle",
                                         "MinimiseWindow"}
    assert app_mod.CHROME_HOVER["CloseWindow"] != \
        app_mod.CHROME_HOVER["MinimiseWindow"]


# ---------------------------------------------------------------------------
# process-level setup
# ---------------------------------------------------------------------------

def test_a_build_with_no_bundled_fonts_loads_none(qapp, monkeypatch):
    """A trimmed install has no font folder, and that is not a failure."""
    from PySide6.QtGui import QFontDatabase

    loaded = []

    monkeypatch.setattr(os.path, "isdir", lambda _path: False)
    monkeypatch.setattr(QFontDatabase, "addApplicationFont",
                        staticmethod(lambda path: loaded.append(path) or -1))

    app_mod._load_bundled_fonts()

    # NONE, and asserted rather than left to a raising stub: the point is
    # that a missing folder is not searched, not merely that nothing blew
    # up on the way past it.
    assert loaded == []


def test_the_crash_dump_goes_beside_the_ordinary_log(tmp_path, monkeypatch):
    """faulthandler writes from a signal handler, where opening a file is not
    allowed -- so the file is opened once and kept."""
    import faulthandler

    from spacr import logging_util

    monkeypatch.setattr(logging_util, "log_dir", lambda: str(tmp_path))
    enabled = []
    monkeypatch.setattr(faulthandler, "enable",
                        lambda file=None, all_threads=True:
                        enabled.append(file))

    path = app_mod._install_crash_dump()
    try:
        assert path == str(tmp_path / app_mod.CRASH_DUMP_NAME)
        assert os.path.isfile(path)
        assert "spaCR started" in open(path).read()
        assert enabled and enabled[0] is app_mod._CRASH_DUMP_FILE
    finally:
        handle = app_mod.__dict__.pop("_CRASH_DUMP_FILE", None)
        if handle is not None:
            handle.close()


def test_a_log_folder_that_cannot_be_made_costs_the_crash_dump_only(
        tmp_path, monkeypatch):
    """The dump is diagnostics; the application is not."""
    from spacr import logging_util

    def explode():
        raise RuntimeError("the log directory cannot be resolved")

    monkeypatch.setattr(logging_util, "log_dir", explode)
    monkeypatch.setattr(os, "makedirs",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            OSError("read-only filesystem")))

    assert app_mod._install_crash_dump() == ""


# ---------------------------------------------------------------------------
# where a nav icon comes from
# ---------------------------------------------------------------------------

def test_a_plugin_may_ship_its_own_icon_file(qapp, tmp_path, monkeypatch):
    """A contributed app's PNG is used as it is; a glyph name is themed."""
    import spacr.plugins as plugins
    from PIL import Image

    icon_path = tmp_path / "plugin.png"
    Image.new("RGBA", (16, 16), (200, 30, 30, 255)).save(icon_path)

    monkeypatch.setattr(
        plugins, "get_app",
        lambda key: types.SimpleNamespace(icon=str(icon_path)))
    icon = app_mod._icon_for_app("w2_2_probe")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()

    monkeypatch.setattr(plugins, "get_app",
                        lambda key: types.SimpleNamespace(icon="flask"))
    assert isinstance(app_mod._icon_for_app("w2_2_probe"), QIcon)


def test_a_plugin_registry_that_cannot_be_read_still_yields_an_icon(
        qapp, monkeypatch):
    """A sidebar row without an icon is worse than the built-in one."""
    import spacr.plugins as plugins

    def explode(_key):
        raise RuntimeError("the plugin registry is unreadable")

    monkeypatch.setattr(plugins, "get_app", explode)
    assert app_mod._icon_for_app(APPS[0][0]) is not None


def test_a_key_that_wants_a_glyph_gets_the_glyph_not_the_png(qapp,
                                                             monkeypatch):
    """Some keys were given a fresh qtawesome glyph rather than a drawing."""
    monkeypatch.setattr(app_mod, "_FORCE_GLYPH", {"w2_2_probe"})
    icon = app_mod._icon_for_app("w2_2_probe")
    assert isinstance(icon, QIcon)


# ---------------------------------------------------------------------------
# process-level setup, continued
# ---------------------------------------------------------------------------

def test_the_bundled_fonts_are_registered_with_qt(qapp):
    """Open Sans ships with spaCR; a build with it must actually load it.

    Idempotent, because Qt tracks the file path -- so calling it twice is the
    documented behaviour rather than a second registration.
    """
    from PySide6.QtGui import QFontDatabase

    app_mod._load_bundled_fonts()
    app_mod._load_bundled_fonts()
    families = set(QFontDatabase.families())
    assert any("Open Sans" in family for family in families), \
        "the bundled font never reached QFontDatabase"


def test_a_build_with_no_logging_helper_still_writes_a_crash_dump(tmp_path,
                                                                  monkeypatch):
    """The dump falls back to `~/.spacr/logs` when the helper is absent."""
    import faulthandler

    monkeypatch.setitem(sys.modules, "spacr.logging_util", None)
    monkeypatch.setattr(os.path, "expanduser", lambda _p: str(tmp_path))
    monkeypatch.setattr(faulthandler, "enable",
                        lambda file=None, all_threads=True: None)

    path = app_mod._install_crash_dump()
    try:
        assert path == os.path.join(str(tmp_path), ".spacr", "logs",
                                    app_mod.CRASH_DUMP_NAME)
        assert os.path.isfile(path)
    finally:
        handle = app_mod.__dict__.pop("_CRASH_DUMP_FILE", None)
        if handle is not None:
            handle.close()
