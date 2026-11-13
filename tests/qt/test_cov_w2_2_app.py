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


import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import QWidget

from spacr.qt import app as app_mod
from spacr.qt.app import (APPS, SECTION_ORDER, SECTIONS,
                          MainWindow, Sidebar, _ChromeButton, _LiveSections,
                          _PipelinePreloader, _UpdateWorker,
                          _call_screen_factory, _insert_position,
                          app_is_visible, app_stage,
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


def test_a_row_in_an_unknown_section_does_not_break_the_ordering(qapp,
                                                                 monkeypatch):
    """A plugin can file a row this build's SECTION_ORDER does not name."""
    APPS.append(("w2_2_orphan", "Orphan", "from elsewhere", "No Such Section"))
    try:
        position = _insert_position(SECTION_ORDER[0])
        assert 0 <= position <= len(APPS)
    finally:
        APPS[:] = [row for row in APPS if row[0] != "w2_2_orphan"]


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


def test_an_interactive_only_app_has_no_pipeline_entry(qapp):
    """`None` is what an app with its own screen and no Run button gives."""
    register_app("w2_2_probe", "W2 2 Probe", "a test app", SECTIONS[0])
    try:
        assert registered_entry("w2_2_probe") is None
    finally:
        unregister_app("w2_2_probe")

    assert registered_entry("no_such_app_key_at_all") is None


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



# ---------------------------------------------------------------------------
# the sidebar
# ---------------------------------------------------------------------------

@pytest.fixture
def sidebar(qapp):
    made = Sidebar()
    yield made
    made.deleteLater()


def test_the_column_is_wide_enough_for_its_longest_label(sidebar, qapp):
    """Names longer than the maximum elide with a tooltip rather than pushing
    the column across the window."""
    from spacr.qt.preferences import scaled_px

    width = sidebar.fitting_width()
    assert scaled_px(Sidebar.WIDTH_MIN) <= width <= scaled_px(
        Sidebar.WIDTH_MAX)
    assert sidebar.clipped_items() == []


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

    THE PAINTER TAKES A SIZE AS WELL AS A COLOUR, since 433 made the marks
    follow the interface scale: the glyph is baked into a pixmap, so it has
    to be painted at the size it will be drawn at rather than painted small
    and stretched. The double accepts both because the three real painters
    do; a double that accepted less would be testing a contract nothing has.
    """
    asked = []

    def painter(size=18, colour=None):
        asked.append(colour)
        return MainWindow._close_icon(size=size, colour=colour)

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
                                         "MaximiseWindow", "MinimiseWindow"}
    assert app_mod.CHROME_HOVER["CloseWindow"] != \
        app_mod.CHROME_HOVER["MinimiseWindow"]


# ---------------------------------------------------------------------------
# process-level setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# where a nav icon comes from
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# process-level setup, continued
# ---------------------------------------------------------------------------

