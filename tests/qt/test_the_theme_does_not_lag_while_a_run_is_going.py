"""Instruction 126 — "the theme starts lagging" while a run is going.

MEASURED BEFORE ANYTHING WAS WRITTEN, because the instruction says to and
because its three candidate causes want three different fixes. Offscreen,
1280x800, `blobs` with its shading already on `ambient._FrameProducer`, a
16 ms timer, two worker threads:

    idle                                 median 16.00 ms   p95  16.04 ms
    numpy worker (what a run mostly is)  median 16.00 ms   p95  20.05 ms
    pure-Python worker                   median 42.42 ms   p95 118.63 ms
    pure-Python, switchinterval 0.001    median 17.74 ms   p95  48.46 ms

So the producer thread had already fixed the common case; what was left was
cause 1, the interpreter lock, against which -- exactly as the instruction
predicted -- moving the shading to another Python thread does nothing.

These tests hold the CONTRACT (claimed for the run, restored after, nesting)
rather than the milliseconds. A timing assertion on a shared CI machine is a
test that fails for reasons that have nothing to do with the code; the
numbers above are the evidence and they are written down where they were
taken.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.gil_priority import (BUSY_INTERVAL, active,  # noqa: E402
                                   claim, release, responsive_gui)


@pytest.fixture(autouse=True)
def _restore():
    before = sys.getswitchinterval()
    while active():
        release()
    yield
    while active():
        release()
    sys.setswitchinterval(before)


def test_a_running_worker_asks_for_the_lock_more_often():
    before = sys.getswitchinterval()
    with responsive_gui():
        assert sys.getswitchinterval() == BUSY_INTERVAL
    assert sys.getswitchinterval() == before


def test_it_is_given_back_when_the_run_raises():
    """Otherwise the process pays 1 ms for as long as it lives."""
    before = sys.getswitchinterval()
    with pytest.raises(RuntimeError):
        with responsive_gui():
            raise RuntimeError("the design is not identifiable")
    assert sys.getswitchinterval() == before


def test_two_modules_running_at_once_do_not_hand_it_back_early():
    """Mask and Measure together: the first to finish must not undo it."""
    before = sys.getswitchinterval()
    claim()
    claim()
    release()
    assert sys.getswitchinterval() == BUSY_INTERVAL, "the second run lost it"
    release()
    assert sys.getswitchinterval() == before


def test_releasing_more_than_was_claimed_does_not_go_negative():
    before = sys.getswitchinterval()
    release()
    release()
    assert active() is False
    with responsive_gui():
        assert sys.getswitchinterval() == BUSY_INTERVAL
    assert sys.getswitchinterval() == before


def test_it_is_claimed_from_several_threads_without_losing_count():
    started = threading.Barrier(5)

    def worker():
        started.wait()
        with responsive_gui():
            pass

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert active() is False


def test_the_pipeline_worker_holds_it_for_the_length_of_the_run():
    """The claim is on the RUN, not on the application."""
    pytest.importorskip("PySide6")
    import inspect

    from spacr.qt import bridge

    source = inspect.getsource(bridge.PipelineWorker.run)
    assert "responsive_gui()" in source


def test_a_headless_run_pays_nothing_for_a_window_that_is_not_there():
    """A process-wide 1 ms would tax every `spacr-run` in the interpreter."""
    before = sys.getswitchinterval()
    import spacr.qt.gil_priority          # noqa: F401  (importing is the test)
    assert sys.getswitchinterval() == before


def test_the_backdrop_shades_on_its_own_thread_when_it_has_a_frame_to_hand():
    """The half that was already built, and the half that fixes numpy runs."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.ambient import AmbientWidget, _BufferedEngine

    app = QApplication.instance() or QApplication([])
    widget = AmbientWidget()
    widget.resize(320, 240)
    widget.show()
    widget.start()
    try:
        if isinstance(widget._engine, _BufferedEngine):
            assert widget._producer_box[0] is not None
        widget.stop()
        # And it costs nothing while stopped: no thread, no frame held.
        assert widget._producer_box[0] is None
    finally:
        widget.stop()
        widget.deleteLater()
        app.processEvents()



# ---------------------------------------------------------------------------
# 284, 2026-09-26: the application-wide event filters share one hub.
#
# Every Python event filter on the QApplication was called for every event in
# the process. A screen's first stylesheet delivers ~7 events per widget
# (6,288 on Make Masks), so thirteen filters turned one setStyleSheet into
# ~80,000 C++ -> Python crossings: 650-700 ms of thread CPU against 200-290 ms
# behind the hub (same screen, same process, load 90-100). These tests hold
# the contract the hub has to keep so the filters behind it behave exactly as
# they did when Qt called them itself.
# ---------------------------------------------------------------------------


def _hub_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _recorder(name, calls, *, consume=False, raises=False):
    from PySide6.QtCore import QObject

    class _Recorder(QObject):
        def eventFilter(self, watched, event):  # noqa: N802
            calls.append((name, event.type()))
            if raises:
                raise RuntimeError("a watcher that fails")
            return consume

    return _Recorder()


def _deliver(kind):
    from PySide6.QtCore import QCoreApplication, QEvent, QObject

    target = QObject()
    return QCoreApplication.sendEvent(target, QEvent(kind))


def test_a_watcher_is_asked_only_about_the_kinds_it_named():
    from PySide6.QtCore import QEvent

    from spacr.qt.gil_priority import (_stop_watching_application_events,
                                       _watch_application_events)

    app = _hub_app()
    calls = []
    watcher = _recorder("a", calls)
    assert _watch_application_events(app, watcher, (QEvent.Type.User,))
    try:
        _deliver(QEvent.Type.PaletteChange)
        _deliver(QEvent.Type.FontChange)
        assert calls == []
        _deliver(QEvent.Type.User)
        assert calls == [("a", QEvent.Type.User)]
    finally:
        assert _stop_watching_application_events(app, watcher)
    _deliver(QEvent.Type.User)
    assert calls == [("a", QEvent.Type.User)]


def test_the_last_registered_is_asked_first_and_true_ends_the_event():
    from PySide6.QtCore import QEvent

    from spacr.qt.gil_priority import (_application_watchers,
                                       _stop_watching_application_events,
                                       _watch_application_events)

    app = _hub_app()
    calls = []
    first = _recorder("first", calls)
    second = _recorder("second", calls, consume=True)
    kind = QEvent.Type.User
    _watch_application_events(app, first, (kind,))
    _watch_application_events(app, second, (kind,))
    try:
        watchers = _application_watchers(app)
        assert watchers.index(second) < watchers.index(first)
        assert _deliver(kind) is True
        assert [name for name, _kind in calls] == ["second"]

        calls.clear()
        _watch_application_events(app, first, (kind,))
        _deliver(kind)
        assert [name for name, _kind in calls] == ["first", "second"]
    finally:
        _stop_watching_application_events(app, first)
        _stop_watching_application_events(app, second)


def test_a_watcher_removed_mid_event_is_not_asked_about_it():
    from PySide6.QtCore import QEvent, QObject

    from spacr.qt.gil_priority import (_stop_watching_application_events,
                                       _watch_application_events)

    app = _hub_app()
    kind = QEvent.Type.User
    calls = []
    later = _recorder("later", calls)

    class _Remover(QObject):
        def eventFilter(self, watched, event):  # noqa: N802
            calls.append(("remover", event.type()))
            _stop_watching_application_events(app, later)
            return False

    remover = _Remover()
    _watch_application_events(app, later, (kind,))
    _watch_application_events(app, remover, (kind,))
    try:
        _deliver(kind)
        assert [name for name, _kind in calls] == ["remover"]
    finally:
        _stop_watching_application_events(app, remover)
        _stop_watching_application_events(app, later)


def test_a_destroyed_watcher_is_skipped_and_a_failing_one_does_not_stop_the_rest(
        monkeypatch):
    from PySide6.QtCore import QEvent
    from shiboken6 import delete

    from spacr.qt.gil_priority import (_stop_watching_application_events,
                                       _watch_application_events)

    app = _hub_app()
    kind = QEvent.Type.User
    calls = []
    hooked = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda *exc_info: hooked.append(exc_info[0]))
    survivor = _recorder("survivor", calls)
    failing = _recorder("failing", calls, raises=True)
    doomed = _recorder("doomed", calls)
    _watch_application_events(app, survivor, (kind,))
    _watch_application_events(app, failing, (kind,))
    _watch_application_events(app, doomed, (kind,))
    try:
        delete(doomed)
        _deliver(kind)
        assert [name for name, _kind in calls] == ["failing", "survivor"]
        assert hooked == [RuntimeError]
    finally:
        _stop_watching_application_events(app, failing)
        _stop_watching_application_events(app, survivor)


def test_a_stand_in_application_gets_the_filter_installed_directly():
    from spacr.qt.gil_priority import (_stop_watching_application_events,
                                       _watch_application_events)

    class _StandIn:
        def __init__(self):
            self.installed = []

        def installEventFilter(self, obj):  # noqa: N802
            self.installed.append(obj)

        def removeEventFilter(self, obj):  # noqa: N802
            self.installed.remove(obj)

    stand_in = _StandIn()
    watcher = object()
    assert _watch_application_events(stand_in, watcher, ())
    assert stand_in.installed == [watcher]
    assert _stop_watching_application_events(stand_in, watcher)
    assert stand_in.installed == []


def test_spacrs_application_filters_are_behind_the_hub_and_none_hears_a_restyle():
    """The storm a stylesheet sets off reaches none of them."""
    from PySide6.QtCore import QEvent

    from spacr.qt.button_roles import install_button_roles
    from spacr.qt.gil_priority import _application_event_hub
    from spacr.qt.i18n import install_dialog_translation
    from spacr.qt.live_zoom import install_column_text_scale, install_live_zoom
    from spacr.qt.tooltip_policy import install_tooltip_policy
    from spacr.qt.widgets.feature_dictionary import install_context_menu_filter
    from spacr.qt.widgets.field_fade import install_field_fade
    from spacr.qt.widgets.glass import install_glass_everywhere

    app = _hub_app()
    install_button_roles(app)
    install_dialog_translation(app)
    install_column_text_scale(app)
    install_live_zoom(app)
    install_tooltip_policy(app)
    install_context_menu_filter(app)
    install_field_fade(app)
    install_glass_everywhere(app)

    hub = _application_event_hub(app, create=False)
    assert hub is not None
    watched = {type(watcher).__name__ for watcher in hub.watchers()}
    assert {"_SemanticButtonFilter", "_DialogTranslationFilter",
            "ColumnTextScale", "LiveZoomFilter", "_TooltipFilter",
            "FeatureHelpFilter", "_FieldFadeFilter", "_GlassInstaller",
            "_CursorPolicy"} <= watched
    storm = {QEvent.Type.PaletteChange, QEvent.Type.FontChange,
             QEvent.Type.DynamicPropertyChange, QEvent.Type.Resize,
             QEvent.Type.Move, QEvent.Type.ChildAdded}
    assert not storm & set(hub._by_kind)
