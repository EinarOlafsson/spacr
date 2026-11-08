"""The Embeddings screen's export refusals and the collector's Qt-free fallbacks.

The export step says why it did not start (no source, no folder chosen, a
spec that does not build) and reports an embedding failure in its status
line. The garbage-collection policy treats a missing application, a missing
watcher and a failing Qt call as "no event in progress" or "sweep owed"
rather than raising.
"""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _screen(qtbot):
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def _with_source(screen, tmp_path):
    folder = tmp_path / "crops"
    folder.mkdir()
    screen._where.setCurrentIndex(1)
    screen._path.setText(str(folder))
    screen._on_path_changed()


def test_export_without_a_chosen_folder_starts_nothing(qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    screen = _screen(qtbot)
    _with_source(screen, tmp_path)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    assert screen._embed_and_export() == ""
    assert screen._export_store.isEnabled()


def test_export_asks_for_a_folder_when_none_is_given(qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    import spacr.embeddings as emb

    screen = _screen(qtbot)
    _with_source(screen, tmp_path)
    asked = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: asked.append(1) or str(tmp_path)))

    def boom(*a, **k):
        raise RuntimeError("stop")

    monkeypatch.setattr(emb, "embed_store", boom)
    assert screen._embed_and_export() == str(tmp_path)
    assert asked == [1]


def test_export_reports_a_spec_that_does_not_build(qtbot, monkeypatch, tmp_path):
    screen = _screen(qtbot)
    _with_source(screen, tmp_path)

    def bad():
        raise ValueError("the spec is wrong")

    monkeypatch.setattr(screen, "spec", bad)
    monkeypatch.setattr(screen._foundation, "currentData", lambda *a: "other")
    assert screen._embed_and_export(str(tmp_path)) == ""
    assert screen._status.text() == "the spec is wrong"


def test_export_reports_an_embedding_failure_in_the_status_line(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    screen = _screen(qtbot)
    _with_source(screen, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("no encoder")

    monkeypatch.setattr(emb, "embed_store", boom)
    assert screen._embed_and_export(str(tmp_path)) == str(tmp_path)
    assert screen._status.text() == "RuntimeError: no encoder"
    assert screen._export_store.isEnabled()


def test_export_with_the_automatic_dino_choice_uses_the_dino_note(qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    screen = _screen(qtbot)
    _with_source(screen, tmp_path)
    seen = []
    monkeypatch.setattr(screen._foundation, "currentData", lambda *a: "dino_auto")
    monkeypatch.setattr(screen, "_dino_note", lambda: ("DINO-SPEC", ""))

    def record(query, run, **k):
        seen.append(run)
        raise RuntimeError("stop")

    monkeypatch.setattr(emb, "embed_store", record)
    screen._embed_and_export(str(tmp_path))
    assert seen == ["DINO-SPEC"]


@pytest.fixture
def show_watch(monkeypatch):
    from spacr.qt import gc_policy

    for key, value in (("watcher", None), ("class", None), ("frozen", None),
                       ("lost", False)):
        monkeypatch.setitem(gc_policy._SHOW_WATCH, key, value)
    return gc_policy


def test_a_known_show_watcher_class_is_reused(show_watch):
    sentinel = type("Known", (), {})
    show_watch._SHOW_WATCH["class"] = sentinel
    assert show_watch._window_show_watcher_class() is sentinel


def test_watching_window_shows_makes_one_watcher_and_survives_a_hub_failure(
        show_watch, qapp, monkeypatch):
    from spacr.qt import gil_priority

    calls = []
    monkeypatch.setattr(gil_priority, "_watch_application_events",
                        lambda app, watcher, kinds: calls.append(watcher))
    show_watch._watch_window_shows(qapp)
    first = show_watch._SHOW_WATCH["watcher"]
    assert first is not None and calls == [first]
    show_watch._watch_window_shows(qapp)
    assert calls == [first, first]

    def broken(*a, **k):
        raise RuntimeError("hub gone")

    monkeypatch.setattr(gil_priority, "_watch_application_events", broken)
    show_watch._watch_window_shows(qapp)
    assert show_watch._SHOW_WATCH["watcher"] is first


def test_stopping_the_show_watch_clears_state_with_or_without_a_watcher(
        show_watch, qapp, monkeypatch):
    from spacr.qt import gil_priority

    show_watch._SHOW_WATCH["frozen"] = 1
    show_watch._stop_watching_window_shows()
    assert show_watch._SHOW_WATCH["frozen"] is None

    stopped = []
    monkeypatch.setattr(gil_priority, "_stop_watching_application_events",
                        lambda app, watcher: stopped.append(watcher))
    show_watch._SHOW_WATCH["watcher"] = "w"
    show_watch._stop_watching_window_shows()
    assert stopped == ["w"]

    def broken(*a, **k):
        raise RuntimeError("gone")

    monkeypatch.setattr(gil_priority, "_stop_watching_application_events", broken)
    show_watch._stop_watching_window_shows()
    assert show_watch._SHOW_WATCH["frozen"] is None


def test_a_sweep_is_owed_when_no_watcher_hears_the_shows(show_watch, qapp):
    assert show_watch._a_sweep_is_owed_before_the_freeze() is True


def test_a_sweep_is_owed_when_the_watcher_is_not_installed_or_no_app(show_watch, qapp, monkeypatch):
    from spacr.qt import gil_priority

    show_watch._SHOW_WATCH["watcher"] = object()
    monkeypatch.setattr(gil_priority, "_application_watchers", lambda app: [])
    assert show_watch._a_sweep_is_owed_before_the_freeze() is True
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
    assert show_watch._a_sweep_is_owed_before_the_freeze() is True


def test_no_application_means_no_pointer_pressed_and_no_event_in_progress(show_watch, monkeypatch):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    assert show_watch._pointer_pressed() is False
    assert show_watch._mid_event() is False


def test_a_failing_qt_call_means_no_event_in_progress(show_watch, monkeypatch):
    def broken():
        raise RuntimeError("qt gone")

    monkeypatch.setattr(show_watch, "_pointer_pressed", broken)
    assert show_watch._mid_event() is False


def test_a_pointer_check_that_fails_does_not_stop_the_tick_from_deciding(show_watch, monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            return True
        raise RuntimeError("qt gone")

    monkeypatch.setattr(show_watch, "_pointer_pressed", flaky)
    monkeypatch.setattr(show_watch, "_mid_event_since", [None])
    assert show_watch._collect_on_tick() == -1
