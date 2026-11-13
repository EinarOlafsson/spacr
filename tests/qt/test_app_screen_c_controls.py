"""Toggles, layout memory and the run's own reporting, on a partial screen.

Every card on a module screen is optional: the sweep, the live preview, the
UMAP explorer, the hyperparameter panel, the runtime splitter, the two hint
strips. A toggle wired to a card this module does not build is the ordinary
case, so each handler answers "is the thing I am about to move here?", and
the ones that seed a card from the form must still open it when seeding
fails. One regression screen is built per test and shared by its cases, each
of which makes its overrides inside ``monkeypatch.context()``.
"""
from __future__ import annotations

import os
import time
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QSplitter, QWidget                 # noqa: E402

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _splitter(qtbot, panes=2):
    splitter = QSplitter()
    qtbot.addWidget(splitter)
    for _ in range(panes):
        splitter.addWidget(QWidget())
    return splitter


def test_the_runtime_split_is_remembered_or_left_at_its_default(
        screen, monkeypatch, subtests, qtbot):
    """The layout store is a convenience; the splitter is always adopted.

    Invariant: no splitter remembers nothing; a missing layout store, a
    saved blob that cannot be read, or a store that refuses a save during a
    drag all leave the splitter adopted (the drag tries the save once, under
    ``<app>::runtime``); a readable saved state is restored.
    """
    import builtins

    from spacr.qt.widgets import console_panel

    with subtests.test(case="no splitter"):
        screen._remember_runtime_splitter(None)
        assert screen._runtime_splitter is None

    with subtests.test(case="no layout store"), monkeypatch.context() as m:
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.endswith("console_panel"):
                raise ImportError("no layout store")
            return real_import(name, *args, **kwargs)

        splitter = _splitter(qtbot, panes=0)
        m.setattr(builtins, "__import__", refuse)
        screen._remember_runtime_splitter(splitter)
        assert screen._runtime_splitter is splitter

    with subtests.test(case="saved blob unreadable"), \
            monkeypatch.context() as m:
        splitter = _splitter(qtbot, panes=0)
        m.setattr(console_panel, "get_split_state", _boom)
        screen._remember_runtime_splitter(splitter)
        assert screen._runtime_splitter is splitter

    with subtests.test(case="a save refused mid-drag"), \
            monkeypatch.context() as m:
        splitter = _splitter(qtbot)
        attempts = []

        def refusing_store(key, state):
            attempts.append(key)
            raise RuntimeError("the layout store is unavailable")

        m.setattr(console_panel, "get_split_state", lambda _k: None)
        m.setattr(console_panel, "set_split_state", refusing_store)
        screen._remember_runtime_splitter(splitter)
        splitter.splitterMoved.emit(10, 1)
        assert attempts == ["regression::runtime"]
        assert screen._runtime_splitter is splitter

    with subtests.test(case="a saved state is restored"), \
            monkeypatch.context() as m:
        splitter = _splitter(qtbot)
        state = splitter.saveState()
        restored = []
        m.setattr(console_panel, "get_split_state", lambda key: state)
        m.setattr(console_panel, "set_split_state", lambda key, value: None)
        m.setattr(splitter, "restoreState",
                  lambda value: restored.append(value) or True)
        ns = types.SimpleNamespace(app_key="mask",
                                   RUNTIME_SPLIT_SUFFIX=".runtime",
                                   _runtime_splitter=None)
        AppScreen._remember_runtime_splitter(ns, splitter)
        assert restored == [state]


def test_optional_cards_follow_their_toggles_only_when_present(
        screen, monkeypatch, subtests, qtbot):
    """A toggle moves its card if the module built one, and nothing else.

    Invariant: with no live-preview, preview, sweep card, hint strip or
    explorer the toggle changes nothing; the live-preview card shows and
    hides with its switch; the sweep opens seeded from the form, and still
    opens when seeding raises; only the image-UMAP screen forwards the GPU
    switch; a static figure flips the interactive switch only when an
    embedding is ready and a switch exists, even when the console cannot
    take the explanation; a pinned category comes back when the pointer
    leaves; the late-caption watcher installs with hosts missing.
    """
    def card():
        widget = QWidget()
        qtbot.addWidget(widget)
        widget.hide()
        return widget

    with subtests.test(case="no hint strips"), monkeypatch.context() as m:
        m.setattr(screen, "_hint_strip", None, raising=False)
        m.setattr(screen, "_category_hint", None, raising=False)
        screen._sync_hint_strip_height()
        screen._sync_category_hint_height()
        screen.show_category_hint("Paths")
        screen.clear_category_hint()
        assert (screen._hint_strip, screen._category_hint) == (None, None)

    with subtests.test(case="pinned category comes back"), \
            monkeypatch.context() as m:
        m.setattr(screen, "_category_hint_pinned", "Paths", raising=False)
        screen.clear_category_hint()
        assert "PATHS" in screen._category_hint.text().upper()

    absent = [
        ("no live preview card", "_live_preview_card", None,
         lambda s: s._on_lp_switch(True)),
        ("no preview card", "_preview_card_attr", "",
         lambda s: s._on_preview_switch(True)),
        ("no sweep card", "_sweep_card", None,
         lambda s: s._on_sweep_switch(True)),
    ]
    for name, attr, value, toggle in absent:
        with subtests.test(case=name), monkeypatch.context() as m:
            m.setattr(screen, attr, value, raising=False)
            toggle(screen)
            assert getattr(screen, attr) == value

    with subtests.test(case="live preview follows its toggle"), \
            monkeypatch.context() as m:
        live = card()
        m.setattr(screen, "_live_preview_card", live, raising=False)
        screen._on_lp_switch(True)
        assert not live.isHidden()
        screen._on_lp_switch(False)
        assert live.isHidden()

    for name, seed, seeds in [("sweep seeded from the form", "record", 1),
                              ("sweep seeding raises", _boom, 0)]:
        with subtests.test(case=name), monkeypatch.context() as m:
            sweep, seeded = card(), []
            m.setattr(screen, "_sweep_card", sweep, raising=False)
            m.setattr(screen, "_sweep", types.SimpleNamespace(
                apply_settings=seeded.append if seed == "record" else seed),
                raising=False)
            screen._on_sweep_switch(True)
            assert not sweep.isHidden()
            assert len(seeded) == seeds
            assert all(isinstance(values, dict) for values in seeded)

    with subtests.test(case="GPU switch off the image-UMAP screen"), \
            monkeypatch.context() as m:
        requests = []
        m.setattr(screen, "_hyperparam", types.SimpleNamespace(
            request_gpu_enabled=lambda *a, **k: requests.append((a, k))),
            raising=False)
        screen._on_umap_gpu_switch(True)
        assert requests == []

    with subtests.test(case="no explorer for the interactive toggle"), \
            monkeypatch.context() as m:
        actions = []
        m.setattr(screen, "_umap_explorer", None, raising=False)
        m.setattr(screen, "_figure_queue", types.SimpleNamespace(
            hide=lambda: actions.append("hide"),
            show=lambda: actions.append("show"), count=lambda: 1),
            raising=False)
        screen._on_interactive_switch(True)
        assert actions == []

    # (case, interactive switch, console append_notice, switched, said)
    static = [
        ("static figure with no switch", None, "record", [], []),
        ("console refuses the explanation", "switch", _boom, [True], None),
    ]
    for name, switch, notice, switched_expected, said_expected in static:
        with subtests.test(case=name), monkeypatch.context() as m:
            said, switched = [], []
            m.setattr(screen, "_console", types.SimpleNamespace(
                append_notice=said.append if notice == "record" else notice))
            m.setattr(screen, "_umap_payload_ready", True, raising=False)
            m.setattr(screen, "_interactive_switch", None if switch is None
                      else types.SimpleNamespace(isChecked=lambda: False,
                                                 setChecked=switched.append),
                      raising=False)
            screen._on_static_figure_clicked()
            assert switched == switched_expected
            if said_expected is not None:
                assert said == said_expected

    with subtests.test(case="late captions with hosts missing"), \
            monkeypatch.context() as m:
        m.setattr(screen, "_runtime_wrap", None, raising=False)
        m.setattr(screen, "_body_splitter", None, raising=False)
        m.setattr(screen, "_late_caption_watcher", None, raising=False)
        screen._watch_for_late_captions()
        assert screen._late_caption_watcher is not None


def test_preferences_and_buttons_never_break_the_screen(
        screen, monkeypatch, subtests, caplog):
    """A preference or a button failing is said or logged, never raised.

    Invariant: the AI switch is set on only when the preference says on and
    is readable (a switch that refuses is not fatal; off means untouched);
    a console that will not copy says so in the console; Preferences that
    will not open are logged; settings that will not collect stop a remote
    submit with a warning and send nothing.
    """
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt import preferences

    def unreadable():
        raise RuntimeError("the preference store is unavailable")

    def refusing_switch(checks):
        def set_checked(value):
            checks.append(value)
            raise RuntimeError("the switch has gone")
        return set_checked

    # (case, preference, switch refuses, values the switch was given)
    ai = [
        ("preference unreadable", unreadable, False, []),
        ("switch refuses", lambda: True, True, [True]),
        ("preference off", lambda: False, False, []),
        ("preference on", lambda: True, False, [True]),
    ]
    for name, preference, refuses, checked in ai:
        with subtests.test(case=f"AI default: {name}"), \
                monkeypatch.context() as m:
            checks = []
            m.setattr(preferences, "get_ai_on_by_default", preference)
            m.setattr(screen, "_ai_switch", types.SimpleNamespace(
                setChecked=refusing_switch(checks) if refuses
                else checks.append), raising=False)
            screen._apply_ai_default()
            assert checks == checked

    with subtests.test(case="console will not copy"), \
            monkeypatch.context() as m:
        errors = []
        m.setattr(screen, "_console", types.SimpleNamespace(
            copy_all=_boom, append_error=errors.append))
        screen._on_copy_console()
        assert errors and "Could not copy the console" in errors[0]

    with subtests.test(case="Preferences will not open"), \
            monkeypatch.context() as m:
        m.setattr(preferences, "PreferencesDialog", _boom)
        caplog.clear()
        with caplog.at_level("ERROR"):
            screen._open_preferences_dialog()
        assert any("could not open Preferences" in record.message
                   for record in caplog.records)

    with subtests.test(case="remote submit with unreadable settings"), \
            monkeypatch.context() as m:
        warned, sent = [], []
        m.setattr(QMessageBox, "warning",
                  staticmethod(lambda *args, **kwargs: warned.append(args)))
        m.setattr(screen._settings_model, "collect", _boom)
        screen.remote_submit_requested.connect(lambda *a: sent.append(a))
        screen._on_remote_submit()
        assert warned and sent == []


def test_the_run_reports_its_progress_once_per_mark(screen, monkeypatch,
                                                    subtests):
    """The heartbeat and the design scan are sentences beside the run.

    Invariant: the heartbeat says nothing before a start is recorded or at
    a mark already spoken; an ordinary run says only how long it has been
    going (with its elapsed time); a slow fit explains the single core. A
    design scan says nothing for an unusable answer, reports a partial
    design with its reason, and a scan that cannot be submitted is tried
    once and is not fatal.
    """
    # (case, seconds since start or None, slow fit, sentence expected)
    beats = [
        ("no start recorded", None, False, None),
        ("mark already spoken", 0.0, False, None),
        ("ordinary run", 400.0, False, "Still running"),
        ("slow fit", 400.0, True, "runs mostly on one CPU core"),
    ]
    for name, since, slow, sentence in beats:
        with subtests.test(case=f"heartbeat: {name}"), \
                monkeypatch.context() as m:
            said = []
            m.setattr(screen, "_console", types.SimpleNamespace(
                append_notice=lambda text, **kw: said.append((text, kw))))
            m.setattr(screen, "_thread",
                      types.SimpleNamespace(isRunning=lambda: True))
            m.setattr(screen, "_run_started_at",
                      None if since is None else time.time() - since,
                      raising=False)
            m.setattr(screen, "_slow_fit", slow, raising=False)
            # The spoken mark is screen state; each case starts unspoken.
            m.setattr(screen, "_heartbeat_said", 0.0, raising=False)
            screen._on_heartbeat()
            if sentence is None:
                assert said == []
            else:
                assert len(said) == 1 and sentence in said[0][0]
                if not slow:
                    assert said[0][1]["elapsed"] == "6 min 40 s"

    with subtests.test(case="design scan: nothing usable"), \
            monkeypatch.context() as m:
        said = []
        m.setattr(screen, "_console",
                  types.SimpleNamespace(append_notice=said.append))
        screen._on_design_scanned(None)
        screen._on_design_scanned("not a design")
        assert said == []

    with subtests.test(case="design scan: partial design"), \
            monkeypatch.context() as m:
        said = []
        m.setattr(screen, "_console", types.SimpleNamespace(
            append_notice=lambda text, **kw: said.append((text, kw))))
        screen._on_design_scanned(
            {"guides": 4096, "rows": 50000, "note": "no gene column"})
        assert "distinct gRNAs" in said[0][0]
        assert said[0][1]["why"] == "no gene column"

    with subtests.test(case="design scan cannot be submitted"), \
            monkeypatch.context() as m:
        submissions = []

        def refusing_submit(work, callback):
            submissions.append(callback)
            raise RuntimeError("the job runner is shutting down")

        m.setattr(screen._jobs, "submit", refusing_submit)
        screen._announce_the_fit({"src": "/data"})
        assert submissions == [screen._on_design_scanned]


def test_a_partial_screen_still_builds_and_closes(qtbot, monkeypatch,
                                                  subtests):
    """The results panel is a convenience of the module, not the module.

    Invariant: a saved fold layout that cannot be restored still gives the
    measurements tab and the results tabs; a results half that will not
    build falls back to a parented figure queue with no results panel,
    figure stack or Cells view; assigning an attribute of a part the screen
    still owes builds the part first and the assignment wins; closing with
    a worker thread that cannot be asked, or a figure queue that cannot be
    cleared, still finishes the close.
    """
    from spacr.qt.screens import app_screen as aps
    from spacr.qt.widgets import cell_montage_view
    from spacr.qt.widgets import measurement_scan_panel as scan_module

    def fresh():
        built = AppScreen("regression")
        qtbot.addWidget(built)
        return built

    with subtests.test(case="assigning an owed attribute builds it first"):
        built = fresh()
        assert built._part_is_owed(aps._REGRESSION_RESULTS)
        panel = types.SimpleNamespace(results_frame=lambda: None)
        built._results_panel = panel
        assert not built._part_is_owed(aps._REGRESSION_RESULTS)
        assert built._results_panel is panel
        assert built._results_tabs is not None

    for name, attr, value in [
            ("closing with a thread that cannot be asked", "_thread",
             object()),
            ("closing with a queue that cannot be cleared", "_figure_queue",
             types.SimpleNamespace())]:
        with subtests.test(case=name):
            built = fresh()
            setattr(built, attr, value)
            built.close()
            assert built._thread is None
            assert built._backdrops_ready is False

    with subtests.test(case="fold layout cannot be restored"), \
            monkeypatch.context() as m:
        m.setattr(scan_module.MeasurementScanPanel, "restore_section_layout",
                  _boom)
        built = AppScreen("regression")
        qtbot.addWidget(built)
        assert built._scan_panel is not None
        assert built._results_tabs is not None

    with subtests.test(case="results half will not build"), \
            monkeypatch.context() as m:
        m.setattr(cell_montage_view, "CellMontageView", _boom)
        built = AppScreen("regression")
        qtbot.addWidget(built)
        assert built._results_panel is None
        assert built._figures_stack is None
        assert built._cell_montage is None
        assert built._figure_queue.parent() is not None
