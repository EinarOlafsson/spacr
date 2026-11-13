"""Which run is on screen: beside, removed, a trial, a workspace, a finished run.

Two kinds of target appear. Most cases drive one ``AppScreen`` method on a
stand-in ``self`` that holds only the attributes the method reads -- enough to
state which notice the user is given and what is handed back without building
a screen. The rest drive the same methods on a real regression screen, where
the method also reaches the screen's own collaborators; those reuse one screen
per test with every override inside ``monkeypatch.context()``.
"""
from __future__ import annotations

import os
import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    lines = []
    widget._console = types.SimpleNamespace(
        said=lines, append_notice=lines.append, append_stdout=lines.append)
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


class _Console:
    def __init__(self):
        self.notices = []
        self.stdout = []

    def append_notice(self, text, **kw):
        self.notices.append(text.format(**kw) if kw else text)

    def append_stdout(self, text, **_kw):
        self.stdout.append(text)


def _ns(**attrs):
    ns = types.SimpleNamespace(_console=_Console(), **attrs)
    ns._same_run_folder = AppScreen._same_run_folder
    ns.MAX_LIVE_RUNS = AppScreen.MAX_LIVE_RUNS
    ns.LIVE_RUNS_NOTE = AppScreen.LIVE_RUNS_NOTE
    ns.live_run_count = lambda: AppScreen.live_run_count(ns)
    ns._say = lambda msg: AppScreen._say(ns, msg)
    ns._the_run_did_not_open = lambda why: AppScreen._the_run_did_not_open(
        ns, why)
    return ns


class _Panel:
    def __init__(self, folder="/runs/a", loads=True, grab=None):
        self._folder = folder
        self._loads = loads
        self.volcano = types.SimpleNamespace(grab=grab or (lambda: None))
        self.parent_cleared = False
        self.deleted = False

    def run_folder(self):
        return self._folder

    def run_name(self):
        return "run-b"

    def load(self, folder):
        self._folder = folder
        return self._loads

    def setParent(self, parent):
        self.parent_cleared = parent is None

    def deleteLater(self):
        self.deleted = True


class _Split:
    def __init__(self):
        self.added = []
        self.sizes = None

    def addWidget(self, panel):
        self.added.append(panel)

    def setSizes(self, sizes):
        self.sizes = sizes


def test_opening_a_second_run_beside_answers_every_gesture(
        screen, monkeypatch, subtests, tmp_path):
    """The gesture comes from a menu, so it opens a run or says why not.

    Invariant: a second run opens beside the loaded one (added to the split,
    halves equal, Results tab raised, named in the console); a run that will
    not load is detached and deleted rather than left beside; no folder, the
    run already on screen, and a third live run are each refused with their
    own reason (a folder array is never the run on screen); without a
    loaded run nothing opens and nothing is said.
    """
    from spacr.qt.widgets import regression_results

    # (case, record, compare panel, panel the factory makes, returns, notice)
    cases = [
        ("second run opens beside", {"folder": "/runs/b"}, None,
         _Panel("/x"), True, "run-b is open beside the loaded run"),
        ("run will not load", "/runs/b", None, _Panel(loads=False), False,
         "No results in /runs/b"),
        ("record with no folder", {"folder": ""}, None, None, False,
         "no folder on disk"),
        ("run already on screen", "/runs/a", None, None, False,
         "already on screen"),
        ("a third live run", "/runs/b", _Panel("/runs/c"), None, False,
         "Two runs can be live"),
    ]
    for name, record, compare, made, returns, notice in cases:
        with subtests.test(case=name), monkeypatch.context() as m:
            if made is not None:
                m.setattr(regression_results, "RegressionResultsPanel",
                          lambda parent, made=made: made)
            split = _Split()
            ns = _ns(_results_panel=_Panel("/runs/a"), _results_split=split,
                     _compare_panel=compare)
            raised = []
            ns._raise_the_results_tab = lambda: raised.append(True)
            assert AppScreen.open_run_beside(ns, record) is returns
            assert notice in ns._console.notices[-1]
            if returns:
                assert ns._compare_panel is made
                assert split.added == [made] and split.sizes == [1, 1]
                assert raised == [True]
            elif made is not None:
                assert made.parent_cleared and made.deleted
                assert ns._compare_panel is None
            else:
                assert ns._compare_panel is compare and split.added == []

    import numpy as np

    # A folder read off a concatenated frame can be an array of paths, which
    # is ambiguous when tested for emptiness: it is "not the same run".
    for name, left, same in [
            ("folder array matches no run",
             np.array(["/runs/a", "/runs/other"]), False),
            ("the same folder is the same run", "/runs/a", True)]:
        with subtests.test(case=name):
            assert AppScreen._same_run_folder(left, "/runs/a") is same

    with subtests.test(case="stand-in: no loaded run"):
        ns = _ns(_results_panel=None, _results_split=None)
        assert AppScreen.open_run_beside(ns, "/runs/b") is False
        assert ns._console.notices == []

    # The same refusals on a real screen, which reads its own split.
    real = [
        ("screen: record with no folder", {}, {"folder": ""},
         "nothing to open beside"),
        ("screen: run already on screen",
         {"_results_panel": types.SimpleNamespace(
             run_folder=lambda: str(tmp_path))},
         {"folder": str(tmp_path)}, "already on screen"),
        ("screen: no results panel", {"_results_panel": None},
         {"folder": "/data/run"}, None),
    ]
    for name, attrs, record, said in real:
        with subtests.test(case=name), monkeypatch.context() as m:
            screen._console.said.clear()
            for attr, value in attrs.items():
                m.setattr(screen, attr, value)
            assert screen.open_run_beside(record) is False
            if said is None:
                assert screen._console.said == []
            else:
                assert any(said in line for line in screen._console.said)


def test_closing_a_run_beside_keeps_only_a_real_photograph(
        screen, monkeypatch, subtests, qtbot):
    """The still is a convenience; the panel closing is not.

    Invariant: closing answers whether there was a second run; the panel is
    always gone afterwards; a photograph is kept (under the absolute folder)
    only when ``grab`` returned one, and is looked up the same way; the live
    run count is the number of panels that exist.
    """
    from PySide6.QtWidgets import QWidget

    class Photo:
        def isNull(self):
            return False

    photo = Photo()
    key = os.path.abspath("/runs/b")
    for name, grab, kept in [("a photograph is taken", lambda: photo,
                              {key: photo}),
                             ("grab raises", _boom, {})]:
        with subtests.test(case=name):
            ns = types.SimpleNamespace(
                _compare_panel=_Panel("/runs/b", grab=grab),
                _run_photographs={})
            assert AppScreen.close_run_beside(ns) is True
            assert ns._run_photographs == kept
            assert ns._compare_panel is None

    with subtests.test(case="nothing beside"):
        assert AppScreen.close_run_beside(
            types.SimpleNamespace(_compare_panel=None)) is False

    with subtests.test(case="screen: panel will not be photographed"), \
            monkeypatch.context() as m:
        panel = QWidget()
        qtbot.addWidget(panel)
        panel.run_folder = lambda: "/data/run"
        panel.volcano = types.SimpleNamespace(grab=_boom)
        m.setattr(screen, "_compare_panel", panel, raising=False)
        screen.close_run_beside()
        assert screen._compare_panel is None
        assert os.path.abspath("/data/run") not in screen._run_photographs

    lookups = types.SimpleNamespace(_run_photographs={key: "still"})
    for folder, expected in [("/runs/b", "still"), ("", None),
                             ("/runs/none", None)]:
        with subtests.test(case=f"photograph of {folder!r}"):
            assert AppScreen.run_photograph(lookups, folder) == expected

    for attrs, count in [({}, 0), ({"_results_panel": object()}, 1),
                         ({"_results_panel": object(),
                           "_compare_panel": object()}, 2)]:
        with subtests.test(case=f"{count} live runs"):
            assert AppScreen.live_run_count(
                types.SimpleNamespace(**attrs)) == count


def test_a_removed_run_is_forgotten_everywhere(screen, monkeypatch, subtests):
    """Plot state, figures, photograph and the view beside are separate stores.

    Invariant: every removed record with a folder is forgotten by the panel
    (by folder) and the figure queue (by run label), its photograph dropped,
    the run beside closed when it was that run, and the grid refreshed; a
    store that raises does not stop the others; records with no folder and
    a screen with no panel or no records forget nothing.
    """
    photo_key = os.path.abspath("/runs/b")

    with subtests.test(case="stand-in: everything goes"):
        forgotten, refreshed = [], []
        ns = _ns(_results_panel=types.SimpleNamespace(
                     forget_run=forgotten.append),
                 _compare_panel=_Panel("/runs/b"),
                 _run_photographs={photo_key: "still"},
                 _figure_queue=types.SimpleNamespace(
                     forget_run=lambda label: forgotten.append(label) or True))
        ns.close_run_beside = lambda: setattr(ns, "_compare_panel", None)
        ns._queue_figure_grid_refresh = lambda: refreshed.append(True)
        AppScreen._on_runs_removed(
            ns, [{"folder": "/runs/b", "run": "B"}, {"folder": ""}, None])
        assert photo_key not in ns._run_photographs
        assert forgotten == ["/runs/b", "B"] and refreshed == [True]
        assert ns._compare_panel is None

    with subtests.test(case="stand-in: both stores refuse"):
        asked = []

        def refusing(store):
            def forget(value):
                asked.append((store, value))
                raise RuntimeError("busy")
            return forget

        ns = _ns(_results_panel=types.SimpleNamespace(
                     forget_run=refusing("panel")),
                 _compare_panel=None, _run_photographs={photo_key: "still"},
                 _figure_queue=types.SimpleNamespace(
                     forget_run=refusing("queue")))
        AppScreen._on_runs_removed(ns, [{"folder": "/runs/b", "run": "B"}])
        assert asked == [("panel", "/runs/b"), ("queue", "B")]
        assert ns._run_photographs == {}

    with subtests.test(case="stand-in: no panel, no records"):
        bare = types.SimpleNamespace()
        assert AppScreen._on_runs_removed(bare, [{"folder": "x"}]) is None
        assert vars(bare) == {}
        panel = object()
        without = types.SimpleNamespace(_results_panel=panel,
                                        _run_photographs={"x": 1})
        assert AppScreen._on_runs_removed(without, []) is None
        assert vars(without) == {"_results_panel": panel,
                                 "_run_photographs": {"x": 1}}

    def never(_value):
        pytest.fail("nothing should have been forgotten")

    # (case, panel forget, queue forget refuses, records, panel log, queue log)
    real = [
        ("screen: no panel", None, False, [{"folder": "/data/run"}], [], []),
        ("screen: no records", never, False, [], [], []),
        ("screen: rows with no folder", never, False, [{"folder": ""}, None],
         [], []),
        ("screen: panel refuses, figures still go", _boom, False,
         [{"folder": "/data/run", "run": "run 3"}], [], ["run 3"]),
        ("screen: queue refuses, photograph still goes", "record", True,
         [{"folder": "/data/run", "run": "run 3"}], ["/data/run"], ["run 3"]),
    ]
    for name, forget, queue_refuses, records, panel_log, queue_log in real:
        with subtests.test(case=name), monkeypatch.context() as m:
            panel_seen, queue_seen = [], []

            def queue_forget(label, refuse=queue_refuses):
                queue_seen.append(label)
                if refuse:
                    raise RuntimeError("the figure queue has gone")
                return True

            if forget is None:
                m.setattr(screen, "_results_panel", None)
            else:
                m.setattr(screen, "_results_panel", types.SimpleNamespace(
                    forget_run=panel_seen.append if forget == "record"
                    else forget))
            m.setattr(screen, "_compare_panel", None, raising=False)
            m.setattr(screen, "_figure_queue",
                      types.SimpleNamespace(forget_run=queue_forget))
            m.setitem(screen._run_photographs,
                      os.path.abspath("/data/run"), object())
            screen._on_runs_removed(records)
            assert panel_seen == panel_log
            assert queue_seen == queue_log
            gone = bool(panel_log or queue_log)
            assert (os.path.abspath("/data/run")
                    not in screen._run_photographs) is gone


def test_showing_a_trial_from_the_runs_tab(screen, monkeypatch, subtests,
                                           tmp_path):
    """A row of the Runs tab is shown, named, or refused -- once.

    Invariant: a running trial says it is still going, a failed one gives
    its reason, one with no folder on disk is named not shown, the loaded
    trial is not loaded twice (but its card is shown); a load is wired to
    ``load_finished`` once and a refused start stops before showing; a row
    that is not a record, or a screen with no results panel, is ignored; a
    load answer arriving with nothing pending loads no figures.
    """
    def stand_in(panel):
        shown = []
        ns = _ns(_results_panel=panel, _figures_card=types.SimpleNamespace(
            show=lambda: shown.append(True)))
        ns.shown = shown
        return ns

    said = [
        ("running", None, {"run": "T1", "status": "running"},
         ["T1 is still going"], []),
        ("failed", None, {"trial_id": 4, "status": "error",
                          "error_type": "ValueError", "error": "singular"},
         ["Trial 4 did not produce a regression", "singular"], []),
        ("no folder on disk", _Panel("/runs/a"),
         {"run": "T2", "folder": str(tmp_path / "gone")},
         ["no saved results on disk"], []),
        ("already loaded", _Panel(str(tmp_path)),
         {"run": "T3", "folder": str(tmp_path)}, ["already loaded"], [True]),
    ]
    for name, panel, record, words, shown in said:
        with subtests.test(case=name):
            ns = stand_in(panel)
            AppScreen._show_trial(ns, record)
            for word in words:
                assert word in ns._console.stdout[-1]
            assert ns.shown == shown

    with subtests.test(case="load wired once, refused start stops"):
        connected = []

        class Refusing(_Panel):
            load_finished = types.SimpleNamespace(
                connect=lambda slot: connected.append(slot))

            def start_load(self, folder):
                return False

        ns = stand_in(Refusing("/runs/a"))
        ns._on_trial_loaded = lambda ok: None
        AppScreen._show_trial(ns, {"run": "T4", "folder": str(tmp_path)})
        AppScreen._show_trial(ns, {"run": "T4", "folder": str(tmp_path)})
        assert len(connected) == 1
        assert ns._pending_trial == ("T4", str(tmp_path))
        assert ns.shown == []

    for record in ("not a record", {"run": "T5"}):
        with subtests.test(case=f"stand-in ignores {record!r}"):
            ns = stand_in(None)
            assert AppScreen._show_trial(ns, record) is None
            assert ns._console.stdout == [] and ns.shown == []
            assert not hasattr(ns, "_pending_trial")

    pending = ("the current run", "/data/current")
    with subtests.test(case="screen: not a record"), \
            monkeypatch.context() as m:
        m.setattr(screen, "_pending_trial", pending, raising=False)
        screen._show_trial("not a record")
        assert screen._pending_trial == pending

    with subtests.test(case="screen: no results panel"), \
            monkeypatch.context() as m:
        screen._console.said.clear()
        m.setattr(screen, "_results_panel", None)
        screen._show_trial({"run": "run 3", "status": "ok",
                            "folder": "/data/run"})
        assert screen._console.said == []

    with subtests.test(case="screen: load answer with nothing pending"), \
            monkeypatch.context() as m:
        loaded = []
        m.setattr(screen, "_pending_trial", None, raising=False)
        m.setattr(screen, "_load_trial_figures", loaded.append)
        screen._on_trial_loaded(True)
        assert loaded == [] and screen._pending_trial is None


def test_restoring_a_run_workspace_reports_what_happened(
        monkeypatch, subtests, tmp_path):
    """The workspace of a run is restored and reported, or the absence said.

    Invariant: a record with no folder restores nothing (empty report); a
    folder with no saved workspace says so; a saved one is restored and its
    report text printed; ``_say`` falls back to the log when the console
    raises or is missing.
    """
    import logging

    from spacr import workspace

    with subtests.test(case="no folder"):
        assert AppScreen.restore_run_workspace(types.SimpleNamespace(), {}) \
            == {"restored": [], "skipped": [], "files": []}

    with subtests.test(case="no saved workspace"):
        ns = _ns()
        assert AppScreen.restore_run_workspace(ns, str(tmp_path))[
            "restored"] == []
        assert "carries no saved workspace" in ns._console.stdout[-1]

    with subtests.test(case="restored and reported"), \
            monkeypatch.context() as m:
        m.setattr(workspace, "load", lambda folder: {"doc": 1})
        m.setattr(workspace, "providers", lambda: {})
        m.setattr(workspace, "restore",
                  lambda prov, doc, run_dir: {"restored": ["a"]})
        m.setattr(workspace, "report_text", lambda r: "1 section")
        ns = _ns()
        assert AppScreen.restore_run_workspace(
            ns, {"folder": str(tmp_path)}) == {"restored": ["a"]}
        assert "Restored the workspace of" in ns._console.stdout[-1]
        assert "1 section" in ns._console.stdout[-1]

    with subtests.test(case="say falls back to the log"):
        def refuse(_text):
            raise RuntimeError("closed")

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        aps.LOG.addHandler(handler)
        level = aps.LOG.level
        aps.LOG.setLevel(logging.INFO)
        try:
            AppScreen._say(types.SimpleNamespace(
                _console=types.SimpleNamespace(append_stdout=refuse)),
                "hello")
            AppScreen._say(types.SimpleNamespace(), "again")
        finally:
            aps.LOG.removeHandler(handler)
            aps.LOG.setLevel(level)
        assert [r.getMessage() for r in records
                if r.levelno == logging.INFO] == ["hello", "again"]


def test_the_newest_results_table_the_run_wrote_is_loaded(
        screen, monkeypatch, subtests, tmp_path):
    """After a run, the table on disk is found across every root it names.

    Invariant: ``_load_regression_results`` answers True only when a table
    loaded; the newest table across ``src`` and the count folder wins; a
    list-valued count setting names its folder; a table that vanished
    between listing and stat is skipped; a table that will not load leaves
    the panel's own reason (nothing extra said); settings naming no folder
    say where to point; no results panel loads nothing.
    """
    from spacr.qt.widgets import regression_results

    old, new = tmp_path / "old", tmp_path / "new"
    for folder in (old, new):
        folder.mkdir()
        (folder / "coefficients.csv").write_text("a\n", encoding="utf-8")
    os.utime(old / "coefficients.csv", (1_000_000, 1_000_000))
    os.utime(new / "coefficients.csv", (2_000_000, 2_000_000))
    tables = {str(old): [str(old / "coefficients.csv")],
              str(new): [str(new / "coefficients.csv")],
              str(tmp_path): [str(tmp_path / "never-written.csv")]}

    # (case, settings, panel loads, returns, first folder loaded, said)
    cases = [
        ("newest table across roots wins",
         {"src": str(old), "count_data": str(new / "counts.csv")}, True, True,
         str(new), None),
        ("list-valued count setting",
         {"src": "", "count_data": [str(old / "counts.csv")]}, True, True,
         str(old), None),
        ("table vanished before stat", {"src": str(tmp_path)}, True, False,
         "unchecked", None),
        ("table will not load", {"src": str(new)}, False, False, str(new),
         []),
        ("no folder at all", {}, True, False, None, "Load results"),
    ]
    for name, settings, loads, returns, first, said in cases:
        with subtests.test(case=name), monkeypatch.context() as m:
            screen._console.said.clear()
            m.setattr(regression_results, "find_results_tables",
                      lambda folder: tables.get(str(folder), []))
            loaded = []
            m.setattr(screen, "_results_panel", types.SimpleNamespace(
                load=lambda folder, loads=loads: loaded.append(str(folder))
                or loads,
                say=screen._console.said.append))
            m.setattr(screen, "_show_figure_grid", lambda: None)
            m.setattr(screen._settings_model, "collect",
                      lambda settings=settings: dict(settings))
            assert screen._load_regression_results() is returns
            if first != "unchecked":
                assert (loaded[0] if loaded else None) == first
            if said == []:
                assert screen._console.said == []
            elif said:
                assert any(said in line for line in screen._console.said)

    with subtests.test(case="no results panel"), monkeypatch.context() as m:
        m.setattr(screen, "_results_panel", None)
        assert screen._load_regression_results() is False


def test_a_finished_run_is_handed_over_from_memory(screen, monkeypatch,
                                                   subtests):
    """The run's own frame is shown; no path is guessed.

    Invariant: a payload with rows hands its frame, settings and model to the
    panel and marks the results as loaded in memory with the run folder
    remembered; empty or missing rows hand nothing; a panel refusing the
    settings or the model still keeps the table; a panel refusing the frame
    still remembers the folder; a payload-less result records nothing; with
    no results panel the Runs-tab row is still recorded; a finished run
    whose results will not open still re-enables Run.
    """
    def payload(frame):
        return {"results": frame, "res_folder": "/data/plate1/results/run_1",
                "settings": {"regression_type": "ols"}, "model": object(),
                "regression_type": "ols"}

    def panel(handed, **overrides):
        def set_frame(frame, source=""):
            handed["frame"] = frame
            return True

        fake = types.SimpleNamespace(
            set_frame=set_frame,
            set_run_settings=lambda s: handed.update(settings=s),
            set_diagnostics=lambda model, regression_type=None:
            handed.update(model=model),
            set_summary=lambda model, regression_type=None:
            handed.update(summary=model))
        for attr, value in overrides.items():
            setattr(fake, attr, value)
        return fake

    one_row = pd.DataFrame({"coefficient": [0.5]})
    run_1 = "/data/plate1/results/run_1"
    # (case, panel overrides, frames sent, frame handed, in memory, folder)
    cases = [
        ("shown from memory", {}, [one_row], one_row, True, run_1),
        ("no rows are not handed", {}, [pd.DataFrame(), None], None, False,
         None),
        ("settings refused, table kept", {"set_run_settings": _boom},
         [one_row], one_row, True, run_1),
        ("model refused, table kept", {"set_diagnostics": _boom}, [one_row],
         one_row, True, run_1),
        ("frame refused, folder kept", {"set_frame": _boom}, [one_row], None,
         False, run_1),
    ]
    for name, overrides, frames, handed_frame, in_memory, folder in cases:
        with subtests.test(case=name), monkeypatch.context() as m:
            screen._console.said.clear()
            handed = {"frame": None}
            m.setattr(screen, "_results_panel", panel(handed, **overrides))
            m.setattr(screen, "_update_run_in_runs_tab", lambda **f: True)
            m.setattr(screen, "_say_the_qc_verdict", lambda _p: "")
            m.setattr(screen, "_show_figure_grid", lambda: None)
            m.setattr(screen, "_results_loaded_in_memory", False,
                      raising=False)
            m.setattr(screen, "_last_run_folder", None, raising=False)
            for frame in frames:
                screen._on_pipeline_result(payload(frame))
            assert handed["frame"] is handed_frame
            assert screen._results_loaded_in_memory is in_memory
            assert screen._last_run_folder == folder
            if name == "shown from memory":
                assert handed["settings"] == {"regression_type": "ols"}
                assert any("coefficients loaded from the run itself" in line
                           for line in screen._console.said)

    with subtests.test(case="no payload changes nothing"), \
            monkeypatch.context() as m:
        updates = []
        m.setattr(screen, "_update_run_in_runs_tab",
                  lambda **fields: updates.append(fields))
        screen._on_pipeline_result(None)
        screen._on_pipeline_result("not a payload")
        assert updates == []

    with subtests.test(case="no results panel still records the row"), \
            monkeypatch.context() as m:
        recorded = {}
        m.setattr(screen, "_update_run_in_runs_tab",
                  lambda **fields: recorded.update(fields) or True)
        m.setattr(screen, "_say_the_qc_verdict", lambda _p: "")
        m.setattr(screen, "_results_panel", None)
        screen._on_pipeline_result({"res_folder": "/data/run",
                                    "results": None})
        assert recorded["folder"] == "/data/run"

    with subtests.test(case="results will not open, run still finishes"), \
            monkeypatch.context() as m:
        m.setattr(screen, "_results_panel", types.SimpleNamespace(),
                  raising=False)
        m.setattr(screen, "_results_loaded_in_memory", False, raising=False)
        m.setattr(screen, "_load_regression_results", _boom)
        screen._on_finished(True)
        assert screen._btn_run.isEnabled()
        assert not screen._progress.isVisible()
