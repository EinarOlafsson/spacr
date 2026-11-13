"""What a (regression) module screen answers when a part of itself is missing.

The screen holds a settings model, a console, fold switches, a runs tab and a
results panel, and each can be absent, half-built or already destroyed when
it is asked something: replaying a workspace, a bulk apply whose dependency
pass throws, a restart whose settings cannot be collected, "which folder is
this run in?". The rule in every case: do the part it can, say what it could
not, answer "I do not know" rather than guess, and never raise on the GUI
thread. Cases share one screen per test and patch it through a per-case
``MonkeyPatch.context()``, so each case's changes are undone before the next.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QWidget                              # noqa: E402

from tests.qt.test_app_screen_a_construction import (               # noqa: E402,F401
    _reset_hover_tooltip_around_each_test,
    built,
)

pytestmark = pytest.mark.qt


def _raises(message, exc=RuntimeError):
    def _fn(*a, **k):
        raise exc(message)
    return _fn


def _recording_console(mp, scr, attr="append_stdout"):
    said = []
    mp.setattr(scr, "_console", types.SimpleNamespace(**{attr: said.append}))
    return said


def test_sentences_and_qc_verdicts_reach_the_console_or_the_log(
        qtbot, subtests, caplog):
    """``_say`` writes to the console, and to the log when the console is
    missing or refuses. ``_say_the_qc_verdict`` says nothing without a
    verdict, one ok line for a pass, and names the failing check and the
    report file for a failure.
    """
    with built("regression") as scr:
        with subtests.test(case="say: console"), \
                pytest.MonkeyPatch.context() as mp:
            said = _recording_console(mp, scr)
            scr._say("the run finished")
            assert said == ["the run finished"]
        for name, console in (
                ("say: console refuses", types.SimpleNamespace(
                    append_stdout=_raises("Internal C++ object already "
                                          "deleted."))),
                ("say: no console", None)):
            with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
                mp.setattr(scr, "_console", console)
                caplog.clear()
                with caplog.at_level("INFO"):
                    scr._say("the run finished")
                assert any("the run finished" in r.message
                           for r in caplog.records)

        for payload in (None, {}, {"qc_verdict": None}):
            with subtests.test(case=f"qc: {payload!r}"):
                assert scr._say_the_qc_verdict(payload) == ""

        verdict = types.SimpleNamespace(detail="the design is rank deficient",
                                        name="collinearity")
        cases = [
            ("qc: ok", {"qc_verdict": "every panel passed",
                        "qc_verdict_level": "ok"},
             ["Regression QC: ok", "every panel passed"]),
            ("qc: fail", {"qc_verdict": verdict, "qc_verdict_level": "fail"},
             ["REGRESSION QC: FAIL", "(collinearity)",
              "regression_qc_report.txt"]),
        ]
        for name, payload, parts in cases:
            with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
                said = _recording_console(mp, scr)
                line = scr._say_the_qc_verdict(payload)
                assert all(p in line for p in parts)
                assert said == [line]


def test_workspace_state_is_saved_and_replayed_only_into_its_own_module(
        qtbot, subtests):
    """``workspace_state`` always reports the app key, with ``{}`` settings
    when collect raises or answers None; ``apply_workspace_state`` refuses
    non-records, another module's record and records without a settings
    dict, and applies its own (an ``app_key``-less record counts as own);
    ``restore_run_workspace`` without a folder restores nothing.
    """
    empty = {"restored": [], "skipped": [], "files": []}
    with built("mask") as scr:
        with subtests.test(case="built screen"):
            state = scr.workspace_state()
            assert state["app_key"] == "mask"
            assert isinstance(state["settings"], dict) and state["settings"]
        for name, collect in (("collect raises",
                               _raises("a widget was deleted mid-collect")),
                              ("collect answers None", lambda: None)):
            with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
                mp.setattr(scr._settings_model, "collect", collect)
                assert scr.workspace_state() == {"app_key": "mask",
                                                 "settings": {}}

    with built("regression") as scr:
        replays = [
            ("None", None, False),
            ("a string", "regression", False),
            ("another module", {"app_key": "measure",
                                "settings": {"verbose": True}}, False),
            ("empty settings", {"app_key": "regression", "settings": {}},
             False),
            ("settings not a dict", {"app_key": "regression",
                                     "settings": "verbose=True"}, False),
            ("own record", {"settings": {"verbose": True}}, True),
        ]
        for name, state, applied in replays:
            with subtests.test(case=f"replay {name}"):
                assert bool(scr.apply_workspace_state(state)) is applied
        for record in ({}, ""):
            with subtests.test(case=f"restore {record!r}"):
                assert scr.restore_run_workspace(record) == empty


def test_a_bulk_apply_keeps_its_values_when_a_follow_up_pass_fails(
        qtbot, subtests, caplog):
    """The values are the point: a dependency pass that throws after the
    apply, or dimension switches that throw, still leave the values in and
    ``_applying_settings`` cleared; fold switches that cannot be read report
    nothing switched on (and remember that).
    """
    from spacr.qt.screens import mask

    with built("regression") as scr:
        model = scr._settings_model
        with subtests.test(case="dependency pass throws"), \
                pytest.MonkeyPatch.context() as mp:
            real = model._refresh_setting_dependencies

            def broken_on_the_final_pass():
                if model._applying_settings:
                    return real()
                raise RuntimeError("a rule read a widget that had gone")

            mp.setattr(model, "_refresh_setting_dependencies",
                       broken_on_the_final_pass)
            with caplog.at_level("DEBUG"):
                assert scr.apply_settings_dict({"verbose": True}) >= 1
            assert model._applying_settings is False

        with subtests.test(case="dimension switches throw"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(scr, "_sync_dimension_switches",
                       _raises("the switches are not built yet"))
            assert scr.apply_settings_dict({"verbose": True}) >= 1
            assert model._applying_settings is False

        with subtests.test(case="fold switches unreadable"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(mask, "sync_folds", _raises("the fold strip is half "
                                                   "built"))
            assert scr._sync_folded_switches({"timelapse": True}) == ()
            assert scr._folds_last_switched_on == ()


def test_force_restart_goes_ahead_and_warns_honestly(
        qtbot, subtests, tmp_path, monkeypatch):
    """A restart launches even when settings cannot be collected; one that
    could not save says nothing was restarted; the restart warning names the
    open run folder and is empty (not wrong) when it cannot be composed.
    """
    import spacr.restart_state as restart_state

    # Keep the restart record out of the shared HOME: left in ~/.spacr it
    # would open the next MainWindow on Regression instead of Home.
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "spacr_home"))
    with built("regression") as scr:
        with subtests.test(case="collect raises"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(scr._settings_model, "collect",
                       _raises("the panel is wedged"))
            seen = {}

            def launcher(*args, **kwargs):
                seen["launched"] = True
                return True

            assert scr.force_restart(launcher=launcher,
                                     exiter=lambda *a, **k: None) is True
            assert seen.get("launched") is True

        with subtests.test(case="restart could not save"), \
                pytest.MonkeyPatch.context() as mp:
            said = _recording_console(mp, scr)
            mp.setattr("spacr.qt.shutdown.restart_spacr",
                       lambda *a, **k: False)
            assert scr.force_restart() is False
            assert any("did NOT restart" in line for line in said)

        with subtests.test(case="warning names the open run"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(scr, "_last_run_folder", "/data/plate1/results",
                       raising=False)
            assert "/data/plate1/results" in scr._restart_warning()

        with subtests.test(case="warning cannot be composed"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(restart_state, "warning_text",
                       _raises("the running list is malformed"))
            assert scr._restart_warning() == ""


def _point_at(mp, scr, *, runs=None, panel=None):
    mp.setattr(scr, "_sweep_runs", runs)
    mp.setattr(scr, "_results_panel", panel)


def test_which_folder_and_frame_a_run_is_in_is_never_guessed(
        qtbot, subtests, tmp_path):
    """``_results_source_path`` prefers the selected run, falls back to the
    panel's run folder (also when the runs tab raises) or its kept ``_path``,
    else "". ``_scan_source_frame`` reads the run's ``regression_data.csv``
    or older ``merged_data.csv`` and is None for an unreadable file, neither
    file, or no folder. The effects grid is written beside the run (next to
    a CSV when the path is a file), not at all without effects, and a
    montage writer that raises does not fail the sweep.
    """
    from spacr import cell_montage

    runs = types.SimpleNamespace(
        loaded_run_folder=lambda: "/data/plate1/results/run_3")
    refusing_runs = types.SimpleNamespace(
        loaded_run_folder=_raises("the runs model is not loaded"))
    sources = [
        ("selected run wins", runs,
         types.SimpleNamespace(run_folder=lambda: "/somewhere/else"),
         "/data/plate1/results/run_3"),
        ("runs tab refuses", refusing_runs,
         types.SimpleNamespace(run_folder=lambda: "/data/plate1/results"),
         "/data/plate1/results"),
        ("neither", None, None, ""),
        ("older panel's kept path", None,
         types.SimpleNamespace(_path="/data/plate1/results/coefficients.csv"),
         "/data/plate1/results/coefficients.csv"),
    ]
    frames = [
        ("regression_data.csv", "gene,plateID,response\nA,p1,0.5\n",
         ["gene", "plateID", "response"]),
        ("merged_data.csv", "gene,response\nA,0.5\n", ["gene", "response"]),
        ("regression_data.csv", "", None),
        (None, None, None),
    ]
    with built("regression") as scr:
        for name, sweep_runs, panel, expected in sources:
            with subtests.test(case=f"source: {name}"), \
                    pytest.MonkeyPatch.context() as mp:
                _point_at(mp, scr, runs=sweep_runs, panel=panel)
                assert scr._results_source_path() == expected

        for i, (filename, text, columns) in enumerate(frames):
            folder = tmp_path / f"frame_{i}"
            folder.mkdir()
            if filename:
                (folder / filename).write_text(text, encoding="utf-8")
            with subtests.test(case=f"frame: {filename!r} {columns}"), \
                    pytest.MonkeyPatch.context() as mp:
                _point_at(mp, scr, panel=types.SimpleNamespace(
                    run_folder=lambda f=str(folder): f))
                frame = scr._scan_source_frame()
                if columns is None:
                    assert frame is None
                else:
                    assert list(frame.columns) == columns and len(frame) == 1
        with subtests.test(case="frame: no folder"), \
                pytest.MonkeyPatch.context() as mp:
            _point_at(mp, scr)
            assert scr._scan_source_frame() is None

        table = tmp_path / "coefficients.csv"
        table.write_text("a\n", encoding="utf-8")
        effects = types.SimpleNamespace(effects=[{"gene": "A"}])
        grids = [
            ("beside the run", types.SimpleNamespace(
                run_folder=lambda: str(tmp_path)), None),
            ("next to a CSV", types.SimpleNamespace(_path=str(table)), None),
            ("writer raises", types.SimpleNamespace(
                run_folder=lambda: str(tmp_path)),
             OSError("the run folder is read-only")),
        ]
        for name, panel, error in grids:
            with subtests.test(case=f"grid: {name}"), \
                    pytest.MonkeyPatch.context() as mp:
                attempts = []

                def writer(eff, folder, _e=error):
                    attempts.append((eff, folder))
                    if _e is not None:
                        raise _e

                mp.setattr(cell_montage, "write_effects_grid", writer)
                _point_at(mp, scr, panel=panel)
                scr._keep_the_effects_grid(effects)
                assert attempts == [([{"gene": "A"}], str(tmp_path))]

        with subtests.test(case="grid: no effects"), \
                pytest.MonkeyPatch.context() as mp:
            calls = []
            mp.setattr(cell_montage, "write_effects_grid",
                       lambda *a, **k: calls.append(a))
            for result in (types.SimpleNamespace(effects=[]),
                           types.SimpleNamespace(effects=None), object()):
                scr._keep_the_effects_grid(result)
            assert calls == []


def test_sweep_inputs_and_panel_settings_say_none_rather_than_guess(
        qtbot, subtests, tmp_path):
    """Scores are concatenated from every attached database's score file and
    are None (absent, not zero) with no file or an unparsable one; counts are
    None when missing or unconvertible; a sweep card without a destination
    names no folder; unreadable attached databases are []; column-fit and
    UMAP-display settings read the live panel ({} with no panel or when
    collect raises; UMAP keeps only set display keys); merged measurements
    land beside the attached database even when the panel is mute.
    """
    from spacr import cell_montage

    first, second, broken = (tmp_path / "a.csv", tmp_path / "b.csv",
                             tmp_path / "broken.csv")
    first.write_text("prcfo,plate,row,col,score\nx,p1,r1,c1,0.9\n",
                     encoding="utf-8")
    second.write_text("prcfo,plate,row,col,score\ny,p1,r1,c2,0.1\n",
                      encoding="utf-8")
    broken.write_text("", encoding="utf-8")
    plate = tmp_path / "plate1"
    database = plate / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    database.write_text("", encoding="utf-8")
    mute = _raises("a widget was destroyed mid-collect")

    with built("regression") as scr:
        scores = [
            ("two score files", [{"score": str(first)},
                                 {"score": str(second)}], 2),
            ("no score file", [{"score": ""}, "not a row"], None),
            ("unparsable score file", [{"score": str(broken)}], None),
        ]
        for name, rows, length in scores:
            with subtests.test(case=f"scores: {name}"), \
                    pytest.MonkeyPatch.context() as mp:
                mp.setattr(scr, "_attached_database_rows", lambda r=rows: r)
                frame = scr._sweep_scores()
                if length is None:
                    assert frame is None
                else:
                    assert len(frame) == length and "score" in frame.columns

        for name, rows in (("counts unconvertible",
                            [{"count": "/data/counts.csv"}]),
                           ("no count file", [{"count": ""}])):
            with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
                mp.setattr(cell_montage, "fractions_from_counts",
                           _raises("the count file has no grna column",
                                   ValueError))
                mp.setattr(scr, "_attached_database_rows", lambda r=rows: r)
                assert scr._sweep_counts() is None

        with subtests.test(case="sweep card without destination"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(scr, "_sweep", types.SimpleNamespace(), raising=False)
            assert scr._sweep_destination() == ""

        with subtests.test(case="attached databases unreadable"), \
                pytest.MonkeyPatch.context() as mp:
            class Refusing(QWidget):
                def get_value(self):
                    raise RuntimeError("the table model has gone")

            mp.setitem(scr._settings_model._widgets, "paired_data", Refusing())
            assert scr._attached_database_rows() == []

        with subtests.test(case="column fit reads the live panel"):
            settings = scr._column_fit_settings()
            assert isinstance(settings, dict) and settings

        with subtests.test(case="umap display keeps only set keys"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(scr._settings_model, "collect", lambda: {
                "figuresize": 10, "image_nr": 16, "img_zoom": None,
                "verbose": True})
            assert scr._umap_display_defaults() == {"figuresize": 10,
                                                    "image_nr": 16}

        for name, patch in (("no panel", ("_settings_model", None)),
                            ("panel mute", None)):
            with subtests.test(case=f"panel settings: {name}"), \
                    pytest.MonkeyPatch.context() as mp:
                if patch:
                    mp.setattr(scr, *patch)
                else:
                    mp.setattr(scr._settings_model, "collect", mute)
                assert scr._column_fit_settings() == {}
                assert scr._umap_display_defaults() == {}

        with subtests.test(case="measurements destination, mute panel"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(scr._settings_model, "collect", mute)
            mp.setattr(scr, "_attached_database_rows",
                       lambda: [{"database": str(database)}])
            destination = scr._measurements_destination()
            assert destination.endswith("measurements")
            assert str(tmp_path) in destination
