"""The macro recorder, driven through the real Qt GUI run path.

``tests/test_macro.py`` proves the recorder by executing what it emits;
it does so through the two statements ``PipelineWorker.run`` wraps every
pipeline in. This file closes the last gap: the same round trip, but
started the way a user starts one — ``resolve_pipeline_entry`` for the
module the Run button is on, ``make_thread`` for the worker, a real
``QThread``, and the emitted script executed in a fresh interpreter
afterwards.

It also pins the coupling that decides whether a script exists at all:
the recorder hangs off the run journal, so a job that opts out of
journalling (``journal=False``, which is what read-only UI housekeeping
uses) records no macro — by design, and asserted here so the decision is
visible rather than discovered.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import macro

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

DEMO_MODULE = '''\
"""A miniature spaCR module, for the macro recorder's GUI round trip."""
import json
import os


def run_demo(settings):
    """Sum the numbers in every input file and write the totals."""
    src = settings["src"]
    if isinstance(src, (list, tuple)):
        src = src[0]
    scale = settings.get("scale", 1)
    label = settings.get("label", "demo")
    totals = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(src, name)) as handle:
            totals[name] = sum(
                int(line) for line in handle.read().split() if line) * scale
    out = os.path.join(src, f"{label}.json")
    with open(out, "w") as handle:
        json.dump({"label": label, "scale": scale, "totals": totals},
                  handle, indent=2, sort_keys=True)
    print(f"demo wrote {out}")
    return out


def demo_defaults(settings=None):
    """The module's defaults, through the `register_defaults` seam."""
    values = dict(settings or {})
    values.setdefault("src", "")
    values.setdefault("scale", 1)
    values.setdefault("label", "demo")
    return values
'''

DEMO_KEY = "macro_gui_demo"


@pytest.fixture
def gui_demo(tmp_path, monkeypatch):
    """A registered pipeline app whose entry point is a real, cheap function.

    Registered through ``register_defaults`` + ``register_app(entry=...)``,
    the seams a module is meant to join through, and unregistered again:
    the app registry is process-global and a leaked row becomes another
    file's mysterious extra tile.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "spacr_macro_gui_demo.py").write_text(DEMO_MODULE)
    monkeypatch.syspath_prepend(str(package))

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))

    from spacr.settings import register_defaults, unregister_defaults
    from spacr.qt.app import register_app, unregister_app, SECTION_ORDER
    import spacr_macro_gui_demo

    register_defaults(DEMO_KEY, spacr_macro_gui_demo.demo_defaults,
                      replace=True)
    register_app(DEMO_KEY, "Macro GUI Demo", "round-trip fixture",
                 SECTION_ORDER[0], entry="spacr_macro_gui_demo:run_demo")
    macro.reset()
    try:
        yield {"key": DEMO_KEY, "package": str(package), "home": str(home)}
    finally:
        macro.reset()
        unregister_defaults(DEMO_KEY)
        unregister_app(DEMO_KEY)


def make_inputs(root: Path) -> Path:
    """Create the demo module's inputs."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("1\n2\n3\n")
    (root / "b.txt").write_text("10\n20\n")
    return root


def _retired(thread) -> bool:
    """Has ``thread`` stopped — including "its C++ half is already gone"?

    ``make_thread`` connects ``thread.finished`` to ``deleteLater``, so the
    same event pump that lets the job finish also reaps the wrapper.
    Polling it afterwards raises rather than returning False, and a reaped
    wrapper is the strongest evidence of retirement there is. The same
    helper ``tests/qt/test_qt_worker_teardown.py`` uses, for the same
    reason.
    """
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


def run_through_the_button(qtbot, key, settings, *, journal=True):
    """Start a run exactly as an AppScreen's Run button does, and join it."""
    from spacr.qt.bridge import make_thread, resolve_pipeline_entry
    entry = resolve_pipeline_entry(key)
    assert entry is not None, f"no pipeline entry resolved for {key!r}"
    thread, worker = make_thread(entry, settings, app_key=key, journal=journal)
    outcome = {}
    worker.error.connect(lambda tb: outcome.setdefault("error", tb))
    thread.start()
    qtbot.waitUntil(lambda: _retired(thread), timeout=30000)
    assert "error" not in outcome, outcome["error"]
    return worker


def newest_run_dir(home):
    """Return the most recent run journal folder under a temporary home."""
    runs = Path(home, ".spacr", "runs")
    folders = sorted((path for path in runs.iterdir() if path.is_dir()),
                     key=lambda path: path.stat().st_mtime)
    assert folders, f"the run wrote no journal under {runs}"
    return folders[-1]


class TestTheGuiPathEmitsARunnableScript:
    """Press Run, get a script, run the script, compare the outputs."""

    def test_round_trip_through_make_thread(self, qtbot, tmp_path, gui_demo):
        src = make_inputs(tmp_path / "plate")
        settings = {"src": str(src), "scale": 5, "label": "gui"}

        run_through_the_button(qtbot, gui_demo["key"], settings)

        produced = src / "gui.json"
        assert produced.is_file(), "the GUI run produced nothing"
        expected = produced.read_bytes()

        script = Path(macro.macro_path(newest_run_dir(gui_demo["home"])))
        assert script.is_file(), "no macro.py in the run journal folder"

        produced.unlink()
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                [gui_demo["package"], REPO_ROOT])})
        assert result.returncode == 0, (
            f"emitted script failed:\n{result.stdout}\n{result.stderr}")
        assert produced.read_bytes() == expected, (
            "the emitted script produced a different result than the GUI run")

    def test_the_setting_the_user_changed_is_in_the_script(
            self, qtbot, tmp_path, gui_demo):
        """A knob moved on the settings panel reaches the emitted code."""
        src = make_inputs(tmp_path / "plate")
        run_through_the_button(
            qtbot, gui_demo["key"],
            {"src": str(src), "scale": 11, "label": "changed"})

        record = macro.read_macro(
            macro.macro_path(newest_run_dir(gui_demo["home"])))
        step = record["steps"][0]
        assert step["settings"]["scale"] == 11
        assert "scale" in step["user_set"]
        assert "scale" not in step["defaulted"]
        assert step["status"] == "success"

    def test_two_gui_runs_on_one_project_are_one_script(
            self, qtbot, tmp_path, gui_demo):
        src = make_inputs(tmp_path / "plate")
        for label in ("first", "second"):
            run_through_the_button(
                qtbot, gui_demo["key"], {"src": str(src), "label": label})

        record = macro.read_macro(
            macro.macro_path(newest_run_dir(gui_demo["home"])))
        assert [step["settings"]["label"] for step in record["steps"]] == [
            "first", "second"]

    def test_a_job_that_opts_out_of_journalling_records_nothing(
            self, qtbot, tmp_path, gui_demo):
        """The coupling, stated out loud: no journal, no macro.

        ``make_thread(..., journal=False)`` is for read-only UI
        housekeeping, which is not an analysis run and has no method to
        record. If that ever becomes the default for real runs, this test
        is where the loss of the macro shows up.
        """
        src = make_inputs(tmp_path / "plate")
        run_through_the_button(
            qtbot, gui_demo["key"], {"src": str(src), "label": "quiet"},
            journal=False)
        assert (src / "quiet.json").is_file(), "the job did not run"
        assert not Path(gui_demo["home"], ".spacr", "runs").exists() or not any(
            Path(gui_demo["home"], ".spacr", "runs").iterdir())
        assert macro.current_macro() is None
