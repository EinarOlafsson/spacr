"""Open Sans is matplotlib's default in the app and in pipeline runs -- only.

Item 291. The maintainer's decision, 2026-09-15, verbatim: "Global in the app
only (Recommended)". The option read: "The spaCR GUI and spaCR's pipeline runs
set Open Sans as matplotlib's default; plain `import spacr` in a notebook
leaves the user's matplotlib alone."

The library half -- importing spaCR changes no rcParam -- is held by
``tests/test_one_visual_system_not_twelve.py::
test_importing_spacr_leaves_every_rcparam_alone``. This file holds the other
half, per entry point:

  (a) the GUI startup path, for every GUI console script in setup.py;
  (b) the pipeline entry points: ``spacr-run``, a batch-queue job, a replay;
  (c) in each, a PLAIN ``Figure()`` -- no house style, the way
      ``illumination.py`` builds its QC figure -- resolves to the Open Sans
      file spaCR ships.

EVERY CHECK RUNS IN A CHILD INTERPRETER, for two reasons. The entry points
hold the default for the life of the process, which is the process under
test and must not be this one. And the answer has to come from a font manager
with every Open Sans on the machine taken out of it: the item is about "a
figure drawn on a machine without Open Sans installed", and a machine that
happens to have it installed would pass a broken fix. So the assertion is on
the font FILE ``findfont`` resolved for the title's own FontProperties, never
on the rcParam string -- a family that is named but cannot be found falls back
to DejaVu Sans without an error.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.child_env import child_env

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The faces the package ships, worked out from the checkout rather than from
#: the code under test.
BUNDLED = os.path.realpath(
    REPO_ROOT / "spacr" / "resources" / "font" / "open_sans" / "static")

#: The environment variable a run sets so its worker processes can follow it.
MARKER = "SPACR_FIGURES_IN_OPEN_SANS"

_NEEDS_QT = pytest.mark.skipif(find_spec("PySide6") is None,
                               reason="the Qt extra is not installed")

#: Runs first in every child: a bare machine, and a stand-in that draws.
PREAMBLE = r'''
import json, os, sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager

import spacr
from spacr import figure_font

# The machine the item is about: no Open Sans installed anywhere. Only the
# bundled files, registered by the code under test, can make the name resolve.
_manager = font_manager.fontManager
_manager.ttflist = [entry for entry in _manager.ttflist
                    if "open sans" not in str(entry.name).lower()]
_manager._findfont_cached.cache_clear()
figure_font._registered = False
figure_font._resolved = False

SEEN = []


def draw_a_plain_figure(*_args, **_kwargs):
    """What illumination.py does for its QC figure: a bare Figure()."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure()
    FigureCanvasAgg(figure)
    title = figure.suptitle("a QC figure")
    figure.canvas.draw()
    SEEN.append({
        "family": list(matplotlib.rcParams["font.family"]),
        "file": os.path.realpath(
            font_manager.findfont(title.get_fontproperties())),
        "marker": os.environ.get("SPACR_FIGURES_IN_OPEN_SANS"),
    })
    return 0


def report(**values):
    print("PROBE " + json.dumps(dict(
        values,
        spacr=spacr.__file__,
        seen=SEEN,
        family_after=list(matplotlib.rcParams["font.family"]),
        marker_after=os.environ.get("SPACR_FIGURES_IN_OPEN_SANS"),
        registered=bool(figure_font._registered),
    )))
'''


def _probe(tmp_path: Path, body: str, *, qt: bool = False,
           marker: str | None = None) -> dict:
    """Run ``body`` after :data:`PREAMBLE` in a fresh interpreter.

    :param body: code that calls an entry point and binds its exit to
        ``code``.
    :param marker: :data:`MARKER` for the child, or ``None`` for unset.
    :returns: what the child reported.
    """
    env = child_env(home=str(tmp_path), pythonpath=str(REPO_ROOT), qt=qt)
    env.pop(MARKER, None)
    if marker is not None:
        env[MARKER] = marker
    source = PREAMBLE + textwrap.dedent(body) + "\nreport(code=code)\n"
    finished = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
        env=env)
    lines = [line for line in finished.stdout.splitlines()
             if line.startswith("PROBE ")]
    assert finished.returncode == 0 and lines, (
        f"the child did not finish (exit {finished.returncode}):\n"
        f"{finished.stdout[-3000:]}\n{finished.stderr[-5000:]}")
    probe = json.loads(lines[-1][len("PROBE "):])
    # The child imported THIS checkout, not an installed spaCR.
    assert probe["spacr"].startswith(str(REPO_ROOT / "spacr")), probe["spacr"]
    return probe


def _is_bundled(path: str) -> bool:
    return os.path.commonpath([BUNDLED, path]) == BUNDLED


def _drew_in_open_sans(probe: dict) -> None:
    """(c): the plain figure's title resolved to a face spaCR ships."""
    assert probe["seen"], "the entry point never reached the code that draws"
    drawn = probe["seen"][0]
    assert _is_bundled(drawn["file"]), (
        f"a plain Figure() drawn by this entry point resolved its title to "
        f"{drawn['file']}, not to one of the faces spaCR ships in {BUNDLED}")
    assert os.path.basename(drawn["file"]).startswith("OpenSans-"), drawn
    assert drawn["family"][0] == "Open Sans", drawn
    assert drawn["marker"] == "1", (
        "the run did not tell its worker processes to follow it", drawn)


def _drew_in_the_stock_default(probe: dict) -> None:
    assert probe["seen"], "the entry point never reached the code that draws"
    drawn = probe["seen"][0]
    assert not _is_bundled(drawn["file"]), drawn
    assert os.path.basename(drawn["file"]).startswith("DejaVuSans"), drawn
    assert drawn["family"] == ["sans-serif"], drawn


def _gave_matplotlib_back(probe: dict) -> None:
    """The default is the run's: an in-process caller gets its own back."""
    assert probe["family_after"] == ["sans-serif"], probe["family_after"]
    assert probe["marker_after"] is None, probe["marker_after"]


# ---------------------------------------------------------------------------
# (a) the app
# ---------------------------------------------------------------------------

#: `launch` builds a QApplication and runs its event loop; everything before
#: it is the startup path under test, so only `launch` is stood in for.
_STUB_LAUNCH = r'''
import spacr.qt.app as _app
_app.launch = draw_a_plain_figure
'''

#: Every GUI console script in setup.py, as its target, and how to call it.
GUI_CASES = {
    "spacr.qt:run": _STUB_LAUNCH + r'''
from spacr.qt import run
code = run([])
''',
    "spacr.qt:run_without_setup": _STUB_LAUNCH + r'''
from spacr.qt import run_without_setup
code = run_without_setup([])
''',
    "spacr.qt.spaceout:main": _STUB_LAUNCH + r'''
from spacr.qt.spaceout import main
code = main([])
''',
    "spacr.qt.safespacr:main": _STUB_LAUNCH + r'''
from spacr.qt.safespacr import main
code = main([])
''',
    "spacr.qt.tutorial.__main__:main": r'''
import types
import spacr.qt.tutorial.__main__ as tutorial


def _render(name, **_kwargs):
    draw_a_plain_figure()
    return types.SimpleNamespace(mp4="t.mp4", srt="t.srt", duration_s=0.0,
                                 frames=0)


tutorial.render_tutorial = _render
code = tutorial.main([list(tutorial.AVAILABLE_TUTORIALS)[0],
                      "--out", sys.argv[1]])
''',
    # Not a console script, but the documented `python -m spacr`.
    "spacr.__main__:main": _STUB_LAUNCH + r'''
from spacr.__main__ import main
code = main(["gui"])
''',
}

#: Safe mode is the least spaCR that can still change a setting (item 296):
#: it does not register fonts with matplotlib.
SAFE_MODE = "spacr.qt.safespacr:main"


def test_every_gui_console_script_is_asked():
    """A new GUI script added to setup.py is not silently outside the rule."""
    text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    targets = set(re.findall(r"'[\w-]+=(spacr\.qt[\w.]*:\w+)'", text))
    assert "spacr.qt:run" in targets, "the setup.py scan found no GUI script"
    missing = sorted(targets - set(GUI_CASES))
    assert not missing, (
        f"GUI console scripts with no Open Sans case here: {missing}. Add "
        "each to GUI_CASES, expecting Open Sans unless it is safe mode")


@_NEEDS_QT
@pytest.mark.qt
@pytest.mark.parametrize("target", sorted(set(GUI_CASES) - {SAFE_MODE}))
def test_the_gui_startup_path_makes_open_sans_the_default(tmp_path, target):
    """(a) and (c): offscreen, up to the window, for each way in."""
    probe = _probe(tmp_path, GUI_CASES[target], qt=True)
    assert probe["code"] == 0
    _drew_in_open_sans(probe)
    _gave_matplotlib_back(probe)


@_NEEDS_QT
@pytest.mark.qt
def test_safe_mode_starts_without_registering_a_font(tmp_path):
    probe = _probe(tmp_path, GUI_CASES[SAFE_MODE], qt=True)
    assert probe["code"] == 0
    _drew_in_the_stock_default(probe)
    assert probe["registered"] is False, (
        "safespacr registered the bundled faces with matplotlib; item 296 "
        "wants safe mode to start with the bare minimum")


# ---------------------------------------------------------------------------
# (b) pipeline runs
# ---------------------------------------------------------------------------

#: A module `spacr-run` can really run, whose work is drawing a plain figure.
_FAKE_PIPELINE = r'''
import pathlib, types
from spacr import cli

_fake = types.ModuleType("spacr_open_sans_fake_pipeline")
_fake.run = draw_a_plain_figure
sys.modules[_fake.__name__] = _fake
cli.MODULES["_fake"] = cli.Module(
    key="_fake", summary="test-only module", entry=_fake.__name__ + ":run",
    defaults=None, validate_key="", requires=("src",), writes=("nothing",))
_tmp = pathlib.Path(sys.argv[1])
(_tmp / "plate").mkdir(exist_ok=True)
_settings = _tmp / "settings.csv"
_settings.write_text("Key,Value\nsrc,%s\n" % (_tmp / "plate"))
'''

PIPELINE_CASES = {
    "spacr-run": _FAKE_PIPELINE + r'''
code = cli.main(["_fake", "--settings", str(_settings), "--no-preflight"])
''',
    "a batch-queue job": _FAKE_PIPELINE + r'''
from spacr.batch import Job, inprocess_runner
code = inprocess_runner(Job(module="_fake", settings=str(_settings)),
                        str(_settings), str(_tmp / "job.log"))
''',
    "spacr-repro": r'''
import json, pathlib
from spacr import cli_repro

_run = pathlib.Path(sys.argv[1]) / "recorded_run"
_run.mkdir()
(_run / "manifest.json").write_text(json.dumps({"app_key": "mask"}))
(_run / "settings.json").write_text(json.dumps({"src": sys.argv[1]}))
cli_repro._resolve_pipeline = lambda key: draw_a_plain_figure
code = cli_repro.main([str(_run)])
''',
}


@pytest.mark.parametrize("entry", sorted(PIPELINE_CASES))
def test_a_pipeline_run_makes_open_sans_the_default(tmp_path, entry):
    """(b) and (c)."""
    probe = _probe(tmp_path, PIPELINE_CASES[entry])
    assert probe["code"] == 0
    _drew_in_open_sans(probe)
    _gave_matplotlib_back(probe)


def test_a_queued_job_in_its_own_process_is_a_spacr_run():
    """The default queue runner starts `python -m spacr.cli`, which is the
    `spacr-run` case above -- so that case covers it."""
    from spacr.batch import Job, job_command

    command = job_command(Job(module="mask", settings="s.csv"), "s.csv")
    assert command[1:3] == ["-m", "spacr.cli"], command
    source = (REPO_ROOT / "spacr" / "cli.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":\n    raise SystemExit(main())' in source


# ---------------------------------------------------------------------------
# worker processes a run starts
# ---------------------------------------------------------------------------

#: `perform_regression` is what a sweep trial draws through; stood in for.
_STUB_REGRESSION = r'''
import json, pathlib, types
_ml = types.ModuleType("spacr.ml")
_ml.perform_regression = draw_a_plain_figure
sys.modules["spacr.ml"] = _ml
_tmp = pathlib.Path(sys.argv[1])
'''

_SWEEP_CHILD = _STUB_REGRESSION + r'''
from spacr import sweep_child
(_tmp / "trial.json").write_text(
    json.dumps({"settings": {"src": str(_tmp)}, "trial_id": 1}))
code = sweep_child.main([str(_tmp / "trial.json"), str(_tmp / "out.json")])
'''

_SWEEP_POOL_WORKER = _STUB_REGRESSION + r'''
from spacr import parameter_sweep
parameter_sweep._execute_trial(({}, {"trial_id": 1}, str(_tmp), {}, False))
code = 0
'''


@pytest.mark.parametrize("worker", ["sweep_child", "sweep pool worker"])
def test_a_worker_a_run_started_follows_it(tmp_path, worker):
    body = _SWEEP_CHILD if worker == "sweep_child" else _SWEEP_POOL_WORKER
    probe = _probe(tmp_path, body, marker="1")
    _drew_in_open_sans(probe)


def test_a_worker_a_notebook_started_leaves_matplotlib_alone(tmp_path):
    """The same child, with no run above it: the notebook's own default."""
    probe = _probe(tmp_path, _SWEEP_CHILD)
    _drew_in_the_stock_default(probe)
    assert probe["registered"] is False
