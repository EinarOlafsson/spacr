"""Import-cost guards, and the architectural invariant behind them.

``spacr.utils`` is the scientific stack's front door: importing it pulls
torch, cv2, pandas, matplotlib, sklearn and skimage, and costs a measured
**3.2 s and ~900 MB of resident memory**. The Qt interface is deliberately
built never to touch it — a pipeline run imports it on a worker, a GUI that
is only drawing a window must not — and half a dozen modules in ``spacr/qt``
carry comments saying so (``preview_controls._get_regex_callable`` goes as far
as compiling one function out of the source file rather than pay it).

A comment is not a guarantee. This file asserts it:

=========================================  =========  ========  ==========
what is guarded                            documented  measured  ceiling
=========================================  =========  ========  ==========
``import spacr.utils``, wall clock            3.2 s     3.3 s      15 s
``import spacr.utils``, peak resident         900 MB    833 MB     1600 MB
the ``spacr-qt`` launch path, wall clock      --        0.57 s     4 s
the ``spacr-qt`` launch path, peak resident   --        172 MB     600 MB
``spacr.utils`` after the Qt launch path      absent    absent     absent
``spacr.utils`` after importing every one
of the 127 ``spacr.qt`` modules               absent    absent     absent
=========================================  =========  ========  ==========

Every measurement is taken in a **fresh subprocess** running the same
interpreter as this test. By the time this file runs, pytest has imported most
of spaCR, so ``sys.modules`` in this process says nothing at all about what a
launch costs — and a timing taken here would be timing an import that already
happened.

The two absence assertions are worth more than the four numbers. A number
tells you the tree got slower; ``'spacr.utils' in sys.modules`` tells you
*which* line did it, and it cannot be blamed on a busy machine.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

import spacr

#: Repo root: the tree this test process itself imported, so the subprocesses
#: measure the same checkout rather than whatever is installed.
REPO_ROOT = Path(spacr.__file__).resolve().parents[1]

#: Line prefix the payloads print their JSON under, so ordinary import chatter
#: (spaCR modules do print on import) cannot be mistaken for the result.
SENTINEL = "PERF-GUARD-JSON "

_HAS_PYSIDE6 = find_spec("PySide6") is not None
_NEEDS_QT = pytest.mark.skipif(not _HAS_PYSIDE6,
                               reason="the Qt extra is not installed")

#: Imports that make ``spacr.utils`` cost what it costs. Named individually so
#: a failure says which one arrived rather than "something got heavier".
HEAVY = ("spacr.utils", "torch", "cellpose", "tensorflow", "cv2", "tkinter",
         "IPython", "matplotlib.pyplot")

_PREAMBLE = f"""
import json, resource, sys, time
HEAVY = {HEAVY!r}


def report(**values):
    values["rss_mb"] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / (1024.0 * 1024.0 if sys.platform == "darwin"
                           else 1024.0))
    values["heavy"] = [m for m in HEAVY if m in sys.modules]
    values["modules"] = len(sys.modules)
    print({SENTINEL!r} + json.dumps(values))
"""

#: What ``spacr.qt.run`` does, short of the call to ``launch`` that would open
#: a window: the quiet-logging setup, the import of the main window module and
#: the registration pass over the self-registering apps. Anything that a real
#: ``spacr-qt`` start imports before the first frame is imported here.
QT_LAUNCH_PATH = _PREAMBLE + """
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
start = time.perf_counter()
import spacr.qt
spacr.qt._quiet_gtk_accessibility()
spacr.qt._install_quiet_qt_logging()
from spacr.qt.app import launch
registered = spacr.qt.register_self_registering_modules()
report(seconds=time.perf_counter() - start,
       registered=list(registered),
       expected=list(spacr.qt.SELF_REGISTERING_MODULES))
"""

#: Every module in the Qt package, imported one by one. ``walk_packages``
#: enumerates them from the filesystem, so a module added tomorrow is covered
#: without anyone remembering to list it here.
QT_WHOLE_PACKAGE = _PREAMBLE + """
import importlib, os, pkgutil
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import spacr.qt
names = sorted(m.name for m in pkgutil.walk_packages(spacr.qt.__path__,
                                                     "spacr.qt."))
failed = {}
start = time.perf_counter()
for name in names:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed[name] = "%s: %s" % (type(exc).__name__, exc)
report(seconds=time.perf_counter() - start, imported=names, failed=failed)
"""

SPACR_UTILS = _PREAMBLE + """
start = time.perf_counter()
import spacr.utils
report(seconds=time.perf_counter() - start)
"""


def run_payload(code: str) -> dict:
    """Run ``code`` in a fresh interpreter and return the JSON it reported.

    The GPU is hidden from the subprocess, and that is what makes the numbers
    in this file mean the same thing everywhere. Every ceiling here was
    measured on a CPU-only runner, where ``import spacr.utils`` holds 833 MB.
    On a workstation with a CUDA device the same import can initialise a CUDA
    context — the driver, cuBLAS and cuDNN, not spaCR — and the identical
    import measures **3721 MB**, four and a half times the ceiling.

    It did not even fail consistently, which is how it read as flakiness for
    an evening: run this file alone and the import took the CPU path at
    839 MB; run it after ``tests/test_models_regression_umap.py`` in the slow
    suite and it took the CUDA path at 3721 MB, deterministically, in both
    directions, every time.

    ``CUDA_VISIBLE_DEVICES=""`` pins it to the path the docstrings describe:
    what a pipeline start costs and what the GUI would have to hold. Measured
    with it set, on the GPU box, the answer is 839 MB — the CI number, to
    within noise. Nothing here asserts anything about CUDA.

    The OpenMP and TensorFlow variables go too, and they are the other half of
    the same story. With CUDA hidden the import still measured 1877 MB after
    that same neighbour, because something it imports pulls TENSORFLOW into
    the pytest process, and TensorFlow sets ``KMP_DUPLICATE_LIB_OK=True``,
    ``KMP_INIT_AT_FORK=FALSE``, ``TF2_BEHAVIOR`` and ``TPU_ML_PLATFORM`` in
    ``os.environ`` at import. ``run_payload`` copied the environment, so those
    reached the measured subprocess and more than doubled its resident set
    through OpenMP alone. A guard whose number depends on which test ran
    before it is not a guard, so the variables that change the answer are
    named and dropped here. (spaCR itself never imports TensorFlow —
    ``tests/test_no_tensorflow_guard.py`` — and CI does not install it, which
    is why this only ever bit a workstation.)
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in list(env):
        if name.startswith(("KMP_", "OMP_", "TPU_ML_PLATFORM")) or name in (
                "TF2_BEHAVIOR", "ENABLE_RUNTIME_UPTIME_TELEMETRY"):
            del env[name]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO_ROOT), env=env, timeout=900)
    assert out.returncode == 0, (
        f"the measured subprocess failed:\n{out.stderr[-3000:]}")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith(SENTINEL)]
    assert len(lines) == 1, f"expected one result line, got {len(lines)}"
    return json.loads(lines[0][len(SENTINEL):])


def best_of(code: str, runs: int, key: str = "seconds") -> dict:
    """The cheapest of ``runs`` fresh subprocesses, by ``key``.

    The best rather than the mean, for the reason every timing in this suite
    takes the best: the machine is shared, and the mean measures the
    neighbour.
    """
    return min((run_payload(code) for _ in range(runs)), key=lambda r: r[key])


# ---------------------------------------------------------------------------
# What the expensive import actually costs
# ---------------------------------------------------------------------------

@pytest.mark.heavy
def test_importing_spacr_utils_stays_within_its_time_and_memory_ceiling():
    """Measured 3.3 s and 833 MB; ceilings 15 s and 1600 MB.

    Both halves matter and they fail differently. The seconds are what a user
    waits when a pipeline starts; the resident memory is what the GUI would
    have to hold for the rest of the session if any Qt module ever imported
    this. 900 MB is more than every widget, screen and image in the interface
    put together (172 MB, asserted below).

    The time ceiling is deliberately loose — 4.5x — because import time is
    dominated by reading shared objects off disk and by whatever else the
    machine is doing. Memory is not noisy in that way, so its ceiling is 1.9x
    and means something.
    """
    result = best_of(SPACR_UTILS, runs=2)
    assert result["seconds"] < 15.0, (
        f"importing spacr.utils took {result['seconds']:.1f} s; it is "
        f"documented at 3.2 s and measured 3.3 s. It pulls "
        f"{', '.join(result['heavy'])}")
    assert result["rss_mb"] < 1600.0, (
        f"importing spacr.utils held {result['rss_mb']:.0f} MB resident; it "
        "is documented at ~900 MB and measured 833 MB")
    # The reason it is expensive, asserted so the ceilings above stay legible:
    # if torch ever stops arriving here, these numbers should be re-taken
    # rather than left standing.
    assert "torch" in result["heavy"], (
        "spacr.utils no longer imports torch — its 3.2 s / 900 MB figures, "
        "and every ceiling in this file that is justified by them, need "
        "re-measuring")


# ---------------------------------------------------------------------------
# The invariant: the Qt layer never reaches spacr.utils
# ---------------------------------------------------------------------------

def _console_scripts() -> dict:
    """``{command: target}`` for every console script in setup.py."""
    text = (REPO_ROOT / "setup.py").read_text(encoding="utf8")
    block = text.split("console_scripts", 1)[1]
    return dict(re.findall(r"'([\w.-]+)=([\w.]+:[\w.]+)'", block))


def test_the_spacr_qt_console_script_points_at_the_module_these_guards_cover():
    """Pins what "the Qt entry point" means, so the guard cannot drift.

    Three commands share it — ``spacr``, ``spacr-qt`` and ``spacr-nightly``
    are all ``spacr.qt:run`` — and ``spacr-tutorial`` is the fourth Qt-package
    entry point. Every one of them is inside the surface the next two tests
    import.
    """
    scripts = _console_scripts()
    assert scripts.get("spacr-qt") == "spacr.qt:run", scripts
    assert scripts.get("spacr") == "spacr.qt:run", scripts
    qt_targets = {t.split(":")[0] for t in scripts.values()
                  if t.startswith("spacr.qt")}
    assert qt_targets == {"spacr.qt", "spacr.qt.tutorial.__main__"}, (
        f"a new Qt console script appeared: {sorted(qt_targets)}. Check it is "
        "covered by test_no_module_in_the_qt_package_imports_spacr_utils")


@_NEEDS_QT
@pytest.mark.qt
def test_the_spacr_qt_launch_path_never_imports_spacr_utils():
    """THE architectural guard: starting the GUI costs 0.57 s, not 3.8 s.

    The subprocess runs exactly what ``spacr.qt.run`` runs before it hands
    control to ``QApplication.exec`` — the quiet-logging setup, ``from
    spacr.qt.app import launch``, and the registration pass over the
    self-registering apps (feature dictionary, chaining, prerun, run
    compare). That is the whole import cost of ``spacr-qt``.

    ``spacr.utils`` must not be among what it imported. Nor may torch, cv2,
    tkinter, IPython or ``matplotlib.pyplot``: they are the reason it is
    expensive, and any of them arriving by another road costs the same
    seconds.

    This is asserted as an absence rather than as a duration on purpose. A
    duration on a shared machine is a coin flip; ``'spacr.utils' in
    sys.modules`` is a fact, and it names the regression.
    """
    result = best_of(QT_LAUNCH_PATH, runs=2)
    assert result["heavy"] == [], (
        f"starting spacr-qt imported {', '.join(result['heavy'])}. The Qt "
        "layer is built never to: spacr.utils alone is 3.2 s and 900 MB "
        "before the first window is drawn. Whatever needs it must import it "
        "inside the function that runs, not at module level")
    # The registration pass really ran — otherwise the four self-registering
    # modules (and whatever they import) would be outside this guard.
    assert result["registered"] == result["expected"], (
        f"only {result['registered']} of {result['expected']} registered")
    # Measured 0.57 s against spacr.utils' 3.3 s. A 4 s absolute ceiling on
    # that looked generous and was not: this box runs the whole suite beside
    # several agents, and the guard failed on wall clock alone inside a batch
    # where it had passed on its own moments before. A guard that flakes gets
    # deleted, and deleting this one would cost the invariant above.
    #
    # So the duration is claimed as a RATIO against the import it exists to
    # avoid, taken in the same conditions seconds apart: whatever the machine
    # is doing to one subprocess it is doing to the other. Measured 0.17x
    # (0.57 / 3.3); a Qt layer that reached spacr.utils by any road could not
    # be below 1.0. The absolute ceiling stays, at a value only a catastrophe
    # reaches, because a ratio alone would pass if both halves got slow
    # together.
    heavy_cost = best_of(SPACR_UTILS, runs=2)["seconds"]
    ratio = result["seconds"] / heavy_cost
    assert ratio < 0.5, (
        f"the spacr-qt launch path cost {result['seconds']:.2f} s against "
        f"spacr.utils' {heavy_cost:.2f} s in the same conditions — "
        f"{ratio:.2f}x, where it measured 0.17x. Something heavy arrived at "
        "module level")
    assert result["seconds"] < 15.0, (
        f"the spacr-qt launch path imported in {result['seconds']:.2f} s; it "
        "measured 0.57 s")
    assert result["rss_mb"] < 600.0, (
        f"the spacr-qt launch path held {result['rss_mb']:.0f} MB resident; "
        "it measured 172 MB. Memory is not noisy the way wall clock is, so "
        "this ceiling is 3.5x and means something")


@_NEEDS_QT
@pytest.mark.qt
@pytest.mark.heavy
def test_no_module_in_the_qt_package_imports_spacr_utils():
    """The same invariant over the *whole* package, not just the entry point.

    ``pkgutil.walk_packages`` enumerates every module under ``spacr/qt`` —
    127 of them on this tree, screens, widgets, the AI panel, the tutorial
    renderer, both ``__main__`` files — and each is imported in one fresh
    process. None of them may drag in ``spacr.utils``.

    What the whole package *does* import is pandas, sklearn and skimage, for
    0.65 s and 289 MB all told. That is the budget this invariant buys: the
    entire Qt interface, every screen and widget in it, costs a third of what
    one import of ``spacr.utils`` costs.

    That is a stronger statement than the entry-point test: it holds for the
    screens a user never opens and the widgets a screen only builds lazily, so
    the invariant cannot be broken by a module that merely happens not to be
    on the startup path today.

    Modules whose *third-party* dependency is missing are tolerated and
    named — an optional reader that is not installed is an environment fact,
    not a defect. Anything else failing to import is reported as a failure,
    because a Qt module that cannot be imported cannot be shown either.
    """
    result = run_payload(QT_WHOLE_PACKAGE)
    optional, broken = {}, {}
    for name, error in result["failed"].items():
        (optional if error.startswith("ModuleNotFoundError")
         and "spacr" not in error else broken)[name] = error

    assert result["heavy"] == [], (
        f"a module in spacr.qt imports {', '.join(result['heavy'])} at module "
        f"level. All {len(result['imported'])} were imported; the offender is "
        "whichever one names it outside a function")
    assert not broken, f"Qt modules that failed to import: {broken}"
    assert len(result["imported"]) > 100, (
        f"only {len(result['imported'])} modules were walked; the package has "
        "over a hundred, so the walk did not cover it")
    # The entry points the console scripts name are inside what was walked.
    for target in ("spacr.qt.app", "spacr.qt.tutorial.__main__"):
        assert target in result["imported"], f"{target} was not covered"
    if optional:                                    # pragma: no cover - env
        print(f"optional dependencies missing: {sorted(optional)}")
