"""The Qt GUI installs with spaCR, and still does not load headless.

Asked for on 2026-08-17: "lets stop hiding the qt behind a qt. allways
install qt as well as the tkinter stuff".

`spacr-qt` is a console entry point declared unconditionally, so a plain
`pip install spacr` installed a command that could not run. tkinter needs no
declaration -- it ships with CPython -- so the Tk GUI always worked out of
the box and the PRIMARY GUI did not.

The two things that could go wrong are opposite, and both are pinned here: a
core dependency with no wheel is an INSTALL FAILURE on an interpreter that
used to work headless, and a core dependency that gets IMPORTED by the
pipeline puts Qt on a cluster's import path.
"""
from __future__ import annotations

import pathlib
import runpy
import subprocess
import sys

import pytest


def _metadata() -> dict:
    """setup.py's own call arguments, the way the packaging scripts read it."""
    import setuptools

    captured: dict = {}
    real = setuptools.setup
    setuptools.setup = lambda **kw: captured.update(kw)
    try:
        import spacr

        path = pathlib.Path(spacr.__file__).parent.parent / "setup.py"
        runpy.run_path(str(path), run_name="__not_main__")
    finally:
        setuptools.setup = real
    return captured


# --------------------------------------------------------------------------- #
#  It ships
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("package", ["PySide6", "qtawesome", "pyqtgraph"])
def test_the_qt_stack_is_a_core_dependency(package):
    deps = _metadata()["install_requires"]

    assert any(package.lower() in d.lower() for d in deps), (
        f"{package} is not a core dependency, so `pip install spacr` leaves "
        f"`spacr-qt` unable to start")


def test_spacr_qt_is_declared_unconditionally():
    """Which is why the dependency has to be unconditional too -- an entry
    point that cannot run is worse than one that is absent."""
    entry_points = _metadata()["entry_points"]["console_scripts"]

    assert any(e.startswith("spacr-qt=") for e in entry_points), entry_points


def test_the_qt_extra_still_resolves():
    """Existing instructions, the three packaging scripts, the README and
    nine translations all say `spacr[qt]`. Removing the name would break
    printed instructions to no benefit."""
    extras = _metadata()["extras_require"]

    assert "qt" in extras
    assert any("PySide6" in d for d in extras["qt"])


# --------------------------------------------------------------------------- #
#  It still does not load where there is no display
# --------------------------------------------------------------------------- #

def test_the_pipeline_does_not_import_pyside6():
    """BEING A DEPENDENCY IS NOT BEING AN IMPORT. A cluster install carries
    the wheel and must never load it: importing PySide6 on a machine with no
    display is how a headless run dies at module scope.

    In a subprocess, because an import cannot be undone and every Qt test in
    this suite has already imported PySide6 by the time this runs.
    """
    script = (
        "import sys\n"
        "import spacr.core, spacr.ml, spacr.measure, spacr.io, spacr.utils\n"
        "print('PYSIDE6:' + str('PySide6' in sys.modules))\n"
        "print('PYQTGRAPH:' + str('pyqtgraph' in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]

    assert "PYSIDE6:False" in out.stdout, (
        "importing the pipeline pulled in PySide6")
    assert "PYQTGRAPH:False" in out.stdout, (
        "importing the pipeline pulled in pyqtgraph")


# --------------------------------------------------------------------------- #
#  It installs on every interpreter the package claims
# --------------------------------------------------------------------------- #

def test_the_qt_pins_span_the_whole_support_matrix():
    """A core dependency with no wheel for a supported interpreter turns a
    working headless install into a resolver error.

    Checked against the ranges rather than the newest release: PySide6 6.11
    declares >=3.10, and spaCR supports 3.9 -- so the pin has to admit the
    6.6/6.7 line, which does. Same for pyqtgraph, where 0.14 is >=3.10 and
    0.13 covers 3.9.
    """
    deps = _metadata()["install_requires"]
    pyside = next(d for d in deps if d.lower().startswith("pyside6"))
    graph = next(d for d in deps if d.lower().startswith("pyqtgraph"))

    assert ">=6.6" in pyside, (
        f"{pyside} must admit the 6.6 line, the last one supporting 3.9")
    assert ">=0.13" in graph, (
        f"{graph} must admit 0.13, the last line supporting 3.9")
