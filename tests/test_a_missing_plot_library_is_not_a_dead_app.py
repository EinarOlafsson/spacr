"""pyqtgraph is optional, so its absence must cost the plots and nothing else.

Reported from a real install on 2026-08-17: a machine with PySide6 and no
pyqtgraph could not open ANY module. The traceback ran

    _on_nav_selected -> _build_screen -> AppScreen.__init__
      -> _build_runtime_panel -> build_parameter_sweep_card
      -> ParameterSweepScreen.__init__ -> RegressionResultsPanel.__init__
      -> VolcanoPlot() -> _require_pyqtgraph() -> RuntimeError

so an optional plotting library took down mask, measure, classify and
everything else, none of which draws a volcano. INVARIANTS 10 -- decoration
must cost the decoration.

The message it raised made it worse: "or use the matplotlib figures" named a
fallback that does not exist. Telling a user there is another way and then
dying is worse than dying.

Two causes, both fixed and both pinned here: pyqtgraph was not a declared
dependency at all, and the widget raised instead of degrading.
"""
from __future__ import annotations

import builtins
import os
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  It is declared
# --------------------------------------------------------------------------- #

def test_pyqtgraph_is_a_declared_dependency():
    """`pip install spacr[qt]` must bring the interactive plots with it.

    The comment at the top of setup.py records pyqtgraph being REMOVED as a
    second, unused Qt binding -- true when written, since nothing imported
    it. fast_plots.py made it load-bearing and nothing updated the
    declaration.
    """
    import pathlib

    import spacr

    source = pathlib.Path(spacr.__file__).parent.parent / "setup.py"
    text = source.read_text()
    assert "pyqtgraph" in text
    # In the extra that also carries PySide6 -- the same install situation.
    qt_extra = text.split("'qt': [", 1)[1].split("],", 1)[0]
    assert "pyqtgraph" in qt_extra, qt_extra


# --------------------------------------------------------------------------- #
#  Its absence degrades
# --------------------------------------------------------------------------- #

def _without_pyqtgraph(body: str) -> str:
    """Run ``body`` in a subprocess where pyqtgraph cannot be imported.

    A SUBPROCESS because an import cannot be undone: pyqtgraph is installed
    in this environment and already in sys.modules by the time any test runs,
    so patching it out in-process tests a half-loaded state rather than a
    machine that never had it.
    """
    script = textwrap.dedent("""
        import builtins, os, sys
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        _real = builtins.__import__
        def _blocked(name, *a, **k):
            if name == 'pyqtgraph' or name.startswith('pyqtgraph.'):
                raise ImportError('blocked')
            return _real(name, *a, **k)
        builtins.__import__ = _blocked
        for _m in [m for m in sys.modules if m.startswith('pyqtgraph')]:
            del sys.modules[_m]
        from PySide6.QtWidgets import QApplication
        _app = QApplication([])
        from spacr.qt.widgets import fast_plots
        assert fast_plots.HAVE_PYQTGRAPH is False, 'the block did not take'
    """) + textwrap.dedent(body)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_the_results_panel_still_builds():
    """The exact constructor from the reported traceback."""
    out = _without_pyqtgraph("""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel
        panel = RegressionResultsPanel()
        print('BUILT', panel is not None)
    """)
    assert "BUILT True" in out


def test_the_whole_app_screen_still_builds():
    """The failure was three frames further out than the plot: every module
    in the application died, not only the regression one."""
    out = _without_pyqtgraph("""
        from spacr.qt.screens.app_screen import AppScreen
        for key in ('regression', 'mask', 'measure'):
            AppScreen(key)
            print('OK', key)
    """)
    for key in ("regression", "mask", "measure"):
        assert f"OK {key}" in out, out


def test_it_still_takes_a_table():
    """A panel that builds and then dies on its first redraw is a WORSE
    failure than the original: the app looks fine until data arrives."""
    out = _without_pyqtgraph("""
        import numpy as np, pandas as pd
        from spacr.qt.widgets.regression_results import RegressionResultsPanel
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            'feature': [f'fraction:grna[{i}_1]' for i in range(200)],
            'coefficient': rng.normal(0, .5, 200),
            'p_value': rng.uniform(size=200),
            'condition': list(rng.choice(['nc','pc','other'], 200,
                                         p=[.1,.05,.85]))})
        panel = RegressionResultsPanel()
        print('LOADED', panel.set_frame(frame))
    """)
    assert "LOADED True" in out


def test_it_says_what_is_missing_and_how_to_fix_it():
    """An empty box that does not say why is indistinguishable from a bug."""
    out = _without_pyqtgraph("""
        from PySide6.QtWidgets import QLabel
        from spacr.qt.widgets.regression_results import RegressionResultsPanel
        panel = RegressionResultsPanel()
        texts = ' '.join(w.text() for w in panel.volcano.findChildren(QLabel))
        print('SAYS', 'pyqtgraph' in texts)
        print('EXTRA', 'spacr[qt]' in texts)
        print('NO_PHANTOM_FALLBACK', 'matplotlib figures' not in texts)
    """)
    assert "SAYS True" in out
    # Names the EXTRA, not the bare distribution: a bare `pip install
    # pyqtgraph` into an env installed from an extra is removed again on the
    # next upgrade.
    assert "EXTRA True" in out
    # And it does not offer the fallback that does not exist.
    assert "NO_PHANTOM_FALLBACK True" in out


def test_the_flag_says_so_rather_than_the_widget_lying():
    out = _without_pyqtgraph("""
        from spacr.qt.widgets.fast_plots import VolcanoPlot
        print('AVAILABLE', VolcanoPlot().plots_available)
    """)
    assert "AVAILABLE False" in out


# --------------------------------------------------------------------------- #
#  With it present, nothing changed
# --------------------------------------------------------------------------- #

def test_the_plots_still_work_when_it_is_installed(qtbot):
    pytest.importorskip("pyqtgraph")
    import numpy as np
    import pandas as pd

    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(200)],
        "coefficient": rng.normal(0, .5, 200),
        "p_value": rng.uniform(size=200)})
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    assert panel.volcano.plots_available is True
    assert panel.set_frame(frame) is True
    assert len(panel.volcano._row_xy) == 200
