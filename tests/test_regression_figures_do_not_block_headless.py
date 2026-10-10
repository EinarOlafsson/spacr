"""``spacr.ml.perform_regression`` must not hang on ``plt.show()`` headless.

Reported 2026-10-10: a scripted regression run on a machine with a display
variable set (matplotlib picks QtAgg) and ``QT_QPA_PLATFORM=offscreen`` sat
forever on the first plate heatmap, because ``plt.show()`` entered a Qt event
loop that nothing could end. Only ``MPLBACKEND=Agg`` got it through.

The regression figure helpers now go through ``_show_if_interactive`` (it
lives in :mod:`spacr.figure_sink`; ``spacr.ml`` keeps the names): a host
that captured ``pyplot.show`` and a notebook backend are always shown, a GUI
backend only from an interactive session on a real platform, Agg never.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import pytest

from spacr import figure_sink, ml


def _pyplot_show(calls):
    """A stand-in that looks like matplotlib's own ``pyplot.show``."""
    def show(*_args, **_kwargs):
        calls.append(1)
    show.__module__ = "matplotlib.pyplot"
    return show


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(plt, "show", _pyplot_show(recorded))
    return recorded


def _backend(monkeypatch, name):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: name)


def test_qt_offscreen_is_never_shown(monkeypatch, calls):
    _backend(monkeypatch, "QtAgg")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(figure_sink, "_interactive_session", lambda: True)
    assert ml._show_if_interactive() is False
    assert calls == []


def test_a_gui_backend_in_a_script_is_not_shown(monkeypatch, calls):
    _backend(monkeypatch, "TkAgg")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(figure_sink, "_interactive_session", lambda: False)
    assert ml._show_if_interactive() is False
    assert calls == []


def test_a_gui_backend_in_an_interactive_session_is_shown(monkeypatch, calls):
    _backend(monkeypatch, "QtAgg")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(figure_sink, "_interactive_session", lambda: True)
    assert ml._show_if_interactive() is True
    assert calls == [1]


def test_agg_is_not_shown(monkeypatch, calls):
    _backend(monkeypatch, "agg")
    monkeypatch.setattr(figure_sink, "_interactive_session", lambda: True)
    assert ml._show_if_interactive() is False
    assert calls == []


def test_a_notebook_backend_is_shown(monkeypatch, calls):
    _backend(monkeypatch, "module://matplotlib_inline.backend_inline")
    assert ml._show_if_interactive() is True
    assert calls == [1]


def test_a_captured_show_is_always_called(monkeypatch):
    recorded = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: recorded.append(1))
    _backend(monkeypatch, "agg")
    assert ml._show_if_interactive() is True
    assert recorded == [1]


def test_the_session_test_follows_ion(monkeypatch):
    monkeypatch.delattr(sys, "ps1", raising=False)
    monkeypatch.setattr(sys, "flags", type("F", (), {"interactive": 0})())
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setattr(matplotlib, "is_interactive", lambda: False)
    assert ml._interactive_session() is False
    monkeypatch.setattr(matplotlib, "is_interactive", lambda: True)
    assert ml._interactive_session() is True


def test_an_ipython_kernel_is_interactive(monkeypatch):
    monkeypatch.delattr(sys, "ps1", raising=False)
    monkeypatch.setattr(sys, "flags", type("F", (), {"interactive": 0})())
    monkeypatch.setattr(matplotlib, "is_interactive", lambda: False)
    fake = type("IPython", (), {"get_ipython": staticmethod(lambda: object())})
    monkeypatch.setitem(sys.modules, "IPython", fake)
    assert ml._interactive_session() is True
    broken = type("IPython", (), {"get_ipython": staticmethod(
        lambda: (_ for _ in ()).throw(RuntimeError("no shell")))})
    monkeypatch.setitem(sys.modules, "IPython", broken)
    assert ml._interactive_session() is False


def test_the_gui_backend_list_names_qtagg():
    assert "qtagg" in ml._interactive_backends()
    assert "agg" not in ml._interactive_backends()


def test_the_plate_panel_does_not_show_on_offscreen_qt(monkeypatch, calls):
    pd = pytest.importorskip("pandas")
    _backend(monkeypatch, "QtAgg")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    frame = pd.DataFrame(
        [{"plateID": "plate1", "rowID": row, "columnID": column,
          "prc": f"plate1_{row}_{column}", "fraction": 0.1 * column}
         for row in "ABCDEFGH" for column in range(1, 13)])
    assert ml._show_plates(frame, "fraction", None) is True
    assert calls == []


def test_a_real_qtagg_offscreen_show_returns():
    """End to end in a child: real QtAgg, offscreen, no MPLBACKEND=Agg."""
    pytest.importorskip("PySide6")
    script = textwrap.dedent("""
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        from spacr import ml
        plt.figure()
        print("shown" if ml._show_if_interactive() else "skipped")
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env.pop("MPLBACKEND", None)
    done = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "skipped"


def test_the_backend_registry_is_used_when_matplotlib_has_one(monkeypatch):
    import matplotlib.backends as backends

    registry = type("Registry", (), {"list_builtin": staticmethod(
        lambda kind: ["QtAgg", "MacOSX"] if kind == "gui" else [])})
    monkeypatch.setattr(backends, "BackendFilter",
                        type("Filter", (), {"INTERACTIVE": "gui"}),
                        raising=False)
    monkeypatch.setattr(backends, "backend_registry", registry, raising=False)
    assert ml._interactive_backends() == {"qtagg", "macosx"}


def test_the_rcsetup_list_is_used_without_a_registry(monkeypatch):
    import matplotlib.backends as backends
    from matplotlib import rcsetup

    monkeypatch.delattr(backends, "backend_registry", raising=False)
    monkeypatch.setattr(rcsetup, "interactive_bk", ["TkAgg"], raising=False)
    assert ml._interactive_backends() == {"tkagg"}


def test_the_run_guard_silences_show_where_it_would_block(monkeypatch, calls):
    _backend(monkeypatch, "QtAgg")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    original = plt.show
    with ml._ShowOnlyWhereItReturns():
        assert plt.show is not original
        plt.show()
    assert plt.show is original
    assert calls == []


def test_the_run_guard_leaves_a_working_show_alone(monkeypatch, calls):
    _backend(monkeypatch, "QtAgg")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(figure_sink, "_interactive_session", lambda: True)
    original = plt.show
    with ml._ShowOnlyWhereItReturns():
        assert plt.show is original
        plt.show()
    assert calls == [1]


def test_perform_regression_runs_under_the_guard(monkeypatch, calls):
    _backend(monkeypatch, "QtAgg")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    seen = []

    def fake_run(settings):
        plt.show()
        seen.append(plt.show.__module__ == "spacr.figure_sink")
        return {}

    monkeypatch.setattr(ml, "_perform_regression", fake_run)
    monkeypatch.setattr(ml, "_write_regression_panel_packages",
                        lambda *a, **k: None)
    monkeypatch.setattr(ml, "_write_fit_resources", lambda *a, **k: "")
    assert ml.perform_regression({}) == {}
    assert seen == [True] and calls == []
    assert plt.show.__module__ == "matplotlib.pyplot"


def test_ml_keeps_the_shared_helpers_as_aliases():
    assert ml._show_if_interactive is figure_sink._show_if_interactive
    assert ml._can_show is figure_sink._can_show
    assert ml._ShowOnlyWhereItReturns is figure_sink._ShowOnlyWhereItReturns
