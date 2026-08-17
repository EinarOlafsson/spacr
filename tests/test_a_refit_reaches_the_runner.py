"""The re-fit the panel asks for is the run the screen starts.

The panel decides WHAT to run; it has no worker, no console and no Stop
button, so the screen that owns those starts it. This is the seam between
them, and it is worth pinning because both halves look correct on their own
when the signal is not connected at all.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _settings(**over):
    settings = {
        "count_data": ["/data/screen/counts.csv"],
        "score_data": ["/data/screen/scores.csv"],
        "regression_type": "rlm",
        "multiple_testing_method": "bonferroni",
        "plot": True,
    }
    settings.update(over)
    return settings


def _screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    return screen


def test_the_panel_is_connected_to_the_runner(qtbot):
    """Both halves look right on their own when the signal is not connected,
    and then right-clicking does nothing at all."""
    screen = _screen(qtbot)
    panel = getattr(screen, "_results_panel", None)
    if panel is None:
        pytest.skip("this build places no results panel on the screen")

    started = []
    screen._on_run = lambda *a, **k: started.append(k.get("override"))
    panel.refit_requested.emit(_settings())

    assert started and started[0]["regression_type"] == "rlm"


def test_the_override_is_run_instead_of_the_panel(qtbot, monkeypatch):
    """A re-fit builds its settings from the run on screen, not from the
    widgets -- which may have been edited since, so a re-fit that picked
    those up would compare two runs differing in more ways than the one the
    user chose."""
    screen = _screen(qtbot)
    started = []
    monkeypatch.setattr("spacr.qt.screens.app_screen.make_thread",
                        lambda entry, settings: started.append(settings)
                        or (_FakeThread(), _FakeWorker()))
    monkeypatch.setattr(screen._settings_model, "collect",
                        lambda: {"regression_type": "EDITED"})

    screen._on_run(override=_settings())

    assert started, "no run was started"
    assert started[0]["regression_type"] == "rlm"


def test_the_clicked_bool_is_not_a_settings_dict(qtbot, monkeypatch):
    """`clicked` passes False as the first positional argument. If the
    override were positional, pressing Run would try to run `False`."""
    screen = _screen(qtbot)
    started = []
    monkeypatch.setattr("spacr.qt.screens.app_screen.make_thread",
                        lambda entry, settings: started.append(settings)
                        or (_FakeThread(), _FakeWorker()))
    monkeypatch.setattr(screen._settings_model, "collect",
                        lambda: {"regression_type": "from the panel"})

    screen._on_run(False)

    assert started and started[0]["regression_type"] == "from the panel"


def test_a_refit_is_refused_while_a_run_is_going(qtbot, monkeypatch):
    """Two regressions writing at once is not a comparison. The folder rule
    claims a name by LOOKING, so both would look before either wrote and both
    would claim it."""
    screen = _screen(qtbot)
    started = []
    monkeypatch.setattr(screen, "_on_run",
                        lambda *a, **k: started.append(k))

    class _Running:
        def isRunning(self):
            return True

    screen._thread = _Running()
    assert screen._on_refit(_settings()) is False
    assert started == []


def test_a_refit_says_it_is_re_running(qtbot, monkeypatch):
    """"it must say it is re-running, and its output must not silently
    replace the run the user is looking at"."""
    screen = _screen(qtbot)
    monkeypatch.setattr(screen, "_on_run", lambda *a, **k: None)
    said = []
    monkeypatch.setattr(screen._console, "append_notice",
                        lambda source, **values: said.append(source))

    screen._thread = None
    screen._on_refit(_settings())

    assert any("NEW run" in text for text in said), said


def test_the_run_hands_its_settings_to_the_panel(qtbot):
    """Better than anything read off disk: the shared settings/ copy is
    overwritten by every later run of the same screen."""
    import numpy as np
    import pandas as pd

    screen = _screen(qtbot)
    panel = getattr(screen, "_results_panel", None)
    if panel is None:
        pytest.skip("this build places no results panel on the screen")

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(50)],
        "coefficient": rng.normal(size=50),
        "p_value": rng.uniform(size=50),
    })
    screen._on_pipeline_result({"results": frame, "res_folder": "",
                                "settings": _settings()})

    assert panel._run_settings is not None
    assert panel._run_settings["regression_type"] == "rlm"


def test_the_regression_returns_its_settings():
    """The panel can only be handed them if the run returns them."""
    import inspect

    from spacr import ml

    source = inspect.getsource(ml.perform_regression)
    assert "'settings': dict(settings)" in source


class _FakeThread:
    def __init__(self):
        self.started = _Signal()
        self.finished = _Signal()

    def start(self):
        pass

    def isRunning(self):
        return False


class _FakeWorker:
    def __init__(self):
        for name in ("line_ready", "error", "figure_ready", "result_ready",
                     "finished"):
            setattr(self, name, _Signal())


class _Signal:
    def connect(self, *_args, **_kwargs):
        pass
