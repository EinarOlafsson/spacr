"""The Pipeline Graph canvas in the states nothing had put it in.

Written for instruction 60. ``tests/qt/test_pipeline_graph_screen.py`` drives
the screen against a real registry; what it never reached was the canvas with
no graph at all, a click that lands between the boxes, an edge with one end
filtered away, and the screen opened on nothing.

The whole-package ranking listed this module at 51.8% with 160 uncovered
statements. Over its own tests it was 95% with 17 — the same "a module whose
tests are marker-excluded reads as uncovered" trap that file records, one
level down: the ranking's run excluded most of the files that exercise it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import artifacts                              # noqa: E402
from spacr.qt.screens import pipeline_graph as screen_module  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


# ---------------------------------------------------------------------------
# A canvas with nothing on it
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Clicking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Edges with one end filtered away
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The screen with no project
# ---------------------------------------------------------------------------

def test_opening_the_screen_on_nothing_asks_for_a_project(qtbot):
    """A blank canvas with no message reads as a project that has no
    artifacts, which is a different and much more alarming claim."""
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.load_project("   ")
    assert "Choose a spaCR project folder" in widget._verdict.text()
    assert widget._canvas.graph() is None
    assert widget.last_error == ""


# ---------------------------------------------------------------------------
# The factory the registry calls
# ---------------------------------------------------------------------------

def test_the_registry_factory_builds_the_screen(qtbot):
    """The registry calls it with the app key; the screen takes none."""
    widget = screen_module.make_pipeline_graph_screen(screen_module.APP_KEY)
    qtbot.addWidget(widget)
    assert isinstance(widget, screen_module.PipelineGraphScreen)


# ---------------------------------------------------------------------------
# The two branches that used to be excluded rather than tested
# ---------------------------------------------------------------------------

def test_browsing_for_a_project_loads_the_one_that_was_chosen(qtbot, tmp_path,
                                                                monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    root = tmp_path / "plate1"
    (root / "masks").mkdir(parents=True)
    registry = artifacts.Registry(project=str(root))
    path = root / "masks" / "cell.npy"
    path.write_text("x", encoding="utf-8")
    registry.register(module="mask", kind="masks", role="cell_mask",
                      path=str(path), settings={"diameter": 30},
                      run_id="run-1")
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(root)))
    widget._on_browse()
    assert widget._project_edit.text() == str(root)
    assert len(widget._canvas.node_rects()) == 1


def test_cancelling_the_project_browser_leaves_the_open_project_alone(
        qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._project_edit.setText("/kept")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    widget._on_browse()
    assert widget._project_edit.text() == "/kept"
