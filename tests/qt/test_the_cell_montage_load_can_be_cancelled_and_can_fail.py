"""The montage load stops when asked and says when its runner fails.

Pinned here, each as what the user sees or gets:

* a load asked to stop before it begins reports "cancelled" and never
  reports a stage;
* cutting crops without a stage callback still reports the folders that
  have no crop source;
* progress from a load the user has cancelled is not shown, a stage from an
  older load is ignored, and a load whose progress receiver has gone is
  cancelled rather than crashing;
* a runner that itself raised says so on the status line and emits
  ``montage_failed``;
* a mirrored control of a kind the settings window does not know is left
  alone and not reported back.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QCheckBox  # noqa: E402

from spacr.qt.widgets import cell_montage_view as cmv  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def view(qtbot):
    widget = cmv.CellMontageView(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def _request(**over):
    fields = {"name": "guide_1", "effect": 0.5}
    fields.update(over)
    return cmv.MontageRequest(**fields)


def _ready_to_build(view, monkeypatch, fake_load):
    monkeypatch.setattr(view, "request", _request)
    monkeypatch.setattr(view, "_multivariate_is_ready", lambda _r: True)
    monkeypatch.setattr(cmv, "load", fake_load)


# ---------------------------------------------------------------------------
# The worker side
# ---------------------------------------------------------------------------

def test_a_load_stopped_before_it_begins_says_cancelled():
    stages = []
    result = cmv.load(_request(results_path="", count_csvs=(), databases=()),
                      progress=stages.append, cancelled=lambda: True)
    assert result.error == "Montage loading cancelled."
    assert stages == []


def test_cutting_without_a_stage_callback_still_names_missing_sources():
    plan = SimpleNamespace(rows=lambda: [{"montage_source_root": "/plates/a"},
                                         {"montage_source_root": "/plates/a"}])
    troubles = []
    crops = cmv._cut(plan, {}, _request(), troubles)
    assert crops == (None, None)
    assert troubles == ["/plates/a has no crop source; its objects are blank"]


# ---------------------------------------------------------------------------
# Progress on the GUI side
# ---------------------------------------------------------------------------

def test_progress_after_a_cancel_is_not_shown(view, monkeypatch):
    def fake_load(request, *, progress=None, cancelled=None):
        progress("Reading 3 crops from /plates/a…")
        assert view.status_text() == "Reading 3 crops from /plates/a…"
        view._cancel_loading()
        progress("Preparing the montage for display…")
        return cmv.MontageLoad(request=request,
                               error="Montage loading cancelled.")

    _ready_to_build(view, monkeypatch, fake_load)
    assert view.build() is True
    assert "Preparing the montage" not in view.status_text()


def test_a_stage_from_an_older_load_is_ignored(view):
    before = view.status_text()
    view._on_load_progress(object(), "Reading 9 crops from /old…")
    assert view.status_text() == before


def test_a_load_whose_receiver_has_gone_is_cancelled(view, monkeypatch):
    seen = []

    class Gone:
        def emit(self, *_args):
            raise RuntimeError("Internal C++ object already deleted.")

    def fake_load(request, *, progress=None, cancelled=None):
        progress("Reading 3 crops from /plates/a…")
        seen.append(cancelled())
        return cmv.MontageLoad(request=request,
                               error="Montage loading cancelled.")

    _ready_to_build(view, monkeypatch, fake_load)
    monkeypatch.setattr(view, "_load_progress", Gone())
    assert view.build() is True
    assert seen == [True]
    assert view._load_cancel.is_set()


def test_a_runner_that_raised_says_so(view, qtbot):
    with qtbot.waitSignal(view.montage_failed, timeout=1000) as failed:
        view._on_job_failed("worker thread died")
    assert failed.args == ["worker thread died"]
    assert view.status_text() == "The montage load failed: worker thread died"
    assert view._pending is None


# ---------------------------------------------------------------------------
# Mirrored settings
# ---------------------------------------------------------------------------

def test_a_control_of_an_unknown_kind_is_left_alone(view, monkeypatch):
    name = view._MIRRORED["cap"]
    odd = QCheckBox()
    monkeypatch.setattr(view, name, odd)
    view._write_back({"cap": 7})
    assert odd.isChecked() is False
    assert "cap" not in view._read_widgets()
