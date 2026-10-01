"""Item 608: Plaque's live preview defaults to the Cellpose-SAM plaque model.

Found by spacr-d7 while re-recording tutorials, 2026-10-01. The preview
spelled an unset ``plaque_model`` as ``'bundled'``, the Cellpose 3
checkpoint Cellpose 4 refuses, while the run read the same unset value as
``toxoplasma_plaque_v2``. The old checkpoint stays selectable and runs
through the Cellpose 3 backend; without it the preview says what to do.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import submodules as SUB                                   # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv                    # noqa: E402
from spacr.qt.widgets import preview_contract as pc                   # noqa: E402

REFUSAL = ("This model does not appear to be a CP4 model. CP3 models are "
           "not compatible with CP4.")


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_an_unseeded_panel_asks_for_the_zoo_model(panel):
    assert panel.current_settings()["plaque_model"] == "toxoplasma_plaque_v2"


def test_settings_without_a_plaque_model_select_the_zoo_model(panel):
    panel.apply_settings({"diameter": 30})
    assert panel._model_box.currentText() == "toxoplasma_plaque_v2"
    assert panel._model_note.text() == ""
    assert panel.current_settings()["plaque_model"] == "toxoplasma_plaque_v2"


def test_the_bundled_checkpoint_stays_selectable(panel):
    choices = ppv.plaque_model_choices([])
    assert choices[0] == "toxoplasma_plaque_v2"
    assert choices[-1] == "bundled"
    panel.apply_settings({"plaque_model": "bundled"})
    assert panel.current_settings()["plaque_model"] == "bundled"
    assert "historical" in panel._model_note.text()


def test_the_preview_resolves_an_unset_model_as_the_run_does(monkeypatch):
    asked = []
    monkeypatch.setattr(SUB, "_resolve_plaque_model",
                        lambda s, fetch=True: asked.append(
                            SUB._requested_plaque_model(s)) or "/m/r5")
    path, note, entry = ppv.resolve_plaque_model({})
    assert (path, note, entry) == ("/m/r5", "", None)
    assert asked == ["toxoplasma_plaque_v2"]


def _refuse_like_cellpose4(path, gpu=None):
    try:
        raise ValueError(REFUSAL)
    except ValueError as exc:
        raise SUB.Cellpose3Checkpoint("preview refusal") from exc


def test_the_preview_runs_a_cellpose3_checkpoint_through_the_backend(
        monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pc, "preview_cellpose_model", _refuse_like_cellpose4)
    monkeypatch.setattr(SUB, "_cellpose3_plaque_backend",
                        lambda path: sentinel)
    ppv._MODELS.clear()
    try:
        assert ppv._cellpose_model("/m/old.CP_model") is sentinel
    finally:
        ppv._MODELS.clear()


def test_without_the_backend_the_preview_says_what_to_do(monkeypatch):
    monkeypatch.setattr(pc, "preview_cellpose_model", _refuse_like_cellpose4)
    monkeypatch.setattr(SUB, "_cellpose3_plaque_backend", lambda path: None)
    ppv._MODELS.clear()
    with pytest.raises(SUB.Cellpose3Checkpoint) as excinfo:
        ppv._cellpose_model("/m/old.CP_model")
    ppv._MODELS.clear()
    text = ppv._explain_model_failure("/m/old.CP_model", excinfo.value)
    assert "Cellpose 3 model" in text
    assert "toxoplasma_plaque_v2" in text
    assert "Cellpose 3 backend" in text
