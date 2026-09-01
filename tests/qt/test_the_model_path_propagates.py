"""Propagate must carry the model, not only the numbers.

`model_name` is the TRAINING module's key. The Mask panel holds one checkpoint
per object -- `pathogen_model_name`, `cell_model_name`, and so on -- so
propagating `model_name` alone wrote to a field the Mask panel never shows: a
custom pathogen checkpoint could be tuned in the live preview, propagated, and
the real run would still segment with cpsam.

Reported 2026-09-01: "the custom parasite model path did not propagate".
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.live_preview import LivePreviewPanel

CUSTOM = "/home/user/.spacr/models/toxoplasma_pv_v1.pth"


@pytest.fixture
def panel(qapp):
    return LivePreviewPanel()


def test_a_custom_pathogen_model_reaches_the_pathogen_field(panel):
    panel._object_box.setCurrentText("pathogen")
    panel._model_box.addItem(CUSTOM)
    panel._model_box.setCurrentText(CUSTOM)

    out = panel.settings_for_propagation()

    assert out["pathogen_model_name"] == CUSTOM, (
        "the checkpoint the preview segmented with never reached the field "
        "the run reads")


def test_each_object_gets_its_own_model_field(panel):
    """Not a pathogen special case."""
    panel._model_box.addItem(CUSTOM)
    panel._model_box.setCurrentText(CUSTOM)
    for obj in ("cell", "nucleus", "pathogen", "organelle"):
        panel._object_box.setCurrentText(obj)
        out = panel.settings_for_propagation()
        assert out[f"{obj}_model_name"] == CUSTOM, obj


def test_the_model_is_not_written_to_some_other_objects_field(panel):
    """Propagating a pathogen checkpoint into `cell_model_name` would make the
    cell pass load a pathogen model -- worse than not propagating at all."""
    panel._object_box.setCurrentText("pathogen")
    panel._model_box.addItem(CUSTOM)
    panel._model_box.setCurrentText(CUSTOM)

    out = panel.settings_for_propagation()

    assert "cell_model_name" not in out
    assert "nucleus_model_name" not in out


def test_the_stock_model_propagates_too(panel):
    """A user who moves BACK to cpsam must have that carried as well, or the
    main panel keeps a custom checkpoint the preview is no longer using."""
    panel._object_box.setCurrentText("pathogen")
    panel._model_box.setCurrentText("cpsam")
    assert panel.settings_for_propagation()["pathogen_model_name"] == "cpsam"


def test_cell_plus_nucleus_writes_the_primary(panel):
    panel._object_box.setCurrentText("cell + nucleus")
    panel._model_box.addItem(CUSTOM)
    panel._model_box.setCurrentText(CUSTOM)
    assert panel.settings_for_propagation()["cell_model_name"] == CUSTOM
