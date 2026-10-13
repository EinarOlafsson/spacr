"""Item 696 on screen: the diameter fields take 0 and auto, and show the default.

The maintainer could not type "auto" because the field carried a " px"
suffix, could not set 0, and the shown default no longer followed the
magnification. Here:

* the Mask object settings grid shows a blank diameter as the magnification
  default, greyed, and moves it when the magnification moves; 0, "auto" and
  "Auto" are stored as "auto"; anything else is refused with a sentence;
* the Live preview's diameter box has no suffix, takes "auto" in any case,
  opens at the diameter the run will use and hands back what it was given;
* the settings panel stores 0 as "auto".
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid  # noqa: E402


def _mask_settings(magnification=20, **over):
    from spacr.settings import get_timelapse_settings

    settings = get_timelapse_settings()
    for number, obj in enumerate(("cell", "nucleus", "pathogen")):
        settings[f"{obj}_channel"] = number
    settings["magnification"] = magnification
    settings.update(over)
    return settings


@pytest.fixture
def grid(qtbot, qt_theme_applied):
    widget = ObjectSettingsGrid()
    qtbot.addWidget(widget)
    widget.set_settings(_mask_settings())
    return widget


def _shown(grid, obj, role=Qt.DisplayRole):
    model = grid._model
    row = grid.questions().index("diameter")
    column = grid.objects().index(obj)
    return model.data(model.index(row, column), role)


def test_a_blank_diameter_shows_the_magnification_default_greyed(grid):
    assert [_shown(grid, obj) for obj in ("cell", "nucleus", "pathogen")] \
        == ["120", "60", "20"]
    assert _shown(grid, "cell", Qt.ForegroundRole) is not None
    assert _shown(grid, "cell", Qt.EditRole) == ""
    assert grid.settings()["cell_diameter"] is None


def test_the_default_follows_the_magnification(grid):
    assert grid._follow_magnification(40) is True
    assert [_shown(grid, obj) for obj in ("cell", "nucleus", "pathogen")] \
        == ["160", "75", "40"]
    assert grid.settings()["cell_diameter"] is None
    assert grid._follow_magnification(40) is False


@pytest.mark.parametrize("typed", ["0", "auto", "Auto", "AUTO", " 0.0 "])
def test_zero_and_auto_are_stored_and_shown_as_auto(grid, typed):
    assert grid.set_value("diameter", "cell", typed) is True
    assert grid.settings()["cell_diameter"] == "auto"
    assert _shown(grid, "cell") == "auto"
    assert _shown(grid, "cell", Qt.ForegroundRole) is None


def test_zero_and_auto_are_the_same_stored_setting(grid):
    grid.set_value("diameter", "cell", "0")
    after_zero = grid.settings()
    grid.set_value("diameter", "cell", "12")
    grid.set_value("diameter", "cell", "auto")
    assert grid.settings() == after_zero


def test_a_number_is_stored_and_clearing_returns_to_the_default(grid):
    assert grid.set_value("diameter", "nucleus", "45") is True
    assert grid.settings()["nucleus_diameter"] == 45
    assert grid.set_value("diameter", "nucleus", "") is True
    assert grid.settings()["nucleus_diameter"] is None
    assert _shown(grid, "nucleus") == "60"


@pytest.mark.parametrize("typed", ["-3", "thirty"])
def test_anything_else_is_refused_with_a_sentence(grid, typed):
    assert grid.set_value("diameter", "cell", typed) is False
    assert grid.settings()["cell_diameter"] is None
    assert "is not a diameter" in grid.status_text()


def test_the_unit_is_in_the_row_label_not_the_cell(grid):
    model = grid._model
    row = grid.questions().index("diameter")
    assert model.headerData(row, Qt.Vertical, Qt.DisplayRole) == "Diameter (px)"
    tip = _shown(grid, "cell", Qt.ToolTipRole)
    assert "magnification default" in tip and "auto" in tip


def test_a_loaded_auto_survives_the_round_trip(qtbot, qt_theme_applied):
    widget = ObjectSettingsGrid()
    qtbot.addWidget(widget)
    widget.set_settings(_mask_settings(cell_diameter="auto",
                                       nucleus_diameter=33))
    assert _shown(widget, "cell") == "auto"
    out = widget.settings()
    assert out["cell_diameter"] == "auto"
    assert out["nucleus_diameter"] == 33


def test_the_binding_moves_the_shown_default_with_the_form(qtbot,
                                                           qt_theme_applied):
    from PySide6.QtWidgets import QLineEdit

    from spacr.qt.widgets.object_grid_binding import ObjectGridBinding

    class _Panel:
        def __init__(self):
            self.values = _mask_settings()
            self.magnification = QLineEdit("20")
            self._widgets = {"magnification": self.magnification}

        def collect(self):
            return dict(self.values,
                        magnification=int(self.magnification.text() or 0))

        def set_value_for_key(self, key, value):
            self.values[key] = value
            return True

    panel = _Panel()
    widget = ObjectSettingsGrid()
    qtbot.addWidget(widget)
    binding = ObjectGridBinding(widget, panel)
    binding.seed()
    assert _shown(widget, "cell") == "120"
    panel.magnification.setText("40")
    assert _shown(widget, "cell") == "160"
    widget.set_value("diameter", "cell", "0")
    assert panel.values["cell_diameter"] == "auto"


def test_the_settings_panel_stores_zero_as_auto():
    from types import SimpleNamespace

    from spacr.qt.screens.settings_model import SettingsWidgets

    holder = SimpleNamespace(CANONICAL_READERS=SettingsWidgets.CANONICAL_READERS)
    for key in ("cell_diameter", "nucleus_diameter", "pathogen_diameter"):
        for raw, stored in ((0, "auto"), ("Auto", "auto"), (None, None),
                            (55, 55), ("thirty", "thirty")):
            assert SettingsWidgets._canonical(holder, key, raw) == stored


# ---------------------------------------------------------------------------
# The Live preview's box
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    widget = LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize("typed", ["auto", "Auto", "AUTO", "0"])
def test_the_box_takes_auto_typed_in_any_case(panel, qtbot, typed):
    box = panel._diameter
    assert box.suffix() == ""
    box.setValue(30.0)
    box.lineEdit().selectAll()
    qtbot.keyClicks(box, typed)
    qtbot.keyClick(box, Qt.Key_Return)
    assert box.value() == 0.0
    assert box.text() == "auto"


def test_the_live_settings_label_carries_the_unit(panel):
    from PySide6.QtWidgets import QFormLayout

    panel.open_live_settings()
    try:
        dialog = panel._live_settings_dialog
        labels = [form.labelForField(panel._diameter)
                  for form in dialog.findChildren(QFormLayout)
                  if form.labelForField(panel._diameter) is not None]
        assert labels and labels[0].text() == "Diameter (px)"
    finally:
        panel._live_settings_dialog.close()


def test_the_box_opens_at_the_magnification_default(panel):
    panel.apply_settings(_mask_settings(cell_diameter=None))
    assert panel._diameter.value() == 120.0
    assert panel.settings_for_propagation()["cell_diameter"] is None


def test_a_moved_box_propagates_its_number(panel):
    panel.apply_settings(_mask_settings(cell_diameter=None))
    panel._diameter.setValue(90.0)
    assert panel.settings_for_propagation()["cell_diameter"] == 90.0


@pytest.mark.parametrize("stored", ["auto", 0])
def test_auto_opens_as_auto_and_goes_back_as_auto(panel, stored):
    panel.apply_settings(_mask_settings(cell_diameter=stored))
    assert panel._diameter.value() == 0.0
    assert panel._diameter.text() == "auto"
    assert panel.settings_for_propagation()["cell_diameter"] == "auto"


def test_switching_the_object_shows_that_objects_diameter(panel):
    panel.apply_settings(_mask_settings(cell_diameter=None,
                                        nucleus_diameter=None))
    panel._object_box.setCurrentIndex(panel._object_box.findData("nucleus"))
    assert panel._primary_object() == "nucleus"
    assert panel._diameter.value() == 60.0


def test_the_backend_preview_hands_auto_to_the_run_function(monkeypatch):
    """A Cellpose 3 preview at "auto" sends the size model 0, as the run does."""
    from spacr import _segmentation_backends
    from spacr.object import _cellpose3_eval_settings
    from spacr.qt.widgets import live_preview as LP

    seen = {}

    def _masks_of(model, images, settings, obj, **kwargs):
        seen["keywords"] = _cellpose3_eval_settings(
            settings, obj, kwargs["default_diameter"])[0]
        return ([np.zeros((8, 8), np.int32)], [None], [None])

    monkeypatch.setattr(_segmentation_backends, "_load_backend",
                        lambda *a, **k: object())
    request = LP.PreviewRequest(image=np.zeros((8, 8), np.uint16),
                                model="cellpose3:cyto3", diameter=0.0)
    LP._backend_preview_pass(request, "cell", np.zeros((8, 8)),
                             ("cellpose3", _masks_of), lambda i: None)
    assert seen["keywords"]["diameter"] == 0.0


# ---------------------------------------------------------------------------
# The Timelapse preview's box, which reads the same settings
# ---------------------------------------------------------------------------

@pytest.fixture
def timelapse(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    widget = TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_timelapse_box_has_no_suffix_and_opens_at_the_default(timelapse):
    assert timelapse._diameter.suffix() == ""
    timelapse.apply_settings(_mask_settings(cell_diameter=None,
                                            timelapse_objects=["cell"]))
    assert timelapse._diameter.value() == 120.0
    assert timelapse.settings_for_propagation()["cell_diameter"] is None


def test_the_timelapse_box_reads_and_writes_auto(timelapse):
    timelapse.apply_settings(_mask_settings(cell_diameter="auto",
                                            timelapse_objects=["cell"]))
    assert timelapse._diameter.text() == "auto"
    assert timelapse.settings_for_propagation()["cell_diameter"] == "auto"
    timelapse._diameter.setValue(55.0)
    assert timelapse.settings_for_propagation()["cell_diameter"] == 55.0
