"""The three settings 364 retired on 2026-09-19, met the way a user meets them.

The maintainer decided that day: `Toxoplasma` and `barcodes` retire ("Retire
both"), and `img_size` becomes `crop_size`, kept distinct from the model's
`image_size`. The model-layer tests beside this one start from files; these
press the controls a user presses:

* Import settings on the Regression screen, in a real window, with a file
  that turned the annotation OFF with the old switch. The form has no widget
  for the switch, so unless the import folds it the annotation field keeps
  its default and the run annotates what the file had turned off;
* the Cells tab restoring a saved run whose picture settings say `img_size`;
* the picture settings window opened on such a blob;
* the Annotate settings window, whose size field used to carry the MODEL's
  input-size tooltip because it was wired to `image_size`.
"""
from __future__ import annotations

import csv
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _old_regression_file(tmp_path, rows):
    path = tmp_path / "regression_settings.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([("Key", "Value")] + list(rows))
    return path


@pytest.fixture
def regression_window(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    assert window.open_module("regression") == "regression"
    qtbot.wait(20)
    return window


def _import(window, monkeypatch, path):
    """Press Import settings on Regression and choose ``path``."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[1:])))
    window._screens["regression"]._on_import_settings()
    return warnings


@pytest.mark.parametrize("rows, expected", [
    ([("Toxoplasma", "False")], ""),
    ([("toxo", "False")], ""),
    ([("toxo", "True")], "toxoplasma"),
])
def test_an_old_regression_file_decides_the_annotation(
        regression_window, tmp_path, monkeypatch, rows, expected):
    from spacr.ml import _annotation_source
    from spacr.settings import get_perform_regression_default_settings

    screen = regression_window._screens["regression"]
    model = screen._settings_model
    assert "Toxoplasma" not in model._widgets
    assert "annotation_source" in model._widgets
    if expected == "toxoplasma":
        model._widgets["annotation_source"].set_value("human")
    before = dict(model.collect())
    assert _annotation_source(before) != expected, (
        "pick a starting form unlike the file or this test proves nothing")

    warnings = _import(regression_window, monkeypatch,
                       _old_regression_file(tmp_path, rows))

    assert warnings == [], f"the import failed: {warnings}"
    after = dict(regression_window._screens["regression"]
                 ._settings_model.collect())
    assert "Toxoplasma" not in after and "toxo" not in after
    run = get_perform_regression_default_settings(dict(after))
    assert _annotation_source(run) == expected, (
        f"the run would annotate {_annotation_source(run)!r}; the file "
        f"said {rows}")


def test_the_cells_tab_restores_an_old_saved_size(qtbot):
    from spacr.qt.widgets.cell_montage_view import (CellMontageView,
                                                    _thumb_px_of)

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    assert view.apply_workspace_state(
        {"picture_settings": {"img_size": 96, "normalize_channels": "r,g,b"}})
    picture = view.picture_settings()
    assert picture["crop_size"] == 96
    assert "img_size" not in picture
    assert _thumb_px_of(picture) == 96


def test_the_picture_window_opens_an_old_blob_at_its_size(qtbot):
    from PySide6.QtWidgets import QSpinBox

    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog
    from spacr.settings import tooltips

    dialog = PictureSettingsDialog(values={"img_size": 96})
    qtbot.addWidget(dialog)
    editor = dialog._editors["crop_size"]
    assert isinstance(editor, QSpinBox)
    assert editor.value() == 96
    assert "img_size" not in dialog._editors
    assert dialog._labels["crop_size"].text() == "crop size"
    assert dialog.values()["crop_size"] == 96
    help_text = dialog._labels["crop_size"].toolTip()
    assert "How many pixels across each cell is drawn" in (
        help_text or tooltips["crop_size"])


def test_the_annotate_size_field_is_the_crop_size_and_says_so(qtbot, tmp_path):
    """Its tooltip used to be `image_size`'s: "Side length in pixels of the
    center crop taken from each object PNG before model input ... Default
    224", on a field that opens at 200."""
    from PySide6.QtWidgets import QFormLayout

    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _SettingsDialog

    dialog = _SettingsDialog(AnnotateSettings(str(tmp_path)))
    qtbot.addWidget(dialog)
    field = dialog._img_size
    assert field.value() == 200
    text = " ".join(filter(None, (
        field.toolTip(), str(field.property("apiTooltipHtml") or ""))))
    assert "before model input" not in text
    assert "How many pixels across each cell is drawn" in text, text

    form = dialog.findChild(QFormLayout)
    label = form.labelForField(field)
    assert label is not None and label.text() == "Crop size (px)"
