"""Qt input validation keeps its exception types and offers a concrete next step."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QFileDialog, QMessageBox

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def english_messages(monkeypatch):
    """Render recovery instructions in English without changing preferences."""
    from spacr.qt import i18n

    monkeypatch.setattr(i18n, "current_language", Mock(return_value="en"))


@pytest.mark.parametrize("problem, advice", [
    ("unrelated", "JSON file created with Share template"),
    ("newer", "open it with the spaCR version that saved it"),
    ("missing_settings", "Save and share the template again"),
    ("invalid_json", "export the template again from its original module"),
])
def test_template_refusal_explains_replacement(tmp_path, problem, advice):
    """Invalid template inputs remain ValueErrors and explain where to get a valid file."""
    from spacr.qt import recipes

    data = {
        "unrelated": {},
        "newer": {"spacr_recipe": recipes.FORMAT_VERSION + 1, "settings": {}},
        "missing_settings": {"spacr_recipe": recipes.FORMAT_VERSION},
    }
    path = tmp_path / "broken.json"
    path.write_text("{unfinished", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        if problem == "invalid_json":
            recipes.load_recipe(str(path))
        else:
            recipes.Recipe.from_json(data[problem])
    assert advice in str(caught.value)
    if problem == "invalid_json":
        assert "broken.json is not valid JSON" in str(caught.value)
    assert path.read_text(encoding="utf-8") == "{unfinished"


@pytest.mark.parametrize("existing_folder", [False, True])
def test_missing_database_explains_source_selection(tmp_path, existing_folder):
    """Missing paths remain FileNotFoundErrors without creating a database."""
    from spacr.qt.screens.db_browser import resolve_db_path

    path = tmp_path / "run"
    if existing_folder:
        path.mkdir()
    with pytest.raises(FileNotFoundError) as caught:
        resolve_db_path(path)
    message = str(caught.value)
    assert str(path) in message
    if existing_folder:
        assert "Choose a measurements.db file from a completed run" in message
    else:
        assert "choose an existing measurements.db file" in message
    assert not list(tmp_path.rglob("*.db"))


@pytest.mark.parametrize("value, declared, advice", [
    ("1.5", "INTEGER", "whole number without a decimal point"),
    ("1,5", "REAL", "using a dot as the decimal separator"),
    ("text", "BLOB", "database editor that supports binary fields"),
])
def test_cell_edit_refusal_explains_valid_input(value, declared, advice):
    """Rejected cell values retain the column and type alongside suitable input advice."""
    from spacr.qt.screens.db_browser import coerce_for_column

    with pytest.raises(ValueError) as caught:
        coerce_for_column(value, declared, "measurement")
    message = str(caught.value)
    assert "measurement" in message and declared in message
    assert advice in message


@pytest.mark.parametrize("existing_destination", [False, True])
def test_paired_copy_refusal_preserves_sources(tmp_path, existing_destination):
    """Pairing and destination errors keep the sources and any existing destination intact."""
    from spacr.qt.screens.make_masks import _copy_paired_layout
    from spacr.tiff_io import write_tiff

    image = tmp_path / "field.tif"
    mask = tmp_path / "field_mask.tif"
    pixels = np.ones((3, 4), dtype=np.uint16)
    write_tiff(image, pixels)
    write_tiff(mask, pixels)
    original = {path: path.read_bytes() for path in (image, mask)}
    destination = tmp_path / "output"
    pairs = []
    error_type = ValueError
    advice = "Select one mask per image with a matching field name"
    if existing_destination:
        destination.mkdir()
        (destination / "keep.txt").write_text("keep", encoding="utf-8")
        pairs = [(image, mask)]
        error_type = FileExistsError
        advice = "Choose a new folder name that does not already exist"
    with pytest.raises(error_type) as caught:
        _copy_paired_layout(pairs, destination)
    assert advice in str(caught.value)
    assert all(path.read_bytes() == data for path, data in original.items())
    assert not list(tmp_path.glob(".spacr-paired-*"))
    if existing_destination:
        assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"
    else:
        assert not destination.exists()


def test_missing_canvas_source_explains_reload_without_mutating_box():
    """A detached canvas image remains a ValueError with its label and reload instructions."""
    from spacr.qt.widgets.picture_export import _canvas_box_pixels

    box = {"label": "Field A", "source": {"type": "field"}}
    with pytest.raises(ValueError) as caught:
        _canvas_box_pixels(box, None, None)
    message = str(caught.value)
    assert "Field A" in message
    assert "Reload the source image in Overview" in message
    assert box == {"label": "Field A", "source": {"type": "field"}}


def test_invalid_graph_style_explains_the_save_action(tmp_path):
    """A foreign JSON file is refused without alteration and names the correct export action."""
    from spacr.qt.widgets.figure_settings import load_graph_style

    path = tmp_path / "unrelated.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_graph_style(path)
    message = str(caught.value)
    assert str(path) in message
    assert "JSON file created with Save graph style" in message
    assert path.read_text(encoding="utf-8") == "{}"


def test_pipeline_export_failure_explains_destination_without_changing_setting(
        tmp_path, monkeypatch):
    """A failed CellProfiler export keeps the current setting and reports a writable-folder action."""
    from spacr.qt.screens import settings_model

    path = tmp_path / "example.cppipe"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", Mock(return_value=(str(path), "")))
    monkeypatch.setattr(settings_model, "_write_cellprofiler_example", Mock(side_effect=OSError("disk full")))
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    edit = SimpleNamespace(setText=Mock(), editingFinished=SimpleNamespace(emit=Mock()))

    assert settings_model._export_cellprofiler_example(edit) is False
    message = warning.call_args.args[2]
    assert "disk full" in message
    assert "existing folder you can write to" in message
    assert "free disk space" in message
    edit.setText.assert_not_called()
    edit.editingFinished.emit.assert_not_called()
    assert not path.exists()
