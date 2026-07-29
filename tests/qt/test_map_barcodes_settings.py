"""Map Barcodes settings UI and bundled-reference regression tests."""
from __future__ import annotations

import csv
import os

from PySide6.QtWidgets import QComboBox, QDialogButtonBox


def test_bundled_barcode_path_resolves_each_packaged_reference():
    from spacr.settings import bundled_barcode_path

    expected = {
        "column": "barcodes_column.csv",
        "grna": "barcodes_grna.csv",
        "row": "barcodes_row.csv",
    }
    for kind, filename in expected.items():
        path = bundled_barcode_path(kind)
        assert os.path.isabs(path)
        assert os.path.basename(path) == filename
        assert os.path.isfile(path)
        with open(path, newline="", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == ["name", "sequence"]


def test_bundled_barcode_path_rejects_unknown_reference_type():
    import pytest

    from spacr.settings import bundled_barcode_path

    with pytest.raises(ValueError, match="Unknown barcode reference"):
        bundled_barcode_path("plate")


def test_map_barcodes_defaults_use_the_bundled_csv_files():
    from spacr.settings import (
        bundled_barcode_path,
        set_default_generate_barecode_mapping,
    )

    settings = set_default_generate_barecode_mapping({})
    assert settings["column_csv"] == bundled_barcode_path("column")
    assert settings["grna_csv"] == bundled_barcode_path("grna")
    assert settings["row_csv"] == bundled_barcode_path("row")


def test_map_barcodes_category_relocation_is_scoped_and_non_mutating():
    from spacr.qt.screens.settings_model import categories_for_app

    source = {
        "Sequencing": ["mode"],
        "Advanced": ["n_jobs", "verbose"],
        "Model Training": ["test", "epochs"],
    }
    mapped = categories_for_app("map_barcodes", source)
    classify = categories_for_app("classify", source)

    assert mapped["Sequencing"] == ["mode", "n_jobs", "test"]
    assert mapped["Advanced"] == ["verbose"]
    assert mapped["Model Training"] == ["epochs"]
    assert classify == source
    assert source["Advanced"] == ["n_jobs", "verbose"]
    assert source["Model Training"] == ["test", "epochs"]


def test_map_barcodes_ui_uses_sequencing_dropdowns_and_no_stray_tabs(
    qtbot,
):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("map_barcodes")
    sections = model.build_sections()
    names = [name for name, _rows in sections]
    rows = {
        section: {label for label, _widget in section_rows}
        for section, section_rows in sections
    }

    assert "Advanced" not in names
    assert "Model Training" not in names
    assert {"N jobs", "Test"} <= rows["Sequencing"]

    expected = {
        "mode": ["paired", "single"],
        "single_direction": ["R1", "R2"],
        "comp_type": ["zlib", "lzo", "bzip2", "blosc"],
    }
    for key, values in expected.items():
        widget = model._widgets[key]
        qtbot.addWidget(widget)
        assert isinstance(widget, QComboBox)
        assert [
            widget.itemData(index) for index in range(widget.count())
        ] == values

    assert model.collect()["mode"] == "paired"
    assert model.collect()["single_direction"] == "R1"
    assert model.collect()["comp_type"] == "zlib"


def test_barcode_regex_validation_reports_syntax_groups_and_captures():
    from spacr.qt.widgets.barcode_regex import (
        EXAMPLE_BARCODE_READ,
        evaluate_barcode_regex,
    )
    from spacr.settings import DEFAULT_BARCODE_REGEX

    syntax = evaluate_barcode_regex("(?P<columnID>[")
    assert syntax.valid is False
    assert "Regex error" in syntax.message

    groups = evaluate_barcode_regex(r"(?P<columnID>.{8})")
    assert groups.valid is False
    assert "grna" in groups.message and "rowID" in groups.message

    match = evaluate_barcode_regex(
        DEFAULT_BARCODE_REGEX,
        EXAMPLE_BARCODE_READ,
    )
    assert match.valid is True
    assert match.captures == {
        "columnID": "TCATAGGC",
        "grna": "GTACTATAATGATATTACGAC",
        "rowID": "CAATGTCG",
    }


def test_barcode_regex_dialog_tests_example_and_saves(qtbot):
    from spacr.qt.widgets.barcode_regex import BarcodeRegexDialog
    from spacr.settings import DEFAULT_BARCODE_REGEX

    dialog = BarcodeRegexDialog(DEFAULT_BARCODE_REGEX)
    qtbot.addWidget(dialog)
    dialog._example_button.click()

    assert "matched successfully" in dialog._status.text()
    assert "TCATAGGC" in dialog._captures.toPlainText()
    save = dialog._buttons.button(QDialogButtonBox.Save)
    assert save.isEnabled()
    save.click()
    assert dialog.regex == DEFAULT_BARCODE_REGEX


def test_settings_model_round_trips_interactive_barcode_regex(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets
    from spacr.qt.widgets.barcode_regex import BarcodeRegexWidget

    model = SettingsWidgets("map_barcodes")
    model.build_sections()
    widget = model._widgets["regex"]
    qtbot.addWidget(widget)

    assert isinstance(widget, BarcodeRegexWidget)
    pattern = r"^(?P<columnID>A)(?P<grna>B)(?P<rowID>C)$"
    widget.set_value(pattern)
    assert model.collect()["regex"] == pattern
    assert widget._status.text() == "✓"
