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


def test_map_barcodes_layout_is_scoped_and_non_mutating():
    """The module's layout replaces the shared buckets; nothing else's does.

    This used to check a two-key relocation into "Sequencing", which left
    thirteen unrelated settings in one drop. `_APP_CATEGORY_SPECS` now names
    all five groups the module has, so `n_jobs` and `test` are placed
    explicitly rather than swept somewhere less wrong. What has not changed
    — and is the load-bearing half — is that the regroup is scoped to this
    one module and does not touch the caller's dict.
    """
    from spacr.qt.screens.settings_model import categories_for_app

    source = {
        "Sequencing": ["mode"],
        "Advanced": ["n_jobs", "verbose"],
        "Model Training": ["test", "epochs"],
    }
    mapped = categories_for_app("map_barcodes", source)
    report = categories_for_app("report", source)

    assert "Sequencing Input" in mapped
    assert "Advanced" not in mapped
    assert "Model Training" not in mapped
    assert mapped["Sequencing Input"] == ["src", "mode", "single_direction"]
    assert mapped["Runtime & Reliability"] == ["chunk_size", "n_jobs", "test"]
    # The layout names every setting the module has, so its groups also name
    # keys this synthetic source does not carry. `build_sections` filters
    # those out at render time; here, what matters is that nothing the
    # source DID carry was dropped.
    placed = {k for keys in mapped.values() for k in keys}
    assert {"mode", "n_jobs", "verbose", "test", "epochs"} <= placed

    assert report == source
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
    assert "Paths" not in names
    assert names == [
        "Sequencing Input", "Barcode References", "Read Parsing",
        "Output & Storage", "Runtime & Reliability",
    ]
    assert {"N jobs", "Test"} <= rows["Runtime & Reliability"]
    assert {"Grna csv", "Row csv", "Column csv"} == rows["Barcode References"]

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


# ---------------------------------------------------------------------------
# One barcode reference is ONE file
# ---------------------------------------------------------------------------
#
# `grna_csv`, `row_csv` and `column_csv` each name a single CSV of
# ``name,sequence`` pairs, and `sequencing.map_sequences_to_names` hands each
# one straight to `pd.read_csv`, which refuses a list outright. The panel gave
# all three the multi-file picker, so opening the screen turned the bundled
# default path into a one-element LIST. That rewrote the user's settings file
# the moment they saved, and `validate` then refused every run from it --
# "column_csv=[...] is a list, but str is expected" -- about a value the user
# had never typed and could not correct from the panel that wrote it.

_ONE_FILE_KEYS = ("grna_csv", "row_csv", "column_csv")


def _barcode_model(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("map_barcodes")
    model.build_sections()
    for widget in model._widgets.values():
        qtbot.addWidget(widget)
    return model


def test_a_barcode_reference_collects_as_the_path_it_was_given(qtbot):
    """Opening the screen and saving must give back what was loaded."""
    from spacr.settings import bundled_barcode_path

    model = _barcode_model(qtbot)
    collected = model.collect()
    for key, kind in zip(_ONE_FILE_KEYS, ("grna", "row", "column")):
        assert collected[key] == bundled_barcode_path(kind), (
            f"opening the screen rewrote {key}")


def test_a_barcode_reference_is_the_type_its_consumer_reads(qtbot):
    """`spacr.settings.expected_types` says `str` for all three, and
    `map_sequences_to_names` passes the value to `pd.read_csv`, which refuses
    a list outright."""
    from spacr.settings import expected_types

    collected = _barcode_model(qtbot).collect()
    for key in _ONE_FILE_KEYS:
        assert expected_types[key] is str, "the declaration moved"
        assert isinstance(collected[key], str), (
            f"{key} reaches pd.read_csv as {type(collected[key]).__name__}")


def test_a_settings_file_that_already_holds_the_list_shape_loads_as_one_path(
        qtbot, tmp_path):
    """The panel wrote lists into real settings files while this was broken.
    Loading one back has to give the single path again rather than carrying
    the wrong shape forward."""
    model = _barcode_model(qtbot)
    csv = tmp_path / "barcodes.csv"
    csv.write_text("name,sequence\nA1,ACGT\n")

    assert model.set_value_for_key("row_csv", [str(csv)]) is True

    assert model.collect()["row_csv"] == str(csv)


def test_choosing_another_barcode_reference_replaces_the_first(qtbot,
                                                               tmp_path):
    """There is one row-barcode CSV per run. A second choice is a CORRECTION,
    and a control that appended left the run reading a file the user thought
    they had replaced."""
    model = _barcode_model(qtbot)
    first, second = tmp_path / "old.csv", tmp_path / "new.csv"
    for path in (first, second):
        path.write_text("name,sequence\nA1,ACGT\n")
    widget = model._widgets["row_csv"]

    widget.add_paths([str(first)])
    widget.add_paths([str(second)])

    assert model.collect()["row_csv"] == str(second)


def test_a_settings_file_holding_the_list_shape_is_still_refused_up_front():
    """The other half of the same defect: files written while the panel was
    wrong are out there. The pre-flight names the key and the shape, so the
    run is refused before it opens a FASTQ rather than inside a worker."""
    from spacr.settings import set_default_generate_barecode_mapping
    from spacr.validate import validate_settings

    settings = set_default_generate_barecode_mapping({})
    settings["src"] = "/tmp"
    settings["column_csv"] = [settings["column_csv"]]

    said = [p for p in validate_settings(settings, "map_barcodes")
            if p.setting == "column_csv"]

    assert len(said) == 1
    assert said[0].severity == "error"
    assert "is a list, but str is expected" in said[0].message
