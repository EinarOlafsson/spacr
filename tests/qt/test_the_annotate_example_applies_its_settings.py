"""The annotate example fills the form in, not just the path.

The dataset ships an `annotate_settings.csv` saying which column holds the
labels, what size the crops are and which channels they carry. A user who has
to work that out for themselves has done most of the work the example was meant
to save.

Reported 2026-09-01: "when annotate example is run the annotate settings should
also be implemented and the local path to the files should be downloaded".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt.annotate_engine import AnnotateSettings
from spacr.qt.screens.annotate import _SettingsDialog


@pytest.fixture
def dialog(qapp, tmp_path):
    made = _SettingsDialog(AnnotateSettings(str(tmp_path)))
    yield made
    made.close()
    made.deleteLater()
    qapp.processEvents()


def _write(tmp_path, rows):
    """Write a settings CSV the way spaCR writes one.

    Through `csv.writer`, not string formatting: a value like `r,g,b` has to
    be QUOTED or the reader sees four columns and takes `r` as the whole
    value. The published files are written this way; a test that was not
    would have "failed" against correct code.
    """
    import csv

    folder = tmp_path / "settings"
    folder.mkdir(exist_ok=True)
    path = folder / "annotate_settings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return path


def test_the_label_column_is_filled_in(dialog, tmp_path):
    path = _write(tmp_path, [("annotation_column", "infected")])
    assert dialog._apply_example_settings(path) == 1
    assert dialog._ann_col.text() == "infected"


def test_the_crop_size_is_filled_in(dialog, tmp_path):
    path = _write(tmp_path, [("image_size", "224")])
    dialog._apply_example_settings(path)
    assert dialog._img_size.value() == 224


def test_the_channels_are_filled_in(dialog, tmp_path):
    path = _write(tmp_path, [("channels", "r,g,b")])
    dialog._apply_example_settings(path)
    assert dialog._channels.text() == "r,g,b"


def test_several_fields_land_together(dialog, tmp_path):
    path = _write(tmp_path, [("annotation_column", "infected"),
                             ("image_size", "128"),
                             ("channels", "r,g,b")])
    assert dialog._apply_example_settings(path) == 3


def test_one_unusable_value_does_not_cost_the_others(dialog, tmp_path):
    """The whole reason a settings file is worth shipping."""
    path = _write(tmp_path, [("image_size", "not a number"),
                             ("annotation_column", "infected")])

    assert dialog._apply_example_settings(path) == 1
    assert dialog._ann_col.text() == "infected"


def test_a_blank_value_is_skipped(dialog, tmp_path):
    dialog._ann_col.setText("kept")
    path = _write(tmp_path, [("annotation_column", ""),
                             ("measurement", "None")])
    assert dialog._apply_example_settings(path) == 0
    assert dialog._ann_col.text() == "kept"


def test_a_missing_file_is_not_an_error(dialog, tmp_path):
    assert dialog._apply_example_settings(tmp_path / "nope.csv") == 0


def test_keys_the_form_does_not_hold_are_ignored(dialog, tmp_path):
    """The published file carries a module's whole settings dict; this form
    holds a fraction of it."""
    path = _write(tmp_path, [("some_unrelated_key", "7"),
                             ("annotation_column", "infected")])
    assert dialog._apply_example_settings(path) == 1


def test_the_example_button_applies_them():
    """A source check: the settings are inert unless the button reads them."""
    source = Path(
        __import__("spacr.qt.screens.annotate", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "self._apply_example_settings(destination / \"settings\"" in source


def test_the_label_column_falls_back_only_when_the_file_is_silent(dialog,
                                                                  tmp_path):
    """A file that names a column must win over the built-in default."""
    path = _write(tmp_path, [("annotation_column", "something_else")])
    dialog._apply_example_settings(path)
    assert dialog._ann_col.text() == "something_else"
