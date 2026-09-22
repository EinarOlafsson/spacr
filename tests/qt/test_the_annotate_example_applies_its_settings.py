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
    path = _write(tmp_path, [("crop_size", "224")])
    dialog._apply_example_settings(path)
    assert dialog._img_size.value() == 224


def test_the_channels_are_filled_in(dialog, tmp_path):
    path = _write(tmp_path, [("channels", "r,g,b")])
    dialog._apply_example_settings(path)
    assert dialog._channels.text() == "r,g,b"


def test_several_fields_land_together(dialog, tmp_path):
    path = _write(tmp_path, [("annotation_column", "infected"),
                             ("crop_size", "128"),
                             ("channels", "r,g,b")])
    assert dialog._apply_example_settings(path) == 3


def test_one_unusable_value_does_not_cost_the_others(dialog, tmp_path):
    """The whole reason a settings file is worth shipping."""
    path = _write(tmp_path, [("crop_size", "not a number"),
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


def test_the_crop_size_is_filled_in_from_the_factory_spelling(dialog, tmp_path):
    """`set_annotate_default_settings` wrote `img_size` until 2026-09-19.

    So a settings CSV made from spaCR's OWN defaults used to set every field
    in this dialog except the crop size. Nothing said so: the form had mostly
    filled itself in, and the one row that had not looked no different from
    a row the file did not carry. Instruction 364's Annotate audit found it.
    The factory writes `crop_size` now, and every file saved before that
    still says `img_size`, so the old spelling is still read.
    """
    # 176, NOT the factory's own 200. `AnnotateSettings.image_size` defaults
    # to (200, 200) and the spin box opens on it, so asserting 200 here would
    # have passed against the broken code by agreeing with the default -- it
    # did, on the first run of this test, which is why the number is odd.
    path = _write(tmp_path, [("img_size", "176")])
    assert dialog._apply_example_settings(path) == 1
    assert dialog._img_size.value() == 176


def test_the_current_spelling_wins_over_the_old_one(dialog, tmp_path):
    """A file carrying both is not ambiguous: `crop_size` is today's name."""
    path = _write(tmp_path, [("img_size", "176"), ("crop_size", "224")])
    assert dialog._apply_example_settings(path) == 1
    assert dialog._img_size.value() == 224


def test_the_models_input_size_is_not_the_crop_size(dialog, tmp_path):
    """`image_size` is the MODEL's input crop, and this field is not it.

    The maintainer's decision of 2026-09-19 renamed `img_size` to
    `crop_size` rather than folding it into `image_size`, because the two
    are different quantities: `image_size` is what training and inference
    feed the network (default 224), and this field is how large each cell
    is drawn (default 200). This screen used to read `image_size` as its
    own spelling of the crop size.
    """
    path = _write(tmp_path, [("image_size", "144")])
    before = dialog._img_size.value()
    assert before != 144, "pick a probe unlike the default"
    assert dialog._apply_example_settings(path) == 0
    assert dialog._img_size.value() == before


def test_the_downloaded_example_draws_its_cells_at_the_crop_size(dialog,
                                                                  tmp_path):
    """The example's `annotate_settings.csv` carries BOTH sizes.

    Its rows, as downloaded and read on 2026-09-19: `image_size,224` and
    `img_size,200`. Before the rename this form took 224, the model's input
    size; the size the file chose for drawing cells is 200.
    """
    path = _write(tmp_path, [("annotation_column", "infected"),
                             ("image_size", "224"), ("img_size", "200")])
    dialog._img_size.setValue(96)
    assert dialog._apply_example_settings(path) == 2
    assert dialog._img_size.value() == 200
