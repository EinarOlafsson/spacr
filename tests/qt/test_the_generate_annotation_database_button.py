"""The Annotation app can build its own set to annotate.

Annotating needs crops, and the only way to get them was a full Measure run.
The objects are already described twice on disk once Measure has run, so a
second set can be built in seconds. Instruction 338.

THE FORM SAYS WHAT THE SOURCES CAN DO. The database stores coordinates and no
masks, so it can only cut a bounding box; the arrays carry the masks and can
cut to the object. Leaving that to the manual would let a user pick the
database, ask for a masked crop, and quietly get a rectangle.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt.annotate_engine import AnnotateSettings
from spacr.qt.screens.annotate import _GenerateAnnotationDatabaseDialog


@pytest.fixture
def dialog(qapp, tmp_path):
    made = _GenerateAnnotationDatabaseDialog(AnnotateSettings(str(tmp_path)))
    yield made
    made.close()
    made.deleteLater()
    qapp.processEvents()


def test_both_sources_are_offered(dialog):
    values = [dialog._source.itemData(i)
              for i in range(dialog._source.count())]
    assert values == ["array", "database"]


def test_the_database_source_forces_a_bounding_box(dialog):
    """It has no mask to cut to, so offering the choice would be offering
    something it cannot honour."""
    dialog._source.setCurrentIndex(1)
    assert dialog._bounding_box.isChecked()
    assert not dialog._bounding_box.isEnabled()


def test_the_array_source_leaves_the_choice_open(dialog):
    dialog._source.setCurrentIndex(0)
    assert dialog._bounding_box.isEnabled()


def test_the_difference_is_written_on_the_form(dialog):
    """Not guessable, so not left to the manual."""
    dialog._source.setCurrentIndex(1)
    assert "coordinates" in dialog._note.text()
    dialog._source.setCurrentIndex(0)
    assert "masks" in dialog._note.text()


def test_the_filters_reach_the_generator(dialog):
    dialog._object.setCurrentText("nucleus")
    dialog._min_size.setValue(10)
    dialog._max_size.setValue(900)
    dialog._max_objects.setValue(5)

    collected = dialog._collected()

    assert collected["object_array"] == "nucleus"
    assert collected["nucleus_min_size"] == 10
    assert collected["nucleus_max_size"] == 900
    assert collected["max_objects"] == 5


def test_the_channels_are_parsed(dialog):
    dialog._channels.setText("3, 1, 0")
    assert dialog._collected()["channel_arrays"] == [3, 1, 0]


def test_empty_channels_fall_back_rather_than_crashing(dialog):
    dialog._channels.setText("")
    assert dialog._collected()["channel_arrays"] == [0, 1, 2]


def test_a_run_that_writes_nothing_says_so(dialog, monkeypatch):
    """A generator that writes nothing and closes looks exactly like one that
    worked."""
    monkeypatch.setattr(
        "spacr.annotation_dataset.generate_annotation_dataset",
        lambda settings: {"written": 0, "fields": 0, "table": "",
                          "trouble": ["every object was filtered out"]})

    dialog._on_generate()

    assert "Nothing was written" in dialog._status.text()
    assert "filtered out" in dialog._status.text()
    assert dialog.written_table() == ""


def test_a_failure_is_reported_rather_than_raised(dialog, monkeypatch):
    def _explode(settings):
        raise RuntimeError("the merged folder is not there")

    monkeypatch.setattr(
        "spacr.annotation_dataset.generate_annotation_dataset", _explode)

    dialog._on_generate()

    assert "not there" in dialog._status.text()
    assert dialog._generate.isEnabled(), "the button must come back"


def test_a_successful_run_reports_the_table(dialog, monkeypatch):
    monkeypatch.setattr(
        "spacr.annotation_dataset.generate_annotation_dataset",
        lambda settings: {"written": 12, "fields": 3, "table": "png_list_2",
                          "trouble": []})

    dialog._on_generate()

    assert dialog.written_table() == "png_list_2"


def test_the_button_exists_on_the_screen():
    source = Path(
        __import__("spacr.qt.screens.annotate", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert 'QPushButton("Generate annotation database…")' in source
    assert "self._btn_generate.clicked.connect(" in source


def test_a_second_table_is_reported_as_not_yet_openable():
    """This screen reads png_list and only png_list. Silently showing the OLD
    set while a new one sits unopened would look like the generator had done
    nothing."""
    source = Path(
        __import__("spacr.qt.screens.annotate", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "This screen currently opens" in source
