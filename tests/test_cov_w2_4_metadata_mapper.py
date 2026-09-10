"""Metadata column dialog — the browse button, the preview, the round trip.

``tests/qt/test_metadata_mapper.py`` covers the happy answer. What is left
is everything around it, and each of these is a path a user takes on a real
plate table:

* **Browse…**, both when a file is chosen and when the save dialog is
  cancelled -- a cancel that ticked "Save mapping" anyway would write the
  map to the placeholder name;
* the **well preview**, which has to say something useful for a value that
  is not a well at all, and go back to its prompt when the user changes
  their mind and picks "Do not derive";
* :func:`resolve_metadata_with_dialog`, driven through the real resolver
  so the answer given in the dialog is the answer that lands in the frame,
  and so a cancelled dialog is still an explicit stop rather than a run
  that quietly continues without its plate column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QDialog, QFileDialog

from spacr.metadata_resolution import (
    MetadataResolutionRequired,
    build_metadata_request,
)
from spacr.qt.widgets import metadata_mapper
from spacr.qt.widgets.metadata_mapper import (
    MetadataColumnDialog,
    resolve_metadata_with_dialog,
)


@pytest.fixture
def plate_frame():
    """A plate table with the right values under the wrong column names."""
    # Deliberately NOT the canonical spellings: ``plate`` and ``site`` are
    # renamed by ``canonicalise_columns`` before the dialog ever sees them,
    # so a frame using them has nothing left to ask about.
    return pd.DataFrame({
        "experiment_plate": ["p1", "p1", "p2", "p2"],
        "well_name": ["A01", "B12", "not_a_well", ""],
        "site_no": [1, 2, 1, 2],
    })


@pytest.fixture
def dialog(qtbot, plate_frame):
    request = build_metadata_request(
        plate_frame, ["plateID", "rowID", "columnID", "fieldID"])
    widget = MetadataColumnDialog(request)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Browse…
# ---------------------------------------------------------------------------

def test_browsing_to_a_file_fills_the_path_and_ticks_save(dialog, tmp_path,
                                                          monkeypatch):
    """Choosing a file is itself the request to save -- one click, not two."""
    chosen = tmp_path / "column_map.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(chosen), "JSON (*.json)")))

    assert dialog.save_mapping.isChecked() is False
    dialog._browse()

    assert dialog.save_path.text() == str(chosen)
    assert dialog.save_mapping.isChecked() is True
    assert dialog.decision().save_path == str(chosen)


def test_cancelling_the_save_dialog_does_not_arm_the_save(dialog,
                                                          monkeypatch):
    """Otherwise a cancel writes the map to the placeholder filename."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    dialog._browse()

    assert dialog.save_path.text() == ""
    assert dialog.save_mapping.isChecked() is False
    assert dialog.decision().save_path is None


# ---------------------------------------------------------------------------
# Well preview
# ---------------------------------------------------------------------------

def test_the_preview_names_the_values_it_cannot_read(dialog):
    """A column of mostly-wells must show WHICH value will not parse."""
    dialog.well_selector.setCurrentText("well_name")
    text = dialog.well_preview.text()
    assert "A01 → r1/c1" in text
    assert "not_a_well → not recognised" in text


def test_choosing_not_to_derive_puts_the_prompt_back(dialog):
    """Changing your mind must not leave the previous column's preview up."""
    dialog.well_selector.setCurrentText("well_name")
    assert "r1/c1" in dialog.well_preview.text()

    dialog.well_selector.setCurrentIndex(0)
    assert dialog.well_preview.text() == (
        "Choose a well column to preview its mapping")
    assert dialog.decision().well_column is None


def test_a_column_with_no_example_values_says_so(qtbot):
    """An all-empty column previews as a sentence, not as an empty label."""
    frame = pd.DataFrame({"experiment_plate": ["p1", "p2"],
                          "well_name": [np.nan, np.nan]})
    widget = MetadataColumnDialog(
        build_metadata_request(frame, ["plateID", "rowID", "columnID"]))
    qtbot.addWidget(widget)
    widget.well_selector.setCurrentText("well_name")
    assert widget.well_preview.text() == "No non-empty values to preview"


# ---------------------------------------------------------------------------
# The round trip through the real resolver
# ---------------------------------------------------------------------------

def test_an_accepted_dialog_resolves_the_frame(monkeypatch):
    """The answer given in the dialog is the answer that reaches the frame.

    Every well here parses: ``_derive_well_columns`` is all-or-nothing, so a
    single unreadable well would leave rowID/columnID missing and the run
    would stop with a message that never names the offending value.
    """
    frame = pd.DataFrame({
        "experiment_plate": ["p1", "p1", "p2"],
        "well_name": ["A01", "B12", "H06"],
        "site_no": [1, 2, 1],
    })
    def answer(self):
        self._selectors["plateID"].setCurrentText("experiment_plate")
        self._selectors["fieldID"].setCurrentText("site_no")
        # The guesses pre-fill these two with the well column, which the
        # resolver would then rename rather than derive from.
        self._selectors["rowID"].setCurrentText("")
        self._selectors["columnID"].setCurrentText("")
        # ``well_name`` is already ``wellID`` by the time the resolver builds
        # the request -- ``canonicalise_columns`` runs first.
        self.well_selector.setCurrentText("wellID")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(MetadataColumnDialog, "exec", answer)

    result = resolve_metadata_with_dialog(
        frame, ["plateID", "rowID", "columnID", "fieldID"])

    assert result.column_map["plateID"] == "experiment_plate"
    assert result.derived_from_well == "wellID"
    assert {"plateID", "rowID", "columnID", "fieldID"} <= set(
        result.frame.columns)
    assert result.frame.loc[0, "rowID"] == "r1"


def test_a_cancelled_dialog_stops_the_run(plate_frame, monkeypatch):
    """Cancel must raise, not return a frame that is missing its identity."""
    monkeypatch.setattr(MetadataColumnDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)

    with pytest.raises(MetadataResolutionRequired) as caught:
        resolve_metadata_with_dialog(plate_frame, ["plateID", "rowID"])

    assert "plateID" in str(caught.value)


def test_the_dialog_is_built_with_the_parent_it_was_given(plate_frame, qtbot,
                                                          monkeypatch):
    """A modal with no parent lands in the wrong place on a second screen."""
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    seen = []

    original = metadata_mapper.MetadataColumnDialog

    class Recording(original):
        def __init__(self, request, parent=None):
            super().__init__(request, parent)
            seen.append(parent)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(metadata_mapper, "MetadataColumnDialog", Recording)
    with pytest.raises(MetadataResolutionRequired):
        resolve_metadata_with_dialog(plate_frame, ["plateID"], parent=host)

    assert seen == [host]
