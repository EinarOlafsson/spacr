"""The metadata column dialog's Browse button and its well preview.

Choosing a file is itself the request to save; cancelling the save dialog
arms nothing; picking "Do not derive" puts the well preview's prompt back.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog                       # noqa: E402

from spacr.metadata_resolution import build_metadata_request    # noqa: E402
from spacr.qt.widgets.metadata_mapper import MetadataColumnDialog  # noqa: E402

pytestmark = pytest.mark.qt

_PROMPT = "Choose a well column to preview its mapping"


def _dialog(qtbot):
    """A dialog over a plate table with the right values under the wrong names."""
    frame = pd.DataFrame({
        "experiment_plate": ["p1", "p1", "p2", "p2"],
        "well_name": ["A01", "B12", "not_a_well", ""],
        "site_no": [1, 2, 1, 2],
    })
    widget = MetadataColumnDialog(build_metadata_request(
        frame, ["plateID", "rowID", "columnID", "fieldID"]))
    qtbot.addWidget(widget)
    return widget


def test_browsing_and_previewing_in_the_metadata_dialog(qtbot, tmp_path,
                                                        monkeypatch, subtests):
    """Save path and save tick after browse or cancel; preview text per selection."""
    chosen = str(tmp_path / "column_map.json")
    for case, answer, path, ticked, saved in (
            ("a chosen file fills the path and ticks save",
             (chosen, "JSON (*.json)"), chosen, True, chosen),
            ("a cancelled save dialog arms nothing", ("", ""), "", False, None),
    ):
        with subtests.test(msg=case):
            dialog = _dialog(qtbot)
            assert dialog.save_mapping.isChecked() is False
            with monkeypatch.context() as patch:
                patch.setattr(QFileDialog, "getSaveFileName",
                              staticmethod(lambda *a, **k: answer))
                dialog._browse()
            assert (dialog.save_path.text(), dialog.save_mapping.isChecked(),
                    dialog.decision().save_path) == (path, ticked, saved)

    with subtests.test(msg="choosing not to derive puts the prompt back"):
        dialog = _dialog(qtbot)
        assert dialog.well_preview.text() == _PROMPT
        dialog.well_selector.setCurrentText("well_name")
        assert dialog.well_preview.text() == (
            "A01 → r1/c1, B12 → r2/c12, not_a_well → not recognised")
        assert dialog.decision().well_column == "well_name"
        dialog.well_selector.setCurrentIndex(0)
        assert dialog.well_preview.text() == _PROMPT
        assert dialog.decision().well_column is None
