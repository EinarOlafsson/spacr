"""The ingestion grid has to keep working on a guess that came back wrong.

Its rows are INFERRED metadata -- from a container file's internal
structure, or from a folder layout -- which is exactly why the user is being
shown them. So every cell can hold something that is not what the column
wants: a field index that is a word, a well with no plate prefix. None of
that may raise while the user is typing, and the canonical filename must
still be a name the pipeline can consume.

The dialog around the grid carries the other half: Apply writes a CSV, and a
write that fails must leave ``written_path`` empty rather than reporting a
file that is not there.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ingest_preview import ROW_COLUMNS         # noqa: E402
from spacr.qt.widgets.metadata_table import (           # noqa: E402
    MetadataTableDialog, MetadataTablePanel,
)

pytestmark = pytest.mark.qt


def _row(**over):
    row = {"original": "img_001.czi", "plate": "plate1", "well": "plate1_A01",
           "field": 1, "channel": 1, "time": 1, "canonical": ""}
    row.update(over)
    return row


def _col(key):
    return ROW_COLUMNS.index(key)


def test_a_field_index_that_is_not_a_number_reads_back_as_one(qtbot):
    """An unparseable inferred index must not propagate out of the grid."""
    panel = MetadataTablePanel([_row(field="abc", channel="", time=None)])
    qtbot.addWidget(panel)

    read_back = panel.rows()

    assert read_back[0]["field"] == 1
    assert read_back[0]["channel"] == 1
    assert read_back[0]["time"] == 1


def test_a_well_without_its_plate_prefix_gets_one(qtbot):
    """The canonical name has to match what convert_to_yokogawa produces."""
    panel = MetadataTablePanel([_row(well="plate1_A01", field="abc")])
    qtbot.addWidget(panel)

    panel._table.item(0, _col("well")).setText("A01")

    canonical = panel._table.item(0, _col("canonical")).text()
    assert canonical.startswith("plate1_A01_")
    assert canonical.endswith(".tif")


def test_editing_a_read_only_cell_does_not_rewrite_the_filename(qtbot):
    """Source and Filename are outputs; a change there is not an edit."""
    panel = MetadataTablePanel([_row()])
    qtbot.addWidget(panel)
    panel._table.item(0, _col("well")).setText("plate1_B02")
    before = panel._table.item(0, _col("canonical")).text()

    panel._table.item(0, _col("original")).setText("renamed_on_disk.czi")

    assert panel._table.item(0, _col("canonical")).text() == before
    assert panel.rows()[0]["original"] == "renamed_on_disk.czi"


def test_the_panel_is_built_even_when_the_theme_cannot_be_read(qtbot,
                                                               monkeypatch):
    """A palette that cannot be resolved costs a colour, not the widget."""
    import spacr.qt.theme as theme

    def boom():
        raise RuntimeError("no palette in this context")

    monkeypatch.setattr(theme, "active_palette", boom)

    panel = MetadataTablePanel([_row()])
    qtbot.addWidget(panel)

    assert panel._summary.styleSheet() == ""
    assert panel.rows()[0]["plate"] == "plate1"


def test_a_failed_write_leaves_no_written_path(qtbot, tmp_path, monkeypatch):
    """Apply must not report a filename_map.csv that was never written."""
    import spacr.qt.folder_metadata as folder_metadata

    def boom(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(folder_metadata, "save_filename_map", boom)
    called = []
    dialog = MetadataTableDialog([_row()], tmp_path / "map.csv",
                                 on_apply=called.append)
    qtbot.addWidget(dialog)

    dialog._apply()

    assert dialog.written_path is None
    assert called == []


def test_a_successful_write_reports_the_path_it_wrote(qtbot, tmp_path):
    """The success half, so the failure assertion above means something."""
    called = []
    dialog = MetadataTableDialog([_row()], tmp_path / "map.csv",
                                 on_apply=called.append)
    qtbot.addWidget(dialog)

    dialog._apply()

    assert dialog.written_path is not None
    assert dialog.written_path.is_file()
    assert called == [dialog.written_path]
