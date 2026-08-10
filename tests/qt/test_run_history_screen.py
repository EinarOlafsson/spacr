"""Run History dashboard: search, details, settings hand-off, registration."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from spacr.qt.screens.run_history import (
    APP_INTRO,
    APP_KEY,
    APP_NAME,
    APP_SECTION,
    RunHistoryScreen,
)


@pytest.fixture
def history_root(tmp_path, monkeypatch):
    """One successful classify run with hashing ON, and one failed run.

    ``hash_inputs`` is passed explicitly because input/output hashing became
    OPT-IN on 2026-08-08 (03838b8b, "Input hashing becomes opt-in, and the
    manifest says which it was"): its cost is proportional to the DATA rather
    than to the run, so ``Run.record_input``/``record_output`` are no-ops
    unless the settings ask for it. Without the key this run records no
    digests at all and the "Files & models" pane has nothing to show. The
    now-default OFF path is defended by
    :func:`test_hashes_pane_is_empty_when_hashing_is_off`.
    """
    from spacr import run_journal as journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(journal, "runs_root", lambda: root)
    plate = tmp_path / "plate"
    plate.mkdir()
    (plate / "image.tif").write_bytes(b"pixels")
    output = plate / "scores.csv"
    with journal.open_run(
        "classify",
        {"src": str(plate), "output_path": str(output), "optimizer": "adamw",
         "hash_inputs": True},
    ) as run:
        run.record_warning("class imbalance")
        output.write_text("class,score\nA,0.9\n")
    with pytest.raises(RuntimeError):
        with journal.open_run("measure", {"src": str(plate)}):
            raise RuntimeError("database locked")
    return root, plate, output


@pytest.fixture
def screen(qtbot, qt_theme_applied, history_root):
    widget = RunHistoryScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


def _select_module(screen, module):
    for row in range(screen._table.rowCount()):
        if screen._table.item(row, 1).text() == module:
            screen._table.selectRow(row)
            return
    raise AssertionError(f"{module!r} was not listed")


def test_registration_metadata_matches_app_registry():
    from spacr.qt.app import APPS

    row = next(item for item in APPS if item[0] == APP_KEY)
    assert row[1] == APP_NAME == "Run History"
    assert row[3] == APP_SECTION == "Results & QC"
    assert APP_INTRO


def test_dashboard_lists_performance_and_failure(screen):
    assert screen._table.rowCount() == 2
    _select_module(screen, "classify")
    assert "adamw" in screen._settings.toPlainText()
    hashes = screen._outputs.toPlainText()
    # Both halves of the provenance, not just the output. The run READ
    # image.tif and WROTE scores.csv; a pane showing only one of them would
    # still satisfy the old single assertion while half the record was lost.
    assert "scores.csv" in hashes
    assert "image.tif" in hashes
    assert "class imbalance" in screen._problems.toPlainText()
    assert "output_files" in screen._overview.toPlainText()

    _select_module(screen, "measure")
    assert "database locked" in screen._problems.toPlainText()
    assert "failed" in screen._selection_label.text()


def test_hashes_pane_is_empty_when_hashing_is_off(qtbot, qt_theme_applied,
                                                  tmp_path, monkeypatch):
    """A default run records no digests, and the manifest says which it was.

    Input hashing became opt-in on 2026-08-08 (03838b8b): a run opened
    without ``hash_inputs`` — which is every run nobody configured — writes
    empty input/output hash maps, so the "Files & models" pane is correctly
    empty. That emptiness is a choice, not a lost record, and the manifest is
    what tells the two apart: ``input_hashing: "skipped"`` is written rather
    than the field being silently absent.
    """
    import json

    from spacr import run_journal as journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(journal, "runs_root", lambda: root)
    plate = tmp_path / "plate"
    plate.mkdir()
    (plate / "image.tif").write_bytes(b"pixels")
    output = plate / "scores.csv"
    with journal.open_run(
        "classify", {"src": str(plate), "output_path": str(output)},
    ):
        output.write_text("class,score\nA,0.9\n")

    widget = RunHistoryScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.refresh()
    _select_module(widget, "classify")
    assert json.loads(widget._outputs.toPlainText()) == {
        "inputs": {}, "models": {}, "outputs": {},
    }

    run_dir = next(
        path for path in root.iterdir() if path.name.endswith("__classify")
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["input_hashing"] == "skipped"
    assert manifest["input_hashes"] == {}
    assert manifest["output_hashes"] == {}


def test_search_and_filters_are_combined(screen):
    screen._search.setText("adamw scores.csv")
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 1).text() == "classify"

    screen._status_filter.setCurrentIndex(
        screen._status_filter.findData("failed")
    )
    assert screen._table.rowCount() == 0
    screen._search.clear()
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 1).text() == "measure"


def test_load_settings_emits_exact_module_and_settings(screen, qtbot):
    _select_module(screen, "classify")
    received = []
    screen.settings_requested.connect(
        lambda app_key, settings: received.append((app_key, settings))
    )
    qtbot.mouseClick(screen._load_settings, Qt.LeftButton)
    assert received
    assert received[0][0] == "classify"
    assert received[0][1]["optimizer"] == "adamw"


def test_select_run_and_copy_path(screen, history_root, monkeypatch):
    run_dir = next(
        path for path in history_root[0].iterdir()
        if path.name.endswith("__classify")
    )
    assert screen.select_run(run_dir)
    copied = []

    class Clipboard:
        def setText(self, text):
            copied.append(text)

    monkeypatch.setattr(
        "spacr.qt.screens.run_history.QApplication.clipboard",
        lambda: Clipboard(),
    )
    screen._copy_selected_path()
    assert copied == [str(run_dir)]


def test_empty_history_is_explicit(qtbot, qt_theme_applied, tmp_path,
                                   monkeypatch):
    from spacr import run_journal

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: empty)
    widget = RunHistoryScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.refresh()
    assert widget._table.rowCount() == 0
    assert "0 recorded run" in widget._status.text()
