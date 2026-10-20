"""The 633 example loaders (dose, Control Chart, Investigate Hit) on their
failure paths: a download that leaves nothing behind, a screen that cannot
open what arrived, a record that is not JSON, and a button placed at a
given position or kept out of any layout.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from spacr.qt.widgets import measurements_example as mx

pytestmark = pytest.mark.qt


@pytest.fixture
def examples(tmp_path, monkeypatch):
    """Example folders under ``tmp_path``, never the user's cache."""
    plate = tmp_path / "examples" / "measure"
    monkeypatch.setattr(mx, "example_measurements_folder", lambda: plate)
    return plate.parent


def _screen(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    said = []
    screen._test_data_say = said.append
    return screen, said


def _row(screen):
    row = QHBoxLayout(screen)
    row.addWidget(QLabel("first", screen))
    row.addWidget(QLabel("last", screen))
    return row


def test_a_dose_download_that_leaves_no_plate_is_reported(qtbot, examples):
    screen, said = _screen(qtbot)
    asked = []

    assert mx._load_dose_test_data(screen, ask=asked.append) is False

    assert asked == [mx._dose_example_folder()]
    assert said and str(mx._dose_example_folder() / mx._DOSE_PLATE) in said[0]


def test_a_failing_explanation_falls_back_to_the_error_text(monkeypatch):
    from spacr import example_archives

    def broken(_exc):
        raise RuntimeError("explainer broke")

    monkeypatch.setattr(example_archives, "explain_download_failure", broken)

    assert "socket closed" in mx._download_failure(OSError("socket closed"))
    assert "ValueError" in mx._download_failure(ValueError())


@pytest.mark.parametrize("install", [
    mx._install_dose_test_data_button,
    mx._install_control_chart_test_data_button,
    mx._install_hit_test_data_button,
])
def test_a_test_data_button_goes_where_it_is_asked(qtbot, install):
    screen, _said = _screen(qtbot)
    row = _row(screen)

    placed = install(screen, row, lambda *args: None, index=1)
    appended = install(screen, row, lambda *args: None)
    loose = install(screen, None, lambda *args: None)

    assert row.indexOf(placed) == 1
    assert row.indexOf(appended) == row.count() - 1
    assert row.indexOf(loose) == -1
    assert placed.text() == loose.text()


def test_control_chart_download_without_a_table_is_reported(qtbot, examples):
    screen, said = _screen(qtbot)
    mx._install_control_chart_test_data_button(screen, None,
                                               lambda table: None,
                                               say=said.append)

    assert mx._load_control_chart_test_data(screen, ask=lambda folder: None) \
        is False
    assert mx._CONTROL_CHART_TABLE in said[0]


def _write_table(folder):
    folder.mkdir(parents=True, exist_ok=True)
    table = folder / mx._CONTROL_CHART_TABLE
    table.write_text("plate,well\nP1,A01\n", encoding="utf-8")
    return table


def test_control_chart_table_reaches_the_screen_or_its_error_is_said(
        qtbot, examples):
    screen, said = _screen(qtbot)
    table = _write_table(mx._control_chart_example_folder())

    assert mx._load_control_chart_test_data(screen) is True
    received = []
    screen._control_chart_test_data_apply = received.append
    assert mx._load_control_chart_test_data(screen) is True
    assert received == [table]

    def refuse(_table):
        raise ValueError("no DMSO wells")

    screen._control_chart_test_data_apply = refuse
    assert mx._load_control_chart_test_data(screen) is False
    assert said == ["no DMSO wells"]


def test_a_hit_download_without_a_record_is_reported(qtbot, examples):
    screen, said = _screen(qtbot)

    assert mx._load_hit_test_data(screen, ask=lambda folder: None) is False
    assert mx._HIT_RECORD in said[0]


def test_the_hit_record_reaches_the_screen_or_its_error_is_said(
        qtbot, examples):
    screen, said = _screen(qtbot)
    folder = mx._hit_example_folder()
    folder.mkdir(parents=True)
    record = folder / mx._HIT_RECORD
    record.write_text(json.dumps({"gene": "239740"}), encoding="utf-8")

    assert mx._load_hit_test_data(screen) is True
    received = []
    screen._hit_test_data_apply = lambda where, hit: received.append(
        (where, hit))
    assert mx._load_hit_test_data(screen) is True
    assert received == [(folder, {"gene": "239740"})]

    record.write_text("{not json", encoding="utf-8")
    assert mx._load_hit_test_data(screen) is False
    assert said and "Expecting" in said[0]
    assert len(received) == 1


def test_the_dose_folder_reaches_the_screen_or_its_error_is_said(
        qtbot, examples):
    screen, said = _screen(qtbot)
    folder = mx._dose_example_folder()
    folder.mkdir(parents=True)
    (folder / mx._DOSE_PLATE).write_text("plate,well\n", encoding="utf-8")

    assert mx._load_dose_test_data(screen) is True
    received = []
    screen._dose_test_data_apply = received.append
    assert mx._load_dose_test_data(screen) is True
    assert received == [folder]

    def refuse(_folder):
        raise KeyError()

    screen._dose_test_data_apply = refuse
    assert mx._load_dose_test_data(screen) is False
    assert said == ["KeyError"]


def test_the_control_chart_and_hit_downloads_name_their_example_sets(
        tmp_path, monkeypatch):
    from spacr import example_archives

    fetched, absolute = [], []
    monkeypatch.setattr(mx, "_fetch_example_set",
                        lambda key, folder: fetched.append((key, folder)))
    monkeypatch.setattr(example_archives, "make_the_example_paths_absolute",
                        absolute.append)

    mx._fetch_control_chart_example(tmp_path / "cc")
    mx._fetch_hit_example(tmp_path / "hit")

    assert fetched == [(mx._CONTROL_CHART_KEY, tmp_path / "cc"),
                       (mx._HIT_KEY, tmp_path / "hit")]
    assert absolute == [tmp_path / "hit"]


def test_a_journal_mark_whose_runs_are_gone_is_written_again(
        examples, tmp_path, monkeypatch):
    from spacr import run_journal

    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: journal)
    folder = mx._dose_example_folder()
    folder.mkdir(parents=True)
    (folder / mx._DOSE_JOURNAL_MARK).write_text("\n", encoding="utf-8")

    assert mx._journal_dose_runs(folder) == 0
    assert (folder / mx._DOSE_JOURNAL_MARK).read_text(encoding="utf-8") == "\n"
