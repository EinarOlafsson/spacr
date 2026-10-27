"""Fresh Measure launches ask before the backend preserves previous output tables."""
import sqlite3

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QWidget

from spacr.qt.screens import app_screen


@pytest.fixture
def previous_output(tmp_path):
    database = tmp_path / 'measurements' / 'measurements.db'
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (object_label INTEGER, cell_area REAL)')
        connection.execute('INSERT INTO cell VALUES (1, 24)')
        connection.execute('CREATE TABLE png_list (png_path TEXT)')
        connection.execute("INSERT INTO png_list VALUES ('old.png')")
    return database


@pytest.mark.parametrize('accept', [False, True])
def test_real_warning_cancel_or_confirm_does_not_archive_in_gui(
        qtbot, monkeypatch, previous_output, accept):
    from spacr import measure

    parent = QWidget()
    qtbot.addWidget(parent)
    before = previous_output.read_bytes()
    settings = {'src': str(previous_output.parent.parent), 'resume': False}
    received = []

    def plan(value):
        received.append(dict(value))
        return [{'db_path': str(previous_output),
                 'tables': {'cell': 'cell_1', 'png_list': 'png_list_1'}}]

    monkeypatch.setattr(measure, '_measurement_run_archive_plan', plan, raising=False)
    inspected = []

    def answer(box):
        box.show()
        assert isinstance(box, QMessageBox)
        assert box.icon() == QMessageBox.Warning
        assert box.defaultButton() is box.button(QMessageBox.Cancel)
        assert 'object labels stay separate' in box.text()
        assert str(previous_output) in box.detailedText()
        assert 'cell → cell_1' in box.detailedText()
        assert 'png_list → png_list_1' in box.detailedText()
        inspected.append(True)
        qtbot.mouseClick(box.button(QMessageBox.Yes if accept else QMessageBox.Cancel), Qt.LeftButton)
        return box.result()

    monkeypatch.setattr(QMessageBox, 'exec', answer)
    assert app_screen.AppScreen._confirm_measure_rerun(parent, settings) is accept
    assert inspected and received == [settings]
    assert previous_output.read_bytes() == before
    with sqlite3.connect(previous_output) as connection:
        assert connection.execute('SELECT * FROM cell').fetchall() == [(1, 24)]
        assert connection.execute('SELECT * FROM png_list').fetchall() == [('old.png',)]
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='cell_1'").fetchall()


@pytest.mark.parametrize('settings', [{}, {'resume': True}, {'dry_run': True}])
def test_empty_plan_does_not_interrupt_run(qtbot, monkeypatch, settings):
    from spacr import measure

    parent = QWidget()
    qtbot.addWidget(parent)
    monkeypatch.setattr(measure, '_measurement_run_archive_plan', lambda value: [], raising=False)
    monkeypatch.setattr(QMessageBox, 'exec', lambda box: pytest.fail('empty plan asked for confirmation'))
    assert app_screen.AppScreen._confirm_measure_rerun(parent, settings)


def test_failed_output_inspection_blocks_launch(qtbot, monkeypatch, previous_output):
    from spacr import measure

    parent = QWidget()
    qtbot.addWidget(parent)
    before = previous_output.read_bytes()

    def refuse(settings):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(measure, '_measurement_run_archive_plan', refuse, raising=False)
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args))
    assert not app_screen.AppScreen._confirm_measure_rerun(parent, {'src': str(previous_output.parent.parent)})
    assert warnings and warnings[0][1:] == ('Bad settings', 'database is locked')
    assert previous_output.read_bytes() == before


def test_real_run_preflight_cancellation_stops_before_launch(qtbot, monkeypatch, previous_output):
    from spacr import measure

    screen = app_screen.AppScreen('measure')
    qtbot.addWidget(screen)
    before = previous_output.read_bytes()
    checks = []
    monkeypatch.setattr(screen, '_wait_for_the_entry_import', lambda override: False)
    monkeypatch.setattr(app_screen, 'resolve_pipeline_entry', lambda key: lambda settings: None)
    monkeypatch.setattr(screen, '_claim_the_project', lambda settings: True)
    monkeypatch.setattr(screen, '_confirm_measure_plane_layout', lambda settings: checks.append('planes') or True)
    monkeypatch.setattr(screen, '_confirm_crop_choices', lambda settings: checks.append('crops') or True)
    monkeypatch.setattr(screen, '_confirm_ram_guard', lambda settings: checks.append('ram') or True)
    monkeypatch.setattr(app_screen, 'make_thread', lambda *args: pytest.fail('cancelled preflight started a worker'))

    def plan(settings):
        checks.append('archive')
        return [{'db_path': str(previous_output), 'tables': {'cell': 'cell_1'}}]

    monkeypatch.setattr(measure, '_measurement_run_archive_plan', plan, raising=False)
    monkeypatch.setattr(QMessageBox, 'exec', lambda box: QMessageBox.Cancel)
    try:
        screen._on_run(override={'src': str(previous_output.parent.parent), 'resume': False})
        assert checks == ['planes', 'crops', 'ram', 'archive']
        assert screen._btn_run.isEnabled()
        assert getattr(screen, '_worker', None) is None
        assert previous_output.read_bytes() == before
    finally:
        screen.close()
