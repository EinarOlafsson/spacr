"""History deletion acts on the chosen runs and preserves project outputs."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt

from spacr import run_journal
from spacr.qt.screens import run_history as history


@pytest.fixture
def dashboard(qtbot, qt_theme_applied, monkeypatch, tmp_path):
    root = tmp_path / 'journal'
    root.mkdir()
    project = tmp_path / 'project'
    project.mkdir()
    output = project / 'measurements.csv'
    output.write_text('cell,area\n1,42\n')
    records = []
    for name in ('first', 'second', 'third'):
        directory = root / name
        directory.mkdir()
        (directory / 'manifest.json').write_text('{}')
        records.append(dict(run_id=name, dir=directory, app_key='measure',
                            status='success', settings={'src': str(project)}))
    monkeypatch.setattr(run_journal, 'runs_root', lambda: root)
    monkeypatch.setattr(run_journal, 'current_run', lambda: None)
    monkeypatch.setattr(history, 'search_runs',
                        lambda: [record for record in records if record['dir'].exists()])
    widget = history.RunHistoryScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget, records, output


def _select(widget, *names):
    widget._table.clearSelection()
    selection = widget._table.selectionModel()
    for row in range(widget._table.rowCount()):
        if widget._table.item(row, 0).data(Qt.UserRole) in names:
            selection.select(widget._table.model().index(row, 0),
                             QItemSelectionModel.Select | QItemSelectionModel.Rows)


@pytest.mark.parametrize('answer', [history.QMessageBox.Cancel, history.QMessageBox.Yes])
def test_confirmation_defaults_to_cancel_and_deletes_only_selected_journals(
        dashboard, monkeypatch, answer):
    widget, records, output = dashboard
    _select(widget, 'first', 'third')
    chosen = widget._selected_records()
    assert {record['run_id'] for record in chosen} == {'first', 'third'}
    prompts = []

    def respond(dialog):
        assert dialog.defaultButton() is dialog.button(history.QMessageBox.Cancel)
        assert 'these 2 runs' in dialog.text()
        assert 'Outputs written into your projects are not touched' in dialog.informativeText()
        prompts.append(dialog.text())
        return answer

    monkeypatch.setattr(history.QMessageBox, 'exec', respond)
    widget._delete_records(chosen)
    assert len(prompts) == 1
    assert records[1]['dir'].is_dir()
    expected_remaining = answer == history.QMessageBox.Cancel
    assert records[0]['dir'].exists() is expected_remaining
    assert records[2]['dir'].exists() is expected_remaining
    assert output.read_text() == 'cell,area\n1,42\n'
    assert widget._table.rowCount() == (3 if expected_remaining else 1)


def test_clear_all_uses_loaded_records_and_keeps_a_new_unseen_run(dashboard, monkeypatch):
    widget, records, output = dashboard
    unseen = records[0]['dir'].parent / 'arrived-after-refresh'
    unseen.mkdir()
    monkeypatch.setattr(history.QMessageBox, 'exec', lambda dialog: history.QMessageBox.Yes)
    widget._delete_every_run()
    assert all(not record['dir'].exists() for record in records)
    assert unseen.is_dir()
    assert output.is_file()


@pytest.mark.parametrize('refusal_count', [1, 4])
def test_partial_deletion_preserves_outside_folders_and_reports_refusals(
        dashboard, monkeypatch, tmp_path, refusal_count):
    widget, records, output = dashboard
    outside = []
    for index in range(refusal_count):
        folder = tmp_path / f'outside-{index}'
        folder.mkdir()
        outside.append(dict(run_id=f'outside-{index}', dir=folder))
    notices = []
    original_status = widget._set_status

    def status(text, **kwargs):
        notices.append(text)
        original_status(text, **kwargs)

    monkeypatch.setattr(widget, '_set_status', status)
    monkeypatch.setattr(history.QMessageBox, 'exec', lambda dialog: history.QMessageBox.Yes)
    widget._delete_records([records[0], *outside])
    assert not records[0]['dir'].exists()
    assert all(record['dir'].is_dir() for record in outside)
    assert output.is_file()
    report = next(text for text in notices if text.startswith('Deleted 1;'))
    assert f'kept {refusal_count}' in report
    assert 'outside-0' in report
    assert report.endswith('…') is (refusal_count > 3)
    if refusal_count > 3:
        assert 'outside-3' not in report


@pytest.mark.parametrize('selection, action', [
    (('second',), 'open'), (('first', 'third'), 'open'),
    (('second',), 'delete'), (('first', 'third'), 'dismiss'),
])
def test_context_menu_acts_on_selected_rows(dashboard, monkeypatch, selection, action):
    widget, records, _output = dashboard
    _select(widget, *selection)
    opened, deleted = [], []
    monkeypatch.setattr(history.QDesktopServices, 'openUrl',
                        lambda url: opened.append(Path(url.toLocalFile())) or True)
    monkeypatch.setattr(widget, '_delete_records', lambda rows: deleted.extend(rows))

    def choose(menu, _position):
        actions = menu.actions()
        count = len(selection)
        assert actions[0].text() == ('Open run folder' if count == 1 else f'Open {count} run folders')
        assert actions[1].text() == ('Delete run…' if count == 1 else f'Delete {count} runs…')
        return actions[0] if action == 'open' else actions[1] if action == 'delete' else None

    menu_class = history.QMenu

    def menu_for_test(parent):
        menu = menu_class(parent)
        return SimpleNamespace(addAction=menu.addAction,
                               exec=lambda position: choose(menu, position))

    monkeypatch.setattr(history, 'QMenu', menu_for_test)
    widget._show_row_menu(QPoint(0, 0))
    assert {path.name for path in opened} == (set(selection) if action == 'open' else set())
    assert {record['run_id'] for record in deleted} == (set(selection) if action == 'delete' else set())
    assert all(record['dir'].exists() for record in records)


def test_empty_selection_opens_no_menu_or_confirmation(dashboard, monkeypatch):
    widget, _records, _output = dashboard
    widget._table.clearSelection()
    monkeypatch.setattr(history.QMenu, 'exec', lambda *_: pytest.fail('unexpected menu'))
    monkeypatch.setattr(history.QMessageBox, 'exec', lambda *_: pytest.fail('unexpected confirmation'))
    assert widget._selected_records() == []
    widget._show_row_menu(QPoint(0, 0))
    widget._delete_records([])
