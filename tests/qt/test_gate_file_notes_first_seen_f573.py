"""Save/Load notes share first-seen and integrity policy with live edits/exports."""
import json
from pathlib import Path

from spacr import run_journal as rj
from spacr.qt.screens.gate_editor import GateEditorScreen
from tests.qt.test_gate_export_analysis_lock import database, gates, journal  # noqa: F401


def test_file_note_observes_edit_before_unblinding_and_keeps_it_a_deviation(journal):  # noqa: F811
    folder, now = journal
    strategy = folder / 'gates.json'
    gates().save(str(strategy))
    key = rj.start_blinding(['field1'], scope='annotate', src=str(folder))
    record = rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=str(strategy))
    now['time'] = '2026-10-01T10:01:00.000000Z'
    gates(200).save(str(strategy))
    assert 'a deviation' in rj._gate_file_lock_notes(strategy)[0]
    events = [json.loads(line) for line in rj._seen_log(record).read_text().splitlines()]
    assert len(events) == 1 and events[0]['utc'] == now['time']
    now['time'] = '2026-10-01T10:02:00.000000Z'
    rj.unblind(key['key_id'], reason='finished')
    now['time'] = '2026-10-01T10:03:00.000000Z'
    assert 'a deviation' in rj._gate_file_lock_notes(strategy)[0]
    assert 'post-hoc' not in rj._gate_file_lock_notes(strategy)[0]
    gates(250).save(str(strategy))
    assert 'post-hoc' in rj._gate_file_lock_notes(strategy)[0]


def test_matching_file_never_claims_match_to_a_tampered_lock(journal):  # noqa: F811
    folder, _now = journal
    strategy = folder / 'gates.json'
    gates().save(str(strategy))
    record = rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=str(strategy))
    path = rj._locks_root() / (record['lock_id'] + '.json')
    altered = json.loads(path.read_text())
    altered['locked_by'] = 'edited author'
    path.write_text(json.dumps(altered))
    note = rj._gate_file_lock_notes(strategy)[0]
    assert 'tampered' in note.lower()
    assert 'these gates match' not in note


def test_matching_file_locked_after_unblinding_is_not_preregistered(journal):  # noqa: F811
    folder, now = journal
    strategy = folder / 'gates.json'
    gates().save(str(strategy))
    key = rj.start_blinding(['field1'], scope='annotate', src=str(folder))
    now['time'] = '2026-10-01T10:01:00.000000Z'
    rj.unblind(key['key_id'], reason='finished')
    now['time'] = '2026-10-01T10:02:00.000000Z'
    rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=str(strategy))
    note = rj._gate_file_lock_notes(strategy)[0]
    assert 'not preregistered' in note.lower()
    assert 'these gates match' not in note


def test_file_notes_do_not_open_unrelated_locked_strategy(journal, monkeypatch):  # noqa: F811
    folder, _now = journal
    first, other = folder / 'first.json', folder / 'remote.json'
    gates().save(str(first))
    gates(300).save(str(other))
    rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=[str(first), str(other)])
    original = rj._gate_payload
    def checked(value):
        if isinstance(value, (str, Path)):
            assert str(value) != str(other)
        return original(value)
    monkeypatch.setattr(rj, '_gate_payload', checked)
    assert 'a deviation' in rj._gate_file_lock_notes(first, gates(200))[0]


def test_editor_save_and_reload_keep_pre_unblind_live_edit_status(qtbot, journal):  # noqa: F811
    folder, now = journal
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(database(folder), 'cell')
    strategy = folder / 'strategy.json'
    screen.gates.set_gates(gates())
    screen.save_gates(str(strategy))
    key = rj.start_blinding(['field1'], scope='annotate', src=str(folder))
    rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=str(strategy))
    now['time'] = '2026-10-01T10:01:00.000000Z'
    screen.gates.set_gates(gates(200))
    now['time'] = '2026-10-01T10:02:00.000000Z'
    rj.unblind(key['key_id'], reason='finished')
    now['time'] = '2026-10-01T10:03:00.000000Z'
    screen.save_gates(str(strategy))
    assert 'a deviation' in screen._source.text()
    assert 'post-hoc' not in screen._source.text()
    assert screen.load_gates(str(strategy))
    assert 'a deviation' in screen._source.text()
    assert 'post-hoc' not in screen._source.text()
