"""Committed live gate edits are observed before a later unblinding or export."""
import json
from pathlib import Path

from spacr import run_journal as rj
from spacr.qt.screens import gate_editor as screen_module
from spacr.qt.screens.gate_editor import GateEditorScreen, _gate_export_lock_verdicts
from tests.qt.test_gate_export_analysis_lock import database, gates, journal  # noqa: F401


def make_screen(qtbot, folder):
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(database(folder), 'cell')
    return screen


def test_threshold_edit_before_unblinding_keeps_original_first_seen_at_export(qtbot, journal):  # noqa: F811
    folder, now = journal
    screen = make_screen(qtbot, folder)
    strategy = folder / 'gates.json'
    screen.gates.set_gates(gates())
    screen.save_gates(str(strategy))
    key = rj.start_blinding(['field1'], scope='annotate', src=str(folder))
    record = rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=str(strategy))
    now['time'] = '2026-10-01T10:01:00.000000Z'
    screen.gates.tree.select('large')
    low, _high = screen.gates.tree._threshold_rows['area']
    low.setText('200')
    low.editingFinished.emit()
    assert 'DEVIATION' in screen._source.text().upper()
    assert 'VERIFIED' not in screen._source.text().upper()
    assert json.loads(strategy.read_text())['gates'][0]['low'] == 150
    observed = [json.loads(line) for line in rj._seen_log(record).read_text().splitlines()]
    assert len(observed) == 1 and observed[0]['utc'] == now['time']
    now['time'] = '2026-10-01T10:02:00.000000Z'
    rj.unblind(key['key_id'], reason='scoring complete')
    now['time'] = '2026-10-01T10:03:00.000000Z'
    screen.export_gates()
    verdict = _gate_export_lock_verdicts(str(strategy), screen.gates.gates)[0]
    assert verdict['status'] == 'deviation'
    assert verdict['deviations'][0]['first_seen_utc'] == observed[0]['utc']
    screen.gates.set_gates(gates(250))
    assert 'POST-HOC' in screen._source.text().upper()
    assert _gate_export_lock_verdicts(str(strategy), screen.gates.gates)[0]['status'] == 'post_hoc'


def test_loading_another_strategy_does_not_stamp_a_false_edit_on_previous_lock(qtbot, journal):  # noqa: F811
    folder, _now = journal
    screen = make_screen(qtbot, folder)
    first, second = folder / 'first.json', folder / 'second.json'
    gates(150).save(str(first))
    gates(250).save(str(second))
    record = rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=[str(first), str(second)])
    assert screen.load_gates(str(first))
    assert screen.load_gates(str(second))
    assert screen._gate_strategy_path == str(second.resolve())
    assert not rj._seen_log(record).exists()
    assert 'match analysis lock' in screen._source.text().lower()
    screen.gates.set_gates(gates(275))
    assert 'match analysis lock' not in screen._source.text().lower()
    assert 'DEVIATION' in screen._source.text().upper()


def test_unsaved_strategy_never_claims_verification_and_duplicate_signals_do_not_recheck(qtbot, journal, monkeypatch):  # noqa: F811
    folder, _now = journal
    screen = make_screen(qtbot, folder)
    calls = []
    monkeypatch.setattr(screen_module, '_gate_export_lock_verdicts', lambda *a, **k: calls.append(k) or [])
    screen.gates.set_gates(gates())
    assert not calls
    assert 'verified' not in screen._source.text().lower()
    screen.save_gates(str(folder / 'strategy.json'))
    screen.gates.set_gates(gates(200))
    assert len(calls) == 1
    assert calls[0] == {'scope': 'live_gating_strategy', 'resolved': True}
    screen.gates.gates_changed.emit()
    screen.gates.gates_changed.emit()
    assert len(calls) == 1


def test_live_check_does_not_read_other_locked_gate_files_or_resolve_source_path(journal, monkeypatch):  # noqa: F811
    folder, _now = journal
    first, remote = folder / 'first.json', folder / 'remote.json'
    gates().save(str(first))
    gates(300).save(str(remote))
    rj.lock_analysis({'src': str(folder)}, app_key='measure', gates=[str(first), str(remote)])
    original_payload = rj._gate_payload
    def payload(value):
        assert not isinstance(value, (str, Path)), 'A live edit must compare in-memory gates only'
        return original_payload(value)
    monkeypatch.setattr(rj, '_gate_payload', payload)
    def resolve_forbidden(*_args, **_kwargs):
        raise AssertionError('Strategy was already canonicalized during save/load')
    monkeypatch.setattr(Path, 'resolve', resolve_forbidden)
    verdict = _gate_export_lock_verdicts(str(first), gates(200), scope='live_gating_strategy', resolved=True)[0]
    assert verdict['status'] == 'deviation'
    assert verdict['scope'] == 'live_gating_strategy'


def test_failed_live_check_keeps_edit_and_reports_no_verification(qtbot, journal, monkeypatch):  # noqa: F811
    folder, _now = journal
    screen = make_screen(qtbot, folder)
    screen.gates.set_gates(gates())
    screen.save_gates(str(folder / 'strategy.json'))
    def failed(*_args, **_kwargs):
        raise PermissionError('journal is read-only')
    monkeypatch.setattr(screen_module, '_gate_export_lock_verdicts', failed)
    screen.gates.set_gates(gates(225))
    assert screen.gates.gates.get('large').low == 225
    assert 'Gate lock check failed' in screen._source.text()
    assert 'read-only' in screen._source.text()
    assert 'verified' not in screen._source.text().lower()
