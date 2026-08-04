"""Tests for the fail-loud error policy (:mod:`spacr.errors`).

Two halves:

1. The :class:`~spacr.errors.RunLedger` contract itself — counting,
   grouping, the two exception types it must re-raise rather than
   record, the abort threshold, and stamping/reading a real artifact.
2. An integration test that drives a converted product loop
   (:func:`spacr.io.convert_to_yokogawa`) over a batch containing one
   deliberately corrupt file, and asserts the three things that matter:
   the good files still converted, the bad one is named in the ledger,
   and the artifact on disk knows it is partial.

CPU-only, offline, no torch needed for the ledger half.
"""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from spacr.errors import (
    ConfigurationError,
    DataIntegrityError,
    Failure,
    PartialRunError,
    RUN_STATUS_SUFFIX,
    RUN_STATUS_TABLE,
    RunLedger,
    STATUS_COMPLETE,
    STATUS_EMPTY,
    STATUS_PARTIAL,
    STRICT_ENV_VAR,
    SpacrError,
    assert_run_complete,
    raise_if_strict,
    read_run_status,
    run_is_complete,
    strict_errors,
)


# ---------------------------------------------------------------------------
# exception hierarchy
# ---------------------------------------------------------------------------

def test_every_spacr_error_shares_one_base():
    """One `except SpacrError` catches everything spaCR raises on purpose."""
    assert issubclass(ConfigurationError, SpacrError)
    assert issubclass(DataIntegrityError, SpacrError)
    # PartialRunError is a DataIntegrityError so "the answer is wrong"
    # handlers do not have to know about the threshold mechanism.
    assert issubclass(PartialRunError, DataIntegrityError)


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------

def test_failure_round_trips_through_a_dict():
    f = Failure(item='A01', stage='seg', exc_type='ValueError',
                message='bad shape', traceback_str='Traceback...')
    d = f.to_dict()
    assert d['item'] == 'A01' and d['stage'] == 'seg'
    assert d['exc_type'] == 'ValueError' and d['message'] == 'bad shape'
    assert d['traceback_str'] == 'Traceback...'
    assert isinstance(d['timestamp'], float)
    assert f.short() == 'A01: bad shape'


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------

def test_counts_and_rate_under_mixed_success_and_failure():
    ledger = RunLedger('mixed')
    for i in range(10):
        with ledger.item(f'well_{i}', stage='measure'):
            if i % 5 == 0:                       # 2 of 10 fail
                raise ValueError(f'bad well {i}')

    assert ledger.n_attempted == 10
    assert ledger.n_succeeded == 8
    assert ledger.n_failed == 2
    assert ledger.failure_rate == pytest.approx(0.2)
    assert ledger.status == STATUS_PARTIAL
    assert ledger.is_complete is False


def test_a_clean_run_reports_complete():
    ledger = RunLedger('clean')
    for i in range(3):
        with ledger.item(f'well_{i}'):
            pass
    assert (ledger.n_attempted, ledger.n_succeeded, ledger.n_failed) == (3, 3, 0)
    assert ledger.failure_rate == 0.0
    assert ledger.status == STATUS_COMPLETE
    assert ledger.is_complete is True


def test_a_ledger_that_ran_nothing_is_empty_not_complete_looking():
    ledger = RunLedger('nothing')
    assert ledger.n_attempted == 0
    # No division by zero, and the status distinguishes "nothing ran" from
    # "everything ran and passed".
    assert ledger.failure_rate == 0.0
    assert ledger.status == STATUS_EMPTY


def test_record_success_and_record_failure_can_be_called_directly():
    ledger = RunLedger('manual')
    ledger.record_success('a', stage='s')
    ledger.record_failure('b', stage='s', exc=RuntimeError('boom'))
    assert (ledger.n_succeeded, ledger.n_failed, ledger.n_attempted) == (1, 1, 2)


def test_record_failure_accepts_a_plain_string_for_detected_failures():
    """cv2.imread returns None instead of raising; that still has to count."""
    ledger = RunLedger('detected')
    failure = ledger.record_failure('mask.tif', stage='read',
                                    exc='cv2.imread returned None')
    assert failure.exc_type == 'Failure'
    assert failure.message == 'cv2.imread returned None'
    assert failure.traceback_str == ''
    assert ledger.n_failed == 1


def test_record_failure_with_no_exception_still_records():
    ledger = RunLedger('bare')
    failure = ledger.record_failure('x')
    assert failure.exc_type == 'Failure'
    assert failure.message == 'unspecified failure'
    assert failure.stage == 'bare'          # defaults to the ledger name


def test_record_failure_keeps_the_traceback_and_logs_the_item_at_error(caplog):
    ledger = RunLedger('logged')
    with caplog.at_level('ERROR', logger='spacr.errors'):
        with ledger.item('plate1_A01', stage='segment'):
            raise FileNotFoundError('/data/plate1_A01.npy')

    failure = ledger.failures[0]
    assert failure.item == 'plate1_A01'
    assert failure.stage == 'segment'
    assert failure.exc_type == 'FileNotFoundError'
    assert 'FileNotFoundError' in failure.traceback_str
    # The item id has to be in the ERROR record or the log is useless.
    assert any('plate1_A01' in rec.message and rec.levelname == 'ERROR'
               for rec in caplog.records)


def test_an_exception_with_no_message_still_gets_a_readable_one():
    ledger = RunLedger('silent')
    with ledger.item('x'):
        raise RuntimeError()
    assert ledger.failures[0].message == 'RuntimeError'


def test_repr_shows_the_verdict_at_a_glance():
    ledger = RunLedger('shown')
    ledger.record_failure('a', exc=ValueError('v'))
    text = repr(ledger)
    assert 'shown' in text and 'partial' in text and 'failed=1' in text


# ---------------------------------------------------------------------------
# item(): swallow-and-continue, but re-raise the two things that must not be
# ---------------------------------------------------------------------------

def test_item_swallows_a_normal_exception_and_the_loop_continues():
    ledger = RunLedger('batch')
    processed = []
    for name in ['a', 'b', 'c']:
        with ledger.item(name, stage='load'):
            if name == 'b':
                raise OSError('unreadable')
            processed.append(name)

    # The whole point: b failed, a and c still ran.
    assert processed == ['a', 'c']
    assert [f.item for f in ledger.failures] == ['b']


def test_item_re_raises_configuration_error():
    """A wrong src path is not a per-item failure — one mistake must not be
    filed as N data errors."""
    ledger = RunLedger('cfg')
    with pytest.raises(ConfigurationError):
        with ledger.item('w1', stage='setup'):
            raise ConfigurationError('src does not exist')
    assert ledger.n_failed == 0
    assert ledger.n_attempted == 0


def test_item_re_raises_keyboard_interrupt():
    """Ctrl-C must abort the run, not be recorded as a corrupt image."""
    ledger = RunLedger('interrupt')
    with pytest.raises(KeyboardInterrupt):
        with ledger.item('w1', stage='measure'):
            raise KeyboardInterrupt
    assert ledger.n_failed == 0
    assert ledger.n_attempted == 0


def test_item_re_raises_system_exit():
    ledger = RunLedger('exiting')
    with pytest.raises(SystemExit):
        with ledger.item('w1'):
            raise SystemExit(1)
    assert ledger.n_attempted == 0


def test_item_echoes_the_legacy_console_message_on_failure(capsys):
    """Adoption sites keep the exact stdout users already grep for."""
    ledger = RunLedger('echoing')
    with ledger.item('bad.tif', stage='tiff', echo='Error processing bad.tif'):
        raise ValueError('not a TIFF')
    out = capsys.readouterr().out
    assert 'Error processing bad.tif: not a TIFF' in out


def test_item_echo_is_silent_when_the_item_succeeds(capsys):
    ledger = RunLedger('quiet')
    with ledger.item('good.tif', echo='Error processing good.tif'):
        pass
    assert capsys.readouterr().out == ''


def test_item_yields_the_ledger_so_the_body_can_use_it():
    ledger = RunLedger('yielded')
    with ledger.item('a') as handle:
        assert handle is ledger


def test_continue_inside_item_still_counts_as_a_success():
    """`continue` inside the with-block exits it cleanly, as several
    converted loops in io.py rely on."""
    ledger = RunLedger('continuing')
    for name in ['a', 'b']:
        with ledger.item(name):
            if name == 'a':
                continue
    assert ledger.n_succeeded == 2
    assert ledger.n_failed == 0


# ---------------------------------------------------------------------------
# grouping — 40 identical errors are one problem, not 40
# ---------------------------------------------------------------------------

def test_forty_identical_failures_are_one_group_of_forty():
    ledger = RunLedger('plate')
    for i in range(40):
        with ledger.item(f'well_{i:03d}', stage='segment'):
            raise FileNotFoundError('/data/missing.npy')

    groups = ledger.grouped_failures()
    assert list(groups) == ['FileNotFoundError']
    assert len(groups['FileNotFoundError']) == 40

    summary = ledger.summary()
    assert 'FileNotFoundError x40' in summary
    # One distinct message, annotated with its count — not forty lines.
    assert summary.count('/data/missing.npy') == 1
    assert '(x40)' in summary
    assert 'RUN INCOMPLETE' in summary
    assert 'attempted : 40' in summary


def test_groups_keep_first_seen_order_and_split_by_exception_type():
    ledger = RunLedger('mixed_types')
    for exc in [ValueError('a'), OSError('b'), ValueError('c')]:
        with ledger.item('item', stage='s'):
            raise exc
    groups = ledger.grouped_failures()
    assert list(groups) == ['ValueError', 'OSError']
    assert len(groups['ValueError']) == 2


def test_summary_truncates_distinct_messages_but_says_how_many_it_hid():
    ledger = RunLedger('many_messages')
    for i in range(6):
        with ledger.item(f'w{i}'):
            raise ValueError(f'distinct problem {i}')
    summary = ledger.summary(max_examples=2)
    assert 'distinct problem 0' in summary
    assert 'distinct problem 1' in summary
    assert 'distinct problem 5' not in summary
    assert 'more distinct message(s)' in summary


def test_summary_truncates_exception_groups_but_says_how_many_it_hid():
    ledger = RunLedger('many_types')
    for exc in [ValueError('v'), OSError('o'), TypeError('t'), KeyError('k')]:
        with ledger.item('i'):
            raise exc
    summary = ledger.summary(max_groups=2)
    assert 'ValueError x1' in summary
    assert 'TypeError' not in summary
    assert 'more exception type(s)' in summary


def test_summary_of_a_clean_run_says_complete_and_lists_no_failures():
    ledger = RunLedger('clean')
    with ledger.item('a'):
        pass
    summary = ledger.summary()
    assert 'run complete' in summary
    assert 'RUN INCOMPLETE' not in summary
    assert 'Failures grouped by' not in summary


def test_summary_of_an_empty_run_says_it_processed_nothing():
    summary = RunLedger('void').summary()
    assert 'processed nothing' in summary


# ---------------------------------------------------------------------------
# raise_if_worse_than
# ---------------------------------------------------------------------------

def _ledger_with_rate(n_fail, n_ok, name='rated'):
    ledger = RunLedger(name)
    for i in range(n_fail):
        ledger.record_failure(f'bad_{i}', exc=ValueError('nope'))
    for i in range(n_ok):
        ledger.record_success(f'ok_{i}')
    return ledger


def test_raise_if_worse_than_fires_above_the_threshold():
    ledger = _ledger_with_rate(n_fail=3, n_ok=2)     # 60% failed
    with pytest.raises(PartialRunError) as excinfo:
        ledger.raise_if_worse_than(0.5)
    text = str(excinfo.value)
    assert '3 of 5' in text
    # The summary travels with the error so the abort is self-explanatory.
    assert 'RUN INCOMPLETE' in text


def test_raise_if_worse_than_does_not_fire_below_the_threshold():
    ledger = _ledger_with_rate(n_fail=1, n_ok=9)     # 10% failed
    assert ledger.raise_if_worse_than(0.5) is ledger


def test_raise_if_worse_than_does_not_fire_exactly_at_the_threshold():
    ledger = _ledger_with_rate(n_fail=1, n_ok=1)     # exactly 50%
    assert ledger.raise_if_worse_than(0.5) is ledger


def test_raise_if_worse_than_is_a_no_op_when_nothing_was_attempted():
    ledger = RunLedger('void')
    assert ledger.raise_if_worse_than(0.0) is ledger


def test_raise_if_worse_than_accepts_a_custom_message():
    ledger = _ledger_with_rate(n_fail=2, n_ok=0)
    with pytest.raises(PartialRunError, match='cross-validation is meaningless'):
        ledger.raise_if_worse_than(0.5, message='cross-validation is meaningless')


def test_partial_run_error_is_catchable_as_a_data_integrity_error():
    ledger = _ledger_with_rate(n_fail=2, n_ok=0)
    with pytest.raises(DataIntegrityError):
        ledger.raise_if_worse_than(0.5)


# ---------------------------------------------------------------------------
# stamping a real sqlite artifact
# ---------------------------------------------------------------------------

def _make_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute('CREATE TABLE measurements (well TEXT)')
    conn.execute("INSERT INTO measurements VALUES ('A01')")
    conn.commit()
    conn.close()


def test_stamping_a_real_sqlite_db_and_reading_the_status_back(tmp_path):
    db = tmp_path / 'measurements.db'
    _make_db(db)

    ledger = RunLedger('measure_crop')
    for i in range(4):
        with ledger.item(f'field_{i}.npy', stage='measure'):
            if i == 2:
                raise ValueError('mask stack has the wrong dtype')

    assert ledger.stamp(db) == db

    records = read_run_status(db)
    assert len(records) == 1
    record = records[0]
    assert record['name'] == 'measure_crop'
    assert record['status'] == STATUS_PARTIAL
    assert (record['n_attempted'], record['n_succeeded'], record['n_failed']) == (4, 3, 1)
    assert record['failure_rate'] == pytest.approx(0.25)
    assert [f['item'] for f in record['failures']] == ['field_2.npy']
    assert 'RUN INCOMPLETE' in record['summary']

    # The artifact itself now knows it is suspect.
    assert run_is_complete(db) is False
    with pytest.raises(DataIntegrityError, match='did not complete'):
        assert_run_complete(db)


def test_stamping_leaves_the_rest_of_the_database_untouched(tmp_path):
    db = tmp_path / 'measurements.db'
    _make_db(db)
    RunLedger('m').record_failure('x', exc=ValueError('v'))
    RunLedger('m').stamp(db)

    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute('SELECT well FROM measurements').fetchall() == [('A01',)]
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {'measurements', RUN_STATUS_TABLE} <= names


def test_a_clean_run_stamps_a_status_that_reads_as_complete(tmp_path):
    db = tmp_path / 'measurements.db'
    _make_db(db)

    ledger = RunLedger('measure_crop')
    for i in range(3):
        with ledger.item(f'field_{i}.npy'):
            pass
    ledger.stamp(db)

    records = read_run_status(db)
    assert len(records) == 1
    assert records[0]['status'] == STATUS_COMPLETE
    assert records[0]['n_failed'] == 0
    assert records[0]['failures'] == []
    assert run_is_complete(db) is True
    assert_run_complete(db)                       # must not raise


def test_stamps_accumulate_one_row_per_stage(tmp_path):
    db = tmp_path / 'measurements.db'
    _make_db(db)

    first = RunLedger('preprocess')
    first.record_success('plate1')
    first.stamp(db)

    second = RunLedger('measure_crop')
    second.record_failure('field_1.npy', exc=ValueError('v'))
    second.stamp(db)

    records = read_run_status(db)
    assert [r['name'] for r in records] == ['preprocess', 'measure_crop']
    assert run_is_complete(db) is False


def test_stamping_a_db_that_does_not_exist_yet_creates_it(tmp_path):
    db = tmp_path / 'fresh.db'
    RunLedger('new').stamp(db)
    assert db.is_file()
    assert read_run_status(db)[0]['status'] == STATUS_EMPTY


def test_reading_a_db_without_a_run_status_table_returns_nothing(tmp_path):
    db = tmp_path / 'legacy.db'
    _make_db(db)
    assert read_run_status(db) == []
    # An unstamped legacy artifact must not be reported as broken.
    assert run_is_complete(db) is True
    assert_run_complete(db)


def test_reading_a_db_that_does_not_exist_returns_nothing(tmp_path):
    assert read_run_status(tmp_path / 'nope.db') == []
    assert run_is_complete(tmp_path / 'nope.db') is True


# ---------------------------------------------------------------------------
# stamping a non-database artifact
# ---------------------------------------------------------------------------

def test_stamping_a_csv_writes_a_sibling_run_status_json(tmp_path):
    csv = tmp_path / 'rename_log.csv'
    csv.write_text('Original File,Renamed TIFF\n')

    ledger = RunLedger('convert')
    ledger.record_failure('bad.tif', stage='tiff', exc=ValueError('not a TIFF'))
    sidecar = ledger.stamp(csv)

    assert sidecar == tmp_path / ('rename_log' + RUN_STATUS_SUFFIX)
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]['status'] == STATUS_PARTIAL

    # read_run_status finds the sidecar from the artifact path.
    records = read_run_status(csv)
    assert records[0]['failures'][0]['item'] == 'bad.tif'
    assert run_is_complete(csv) is False


def test_sidecar_stamps_accumulate(tmp_path):
    csv = tmp_path / 'log.csv'
    csv.write_text('x\n')
    RunLedger('first').stamp(csv)
    RunLedger('second').stamp(csv)
    assert [r['name'] for r in read_run_status(csv)] == ['first', 'second']


def test_reading_a_sidecar_by_its_own_path_works(tmp_path):
    csv = tmp_path / 'log.csv'
    csv.write_text('x\n')
    sidecar = RunLedger('only').stamp(csv)
    assert [r['name'] for r in read_run_status(sidecar)] == ['only']


def test_a_sidecar_holding_a_bare_object_is_upgraded_to_a_list(tmp_path):
    """Tolerate a hand-written or older single-object sidecar."""
    csv = tmp_path / 'log.csv'
    csv.write_text('x\n')
    sidecar = tmp_path / ('log' + RUN_STATUS_SUFFIX)
    sidecar.write_text(json.dumps({'name': 'legacy', 'n_failed': 0}))

    assert [r['name'] for r in read_run_status(csv)] == ['legacy']
    RunLedger('new').stamp(csv)
    assert [r['name'] for r in read_run_status(csv)] == ['legacy', 'new']


def test_reading_an_unstamped_non_db_artifact_returns_nothing(tmp_path):
    csv = tmp_path / 'log.csv'
    csv.write_text('x\n')
    assert read_run_status(csv) == []
    assert run_is_complete(csv) is True


def test_stamp_creates_the_sidecar_directory_when_missing(tmp_path):
    artifact = tmp_path / 'nested' / 'out.csv'
    sidecar = RunLedger('n').stamp(artifact)
    assert sidecar.is_file()


# ---------------------------------------------------------------------------
# to_json / to_dict
# ---------------------------------------------------------------------------

def test_to_json_writes_the_whole_ledger_and_makes_parent_dirs(tmp_path):
    ledger = RunLedger('json_run')
    with ledger.item('a', stage='s'):
        raise ValueError('kaput')
    ledger.record_success('b', stage='s')

    target = ledger.to_json(tmp_path / 'deep' / 'ledger.json')
    payload = json.loads(target.read_text())
    assert payload['name'] == 'json_run'
    assert payload['status'] == STATUS_PARTIAL
    assert payload['n_attempted'] == 2
    assert payload['success_by_stage'] == {'s': 1}
    assert payload['failures'][0]['exc_type'] == 'ValueError'
    assert 'summary' in payload


def test_to_dict_is_json_serialisable_even_with_tracebacks():
    """The traceback is the field that makes this hard, so assert it survives.

    A ``to_dict`` that simply dropped ``traceback_str`` would serialise
    perfectly and pass a bare ``json.dumps(...)``; one that kept the live
    traceback object would not serialise at all. Only checking the round trip
    *and* the text distinguishes the two from a correct implementation.
    """
    ledger = RunLedger('serialisable')
    with ledger.item('a'):
        raise ValueError('x')

    payload = ledger.to_dict()
    assert json.loads(json.dumps(payload)) == payload, \
        'a value survived to_dict() but changed identity through JSON'

    failure, = payload['failures']
    assert failure['exc_type'] == 'ValueError'
    assert failure['message'] == 'x'
    assert isinstance(failure['timestamp'], float)
    assert 'Traceback (most recent call last)' in failure['traceback_str']
    assert "raise ValueError('x')" in failure['traceback_str']


def test_run_id_and_started_time_are_recorded():
    ledger = RunLedger('ided')
    assert len(ledger.run_id) == 12
    assert ledger.started_utc.startswith('20')
    assert ledger.to_dict()['run_id'] == ledger.run_id


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def test_finalize_prints_the_loud_block_when_something_failed(capsys, tmp_path):
    db = tmp_path / 'measurements.db'
    _make_db(db)

    ledger = RunLedger('measure_crop')
    ledger.record_success('a')
    ledger.record_failure('b', exc=ValueError('boom'))
    assert ledger.finalize(artifact=db) is ledger

    out = capsys.readouterr().out
    assert 'RUN INCOMPLETE' in out
    assert 'ARTIFACTS FROM THIS RUN ARE INCOMPLETE' in out
    assert read_run_status(db)[0]['status'] == STATUS_PARTIAL


def test_finalize_is_silent_on_stdout_for_a_clean_run(capsys):
    ledger = RunLedger('clean')
    ledger.record_success('a')
    ledger.finalize()
    assert capsys.readouterr().out == ''


def test_finalize_can_be_asked_to_print_a_clean_run_too(capsys):
    ledger = RunLedger('clean')
    ledger.record_success('a')
    ledger.finalize(quiet_when_clean=False)
    assert 'run complete' in capsys.readouterr().out


def test_finalize_logs_one_line_not_the_whole_block(caplog):
    """Logging the block too rendered the summary twice in a terminal."""
    ledger = RunLedger('once')
    ledger.record_failure('a', exc=ValueError('v'))
    with caplog.at_level('ERROR', logger='spacr.errors'):
        ledger.finalize()
    finalize_records = [r for r in caplog.records if 'RUN INCOMPLETE' in r.message]
    assert len(finalize_records) == 1
    assert '=====' not in finalize_records[0].message


def test_finalize_stamps_before_it_raises_so_the_evidence_survives(tmp_path):
    db = tmp_path / 'measurements.db'
    _make_db(db)

    ledger = RunLedger('doomed')
    for i in range(4):
        ledger.record_failure(f'x{i}', exc=ValueError('v'))
    ledger.record_success('ok')

    with pytest.raises(PartialRunError):
        ledger.finalize(artifact=db, threshold=0.5)

    # The abort must not lose the stamp.
    assert read_run_status(db)[0]['n_failed'] == 4


def test_finalize_with_a_threshold_does_not_raise_on_an_acceptable_run():
    ledger = RunLedger('fine')
    ledger.record_success('a')
    ledger.record_failure('b', exc=ValueError('v'))
    assert ledger.finalize(threshold=0.9) is ledger


# ---------------------------------------------------------------------------
# strict mode
# ---------------------------------------------------------------------------

def test_strict_errors_is_off_by_default(monkeypatch):
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    assert strict_errors() is False
    assert strict_errors({}) is False


@pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'on', ' y '])
def test_strict_errors_reads_truthy_env_values(monkeypatch, value):
    monkeypatch.setenv(STRICT_ENV_VAR, value)
    assert strict_errors() is True


@pytest.mark.parametrize('value', ['0', 'false', 'no', '', 'maybe'])
def test_strict_errors_ignores_other_env_values(monkeypatch, value):
    monkeypatch.setenv(STRICT_ENV_VAR, value)
    assert strict_errors() is False


def test_a_settings_key_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, '1')
    assert strict_errors({'strict_errors': False}) is False
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    assert strict_errors({'strict_errors': True}) is True


def test_a_none_settings_value_falls_through_to_the_environment(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, '1')
    assert strict_errors({'strict_errors': None}) is True


def test_raise_if_strict_only_logs_when_not_strict(monkeypatch, caplog):
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    with caplog.at_level('ERROR', logger='spacr.errors'):
        assert raise_if_strict('the src folder is empty') is False
    assert 'the src folder is empty' in caplog.text


def test_raise_if_strict_raises_configuration_error_in_strict_mode(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, '1')
    with pytest.raises(ConfigurationError, match='the src folder is empty'):
        raise_if_strict('the src folder is empty')


def test_raise_if_strict_chains_the_original_exception(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, '1')
    original = OSError('no such directory')
    with pytest.raises(ConfigurationError) as excinfo:
        raise_if_strict('bad src', exc=original)
    assert excinfo.value.__cause__ is original


def test_raise_if_strict_can_raise_a_different_type(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, '1')
    with pytest.raises(DataIntegrityError):
        raise_if_strict('columns disagree', error_type=DataIntegrityError)


def test_raise_if_strict_honours_a_settings_dict(monkeypatch):
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    with pytest.raises(ConfigurationError):
        raise_if_strict('bad src', settings={'strict_errors': True})


# ---------------------------------------------------------------------------
# the module must stay stdlib-only
# ---------------------------------------------------------------------------

def test_importing_errors_does_not_drag_in_torch_or_cellpose():
    """errors.py is imported at module scope by io/core/measure/plot/
    deep_spacr, so it must never pull a heavyweight dependency."""
    import spacr

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(spacr.__file__)))
    code = (
        'import sys; import spacr.errors; '
        'print(",".join(sorted(m for m in '
        '("torch", "cellpose", "pandas", "numpy", "tensorflow") '
        'if m in sys.modules)))'
    )
    # An explicit PYTHONPATH: the ambient one may carry a sitecustomize that
    # pre-imports torch, which would make this assertion meaningless.
    env = dict(os.environ)
    env['PYTHONPATH'] = repo_root
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ''


def test_errors_module_only_imports_the_stdlib_database_sibling():
    """No heavy sibling dependency can sneak into the error hot path.

    Parsed rather than grepped, so the usage examples in the module
    docstring are not mistaken for real imports. ``database_concurrency`` is
    deliberately standard-library-only and is the one approved sibling: it
    makes run-status writes transactional without importing an analysis stack.
    """
    import ast

    import spacr.errors as errors_module
    import spacr.database_concurrency as concurrency_module

    with open(errors_module.__file__, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())

    imported = set()
    relative = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                relative.add(node.module)
            else:
                imported.add((node.module or '').split('.')[0])

    assert relative == {'database_concurrency'}
    assert 'spacr' not in imported
    assert imported.isdisjoint({'torch', 'cellpose', 'pandas', 'numpy',
                                'tensorflow', 'skimage', 'cv2'})

    with open(concurrency_module.__file__, encoding='utf-8') as handle:
        concurrency_tree = ast.parse(handle.read())
    concurrency_imports = set()
    for node in ast.walk(concurrency_tree):
        if isinstance(node, ast.Import):
            concurrency_imports.update(
                alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, (
                f'database_concurrency imports sibling {node.module!r}')
            concurrency_imports.add((node.module or '').split('.')[0])
    assert concurrency_imports.isdisjoint({
        'spacr', 'torch', 'cellpose', 'pandas', 'numpy', 'tensorflow',
        'skimage', 'cv2',
    })


# ---------------------------------------------------------------------------
# INTEGRATION — a real converted loop, one deliberately corrupt item
# ---------------------------------------------------------------------------

def test_convert_to_yokogawa_survives_one_corrupt_file_and_says_so(tmp_path, capsys):
    """The whole point of the item, end to end.

    Three TIFFs go into a folder, one of them is not a TIFF at all. The
    two good files must still convert, the bad one must be named in the
    ledger, and ``rename_log.csv`` must be stamped as partial so a
    downstream reader can tell it is looking at a subset.
    """
    import numpy as np
    import tifffile

    from spacr.io import convert_to_yokogawa

    folder = tmp_path / 'raw'
    folder.mkdir()
    tifffile.imwrite(str(folder / 'good_a.tif'), np.full((4, 4), 11, np.uint16))
    tifffile.imwrite(str(folder / 'good_b.tif'), np.full((4, 4), 22, np.uint16))
    (folder / 'corrupt.tif').write_bytes(b'this is definitely not a TIFF')

    ledger = convert_to_yokogawa(str(folder))

    # (a) the other items still processed
    converted = sorted(p.name for p in folder.glob('plate*.tif'))
    assert len(converted) == 2
    import pandas as pd
    assert len(pd.read_csv(folder / 'rename_log.csv')) == 2

    # (b) the failure is in the ledger, named
    assert ledger.n_attempted == 3
    assert ledger.n_succeeded == 2
    assert [f.item for f in ledger.failures] == ['corrupt.tif']
    assert ledger.failures[0].stage == 'tiff'

    # ...and it was announced loudly, last, on stdout
    out = capsys.readouterr().out
    assert 'RUN INCOMPLETE' in out
    assert 'corrupt.tif' in out
    assert out.rindex('ARTIFACTS FROM THIS RUN ARE INCOMPLETE') > out.rindex(
        'Processing complete.')

    # (c) the output is marked partial, on disk, for whoever reads it next
    csv_path = folder / 'rename_log.csv'
    assert run_is_complete(csv_path) is False
    record = read_run_status(csv_path)[0]
    assert record['name'] == 'convert_to_yokogawa'
    assert record['status'] == STATUS_PARTIAL
    assert (record['n_attempted'], record['n_failed']) == (3, 1)
    assert record['failures'][0]['item'] == 'corrupt.tif'
    with pytest.raises(DataIntegrityError):
        assert_run_complete(csv_path)


def test_convert_to_yokogawa_stamps_a_clean_batch_as_complete(tmp_path, capsys):
    """The clean case must be stamped too, or 'complete' means nothing."""
    import numpy as np
    import tifffile

    from spacr.io import convert_to_yokogawa

    folder = tmp_path / 'raw'
    folder.mkdir()
    for name in ('a.tif', 'b.tif'):
        tifffile.imwrite(str(folder / name), np.ones((4, 4), np.uint16))

    ledger = convert_to_yokogawa(str(folder))

    assert ledger.is_complete
    assert 'RUN INCOMPLETE' not in capsys.readouterr().out

    csv_path = folder / 'rename_log.csv'
    record = read_run_status(csv_path)[0]
    assert record['status'] == STATUS_COMPLETE
    assert record['n_attempted'] == 2 and record['n_failed'] == 0
    assert run_is_complete(csv_path) is True


def test_measure_crop_still_returns_none_on_bad_settings_by_default(tmp_path, capsys,
                                                                    monkeypatch):
    """Default behaviour is unchanged: print the warning, return None."""
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    from spacr.measure import measure_crop

    merged = tmp_path / 'merged'
    merged.mkdir()
    assert measure_crop({'src': str(merged), 'timelapse': False,
                         'normalize': True}) is None
    assert 'normalize' in capsys.readouterr().out


def test_measure_crop_raises_on_bad_settings_under_strict_mode(tmp_path, monkeypatch):
    """Opting in turns the silent None into a ConfigurationError."""
    monkeypatch.setenv(STRICT_ENV_VAR, '1')
    from spacr.measure import measure_crop

    merged = tmp_path / 'merged'
    merged.mkdir()
    with pytest.raises(ConfigurationError, match='list of two percentiles'):
        measure_crop({'src': str(merged), 'timelapse': False, 'normalize': True})


def _synthetic_merged_stack(rng):
    """Build a (48, 48, 7) merged stack: 4 intensity channels + 3 masks."""
    import numpy as np

    cell = np.zeros((48, 48), np.uint16)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    for i, (r, c) in enumerate([(6, 6), (6, 26), (26, 6)], start=1):
        cell[r:r + 14, c:c + 14] = i
        nucleus[r + 3:r + 8, c + 3:c + 8] = i
        pathogen[r + 9:r + 12, c + 9:c + 12] = i
    channels = []
    for _ in range(4):
        base = rng.integers(50, 200, size=(48, 48)).astype(np.uint16)
        base[cell > 0] += 3000
        channels.append(base)
    return np.stack(channels + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


def test_measure_crop_marks_measurements_db_partial_when_a_field_fails(tmp_path, capsys):
    """The scenario the whole item exists for.

    Four fields go in, one is unreadable. The other three must still be
    measured into ``measurements.db``, and the database must carry a
    ``run_status`` row saying it covers 3 of 4 — so a downstream
    regression can tell it is about to analyse a subset.
    """
    import numpy as np

    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / 'merged'
    merged.mkdir()
    (tmp_path / 'measurements').mkdir()
    rng = np.random.default_rng(0)
    for name in ('plate1_A01_F001.npy', 'plate1_A02_F001.npy', 'plate1_A03_F001.npy'):
        np.save(merged / name, _synthetic_merged_stack(rng))
    # The deliberately corrupt field.
    (merged / 'plate1_A04_F001.npy').write_bytes(b'not a numpy file at all')

    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(merged), 'channels': [0, 1, 2, 3],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'png_dims': [0, 1, 2], 'png_size': [32, 32],
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 1, 'test_mode': False, 'cytoplasm': True,
    })
    measure_crop(dict(settings))

    db = tmp_path / 'measurements' / 'measurements.db'

    # (a) the other fields still processed
    conn = sqlite3.connect(str(db))
    try:
        measured = {row[0] for row in
                    conn.execute('SELECT DISTINCT file_name FROM cell')}
    finally:
        conn.close()
    assert len(measured) == 3

    # (b) the failure is accounted for, by name
    record = read_run_status(db)[-1]
    assert record['name'] == 'measure_crop'
    assert (record['n_attempted'], record['n_succeeded'], record['n_failed']) == (4, 3, 1)
    assert record['failures'][0]['item'] == 'plate1_A04_F001.npy'

    # (c) the artifact is marked partial, and does NOT claim success
    assert record['status'] == STATUS_PARTIAL
    assert run_is_complete(db) is False
    with pytest.raises(DataIntegrityError):
        assert_run_complete(db)

    out = capsys.readouterr().out
    assert 'RUN INCOMPLETE' in out
    assert 'Successfully completed run' not in out


def test_load_images_from_paths_reports_what_it_dropped(capsys, tmp_path):
    """A second converted loop: the returned dict is short, and now says so."""
    import numpy as np
    import tifffile

    from spacr.io import load_images_from_paths

    good = tmp_path / 'good.tif'
    tifffile.imwrite(str(good), np.ones((3, 3), np.uint16))
    corrupt = tmp_path / 'corrupt.tif'
    corrupt.write_text('not a tiff')

    out = load_images_from_paths({'fov': [str(good), str(corrupt)]})

    assert len(out['fov']) == 1                 # the good one still loaded
    printed = capsys.readouterr().out
    assert 'Error loading image from' in printed
    assert 'RUN INCOMPLETE' in printed          # and the shortfall is announced
    assert 'corrupt.tif' in printed
