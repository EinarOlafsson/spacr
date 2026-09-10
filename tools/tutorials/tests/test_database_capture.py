"""Tiny SQLite/CSV tests; no Qt application, spaCR import or real-data reads."""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_database as recorder


COLUMNS = ['plateID', 'rowID', 'columnID', 'fieldID', 'object_label',
           'cell_area', 'cell_channel_1_mean_intensity', 'note']
ROWS = [
    ('plate1', 'r2', 'c1', 'f1', 1.0, 18000.0, 3.5, 'first, with comma'),
    ('plate1', 'r1', 'c1', 'f1', 1.0, 9000.0, 2.25, 'second'),
    ('plate1', 'r2', 'c1', 'f1', 2.0, 12000.0, 7.75, 'third'),
    ('plate1', 'r1', 'c1', 'f1', 2.0, 18000.0, 4.0, 'fourth\nsecond line'),
    ('plate1', 'r1', 'c2', 'f1', 1.0, 3000.0, None, 'fifth'),
    ('plate1', 'r1', 'c2', 'f1', 2.0, 12000.0, 8.0, 'sixth — μ'),
    ('plate1', 'r1', 'c2', 'f2', 1.0, 10000.0, 4.25, 'seventh'),
    ('plate1', 'r1', 'c2', 'f2', 2.0, 8500.0, 6.0, None),
]
FILTERED = [ROWS[i] for i in (0, 2, 3, 5, 6)]


@pytest.fixture
def database(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    database = source / 'measurements.db'
    with sqlite3.connect(database) as con:
        con.execute('CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, '
                    'fieldID TEXT, object_label REAL, cell_area REAL, '
                    'cell_channel_1_mean_intensity REAL, note TEXT)')
        con.executemany('INSERT INTO cell VALUES (?,?,?,?,?,?,?,?)', ROWS)
        con.execute('CREATE TABLE settings (name TEXT, value TEXT)')
    return database


def make_csv(path, rows=FILTERED, columns=COLUMNS):
    indices = [COLUMNS.index(name) for name in columns]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows([[row[i] for i in indices] for row in rows])
    return path


def test_copy_is_exact_private_exclusive_and_preserves_existing_sidecars(database, tmp_path):
    Path(str(database) + '-wal').touch()
    Path(str(database) + '-shm').write_bytes(b'preserved fixture sidecar')
    before = recorder.file_bundle(database)
    private = tmp_path / 'private'
    private.mkdir()
    copied = recorder.prepare_database_copy(database, private / 'measurements.db')
    assert copied['copy_is_byte_identical'] is True
    assert copied['source_opened_in_gui'] is False
    assert Path(copied['database']).read_bytes() == database.read_bytes()
    assert copied['source_bundle'] == before == recorder.file_bundle(database)
    assert list(private.iterdir()) == [private / 'measurements.db']
    with pytest.raises(FileExistsError):
        recorder.prepare_database_copy(database, private / 'measurements.db')
    assert copied['source_bundle'] == recorder.file_bundle(database)


@pytest.mark.parametrize('suffix', ['-wal', '-journal'])
def test_nonempty_pending_write_file_refuses_copy_before_output(database, tmp_path, suffix):
    Path(str(database) + suffix).write_bytes(b'not checkpointed')
    target = tmp_path / 'new.db'
    with pytest.raises(ValueError, match='nonempty WAL/journal'):
        recorder.prepare_database_copy(database, target)
    assert not target.exists()


def test_wrong_audited_hash_refuses_copy_before_output(database, tmp_path):
    target = tmp_path / 'new.db'
    with pytest.raises(ValueError, match='audited source hash'):
        recorder.prepare_database_copy(database, target, expected_sha256='0' * 64)
    assert not target.exists()


@pytest.mark.parametrize('suffix', ['-wal', '-shm', '-journal'])
def test_orphaned_destination_sidecar_is_not_overwritten_or_adopted(database, tmp_path, suffix):
    target = tmp_path / 'new.db'
    sidecar = Path(str(target) + suffix)
    sidecar.write_bytes(b'existing private data')
    with pytest.raises(FileExistsError):
        recorder.prepare_database_copy(database, target)
    assert not target.exists()
    assert sidecar.read_bytes() == b'existing private data'


def test_no_copy_inside_source_or_through_source_symlink(database, tmp_path):
    with pytest.raises(ValueError, match='outside the source'):
        recorder.prepare_database_copy(database, database.parent / 'other.db')
    alias = tmp_path / 'alias'
    alias.symlink_to(database.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='outside the source'):
        recorder.prepare_database_copy(database, alias / 'other.db')
    direct = tmp_path / 'alias.db'
    direct.symlink_to(database)
    with pytest.raises(FileExistsError):
        recorder.prepare_database_copy(database, direct)


def test_copy_detects_source_change_during_copy(database, tmp_path, monkeypatch):
    actual_copy = recorder.shutil.copyfileobj
    def mutate_private_fixture_source(source, target, length):
        actual_copy(source, target, length)
        with sqlite3.connect(database) as con:
            con.execute('UPDATE cell SET note=? WHERE _rowid_=1', ('changed fixture',))
    monkeypatch.setattr(recorder.shutil, 'copyfileobj', mutate_private_fixture_source)
    with pytest.raises(ValueError, match='sidecars changed'):
        recorder.prepare_database_copy(database, tmp_path / 'new.db')


def test_readonly_inspection_is_bounded_uses_full_keys_and_unshadowed_rowid(database):
    before = recorder.file_bundle(database)
    facts = recorder.inspect_database(database)
    assert facts['rows'] == 8
    assert facts['filtered_rows'] == 5
    assert facts['columns'] == COLUMNS
    assert facts['tables'] == ['cell', 'settings']
    assert facts['rowid_alias'] == '_rowid_'
    assert facts['identity_columns'] == list(recorder.IDENTITY)
    assert facts['full_object_identities_unique'] is True
    assert recorder.file_bundle(database) == before
    with recorder._readonly(database) as con:
        assert con.execute('PRAGMA query_only').fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError):
            con.execute('DELETE FROM cell')


@pytest.mark.parametrize('case', ['duplicate', 'null', 'missing_column'])
def test_incomplete_or_duplicate_full_identity_is_rejected(database, case):
    with sqlite3.connect(database) as con:
        if case == 'duplicate':
            con.execute('INSERT INTO cell SELECT * FROM cell WHERE _rowid_=1')
        elif case == 'null':
            con.execute('UPDATE cell SET fieldID=NULL WHERE _rowid_=1')
        else:
            con.execute('ALTER TABLE cell RENAME COLUMN fieldID TO missing')
    with pytest.raises(ValueError, match='identit'):
        recorder.inspect_database(database)


@pytest.mark.parametrize('threshold', [0, 999999])
def test_filter_must_be_nonempty_strict_subset(database, threshold):
    with pytest.raises(ValueError, match='nonempty strict subset'):
        recorder.inspect_database(database, threshold=threshold)


def test_schema_and_row_budgets_are_enforced(database):
    with pytest.raises(ValueError, match='row budget'):
        recorder.inspect_database(database, max_rows=3)
    with pytest.raises(ValueError, match='schema'):
        recorder.inspect_database(database, table='cell; DROP TABLE cell')
    with pytest.raises(ValueError, match='identity'):
        recorder.inspect_database(database, table='settings')


@pytest.mark.parametrize('threshold', [True, float('nan'), float('inf'), '10000 OR 1=1'])
def test_threshold_cannot_be_nonfinite_or_inject_sql(database, threshold):
    with pytest.raises(ValueError, match='threshold must be finite'):
        recorder.inspect_database(database, threshold=threshold)


@pytest.mark.parametrize('sort,rows', [
    (None, ROWS[:2]),
    (('cell_area', True), [ROWS[0], ROWS[3]]),
    (('cell_area', False), [ROWS[4], ROWS[7]]),
])
def test_preview_identity_order_values_and_total_are_independent(database, sort, rows):
    before = deepcopy(rows)
    proof = recorder.verify_preview(database, COLUMNS, rows, 8, sort=sort, max_loaded=2)
    assert proof['loaded_rows'] == 2
    assert proof['exact_total_rows'] == 8
    assert proof['identities_and_values_exact'] is True
    assert rows == before


def test_filtered_preview_count_is_whole_query_not_drawn_page(database):
    proof = recorder.verify_preview(database, COLUMNS, FILTERED[:2], 5,
                                    threshold=10000, max_loaded=2)
    assert proof['loaded_rows'] == 2 and proof['exact_total_rows'] == 5
    with pytest.raises(ValueError, match='complete count'):
        recorder.verify_preview(database, COLUMNS, FILTERED[:2], 2, threshold=10000)


@pytest.mark.parametrize('case', ['identity', 'mean', 'order', 'partial_sort', 'column'])
def test_preview_rejects_plausible_counts_with_wrong_data(database, case):
    columns, rows = COLUMNS[:], [list(row) for row in ROWS[:2]]
    sort = None
    if case == 'identity':
        rows[0][1] = 'another-row'
    elif case == 'mean':
        rows[0][6] += 1
    elif case == 'order':
        rows.reverse()
    elif case == 'partial_sort':
        sort = ('cell_area', True)  # largest of the first page is NOT the whole-table top two.
    else:
        columns.reverse()
    with pytest.raises(ValueError, match='Preview'):
        recorder.verify_preview(database, columns, rows, 8, sort=sort)


@pytest.mark.parametrize('rows,budget', [([], 1000), (ROWS, 2), (ROWS, 1001)])
def test_preview_never_silently_accepts_empty_or_unbounded_reads(database, rows, budget):
    with pytest.raises(ValueError, match='page budget'):
        recorder.verify_preview(database, COLUMNS, rows, 8, max_loaded=budget)


def test_exact_full_export_preserves_float_ids_unicode_nulls_and_quoted_csv(database, tmp_path):
    exported = make_csv(tmp_path / 'full.csv')
    proof = recorder.verify_export(database, exported, COLUMNS, 5)
    assert proof['rows'] == 5
    assert proof['full_object_identities_exact'] and proof['all_values_exact']
    assert proof['sha256'] == hashlib.sha256(exported.read_bytes()).hexdigest()
    assert len(proof['ordered_identity_sha256']) == 64
    assert '1.0' in exported.read_text()


def test_visible_projection_with_full_identity_is_also_verified(database, tmp_path):
    columns = list(recorder.IDENTITY) + ['cell_area']
    exported = make_csv(tmp_path / 'visible.csv', columns=columns)
    assert recorder.verify_export(database, exported, columns, 5)['columns'] == 6


@pytest.mark.parametrize('case', ['missing', 'extra', 'identity', 'mean', 'swapped', 'unfiltered', 'count'])
def test_export_requires_complete_exact_rows_not_only_a_matching_row_count(database, tmp_path, case):
    rows = [list(row) for row in FILTERED]
    expected_count = 5
    if case == 'missing':
        rows.pop()
    elif case == 'extra':
        rows.append(list(ROWS[0]))
    elif case == 'identity':
        rows[0][3] = 'wrong-field'
    elif case == 'mean':
        rows[0][6] += 0.01
    elif case == 'swapped':
        rows[0], rows[1] = rows[1], rows[0]
    elif case == 'unfiltered':
        rows = ROWS
    else:
        expected_count = 4
    exported = make_csv(tmp_path / 'wrong.csv', rows=rows)
    with pytest.raises(ValueError, match='CSV'):
        recorder.verify_export(database, exported, COLUMNS, expected_count)


def test_header_must_match_actual_selection_and_export_must_carry_full_keys(database, tmp_path):
    exported = make_csv(tmp_path / 'wrong_header.csv', columns=COLUMNS[::-1])
    with pytest.raises(ValueError, match='header'):
        recorder.verify_export(database, exported, COLUMNS, 5)
    with pytest.raises(ValueError, match='identity column'):
        recorder.verify_export(database, exported, ['cell_area'], 5)


@pytest.mark.parametrize('change', ['database', 'sidecar', 'added_sidecar'])
def test_original_byte_or_sidecar_changes_are_detected(database, change):
    if change == 'sidecar':
        Path(str(database) + '-shm').write_bytes(b'before')
    before = recorder.file_bundle(database)
    if change == 'database':
        with sqlite3.connect(database) as con:
            con.execute('UPDATE cell SET cell_area=cell_area+1 WHERE _rowid_=1')
    else:
        Path(str(database) + '-shm').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sidecars changed'):
        recorder.require_unchanged_source(database, before)


def test_retained_eight_sentence_mapping_matches_real_catalog_without_expansion():
    catalog_path = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials/catalog/lessons_en.json'
    catalog = json.loads(catalog_path.resolve().read_text())
    original = deepcopy(catalog)
    result = recorder.retained_scene_mapping(catalog)
    assert result['english_sha256'] == recorder.ENGLISH_SHA256
    assert result['narration_changed'] is False
    assert result['existing_voices_reusable'] is True
    assert [s['narration'] for s in result['scenes']] == list(recorder.NARRATIONS)
    assert [s['visual'] for s in result['scenes']] == list(recorder.SCENE_FRAMES)
    assert len(result['scenes']) == 8
    assert catalog == original
    lesson = next(x for x in catalog['lessons'] if x['id'] == '34_database')
    lesson['scenes'][0]['narration'] += ' This must not silently replace the old voices.'
    with pytest.raises(ValueError, match='eight-sentence lesson changed'):
        recorder.retained_scene_mapping(catalog)


class JobStates:
    """Pure lifecycle fixture, not a mocked Qt/app pipeline."""
    def __init__(self, states):
        self.states = list(states)
        self.polls = 0

    def is_busy(self):
        return self.states[0][0]

    def active_jobs(self):
        return self.states[0][1]

    def queued_jobs(self):
        return self.states[0][2]

    def settle(self, seconds):
        assert seconds == .05
        assert len(self.states) > 1, 'No event pumping is needed once all jobs retire'
        self.states.pop(0)
        self.polls += 1


def test_job_retirement_drains_busy_queued_and_winding_down_states():
    screen = JobStates([(True, 1, 1), (False, 1, 0), (False, 0, 1),
                        (True, 1, 0), (False, 0, 0)])
    result = recorder._retire_database_jobs(screen, screen.settle)
    assert result['event_processing_polls'] == 4
    assert result['busy'] is False
    assert result['active_jobs'] == result['queued_jobs'] == 0
    assert result['workers_forcibly_stopped'] is False


@pytest.mark.parametrize('screen', [None, JobStates([(False, 0, 0)])])
def test_no_worker_means_no_unnecessary_event_wait(screen):
    def forbidden_wait(_seconds):
        raise AssertionError('Unexpected wait with no pending job')
    assert recorder._retire_database_jobs(screen, forbidden_wait)['event_processing_polls'] == 0


@pytest.mark.parametrize('original', [AssertionError('wrong captured data'),
                                     TimeoutError('capture deadline expired'),
                                     KeyboardInterrupt('capture interrupted')])
def test_cleanup_retires_workers_and_reraises_the_same_original_failure(original):
    screen = JobStates([(True, 1, 0), (False, 1, 0), (False, 0, 0)])
    reports = []
    with pytest.raises(type(original)) as raised:
        with recorder._database_job_lifecycle(lambda: screen, screen.settle,
                                              lambda proof, error: reports.append((proof, error))):
            raise original
    assert raised.value is original
    assert reports[0][1] is original
    assert reports[0][0]['event_processing_polls'] == 2
    assert screen.active_jobs() == 0


def test_successful_body_also_retires_workers_before_reporting():
    screen = JobStates([(False, 1, 0), (False, 0, 0)])
    reports = []
    with recorder._database_job_lifecycle(lambda: screen, screen.settle,
                                          lambda proof, error: reports.append((proof, error))):
        pass
    assert reports[0][1] is None and reports[0][0]['active_jobs'] == 0
    assert screen.polls == 1


def test_cleanup_reporting_failure_does_not_replace_the_capture_failure():
    screen = JobStates([(True, 1, 0), (False, 0, 0)])
    original = RuntimeError('the original capture failure')
    def failed_report(_proof, _error):
        raise OSError('private evidence volume unavailable')
    with pytest.raises(RuntimeError) as raised:
        with recorder._database_job_lifecycle(lambda: screen, screen.settle, failed_report):
            raise original
    assert raised.value is original
    assert screen.active_jobs() == 0
    assert any('private evidence volume unavailable' in note for note in original.__notes__)


def test_cleanup_error_is_not_silenced_when_there_was_no_original_error():
    screen = JobStates([(False, 0, 0)])
    def failed_report(_proof, _error):
        raise OSError('failed retirement evidence')
    with pytest.raises(OSError, match='failed retirement evidence'):
        with recorder._database_job_lifecycle(lambda: screen, screen.settle, failed_report):
            pass


def test_retirement_does_not_hide_an_event_loop_error():
    screen = JobStates([(True, 1, 0)])
    def failed_settle(_seconds):
        raise RuntimeError('event loop processing failed')
    with pytest.raises(RuntimeError, match='event loop processing failed'):
        recorder._retire_database_jobs(screen, failed_settle)


def test_header_index_is_relative_to_visible_model_not_full_schema():
    assert COLUMNS.index('cell_area') == 5
    assert recorder._visible_header_section(['cell_area'], 'cell_area') == 0
    assert recorder._visible_header_section(['object_label', 'cell_area'], 'cell_area') == 1
    # The real table has three substring matches, not an exact-match search.
    assert recorder._visible_header_section(
        ['cell_area', 'cell_area_filled', 'cell_area_bbox'], 'cell_area') == 0
    for columns in [[], ['another_column'], ['cell_area', 'cell_area']]:
        with pytest.raises(ValueError, match='absent or ambiguous'):
            recorder._visible_header_section(columns, 'cell_area')


class EditorKeys:
    Key_A, Key_Backspace, Key_Tab, ControlModifier = 'A', 'Backspace', 'Tab', 'Control'


class TextEditor:
    def __init__(self, text):
        self.text, self.selected, self.focused = text, False, False

    def setFocus(self):
        self.focused = True


class KeyboardEvents:
    """Unit-level keyboard semantics; no Qt application or result replacement."""
    def __init__(self):
        self.events = []

    def keyClick(self, widget, key, modifier=None):
        self.events.append((key, modifier))
        if key == 'A' and modifier == 'Control':
            widget.selected = True
        elif key == 'Backspace' and widget.selected:
            widget.text, widget.selected = '', False

    def keyClicks(self, widget, text):
        self.events.append(('type', text))
        if text:
            widget.text = text if widget.selected else widget.text + text
            widget.selected = False


@pytest.mark.parametrize('value,expected', [('', ''), ('cell_area', 'cell_area'), (10000, '10000')])
def test_native_text_replacement_clears_old_text_for_empty_and_populated_values(value, expected):
    editor, keyboard = TextEditor('cell_channel_1_mean_intensity'), KeyboardEvents()
    recorder._native_replace_text(editor, value, keyboard, EditorKeys)
    assert editor.text == expected
    assert editor.focused is True
    assert keyboard.events[0] == ('A', 'Control')
    assert keyboard.events[1] == ('Backspace', None)
    assert keyboard.events[-1] == ('Tab', None)


def test_column_failure_diagnostic_records_actual_search_and_full_keys_without_mutation():
    before = deepcopy(ROWS[:2])
    snapshot = recorder._column_view_snapshot('cell_area', ['cell_area'], COLUMNS, ROWS[:2])
    assert snapshot['column_search_text'] == 'cell_area'
    assert snapshot['visible_columns'] == ['cell_area']
    assert snapshot['loaded_rows'] == 2
    assert snapshot['identity_columns'] == list(recorder.IDENTITY)
    assert snapshot['row_identities'] == [list(row[:5]) for row in ROWS[:2]]
    assert ROWS[:2] == before
    with pytest.raises(ValueError, match='page budget'):
        recorder._column_view_snapshot('', COLUMNS, COLUMNS, [ROWS[0]] * 1001)
