"""Tiny source projects exercise subset integrity without importing spaCR."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recruitment_data
from recruitment_data import prepare_subset, TABLES


FIELDS = ("field'one", "field_two")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def small_npy(value):
    header = repr({'descr': '<u2', 'fortran_order': False, 'shape': (2, 2, 7)})
    padding = (-10 - len(header) - 1) % 64
    header = (header + ' ' * padding + '\n').encode('ascii')
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<28H', *([value] * 28))


@pytest.fixture
def source(tmp_path, request):
    project = tmp_path / 'source'
    (project / 'measurements').mkdir(parents=True)
    (project / 'merged').mkdir()
    database = project / 'measurements' / 'measurements.db'
    real_ids = getattr(request, 'param', False)
    with closing(sqlite3.connect(database)) as connection, connection:
        for table in TABLES:
            parent_type = 'REAL' if real_ids else 'INTEGER'
            label_type = 'REAL' if real_ids == 'all' else 'INTEGER'
            child = f', cell_id {parent_type} NOT NULL' if table in ('nucleus', 'pathogen') else ''
            connection.execute(f'''CREATE TABLE {table} (
                object_label {label_type} NOT NULL CHECK(object_label > 0),
                plateID TEXT NOT NULL, rowID TEXT NOT NULL,
                columnID TEXT NOT NULL, fieldID TEXT NOT NULL,
                file_name TEXT NOT NULL, path_name TEXT, prcf TEXT,
                measurement REAL, arbitrary_blob BLOB, nullable_feature TEXT
                {child})''')
            connection.execute(f'CREATE INDEX {table}_filename ON {table}(file_name)')
        connection.execute('CREATE TRIGGER keep_cell_schema AFTER UPDATE ON cell BEGIN SELECT 1; END')
        for index, stem in enumerate((*FIELDS, 'not_selected'), start=1):
            labels = (1, 2) if index == 2 else (1,)
            for label in labels:
                for table in TABLES:
                    child = table in ('nucleus', 'pathogen')
                    row = [label + 10 if child else label,
                           'plate1', 'r1', 'c1', f'f{index}', stem,
                           f'/original/acquisition/{stem}.npy',
                           f'plate1_r1_c1_f{index}', 1.25 * label,
                           bytes([index, label, 255]), None]
                    if child:
                        row.append(label)
                    connection.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", row)
            (project / 'merged' / (stem + '.npy')).write_bytes(small_npy(index))
    return project


def mutate(source, sql, parameters=()):
    database = source / 'measurements' / 'measurements.db'
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(sql, parameters)


def test_exact_rows_schemas_arrays_and_provenance_are_preserved(source, tmp_path, monkeypatch):
    before = {str(path.relative_to(source)): digest(path)
              for path in source.rglob('*') if path.is_file()}
    connections = []
    original_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        connections.append((str(database), kwargs.copy()))
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(recruitment_data.sqlite3, 'connect', connect)
    destination = tmp_path / 'subset'
    manifest = prepare_subset(source, destination, fields=FIELDS)
    assert '?mode=ro&immutable=1' in connections[0][0]
    assert connections[0][1]['uri'] is True
    assert json.loads((destination / 'provenance.json').read_text()) == manifest
    assert manifest['source_database_sha256'] == before['measurements/measurements.db']
    subset_database = destination / 'measurements' / 'measurements.db'
    assert manifest['subset_database_sha256'] == digest(subset_database)
    with closing(sqlite3.connect(source / 'measurements' / 'measurements.db')) as original, \
            closing(sqlite3.connect(subset_database)) as copied:
        for table in TABLES:
            expected = original.execute(
                f'SELECT * FROM {table} WHERE file_name IN (?,?) ORDER BY fieldID,object_label', FIELDS).fetchall()
            assert copied.execute(f'SELECT * FROM {table} ORDER BY fieldID,object_label').fetchall() == expected
            schema = "SELECT type,name,sql FROM sqlite_schema WHERE tbl_name=? ORDER BY type,name"
            assert copied.execute(schema, (table,)).fetchall() == original.execute(schema, (table,)).fetchall()
            assert manifest['tables'][table]['row_count'] == 3
            assert manifest['tables'][table]['per_field'] == {FIELDS[0]: 1, FIELDS[1]: 2}
            assert len(manifest['tables'][table]['rows_sha256']) == 64
        # label 1 is deliberately reused in different fields, and both survive.
        assert copied.execute('SELECT COUNT(*) FROM cell WHERE object_label=1').fetchone()[0] == 2
    assert {p.name for p in (destination / 'merged').iterdir()} == {f + '.npy' for f in FIELDS}
    for array in manifest['arrays']:
        assert digest(Path(array['destination'])) == array['sha256']
        assert Path(array['destination']).read_bytes() == Path(array['source']).read_bytes()
        assert array['identity']['plateID'] == 'plate1'
    after = {str(path.relative_to(source)): digest(path)
             for path in source.rglob('*') if path.is_file()}
    assert before == after


def test_a_host_in_another_field_does_not_satisfy_a_child_link(source, tmp_path):
    # Cell 2 exists only in field_two. A label-only join would accept this.
    mutate(source, 'UPDATE pathogen SET cell_id=2 WHERE file_name=?', (FIELDS[0],))
    destination = tmp_path / 'broken'
    with pytest.raises(ValueError, match='Broken pathogen host-cell link'):
        prepare_subset(source, destination, fields=FIELDS)
    assert not destination.exists()
    assert not list(tmp_path.glob('.recruitment-subset-*'))


@pytest.mark.parametrize('source', ['children', 'all'], indirect=True)
def test_integral_real_ids_keep_exact_sqlite_values_and_links(source, tmp_path):
    database = source / 'measurements' / 'measurements.db'
    before = digest(database)
    destination = tmp_path / 'real_ids'
    manifest = prepare_subset(source, destination, fields=FIELDS)
    with closing(sqlite3.connect(database)) as original, \
            closing(sqlite3.connect(destination / 'measurements' / 'measurements.db')) as copied:
        for table in ('nucleus', 'pathogen'):
            query = (f'SELECT object_label,typeof(object_label),cell_id,typeof(cell_id) '
                     f'FROM {table} WHERE file_name IN (?,?) ORDER BY fieldID,object_label')
            expected = original.execute(query, FIELDS).fetchall()
            assert len(expected) == 3
            assert all(type(row[2]) is float and row[3] == 'real' for row in expected)
            assert copied.execute(query, FIELDS).fetchall() == expected
            assert all(type(row['cell_id']) is float
                       for row in manifest['tables'][table]['identities'])
        for table in TABLES:
            query = f'SELECT * FROM {table} WHERE file_name IN (?,?) ORDER BY fieldID,object_label'
            assert copied.execute(query, FIELDS).fetchall() == original.execute(query, FIELDS).fetchall()
    assert digest(database) == before


@pytest.mark.parametrize('source', ['children'], indirect=True)
def test_real_parent_id_in_another_field_is_still_rejected(source, tmp_path):
    mutate(source, 'UPDATE pathogen SET cell_id=2.0 WHERE file_name=?', (FIELDS[0],))
    with pytest.raises(ValueError, match='Broken pathogen host-cell link'):
        prepare_subset(source, tmp_path / 'wrong_real_parent', fields=FIELDS)


@pytest.mark.parametrize('source', ['children'], indirect=True)
@pytest.mark.parametrize('parent', [1.5, float('inf'), float('-inf'), 0.0, -1.0])
def test_invalid_real_parent_ids_are_rejected(source, tmp_path, parent):
    mutate(source, 'UPDATE nucleus SET cell_id=? WHERE file_name=?', (parent, FIELDS[0]))
    with pytest.raises(ValueError, match='nucleus has an invalid cell_id'):
        prepare_subset(source, tmp_path / 'invalid_real_parent', fields=FIELDS)


@pytest.mark.parametrize('value', [True, False, 1.5, float('nan'), float('inf'),
                                  float('-inf'), 0, 0.0, -1, -1.0, '1', None])
def test_identity_validation_rejects_non_integral_or_non_numeric_ids(value):
    # SQLite maps bound NaN to NULL and bool to integer, so exercise these
    # distinctions before SQLite erases their original Python types.
    assert not recruitment_data._positive_integral_id(value)
    record = dict(zip(recruitment_data.LOCATION, ('plate1', 'r1', 'c1', 'f1')))
    record.update(object_label=value, file_name='one')
    with pytest.raises(ValueError, match='invalid object_label'):
        recruitment_data._validate_identity('cell', record, {}, set())


def test_cytoplasm_requires_the_same_full_cell_identity(source, tmp_path):
    mutate(source, 'UPDATE cytoplasm SET object_label=2 WHERE file_name=?', (FIELDS[0],))
    with pytest.raises(ValueError, match='Cytoplasm identities'):
        prepare_subset(source, tmp_path / 'wrong_cytoplasm', fields=FIELDS)
    assert not (tmp_path / 'wrong_cytoplasm').exists()


@pytest.mark.parametrize('table', ['cell', 'nucleus', 'pathogen', 'cytoplasm'])
def test_duplicate_object_identities_are_rejected(source, tmp_path, table):
    mutate(source, f'INSERT INTO {table} SELECT * FROM {table} WHERE file_name=?', (FIELDS[0],))
    with pytest.raises(ValueError, match=f'Duplicate {table} object identity'):
        prepare_subset(source, tmp_path / 'duplicates', fields=FIELDS)


def test_conflicting_location_metadata_is_rejected(source, tmp_path):
    mutate(source, 'UPDATE nucleus SET columnID=? WHERE file_name=?', ('c2', FIELDS[0]))
    with pytest.raises(ValueError, match='Conflicting field identity'):
        prepare_subset(source, tmp_path / 'conflicting', fields=FIELDS)


def test_every_selected_field_must_have_every_table(source, tmp_path):
    mutate(source, 'DELETE FROM nucleus WHERE file_name=?', (FIELDS[0],))
    with pytest.raises(ValueError, match='nucleus has missing selected fields'):
        prepare_subset(source, tmp_path / 'missing', fields=FIELDS)


def test_source_and_existing_destinations_are_never_overwritten(source, tmp_path):
    existing = tmp_path / 'existing'
    existing.mkdir()
    sentinel = existing / 'precious.txt'
    sentinel.write_text('preserve me')
    for destination in (source, existing):
        with pytest.raises(FileExistsError):
            prepare_subset(source, destination, fields=FIELDS)
    assert sentinel.read_text() == 'preserve me'
    with pytest.raises(ValueError, match='outside the source'):
        prepare_subset(source, source / 'subset', fields=FIELDS)
    assert not (source / 'subset').exists()


def test_existing_dangling_symlink_is_not_replaced(source, tmp_path):
    destination = tmp_path / 'reserved'
    destination.symlink_to(tmp_path / 'absent')
    with pytest.raises(FileExistsError):
        prepare_subset(source, destination, fields=FIELDS)
    assert destination.is_symlink()


def test_row_bound_rejects_instead_of_truncating(source, tmp_path):
    with pytest.raises(ValueError, match='exceeds max_rows_per_table'):
        prepare_subset(source, tmp_path / 'too_many', fields=FIELDS, max_rows_per_table=2)
    assert not (tmp_path / 'too_many').exists()


def test_missing_array_does_not_publish_a_partial_project(source, tmp_path):
    (source / 'merged' / (FIELDS[0] + '.npy')).unlink()
    with pytest.raises(FileNotFoundError):
        prepare_subset(source, tmp_path / 'missing_array', fields=FIELDS)
    assert not (tmp_path / 'missing_array').exists()


def test_nonempty_wal_is_not_silently_ignored(source, tmp_path):
    wal = source / 'measurements' / 'measurements.db-wal'
    wal.write_bytes(b'uncheckpointed fixture data')
    with pytest.raises(ValueError, match='nonempty WAL'):
        prepare_subset(source, tmp_path / 'pending_wal', fields=FIELDS)
    assert wal.read_bytes() == b'uncheckpointed fixture data'


def test_source_change_during_copy_prevents_publication(source, tmp_path, monkeypatch):
    original_hash = recruitment_data._sha256

    def changed_during_hash(path):
        result = original_hash(path)
        if path == source / 'measurements' / 'measurements.db':
            mutate(source, 'UPDATE cell SET measurement=99 WHERE file_name=?', (FIELDS[0],))
        return result

    monkeypatch.setattr(recruitment_data, '_sha256', changed_during_hash)
    with pytest.raises(ValueError, match='Source database changed'):
        prepare_subset(source, tmp_path / 'changed', fields=FIELDS)
    assert not (tmp_path / 'changed').exists()


@pytest.mark.parametrize('fields', [(), ('../escape',), ('field.npy',), (FIELDS[0], FIELDS[0])])
def test_invalid_selections_are_rejected(source, tmp_path, fields):
    with pytest.raises(ValueError):
        prepare_subset(source, tmp_path / 'invalid', fields=fields)
