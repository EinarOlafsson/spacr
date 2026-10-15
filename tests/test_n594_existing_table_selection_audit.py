"""Native reader evidence for the existing Classify table selection contract."""
import sqlite3

import pytest


DEFAULT = ['cell', 'nucleus', 'pathogen', 'cytoplasm']
SELECTED = ['cell', 'nucleus']


def plate(root):
    folder = root / 'measurements'
    folder.mkdir()
    path = folder / 'measurements.db'
    with sqlite3.connect(path) as con:
        for role in DEFAULT:
            parent = ', cell_id INTEGER' if role in {'nucleus', 'pathogen'} else ''
            con.execute(f'CREATE TABLE {role} (object_label INTEGER, plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, prcf TEXT, prc TEXT, {role}_area REAL{parent})')
            labels = [1, 2, 4] if role == 'nucleus' else [1, 2, 3] if role == 'pathogen' else [1, 2, 3, 4]
            for label in labels:
                row = [label, 'plate1', 'r1', 'c1', 'f1', 'plate1_r1_c1_f1', 'plate1_r1_c1', 100.0 + label]
                if parent:
                    row.append(label)
                con.execute(f'INSERT INTO {role} VALUES ({",".join("?" for _ in row)})', row)
    return path


def test_native_selected_reader_keeps_existing_join_semantics(tmp_path):
    from spacr.io import _read_and_merge_data

    database = plate(tmp_path)
    selected, _ = _read_and_merge_data([str(database)], SELECTED, nuclei_limit=True, pathogen_limit=3)
    default, _ = _read_and_merge_data([str(database)], DEFAULT, nuclei_limit=True, pathogen_limit=3)
    assert set(selected.index) == set(default.index) == {
        'plate1_r1_c1_f1_o1', 'plate1_r1_c1_f1_o2', 'plate1_r1_c1_f1_o4'}
    assert 'nucleus_area' in selected.columns and 'cell_area' in selected.columns
    assert 'pathogen_area' not in selected.columns and 'cytoplasm_area' not in selected.columns
    assert 'pathogen_area' in default.columns and 'cytoplasm_area' in default.columns


def test_actual_ml_entrypoint_preserves_existing_selected_tables_before_fit(tmp_path, monkeypatch):
    from spacr import ml, training_basis, utils

    database = plate(tmp_path)
    observed = []

    class ReachedFitBoundary(Exception):
        pass

    def capture(frame, *args):
        observed.append(frame.copy())
        return frame

    def stop(settings):
        raise ReachedFitBoundary

    monkeypatch.setattr(utils, 'save_settings', lambda *args, **kwargs: None)
    monkeypatch.setattr(utils, 'calculate_shortest_distance', capture)
    monkeypatch.setattr(training_basis, 'resolve_basis', stop)
    with pytest.raises(ReachedFitBoundary):
        ml.generate_ml_scores({'src': str(tmp_path), 'tables': SELECTED.copy(),
                               'dataset_mode': 'metadata', 'verbose': False})
    assert len(observed) == 1
    frame = observed[0]
    assert 'nucleus_area' in frame.columns and 'cell_area' in frame.columns
    assert 'pathogen_area' not in frame.columns and 'cytoplasm_area' not in frame.columns
    assert set(frame.index) == {'plate1_r1_c1_f1_o1', 'plate1_r1_c1_f1_o2', 'plate1_r1_c1_f1_o4'}
    assert database.is_file()


def add_object_table(database, role):
    with sqlite3.connect(database) as con:
        con.execute(f'CREATE TABLE {role} (object_label INTEGER, cell_id INTEGER, plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, prcf TEXT, prc TEXT, {role}_area REAL)')
        for label in [1, 2, 3, 4]:
            con.execute(f'INSERT INTO {role} VALUES (?,?,?,?,?,?,?,?,?)',
                        [label, label, 'plate1', 'r1', 'c1', 'f1', 'plate1_r1_c1_f1', 'plate1_r1_c1', 200.0 + label])


def add_crops(database):
    with sqlite3.connect(database) as con:
        con.execute('CREATE TABLE png_list (cell_id TEXT, plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, prcf TEXT, prcfo TEXT, png_path TEXT, crop_score REAL)')
        for label in [1, 2, 3, 4]:
            con.execute('INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?,?)',
                        [f'o{label}', 'plate1', 'r1', 'c1', 'f1', 'plate1_r1_c1_f1', f'plate1_r1_c1_f1_o{label}', f'/unused/{label}.png', 0.1 * label])


def capture_before_fit(settings, monkeypatch):
    from spacr import io, ml, utils

    observed, table_arguments = [], []
    read = io._read_and_merge_data

    class ReachedFitBoundary(Exception):
        pass

    def capture(frame, *args):
        return frame

    def reader(locs, tables, *args, **kwargs):
        table_arguments.append(list(tables))
        return read(locs, tables, *args, **kwargs)

    def stop(frame, *args, **kwargs):
        observed.append(frame.copy())
        raise ReachedFitBoundary

    monkeypatch.setattr(utils, 'save_settings', lambda *args, **kwargs: None)
    monkeypatch.setattr(utils, 'calculate_shortest_distance', capture)
    monkeypatch.setattr(io, '_read_and_merge_data', reader)
    monkeypatch.setattr(ml, 'ml_analysis', stop)
    with pytest.raises(ReachedFitBoundary):
        ml.generate_ml_scores(settings)
    assert len(observed) == len(table_arguments) == 1
    return observed[0], table_arguments[0]


def test_missing_choice_preserves_exact_default_native_frame_and_order(tmp_path, monkeypatch):
    import pandas as pd
    from spacr.io import _read_and_merge_data

    database = plate(tmp_path)
    expected, _ = _read_and_merge_data([str(database)], DEFAULT, nuclei_limit=True, pathogen_limit=3)
    settings = {'src': str(tmp_path), 'dataset_mode': 'metadata', 'verbose': False, 'n_jobs': 1}
    actual, tables = capture_before_fit(settings, monkeypatch)
    pd.testing.assert_frame_equal(actual, expected)
    assert tables == DEFAULT and 'tables' not in settings


@pytest.mark.parametrize('selection,expected,absent', [
    (tuple(SELECTED), 'nucleus_area', 'pathogen_area'),
    (['cell', 'nucleus', 'organelle'], 'organelle_area', 'pathogen_area'),
    (['cell', 'nucleus', 'png_list'], 'crop_score', 'pathogen_area'),
])
def test_existing_registered_object_and_optional_crop_tables_reach_native_loader(
        tmp_path, monkeypatch, selection, expected, absent):
    import copy

    database = plate(tmp_path)
    add_object_table(database, 'organelle')
    add_crops(database)
    original = copy.deepcopy(selection)
    settings = {'src': str(tmp_path), 'tables': selection,
                'dataset_mode': 'metadata', 'verbose': False, 'n_jobs': 1}
    actual, tables = capture_before_fit(settings, monkeypatch)
    assert expected in actual.columns and absent not in actual.columns
    assert set(actual.index) == {'plate1_r1_c1_f1_o1', 'plate1_r1_c1_f1_o2', 'plate1_r1_c1_f1_o4'}
    assert tables == list(original) and settings['tables'] == original


@pytest.mark.parametrize('selection', [None, [], (), 'cell', ['cell', 'cell'],
                                      ['unknown'], ['Cell'], ['png_list'], False,
                                      [['cell']], {'cell': True}, ['cell', None]])
def test_invalid_explicit_tables_refuse_before_settings_or_database_io(
        tmp_path, monkeypatch, selection):
    from spacr import io, ml, utils

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid tables reached settings publication or database reads')

    monkeypatch.setattr(utils, 'save_settings', forbidden)
    monkeypatch.setattr(io, '_read_and_merge_data', forbidden)
    with pytest.raises(ValueError, match='tables='):
        ml.generate_ml_scores({'src': str(tmp_path), 'tables': selection,
                               'dataset_mode': 'metadata', 'verbose': False, 'n_jobs': 1})
    assert not list(tmp_path.iterdir())


def test_optional_crops_and_annotation_basis_use_one_authoritative_target(
        tmp_path, monkeypatch):
    database = plate(tmp_path)
    add_crops(database)
    with sqlite3.connect(database) as con:
        con.execute('ALTER TABLE png_list ADD COLUMN test INTEGER')
        for label in [1, 2, 3, 4]:
            con.execute('UPDATE png_list SET test=? WHERE cell_id=?', [label % 2, f'o{label}'])
    settings = {'src': str(tmp_path), 'tables': ['cell', 'nucleus', 'png_list'],
                'dataset_mode': 'annotation', 'annotation_column': 'test',
                'positive_control_id': None, 'negative_control_id': None,
                'verbose': False, 'n_jobs': 1}
    actual, tables = capture_before_fit(settings, monkeypatch)
    assert 'test' in actual.columns and 'test_x' not in actual.columns and 'test_y' not in actual.columns
    assert set(actual['test']) == {0, 1}
    assert actual.loc['plate1_r1_c1_f1_o1', 'test'] == 1
    assert actual.loc['plate1_r1_c1_f1_o2', 'test'] == 0
    assert actual.loc['plate1_r1_c1_f1_o4', 'test'] == 0
    assert tables == settings['tables']
