"""The shipped synthetic sample must exercise the real analysis and known geometry."""

import hashlib
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.host_pathogen_example import build_example, is_present


@pytest.fixture
def project(tmp_path):
    return build_example(tmp_path / 'sample')


def test_sample_has_consistent_geometry_and_explicit_parent_links(project):
    from spacr.host_pathogen import vacuole_links

    with sqlite3.connect(project / 'measurements/measurements.db') as connection:
        parasite_rows = pd.read_sql_query('select * from organelle', connection)
        hosts = pd.read_sql_query('select * from cell', connection)
    assert len(hosts) == 24
    assert len(parasite_rows) == 88
    for path in (project / 'merged').glob('*.npy'):
        stack = np.load(path)
        assert stack.dtype == np.uint16
        assert stack.shape == (256, 384, 8)
        links = vacuole_links(stack[..., 7], stack[..., 6])
        column = 'c' + str(int(path.stem.split('_')[-3][1:]))
        field = path.stem.split('_')[-2]
        recorded = parasite_rows[(parasite_rows.columnID == column) & (parasite_rows.fieldID == field)]
        joined = links.merge(recorded, left_on='label', right_on='object_label', validate='one_to_one')
        assert len(joined) == 22
        np.testing.assert_allclose(joined.pathogen_id_x, joined.pathogen_id_y, equal_nan=True)
        assert (stack[..., 4][stack[..., 6] == 6] == 0).all()


def test_settings_pack_runs_and_matches_independent_expected_results(project):
    from spacr.host_pathogen import default_settings, analyze_host_pathogen
    from spacr.qt.settings_pack import settings_from_pack

    config, report = settings_from_pack('host_pathogen', project / 'settings', defaults=default_settings())
    assert report.applied and not report.dropped and not report.malformed
    assert config['hp_parasite_table'] == 'organelle'
    original = (project / 'measurements/measurements.db').read_bytes()
    result = analyze_host_pathogen(config)
    expected = pd.read_csv(project / 'expected' / 'vacuoles.csv', dtype={'fieldID': str, 'timeID': str})
    observed = result['vacuoles'].copy()
    observed[['fieldID', 'timeID']] = observed[['fieldID', 'timeID']].astype(str)
    keys = ['plateID', 'rowID', 'columnID', 'fieldID', 'timeID', 'vacuole_id']
    joined = observed.merge(expected, on=keys, suffixes=('_actual', '_expected'), validate='one_to_one')
    assert len(joined) == 28
    for name in ('parasite_count', 'channel_0_recruitment_ratio', 'channel_1_recruitment_ratio'):
        np.testing.assert_allclose(joined[name + '_actual'].astype(float), joined[name + '_expected'], equal_nan=True)
    for name in ('channel_0_state', 'channel_1_state'):
        assert joined[name + '_actual'].tolist() == joined[name + '_expected'].tolist()
    wells = result['wells'].merge(pd.read_csv(project / 'expected' / 'wells.csv'),
                                  on=['plateID', 'rowID', 'columnID'], suffixes=('_actual', '_expected'))
    for name in ('host_cells', 'infected_cells', 'multiply_infected_cells', 'infection_fraction',
                 'vacuoles', 'linked_vacuoles', 'vacuoles_with_counts'):
        np.testing.assert_allclose(wells[name + '_actual'], wells[name + '_expected'])
    assert len(result['orphan_parasites']) == 4
    assert result['orphan_parasites'].orphan_reason.eq('unmatched_vacuole').all()
    assert result['cells'].infected.sum() == 20
    assert len(list((project / 'results' / 'host_pathogen').glob('*.csv'))) == 6
    assert (project / 'measurements/measurements.db').read_bytes() == original


def test_reusing_sample_preserves_changes_and_unknown_folders_are_not_replaced(project, tmp_path):
    changed = project / 'expected' / 'wells.csv'
    changed.write_text('user kept notes here\n')
    assert build_example(project) == project
    assert changed.read_text() == 'user kept notes here\n'
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    (unrelated / 'data.txt').write_text('preserve me')
    with pytest.raises(FileExistsError):
        build_example(unrelated)
    assert (unrelated / 'data.txt').read_text() == 'preserve me'


def test_cancelled_sample_does_not_publish_a_partial_project(tmp_path):
    progress = []
    dest = tmp_path / 'cancelled'
    with pytest.raises(RuntimeError, match='cancelled'):
        build_example(dest, progress=lambda done, total: progress.append(done),
                      cancelled=lambda: bool(progress))
    assert progress == [1]
    assert not dest.exists()
    assert not list(tmp_path.glob('.spacr-host-pathogen-*'))


def test_rebuild_is_deterministic_and_missing_inputs_are_detected(project, tmp_path):
    second = build_example(tmp_path / 'second')
    for path in (project / 'merged').glob('*.npy'):
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(
            (second / 'merged' / path.name).read_bytes()).digest()
    assert is_present(project)
    next((project / 'merged').glob('*.npy')).unlink()
    assert not is_present(project)
