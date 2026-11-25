"""A3-BUG2: Measure and the OPS phenotype path measure ONE outside region.

The paper measured one field both ways and the ``outside`` columns differed
(channel 0 mean max |diff| 8.94, channel 1 13.61). Two producers wrote the
column ``<object>_channel_<i>_outside_<stat>``: Measure's outside ring block,
which ignored ``profile_outside_rule`` and always kept the neighbouring
nuclei in the nucleus ring, and the ``outside`` profiling family, which
follows the rule and is the only producer in OPS. Under ``'background'``
(also every settings file older than schema 965) the two rings differed.

Now both take the ring from :func:`spacr.measure._outside_ring_selections`
and read the rule through :func:`spacr.measure._resolve_outside_rule`. These
tests run one synthetic field through Measure (``_measure_crop_core``, the
per-field Measure run, written to its database) and through the OPS
phenotype task (``ops_engine._ph_measure_table`` with the rule from
``_ph_outside_rule_option``), and require equal outside columns. Settings
files stamped before 979 keep the old split through ``'legacy'``.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M
from spacr import ops_engine
from spacr import settings as S
from spacr.settings import get_measure_crop_settings

ROLES = ('cell', 'nucleus', 'cytoplasm')
NAMES = ['channel_0', 'channel_1']


def _field(size=64):
    """Two touching cells whose nuclei sit 2 px apart across the shared edge,
    so the nucleus ring holds the neighbouring nucleus, and one free cell."""
    cell = np.zeros((size, size), np.uint16)
    nucleus = np.zeros_like(cell)
    cell[8:40, 4:30] = 1
    cell[8:40, 30:58] = 2
    cell[42:60, 10:50] = 3
    nucleus[18:28, 22:29] = 1
    nucleus[18:28, 31:38] = 2
    nucleus[46:54, 24:34] = 3
    rng = np.random.default_rng(7)
    channels = []
    for c in range(2):
        plane = rng.integers(50, 200, size=(size, size)).astype(np.uint16)
        plane[cell > 0] += np.uint16(500 * (c + 1))
        plane[nucleus > 0] += np.uint16(2000 + 300 * c)
        channels.append(plane)
    return np.stack(channels + [cell, nucleus], axis=-1).astype(np.uint16)


def _measure(tmp_path, **settings):
    root = tmp_path / 'measure'
    merged = root / 'merged'
    merged.mkdir(parents=True)
    (root / 'measurements').mkdir()
    np.save(merged / 'plate1_A01_F001.npy', _field())
    run = get_measure_crop_settings(settings={})
    run.update({
        'src': str(merged), 'channels': [0, 1],
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cytoplasm_min_size': 0,
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': None,
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'test_mode': False, 'cytoplasm': True, 'n_jobs': 1,
        'experiment': 'exp'})
    run.update(settings)
    result = M._measure_crop_core(0, [], 'plate1_A01_F001.npy', run)
    assert result[4] in ('', None), result[4]
    with sqlite3.connect(root / 'measurements' / 'measurements.db') as conn:
        return {role: pd.read_sql(f'SELECT * FROM "{role}"', conn)
                .set_index('object_label').sort_index() for role in ROLES}


def _ops(tmp_path, **settings):
    """The OPS phenotype task of the field, as a run builds and measures it."""
    field = _field()
    root = tmp_path / 'ops'
    root.mkdir(parents=True)
    np.save(root / 'stack.npy', field[..., :2].astype(np.float32))
    np.savez(root / 'masks.npz', nuclei=field[..., 3].astype(np.int32),
             cells=field[..., 2].astype(np.int32))
    families = ops_engine._ph_profile_features(settings)
    task = {'site': 0, 'staged': str(root / 'stack.npy'),
            'mask_path': str(root / 'masks.npz'), 'files': None,
            'channels': NAMES, 'texture': False,
            'profile_features': families,
            **ops_engine._ph_outside_rule_option(settings)}
    table, unreadable, _shape = ops_engine._ph_measure_table(task)
    assert not unreadable
    return table.set_index('label').sort_index()


def _outside_columns(frame, role):
    return [c for c in frame.columns
            if c.startswith(f'{role}_channel_') and '_outside_' in c
            and '__dup' not in c]


@pytest.mark.parametrize('rule', ['object', 'background'])
@pytest.mark.parametrize('measure_families', [['outside'], None])
def test_measure_and_ops_measure_the_same_outside_region(
        tmp_path, rule, measure_families):
    """One field, equivalent settings: every outside column is equal.

    ``measure_families=None`` is Measure's default, its outside ring block
    alone; OPS writes the columns through the ``outside`` family.
    """
    measured = _measure(tmp_path, profile_features=measure_families,
                        profile_outside_rule=rule)
    ops = _ops(tmp_path, ops_ph_profile_features=['outside'],
               ops_ph_profile_outside_rule=rule)
    compared = 0
    for role in ('cell', 'nucleus'):
        frame = measured[role]
        columns = _outside_columns(frame, role)
        assert len(columns) == 16, columns   # 2 channels x mean + 7 percentiles
        for column in columns:
            want = ops[column].reindex(frame.index).to_numpy(np.float64)
            got = frame[column].to_numpy(np.float64)
            assert np.array_equal(got, want, equal_nan=True), (
                rule, role, column, np.nanmax(np.abs(got - want)))
            compared += 1
    if measure_families:
        columns = _outside_columns(measured['cytoplasm'], 'cytoplasm')
        assert columns
        for column in columns:
            want = ops[column].reindex(
                measured['cytoplasm'].index).to_numpy(np.float64)
            assert np.array_equal(
                measured['cytoplasm'][column].to_numpy(np.float64), want,
                equal_nan=True), (rule, column)
    assert compared == 32


def test_the_rule_changes_the_nucleus_ring_of_this_field(tmp_path):
    """The field is one where the rule matters, so the test above can fail."""
    near = _ops(tmp_path / 'a', ops_ph_profile_features=['outside'],
                ops_ph_profile_outside_rule='object')
    far = _ops(tmp_path / 'b', ops_ph_profile_features=['outside'],
               ops_ph_profile_outside_rule='background')
    column = 'nucleus_channel_0_outside_mean'
    assert not np.allclose(near[column], far[column])
    np.testing.assert_array_equal(near['cell_channel_0_outside_mean'],
                                  far['cell_channel_0_outside_mean'])


def test_measure_writes_one_meaning_per_outside_column(tmp_path):
    """With block and family both on, Measure's two copies agree."""
    measured = _measure(tmp_path, profile_features=['outside'],
                        profile_outside_rule='background')
    frame = measured['nucleus']
    for column in _outside_columns(frame, 'nucleus'):
        np.testing.assert_array_equal(frame[column],
                                      frame[f'{column}__dup1'])


def test_both_settings_resolve_through_one_reading():
    assert M._resolve_outside_rule(None) == 'object'
    assert M._resolve_outside_rule('') == 'object'
    assert M._resolve_outside_rule(' Background ') == 'background'
    assert ops_engine._ph_outside_rule_option(
        {'ops_ph_profile_features': ['outside'],
         'ops_ph_profile_outside_rule': None}) == {'outside_rule': 'object'}
    with pytest.raises(ValueError, match='ops_ph_profile_outside_rule'):
        ops_engine._ph_outside_rule_option(
            {'ops_ph_profile_features': ['outside'],
             'ops_ph_profile_outside_rule': 'legacy'})
    with pytest.raises(ValueError, match='profile_outside_rule'):
        M._resolve_outside_rule('ring')


def test_older_measure_files_replay_the_old_split(tmp_path):
    """Schema 979: a Measure file older than 979 under 'background' reads
    'legacy', the old block ('object' rings) and family (background)."""
    assert S.SETTINGS_SCHEMA_VERSION >= 979
    assert 979 in S.SETTINGS_FORMAT_VERSIONS and 979 not in S.DEFAULT_CHANGES
    unstamped = {}
    S._fold_older_defaults(unstamped, 0)
    assert unstamped['profile_outside_rule'] == 'legacy'
    assert unstamped['ops_ph_profile_outside_rule'] == 'background'
    explicit = {'profile_outside_rule': 'background'}
    S._fold_older_defaults(explicit, 976)
    assert explicit['profile_outside_rule'] == 'legacy'
    current = {'profile_outside_rule': 'background'}
    S._fold_older_defaults(current, S.SETTINGS_SCHEMA_VERSION)
    assert current['profile_outside_rule'] == 'background'
    kept = {'profile_outside_rule': 'object'}
    S._fold_older_defaults(kept, 970)
    assert kept['profile_outside_rule'] == 'object'

    legacy = _measure(tmp_path / 'legacy', profile_features=['outside'],
                      profile_outside_rule='legacy')['nucleus']
    near = _ops(tmp_path / 'near', ops_ph_profile_features=['outside'],
                ops_ph_profile_outside_rule='object')
    far = _ops(tmp_path / 'far', ops_ph_profile_features=['outside'],
               ops_ph_profile_outside_rule='background')
    for column in _outside_columns(legacy, 'nucleus'):
        np.testing.assert_array_equal(
            legacy[column], near[column].reindex(legacy.index))
        np.testing.assert_array_equal(
            legacy[f'{column}__dup1'], far[column].reindex(legacy.index))
