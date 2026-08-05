"""Every join in ``spacr.measure`` states the relationship it expects, and the
child->parent joins state ``one_to_one``.

A child object -- a nucleus, a pathogen -- belongs to exactly one cell
everywhere downstream of this module. ``ObjectTableSchema.row_key_columns()``
keys the ``nucleus`` and ``pathogen`` tables on one row per ``object_label``
per field; those tables carry a single scalar ``cell_id``; and
``spacr.utils._merge_and_save_to_database`` joins morphology to intensity on
``object_label`` with ``validate='one_to_one'``. A frame holding the same label
twice is not a shape ``measurements.db`` can store.

So the only open question is *where* a fan-out stops, and that question has one
right answer. ``_measure_crop_core`` writes the object tables one call at a
time -- cell, then nucleus, then pathogen -- so a fan-out allowed through the
morphology pass is not caught until the write for its own table, with the
earlier tables for that field already committed. This module pins both halves:
the contract, and the atomicity that the contract buys.

Previously tried and backed out: relaxing these two joins to ``one_to_many``,
on the theory that a nucleus straddling two touching cells is a legitimate
shape to carry forward. ``test_the_pipeline_resolves_a_straddling_child_before_measuring``
shows the pipeline removes that shape before the join is even reached, and
``test_a_fanned_out_field_writes_no_table_at_all`` shows what the relaxation
cost when it was reached: a half-written field.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _straddling_field():
    """Two touching cells; the nucleus and the pathogen sit on the border."""
    cell = np.zeros((24, 48), dtype=np.uint16)
    cell[2:22, 2:24] = 1
    cell[2:22, 24:46] = 2

    nucleus = np.zeros((24, 48), dtype=np.uint16)
    nucleus[10:14, 20:28] = 1          # crosses x=24

    pathogen = np.zeros((24, 48), dtype=np.uint16)
    pathogen[4:8, 21:27] = 1           # crosses x=24 too

    return cell, nucleus, pathogen


def _settings(**over):
    s = {
        'cell_mask_dim': 0, 'nucleus_mask_dim': 1, 'pathogen_mask_dim': 2,
        'organelle_mask_dim': None, 'cytoplasm': False,
    }
    s.update(over)
    return s


def _crop_settings(src, **over):
    """Every key ``_measure_crop_core`` reads on the 2-D measurement path."""
    s = {
        'src': str(src),
        'channels': [0, 1],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'organelle_mask_dim': None,
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cytoplasm_min_size': 0,
        # cytoplasm=True because _exclude_objects drops every cell without one.
        'cytoplasm': True, 'uninfected': True,
        'merge_edge_pathogen_cells': False,
        'timelapse': False, 'timelapse_objects': ['cell'],
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'experiment': 'exp',
        'radial_dist': False, 'calculate_correlation': False,
        'manders_thresholds': [15, 85], 'homogeneity': False,
        'homogeneity_distances': [8, 16], 'distance_gaussian_sigma': 0,
        'strict_errors': False,
    }
    s.update(over)
    return s


def _field_with_a_straddling_pathogen(tmp_path):
    """A merged field whose pathogen overlaps two cells; returns its folder.

    The two nuclei sit well inside their own cells, so the unconditional
    nucleus/cell mask repair leaves the field alone and the pathogen join is
    the one that has to make the decision.
    """
    src = tmp_path / 'merged'
    src.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir(parents=True)

    y, x = 48, 96
    data = np.zeros((y, x, 7), np.uint16)
    data[..., 0] = 11
    data[..., 1] = 22
    cell = np.zeros((y, x), np.uint16)
    cell[4:44, 4:46] = 1
    cell[4:44, 50:92] = 2
    nucleus = np.zeros((y, x), np.uint16)
    nucleus[16:28, 14:26] = 1
    nucleus[16:28, 60:72] = 2
    pathogen = np.zeros((y, x), np.uint16)
    pathogen[8:14, 40:60] = 1          # reaches into both cells
    data[..., 4] = cell
    data[..., 5] = nucleus
    data[..., 6] = pathogen
    np.save(src / 'plate1_A01_f1.npy', data)
    return src


def _tables(db):
    if not db.is_file():
        return {}
    conn = sqlite3.connect(db)
    try:
        names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn)['name']
        return {
            name: pd.read_sql_query(f'SELECT COUNT(*) c FROM "{name}"',
                                    conn)['c'][0]
            for name in names
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# what the schema says a child row is
# ---------------------------------------------------------------------------

def test_the_child_tables_are_declared_one_row_per_object_label():
    """The join contract is downstream of this, not an independent opinion.

    If ``row_key_columns`` ever stops keying on ``object_label`` alone, the
    ``one_to_one`` joins below are the thing to revisit.
    """
    from spacr.schema import object_table_schema

    for table in ('nucleus', 'pathogen'):
        keys = object_table_schema(table).row_key_columns()
        assert keys[-1] == 'object_label'
        assert 'cell_id' not in keys, (
            f'{table} is keyed by object alone, so one object cannot hold two '
            f'cell_ids')


# ---------------------------------------------------------------------------
# the fan-out the relaxation was justified by does not reach the join
# ---------------------------------------------------------------------------

def test_get_components_lists_a_straddling_child_under_two_cells():
    """True of the raw masks -- which is not the same as true at the join."""
    cell, nucleus, pathogen = _straddling_field()
    nucleus_map, pathogen_map = M.get_components(cell, nucleus, pathogen)

    assert sorted(nucleus_map['cell_id'].tolist()) == [1, 2]
    assert nucleus_map['nucleus'].tolist() == [1, 1]
    assert sorted(pathogen_map['cell_id'].tolist()) == [1, 2]
    assert pathogen_map['pathogen'].tolist() == [1, 1]


def test_the_pipeline_resolves_a_straddling_child_before_measuring():
    """``_measure_crop_core`` repairs the masks first, so the join never sees it.

    This is the premise the ``one_to_many`` relaxation rested on, and it is
    false on the caller path: ``_merge_overlapping_objects`` runs on
    (nucleus, cell) unconditionally and on (pathogen, cell) under
    ``merge_edge_pathogen_cells``, and it resolves a straddling child to one
    cell -- here by merging the two touching cells into one label.
    """
    from spacr.utils import _merge_overlapping_objects

    cell, nucleus, pathogen = _straddling_field()
    nucleus, cell = _merge_overlapping_objects(mask1=nucleus, mask2=cell)
    pathogen, cell = _merge_overlapping_objects(mask1=pathogen, mask2=cell)

    nucleus_map, pathogen_map = M.get_components(cell, nucleus, pathogen)
    assert not nucleus_map['nucleus'].duplicated().any()
    assert not pathogen_map['pathogen'].duplicated().any()


# ---------------------------------------------------------------------------
# the ordinary field is unchanged
# ---------------------------------------------------------------------------

def test_a_nucleus_inside_exactly_one_cell_is_unchanged():
    """The common case produces exactly the rows it always produced."""
    cell = np.zeros((24, 48), dtype=np.uint16)
    cell[2:22, 2:24] = 1
    cell[2:22, 24:46] = 2
    nucleus = np.zeros((24, 48), dtype=np.uint16)
    nucleus[10:14, 8:16] = 1
    nucleus[10:14, 30:38] = 2
    pathogen = np.zeros((24, 48), dtype=np.uint16)

    _cell, nucleus_df, _p, _o, _c = M._morphological_measurements(
        cell, nucleus, pathogen, None, None,
        _settings(pathogen_mask_dim=None), zernike=False)

    assert len(nucleus_df) == 2
    assert nucleus_df['label'].tolist() == [1, 2]
    assert nucleus_df['nucleus_cell_id'].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# a fan-out that does reach the join stops there, loudly and completely
# ---------------------------------------------------------------------------

def test_a_shared_nucleus_stops_the_morphology_pass():
    """One nucleus, two cells: refused, because no ``cell_id`` is the answer."""
    cell, nucleus, pathogen = _straddling_field()

    with pytest.raises(pd.errors.MergeError) as excinfo:
        M._morphological_measurements(
            cell, nucleus, pathogen, None, None,
            _settings(pathogen_mask_dim=None), zernike=False)

    message = str(excinfo.value)
    assert 'nucleus' in message
    assert 'Nothing was written' in message


def test_a_shared_pathogen_stops_the_morphology_pass_and_names_the_setting():
    """The message has to be actionable: which labels, and what to change.

    ``_measure_crop_core`` catches its own exceptions and files the field as
    failed, so a bare "Merge keys are not unique in right dataset" is all the
    user would otherwise get.
    """
    cell, nucleus, pathogen = _straddling_field()

    with pytest.raises(pd.errors.MergeError) as excinfo:
        M._morphological_measurements(
            cell, nucleus, pathogen, None, None,
            _settings(nucleus_mask_dim=None), zernike=False)

    message = str(excinfo.value)
    assert 'pathogen' in message
    assert '[1, 2]' in message                     # the two cell_ids
    assert 'merge_edge_pathogen_cells' in message
    assert 'Nothing was written' in message


def test_a_duplicated_props_row_is_re_raised_without_the_cell_story():
    """A repeated label on the LEFT is a different fault with a different fix.

    ``regionprops_table`` emitting one label twice would mean every
    measurement of that object is counted twice. Blaming straddling cells for
    it would send the reader to the wrong place, so the handler stands aside.
    """
    props = pd.DataFrame({'label': [1, 1], 'area': [10.0, 10.0]})
    mapping = pd.DataFrame({'cell_id': [1], 'nucleus': [1]})

    with pytest.raises(pd.errors.MergeError) as excinfo:
        M._join_child_to_parent_cell(props, mapping, 'nucleus',
                                     remedy='should not appear')

    assert 'should not appear' not in str(excinfo.value)
    assert 'left dataset' in str(excinfo.value)


# ---------------------------------------------------------------------------
# THE regression: a refused field must be all-out, not half-in
# ---------------------------------------------------------------------------

def test_a_fanned_out_field_writes_no_table_at_all(tmp_path):
    """Atomicity. This is what ``one_to_many`` gave away.

    With the join relaxed, the pathogen fan-out survived the morphology pass
    and was not caught until ``_merge_and_save_to_database('pathogen', ...)``
    re-imposed ``one_to_one`` on ``object_label`` -- by which point the ``cell``
    and ``nucleus`` tables for this field held 2 rows each. The field was then
    half in ``measurements.db``: cells with no pathogen rows, which reads as
    uninfected rather than as a failure. Refusing in the morphology pass leaves
    nothing behind.
    """
    src = _field_with_a_straddling_pathogen(tmp_path)
    settings = _crop_settings(src, merge_edge_pathogen_cells=False)

    _index, _avg, cells, _figs = M._measure_crop_core(
        0, [], 'plate1_A01_f1.npy', settings)

    # 0 (a plain int) is the cross-process failure sentinel.
    assert cells == 0, 'the field was expected to fail'
    assert _tables(tmp_path / 'measurements' / 'measurements.db') == {}, (
        'a refused field left rows in measurements.db')


def test_the_same_field_measures_cleanly_once_the_masks_are_repaired(tmp_path):
    """And the remedy the message names actually works.

    Same field, ``merge_edge_pathogen_cells=True``: the straddle is resolved
    before the join, every table is written, and the pathogen appears once.
    """
    src = _field_with_a_straddling_pathogen(tmp_path)
    settings = _crop_settings(src, merge_edge_pathogen_cells=True)

    _index, _avg, cells, _figs = M._measure_crop_core(
        0, [], 'plate1_A01_f1.npy', settings)

    assert not isinstance(cells, int), 'the field failed inside _measure_crop_core'
    counts = _tables(tmp_path / 'measurements' / 'measurements.db')
    assert counts.get('pathogen') == 1
    assert counts.get('cell', 0) >= 1
    assert counts.get('nucleus', 0) >= 1


# ---------------------------------------------------------------------------
# _map_child_to_parent: why the organelle joins are one_to_one for a
# different reason
# ---------------------------------------------------------------------------

def test_the_child_to_parent_mapper_emits_exactly_one_row_per_child():
    """An organelle overlapping two cells is resolved to the one it overlaps most.

    So the organelle joins are ``one_to_one`` because the mapper cannot fan
    out at all, not merely because a fan-out would be illegal downstream.
    """
    parent = np.zeros((20, 40), dtype=np.uint16)
    parent[2:18, 2:20] = 1
    parent[2:18, 20:38] = 2
    child = np.zeros((20, 40), dtype=np.uint16)
    child[8:12, 14:26] = 1        # 6 px in cell 1, 6 px in cell 2
    child[4:6, 6:10] = 2          # wholly in cell 1

    mapping = M._map_child_to_parent(child, parent,
                                     child_name='organelle', parent_name='cell')

    assert mapping['organelle'].tolist() == [1, 2]
    assert not mapping['organelle'].duplicated().any()


def test_the_organelle_summary_counts_a_shared_organelle_once():
    """One row per organelle in, one parent assignment out, no double counting."""
    parent = np.zeros((20, 40), dtype=np.uint16)
    parent[2:18, 2:20] = 1
    parent[2:18, 20:38] = 2
    organelle = np.zeros((20, 40), dtype=np.uint16)
    organelle[8:12, 14:26] = 1
    channels = np.ones((20, 40, 1), dtype=np.uint16)

    summary = M._summarize_organelles_per_parent(
        organelle, parent, channels, parent_name='cell')

    assert sorted(summary['label'].tolist()) == [1, 2]
    # The organelle belongs to exactly one parent, so the counts sum to one.
    assert summary['organelle_count'].sum() == 1


# ---------------------------------------------------------------------------
# the per-channel widening in _measure_intensity_distance
# ---------------------------------------------------------------------------

def test_the_per_channel_distance_tables_join_one_row_per_cell():
    """Widening one table across channels, not relating two object types.

    Each channel's frame is built by walking the same ``np.unique(cell_mask)``,
    so ``one_to_one`` is exactly right and a repeat would mean a cell's
    distances were about to be averaged over duplicate rows.
    """
    cell = np.zeros((24, 48), dtype=np.uint16)
    cell[2:22, 2:24] = 1
    cell[2:22, 24:46] = 2
    nucleus = np.zeros((24, 48), dtype=np.uint16)
    nucleus[10:14, 8:16] = 1
    pathogen = np.zeros((24, 48), dtype=np.uint16)
    pathogen[4:8, 30:34] = 1
    channels = np.random.default_rng(0).integers(
        0, 500, size=(24, 48, 3)).astype(np.uint16)

    out = M._measure_intensity_distance(
        cell, nucleus, pathogen, channels, _settings())

    assert out['label'].tolist() == [1, 2]
    for ch in range(3):
        assert f'cell_channel_{ch}_distance_to_nucleus' in out.columns
        assert f'cell_channel_{ch}_distance_to_pathogen' in out.columns
