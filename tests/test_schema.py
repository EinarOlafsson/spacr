"""``spacr.schema`` — the one definition of what a spaCR key is.

The reproduction
----------------
``_map_wells`` was reimplemented five times and the copies disagreed about
malformed wells. This file proves it against **real databases built by the
real writers** (``utils._merge_and_save_to_database`` and
``utils.filepaths_to_database``), never a hand-built schema:

* :func:`test_repro_object_table_and_png_list_disagree_on_a_1536_well` — one
  field, both writers, two different identities, and a join that returns
  **0 rows out of 2 x 2**.
* :func:`test_repro_three_sites_collapse_onto_one_prcf` — three ImageXpress
  sites go in, **one** ``prcf`` comes out.
* :func:`test_repro_a_whole_timelapse_collapses_onto_t0` — three timepoints
  go in, **one** ``timeID`` comes out.

Then the canonical module is tested against every malformed input, and pinned
against the legacy implementation on every input the legacy one gets right,
so the migration is provably a repair and not a change of contract.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pandas as pd
import pytest

from spacr import schema as S
from spacr.schema import (KEY_SEPARATOR, KeyParseError, SchemaError,
                          WellParseError)


# ---------------------------------------------------------------------------
# helpers -- databases are built by the real writers, only read here
# ---------------------------------------------------------------------------

def _read(db, table):
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    finally:
        conn.close()


def _write_object_table(root, stem, labels=(1,), table='cell', timelapse=False):
    """Write one field's worth of rows with the real measurement writer."""
    from spacr.utils import _merge_and_save_to_database

    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    morph = pd.DataFrame({'label': list(labels),
                          f'{table}_area': [10.0 + i for i in range(len(labels))]})
    intensity = pd.DataFrame({'label': list(labels),
                              f'{table}_channel_0_mean_intensity':
                                  [1.0 + i for i in range(len(labels))]})
    _merge_and_save_to_database(morph, intensity, table, root, stem, 'exp',
                                timelapse=timelapse)
    return os.path.join(root, 'measurements', 'measurements.db')


def _write_png_list(root, names, crop_mode='cell', timelapse=False):
    """Write png_list rows with the real crop writer."""
    from spacr.utils import filepaths_to_database

    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    paths = [os.path.join(root, f'{crop_mode}_png', n) for n in names]
    filepaths_to_database(paths, {'timelapse': timelapse}, root, crop_mode)
    return os.path.join(root, 'measurements', 'measurements.db')


# ===========================================================================
# REPRODUCTION -- against real databases
# ===========================================================================

def test_repro_object_table_and_png_list_disagree_on_a_1536_well(tmp_path):
    """One field, two writers, two identities, and a join that finds nothing.

    ``AA01`` is an ordinary well of a 1536 plate (32 rows: A..Z, AA..AF).
    ``_map_wells`` raises inside and returns ``'error'`` in every slot --
    destroying the *plate* along with the well -- while ``_map_wells_png``
    silently reads ``well[1:] == 'A01'`` through ``_safe_int_convert`` and
    gets column ``0``.
    """
    root = str(tmp_path)
    stem = 'plate1_AA01_1'
    db = _write_object_table(root, stem, labels=(1, 2))
    _write_png_list(root, [f'{stem}_1.png', f'{stem}_2.png'])

    cell = _read(db, 'cell')
    png = _read(db, 'png_list')
    assert len(cell) == 2 and len(png) == 2

    # MEASURED: the object table lost every key, including the plate.
    assert cell[['plateID', 'rowID', 'columnID', 'fieldID', 'prcf']] \
        .drop_duplicates().values.tolist() == [['error'] * 5]
    # MEASURED: png_list kept the plate but invented column 0.
    assert png[['plateID', 'rowID', 'columnID', 'fieldID']] \
        .drop_duplicates().values.tolist() == [['plate1', 'r1', 'c0', 'f1']]

    # MEASURED: the join between them is empty. 2 x 2 rows -> 0.
    joined = png.merge(cell, on=['plateID', 'rowID', 'columnID', 'fieldID'])
    assert len(joined) == 0

    # The canonical parser gives both halves the same, correct identity,
    # and the same join would then match every row.
    field = S.parse_field_stem(stem)
    obj = S.parse_object_stem(f'{stem}_1.png')
    assert field.to_dict() == {'plateID': 'plate1', 'rowID': 'r27',
                               'columnID': 'c1', 'fieldID': 'f1'}
    assert obj.field == field
    assert field.well == 'AA01'


def test_repro_three_sites_collapse_onto_one_prcf(tmp_path):
    """Three ImageXpress sites become one field, silently.

    ``_safe_int_convert('s1')`` is ``0``, so ``s1``, ``s2`` and ``s3`` all
    become ``f0`` and share a ``prcf``. Object 1 of each becomes the same
    ``prcfo``: three different cells with one identity.
    """
    root = str(tmp_path)
    for site in ('s1', 's2', 's3'):
        db = _write_object_table(root, f'plate1_A01_{site}')

    cell = _read(db, 'cell')
    # MEASURED: three rows from three distinct files ...
    assert sorted(cell['file_name']) == ['plate1_A01_s1', 'plate1_A01_s2',
                                         'plate1_A01_s3']
    # ... one field id and one prcf between them.
    assert cell['fieldID'].tolist() == ['f0', 'f0', 'f0']
    assert cell['prcf'].nunique() == 1
    assert cell['prcf'].iloc[0] == 'plate1_r1_c1_f0'

    # And the objects collide: same prcf, same label -> same prcfo.
    assert len({S.compose_prcfo('plate1', 'r1', 'c1', 'f0', 1)}) == 1

    # Canonically the three sites stay three fields.
    prcfs = {S.parse_field_stem(f'plate1_A01_{s}').prcf
             for s in ('s1', 's2', 's3')}
    assert prcfs == {'plate1_r1_c1_f1', 'plate1_r1_c1_f2', 'plate1_r1_c1_f3'}


def test_repro_a_whole_timelapse_collapses_onto_t0(tmp_path):
    """Three timepoints, one ``timeID``. ``_safe_int_convert('T0001') == 0``."""
    root = str(tmp_path)
    for stamp in ('T0001', 'T0002', 'T0003'):
        db = _write_object_table(root, f'plate1_A01_1_{stamp}',
                                 timelapse=True)

    cell = _read(db, 'cell')
    assert sorted(cell['file_name']) == ['plate1_A01_1_T0001',
                                         'plate1_A01_1_T0002',
                                         'plate1_A01_1_T0003']
    # MEASURED: 3 distinct timepoints on disk, 1 in the database.
    assert cell['timeID'].tolist() == ['t0', 't0', 't0']
    assert cell['timeID'].nunique() == 1
    assert cell['prcf'].nunique() == 1

    # Canonically the timelapse survives.
    times = [S.parse_field_stem(f'plate1_A01_1_{s}', timelapse=True).timeID
             for s in ('T0001', 'T0002', 'T0003')]
    assert times == ['t1', 't2', 't3']


def test_repro_an_unparseable_field_is_written_into_the_database_as_error(
        tmp_path):
    """A lowercase well makes the *whole* identity the string ``'error'``.

    Those five ``'error'`` strings are not a sentinel any reader checks --
    they are appended to the table as if they were a plate, a row, a column
    and a field, so the rows are unfindable and pollute every group-by.
    """
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_a01_1', labels=(1, 2, 3))
    cell = _read(db, 'cell')

    assert len(cell) == 3
    assert set(cell['plateID']) == {'error'}
    assert set(cell['prcf']) == {'error'}
    # Three real objects, all reachable only under a plate called "error".
    assert sorted(cell['object_label']) == [1, 2, 3]

    # Canonically a lowercase well is just a well.
    assert S.parse_field_stem('plate1_a01_1').prcf == 'plate1_r1_c1_f1'


def test_repro_png_list_reads_the_field_token_as_the_object_too(tmp_path):
    """``plate1_A01_5.png`` becomes field 5 *and* object 5.

    ``_map_wells_png`` takes the field from ``parts[2]`` and the object from
    ``parts[-1]``; in a three-part name those are one token.
    """
    root = str(tmp_path)
    db = _write_png_list(root, ['plate1_A01_5.png'])
    png = _read(db, 'png_list')
    row = png.iloc[0]

    # MEASURED: one token, used twice.
    assert row['fieldID'] == 'f5'
    assert row['cell_id'] == 'o5'
    assert row['prcfo'] == 'plate1_r1_c1_f5_o5'

    # Canonically a name that short is not an object name at all.
    with pytest.raises(KeyParseError, match='plate_well_field_object'):
        S.parse_object_stem('plate1_A01_5.png')


def test_repro_the_five_implementations_disagree():
    """The disagreement itself, as a table. Every row here is a real answer.

    ================  ==================  ======================  ====================
    well              ``_map_wells``      ``_map_wells_png``      ``align._well_ids``
    ================  ==================  ======================  ====================
    ``'A'``           ``error``           ``r1,c0``               ``A,A``
    ``'a01'``         ``error``           ``error``               ``r1,c1``
    ``'AA01'``        ``error``           ``r1,c0``               ``AA01,AA01``
    ================  ==================  ======================  ====================

    This test is also the **migration tripwire**. It pins the pre-migration
    behaviour of every copy, so the first call site that is switched over to
    :mod:`spacr.schema` makes it fail — which is the signal to delete that
    copy's row here rather than to loosen the assertion.
    """
    from spacr.align import _well_ids as align_well_ids
    from spacr.convert import _well_ids as convert_well_ids
    from spacr.resume import field_identity
    from spacr.utils import _map_wells, _map_wells_png

    def utils_stack(well):
        out = _map_wells(f'plate1_{well}_3')
        return out[1], out[2]

    def utils_png(well):
        out = _map_wells_png(f'plate1_{well}_3_7.png')
        return out[1], out[2]

    def resume_ids(well):
        out = field_identity(f'plate1_{well}_3')
        return out['rowID'], out['columnID']

    # 'A' -- letters, no column.
    assert utils_stack('A') == ('error', 'error')
    assert utils_png('A') == ('r1', 'c0')
    assert resume_ids('A') == ('r1', 'c0')
    assert align_well_ids('A') == ('A', 'A')
    assert convert_well_ids('A') == ('A', 'A')
    with pytest.raises(WellParseError, match='no column'):
        S.parse_well('A')

    # lowercase -- two of five raise into 'error', three uppercase it.
    assert utils_stack('a01') == ('error', 'error')
    assert utils_png('a01') == ('error', 'error')
    assert resume_ids('a01') == ('r1', 'c1')
    assert align_well_ids('a01') == ('r1', 'c1')
    assert S.parse_well('a01') == ('r1', 'c1')

    # a 1536-plate row -- three different answers.
    assert utils_stack('AA01') == ('error', 'error')
    assert utils_png('AA01') == ('r1', 'c0')
    assert resume_ids('AA01') == ('r1', 'c0')
    assert align_well_ids('AA01') == ('AA01', 'AA01')
    assert S.parse_well('AA01') == ('r27', 'c1')

    # ... and the cases they all agree on, which the canonical one keeps.
    for well in ('A01', 'A1', 'P24', 'Z01', 'A48'):
        assert utils_stack(well) == utils_png(well) == resume_ids(well) \
            == align_well_ids(well) == convert_well_ids(well) \
            == S.parse_well(well)


# ===========================================================================
# the canonical parser is a strict repair of the legacy one
# ===========================================================================

_LEGACY_GOOD_WELLS = ['A01', 'A1', 'A12', 'B03', 'H12', 'P24', 'Z01', 'A48',
                      'A0', 'A00', 'D6']


@pytest.mark.parametrize('well', _LEGACY_GOOD_WELLS)
@pytest.mark.parametrize('field', ['1', '01', '007', '12'])
def test_canonical_agrees_with_legacy_on_every_name_legacy_gets_right(well,
                                                                      field):
    """No contract change: where ``_map_wells`` works, the answers are equal."""
    from spacr.utils import _map_wells

    name = f'plate1_{well}_{field}'
    plate, row, column, fld, prcf = _map_wells(name)
    assert (plate, row, column, fld) != ('error',) * 4  # legacy did work

    parsed = S.parse_field_stem(name)
    assert (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID) \
        == (plate, row, column, fld)
    assert parsed.prcf == prcf


@pytest.mark.parametrize('well', _LEGACY_GOOD_WELLS)
def test_canonical_agrees_with_legacy_timelapse_ordering(well):
    """The timepoint goes after the field, as it does on disk."""
    from spacr.utils import _map_wells

    name = f'plate1_{well}_2_5'
    plate, row, column, fld, timeid, prcf = _map_wells(name, timelapse=True)
    parsed = S.parse_field_stem(name, timelapse=True)
    assert (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID,
            parsed.timeID) == (plate, row, column, fld, timeid)
    assert parsed.prcf == prcf
    assert parsed.prcf.endswith('_f2_t5')


@pytest.mark.parametrize('well', _LEGACY_GOOD_WELLS)
def test_canonical_prcfo_agrees_with_legacy_png(well):
    from spacr.utils import _map_wells_png

    name = f'plate1_{well}_3_17.png'
    plate, row, column, fld, prcfo, obj = _map_wells_png(name)
    parsed = S.parse_object_stem(name)
    assert (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID,
            parsed.objectID) == (plate, row, column, fld, obj)
    assert parsed.prcfo == prcfo


def test_repro_the_two_safe_int_copies_disagree_on_none():
    """``_safe_int_convert`` and ``resume._safe_int`` are the same bug, twice.

    Both return ``0``; only one of them catches ``TypeError``. So ``None``
    is a crash in one code path and field ``f0`` in the other.
    """
    from spacr.resume import _safe_int
    from spacr.utils import _safe_int_convert

    assert _safe_int(None) == 0
    with pytest.raises(TypeError):
        _safe_int_convert(None)

    # Neither ever says "there is no number here". schema does.
    assert _safe_int('x') == 0 and _safe_int_convert('x') == 0
    assert S.parse_int_token('x') is None


def test_repro_the_ml_regex_only_understands_rows_a_to_p():
    """``ml.py``'s inline copy silently leaves rows past P unchanged."""
    import re

    pattern = re.compile(r'^(?i:plate)\d+_([A-Pa-p])(\d+)_')

    assert pattern.match('PLATE1_A14_1_1_111.png').groups() == ('A', '14')
    # MEASURED: a 384-plate row past P, and any plate not named "plate<n>",
    # do not match -- rowID/columnID are then left as they were.
    assert pattern.match('PLATE1_Q14_1_1_111.png') is None
    assert pattern.match('PLATE1_AA14_1_1_111.png') is None
    assert pattern.match('exp3_A14_1_1_111.png') is None

    # schema parses all four.
    for stem, expected in [('PLATE1_A14_1_1', ('r1', 'c14')),
                           ('PLATE1_Q14_1_1', ('r17', 'c14')),
                           ('PLATE1_AA14_1_1', ('r27', 'c14')),
                           ('exp3_A14_1_1', ('r1', 'c14'))]:
        parsed = S.parse_field_stem(stem)
        assert (parsed.rowID, parsed.columnID) == expected, stem


def test_legacy_helpers_reproduce_utils_exactly():
    """The bug-compatible copies really are bug compatible."""
    from spacr.utils import _map_wells

    for name in ['plate1_A01_1', 'plate1_a01_1', 'plate1_AA01_1',
                 'plate1_A_1', 'plate1__1', 'plate1_A01_x', 'garbage',
                 'plate1_A01', 'plate1_12_3', 'plate1_ A01 _1']:
        assert S.legacy_map_wells(name) == _map_wells(name), name
        assert S.legacy_map_wells(name + '_9', timelapse=True) == \
            _map_wells(name + '_9', timelapse=True), name

    assert S.legacy_safe_int_convert('x') == 0
    assert S.legacy_safe_int_convert('x', default=7) == 7
    assert S.legacy_safe_int_convert('5') == 5
    with pytest.raises(TypeError):
        S.legacy_safe_int_convert(None)


# ===========================================================================
# parse_int_token -- never 0
# ===========================================================================

@pytest.mark.parametrize('token,expected', [
    ('3', 3), ('003', 3), (' 3 ', 3), ('0', 0), ('-2', -2), ('+4', 4),
    (3, 3), (3.0, 3), (b'7', 7), (bytearray(b'8'), 8),
    ('s3', 3), ('S3', 3), ('T0001', 1), ('F003', 3), ('Z01', 1), ('ch2', 2),
])
def test_parse_int_token_parses(token, expected):
    assert S.parse_int_token(token) == expected


@pytest.mark.parametrize('token', [
    None, '', '   ', 'x', 'xy', 'abc', '1a', 'a1b', '3.7', '1e3', '--1',
    'field', float('nan'), float('inf'), float('-inf'), 2.5,
    True, False, [], {}, ('1',), b'\xff\xfe',
])
def test_parse_int_token_returns_none_never_zero(token):
    """The whole point: the honest answer to "no number here" is ``None``."""
    result = S.parse_int_token(token)
    assert result is None, f'{token!r} -> {result!r}'
    assert result != 0


def test_parse_int_token_prefix_can_be_switched_off():
    assert S.parse_int_token('s3', allow_prefix=False) is None
    assert S.parse_int_token('3', allow_prefix=False) == 3


def test_a_three_letter_prefix_is_not_stripped():
    """Only one or two letters. ``'abc12'`` is not field 12."""
    assert S.parse_int_token('abc12') is None
    assert S.parse_int_token('ab12') == 12


# ===========================================================================
# the graded failure policy
# ===========================================================================

def test_an_unparseable_field_keeps_its_token_instead_of_becoming_f0():
    assert S.field_id('xy') == 'fxy'
    assert S.field_id('s3') == 'f3'
    assert S.field_id('3') == 'f3'
    # Distinct tokens stay distinct -- this is what stops the collapse.
    assert len({S.field_id(t) for t in ('xy', 'zz', 'qq')}) == 3


def test_an_empty_field_raises_because_it_is_not_an_identity():
    for token in ('', '   ', None):
        with pytest.raises(KeyParseError, match='empty'):
            S.field_id(token)


def test_strict_promotes_an_unparseable_token_to_an_error():
    assert S.field_id('xy') == 'fxy'
    with pytest.raises(KeyParseError, match='holds no integer'):
        S.field_id('xy', strict=True)
    with pytest.raises(KeyParseError, match='holds no integer'):
        S.time_id('later', strict=True)
    with pytest.raises(KeyParseError, match='holds no integer'):
        S.object_id('xx', strict=True)


def test_a_preserved_token_cannot_break_the_key_separator():
    """A token holding ``_`` would silently add a prcf component."""
    assert S.field_id('a_b') == 'fa-b'
    assert S.compose_prcf('plate1', 'r1', 'c1', 'a_b').count(KEY_SEPARATOR) == 3


def test_object_id_never_invents_o0():
    assert S.object_id(41) == 'o41'
    assert S.object_id('o41') == 'o41'
    assert S.object_id('xx') == 'oxx'
    with pytest.raises(KeyParseError, match='empty'):
        S.object_id('')


# ===========================================================================
# wells -- every malformed input asked for
# ===========================================================================

@pytest.mark.parametrize('well,expected', [
    ('A1', ('r1', 'c1')),
    ('A01', ('r1', 'c1')),
    ('a01', ('r1', 'c1')),
    ('  A01  ', ('r1', 'c1')),
    ('A-01', ('r1', 'c1')),
    ('A_01', ('r1', 'c1')),
    ('A 1', ('r1', 'c1')),
    ('AA01', ('r27', 'c1')),
    ('aa01', ('r27', 'c1')),
    ('AF48', ('r32', 'c48')),
    ('Z1', ('r26', 'c1')),
    ('P24', ('r16', 'c24')),
    ('A25', ('r1', 'c25')),      # past 24 columns: legal on a 1536 plate
    ('A0', ('r1', 'c0')),
    ('B0003', ('r2', 'c3')),
])
def test_parse_well(well, expected):
    assert S.parse_well(well) == expected


@pytest.mark.parametrize('well', ['', '   ', None])
def test_an_empty_well_raises(well):
    with pytest.raises(WellParseError, match='empty'):
        S.parse_well(well)


@pytest.mark.parametrize('well', ['A', 'a', 'AA', 'AF'])
def test_a_well_with_no_column_raises_instead_of_becoming_c0(well):
    with pytest.raises(WellParseError, match='no column'):
        S.parse_well(well)


def test_well_shape_agrees_with_plate_qc():
    """Both modules must answer "is this a well?" the same way."""
    from spacr.plate_qc import _parse_well_label

    for well in ['A01', 'A1', 'H12', 'P24', 'AA01', 'AF48', 'a01', 'A-01',
                 'A 1', ' B7 ']:
        qc = _parse_well_label(well)
        assert qc is not None, well
        assert S.parse_well(well) == (f'r{qc[0]}', f'c{qc[1]}'), well

    for not_a_well in ['A', '12', '1A', 'ABC12', '']:
        assert _parse_well_label(not_a_well) is None, not_a_well
        # schema does not silently invent a row/column for any of them.
        try:
            row, column = S.parse_well(not_a_well)
        except WellParseError:
            continue
        assert S.is_positional_pair(row, column), not_a_well


def test_a_bare_number_well_passes_through_as_all_five_copies_do():
    assert S.parse_well('12') == ('12', '12')
    assert S.is_positional_well('12') is True
    assert S.is_positional_well('A01') is False
    with pytest.raises(WellParseError, match='not knowable'):
        S.parse_well('12', strict=True)


def test_an_int_well_is_treated_as_a_positional_well():
    assert S.parse_well(7) == ('7', '7')


@pytest.mark.parametrize('well', ['1A', 'A1B', '!!', 'A-B'])
def test_a_shapeless_well_falls_through_rather_than_guessing(well):
    assert S.parse_well(well) == (well, well)


def test_row_letters_round_trip_past_z():
    for index, letters in [(1, 'A'), (26, 'Z'), (27, 'AA'), (32, 'AF'),
                           (52, 'AZ'), (53, 'BA'), (702, 'ZZ'), (703, 'AAA')]:
        assert S.row_index_from_letters(letters) == index
        assert S.letters_from_row_index(index) == letters


def test_row_letters_agree_with_plate_qc():
    """``plate_qc`` already got multi-letter rows right; do not fork from it."""
    from spacr.plate_qc import _alpha_to_index, _index_to_alpha
    from spacr.plate_qc import well_id as qc_well_id

    for letters in ['A', 'H', 'P', 'Z', 'AA', 'AF', 'AZ', 'BA', 'ZZ']:
        assert S.row_index_from_letters(letters) == _alpha_to_index(letters)
    for index in range(1, 200):
        assert S.letters_from_row_index(index) == _index_to_alpha(index)
        assert S.well_id(index, 7) == qc_well_id(index, 7)


def test_row_index_from_letters_rejects_non_letters():
    for value in ['', '  ', '1', 'A1', '-', None]:
        assert S.row_index_from_letters(value) is None


def test_letters_from_row_index_rejects_a_non_row():
    for value in [0, -1, 'x', None]:
        with pytest.raises(KeyParseError):
            S.letters_from_row_index(value)


def test_repro_the_inverse_direction_is_broken_past_row_z():
    """Going ``rowID -> well`` uses ``chr(ord('A') + n - 1)`` in two places.

    ``crops.py`` rebuilds a merged-stack path from a row id that way and
    ``utils._convert_cq1_well_id`` builds a CQ1 well from a linear index
    that way. Both walk straight off the end of the alphabet.
    """
    from spacr.utils import _convert_cq1_well_id

    # crops.py:2244, verbatim.
    def crops_style(rowid):
        return chr(ord('A') + int(str(rowid).lstrip('r')) - 1)

    assert crops_style('r26') == 'Z'
    assert crops_style('r27') == '['            # MEASURED: not a row letter
    assert crops_style('r32') == '`'
    assert S.letters_from_row_index(27) == 'AA'
    assert S.letters_from_row_index(32) == 'AF'

    # utils._convert_cq1_well_id: 24 columns hardcoded, so a 1536 plate
    # walks past Z after well 624.
    assert _convert_cq1_well_id(1) == 'A01'
    assert _convert_cq1_well_id(384) == 'P24'
    assert _convert_cq1_well_id(1536) == '\x8024'   # MEASURED

    # The canonical inverse stays inside the alphabet for every 1536 well.
    for index in range(1, 1537):
        row, column = divmod(index - 1, 48)
        name = S.well_id(row + 1, column + 1)
        assert name.isalnum() and S.parse_well(name) == (f'r{row + 1}',
                                                         f'c{column + 1}')


def test_well_id_is_the_inverse_of_parse_well():
    for well in ['A01', 'B07', 'H12', 'P24', 'AA01', 'AF48']:
        row, column = S.parse_well(well)
        assert S.well_id(row, column) == well


def test_well_id_refuses_a_positional_well():
    """``('12','12')`` must not render as the well ``'L12'``."""
    row, column = S.parse_well('12')
    assert S.is_positional_pair(row, column) is True
    with pytest.raises(KeyParseError, match='passthrough'):
        S.well_id(row, column)


@pytest.mark.parametrize('row,column', [
    ('rubbish', 'c1'), ('r0', 'c1'), ('r1', 'nonsense'), ('r1', 'c0'),
    ('r1', None), (None, 'c1'),
])
def test_well_id_refuses_an_unusable_index(row, column):
    with pytest.raises(KeyParseError, match='cannot build a well name'):
        S.well_id(row, column)


def test_fieldid_positional_flags_the_passthrough():
    assert S.FieldID.build('p', well='12', field=1).positional is True
    assert S.FieldID.build('p', well='A01', field=1).positional is False


def test_is_within_plate_format_is_false_for_an_unparseable_position():
    assert S.is_within_plate_format('rubbish', 'c1', 96) is False
    assert S.is_within_plate_format('r1', 'rubbish', 96) is False


def test_parse_prcf_rejects_a_key_with_no_plate():
    with pytest.raises(KeyParseError, match='no plate'):
        S.parse_prcf('r1_c1_f2_t3')


def test_is_positional_pair_does_not_fire_on_a_real_position():
    assert S.is_positional_pair('r1', 'c1') is False
    assert S.is_positional_pair('r3', 'c3') is False
    assert S.is_positional_pair('12', '13') is False
    assert S.is_positional_pair('', '') is False
    assert S.is_positional_pair('A01', 'A01') is True
    # Bare integer indices are an index API call, never a passthrough.
    assert S.is_positional_pair(1, 1) is False
    assert S.well_id(1, 1) == 'A01'


# ===========================================================================
# ids and their inverses
# ===========================================================================

def test_ids_round_trip():
    assert S.row_id(3) == 'r3' and S.row_index('r3') == 3
    assert S.row_id('r3') == 'r3'
    assert S.row_id('C') == 'r3' and S.row_index('C') == 3
    assert S.column_id('012') == 'c12' and S.column_index('c12') == 12
    assert S.field_id('F003') == 'f3' and S.field_index('f3') == 3
    assert S.time_id('T0007') == 't7' and S.time_index('t7') == 7
    assert S.object_id(41) == 'o41' and S.object_index('o41') == 41


def test_indices_of_an_unparseable_id_are_none_not_zero():
    assert S.field_index('fxy') is None
    assert S.row_index('rubbish') is None
    assert S.column_index(None) is None
    assert S.object_index('oxx') is None


def test_row_id_does_not_double_prefix():
    assert S.row_id('r12') == 'r12'
    assert S.column_id('c12') == 'c12'
    assert S.field_id('f12') == 'f12'
    assert S.time_id('t12') == 't12'


def test_strip_prefix():
    assert S.strip_prefix('r12', 'r') == '12'
    assert S.strip_prefix('12', 'r') == '12'
    assert S.strip_prefix('R12', 'r') == '12'
    assert S.strip_prefix(' c7 ', 'c') == '7'


# ===========================================================================
# composition and its inverse
# ===========================================================================

def test_compose_matches_what_the_writers_put_on_disk(tmp_path):
    """Pinned against a real database, not against itself."""
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_B03_2', labels=(9,))
    _write_png_list(root, ['plate1_B03_2_9.png'])

    cell = _read(db, 'cell').iloc[0]
    png = _read(db, 'png_list').iloc[0]

    assert S.compose_prcf('plate1', 'B', 3, 2) == cell['prcf'] \
        == 'plate1_r2_c3_f2'
    assert S.compose_prcfo('plate1', 'B', 3, 2, 9) == png['prcfo'] \
        == 'plate1_r2_c3_f2_o9'
    assert S.compose_prc('plate1', 'B', 3) == 'plate1_r2_c3'


def test_compose_matches_the_timelapse_database(tmp_path):
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_A01_1_3', labels=(17,),
                             timelapse=True)
    _write_png_list(root, ['plate1_A01_1_3_17.png'], timelapse=True)

    cell = _read(db, 'cell').iloc[0]
    png = _read(db, 'png_list').iloc[0]
    assert S.compose_prcf('plate1', 'A', 1, 1, time=3) == cell['prcf']
    assert S.compose_prcfo('plate1', 'A', 1, 1, 17, time=3) == png['prcfo']
    assert cell['timeID'] == png['timeID'] == 't3'


def test_prcfo_matches_the_composition_io_uses(tmp_path):
    """``io._read_and_join_tables`` builds ``prcf + '_' + 'o' + label``."""
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_C05_4', labels=(23,))
    cell = _read(db, 'cell').iloc[0]
    io_style = cell['prcf'] + '_' + 'o' + str(int(cell['object_label']))
    assert S.compose_prcfo('plate1', 'C', 5, 4, 23) == io_style


def test_a_plate_containing_the_separator_is_refused():
    with pytest.raises(KeyParseError, match='key separator'):
        S.compose_prcf('plate_1', 'r1', 'c1', 'f1')
    with pytest.raises(KeyParseError, match='empty plate'):
        S.compose_prcf('', 'r1', 'c1', 'f1')


def test_parse_prcf_round_trips_both_shapes():
    for text in ['plate1_r1_c1_f2', 'plate1_r1_c1_f2_t3',
                 'PLATE-4_r16_c24_f9']:
        assert S.parse_prcf(text).prcf == text


def test_parse_prcfo_round_trips_both_shapes():
    for text in ['plate1_r1_c1_f2_o7', 'plate1_r1_c1_f2_t3_o7']:
        assert S.parse_prcfo(text).prcfo == text


def test_parse_prcf_reads_right_to_left_so_the_timepoint_is_unambiguous():
    """The bug this avoids: ``ml.py`` splits left to right into 5 columns."""
    no_time = S.parse_prcf('plate1_r1_c1_f2')
    with_time = S.parse_prcf('plate1_r1_c1_f2_t3')
    assert no_time.timeID is None
    assert with_time.timeID == 't3'
    assert no_time.fieldID == with_time.fieldID == 'f2'
    # A left-to-right split of the timelapse key puts 't3' in the object slot.
    assert 'plate1_r1_c1_f2_t3_o7'.split('_')[4] == 't3'
    assert S.parse_prcfo('plate1_r1_c1_f2_t3_o7').objectID == 'o7'


@pytest.mark.parametrize('text', ['', 'plate1', 'plate1_r1_c1',
                                  'plate1_r1_c1_x2'])
def test_parse_prcf_rejects_a_non_prcf(text):
    with pytest.raises(KeyParseError):
        S.parse_prcf(text)


@pytest.mark.parametrize('text', ['plate1_r1_c1_f2', 'plate1_r1_c1_f2_x7'])
def test_parse_prcfo_rejects_a_non_prcfo(text):
    with pytest.raises(KeyParseError):
        S.parse_prcfo(text)


def test_parse_prcf_keeps_an_underscored_plate_together():
    """Right-to-left parsing survives what left-to-right cannot."""
    parsed = S.parse_prcf('my_plate_r1_c1_f2')
    assert parsed.plateID == 'my_plate'
    assert parsed.fieldID == 'f2'


# ===========================================================================
# filename parsing
# ===========================================================================

def test_parse_field_stem_accepts_a_path_and_an_extension():
    for name in ['plate1_A01_3', 'plate1_A01_3.npy',
                 '/data/merged/plate1_A01_3.npy',
                 'C:/x/plate1_A01_3.tif']:
        assert S.parse_field_stem(name).prcf == 'plate1_r1_c1_f3'


@pytest.mark.parametrize('name', ['garbage', 'plate1_A01', 'plate1', ''])
def test_parse_field_stem_refuses_a_name_with_too_few_parts(name):
    with pytest.raises(KeyParseError, match='cannot identify a field'):
        S.parse_field_stem(name)


def test_parse_field_stem_timelapse_needs_the_timepoint():
    with pytest.raises(KeyParseError, match='plate_well_field_time'):
        S.parse_field_stem('plate1_A01_3', timelapse=True)
    assert S.parse_field_stem('plate1_A01_3_4', timelapse=True).timeID == 't4'


def test_parse_object_stem_takes_the_object_from_the_end():
    obj = S.parse_object_stem('plate1_A01_3_17.png')
    assert obj.objectID == 'o17'
    assert obj.prcfo == 'plate1_r1_c1_f3_o17'
    assert obj.field.prcf == 'plate1_r1_c1_f3'


def test_parse_object_stem_timelapse():
    obj = S.parse_object_stem('plate1_A01_3_4_17.png', timelapse=True)
    assert obj.timeID == 't4' and obj.objectID == 'o17'
    assert obj.prcfo == 'plate1_r1_c1_f3_t4_o17'


@pytest.mark.parametrize('name', ['plate1_A01_3.png', 'garbage.png'])
def test_parse_object_stem_refuses_a_short_name(name):
    with pytest.raises(KeyParseError):
        S.parse_object_stem(name)


def test_strict_parsing_is_available_as_a_preflight():
    """What a CLI would run before a ten-hour job."""
    assert S.parse_field_stem('plate1_A01_s3', strict=True).fieldID == 'f3'
    with pytest.raises(KeyParseError):
        S.parse_field_stem('plate1_A01_xy', strict=True)
    with pytest.raises(WellParseError):
        S.parse_field_stem('plate1_12_3', strict=True)
    # ... and without strict, both keep going.
    assert S.parse_field_stem('plate1_A01_xy').fieldID == 'fxy'
    assert S.parse_field_stem('plate1_12_3').rowID == '12'


# ===========================================================================
# FieldID / ObjectID
# ===========================================================================

def test_fieldid_build_from_a_well_or_from_row_and_column():
    a = S.FieldID.build('plate1', well='B03', field=2)
    b = S.FieldID.build('plate1', row='r2', column=3, field='s2')
    assert a == b
    assert a.prcf == 'plate1_r2_c3_f2'
    assert a.prc == 'plate1_r2_c3'
    assert a.well == 'B03'


def test_fieldid_build_needs_a_position():
    with pytest.raises(KeyParseError, match='needs either a well'):
        S.FieldID.build('plate1', field=1)


def test_fieldid_to_dict_matches_the_table_columns(tmp_path):
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_D06_7', labels=(1,))
    cell = _read(db, 'cell').iloc[0]

    parsed = S.parse_field_stem('plate1_D06_7').to_dict(include_prcf=True)
    assert parsed['plateID'] == cell['plateID']
    assert parsed['rowID'] == cell['rowID']
    assert parsed['columnID'] == cell['columnID']
    assert parsed['fieldID'] == cell['fieldID']
    assert parsed['prcf'] == cell['prcf']
    assert set(S.FIELD_KEY_COLUMNS) <= set(cell.index)


def test_fieldid_without_a_timepoint_omits_the_key():
    assert 'timeID' not in S.FieldID.build('p', well='A01', field=1).to_dict()
    assert 'timeID' in S.FieldID.build('p', well='A01', field=1,
                                       time=2).to_dict()


def test_fieldid_well_is_none_for_a_positional_well():
    assert S.FieldID.build('p', well='12', field=1).well is None


def test_objectid_derives_its_field():
    obj = S.parse_object_stem('plate1_A01_3_4_17.png', timelapse=True)
    assert obj.field == S.parse_field_stem('plate1_A01_3_4', timelapse=True)
    assert obj.to_dict()['prcfo'] == obj.prcfo


def test_ids_are_hashable_and_comparable():
    a = S.FieldID.build('p', well='A01', field=1)
    b = S.FieldID.build('p', well='a1', field='001')
    assert a == b and len({a, b}) == 1


# ===========================================================================
# column names
# ===========================================================================

def test_every_db_column_rename_utils_knows_is_here_too():
    """schema is a superset of the migration list already in utils."""
    from spacr.utils import DB_COLUMN_RENAMES

    for old, new in DB_COLUMN_RENAMES.items():
        assert S.canonical_column_name(old) == new, old


def test_canonical_column_name_is_case_insensitive_and_leaves_data_alone():
    assert S.canonical_column_name('RowID') == 'rowID'
    assert S.canonical_column_name('column_name') == 'columnID'
    assert S.canonical_column_name('time_id') == 'timeID'
    assert S.canonical_column_name('cell_area') == 'cell_area'
    assert S.canonical_column_name('cell_channel_0_mean_intensity') \
        == 'cell_channel_0_mean_intensity'


def test_canonicalise_columns_never_overwrites_an_existing_canonical_column():
    df = pd.DataFrame({'timeID': ['t1'], 'time_id': ['t9'], 'x': [1]})
    out = S.canonicalise_columns(df)
    assert list(out.columns) == ['timeID', 'time_id', 'x']
    assert out['timeID'].tolist() == ['t1']
    assert out['time_id'].tolist() == ['t9']


def test_canonicalise_columns_repairs_a_legacy_frame(tmp_path):
    """Against a real database that was renamed back to the legacy spelling."""
    root = str(tmp_path)
    db = _write_png_list(root, ['plate1_A01_1_3_17.png'], timelapse=True)
    conn = sqlite3.connect(db)
    try:
        conn.execute('ALTER TABLE png_list RENAME COLUMN "timeID" TO "time_id"')
        conn.commit()
    finally:
        conn.close()

    legacy = _read(db, 'png_list')
    assert 'time_id' in legacy.columns and 'timeID' not in legacy.columns

    fixed = S.canonicalise_columns(legacy)
    assert 'timeID' in fixed.columns and 'time_id' not in fixed.columns
    assert fixed['timeID'].tolist() == legacy['time_id'].tolist() == ['t3']


def test_canonicalise_columns_returns_a_copy():
    df = pd.DataFrame({'x': [1]})
    out = S.canonicalise_columns(df)
    out['x'] = 2
    assert df['x'].tolist() == [1]


# ===========================================================================
# tables
# ===========================================================================

def test_owned_tables_match_the_ones_utils_declares():
    from spacr import utils as U

    assert set(S.PARENT_OBJECT_TABLES) == set(U._PARENT_OBJECT_TABLES)
    assert set(S.CHILD_OBJECT_TABLES) == set(U._CHILD_OBJECT_TABLES)
    assert set(S.ORGANELLE_SUMMARY_TABLES) == set(U._ORGANELLE_SUMMARY_TABLES)


def test_owned_tables_cover_what_a_real_run_writes(tmp_path):
    root = str(tmp_path)
    _write_object_table(root, 'plate1_A01_1', table='cell')
    _write_object_table(root, 'plate1_A01_1', table='nucleus')
    db = _write_png_list(root, ['plate1_A01_1_1.png'])

    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert tables <= set(S.OWNED_TABLES), tables - set(S.OWNED_TABLES)


def test_table_key_columns():
    assert S.table_key_columns('cell') == (
        'plateID', 'rowID', 'columnID', 'fieldID', 'object_label')
    assert S.table_key_columns('cell', timelapse=True) == (
        'plateID', 'rowID', 'columnID', 'fieldID', 'timeID', 'object_label')
    assert S.table_key_columns('png_list')[-1] == 'prcfo'
    assert S.table_key_columns('object_counts') == ('file_name',)
    assert S.table_key_columns('settings') == ()
    with pytest.raises(KeyParseError, match='not a table spaCR owns'):
        S.table_key_columns('some_other_table')


# ===========================================================================
# plate formats
# ===========================================================================

def test_plate_format_for():
    assert S.plate_format_for('r1', 'c1') == 6
    assert S.plate_format_for('r8', 'c12') == 96
    assert S.plate_format_for('r1', 'c24') == 384
    assert S.plate_format_for('r32', 'c48') == 1536
    assert S.plate_format_for('r33', 'c1') is None
    assert S.plate_format_for('rubbish', 'c1') is None


def test_a_column_past_24_is_legal_not_an_error():
    """The 1536 case the task asked about: accepted, and identifiable."""
    assert S.parse_well('A48') == ('r1', 'c48')
    assert S.plate_format_for('r1', 'c48') == 1536
    assert S.is_within_plate_format('r1', 'c48', 1536) is True
    assert S.is_within_plate_format('r1', 'c48', 384) is False
    assert S.is_within_plate_format('r1', 'c25', 384) is False
    assert S.is_within_plate_format('r16', 'c24', 384) is True


def test_is_within_plate_format_rejects_a_nonstandard_size():
    with pytest.raises(KeyParseError, match='not a standard plate format'):
        S.is_within_plate_format('r1', 'c1', 100)


# ===========================================================================
# the pandas helper
# ===========================================================================

def test_add_identity_columns_reproduces_the_writer_on_good_names(tmp_path):
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_B03_2', labels=(1, 2))
    cell = _read(db, 'cell')

    out = S.add_identity_columns(cell[['file_name']].copy())
    assert out['plateID'].tolist() == cell['plateID'].tolist()
    assert out['rowID'].tolist() == cell['rowID'].tolist()
    assert out['columnID'].tolist() == cell['columnID'].tolist()
    assert out['fieldID'].tolist() == cell['fieldID'].tolist()
    assert out['prcf'].tolist() == cell['prcf'].tolist()


def test_add_identity_columns_keeps_three_sites_apart(tmp_path):
    """The f0 collapse, undone."""
    root = str(tmp_path)
    for site in ('s1', 's2', 's3'):
        db = _write_object_table(root, f'plate1_A01_{site}')
    cell = _read(db, 'cell')
    assert cell['prcf'].nunique() == 1            # what is on disk

    fixed = S.add_identity_columns(cell[['file_name']].copy())
    assert fixed['prcf'].nunique() == 3           # what it should have been
    assert sorted(fixed['fieldID']) == ['f1', 'f2', 'f3']


def test_add_identity_columns_objects_and_timelapse(tmp_path):
    root = str(tmp_path)
    db = _write_png_list(root, ['plate1_A01_1_3_17.png',
                                'plate1_A01_1_4_18.png'], timelapse=True)
    png = _read(db, 'png_list')
    out = S.add_identity_columns(png[['file_name']].copy(), timelapse=True,
                                 objects=True)
    assert out['prcfo'].tolist() == png['prcfo'].tolist()
    assert out['timeID'].tolist() == png['timeID'].tolist() == ['t3', 't4']


def test_add_identity_columns_preserves_the_index_and_the_other_columns():
    df = pd.DataFrame({'file_name': ['plate1_A01_1'], 'area': [3.0]},
                      index=[42])
    out = S.add_identity_columns(df)
    assert out.index.tolist() == [42]
    assert out['area'].tolist() == [3.0]
    assert out['prcf'].tolist() == ['plate1_r1_c1_f1']


def test_add_identity_columns_needs_the_column_it_is_told_to_read():
    with pytest.raises(KeyParseError, match='not a column'):
        S.add_identity_columns(pd.DataFrame({'x': [1]}))


def test_add_identity_columns_can_skip_prcf():
    out = S.add_identity_columns(pd.DataFrame({'file_name': ['plate1_A01_1']}),
                                 include_prcf=False)
    assert 'prcf' not in out.columns
    assert 'fieldID' in out.columns


def test_add_identity_columns_on_an_empty_frame():
    out = S.add_identity_columns(pd.DataFrame({'file_name': []}))
    assert len(out) == 0


# ===========================================================================
# dependency hygiene
# ===========================================================================

def test_module_imports_with_only_the_stdlib_and_pandas():
    """Everything wants this module, so it must cost nothing to import.

    The sys.modules delta, not its absolute contents -- the coverage runner's
    sitecustomize pre-imports torch into every interpreter here.
    """
    module_path = os.path.abspath(S.__file__)
    code = (
        "import importlib.util, sys\n"
        "before = set(sys.modules)\n"
        f"spec = importlib.util.spec_from_file_location('schema_probe', {module_path!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['schema_probe'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "added = set(sys.modules) - before\n"
        "banned = sorted(m for m in added\n"
        "                if m.split('.')[0] in ('torch', 'cellpose', 'tensorflow',\n"
        "                                       'skimage', 'scipy', 'cv2', 'PyQt5',\n"
        "                                       'PySide6', 'matplotlib', 'spacr'))\n"
        "assert not banned, banned\n"
        "assert mod.parse_field_stem('plate1_AA01_s3').prcf == 'plate1_r27_c1_f3'\n"
    )
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_errors_are_valueerrors_so_existing_guards_still_catch_them():
    assert issubclass(SchemaError, ValueError)
    assert issubclass(WellParseError, SchemaError)
    assert issubclass(KeyParseError, SchemaError)


def test_public_api_is_exported():
    for name in S.__all__:
        assert hasattr(S, name), name
