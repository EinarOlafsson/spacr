"""``spacr.schema`` — the one definition of what a spaCR key is.

The reproductions, and their repair
-----------------------------------
``_map_wells`` was reimplemented five times and the copies disagreed about
malformed wells. This file proved it against **real databases built by the
real writers** (``utils._merge_and_save_to_database`` and
``utils.filepaths_to_database``), never a hand-built schema. Every one of
those call sites is now routed through :mod:`spacr.schema`, so the tests that
pinned the damage now pin the fix — against the same real writers, with the
measured before-value kept in each docstring so the repair stays legible:

* :func:`test_object_table_and_png_list_agree_on_a_1536_well` — one field,
  both writers, one identity. It used to be two identities and a join that
  returned **0 rows out of 2 x 2**.
* :func:`test_three_sites_stay_three_fields` — three ImageXpress sites go in,
  three ``prcf`` come out. It used to be one.
* :func:`test_a_whole_timelapse_survives` — three timepoints go in, three
  ``timeID`` come out. They used to be one.

Then the canonical module is tested against every malformed input, and pinned
against :func:`spacr.schema.legacy_map_wells` — a bug-compatible copy of the
pre-migration ``_map_wells`` — on every input the legacy one got right, so
the migration is provably a repair and not a change of contract.
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
# THE REPAIR -- against real databases
# ===========================================================================

def test_object_table_and_png_list_agree_on_a_1536_well(tmp_path):
    """One field, two writers, one identity, and a join that finds everything.

    ``AA01`` is an ordinary well of a 1536 plate (32 rows: A..Z, AA..AF).
    Before the migration ``_map_wells`` raised inside and returned ``'error'``
    in every slot -- destroying the *plate* along with the well -- while
    ``_map_wells_png`` silently read ``well[1:] == 'A01'`` through
    ``_safe_int_convert`` and got column ``0``. MEASURED, then::

        cell     : ('error', 'error', 'error', 'error')
        png_list : ('plate1', 'r1', 'c0', 'f1')
        join     : 0 rows out of 2 x 2

    Both writers now go through :func:`spacr.schema.parse_field_stem` /
    :func:`spacr.schema.parse_object_stem`.
    """
    root = str(tmp_path)
    stem = 'plate1_AA01_1'
    db = _write_object_table(root, stem, labels=(1, 2))
    _write_png_list(root, [f'{stem}_1.png', f'{stem}_2.png'])

    cell = _read(db, 'cell')
    png = _read(db, 'png_list')
    assert len(cell) == 2 and len(png) == 2

    # MEASURED: the object table now carries the real well.
    assert cell[['plateID', 'rowID', 'columnID', 'fieldID', 'prcf']] \
        .drop_duplicates().values.tolist() == [
            ['plate1', 'r27', 'c1', 'f1', 'plate1_r27_c1_f1']]
    # MEASURED: png_list agrees, token for token.
    assert png[['plateID', 'rowID', 'columnID', 'fieldID']] \
        .drop_duplicates().values.tolist() == [['plate1', 'r27', 'c1', 'f1']]

    # MEASURED: the join now matches. 2 crops x 2 objects, no key lost.
    joined = png.merge(cell, on=['plateID', 'rowID', 'columnID', 'fieldID'])
    assert len(joined) == 4

    # And both halves are exactly what schema says they are.
    field = S.parse_field_stem(stem)
    obj = S.parse_object_stem(f'{stem}_1.png')
    assert field.to_dict() == {'plateID': 'plate1', 'rowID': 'r27',
                               'columnID': 'c1', 'fieldID': 'f1'}
    assert obj.field == field
    assert field.well == 'AA01'


def test_three_sites_stay_three_fields(tmp_path):
    """Three ImageXpress sites go in, three fields come out.

    ``_safe_int_convert('s1')`` was ``0``, so ``s1``, ``s2`` and ``s3`` all
    became ``f0`` and shared a ``prcf`` (MEASURED: ``fieldID`` was
    ``['f0', 'f0', 'f0']`` and ``prcf.nunique()`` was 1). Object 1 of each
    then became the same ``prcfo``: three different cells, one identity.
    """
    root = str(tmp_path)
    for site in ('s1', 's2', 's3'):
        db = _write_object_table(root, f'plate1_A01_{site}')

    cell = _read(db, 'cell')
    # MEASURED: three rows from three distinct files ...
    assert sorted(cell['file_name']) == ['plate1_A01_s1', 'plate1_A01_s2',
                                         'plate1_A01_s3']
    # ... and now three field ids and three prcf between them.
    assert sorted(cell['fieldID']) == ['f1', 'f2', 'f3']
    assert cell['prcf'].nunique() == 3
    assert set(cell['prcf']) == {'plate1_r1_c1_f1', 'plate1_r1_c1_f2',
                                 'plate1_r1_c1_f3'}

    # The database agrees with schema, name for name.
    assert {S.parse_field_stem(f'plate1_A01_{s}').prcf
            for s in ('s1', 's2', 's3')} == set(cell['prcf'])


def test_a_whole_timelapse_survives(tmp_path):
    """Three timepoints go in, three ``timeID`` come out.

    ``_safe_int_convert('T0001')`` was ``0``, so every frame became ``t0``
    (MEASURED: ``timeID`` was ``['t0', 't0', 't0']``, ``prcf.nunique()`` 1).
    """
    root = str(tmp_path)
    for stamp in ('T0001', 'T0002', 'T0003'):
        db = _write_object_table(root, f'plate1_A01_1_{stamp}',
                                 timelapse=True)

    cell = _read(db, 'cell')
    assert sorted(cell['file_name']) == ['plate1_A01_1_T0001',
                                         'plate1_A01_1_T0002',
                                         'plate1_A01_1_T0003']
    # MEASURED: 3 distinct timepoints on disk, 3 in the database.
    assert sorted(cell['timeID']) == ['t1', 't2', 't3']
    assert cell['timeID'].nunique() == 3
    assert cell['prcf'].nunique() == 3

    assert [S.parse_field_stem(f'plate1_A01_1_{s}', timelapse=True).timeID
            for s in ('T0001', 'T0002', 'T0003')] == ['t1', 't2', 't3']


def test_a_lowercase_well_is_just_a_well(tmp_path):
    """A lowercase well used to make the *whole* identity the string ``'error'``.

    Those five ``'error'`` strings were not a sentinel any reader checks --
    they were appended to the table as if they were a plate, a row, a column
    and a field, so the rows were unfindable and polluted every group-by.
    """
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_a01_1', labels=(1, 2, 3))
    cell = _read(db, 'cell')

    assert len(cell) == 3
    assert set(cell['plateID']) == {'plate1'}
    assert set(cell['prcf']) == {'plate1_r1_c1_f1'}
    assert sorted(cell['object_label']) == [1, 2, 3]

    assert S.parse_field_stem('plate1_a01_1').prcf == 'plate1_r1_c1_f1'


def test_png_list_no_longer_reads_the_field_token_as_the_object_too(tmp_path):
    """``plate1_A01_5.png`` used to become field 5 *and* object 5.

    ``_map_wells_png`` took the field from ``parts[2]`` and the object from
    ``parts[-1]``; in a three-part name those are one token, and it produced
    ``fieldID='f5'``, ``cell_id='o5'``, ``prcfo='plate1_r1_c1_f5_o5'`` -- a
    complete identity for an object that was never identified.

    ``utils._generate_names`` never emits a three-part crop name (it is always
    ``<stem>_<cell_id>``), so refusing costs nothing, and refusing means the
    row is visibly broken rather than invisibly wrong.
    """
    root = str(tmp_path)
    db = _write_png_list(root, ['plate1_A01_5.png'])
    png = _read(db, 'png_list')
    row = png.iloc[0]

    assert row['fieldID'] == 'error'
    assert row['cell_id'] == 'error'
    assert row['prcfo'] == 'error'

    with pytest.raises(KeyParseError, match='plate_well_field_object'):
        S.parse_object_stem('plate1_A01_5.png')


def test_the_five_implementations_now_agree():
    """The disagreement, gone. Every cell here was a different real answer.

    ================  ==================  ======================  ====================
    well              ``_map_wells``      ``_map_wells_png``      ``align._well_ids``
    ================  ==================  ======================  ====================
    ``'A'``           ``error``           ``r1,c0``               ``A,A``
    ``'a01'``         ``error``           ``error``               ``r1,c1``
    ``'AA01'``        ``error``           ``r1,c0``               ``AA01,AA01``
    ================  ==================  ======================  ====================

    All five now delegate to :mod:`spacr.schema`. The one remaining
    difference is deliberate and asserted below: ``align._well_ids`` and
    ``convert._well_ids`` keep the legacy passthrough for a well with no
    column, because a stitch and a conversion must emit a row for every input
    file, whereas a *key writer* must not invent an identity for one.
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

    # 'A' -- letters, no column. Not a well: the three key writers refuse it
    # and the two passthrough sites echo it back unchanged.
    assert utils_stack('A') == ('error', 'error')
    assert utils_png('A') == ('error', 'error')
    with pytest.raises(WellParseError, match='no column'):
        field_identity('plate1_A_3')
    assert align_well_ids('A') == ('A', 'A')
    assert convert_well_ids('A') == ('A', 'A')
    with pytest.raises(WellParseError, match='no column'):
        S.parse_well('A')

    # lowercase -- all five now read it as the well it is.
    for reader in (utils_stack, utils_png, resume_ids, align_well_ids,
                   convert_well_ids):
        assert reader('a01') == ('r1', 'c1'), reader

    # a 1536-plate row -- one answer, and it is the right one.
    for reader in (utils_stack, utils_png, resume_ids, align_well_ids,
                   convert_well_ids):
        assert reader('AA01') == ('r27', 'c1'), reader
    assert S.parse_well('AA01') == ('r27', 'c1')

    # ... and the cases they always agreed on, unchanged.
    for well in ('A01', 'A1', 'P24', 'Z01', 'A48'):
        assert utils_stack(well) == utils_png(well) == resume_ids(well) \
            == align_well_ids(well) == convert_well_ids(well) \
            == S.parse_well(well)


# ===========================================================================
# the canonical parser is a strict repair of the legacy one
#
# Pinned against schema.legacy_map_wells, which reproduces the pre-migration
# _map_wells bit for bit. Pinning against utils._map_wells would now be
# pinning schema against itself.
# ===========================================================================

_LEGACY_GOOD_WELLS = ['A01', 'A1', 'A12', 'B03', 'H12', 'P24', 'Z01', 'A48',
                      'A0', 'A00', 'D6']


@pytest.mark.parametrize('well', _LEGACY_GOOD_WELLS)
@pytest.mark.parametrize('field', ['1', '01', '007', '12'])
def test_canonical_agrees_with_legacy_on_every_name_legacy_gets_right(well,
                                                                      field):
    """No contract change: where the old ``_map_wells`` worked, so does this."""
    from spacr.utils import _map_wells

    name = f'plate1_{well}_{field}'
    plate, row, column, fld, prcf = S.legacy_map_wells(name)
    assert (plate, row, column, fld) != ('error',) * 4  # legacy did work

    parsed = S.parse_field_stem(name)
    assert (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID) \
        == (plate, row, column, fld)
    assert parsed.prcf == prcf
    # ... and the migrated writer still answers exactly that.
    assert _map_wells(name) == (plate, row, column, fld, prcf)


@pytest.mark.parametrize('well', _LEGACY_GOOD_WELLS)
def test_canonical_agrees_with_legacy_timelapse_ordering(well):
    """The timepoint goes after the field, as it does on disk."""
    from spacr.utils import _map_wells

    name = f'plate1_{well}_2_5'
    plate, row, column, fld, timeid, prcf = S.legacy_map_wells(
        name, timelapse=True)
    parsed = S.parse_field_stem(name, timelapse=True)
    assert (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID,
            parsed.timeID) == (plate, row, column, fld, timeid)
    assert parsed.prcf == prcf
    assert parsed.prcf.endswith('_f2_t5')
    assert _map_wells(name, timelapse=True) == (plate, row, column, fld,
                                                timeid, prcf)


@pytest.mark.parametrize('well', _LEGACY_GOOD_WELLS)
def test_canonical_prcfo_agrees_with_legacy_png(well):
    """The migrated ``_map_wells_png`` still answers every good name the same."""
    from spacr.utils import _map_wells_png

    name = f'plate1_{well}_3_17.png'
    plate, row, column, fld, prcfo, obj = _map_wells_png(name)
    parsed = S.parse_object_stem(name)
    assert (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID,
            parsed.objectID) == (plate, row, column, fld, obj)
    assert parsed.prcfo == prcfo
    # ... and the pre-migration parser said the same about the field half.
    assert S.legacy_map_wells(f'plate1_{well}_3')[:4] == (plate, row, column,
                                                          fld)


def test_the_two_safe_int_copies_no_longer_disagree_on_none():
    """``resume._safe_int`` is gone and ``_safe_int_convert`` is not a key.

    They were the same bug twice: both returned ``0``, and only one of them
    caught ``TypeError``, so ``None`` was a crash in one code path and field
    ``f0`` in the other. ``resume.field_identity`` now calls
    :func:`spacr.schema.parse_field_stem`; ``_safe_int_convert`` survives only
    as a de-zero-padding helper and answers "is there an integer here?"
    through :func:`spacr.schema.parse_int_token`.
    """
    import spacr.resume as resume
    from spacr.utils import _safe_int_convert

    assert not hasattr(resume, '_safe_int')
    assert _safe_int_convert(None) == 0            # no longer a TypeError
    assert _safe_int_convert(None, default=-1) == -1
    assert _safe_int_convert('x') == 0
    # Neither ever says "there is no number here". schema does.
    assert S.parse_int_token('x') is None


def test_the_ml_regex_that_only_understood_rows_a_to_p_is_gone():
    """``ml.py``'s inline copy silently left rows past P unchanged."""
    import re

    from spacr import ml

    pattern_source = r"^(?i:plate)\d+_([A-Pa-p])(\d+)_"
    assert pattern_source not in open(ml.__file__).read()

    pattern = re.compile(pattern_source)
    assert pattern.match('PLATE1_A14_1_1_111.png').groups() == ('A', '14')
    # MEASURED: a 384-plate row past P, and any plate not named "plate<n>",
    # did not match -- rowID/columnID were then left as they were.
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


def test_legacy_helpers_still_reproduce_the_pre_migration_behaviour():
    """The bug-compatible copies are the only remaining record of the old keys.

    A database written before the migration carries them, so a reader has to
    be able to reproduce them; that is the whole reason they exist. They are
    pinned here against the measured pre-migration answers rather than against
    ``utils``, which no longer produces them.
    """
    assert S.legacy_map_wells('plate1_A01_1') == (
        'plate1', 'r1', 'c1', 'f1', 'plate1_r1_c1_f1')
    for name in ['plate1_a01_1', 'plate1_AA01_1', 'plate1_A_1', 'plate1__1',
                 'garbage', 'plate1_A01']:
        assert S.legacy_map_wells(name) == ('error',) * 5, name
        assert S.legacy_map_wells(name, timelapse=True) == ('error',) * 6, name
    # The f0 collapse and the t0 collapse, exactly as they were written.
    assert S.legacy_map_wells('plate1_A01_s3')[3] == 'f0'
    assert S.legacy_map_wells('plate1_A01_1_T0003', timelapse=True)[4] == 't0'
    assert S.legacy_map_wells('plate1_12_3') == (
        'plate1', '12', '12', 'f3', 'plate1_12_12_f3')

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


def test_the_inverse_direction_no_longer_breaks_past_row_z(tmp_path):
    """Going ``rowID -> well`` used ``chr(ord('A') + n - 1)`` in two places.

    ``crops.py`` rebuilt a merged-stack path from a row id that way and
    ``utils._convert_cq1_well_id`` built a CQ1 well from a linear index that
    way. Both walked straight off the end of the alphabet; both now call
    :func:`spacr.schema.well_id`.
    """
    from spacr.crops import MergedCropSource
    from spacr.utils import _convert_cq1_well_id

    # The old crops.py:2244 expression, kept here as the measurement.
    def crops_style(rowid):
        return chr(ord('A') + int(str(rowid).lstrip('r')) - 1)

    assert crops_style('r26') == 'Z'
    assert crops_style('r27') == '['            # MEASURED: not a row letter
    assert crops_style('r32') == '`'
    assert S.letters_from_row_index(27) == 'AA'
    assert S.letters_from_row_index(32) == 'AF'

    # crops now rebuilds the real 1536 well.
    cropper = MergedCropSource(merged_root=str(tmp_path))
    rebuilt = cropper.resolve_path({'plateID': 'plate1', 'rowID': 'r27',
                                    'columnID': 'c1', 'fieldID': 'f3'})
    assert os.path.basename(rebuilt) == 'plate1_AA01_3.npy'
    assert S.parse_field_stem(rebuilt).prcf == 'plate1_r27_c1_f3'

    # utils._convert_cq1_well_id: 24 columns per row is the CQ1's own layout
    # and stays, but the row letter no longer leaves the alphabet.
    # MEASURED before: _convert_cq1_well_id(1536) == '\x8024'.
    assert _convert_cq1_well_id(1) == 'A01'
    assert _convert_cq1_well_id(384) == 'P24'
    assert _convert_cq1_well_id(1536) == 'BL24'
    assert S.parse_well(_convert_cq1_well_id(1536)) == ('r64', 'c24')

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
# the object TYPE lives in the object key
# ===========================================================================
#
# The measured failure: a nucleus labelled 1 and a pathogen labelled 1 in one
# field carried the identical key, and a cell's own children are exactly the
# objects most likely to collide. Four objects opened as three crops, and
# which one you got depended on the row order of ``png_list``.


def test_the_key_vocabulary_is_exactly_the_object_tables():
    """A new object table cannot gain a table without gaining a prefix.

    ``OBJECT_TYPES`` is declared above ``object_id`` because the composer
    needs it, and ``OBJECT_TABLES`` far below. Two lists in one module is how
    they drift; this is the pin that says they may not.
    """
    assert set(S.OBJECT_TYPES) == set(S.OBJECT_TABLES)


def test_a_nucleus_and_a_pathogen_with_one_label_are_two_keys():
    """The defect, as a law, at the ``prcfo`` level."""
    nucleus = S.compose_prcfo('p1', 1, 1, 1, 1, object_type='nucleus')
    pathogen = S.compose_prcfo('p1', 1, 1, 1, 1, object_type='pathogen')
    assert nucleus != pathogen
    assert S.parse_prcfo(nucleus).objectType == 'nucleus'
    assert S.parse_prcfo(pathogen).objectType == 'pathogen'
    assert S.parse_prcfo(nucleus).objectLabel == '1'


def test_the_untyped_object_key_is_byte_for_byte_what_it_always_was():
    """Every ``prcfo`` on disk is untyped; none of them may move.

    The type is a *refinement* of the key, not a new spelling of it. If the
    untyped form changed, every key in every exported file and every stored
    selection would have to be migrated — and the ones that were missed would
    silently match nothing.
    """
    assert S.compose_prcfo('p1', 1, 1, 1, 7) == 'p1_r1_c1_f1_o7'
    assert S.compose_prcfo('p1', 1, 1, 1, 7, time=3) == 'p1_r1_c1_f1_t3_o7'
    assert S.object_id(7) == 'o7'
    assert S.parse_prcfo('p1_r1_c1_f1_o7').prcfo == 'p1_r1_c1_f1_o7'


def test_an_untyped_key_is_not_stated_rather_than_a_cell():
    """``None`` is a fact about what we were told, not a default guess.

    Defaulting an untyped key to ``'cell'`` would take every legacy key --
    which is all of them -- and assert a type nobody recorded, which is worse
    than the collision it would be papering over.
    """
    assert S.parse_prcfo('p1_r1_c1_f1_o7').objectType is None
    assert S.split_object_id('o7') == (None, '7')


@pytest.mark.parametrize('object_type', list(S.OBJECT_TYPES))
def test_every_declared_object_type_round_trips(object_type):
    key = S.compose_prcfo('PLATE-4', 'A', 2, 3, 41, object_type=object_type)
    parsed = S.parse_prcfo(key)
    assert parsed.prcfo == key
    assert parsed.objectType == object_type
    assert parsed.objectLabel == '41'
    assert S.object_index(parsed.objectID) == 41


def test_organelle_is_not_read_as_an_untyped_object_labelled_rganelle():
    """Longest-prefix-first, because ``'organelle'`` starts with ``'o'``."""
    assert S.split_object_id('organelle7') == ('organelle', '7')
    assert S.object_id(7, object_type='organelle') == 'organelle7'


def test_a_label_that_would_read_back_as_a_type_is_refused():
    """The one composition the closed vocabulary cannot make injective.

    An untyped object whose preserved (non-numeric) label happens to be
    ``'rganelle7'`` composes to ``'organelle7'``, which reads back as an
    organelle. Two identities, one key — refused rather than written.
    """
    with pytest.raises(KeyParseError, match='would read back'):
        S.object_id('rganelle7')


def test_object_id_is_idempotent_with_and_without_a_type():
    for token in ('o7', 'nucleus7', 'omulti', 'onone', 'oxy'):
        assert S.object_id(S.object_id(token)) == token


def test_a_type_on_the_label_that_disagrees_with_the_argument_is_refused():
    """Guessing which of the two is right is how a nucleus becomes a pathogen."""
    with pytest.raises(KeyParseError, match='already says it is'):
        S.object_id('nucleus7', object_type='pathogen')
    # An *untyped* id may still be given a type: that is stating something
    # unstated, not overwriting something stated.
    assert S.object_id('o7', object_type='pathogen') == 'pathogen7'


@pytest.mark.parametrize('bad', ['x', 'foo', 'my_type', 'cell1', ''])
def test_an_undeclared_object_type_is_refused(bad):
    """The vocabulary is closed so a malformed key stays an error.

    With an open vocabulary ``'plate1_r1_c1_f2_x7'`` would parse as object 7
    of type ``'x'`` — a plausible wrong answer where there used to be a
    refusal.
    """
    with pytest.raises(KeyParseError):
        S.object_type_prefix(bad)
    assert not S.is_object_type(bad)


def test_is_object_type_lets_a_reader_ask_before_it_stamps():
    assert S.is_object_type('nucleus')
    assert S.is_object_type('NUCLEUS')
    assert not S.is_object_type('png_list')
    assert not S.is_object_type(None)


def test_a_typed_prcfo_is_still_refused_by_the_prcf_parser():
    """One level too deep stays loud, for every type."""
    for object_type in S.OBJECT_TYPES:
        key = S.compose_prcfo('p1', 1, 1, 2, 7, object_type=object_type)
        with pytest.raises(KeyParseError):
            S.parse_prcf(key)


def test_a_typed_object_reports_its_type_in_to_dict():
    typed = S.parse_prcfo('p1_r1_c1_f1_nucleus7').to_dict()
    assert typed[S.OBJECT_TYPE_KEY] == 'nucleus'
    # An untyped one emits exactly the dict it always did: a reader that
    # never learned about types sees no new column on data with no type.
    assert S.OBJECT_TYPE_KEY not in S.parse_prcfo('p1_r1_c1_f1_o7').to_dict()


def test_split_object_id_reads_the_bare_label_only_when_asked():
    """``'7'`` is the selection key's object component, not a ``prcfo``'s."""
    assert S.split_object_id('7') == (None, '')
    assert S.split_object_id('7', require_prefix=False) == (None, '7')
    # An unrecognised token is not a bare label either way round.
    assert S.split_object_id('x7', require_prefix=False) == (None, '')


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

def test_there_is_exactly_one_canonical_column_name():
    """Every import path must reach the *same function object*.

    This test used to assert that ``schema`` was a **superset** of the
    rename map in ``utils`` -- i.e. it pinned the gap open. There really were
    two implementations in scope at once: ``schema.canonical_column_name``
    (22 aliases, case-insensitive) and a second one in ``database_schema``
    (11 aliases, case-sensitive) that ``spacr.utils`` re-exported. Measured
    before the repair, in an interpreter that had imported both::

        spacr.utils.canonical_column_name is database_schema...  -> True
        spacr.utils.canonical_column_name is schema...           -> False

    So whether a database column named ``Row`` got canonicalised depended on
    which module the calling file happened to import, and a database
    canonicalised by one path and read through the other joins on keys that
    do not match. Equality of behaviour is not enough to pin that -- two
    functions can be made equal today and drift again tomorrow -- so identity
    is what is asserted.
    """
    from spacr import database_schema as D
    from spacr import utils as U

    assert U.canonical_column_name is S.canonical_column_name
    assert D.canonical_column_name is S.canonical_column_name
    # The constants too: a rename *map* that disagrees with the rename
    # *function* is the same bug one level down.
    assert D.DB_COLUMN_RENAMES is S.LEGACY_COLUMN_NAMES
    assert U.DB_COLUMN_RENAMES is S.LEGACY_COLUMN_NAMES
    assert D.DB_COLUMN_RENAME_PATTERNS is S.LEGACY_COLUMN_PATTERNS


@pytest.mark.parametrize('name,canonical', [
    # The eleven aliases the database implementation did not know. Each one
    # returned itself unchanged before the two were collapsed into one.
    ('plate_id', 'plateID'),
    ('row_id', 'rowID'),
    ('rowid', 'rowID'),
    ('column_id', 'columnID'),
    ('col_name', 'columnID'),
    ('field_id', 'fieldID'),
    ('time', 'timeID'),
    ('timepoint', 'timeID'),
    ('channel_name', 'chanID'),
    ('chan_id', 'chanID'),
    ('slice_id', 'sliceID'),
    # ...and the case sensitivity it did not have.
    ('Row', 'rowID'),
    ('COLUMN', 'columnID'),
    ('Plate_Name', 'plateID'),
    ('RowID', 'rowID'),
    ('TIMEID', 'timeID'),
])
def test_the_aliases_the_database_path_used_to_miss(name, canonical):
    """Pinned through every entry point, because the bug *was* the entry point."""
    from spacr import database_schema as D
    from spacr import utils as U

    for fn in (S.canonical_column_name, D.canonical_column_name,
               U.canonical_column_name):
        assert fn(name) == canonical, (fn.__module__, name)


@pytest.mark.parametrize('name,canonical', [
    ('cell_channel_0_periphery_25_percentile',
     'cell_channel_0_periphery_percentile_25'),
    ('pathogen_channel_1_outside_95_percentile',
     'pathogen_channel_1_outside_percentile_95'),
    ('organelle_summary_organelle_ch0_mean_intensity_per_cell',
     'organelle_summary_organelle_channel_0_mean_intensity_per_cell'),
])
def test_the_feature_rewrites_survived_the_collapse(name, canonical):
    """The half of the rename map only the database path used to know.

    ``schema.canonical_column_name`` handled metadata aliases and nothing
    else, so collapsing onto it would have silently dropped the two feature
    families -- several hundred columns on a four-channel run -- from every
    database migration. They moved into
    :data:`spacr.schema.LEGACY_COLUMN_PATTERNS` instead.
    """
    from spacr import database_schema as D
    from spacr import utils as U

    for fn in (S.canonical_column_name, D.canonical_column_name,
               U.canonical_column_name):
        assert fn(name) == canonical, (fn.__module__, name)
        assert fn(canonical) == canonical, (fn.__module__, canonical)


@pytest.mark.parametrize('name', [
    'cell_area',
    'cell_channel_0_percentile_75',
    'nucleus_channel_0_periphery_mean',
    'my_custom_ch0_score',
    'organelle_summary_organelle_mean_area',
    # a head containing 'outside' must not capture the ring rewrite
    'outside_thing_periphery_25_percentile_extra',
    # case-sensitive on purpose: a user column is not spaCR's to rewrite
    'Outside_25_Percentile',
    '',
])
def test_canonical_column_name_still_leaves_data_columns_alone(name):
    assert S.canonical_column_name(name) == name


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


def test_the_two_frame_canonicalisers_produce_the_same_columns():
    """``schema.canonicalise_columns`` and ``utils.canonicalize_measurement_columns``.

    Two functions with the same job, reached from different modules, that
    used to give different answers on the same frame: the schema one knew the
    metadata aliases and not the feature spellings, the utils one knew the
    feature spellings and eleven fewer aliases. Whichever a caller imported
    decided what its columns were called.
    """
    from spacr.utils import canonicalize_measurement_columns

    columns = {
        'plate_id': ['plate1'],
        'Row': ['r1'],
        'col_name': ['c1'],
        'time': ['t3'],
        'cell_channel_0_periphery_25_percentile': [1.0],
        'organelle_summary_organelle_ch1_mean_intensity': [2.0],
        'cell_area': [3.0],
    }
    schema_out = S.canonicalise_columns(pd.DataFrame(columns))
    utils_out = canonicalize_measurement_columns(pd.DataFrame(columns))

    assert list(schema_out.columns) == list(utils_out.columns)
    assert list(schema_out.columns) == [
        'plateID', 'rowID', 'columnID', 'timeID',
        'cell_channel_0_periphery_percentile_25',
        'organelle_summary_organelle_channel_1_mean_intensity',
        'cell_area',
    ]
    # The values ride along with their column; nothing is reordered or lost.
    assert schema_out['rowID'].tolist() == ['r1']
    assert schema_out['cell_area'].tolist() == [3.0]


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
    """The f0 collapse, undone — and the vectorised helper agrees.

    ``cell['prcf'].nunique()`` was 1 here before the migration: ``s1``,
    ``s2`` and ``s3`` all became ``f0``. Now the writer and
    :func:`add_identity_columns` derive the same three fields from the same
    three names, which is the point — a reader that re-derives the key must
    land on the one already in the table.
    """
    root = str(tmp_path)
    for site in ('s1', 's2', 's3'):
        db = _write_object_table(root, f'plate1_A01_{site}')
    cell = _read(db, 'cell')
    assert cell['prcf'].nunique() == 3

    fixed = S.add_identity_columns(cell[['file_name']].copy())
    assert fixed['prcf'].nunique() == 3
    assert sorted(fixed['fieldID']) == ['f1', 'f2', 'f3']
    assert fixed['prcf'].tolist() == cell['prcf'].tolist()


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
