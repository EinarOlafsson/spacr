"""Every key spaCR writes now comes from :mod:`spacr.schema`.

``tests/test_schema.py`` tests the module. This file tests the **migration**:
that the hand-rolled ``_map_wells`` copies, the ``prcf`` / ``prcfo`` string
surgery and the ``'r' + str(n)`` formatting scattered across the tree are all
gone, and that what replaced them agrees — with schema, and with each other,
on exactly the inputs the copies used to disagree about.

Every database here is built by the **real writers**
(``utils._merge_and_save_to_database``, ``utils.filepaths_to_database``) and
only read back. A hand-built fixture that happens to match the schema is
precisely how this class of bug survived five reimplementations.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr import schema as S
from spacr.schema import KeyParseError, WellParseError


# ---------------------------------------------------------------------------
# helpers -- real writers only
# ---------------------------------------------------------------------------

def _read(db, table):
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    finally:
        conn.close()


def _write_object_table(root, stem, labels=(1,), table='cell',
                        timelapse=False):
    from spacr.utils import _merge_and_save_to_database

    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    morph = pd.DataFrame({
        'label': list(labels),
        f'{table}_area': [10.0 + i for i in range(len(labels))]})
    intensity = pd.DataFrame({
        'label': list(labels),
        f'{table}_channel_0_mean_intensity': [1.0 + i
                                              for i in range(len(labels))]})
    _merge_and_save_to_database(morph, intensity, table, root, stem, 'exp',
                                timelapse=timelapse)
    return os.path.join(root, 'measurements', 'measurements.db')


def _write_png_list(root, names, crop_mode='cell', timelapse=False):
    from spacr.utils import filepaths_to_database

    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    paths = [os.path.join(root, f'{crop_mode}_png', n) for n in names]
    filepaths_to_database(paths, {'timelapse': timelapse}, root, crop_mode)
    return os.path.join(root, 'measurements', 'measurements.db')


def _plate_database(tmp_path, timelapse=False):
    """A small but real measurements.db: 2 wells x 2 fields x 2 objects.

    With ``timelapse``, 2 frames as well. Written by the real writers, so
    every key in it is a key the pipeline actually produces.
    """
    root = str(tmp_path)
    stems = []
    for well in ('A01', 'H12'):
        for field in (1, 2):
            if timelapse:
                for frame in (1, 2):
                    stems.append(f'plate1_{well}_{field}_{frame}')
            else:
                stems.append(f'plate1_{well}_{field}')
    db = None
    for stem in stems:
        db = _write_object_table(root, stem, labels=(1, 2),
                                 timelapse=timelapse)
        _write_png_list(root, [f'{stem}_{label}.png' for label in (1, 2)],
                        timelapse=timelapse)
    return db, stems


# ===========================================================================
# 1. a real database round-trips every key through schema unchanged
# ===========================================================================

@pytest.mark.parametrize('timelapse', [False, True])
def test_every_key_a_real_run_writes_round_trips_through_schema(tmp_path,
                                                                timelapse):
    """Parse each stored key back with schema; nothing moves.

    This is the property the five copies broke: the *writer* and any
    *reader* must agree on what the key means. Both directions are checked
    here — the columns rebuild the composed key, and the composed key
    rebuilds the columns.
    """
    db, stems = _plate_database(tmp_path, timelapse=timelapse)
    cell = _read(db, 'cell')
    png = _read(db, 'png_list')

    expected_rows = len(stems) * 2
    assert len(cell) == expected_rows
    assert len(png) == expected_rows
    # No identity was lost on the way in.
    assert 'error' not in set(cell['plateID']) | set(cell['prcf'])
    assert 'error' not in set(png['plateID']) | set(png['prcfo'])

    for _, row in cell.iterrows():
        parsed = S.parse_field_stem(row['file_name'], timelapse=timelapse)
        # file name -> columns
        assert parsed.plateID == row['plateID']
        assert parsed.rowID == row['rowID']
        assert parsed.columnID == row['columnID']
        assert parsed.fieldID == row['fieldID']
        if timelapse:
            assert parsed.timeID == row['timeID']
        # file name -> composed key
        assert parsed.prcf == row['prcf']
        # columns -> composed key
        assert S.compose_prcf(
            row['plateID'], row['rowID'], row['columnID'], row['fieldID'],
            time=row['timeID'] if timelapse else None) == row['prcf']
        # composed key -> columns (the direction ml.py used to get wrong)
        back = S.parse_prcf(row['prcf'])
        assert back.to_dict() == parsed.to_dict()

    for _, row in png.iterrows():
        parsed = S.parse_object_stem(row['file_name'], timelapse=timelapse)
        assert parsed.prcfo == row['prcfo']
        assert parsed.objectID == row['cell_id']
        assert parsed.field.to_dict() == {
            k: row[k] for k in parsed.field.to_dict()}
        assert S.parse_prcfo(row['prcfo']).prcfo == row['prcfo']
        assert S.compose_prcfo(
            row['plateID'], row['rowID'], row['columnID'], row['fieldID'],
            row['cell_id'],
            time=row['timeID'] if timelapse else None) == row['prcfo']


@pytest.mark.parametrize('timelapse', [False, True])
def test_the_object_table_and_png_list_join_on_the_field_key(tmp_path,
                                                             timelapse):
    """The join the divergence used to empty. Both writers, one identity."""
    db, stems = _plate_database(tmp_path, timelapse=timelapse)
    cell = _read(db, 'cell')
    png = _read(db, 'png_list')

    keys = list(S.TIMEPOINT_KEY_COLUMNS if timelapse
                else S.FIELD_KEY_COLUMNS)
    joined = png.merge(cell, on=keys)
    # 2 crops x 2 objects per field, every field matched.
    assert len(joined) == len(stems) * 4
    assert set(cell['prcf']) == {S.parse_field_stem(s, timelapse=timelapse).prcf
                                 for s in stems}


@pytest.mark.parametrize('timelapse', [False, True])
def test_split_data_rebuilds_the_prcf_the_writer_stored(tmp_path, timelapse):
    """``utils._split_data`` re-derives ``prcf``/``prcfo`` from the columns.

    It is the one composition that is deliberately still vectorised string
    concatenation rather than a schema call — a per-row parse over a
    thousand-column measurement frame is not free. This is what keeps it
    honest: the value it builds must equal the one ``compose_prcf`` /
    ``compose_prcfo`` build from the same columns, and the one the writer
    put in the table.
    """
    from spacr.utils import _split_data

    db, _stems = _plate_database(tmp_path, timelapse=timelapse)
    cell = _read(db, 'cell')
    stored_prcf = cell['prcf'].tolist()

    frame = cell.copy()
    frame['object_label'] = 'o' + frame['object_label'].astype(int).astype(str)
    frame['prcfo'] = frame['prcf'] + '_' + frame['object_label']
    times = frame['timeID'] if timelapse else [None] * len(frame)

    # What schema says these columns compose to ...
    expected_prcf = [S.compose_prcf(p, r, c, f, time=t)
                     for p, r, c, f, t in zip(frame['plateID'],
                                              frame['rowID'],
                                              frame['columnID'],
                                              frame['fieldID'], times)]
    expected_prcfo = [S.compose_prcfo(p, r, c, f, o, time=t)
                      for p, r, c, f, o, t in zip(frame['plateID'],
                                                  frame['rowID'],
                                                  frame['columnID'],
                                                  frame['fieldID'],
                                                  frame['object_label'],
                                                  times)]
    # ... is what the writer stored ...
    assert expected_prcf == stored_prcf
    # ... and what _split_data groups on.
    numeric, non_numeric = _split_data(frame.copy(), 'prcfo', 'object_label')
    assert sorted(numeric.index) == sorted(set(expected_prcfo))
    assert sorted(non_numeric.index) == sorted(set(expected_prcfo))
    assert len(numeric) == len(frame)   # one row per object, none merged


# ===========================================================================
# 2. the malformed inputs the five copies disagreed about
# ===========================================================================

#: What ``schema`` says each of these wells is. ``None`` means "not a well",
#: which the three key writers must refuse rather than invent an answer for.
_WELL_CASES = [
    ('A1',    ('r1', 'c1')),
    ('A01',   ('r1', 'c1')),
    ('a01',   ('r1', 'c1')),          # lowercase
    ('  A01', ('r1', 'c1')),          # whitespace
    ('AA01',  ('r27', 'c1')),         # a real 1536-plate row
    ('A25',   ('r1', 'c25')),         # past 24 columns: legal on a 1536
    ('A48',   ('r1', 'c48')),
    ('P24',   ('r16', 'c24')),
    ('7',     ('7', '7')),            # positional passthrough
    ('A',     None),                  # letters, no column
    ('',      None),                  # nothing at all
]

# The functions that write a key into a database. All three must refuse a
# well that is not a well; inventing ``c0`` for ``'A'`` is what made
# ``png_list`` unjoinable to ``cell``.
def _key_writers():
    from spacr.resume import field_identity
    from spacr.utils import _map_wells, _map_wells_png

    def utils_stack(well):
        out = _map_wells(f'plate1_{well}_3')
        return None if out[0] == 'error' else (out[1], out[2])

    def utils_png(well):
        out = _map_wells_png(f'plate1_{well}_3_7.png')
        return None if out[0] == 'error' else (out[1], out[2])

    def resume_identity(well):
        try:
            out = field_identity(f'plate1_{well}_3')
        except ValueError:
            return None
        return out['rowID'], out['columnID']

    return {'utils._map_wells': utils_stack,
            'utils._map_wells_png': utils_png,
            'resume.field_identity': resume_identity}


@pytest.mark.parametrize('well,expected', _WELL_CASES)
def test_every_key_writer_gives_the_same_answer(well, expected):
    """One well, one answer, whichever writer is asked.

    Before the migration each column of this table was its own answer. On
    ``'AA01'`` — an ordinary 1536-plate well — ``_map_wells`` returned
    ``('error', 'error')`` and destroyed the plate with it, while
    ``_map_wells_png`` and ``resume.field_identity`` both returned
    ``('r1', 'c0')``: the second row letter dropped and a column 0 invented.
    So the object table and ``png_list`` could not be joined, and a resume
    computed a delete key that matched no rows and re-measured a field that
    was already done.

    On ``'A'`` they split the other way: ``_map_wells`` said ``'error'`` and
    the other two said ``('r1', 'c0')``.
    """
    for name, writer in _key_writers().items():
        assert writer(well) == expected, f'{name} disagreed about {well!r}'
    if expected is None:
        with pytest.raises(WellParseError):
            S.parse_well(well)
    else:
        assert S.parse_well(well) == expected


@pytest.mark.parametrize('well,expected', _WELL_CASES)
def test_the_two_passthrough_sites_agree_wherever_the_well_is_a_well(well,
                                                                    expected):
    """``align`` and ``convert`` keep a documented, deliberate variation.

    Both must emit one row per input file — a stitch cannot drop a tile
    because a folder was named oddly, and a conversion map must record every
    source. So where a *key writer* refuses, these two echo the token back
    into both slots, which is what they have always done and what databases
    on disk already carry. Everywhere the well is really a well they are
    byte-identical to schema.
    """
    from spacr.align import _well_ids as align_well_ids
    from spacr.convert import _well_ids as convert_well_ids

    for name, reader in [('align', align_well_ids),
                         ('convert', convert_well_ids)]:
        got = reader(well)
        if expected is None:
            assert got == (well, well), f'{name} on {well!r}'
        else:
            assert got == expected, f'{name} on {well!r}'


def test_none_and_an_int_are_handled_the_same_way_everywhere():
    """``None`` is not a well; an integer is a positional one.

    ``resume._safe_int`` swallowed ``None`` into ``0`` and
    ``utils._safe_int_convert`` raised ``TypeError`` on it — the same value,
    two outcomes, in two functions that were supposed to be copies.
    """
    from spacr.align import _well_ids as align_well_ids
    from spacr.convert import _well_ids as convert_well_ids

    with pytest.raises(WellParseError, match='empty'):
        S.parse_well(None)
    assert align_well_ids(None) == ('None', 'None')
    assert convert_well_ids(None) == ('None', 'None')

    # An int well is unambiguous only as a passthrough: whether '12' means
    # row 1 column 2 or the twelfth well is not knowable.
    assert S.parse_well(7) == ('7', '7')
    assert S.is_positional_well(7) is True
    for reader in (align_well_ids, convert_well_ids):
        assert reader(7) == ('7', '7')


def test_a_well_past_24_columns_is_written_not_rejected(tmp_path):
    """1536 plates have 48 columns; nothing in the key path may cap at 24."""
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_AF48_1')
    _write_png_list(root, ['plate1_AF48_1_1.png'])

    cell = _read(db, 'cell').iloc[0]
    png = _read(db, 'png_list').iloc[0]
    assert (cell['rowID'], cell['columnID']) == ('r32', 'c48')
    assert png['prcfo'] == 'plate1_r32_c48_f1_o1'
    assert S.plate_format_for(cell['rowID'], cell['columnID']) == 1536
    assert S.well_id(cell['rowID'], cell['columnID']) == 'AF48'


def test_a_1536_well_survives_the_whole_writer_to_reader_path(tmp_path):
    """``plate_qc`` and the database now agree about ``AA01``.

    ``plate_qc`` always parsed multi-letter rows correctly and the writers
    did not, so a QC layout and a measurements table described different
    plates. Both go through schema now.
    """
    from spacr.plate_qc import _parse_well_label, well_id as qc_well_id

    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_AA01_1')
    cell = _read(db, 'cell').iloc[0]

    qc = _parse_well_label('AA01')
    assert qc == (27, 1)
    assert (cell['rowID'], cell['columnID']) == (f'r{qc[0]}', f'c{qc[1]}')
    assert qc_well_id(*qc) == S.well_id(cell['rowID'], cell['columnID'])


# ===========================================================================
# 3. prcf / prcfo compose and split, timelapse and not
# ===========================================================================

def test_a_timelapse_prcfo_splits_with_the_object_last(tmp_path):
    """Six tokens, and the fifth is the timepoint, not the object.

    ``ml._assign_prcfo_parts`` used to split left to right into five fixed
    names, so on a timelapse it either raised (six columns against five
    keys) or would have put ``'t1'`` in ``objectID`` and dropped the object
    entirely. It now parses right to left through
    :func:`spacr.schema.parse_prcfo`.
    """
    from spacr.ml import _assign_prcfo_parts

    db, _stems = _plate_database(tmp_path, timelapse=True)
    png = _read(db, 'png_list')
    assert png['prcfo'].str.count('_').unique().tolist() == [5]  # six tokens

    frame = _assign_prcfo_parts(png[['prcfo']].copy(),
                                object_column='objectID')
    for _, row in frame.iterrows():
        parsed = S.parse_prcfo(row['prcfo'])
        assert row['plateID'] == parsed.plateID
        assert row['rowID'] == parsed.rowID
        assert row['columnID'] == parsed.columnID
        assert row['fieldID'] == parsed.fieldID
        assert row['timeID'] == parsed.timeID
        assert row['objectID'] == parsed.objectID
        assert row['objectID'].startswith('o')
        assert row['timeID'].startswith('t')


def test_a_non_timelapse_prcfo_splits_into_five(tmp_path):
    from spacr.ml import _assign_prcfo_parts

    db, _stems = _plate_database(tmp_path, timelapse=False)
    png = _read(db, 'png_list')
    assert png['prcfo'].str.count('_').unique().tolist() == [4]  # five tokens

    frame = _assign_prcfo_parts(png[['prcfo']].copy(),
                                object_column='objectID')
    assert 'timeID' not in frame.columns
    for _, row in frame.iterrows():
        parsed = S.parse_prcfo(row['prcfo'])
        assert (row['plateID'], row['rowID'], row['columnID'],
                row['fieldID'], row['objectID']) == (
                    parsed.plateID, parsed.rowID, parsed.columnID,
                    parsed.fieldID, parsed.objectID)


def test_mixing_a_timelapse_and_a_plain_key_is_still_refused(tmp_path):
    """The width guard schema deliberately does not provide, kept.

    ``schema.parse_prcfo`` answers "what is this one key?". Whether a *frame*
    of keys agrees with itself is a different question, and the answer to it
    is the difference between a timepoint and an object id, so it must stay
    an error rather than a per-row guess.
    """
    from spacr.io import TimelapseKeyMismatch
    from spacr.ml import _assign_prcfo_parts

    mixed = pd.DataFrame({'prcfo': ['plate1_r1_c1_f1_o1',
                                    'plate1_r1_c1_f1_t2_o1']})
    with pytest.raises(TimelapseKeyMismatch, match='mixes'):
        _assign_prcfo_parts(mixed)

    with pytest.raises(ValueError, match='5 tokens'):
        _assign_prcfo_parts(pd.DataFrame({'prcfo': ['plate1_r1_c1']}))


def test_the_timelapse_prcf_carries_the_timepoint(tmp_path):
    """``prcf`` is ``plate_row_column_field_time`` on a timelapse.

    Dropping the timepoint when rebuilding it collapsed every object across
    all of its frames — a 2-field x 2-frame x 2-cell run came back as half
    the rows with the frames averaged together.
    """
    db, stems = _plate_database(tmp_path, timelapse=True)
    cell = _read(db, 'cell')

    assert cell['prcf'].nunique() == len(stems)
    for _, row in cell.iterrows():
        assert row['prcf'].endswith(f"_{row['fieldID']}_{row['timeID']}")
        assert S.parse_prcf(row['prcf']).timeID == row['timeID']

    # ... and without a timelapse there is no timepoint to carry.
    plain_db, plain_stems = _plate_database(tmp_path / 'plain',
                                            timelapse=False)
    plain = _read(plain_db, 'cell')
    assert plain['prcf'].nunique() == len(plain_stems)
    assert all(S.parse_prcf(v).timeID is None for v in plain['prcf'])


def test_the_object_key_is_the_same_string_everywhere(tmp_path):
    """``png_list.prcfo`` and the ``io`` composition are one string.

    ``io._read_and_merge_data`` builds ``prcf + '_' + 'o' + object_label``
    from the object tables, and ``filepaths_to_database`` builds ``prcfo``
    from the crop file name. They key the same rows, so they must agree.
    """
    db, _stems = _plate_database(tmp_path, timelapse=False)
    cell = _read(db, 'cell')
    png = _read(db, 'png_list')

    io_style = {f"{row['prcf']}_o{int(row['object_label'])}"
                for _, row in cell.iterrows()}
    assert io_style == set(png['prcfo'])
    assert io_style == {S.compose_prcfo(row['plateID'], row['rowID'],
                                        row['columnID'], row['fieldID'],
                                        row['object_label'])
                        for _, row in cell.iterrows()}


# ===========================================================================
# 4. the _safe_int_convert decision, pinned both ways
# ===========================================================================

def test_no_key_is_built_through_safe_int_convert_any_more(tmp_path):
    """The decision, stated as a property of the database.

    ``_safe_int_convert`` returning ``0`` was indefensible **as a key
    builder** — three ImageXpress sites went in and one ``prcf`` came out —
    and raising in the middle of a ten-hour ``measure_crop`` is no better.
    The resolution is neither: key construction now uses
    :func:`spacr.schema.field_id`, which grades the failure —

    * parseable (``'3'``, ``'003'``, ``'s3'``, ``'F003'``) -> ``f3``;
    * present but not a number (``'xy'``) -> ``'fxy'``: still distinct per
      token, so three bad fields stay three fields, still visibly not a
      number, and still a usable join key, so the run continues;
    * absent -> raises, and ``_map_wells`` turns that into the ``'error'``
      row it has always written.

    So no field id is ever invented, and no run dies on one bad name.
    """
    root = str(tmp_path)
    for token in ('s1', 's2', 's3', 'xy', 'zz'):
        db = _write_object_table(root, f'plate1_A01_{token}')
    cell = _read(db, 'cell')

    # MEASURED: five distinct field tokens on disk, five in the database.
    # It used to be one ('f0'), for all five.
    assert sorted(cell['fieldID']) == ['f1', 'f2', 'f3', 'fxy', 'fzz']
    assert cell['prcf'].nunique() == 5
    assert '0' not in {S.strip_prefix(v, 'f') for v in cell['fieldID']}


def test_an_unreadable_field_does_not_stop_the_run(tmp_path, capsys):
    """The other half of the decision: it must not raise, either."""
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_A01_xy', labels=(1, 2))
    cell = _read(db, 'cell')
    assert len(cell) == 2
    assert set(cell['fieldID']) == {'fxy'}
    assert set(cell['prcf']) == {'plate1_r1_c1_fxy'}
    # ... and the key it wrote is still a key: it parses back.
    assert S.parse_prcf('plate1_r1_c1_fxy').fieldID == 'fxy'
    assert S.field_index('fxy') is None      # honestly not a number


def test_a_field_that_is_absent_is_an_error_row_not_a_guess(tmp_path, capsys):
    """Tier three: no token at all. ``f`` alone would merge every such row."""
    root = str(tmp_path)
    db = _write_object_table(root, 'plate1_A01_', labels=(1,))
    cell = _read(db, 'cell')
    assert set(cell['fieldID']) == {'error'}
    assert 'Error processing filename' in capsys.readouterr().out


def test_rebuilding_a_merged_path_refuses_metadata_that_names_no_field(
        tmp_path):
    """``crops`` re-derives a path from the key columns; it must not guess.

    ``chr(ord('A') + n - 1)`` never failed — it just produced ``'['`` for row
    27 and a path that cannot exist. Going through
    :func:`spacr.schema.well_id` / :func:`spacr.schema.field_index` means an
    identity that names no well or no field is an error with the reason in
    it, and a 1536 row is a path.
    """
    from spacr.crops import CropError, MergedCropSource

    source = MergedCropSource(merged_root=str(tmp_path))
    good = source.resolve_path({'plateID': 'plate1', 'rowID': 'r27',
                                'columnID': 'c1', 'fieldID': 'f3'})
    assert os.path.basename(good) == 'plate1_AA01_3.npy'

    # 'fxy' is a real fieldID the graded policy can write, but it names no
    # file, so this must say so rather than build 'plate1_AA01_0.npy'.
    with pytest.raises(CropError, match='does not name a field'):
        source.resolve_path({'plateID': 'plate1', 'rowID': 'r27',
                             'columnID': 'c1', 'fieldID': 'fxy'})
    # A positional well has no well name at all.
    with pytest.raises(CropError, match='does not name a field'):
        source.resolve_path({'plateID': 'plate1', 'rowID': '12',
                             'columnID': '12', 'fieldID': 'f1'})


def test_strict_is_available_for_a_preflight():
    """A QC pass wants tier two to be an error; a long run does not."""
    assert S.parse_field_stem('plate1_A01_xy').fieldID == 'fxy'
    with pytest.raises(KeyParseError, match='holds no integer'):
        S.parse_field_stem('plate1_A01_xy', strict=True)


def test_safe_int_convert_is_pinned_both_ways():
    """It parses what schema calls an integer, and defaults on everything else.

    Kept, rather than deleted, because de-zero-padding a regex group is a
    real job and the default is meaningful there. Not kept as a key builder:
    :func:`test_no_key_is_built_through_safe_int_convert_any_more` is the
    assertion that matters.
    """
    from spacr.utils import _int_or_token, _safe_int_convert

    # parses
    for token, value in [('3', 3), ('003', 3), (' 3 ', 3), (3, 3), (3.0, 3),
                         ('-2', -2), ('0', 0)]:
        assert _safe_int_convert(token) == value, token
        assert S.parse_int_token(token, allow_prefix=False) == value, token

    # defaults -- and schema says None for every one of them
    for token in ['x', 'xy', '', '   ', None, 3.7, True, 's3', '1a']:
        assert _safe_int_convert(token) == 0, token
        assert _safe_int_convert(token, default=-1) == -1, token
        assert S.parse_int_token(token, allow_prefix=False) is None, token

    # and the filename parsers keep the token rather than taking the default
    assert _int_or_token('001') == '1'
    assert _int_or_token('1a') == '1a'


def test_the_filename_metadata_parser_keeps_an_unreadable_token(tmp_path):
    """``_extract_filename_metadata`` no longer collapses odd wells onto '0'.

    The tokens are grouped into a key of ``(plate, well, field, channel,
    time, slice)``; when two different wells both became ``'0'`` their
    images were merged into one stack.
    """
    import re

    from spacr.utils import _extract_filename_metadata

    regex = re.compile(
        r'^(?P<plateID>[^_]+)_(?P<wellID>[^_]+)_(?P<fieldID>[^_]+)_'
        r'(?P<chanID>[^_.]+)\.tif$')
    names = ['plate1_01_001_1.tif', 'plate1_1_1_1.tif',
             '  plate1_1a_002_1.tif'.strip(), 'plate1_1b_002_1.tif']
    src = str(tmp_path)
    keys = _extract_filename_metadata(names, src, regex, metadata_type='x')

    wells = {key[1] for key in keys}
    # '01' and '1' are one well; '1a' and '1b' stay two, and neither is '0'.
    assert '1' in wells and '1a' in wells and '1b' in wells
    assert '0' not in wells
    assert len(keys[('plate1', '1', '1', '1', None, None)]) == 2
