"""Ingest of a plate bigger than 384 — :mod:`spacr.convert` and :mod:`spacr.io`.

``1bf5c4b`` taught the heatmap to draw all 32 rows and 48 columns of a
1536-well plate. The ingest side still could not produce one, and it failed
two different ways:

* a source folder named ``Q01`` (row 17) or ``A25`` (column 25) was not
  recognised as a well at all, so it was handed the *next free synthetic*
  address — ``A01``. A real well, silently renamed, in a run that reports
  success. Only the map file's ``source_well`` column says otherwise.
* past 384 distinct wells the run stopped dead with a ``ConfigurationError``
  naming the 384 limit, so a 1536-well plate could not be converted at all.

Everything here goes through the real writers — :func:`spacr.convert.scan`,
:func:`~spacr.convert.plan`, :func:`~spacr.convert.convert` and
:func:`~spacr.convert.read_map` — on synthetic 2x2 TIFFs. Nothing is
hand-built, because a hand-built map file would agree with whatever the code
happened to do.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr import schema
from spacr.errors import ConfigurationError


def _tif(path: str, value: int = 1) -> None:
    """Write the smallest real TIFF that :func:`scan` will open."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full((2, 2), value, np.uint16))


def _wells_of(n_wells: int):
    """Every well id of an ``n_wells`` plate, row-major, from the schema."""
    n_rows, n_columns = schema.PLATE_FORMATS[n_wells]
    return [schema.well_id(row, column)
            for row in range(1, n_rows + 1)
            for column in range(1, n_columns + 1)]


# ---------------------------------------------------------------------------
# A real convert run over a 1536-well plate
# ---------------------------------------------------------------------------

def test_a_1536_well_source_converts_and_every_well_keeps_its_own_address(tmp_path):
    """1536 well folders -> 1536 TIFFs, none of them renamed.

    Before: ``plan`` raised ``ConfigurationError`` — "1536 wells were found
    but a 384-well plate only has 384" — and nothing was written.
    """
    wells = _wells_of(1536)
    assert len(wells) == 1536
    assert wells[0] == 'A01' and wells[-1] == 'AF48'

    root = tmp_path / 'src'
    for index, well in enumerate(wells):
        _tif(str(root / 'plate1536' / well / 'fov01_C1.tif'), value=index % 500 + 1)

    plan = cv.plan(cv.scan(str(root)))
    assert plan.ok, plan.summary()
    assert len(plan) == 1536

    result = cv.convert(plan, str(tmp_path / 'out'))
    assert result.n_written == 1536
    assert result.is_complete

    frame = cv.read_map(result.map_path)
    assert len(frame) == 1536
    # Not one well was renamed: the address out is the folder name in.
    assert set(frame['well']) == set(wells)
    assert (frame['well'] == frame['source_well']).all()

    # The whole plate is addressable: 32 rows and 48 columns, rows past Z
    # spelled AA..AF.
    assert set(frame['rowID']) == {f'r{i}' for i in range(1, 33)}
    assert set(frame['columnID']) == {f'c{i}' for i in range(1, 49)}
    assert (tmp_path / 'out' /
            'plate1_AF48_T0001F001L01A01Z01C01.tif').is_file()
    assert (tmp_path / 'out' /
            'plate1_AA01_T0001F001L01A01Z01C01.tif').is_file()


def test_a_source_folder_named_Q01_stays_well_Q01(tmp_path):
    """The silent rename, on its own.

    Before: ``Q01`` -> ``A01``, ``A25`` -> ``A02``, ``P25`` -> ``A03`` —
    three real wells renamed, and the run reported success.
    """
    root = tmp_path / 'src'
    for well in ('Q01', 'A25', 'P25', 'AA13'):
        _tif(str(root / 'run1' / well / 'fov01_C1.tif'))

    plan = cv.plan(cv.scan(str(root)))
    assert plan.ok, plan.summary()
    result = cv.convert(plan, str(tmp_path / 'out'))

    frame = cv.read_map(result.map_path)
    assert dict(zip(frame['source_well'], frame['well'])) == {
        'Q01': 'Q01', 'A25': 'A25', 'P25': 'P25', 'AA13': 'AA13'}
    assert dict(zip(frame['source_well'], frame['rowID'])) == {
        'Q01': 'r17', 'A25': 'r1', 'P25': 'r16', 'AA13': 'r27'}
    assert dict(zip(frame['source_well'], frame['columnID'])) == {
        'Q01': 'c1', 'A25': 'c25', 'P25': 'c25', 'AA13': 'c13'}
    assert (tmp_path / 'out' /
            'plate1_Q01_T0001F001L01A01Z01C01.tif').is_file()


def test_normalise_well_accepts_every_address_a_1536_plate_has():
    assert cv.normalise_well('Q01') == 'Q01'
    assert cv.normalise_well('A25') == 'A25'
    assert cv.normalise_well('P25') == 'P25'
    assert cv.normalise_well('aa1') == 'AA01'
    assert cv.normalise_well('AF-48') == 'AF48'
    # Still not wells.
    assert cv.normalise_well('wt') is None
    assert cv.normalise_well('') is None
    assert cv.normalise_well('A0') is None          # column 0 is no column
    assert cv.normalise_well('AG01') is None        # row 33: past every plate
    assert cv.normalise_well('A49') is None         # column 49: past every plate
    assert cv.normalise_well('12') is None          # positional, not an address


def test_a_canonical_well_past_384_does_not_take_someone_elses_id():
    """``Q01`` claims Q01, and the synthetic names go elsewhere."""
    assigned = cv.assign_wells(['Q01', 'wt', 'ko'])
    assert assigned['Q01'] == 'Q01'
    assert assigned['wt'] != 'Q01' and assigned['ko'] != 'Q01'
    assert len(set(assigned.values())) == 3


def test_1536_synthetic_names_all_fit_on_one_plate():
    assigned = cv.assign_wells([f'sample{i}' for i in range(1536)])
    assert len(set(assigned.values())) == 1536
    assert set(assigned.values()) == set(_wells_of(1536))


def test_more_wells_than_the_largest_plate_names_the_real_limit():
    with pytest.raises(ConfigurationError) as excinfo:
        cv.assign_wells([f'sample{i}' for i in range(1537)])
    assert '1536' in str(excinfo.value)
    assert '1537' in str(excinfo.value)


# ---------------------------------------------------------------------------
# A synthetic address is never handed out silently
# ---------------------------------------------------------------------------

def test_synthetic_well_assignment_is_reported_by_name(tmp_path):
    root = tmp_path / 'src'
    for name in ('wt', 'ko'):
        _tif(str(root / 'run1' / name / 'fov01_C1.tif'))
    plan = cv.plan(cv.scan(str(root)))
    text = '\n'.join(plan.notes + plan.warnings)
    assert "'wt'" in text and "'ko'" in text
    assert 'A01' in text and 'A02' in text
    assert 'source_well' in text


def test_a_well_shaped_name_on_no_standard_plate_is_a_warning(tmp_path):
    """``ZZ99`` reads as a position, but no plate has it. Say so, by name."""
    root = tmp_path / 'src'
    for name in ('ZZ99', 'A01'):
        _tif(str(root / 'run1' / name / 'fov01_C1.tif'))
    plan = cv.plan(cv.scan(str(root)))
    warning = '\n'.join(plan.warnings)
    assert 'ZZ99' in warning
    assert 'r702' in warning and 'c99' in warning
    # It still converts — with an address that is recorded, not invented in
    # silence.
    assert plan.ok


def test_the_plan_never_reports_a_real_well_as_synthetic(tmp_path):
    root = tmp_path / 'src'
    for name in ('Q01', 'AF48'):
        _tif(str(root / 'run1' / name / 'fov01_C1.tif'))
    plan = cv.plan(cv.scan(str(root)))
    text = '\n'.join(plan.notes + plan.warnings)
    assert 'synthetic' not in text.lower()
    assert 'no standard plate' not in text


# ---------------------------------------------------------------------------
# spacr.io — the other synthetic assignment
# ---------------------------------------------------------------------------

_SEP_REGEX = (r"(?P<wellID>[A-Za-z0-9]+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
              r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")


def test_convert_separate_files_keeps_a_real_well_past_384(tmp_path):
    """Before: ``Q01`` and ``AA13`` were thrown away and renumbered A01/A02."""
    from spacr.io import convert_separate_files_to_yokogawa

    folder = tmp_path / 'raw'
    folder.mkdir()
    for well in ('Q01', 'AA13', 'A25'):
        _tif(str(folder / f'{well}_F1_T1_C1_Z1.tif'))

    convert_separate_files_to_yokogawa(str(folder), _SEP_REGEX)

    produced = sorted(p.name for p in folder.glob('plate1_*.tif'))
    assert produced == ['plate1_A25_T0001F001L01C01.tif',
                        'plate1_AA13_T0001F001L01C01.tif',
                        'plate1_Q01_T0001F001L01C01.tif']


def test_convert_separate_files_never_hands_out_the_same_well_twice(tmp_path):
    """386 regions used to collide: two of them both got ``plate2_A01``.

    Before: 385 output TIFFs for 386 inputs — the 386th overwrote the 385th,
    with nothing said. ``_get_next_well`` fell out of its loop and returned
    ``plate2_A01`` unconditionally, never checking whether it was taken.
    """
    from spacr.io import convert_separate_files_to_yokogawa

    folder = tmp_path / 'raw'
    folder.mkdir()
    for i in range(1, 387):
        _tif(str(folder / f'{i}_F1_T1_C1_Z1.tif'))

    convert_separate_files_to_yokogawa(str(folder), _SEP_REGEX)

    produced = [p.name for p in folder.glob('plate*_*.tif')]
    assert len(produced) == len(set(produced)) == 386
    log = pd.read_csv(folder / 'rename_log.csv')
    assert len(log) == 386
    assert len(set(log['Renamed TIFF'])) == 386


def test_convert_separate_files_assignment_is_deterministic(tmp_path):
    """The same folder converts to the same wells twice.

    Assignment used to follow ``os.listdir`` order, so which source became
    ``A01`` depended on the filesystem.
    """
    from spacr.io import convert_separate_files_to_yokogawa

    runs = []
    for attempt in range(2):
        folder = tmp_path / f'raw{attempt}'
        folder.mkdir()
        for i in range(1, 8):
            _tif(str(folder / f'{i}_F1_T1_C1_Z1.tif'), value=i)
        convert_separate_files_to_yokogawa(str(folder), _SEP_REGEX)
        log = pd.read_csv(folder / 'rename_log.csv')
        runs.append(dict(zip(log['Original File(s)'], log['Renamed TIFF'])))
    assert runs[0] == runs[1]
    assert runs[0]['1_F1_T1_C1_Z1.tif'] == 'plate1_A01_T0001F001L01C01.tif'
    assert runs[0]['7_F1_T1_C1_Z1.tif'] == 'plate1_A07_T0001F001L01C01.tif'
