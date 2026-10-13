"""Plate recovery must produce newly requested evidence, not merely skip work."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import types

import pytest


_SPEC = importlib.util.spec_from_file_location(
    'ops_plate_driver', Path(__file__).resolve().parents[1] / 'tools/run_ops_plate.py')
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _record(well, stored, count):
    report = {'stitch': {}, 'phenotype': {}, 'objects': {},
              'decode': {'ops_reads_rows': count}}
    return {'plate': 'plate', 'well': well, 'complete': True,
            'settings': {'ops_store_reads': stored},
            'report': report, 'row': driver.well_row(report)}


@pytest.mark.parametrize('stored,reported,actual,requested,reruns', [
    (False, None, None, True, ['B2']),
    (True, 2, None, True, ['B2']),
    (True, 2, 1, True, ['B2']),
    (True, 2, 2, True, []),
    (False, None, None, False, []),
    (True, None, 2, True, ['B2']),
    (True, 0, 0, True, []),
])
def test_real_driver_resumes_only_wells_with_requested_reads(
        tmp_path, monkeypatch, capsys, stored, reported, actual, requested, reruns):
    out = tmp_path / 'output'
    results = out / 'results'
    results.mkdir(parents=True)
    design = tmp_path / 'design.csv'
    design.write_text('dialout,sgRNA,prefix_length\n0,ACGTACGT,4\n')
    for well in ('B1', 'B2'):
        (results / f'{well}.json').write_text(json.dumps(_record(well, stored, reported)))
    b1_before = (results / 'B1.json').read_bytes()
    db = out / 'measurements.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE ops_barcodes (plate TEXT, well TEXT)')
        conn.executemany('INSERT INTO ops_barcodes VALUES (?,?)', [('plate', 'B1'), ('plate', 'B2')])
        if actual is not None:
            conn.execute('CREATE TABLE ops_reads (plate TEXT, well TEXT)')
            conn.executemany('INSERT INTO ops_reads VALUES (?,?)',
                             [('plate', 'B2')] * actual + [('different', 'B2')] * 3 + [('plate', 'B1')] * 4)
    called = []

    def run_ops(settings, wells, phases, library):
        assert settings['ops_store_reads'] and not settings['ops_gpu']
        assert tuple(phases) == driver.PHASES
        assert Path(library).read_text().splitlines() == ['prefix', 'ACGT']
        called.extend(wells)
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS ops_reads (plate TEXT, well TEXT)')
            for well in wells:
                conn.execute('DELETE FROM ops_reads WHERE plate=? AND well=?', ('plate', well))
                conn.executemany('INSERT INTO ops_reads VALUES (?,?)', [('plate', well)] * 2)
        return {'wells': {well: _record(well, True, 2)['report'] for well in wells}}

    monkeypatch.setitem(sys.modules, 'spacr.ops_engine', types.SimpleNamespace(
        run_ops=run_ops, _index_tiles=lambda *args: {'B1': {}, 'B2': {}}))
    args = ['--plate', str(tmp_path / 'plate'), '--design', str(design),
            '--out', str(out), '--no-gpu']
    if requested:
        args += ['--store-reads', 'b2']
    assert driver.main(args) == 0
    assert called == reruns
    assert (results / 'B1.json').read_bytes() == b1_before
    summary = json.loads((out / 'plate_summary.json').read_text())
    assert summary['store_reads_wells'] == (['B2'] if requested else [])
    assert summary['failed'] == {}
    if reruns:
        assert 'requested stored reads are missing or incomplete; rerunning' in capsys.readouterr().out
        called.clear()
        assert driver.main(args) == 0
        assert called == []


@pytest.mark.parametrize('reported', [None, -1, True, '2', 2.0])
def test_invalid_saved_counts_never_establish_read_completeness(tmp_path, reported):
    db = tmp_path / 'missing.db'
    assert not driver._stored_reads_match(db, 'plate', 'B2', _record('B2', True, reported))
    assert not db.exists()


def test_read_check_does_not_create_missing_database(tmp_path):
    db = tmp_path / 'missing.db'
    assert not driver._stored_reads_match(db, 'plate', 'B2', _record('B2', True, 0))
    assert not db.exists()
