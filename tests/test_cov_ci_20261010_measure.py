"""Overload propagation, write-queue budget, verdict dedupe and writer cleanup in measure."""
from __future__ import annotations

import logging
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M
import spacr.database_concurrency as database_concurrency
from spacr.errors import RunLedger


def _merged_field(size=64):
    yy, xx = np.mgrid[:size, :size]
    cell = np.zeros((size, size), np.uint16)
    nucleus = np.zeros((size, size), np.uint16)
    for i, (cy, cx) in enumerate([(18, 18), (18, 46)], start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 100] = i
        nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] = i
    pathogen = np.zeros((size, size), np.uint16)
    pathogen[(yy - 18) ** 2 + (xx - 18) ** 2 <= 9] = 1
    rng = np.random.default_rng(0)
    chans = []
    for _ in range(4):
        base = rng.integers(50, 200, size=(size, size)).astype(np.uint16)
        base[cell > 0] += 3000
        chans.append(base)
    return np.stack(chans + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


@pytest.fixture
def merged_project(tmp_path):
    merged = tmp_path / 'merged'
    merged.mkdir()
    (tmp_path / 'measurements').mkdir()
    np.save(merged / 'plate1_A01_F001.npy', _merged_field())
    return tmp_path


def _settings(merged_dir, **over):
    from spacr.settings import get_measure_crop_settings
    s = get_measure_crop_settings(settings={})
    s.update({
        'src': str(merged_dir), 'channels': [0, 1, 2, 3],
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'png_dims': [0, 1, 2], 'png_size': [32, 32],
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'test_mode': False, 'cytoplasm': True,
        'n_jobs': 1, 'database_write_queue_gib': 0.5,
    })
    s.update(over)
    return s


# --------------------------------------------------------------------------
# _measure_crop_core: an overload inside a captured write packet propagates
# --------------------------------------------------------------------------

def _overloaded_rescale(*args, **kwargs):
    raise MemoryError('cannot allocate memory')


def test_overload_inside_a_write_capture_propagates_for_the_retry_queue(
        merged_project, monkeypatch):
    monkeypatch.setattr(M, '_resolve_intensity_rescale_record', _overloaded_rescale)
    settings = _settings(merged_project / 'merged')
    with database_concurrency._capture_write_packet():
        assert database_concurrency._write_capture_active()
        with pytest.raises(MemoryError, match='cannot allocate memory'):
            M._measure_crop_core(0, [], 'plate1_A01_F001.npy', settings)


def test_overload_outside_a_write_capture_is_reported_as_a_failed_field(
        merged_project, monkeypatch, capsys):
    monkeypatch.setattr(M, '_resolve_intensity_rescale_record', _overloaded_rescale)
    settings = _settings(merged_project / 'merged')
    assert not database_concurrency._write_capture_active()
    result = M._measure_crop_core(0, [], 'plate1_A01_F001.npy', settings)
    assert result[0] == 0
    assert result[2] == 0
    assert 'MemoryError: cannot allocate memory' in result[4]
    assert 'plate1_A01_F001.npy failed' in capsys.readouterr().out


# --------------------------------------------------------------------------
# _measure_write_queue_budget
# --------------------------------------------------------------------------

def test_write_queue_budget_falls_back_to_one_gib_without_the_qt_preferences(
        monkeypatch):
    monkeypatch.setitem(sys.modules, 'spacr.qt.preferences', None)
    assert M._measure_write_queue_budget({}) == 1.0
    assert M._measure_write_queue_budget({'database_write_queue_gib': '2'}) == 2.0


@pytest.mark.parametrize('value', [-1, 65, float('nan'), float('inf')])
def test_write_queue_budget_outside_the_allowed_range_is_refused(value):
    with pytest.raises(ValueError, match='between 0 and 64 GiB'):
        M._measure_write_queue_budget({'database_write_queue_gib': value})


# --------------------------------------------------------------------------
# measure_crop orchestration with an in-process pool and writer
# --------------------------------------------------------------------------

class _Manager:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def list(self):
        return []

    def Event(self):
        import threading
        return threading.Event()


class _Result:
    def __init__(self, value):
        self.value = value

    def get(self, timeout=None):
        return self.value


class _Pool:
    close_error = None
    submitted = []

    def __init__(self, jobs, context=None, **options):
        self.options = options
        options['initializer'](*options['initargs'])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def apply_async(self, function, args=()):
        index, time_ls, file, settings = args[:4]
        _Pool.submitted.append((function, file))
        time_ls.append(0.01)
        return _Result((index, 0.01, np.array([1, 2]), None, '', 'ticket-' + file))

    def close(self):
        if _Pool.close_error is not None:
            raise _Pool.close_error

    def join(self):
        pass


class _Writer:
    instances = []
    finish_error = None
    repeat_verdict = 1

    def __init__(self, path, commit, callback, ram_gib, context=None):
        self.path = path
        self.callback = callback
        self.ram_gib = ram_gib
        self.endpoint = self
        self.events = []
        self.queued = []
        _Writer.instances.append(self)

    def enqueue(self, item, operations):
        self.queued.append(item)
        return 'ticket'

    def start(self):
        self.events.append('start')

    def cancel(self):
        self.events.append('cancel')

    def finish(self):
        self.events.append('finish')
        if _Writer.finish_error is not None:
            raise _Writer.finish_error
        for _ in range(_Writer.repeat_verdict):
            self.callback('plate1_A01_F001.npy', 'ticket', None)


@pytest.fixture
def inline_pool(monkeypatch):
    _Pool.close_error = None
    _Pool.submitted = []
    _Writer.instances = []
    _Writer.finish_error = None
    _Writer.repeat_verdict = 1
    monkeypatch.setattr(M, '_parallel_pool', _Pool)
    monkeypatch.setattr(M, '_start_manager', lambda ctx: _Manager())
    monkeypatch.setattr(database_concurrency, '_DatabaseWriteQueue', _Writer)
    successes = []
    original = RunLedger.record_success

    def spy(self, item, stage=None):
        successes.append((item, stage))
        return original(self, item, stage=stage)

    monkeypatch.setattr(RunLedger, 'record_success', spy)
    return successes


def test_a_repeated_writer_verdict_counts_the_field_once(
        merged_project, inline_pool, monkeypatch, capsys):
    _Writer.repeat_verdict = 2
    monkeypatch.setattr(M, '_ram_guard_plan', lambda src, jobs: None)
    seen = []

    def no_wells(db_path, threshold=None):
        seen.append(db_path)
        return pd.DataFrame(columns=['monolayer_ok'])

    monkeypatch.setattr(M, '_aggregate_confluency_by_well', no_wells)
    db = merged_project / 'measurements' / 'measurements.db'
    sqlite3.connect(db).close()
    M.measure_crop(_settings(merged_project / 'merged', confluency=True))

    writer, = _Writer.instances
    assert writer.events == ['start', 'finish']
    assert writer.ram_gib == 0.5
    assert [file for _, file in _Pool.submitted] == ['plate1_A01_F001.npy']
    assert _Pool.submitted[0][0] is M._measure_crop_queued
    assert inline_pool == [('plate1_A01_F001.npy', 'measure_write')]
    assert seen == [str(db)]
    assert 'Confluency:' not in capsys.readouterr().out


def test_a_writer_that_fails_while_stopping_does_not_mask_the_run_error(
        merged_project, inline_pool, caplog):
    _Pool.close_error = RuntimeError('pool could not close')
    _Writer.finish_error = OSError('queue disk vanished')
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match='pool could not close'):
            M.measure_crop(_settings(merged_project / 'merged'))
    writer, = _Writer.instances
    assert writer.events == ['start', 'cancel', 'finish']
    assert any('Database writer stopped: queue disk vanished' in record.getMessage()
               for record in caplog.records)
    assert inline_pool == []


def test_a_non_list_source_after_normalization_measures_nothing(
        monkeypatch, tmp_path):
    import spacr.utils as utils
    import spacr.ome_zarr as ome_zarr

    monkeypatch.setattr(utils, 'normalize_src_path', lambda src: (src,))
    monkeypatch.setattr(ome_zarr, '_needs_cloud_run', lambda settings: False)
    monkeypatch.setattr(M, 'run_context', lambda *a, **k: pytest.fail('ran'))
    assert M.measure_crop(_settings(tmp_path / 'merged')) is None


# --------------------------------------------------------------------------
# _validate_measurement_calibration_history: owned tables without rows
# --------------------------------------------------------------------------

def test_empty_owned_tables_do_not_count_as_existing_measurements(tmp_path):
    from spacr.resume import MEASURE_OWNED_TABLES

    db = tmp_path / 'measurements.db'
    owned = sorted(MEASURE_OWNED_TABLES - {'png_list', 'intensity_rescale'})
    with sqlite3.connect(db) as connection:
        for table in owned[:2]:
            connection.execute(f'CREATE TABLE "{table}" (prcfo TEXT, value REAL)')
    settings = {M._CALIBRATION_IDENTITY_KEY: 'new-identity'}
    assert M._validate_measurement_calibration_history(settings, str(db)) is None


# --------------------------------------------------------------------------
# _cellprofiler_tables: cached overlaps and tied role candidates
# --------------------------------------------------------------------------

def _cp_reply(tmp_path, planes, rows, columns, *, name, labels=None):
    merged = tmp_path / 'merged'
    merged.mkdir()
    stem = 'plate1_A01_1'
    np.save(merged / (stem + '.npy'), np.stack(planes, axis=-1))
    table = tmp_path / 'objects.npy'
    np.save(table, np.asarray(rows, float))
    reply = {'images': {'1': [stem + '_ch0.tif']},
             'objects': {name: {'columns': columns, 'path': str(table)}}}
    if labels is not None:
        label_path = tmp_path / 'labels.npy'
        np.save(label_path, labels)
        reply['labels'] = {'1': {name: [str(label_path)]}}
    return merged, reply


def test_supplied_label_overlaps_are_computed_once_per_image(tmp_path, monkeypatch):
    spacr_mask = np.zeros((8, 8), np.uint16)
    spacr_mask[:, :4] = 17
    spacr_mask[:, 4:] = 18
    cp_labels = np.zeros((8, 8), np.uint16)
    cp_labels[:, :4] = 1
    cp_labels[:, 4:] = 2
    merged, reply = _cp_reply(
        tmp_path, [spacr_mask], [[1, 1, 32], [1, 2, 32]],
        ['ImageNumber', 'ObjectNumber', 'AreaShape_Area'], name='Cells',
        labels=cp_labels)
    calls = []
    original = M._cellprofiler_overlap_labels

    def counting(paths, mask):
        calls.append(paths)
        return original(paths, mask)

    monkeypatch.setattr(M, '_cellprofiler_overlap_labels', counting)
    frame = M._cellprofiler_tables(reply, merged, {'cell_mask_dim': 0})[
        'cellprofiler_cells']
    assert frame['object_label'].tolist() == [17, 18]
    assert frame['object_type'].tolist() == ['cell', 'cell']
    assert len(calls) == 1


def test_an_unnamed_object_tied_between_roles_keeps_the_first_role(tmp_path):
    cell = np.zeros((8, 8), np.uint16)
    cell[1:4, 1:4] = 5
    nucleus = np.zeros((8, 8), np.uint16)
    nucleus[1:4, 1:4] = 9
    merged, reply = _cp_reply(
        tmp_path, [cell, nucleus], [[1, 1, 2.0, 2.0]],
        ['ImageNumber', 'ObjectNumber', 'Location_Center_X', 'Location_Center_Y'],
        name='Speckles')
    frame = M._cellprofiler_tables(
        reply, merged, {'cell_mask_dim': 0, 'nucleus_mask_dim': 1})[
        'cellprofiler_speckles']
    assert frame['object_type'].tolist() == ['cell']
    assert frame['object_label'].tolist() == [5]


def test_an_unnamed_object_on_only_one_role_picks_that_role(tmp_path):
    cell = np.zeros((8, 8), np.uint16)
    cell[1:4, 1:4] = 5
    nucleus = np.zeros((8, 8), np.uint16)
    nucleus[5:7, 5:7] = 9
    merged, reply = _cp_reply(
        tmp_path, [cell, nucleus], [[1, 1, 2.0, 2.0], [1, 2, 1.0, 1.0]],
        ['ImageNumber', 'ObjectNumber', 'Location_Center_X', 'Location_Center_Y'],
        name='Speckles')
    frame = M._cellprofiler_tables(
        reply, merged, {'cell_mask_dim': 0, 'nucleus_mask_dim': 1})[
        'cellprofiler_speckles']
    assert frame['object_type'].tolist() == ['cell', 'cell']
    assert frame['object_label'].tolist() == [5, 5]


# --------------------------------------------------------------------------
# _fit_dna_content: a peak with no responsibility keeps its position
# --------------------------------------------------------------------------

def test_dna_peaks_without_responsibility_keep_their_seeded_positions(monkeypatch):
    rng = np.random.default_rng(3)
    content = np.concatenate([rng.normal(100, 6, 300), rng.normal(200, 12, 150),
                              rng.uniform(110, 190, 80)])
    x = content[content > 0]
    g1_seed = M._dna_seed(x)
    kept = x[(x >= M._CELL_CYCLE_FIT_RANGE[0] * g1_seed)
             & (x <= M._CELL_CYCLE_FIT_RANGE[1] * g1_seed)]
    low, high = M._CELL_CYCLE_RATIO_BOUNDS
    g2_seed = float(np.clip(M._dna_g2_seed(kept, g1_seed),
                            low * g1_seed, high * g1_seed))
    original = M._DnaFit.densities

    def without_peaks(self, values):
        dens = np.array(original(self, values), dtype=float, copy=True)
        dens[:, 0] = 0.0
        dens[:, 2] = 0.0
        return dens

    monkeypatch.setattr(M._DnaFit, 'densities', without_peaks)
    fit = M._fit_dna_content(content, gates=[2.5, 3.5])
    assert fit.g1 == pytest.approx(g1_seed)
    assert fit.g2 == pytest.approx(g2_seed)
    assert fit.sd1 == pytest.approx(0.02 * g1_seed)
    assert fit.gates == pytest.approx((2.5 * g1_seed / 2, 3.5 * g1_seed / 2))
    assert fit.fitted_gates is False
