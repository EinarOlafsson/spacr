"""A database written by more than one run must remember all of them.

``spacr.io._save_settings_to_db`` writes the ``settings`` table with
``if_exists='replace'``. That is right for the *current* view --- it is what
``spacr.resume.read_recorded_settings`` compares against before a resume, and
that comparison is only meaningful over exactly one run's settings --- but on
its own it means only the last stage survives.

Two things go with it. A source folder measured twice, once for cell crops and
once for pathogen crops, appends both sets of rows to the same ``png_list``,
and afterwards the database says only how the second set was made. And the
call happens *before* the first field is measured, so a run that recorded its
settings and then died replaced the settings of the run that actually produced
the rows on disk.

``settings_history`` is appended on every save, so ``settings`` keeps its
meaning and nothing is lost.

Databases here are built by the real writers ---
``_save_settings_to_db`` itself, plus ``filepaths_to_database`` and
``_merge_and_save_to_database`` for the measurement rows those settings
describe.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.io import (SETTINGS_HISTORY_COLUMNS, SETTINGS_HISTORY_TABLE,
                      SETTINGS_TABLE, _save_settings_to_db,
                      read_settings_history)


@pytest.fixture()
def src(tmp_path):
    """``<root>/stack``, so the database lands in ``<root>/measurements``."""
    path = tmp_path / 'stack'
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def db_for(src_path):
    return os.path.join(os.path.dirname(src_path), 'measurements',
                        'measurements.db')


def measure_a_field(src_path, stem, crop_mode='cell'):
    """Rows for those settings to describe, via the writers measure calls."""
    from spacr.utils import _merge_and_save_to_database, filepaths_to_database

    root = os.path.dirname(src_path)
    labels = [1, 2, 3]
    morphology = pd.DataFrame({'label': labels,
                               'cell_area': [100.0, 101.0, 102.0]})
    intensity = pd.DataFrame({'label': labels,
                              'cell_channel_0_mean_intensity': [5.0] * 3})
    _merge_and_save_to_database(morphology, intensity, 'cell', root, stem,
                                'exp', False)
    folder = os.path.join(root, 'data', f'{crop_mode}_png')
    os.makedirs(folder, exist_ok=True)
    filepaths_to_database(
        [os.path.join(folder, f'{stem}_{i}.png') for i in labels],
        {'timelapse': False}, root, crop_mode)


def settings_rows(db):
    con = sqlite3.connect(db)
    try:
        return dict(con.execute(
            f'SELECT setting_key, setting_value FROM "{SETTINGS_TABLE}"'))
    finally:
        con.close()


def tables_in(db):
    con = sqlite3.connect(db)
    try:
        return sorted(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. the finding
# ---------------------------------------------------------------------------

def test_two_stages_both_survive(src):
    """1 of 2 recoverable before this table existed; 2 of 2 now."""
    _save_settings_to_db({'src': src, 'crop_mode': ['cell'],
                          'channels': [0, 1], 'experiment': 'stage-one'},
                         stage='measure_crop')
    measure_a_field(src, 'plate1_A01_1', 'cell')
    _save_settings_to_db({'src': src, 'crop_mode': ['pathogen'],
                          'channels': [0, 2], 'experiment': 'stage-two'},
                         stage='measure_crop')
    measure_a_field(src, 'plate1_A01_2', 'pathogen')

    db = db_for(src)
    history = read_settings_history(db)

    assert len(history) == 2
    assert [h['settings']['experiment'] for h in history] == ['stage-one',
                                                              'stage-two']
    assert [h['settings']['crop_mode'] for h in history] == ["['cell']",
                                                             "['pathogen']"]
    assert [h['settings']['channels'] for h in history] == ['[0, 1]', '[0, 2]']
    # the two runs are distinguishable even before you read their settings
    assert len({h['run_id'] for h in history}) == 2
    assert all(h['stage'] == 'measure_crop' for h in history)
    assert all(h['stamped_utc'] for h in history)


def test_the_settings_table_still_means_the_latest_run(src):
    """Unchanged, deliberately: ``resume`` compares against exactly this."""
    from spacr.resume import read_recorded_settings

    _save_settings_to_db({'src': src, 'crop_mode': ['cell'],
                          'experiment': 'stage-one'})
    _save_settings_to_db({'src': src, 'crop_mode': ['pathogen'],
                          'experiment': 'stage-two'})
    db = db_for(src)

    assert settings_rows(db)['experiment'] == 'stage-two'
    assert settings_rows(db)['crop_mode'] == "['pathogen']"
    recorded = read_recorded_settings(db)
    assert recorded['experiment'] == 'stage-two'
    # exactly the latest run's keys, not a union across stages
    assert set(recorded) == {'src', 'crop_mode', 'experiment'}


def test_a_run_that_dies_before_measuring_no_longer_erases_its_predecessor(src):
    """The order of operations is the whole hazard.

    ``_save_settings_to_db`` runs *before* the first field. A run that records
    and then dies leaves a database whose ``settings`` describe a run that
    produced nothing, and whose rows were produced by settings that are gone.
    """
    real = {'src': src, 'channels': [0, 1], 'cell_mask_dim': 4,
            'experiment': 'the-run-that-made-the-rows'}
    _save_settings_to_db(real, stage='measure_crop')
    measure_a_field(src, 'plate1_A01_1')

    # a second run starts, records, and is killed before it measures anything
    _save_settings_to_db({'src': src, 'channels': [0, 2], 'cell_mask_dim': 4,
                          'experiment': 'killed-before-it-measured'},
                         stage='measure_crop')

    db = db_for(src)
    assert settings_rows(db)['experiment'] == 'killed-before-it-measured'
    history = read_settings_history(db)
    assert [h['settings']['experiment'] for h in history] == [
        'the-run-that-made-the-rows', 'killed-before-it-measured']
    assert history[0]['settings']['channels'] == '[0, 1]'


# ---------------------------------------------------------------------------
# 2. what is already on disk
# ---------------------------------------------------------------------------

def test_a_database_written_before_the_history_existed_keeps_its_snapshot(src):
    """Migrate the old content rather than leaving it to be overwritten."""
    db = db_for(src)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    con = sqlite3.connect(db)
    try:
        pd.DataFrame({'setting_key': ['src', 'crop_mode', 'channels'],
                      'setting_value': [src, "['cell']", '[0, 1]']}
                     ).to_sql(SETTINGS_TABLE, con, if_exists='replace',
                              index=False)
    finally:
        con.close()

    _save_settings_to_db({'src': src, 'crop_mode': ['pathogen'],
                          'channels': [0, 2]}, stage='measure_crop')

    history = read_settings_history(db)
    assert len(history) == 2
    assert history[0]['stage'] == 'before-history'
    assert history[0]['settings']['crop_mode'] == "['cell']"
    assert history[1]['stage'] == 'measure_crop'
    assert history[1]['settings']['crop_mode'] == "['pathogen']"


def test_the_legacy_snapshot_is_copied_once_not_on_every_save(src):
    db = db_for(src)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    con = sqlite3.connect(db)
    try:
        pd.DataFrame({'setting_key': ['src'], 'setting_value': [src]}
                     ).to_sql(SETTINGS_TABLE, con, if_exists='replace',
                              index=False)
    finally:
        con.close()

    for i in range(3):
        _save_settings_to_db({'src': src, 'pass': i}, stage='measure_crop')

    history = read_settings_history(db)
    assert [h['stage'] for h in history] == ['before-history'] + \
        ['measure_crop'] * 3


def test_reading_a_database_with_no_history_returns_nothing(tmp_path):
    assert read_settings_history(tmp_path / 'nope.db') == []
    empty = tmp_path / 'empty.db'
    sqlite3.connect(str(empty)).close()
    assert read_settings_history(empty) == []


# ---------------------------------------------------------------------------
# 3. shape, naming, and not stepping on anyone else
# ---------------------------------------------------------------------------

def test_the_history_table_reads_like_the_table_it_archives(src):
    _save_settings_to_db({'src': src, 'channels': [0, 1]}, stage='measure_crop')
    db = db_for(src)
    con = sqlite3.connect(db)
    try:
        columns = [r[1] for r in con.execute(
            f'PRAGMA table_info("{SETTINGS_HISTORY_TABLE}")')]
        pairs = dict(con.execute(
            f'SELECT setting_key, setting_value '
            f'FROM "{SETTINGS_HISTORY_TABLE}"'))
    finally:
        con.close()
    assert tuple(columns) == SETTINGS_HISTORY_COLUMNS
    assert pairs == settings_rows(db)


def test_the_stage_label_falls_back_without_inventing_one(src):
    _save_settings_to_db({'src': src, 'a': 1})
    _save_settings_to_db({'src': src, 'a': 2, 'stage': 'mask'})
    _save_settings_to_db({'src': src, 'a': 3, 'module': 'measure'})
    assert [h['stage'] for h in read_settings_history(db_for(src))] == [
        'unknown', 'mask', 'measure']


def test_a_measure_resume_never_clears_the_history(src):
    """``resume`` deletes per-field rows; this table has no field."""
    from spacr.resume import MEASURE_OWNED_TABLES, NON_FIELD_TABLES

    assert SETTINGS_HISTORY_TABLE not in MEASURE_OWNED_TABLES
    _save_settings_to_db({'src': src, 'channels': [0, 1]}, stage='measure_crop')
    measure_a_field(src, 'plate1_A01_1')

    from spacr.resume import discover_field_tables

    db = db_for(src)
    assert SETTINGS_HISTORY_TABLE in tables_in(db)
    assert SETTINGS_HISTORY_TABLE not in discover_field_tables(db)
    assert SETTINGS_HISTORY_TABLE not in discover_field_tables(db,
                                                               owned_only=False)
    # the deny-list is the historical belt-and-braces, not the mechanism
    assert SETTINGS_TABLE in NON_FIELD_TABLES


def test_the_connection_is_closed_even_when_the_write_fails(src, monkeypatch):
    """An open connection holds the lock, and workers start writing next."""
    import spacr.io as IO

    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(IO.sqlite3, 'connect', tracking_connect)

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError('disk I/O error')

    monkeypatch.setattr(pd.DataFrame, 'to_sql', explode)
    with pytest.raises(sqlite3.OperationalError):
        _save_settings_to_db({'src': src, 'a': 1}, stage='measure_crop')

    assert opened, 'no connection was opened'
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute('SELECT 1')
