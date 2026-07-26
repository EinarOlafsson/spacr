"""The object assays' field key must not merge a timelapse's frames.

``prcf`` is the unit of observation for both per-object assays in
:mod:`spacr.submodules`. The replication assay builds every vacuole id from
it; the invasion assay computes one outside-stain threshold per ``prcf``. On a
timelapse the key is ``plate_row_column_field_TIME`` — five tokens — which is
what :func:`spacr.utils._map_wells`, the writer that puts ``prcf`` into the
measurements database, actually writes.

Both assays rebuilt a **four**-token key whenever the column was absent, which
is exactly the state ``change_plate=True`` arranges: it drops the database's
own ``prcf`` so the relabelled plate cannot disagree with it. A four-token key
on a timelapse names a stack, not a frame, so every timepoint of a field folded
into one observation. Same defect class as ``utils._split_data`` (fixed in
5a64981), one level up.

Every database here is written by ``utils._merge_and_save_to_database`` — the
real writer — from real spaCR stack names. The pre-fix behaviour is not
narrated: ``_legacy_field_key`` is the verbatim four-line rebuild those two
assays used to run, monkeypatched back in so the whole assay executes exactly
as it did, against the same database, and the damage is asserted as numbers.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io  # noqa: E402,F401
import spacr.plot  # noqa: E402,F401
import spacr.settings  # noqa: E402,F401
import spacr.submodules as submodules  # noqa: E402
from spacr.submodules import analyze_invasion, analyze_replication  # noqa: E402
from spacr.utils import _merge_and_save_to_database  # noqa: E402

PLATE = "plate1"
PARASITE_AREA = 100.0
PARASITE_DIAMETER = 2.0 * np.sqrt(PARASITE_AREA / np.pi)


@pytest.fixture(autouse=True)
def _no_blocking_show_and_clean_figs(monkeypatch):
    """Never open a window, never let figures accumulate."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the code as it was
# ---------------------------------------------------------------------------

def _legacy_field_key(df, source=None, verbose=False):
    """``analyze_replication`` / ``analyze_invasion``'s prcf rebuild, verbatim.

    Copied character for character out of the pre-fix source, with the same
    signature as :func:`spacr.submodules._ensure_field_key` so it can be
    swapped in and the rest of the assay run untouched.
    """
    if 'prcf' not in df.columns:
        df['prcf'] = (df['plateID'].astype(str) + '_' + df['rowID'].astype(str)
                      + '_' + df['columnID'].astype(str) + '_'
                      + df['fieldID'].astype(str))
    return df


# ---------------------------------------------------------------------------
# databases — real writer only
# ---------------------------------------------------------------------------

def _stem(well, field, time):
    """The stack file-name stem ``_merge_and_save_to_database`` parses."""
    if time is None:
        return f"{PLATE}_{well}_{field}"
    return f"{PLATE}_{well}_{field}_{time}"


def _write_frame(root, well, field, time, parasites, cells=True):
    """Append one field (one frame) of parasite + cell measurements.

    ``parasites`` is a list of dicts, one per parasite, with the per-object
    values that matter to whichever assay is reading: ``cy``/``cx`` centroid
    for the replication assay's spatial clustering, ``outside`` for the
    invasion assay's stain channel, and ``cell`` for the host-cell link.
    """
    timelapse = time is not None
    labels = list(range(1, len(parasites) + 1))
    morph = pd.DataFrame({
        'label': labels,
        'cell_id': [p.get('cell', label) for p, label in zip(parasites, labels)],
        'pathogen_area': [PARASITE_AREA] * len(parasites),
        'pathogen_equivalent_diameter_area': [PARASITE_DIAMETER] * len(parasites),
        'pathogen_channel_0_centroid_weighted-0': [p.get('cy', 50.0) for p in parasites],
        'pathogen_channel_0_centroid_weighted-1': [p.get('cx', 50.0) for p in parasites],
    })
    intensity = pd.DataFrame({
        'label': labels,
        'pathogen_channel_0_mean_intensity': [500.0] * len(parasites),
        'pathogen_channel_1_mean_intensity': [p.get('outside', 1.0) for p in parasites],
    })
    _merge_and_save_to_database(morph, intensity, 'pathogen', root,
                                _stem(well, field, time), 'exp',
                                timelapse=timelapse)
    if cells:
        host_labels = sorted({p.get('cell', label)
                              for p, label in zip(parasites, labels)})
        _merge_and_save_to_database(
            pd.DataFrame({'label': host_labels,
                          'cell_area': [1000.0] * len(host_labels)}),
            pd.DataFrame({'label': host_labels,
                          'cell_channel_0_mean_intensity': [5.0] * len(host_labels)}),
            'cell', root, _stem(well, field, time), 'exp', timelapse=timelapse)


def _replication_db(root, times=(1, 2, 3), wells=("A01", "A02")):
    """One field per well, two host cells, ONE parasite in each, per frame.

    The two host cells sit 400 px apart, far past the ~17 px single-linkage
    threshold the default ``link_factor`` derives from this parasite size, so
    the vacuole count is known by construction: one vacuole of one parasite per
    (well, frame, host cell). With three frames that is 2 x 3 x 2 = 12 vacuoles
    of exactly 1 parasite, every one of them in the ``'1'`` bucket.
    """
    os.makedirs(os.path.join(str(root), 'measurements'), exist_ok=True)
    for well in wells:
        for time in times:
            _write_frame(str(root), well, 1, time, [
                {'cell': 1, 'cy': 50.0, 'cx': 50.0},
                {'cell': 2, 'cy': 450.0, 'cx': 450.0},
            ])
    return os.path.join(str(root), 'measurements', 'measurements.db')


#: Per frame: (invaded stain level, how many, attached stain level, how many).
#: Invasion progresses 2/12 -> 6/12 -> 10/12 while the stain level drifts x3
#: per frame, which is the ordinary reason the threshold is per-field rather
#: than per-plate. Per-frame Otsu lands at 25 / 75 / 225; pooled across the
#: three frames it lands at 240 and calls almost everything invaded.
_INVASION_FRAMES = {1: (10.0, 2, 40.0, 10),
                    2: (30.0, 6, 120.0, 6),
                    3: (90.0, 10, 360.0, 2)}


def _invasion_db(root, times=(1, 2, 3), wells=("A01", "A02")):
    """One field per well, twelve parasites per frame, each in its own host."""
    os.makedirs(os.path.join(str(root), 'measurements'), exist_ok=True)
    for well in wells:
        for time in times:
            low, n_low, high, n_high = _INVASION_FRAMES[time or 1]
            parasites = [{'cell': i + 1, 'outside': level, 'cy': 50.0 * i,
                          'cx': 50.0 * i}
                         for i, level in enumerate([low] * n_low + [high] * n_high)]
            _write_frame(str(root), well, 1, time, parasites)
    return os.path.join(str(root), 'measurements', 'measurements.db')


def _replication_settings(root, **overrides):
    settings = {'src': str(root), 'save': False, 'verbose': False}
    settings.update(overrides)
    return settings


def _invasion_settings(root, **overrides):
    settings = {'src': str(root), 'save': False, 'verbose': False,
                'min_parasites_per_well': 10}
    settings.update(overrides)
    return settings


def _prcf_column(db, table='pathogen'):
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(f'SELECT prcf FROM {table}', conn)['prcf']
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# what the writer puts in the database
# ---------------------------------------------------------------------------

def test_the_writer_puts_the_timepoint_in_prcf(tmp_path):
    """The premise: a timelapse prcf has five tokens, and a plain one has four."""
    timelapse_db = _replication_db(tmp_path / 'tl')
    plain_db = _replication_db(tmp_path / 'plain', times=(None,))

    assert set(_prcf_column(timelapse_db)) == {
        f'{PLATE}_r1_{column}_f1_t{time}'
        for column in ('c1', 'c2') for time in (1, 2, 3)}
    assert set(_prcf_column(plain_db)) == {f'{PLATE}_r1_c1_f1', f'{PLATE}_r1_c2_f1'}


# ---------------------------------------------------------------------------
# replication
# ---------------------------------------------------------------------------

def test_replication_time_blind_key_merges_every_frame_into_one_vacuole(tmp_path, monkeypatch):
    """The pre-fix rebuild turns 12 vacuoles of 1 parasite into 4 of 3.

    Not a description of the bug: ``_legacy_field_key`` is the four-line
    rebuild as it was, and the whole assay runs on the same real database.
    The spatial clustering scopes on ``(prcf, cell_id)``, so with the timepoint
    gone the same host cell photographed at t1/t2/t3 becomes one group, and
    three separate single-parasite vacuoles are reported as one vacuole holding
    three parasites — off the endodyogeny ladder, so every one of them lands in
    the ``non_power_of_two`` bucket. The assay's headline number, the fraction
    of vacuoles that are a power of two, goes from 100 % to 0 %.
    """
    root = tmp_path / 'run'
    _replication_db(root)

    monkeypatch.setattr(submodules, '_ensure_field_key', _legacy_field_key)
    broken = analyze_replication(_replication_settings(root, change_plate=True))
    monkeypatch.undo()
    fixed = analyze_replication(_replication_settings(root, change_plate=True))

    broken_vacuoles = broken['vacuoles']
    assert len(broken_vacuoles) == 4
    assert broken_vacuoles['n_parasites'].tolist() == [3, 3, 3, 3]
    assert broken_vacuoles['replication_bucket'].astype(str).tolist() == \
        ['non_power_of_two'] * 4
    assert not broken_vacuoles['is_power_of_two'].any()
    assert sorted(set(broken_vacuoles['prcf'])) == [
        f'{PLATE}_r1_c1_f1', f'{PLATE}_r1_c2_f1']

    fixed_vacuoles = fixed['vacuoles']
    assert len(fixed_vacuoles) == 12
    assert fixed_vacuoles['n_parasites'].tolist() == [1] * 12
    assert fixed_vacuoles['replication_bucket'].astype(str).tolist() == ['1'] * 12
    assert fixed_vacuoles['is_power_of_two'].all()
    assert sorted(set(fixed_vacuoles['prcf'])) == [
        f'{PLATE}_r1_c1_f1_t1', f'{PLATE}_r1_c1_f1_t2', f'{PLATE}_r1_c1_f1_t3',
        f'{PLATE}_r1_c2_f1_t1', f'{PLATE}_r1_c2_f1_t2', f'{PLATE}_r1_c2_f1_t3']


def test_replication_change_plate_answers_what_the_database_key_answers(tmp_path):
    """``change_plate`` decides a plate label; it must not decide the biology.

    With ``change_plate=False`` the assay uses the database's own five-token
    ``prcf``; with ``True`` it drops it and rebuilds. The two must agree, and
    before the fix they did not: 12 vacuoles against 4.
    """
    root = tmp_path / 'run'
    _replication_db(root)

    kept = analyze_replication(_replication_settings(root, change_plate=False))
    rebuilt = analyze_replication(_replication_settings(root, change_plate=True))

    columns = ['prcf', 'n_parasites', 'replication_bucket', 'cell_id']
    pd.testing.assert_frame_equal(
        kept['vacuoles'][columns].sort_values(['prcf', 'cell_id'])
        .reset_index(drop=True),
        rebuilt['vacuoles'][columns].sort_values(['prcf', 'cell_id'])
        .reset_index(drop=True),
    )
    assert len(kept['vacuoles']) == 12


def test_replication_without_a_timelapse_is_byte_for_byte_unchanged(tmp_path):
    """A database with no timepoint column must come out exactly as it did.

    The old rebuild is run over the same non-timelapse database and the two
    answers are compared to each other, not to a hand-written expectation of
    what the old one produced.
    """
    root = tmp_path / 'run'
    _replication_db(root, times=(None,))

    fixed = analyze_replication(_replication_settings(root, change_plate=True))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(submodules, '_ensure_field_key', _legacy_field_key)
        legacy = analyze_replication(_replication_settings(root, change_plate=True))

    pd.testing.assert_frame_equal(legacy['vacuoles'], fixed['vacuoles'])
    assert set(fixed['vacuoles']['prcf']) == {f'{PLATE}_r1_c1_f1',
                                              f'{PLATE}_r1_c2_f1'}
    assert len(fixed['vacuoles']) == 4        # 2 wells x 2 host cells, one frame


# ---------------------------------------------------------------------------
# invasion
# ---------------------------------------------------------------------------

def test_invasion_time_blind_key_pools_frames_into_one_threshold(tmp_path, monkeypatch):
    """The pre-fix rebuild reports 94.4 % invasion where the truth is 50 %.

    Twelve parasites per frame, invasion progressing 2/12 -> 6/12 -> 10/12
    while the outside-stain level drifts x3 per frame. Per frame, Otsu lands at
    25 / 75 / 225 and reads the composition off correctly. With the timepoint
    gone all three frames become one 36-object 'field', Otsu lands at 240, and
    everything below the last frame's bright objects is called invaded.
    """
    root = tmp_path / 'run'
    _invasion_db(root)

    monkeypatch.setattr(submodules, '_ensure_field_key', _legacy_field_key)
    broken = analyze_invasion(_invasion_settings(root, change_plate=True))
    monkeypatch.undo()
    fixed = analyze_invasion(_invasion_settings(root, change_plate=True))

    broken_fields, fixed_fields = broken['fields'], fixed['fields']
    assert len(broken_fields) == 2
    assert broken_fields['n_objects'].tolist() == [36, 36]
    assert broken_fields['threshold'].tolist() == [240.0, 240.0]
    assert broken['wells']['invasion_efficiency'].round(4).tolist() == [0.9444, 0.9444]
    assert broken['wells']['n_fields'].tolist() == [1, 1]
    assert broken['parasites']['invasion_class'].value_counts().to_dict() == {
        'invaded': 68, 'attached': 4, 'unclassified': 0}

    assert len(fixed_fields) == 6
    assert fixed_fields['n_objects'].tolist() == [12] * 6
    assert fixed_fields['threshold'].tolist() == [25.0, 75.0, 225.0] * 2
    assert fixed_fields['invasion_efficiency'].round(3).tolist() == \
        [0.167, 0.5, 0.833] * 2
    assert fixed['wells']['invasion_efficiency'].tolist() == [0.5, 0.5]
    assert fixed['wells']['n_fields'].tolist() == [3, 3]
    assert fixed['parasites']['invasion_class'].value_counts().to_dict() == {
        'attached': 36, 'invaded': 36, 'unclassified': 0}


def test_invasion_change_plate_answers_what_the_database_key_answers(tmp_path):
    """Same invariant as the replication assay: the plate label is not the biology."""
    root = tmp_path / 'run'
    _invasion_db(root)

    kept = analyze_invasion(_invasion_settings(root, change_plate=False))
    rebuilt = analyze_invasion(_invasion_settings(root, change_plate=True))

    columns = ['prcf', 'n_objects', 'threshold', 'n_attached', 'n_invaded']
    pd.testing.assert_frame_equal(kept['fields'][columns],
                                  rebuilt['fields'][columns])
    assert kept['wells']['invasion_efficiency'].tolist() == [0.5, 0.5]


def test_invasion_without_a_timelapse_is_byte_for_byte_unchanged(tmp_path):
    """A non-timelapse invasion run must survive the change untouched."""
    root = tmp_path / 'run'
    _invasion_db(root, times=(None,))

    fixed = analyze_invasion(_invasion_settings(root, change_plate=True))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(submodules, '_ensure_field_key', _legacy_field_key)
        legacy = analyze_invasion(_invasion_settings(root, change_plate=True))

    pd.testing.assert_frame_equal(legacy['fields'], fixed['fields'])
    pd.testing.assert_frame_equal(legacy['wells'], fixed['wells'])
    assert set(fixed['fields']['prcf']) == {f'{PLATE}_r1_c1_f1',
                                            f'{PLATE}_r1_c2_f1'}


# ---------------------------------------------------------------------------
# the migration: a time-blind prcf already sitting in a database
# ---------------------------------------------------------------------------

def _blind_the_prcf(db, table='pathogen'):
    """Rewrite ``prcf`` to the four-token form a pre-fix writer would leave.

    The database is still built by the real writer; only this one column is
    walked back, which is the situation the repair-on-read exists for. The
    UPDATE runs inside an explicit transaction — sqlite3 opens an implicit one
    for DML, but being explicit is the same discipline commit 0b418d2 imposed
    on ``rename_columns_in_db`` after every ALTER TABLE turned out to be
    autocommitting.
    """
    conn = sqlite3.connect(db)
    try:
        conn.execute('BEGIN')
        conn.execute(
            f"UPDATE {table} SET prcf = plateID || '_' || rowID || '_' "
            f"|| columnID || '_' || fieldID")
        conn.execute('COMMIT')
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    finally:
        conn.close()


def test_a_time_blind_prcf_in_the_database_is_repaired_on_read(tmp_path, capsys):
    """Old data is corrected on read, not preserved.

    ``change_plate=False`` trusts the database's ``prcf``. A database whose
    ``prcf`` was written without the timepoint — but which still carries the
    ``timeID`` column, so the frames are all there — would silently give the
    same 4-vacuoles-of-3 answer. The assay now notices that the stored key is
    exactly this frame's own time-blind key, appends the timepoint, and says so.
    """
    root = tmp_path / 'run'
    db = _replication_db(root)
    _blind_the_prcf(db)
    assert set(_prcf_column(db)) == {f'{PLATE}_r1_c1_f1', f'{PLATE}_r1_c2_f1'}

    output = analyze_replication(_replication_settings(root, change_plate=False))

    assert 'Repaired 12 time-blind prcf value(s)' in capsys.readouterr().out
    assert len(output['vacuoles']) == 12
    assert output['vacuoles']['n_parasites'].tolist() == [1] * 12
    assert sorted(set(output['vacuoles']['prcf'])) == [
        f'{PLATE}_r1_c1_f1_t1', f'{PLATE}_r1_c1_f1_t2', f'{PLATE}_r1_c1_f1_t3',
        f'{PLATE}_r1_c2_f1_t1', f'{PLATE}_r1_c2_f1_t2', f'{PLATE}_r1_c2_f1_t3']


def test_a_prcf_that_is_not_this_frames_key_is_left_alone(tmp_path, capsys):
    """The repair only touches a key it can prove is the time-blind build.

    A ``prcf`` that differs any other way — an imported table, a key from
    :mod:`spacr.foreign`, a plate renamed upstream — is somebody else's
    identifier and is passed through untouched. Rewriting it would be the same
    class of mistake as dropping the timepoint.
    """
    frame = pd.DataFrame({
        'plateID': ['plate1'] * 2,
        'rowID': ['r1'] * 2,
        'columnID': ['c1'] * 2,
        'fieldID': ['f1'] * 2,
        'timeID': ['t1', 't2'],
        'prcf': ['imported_key_a', 'imported_key_b'],
    })
    result = submodules._ensure_field_key(frame.copy(), source='an import')
    assert result['prcf'].tolist() == ['imported_key_a', 'imported_key_b']
    assert 'Repaired' not in capsys.readouterr().out


def test_the_legacy_time_id_spelling_is_accepted(tmp_path):
    """``time_id`` and ``timeID`` both resolve, through ``utils._time_column``."""
    frame = pd.DataFrame({
        'plateID': ['plate1'] * 2,
        'rowID': ['r1'] * 2,
        'columnID': ['c1'] * 2,
        'fieldID': ['f1'] * 2,
        'time_id': ['t1', 't2'],
    })
    built = submodules._ensure_field_key(frame.copy())
    assert built['prcf'].tolist() == ['plate1_r1_c1_f1_t1', 'plate1_r1_c1_f1_t2']


def test_no_timepoint_column_builds_the_four_token_key(tmp_path):
    """No timepoint anywhere means the key is what it always was."""
    frame = pd.DataFrame({
        'plateID': ['plate1'] * 2,
        'rowID': ['r1'] * 2,
        'columnID': ['c1', 'c2'],
        'fieldID': ['f1'] * 2,
    })
    built = submodules._ensure_field_key(frame.copy())
    assert built['prcf'].tolist() == ['plate1_r1_c1_f1', 'plate1_r1_c2_f1']


# ---------------------------------------------------------------------------
# the field-threshold join
# ---------------------------------------------------------------------------

def test_a_duplicated_field_threshold_row_is_caught_not_absorbed(tmp_path):
    """``_invasion_classify`` refuses to let its join grow the parasite table.

    ``_invasion_field_thresholds`` produces one row per ``prcf`` so the join is
    many-to-one, but a field table assembled any other way with a repeated
    ``prcf`` would duplicate every parasite and inflate ``n_total`` silently.
    :func:`spacr.io._report_fan_out` is the established check for exactly that.
    """
    from spacr.io import JoinFanOut

    parasites = pd.DataFrame({
        'prcf': ['plate1_r1_c1_f1_t1'] * 3,
        'outside_intensity': [10.0, 20.0, 30.0],
        'no_host_cell': [False] * 3,
    })
    fields = pd.DataFrame({
        'prcf': ['plate1_r1_c1_f1_t1', 'plate1_r1_c1_f1_t1'],   # written twice
        'threshold': [15.0, 15.0], 'threshold_source': ['field'] * 2,
        'threshold_low': [12.0] * 2, 'threshold_high': [18.0] * 2,
        'automatic_threshold': [15.0] * 2, 'reference_threshold': [15.0] * 2,
        'bimodality_coefficient': [0.7] * 2,
    })
    with pytest.raises(JoinFanOut, match='3 parasite rows into 6'):
        submodules._invasion_classify(parasites, fields, 'outside_intensity',
                                      'attached')
