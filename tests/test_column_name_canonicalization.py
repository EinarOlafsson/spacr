"""Percentile word order and channel spelling: one name per idea.

Two families of measurement column disagreed with the rest of the database
about how to spell themselves:

* the object *interior* percentiles have always been ``percentile_5``, but the
  periphery and outside rings were written ``periphery_5_percentile`` /
  ``outside_5_percentile`` — the same statistic, the reverse word order;
* ``organelle_summary_organelle_ch0_...`` was the only family that abbreviated
  the channel, while every other one writes ``channel_0``.

Both are corrected going forward in :mod:`spacr.measure`, and
:func:`spacr.utils.rename_columns_in_db` — which already runs at the top of
``io._read_db`` and ``io._read_and_join_tables`` — migrates an old database the
first time it is read.

Everything here is built with the real writers: the databases come out of
``measure_crop``, and the "old" database is a real one whose columns are then
renamed *back* to the legacy spelling, so the schema, the types and the values
are the ones spaCR actually produces.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

PERCENTILES = (5, 10, 25, 50, 75, 85, 95)
OBJECTS_WITH_RINGS = ('nucleus', 'pathogen', 'organelle')


# ---------------------------------------------------------------------------
# a real measurements.db, and a real one aged back to the legacy spelling
# ---------------------------------------------------------------------------

def _merged_stack(rng):
    """(48, 48, 6): 2 intensity channels + cell/nucleus/pathogen/organelle masks."""
    cell = np.zeros((48, 48), np.uint16)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    organelle = np.zeros_like(cell)
    for i, (r, c) in enumerate([(6, 6), (6, 26), (26, 6)], start=1):
        cell[r:r + 14, c:c + 14] = i
        nucleus[r + 3:r + 8, c + 3:c + 8] = i
        pathogen[r + 9:r + 12, c + 9:c + 12] = i
        organelle[r + 1:r + 3, c + 1:c + 3] = i
    channels = []
    for _ in range(2):
        base = rng.integers(50, 200, size=(48, 48)).astype(np.uint16)
        base[cell > 0] += 3000
        channels.append(base)
    return np.stack(channels + [cell, nucleus, pathogen, organelle],
                    axis=-1).astype(np.uint16)


def _run_measure_crop(root):
    """Run the real measure_crop over one synthetic field; return the db path."""
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    merged = os.path.join(root, 'merged')
    os.makedirs(merged, exist_ok=True)
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(os.path.join(merged, 'plate1_A01_F001.npy'), _merged_stack(rng))

    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': merged, 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4,
        'organelle_mask_dim': 5,
        'summarize_organelles_by': ['organelle', 'cell'],
        'png_dims': [0, 1], 'png_size': [32, 32],
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 1, 'test_mode': False, 'cytoplasm': True,
    })
    measure_crop(dict(settings))
    return os.path.join(root, 'measurements', 'measurements.db')


def _columns(db_path):
    """Return ``{table: [column, ...]}`` for every user table."""
    con = sqlite3.connect(db_path)
    try:
        tables = [row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: [row[1] for row in
                        con.execute(f'PRAGMA table_info(`{table}`)')]
                for table in tables}
    finally:
        con.close()


def _age_to_legacy_spelling(db_path):
    """Rename canonical feature columns back to the pre-migration spelling.

    The inverse of the migration, applied to a database the current writer
    produced — so the fixture is a real measurements.db from an older release
    rather than a schema invented by the test.

    :returns: the list of ``(table, canonical, legacy)`` names it aged.
    """
    import re

    ring = re.compile(
        r'^(?P<head>.*?)(?P<ring>periphery|outside)_percentile_(?P<p>\d+)$')
    organelle = re.compile(
        r'^organelle_summary_organelle_channel_(?P<c>\d+)_(?P<rest>.+)$')

    aged = []
    con = sqlite3.connect(db_path)
    con.isolation_level = None
    try:
        for table, cols in _columns(db_path).items():
            for col in cols:
                match = ring.match(col)
                if match:
                    old = (f"{match.group('head')}{match.group('ring')}"
                           f"_{match.group('p')}_percentile")
                else:
                    match = organelle.match(col)
                    if not match:
                        continue
                    old = (f"organelle_summary_organelle_ch{match.group('c')}_"
                           f"{match.group('rest')}")
                con.execute(
                    f'ALTER TABLE `{table}` RENAME COLUMN `{col}` TO `{old}`')
                aged.append((table, col, old))
    finally:
        con.close()
    return aged


@pytest.fixture(scope='module')
def fresh_db(tmp_path_factory):
    """A measurements.db written by the current measure_crop."""
    return _run_measure_crop(str(tmp_path_factory.mktemp('fresh')))


@pytest.fixture(scope='module')
def _legacy_db_template(tmp_path_factory):
    """The same database, aged back to the pre-migration column spelling."""
    path = _run_measure_crop(str(tmp_path_factory.mktemp('legacy')))
    aged = _age_to_legacy_spelling(path)
    assert aged, 'nothing to age — the fixture measured no ring percentiles'
    return path


@pytest.fixture
def legacy_db(_legacy_db_template, tmp_path):
    """A private copy of the aged database.

    Per test, not per module: every test here migrates it, and a migration is
    by design a one-way in-place change — sharing one file would mean the
    second test to run saw a database the first had already repaired.
    """
    import shutil

    path = str(tmp_path / 'legacy_measurements.db')
    shutil.copy(_legacy_db_template, path)
    return path


# ---------------------------------------------------------------------------
# what measure_crop writes now
# ---------------------------------------------------------------------------

def test_ring_percentiles_are_written_percentile_p(fresh_db):
    """periphery/outside percentiles use the interior word order."""
    cols = _columns(fresh_db)
    for obj in OBJECTS_WITH_RINGS:
        for ring in ('periphery', 'outside'):
            for p in PERCENTILES:
                assert f'{obj}_channel_0_{ring}_percentile_{p}' in cols[obj]


def test_no_table_still_carries_the_reversed_word_order(fresh_db):
    """The old spelling is gone from every table, not just the ones checked."""
    stale = [f'{table}.{col}'
             for table, cols in _columns(fresh_db).items() for col in cols
             if col.endswith('_percentile')
             and ('periphery' in col or 'outside' in col)]
    assert stale == []


def test_interior_and_ring_percentiles_now_agree(fresh_db):
    """The point of the rename: one statistic, one word order.

    Both families end in ``percentile_<p>``, so sorting a column list groups
    them and a suffix match finds all of them.
    """
    cols = _columns(fresh_db)['nucleus']
    for p in (5, 25, 95):
        assert f'nucleus_channel_0_percentile_{p}' in cols
        assert f'nucleus_channel_0_periphery_percentile_{p}' in cols
        assert f'nucleus_channel_0_outside_percentile_{p}' in cols


def test_organelle_summary_spells_the_channel_out(fresh_db):
    """The last family that abbreviated the channel now writes channel_<c>."""
    cols = _columns(fresh_db)['cell_organelle_summary']
    for c in (0, 1):
        for stat in ('mean', 'std'):
            assert (f'organelle_summary_organelle_channel_{c}_{stat}'
                    f'_intensity_per_cell') in cols
    assert not [col for col in cols if '_organelle_ch0' in col
                or '_organelle_ch1' in col]


def test_organelle_summary_names_do_not_collide_with_the_organelle_table(fresh_db):
    """``organelle_channel_0_mean_intensity`` already existed on the organelle
    table, so spelling the summary's channel out could have produced two
    columns with one name. It does not: the summary keeps its own
    ``organelle_summary_`` prefix, its own ``_per_<parent>`` suffix, and lives
    in a different table."""
    cols = _columns(fresh_db)
    assert 'organelle_channel_0_mean_intensity' in cols['organelle']
    summary = cols['cell_organelle_summary']
    assert 'organelle_channel_0_mean_intensity' not in summary
    assert len(summary) == len(set(summary))


# ---------------------------------------------------------------------------
# what happens to a database written before the rename
# ---------------------------------------------------------------------------

def test_legacy_database_is_migrated_on_read(legacy_db):
    """The established repair-on-read contract, extended to these two families."""
    from spacr.utils import rename_columns_in_db

    # 3 ring objects x 2 channels x 7 percentiles x 2 rings = 84 ring columns,
    # plus 2 channels x (mean, std) = 4 organelle-summary columns.
    expected = 3 * 2 * 7 * 2 + 2 * 2
    assert expected == 88

    before = _columns(legacy_db)
    stale_before = [col for cols in before.values() for col in cols
                    if (col.endswith('_percentile')
                        and ('periphery' in col or 'outside' in col))
                    or '_organelle_ch0_' in col or '_organelle_ch1_' in col]
    assert len(stale_before) == expected

    renamed = rename_columns_in_db(legacy_db)
    assert len(renamed) == expected

    after = _columns(legacy_db)
    stale_after = [col for cols in after.values() for col in cols
                   if (col.endswith('_percentile')
                       and ('periphery' in col or 'outside' in col))
                   or '_organelle_ch0_' in col or '_organelle_ch1_' in col]
    assert stale_after == []


def test_migration_is_idempotent(legacy_db):
    """Running on every read has to be free after the first one."""
    from spacr.utils import rename_columns_in_db

    rename_columns_in_db(legacy_db)
    assert rename_columns_in_db(legacy_db) == []
    assert rename_columns_in_db(legacy_db) == []


def test_a_migrated_database_matches_a_freshly_written_one_exactly(
        legacy_db, fresh_db):
    """The migration has to land on the writer, not merely near it.

    Column for column, table for table — otherwise an old database and a new
    one are two different schemas that merely look alike, and every consumer
    needs to handle both forever.
    """
    from spacr.utils import rename_columns_in_db

    rename_columns_in_db(legacy_db)
    migrated = {table: set(cols) for table, cols in _columns(legacy_db).items()}
    fresh = {table: set(cols) for table, cols in _columns(fresh_db).items()}

    assert set(migrated) == set(fresh)
    for table in fresh:
        assert migrated[table] == fresh[table], table


def test_migration_moves_no_values(legacy_db):
    """A rename is a rename: same rows, same numbers, new name."""
    from spacr.utils import rename_columns_in_db

    con = sqlite3.connect(legacy_db)
    try:
        before = con.execute(
            'SELECT nucleus_channel_0_periphery_95_percentile FROM nucleus '
            'ORDER BY object_label').fetchall()
    finally:
        con.close()
    assert before and any(row[0] is not None for row in before)

    rename_columns_in_db(legacy_db)

    con = sqlite3.connect(legacy_db)
    try:
        after = con.execute(
            'SELECT nucleus_channel_0_periphery_percentile_95 FROM nucleus '
            'ORDER BY object_label').fetchall()
    finally:
        con.close()
    assert after == before


def test_a_table_carrying_both_spellings_keeps_both(legacy_db, tmp_path):
    """Never destructive, the same rule the metadata renames follow.

    Dropping or overwriting one of two columns to tidy a name would destroy
    data. Both stay and a human decides.
    """
    import shutil

    from spacr.utils import rename_columns_in_db

    both = str(tmp_path / 'both.db')
    shutil.copy(legacy_db, both)
    con = sqlite3.connect(both)
    con.isolation_level = None
    try:
        con.execute('ALTER TABLE nucleus ADD COLUMN '
                    '`nucleus_channel_0_periphery_percentile_95` REAL')
        con.execute('UPDATE nucleus SET '
                    '`nucleus_channel_0_periphery_percentile_95` = -1.0')
    finally:
        con.close()

    rename_columns_in_db(both)

    cols = _columns(both)['nucleus']
    assert 'nucleus_channel_0_periphery_95_percentile' in cols
    assert 'nucleus_channel_0_periphery_percentile_95' in cols
    con = sqlite3.connect(both)
    try:
        planted = con.execute(
            'SELECT DISTINCT `nucleus_channel_0_periphery_percentile_95` '
            'FROM nucleus').fetchall()
    finally:
        con.close()
    assert planted == [(-1.0,)]


def test_read_db_migrates_without_being_asked(legacy_db, tmp_path):
    """The repair happens on the ordinary read path, not only when a caller
    thinks to call the migration."""
    import shutil

    from spacr.io import _read_db

    target = str(tmp_path / 'read.db')
    shutil.copy(legacy_db, target)

    frames = _read_db(target, tables=['nucleus'])

    assert 'nucleus_channel_0_periphery_percentile_95' in frames[0].columns
    assert 'nucleus_channel_0_periphery_95_percentile' not in frames[0].columns
    assert 'nucleus_channel_0_periphery_percentile_95' in _columns(
        target)['nucleus']


# ---------------------------------------------------------------------------
# the name-level helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('legacy,canonical', [
    ('nucleus_channel_0_periphery_5_percentile',
     'nucleus_channel_0_periphery_percentile_5'),
    ('pathogen_channel_12_outside_95_percentile',
     'pathogen_channel_12_outside_percentile_95'),
    ('periphery_50_percentile', 'periphery_percentile_50'),
    ('outside_50_percentile', 'outside_percentile_50'),
    ('organelle_summary_organelle_ch0_mean_intensity_per_cell',
     'organelle_summary_organelle_channel_0_mean_intensity_per_cell'),
    ('organelle_summary_organelle_ch3_std_intensity_per_pathogen',
     'organelle_summary_organelle_channel_3_std_intensity_per_pathogen'),
    # the metadata renames still work through the same entry point
    ('row', 'rowID'),
    ('time_id', 'timeID'),
])
def test_canonical_column_name_rewrites_legacy_spellings(legacy, canonical):
    from spacr.utils import canonical_column_name

    assert canonical_column_name(legacy) == canonical
    assert canonical_column_name(canonical) == canonical


@pytest.mark.parametrize('name', [
    'cell_channel_0_percentile_75',
    'nucleus_channel_0_periphery_mean',
    'nucleus_channel_0_outside_mean',
    'cell_area',
    'homogeneity_distance_8',
    # a user feature that merely contains 'ch' followed by a digit must not be
    # dragged into the organelle rewrite
    'my_custom_ch0_score',
    'organelle_summary_organelle_mean_area',
    'rad_dist_channel_0_bin_2',
    '',
])
def test_canonical_column_name_leaves_everything_else_alone(name):
    from spacr.utils import canonical_column_name

    assert canonical_column_name(name) == name


def test_ring_rewrite_picks_the_last_ring_token():
    """A head that itself contains 'outside' must not capture the rewrite."""
    from spacr.utils import canonical_column_name

    assert canonical_column_name('outside_thing_periphery_25_percentile') == \
        'outside_thing_periphery_percentile_25'


def test_canonicalize_measurement_columns_renames_a_frame():
    from spacr.utils import canonicalize_measurement_columns

    df = pd.DataFrame({
        'pathogen_channel_1_outside_75_percentile': [1.0, 2.0],
        'pathogen_channel_1_percentile_75': [3.0, 4.0],
        'row': ['A', 'B'],
    })
    out = canonicalize_measurement_columns(df)

    assert out is df
    assert list(df.columns) == ['pathogen_channel_1_outside_percentile_75',
                               'pathogen_channel_1_percentile_75', 'rowID']
    assert df['pathogen_channel_1_outside_percentile_75'].tolist() == [1.0, 2.0]


def test_canonicalize_measurement_columns_keeps_both_when_both_are_present():
    """Same never-destructive rule as the database migration: renaming onto an
    existing name would silently create a duplicate column and lose one of
    them to the next ``df[name]``."""
    from spacr.utils import canonicalize_measurement_columns

    df = pd.DataFrame({
        'nucleus_channel_0_outside_75_percentile': [1.0],
        'nucleus_channel_0_outside_percentile_75': [2.0],
    })
    canonicalize_measurement_columns(df)

    assert list(df.columns) == ['nucleus_channel_0_outside_75_percentile',
                                'nucleus_channel_0_outside_percentile_75']


def test_calculate_recruitment_accepts_a_pre_migration_frame():
    """A CSV exported by an older release never passed through the database
    migration, so the ratio must still divide rather than raise KeyError."""
    from spacr.utils import _calculate_recruitment

    df = pd.DataFrame({
        'pathogen_channel_1_mean_intensity': [100.0],
        'cell_channel_1_mean_intensity': [10.0],
        'cytoplasm_channel_1_mean_intensity': [5.0],
        'nucleus_channel_1_mean_intensity': [4.0],
        'pathogen_channel_1_percentile_75': [300.0],
        'pathogen_channel_1_outside_mean': [20.0],
        'pathogen_channel_1_outside_75_percentile': [60.0],   # legacy
        'pathogen_channel_1_periphery_mean': [8.0],
    })
    out = _calculate_recruitment(df, channel=1)

    assert out['pathogen_outside_cell_q75_mean'].tolist() == [6.0]
    # and the frame it hands back speaks the canonical spelling
    assert 'pathogen_channel_1_outside_percentile_75' in out.columns


# ---------------------------------------------------------------------------
# the feature dictionary describes both spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('canonical,legacy', [
    ('nucleus_channel_0_periphery_percentile_25',
     'nucleus_channel_0_periphery_25_percentile'),
    ('nucleus_channel_0_outside_percentile_25',
     'nucleus_channel_0_outside_25_percentile'),
    ('organelle_summary_organelle_channel_1_mean_intensity_per_cell',
     'organelle_summary_organelle_ch1_mean_intensity_per_cell'),
    ('organelle_summary_organelle_channel_1_std_intensity_per_nucleus',
     'organelle_summary_organelle_ch1_std_intensity_per_nucleus'),
])
def test_both_spellings_are_described_and_describe_the_same_thing(canonical,
                                                                 legacy):
    from spacr.feature_dict import parse_column

    new = parse_column(canonical)
    old = parse_column(legacy)

    assert new.family == old.family == 'intensity'
    assert new.family != 'unknown'
    assert new.unit == old.unit
    assert new.channel == old.channel
    assert new.object_type == old.object_type
    assert new.description and old.description


@pytest.mark.parametrize('canonical,legacy', [
    ('nucleus_channel_0_periphery_percentile_25',
     'nucleus_channel_0_periphery_25_percentile'),
    ('nucleus_channel_0_outside_percentile_25',
     'nucleus_channel_0_outside_25_percentile'),
    ('organelle_summary_organelle_channel_1_mean_intensity_per_cell',
     'organelle_summary_organelle_ch1_mean_intensity_per_cell'),
    ('organelle_summary_organelle_channel_1_std_intensity_per_nucleus',
     'organelle_summary_organelle_ch1_std_intensity_per_nucleus'),
])
def test_the_legacy_entry_says_it_is_legacy_and_names_the_new_one(canonical,
                                                                 legacy):
    """A user reading an old database must not be told the current name is
    what their file contains."""
    from spacr.feature_dict import parse_column

    old = parse_column(legacy)
    text = f'{old.description} {old.notes}'
    assert 'LEGACY' in old.description.upper() or 'legacy' in old.notes
    assert canonical.split('_channel_')[-1].split('_', 1)[-1] in text or \
        'rename_columns_in_db' in text
    assert 'rename_columns_in_db' in text

    new = parse_column(canonical)
    assert 'LEGACY' not in new.description.upper()


def test_the_canonical_entry_points_at_the_migration():
    """Someone who greps the dictionary for the old name they remember has to
    be told where it went."""
    from spacr.feature_dict import parse_column

    for name in ('nucleus_channel_0_periphery_percentile_5',
                 'nucleus_channel_0_outside_percentile_5',
                 'organelle_summary_organelle_channel_0_mean_intensity_per_cell'):
        entry = parse_column(name)
        assert 'rename_columns_in_db' in entry.notes


def test_every_column_of_a_real_database_is_described(fresh_db):
    """The dictionary has to agree with the writer, which is the whole reason
    it is checked against a database rather than against a list.

    Measurement tables are identified by carrying ``object_label`` — the column
    ``utils._check_integrity`` puts on every per-object frame — rather than by
    a hard-coded name list, so a bookkeeping table added later (``run_status``,
    ``settings``, ``settings_history``, ...) does not turn this into a failure
    about provenance columns that were never features.
    """
    from spacr.feature_dict import parse_column

    tables = _columns(fresh_db)
    measurement_tables = [table for table, cols in tables.items()
                          if 'object_label' in cols]
    assert len(measurement_tables) >= 5, sorted(tables)

    unknown = []
    for table in measurement_tables:
        for col in tables[table]:
            if parse_column(col).family == 'unknown':
                unknown.append(f'{table}.{col}')
    assert unknown == []
