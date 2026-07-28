"""metadata-mode class selection reads the column the user actually named.

``generate_training_dataset(dataset_mode='metadata')`` used to select classes
from a hard-coded ``'condition'`` column of ``png_list``. No spaCR writer
creates that column -- :func:`spacr.utils.filepaths_to_database` writes
``plateID``/``rowID``/``columnID``/``fieldID``/``prcfo`` and nothing else, and
``condition`` appears only after ``annotate_conditions`` has been run over a
plate-metadata map. So the shipped default configuration (``metadata_type_by
= 'columnID'``, ``class_metadata = [['c1'], ['c2']]``) printed

    metadata mode (legacy): 'condition' column not found in png_list;
    got 0 classes.

...and then indexed ``png_df['condition']`` anyway, one line below the guard,
turning a diagnosable misconfiguration into ``KeyError: 'condition'``.

Two behaviours are pinned here:

* the selection column comes from ``settings['metadata_type_by']``, and
* a genuinely absent column raises with the column name, the columns that are
  present and the setting to change -- never a bare ``KeyError``.

The database is built by the real writers only:
:func:`spacr.utils.filepaths_to_database` for ``png_list`` and
:func:`spacr.utils._merge_and_save_to_database` for the object tables, so the
schema is the one ``measure_crop`` leaves behind rather than a hand-built
table that happens to carry the column under test.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from PIL import Image

WELLS = ('A1', 'A2', 'A3')
FIELDS = (1, 2)
OBJECTS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def _real_plate(root):
    """A plate whose measurements.db only spaCR's own writers have touched."""
    from spacr.utils import filepaths_to_database, _merge_and_save_to_database

    root = str(root)
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)
    crops = os.path.join(root, 'data', 'cell_png')
    os.makedirs(crops, exist_ok=True)

    rng = np.random.default_rng(20240727)
    paths = []
    for well in WELLS:
        for field in FIELDS:
            for obj in OBJECTS:
                path = os.path.join(crops, f'plate1_{well}_{field}_{obj}.png')
                Image.fromarray(
                    rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
                paths.append(path)
    filepaths_to_database(paths, {'timelapse': False}, root, 'cell')

    for well in WELLS:
        for field in FIELDS:
            stem = f'plate1_{well}_{field}'
            morph = pd.DataFrame({'label': list(OBJECTS),
                                  'cell_area': rng.uniform(200, 4000, len(OBJECTS))})
            inten = pd.DataFrame({
                'label': list(OBJECTS),
                'cell_channel_1_mean_intensity': rng.uniform(1, 9, len(OBJECTS))})
            _merge_and_save_to_database(morph, inten, 'cell', root, stem, 'spacr_run')

    return os.path.join(root, 'measurements', 'measurements.db')


def _settings(src, **over):
    settings = {
        'src': str(src),
        'dataset_mode': 'metadata',
        'metadata_type_by': 'columnID',
        'class_metadata': [['c1'], ['c2']],
        'png_type': 'cell_png',
        'crop_source': 'png',
        'tables': ['cell'],
        'size': 16,
        'test_split': 0.2,
    }
    settings.update(over)
    return settings


@pytest.fixture
def plate(tmp_path):
    src = tmp_path / 'plate1'
    src.mkdir()
    _real_plate(src)
    return src


# ---------------------------------------------------------------------------
# what the real writers put in png_list
# ---------------------------------------------------------------------------

def test_the_writers_never_create_a_condition_column(plate):
    """The premise: the column the old code selected on does not exist."""
    con = sqlite3.connect(os.path.join(plate, 'measurements', 'measurements.db'))
    try:
        columns = {row[1] for row in con.execute('PRAGMA table_info("png_list")')}
    finally:
        con.close()
    assert 'condition' not in columns
    assert {'columnID', 'rowID', 'png_path'} <= columns


# ---------------------------------------------------------------------------
# the user's configuration now builds the dataset it describes
# ---------------------------------------------------------------------------

def test_metadata_mode_selects_on_metadata_type_by(plate):
    """columnID + [['c1'], ['c2']] -> a c1 class and a c2 class."""
    from spacr.io import generate_training_dataset

    train, test = generate_training_dataset(_settings(plate))

    assert sorted(os.listdir(train)) == ['c1', 'c2']
    assert sorted(os.listdir(test)) == ['c1', 'c2']
    per_class = len(FIELDS) * len(OBJECTS)
    for cls in ('c1', 'c2'):
        kept = (len(os.listdir(os.path.join(train, cls)))
                + len(os.listdir(os.path.join(test, cls))))
        assert kept == per_class


def test_metadata_mode_selects_on_row_id_when_asked(plate):
    """The setting is honoured, not just tolerated: rowID groups everything
    into one class because every well in the fixture is row A."""
    from spacr.io import generate_training_dataset

    train, _ = generate_training_dataset(
        _settings(plate, metadata_type_by='rowID', class_metadata=[['r1']]))
    assert sorted(os.listdir(train)) == ['r1']


def test_inner_lists_group_several_wells_into_one_class(plate):
    """``[['c1','c2'], ['c3']]`` is two classes, not two literal strings.

    The whole entry used to be ``str()``-ed, so ``['c1','c2']`` was matched as
    the text ``"['c1', 'c2']"`` and selected no rows at all.
    """
    from spacr.io import generate_training_dataset

    train, test = generate_training_dataset(
        _settings(plate, class_metadata=[['c1', 'c2'], ['c3']]))

    assert sorted(os.listdir(train)) == ['c1_c2', 'c3']
    both = (len(os.listdir(os.path.join(train, 'c1_c2')))
            + len(os.listdir(os.path.join(test, 'c1_c2'))))
    one = (len(os.listdir(os.path.join(train, 'c3')))
           + len(os.listdir(os.path.join(test, 'c3'))))
    # balance_to_smallest trims the two-well class down to the one-well class
    assert both == one == len(FIELDS) * len(OBJECTS)


def test_plain_string_entries_still_work(plate):
    """The flat form ``['c1','c2']`` predates the nested one and still means
    one class per value."""
    from spacr.io import generate_training_dataset

    train, _ = generate_training_dataset(
        _settings(plate, class_metadata=['c1', 'c2']))
    assert sorted(os.listdir(train)) == ['c1', 'c2']


# ---------------------------------------------------------------------------
# a missing column is reported, not walked into
# ---------------------------------------------------------------------------

def test_missing_column_raises_and_names_what_is_available(plate):
    from spacr.io import generate_training_dataset

    with pytest.raises(ValueError) as excinfo:
        generate_training_dataset(_settings(plate, metadata_type_by='condition'))

    message = str(excinfo.value)
    assert "'condition'" in message          # the column that is missing
    assert 'columnID' in message             # a column that is present
    assert 'metadata_type_by' in message     # the setting to change


def test_missing_column_does_not_raise_key_error(plate):
    """Regression: the guard printed and then fell through to png_df['condition'].

    ``KeyError`` is a subclass of ``LookupError``, not of ``ValueError``, so
    this asserts the failure mode changed rather than merely moved.
    """
    from spacr.io import generate_training_dataset

    with pytest.raises(ValueError):
        generate_training_dataset(_settings(plate, metadata_type_by='no_such_column'))
    with pytest.raises(Exception) as excinfo:
        generate_training_dataset(_settings(plate, metadata_type_by='no_such_column'))
    assert not isinstance(excinfo.value, KeyError)


def test_blank_metadata_type_by_falls_back_to_condition(plate):
    """An unset setting is the only thing that still means 'condition' -- and
    it says so instead of crashing."""
    from spacr.io import generate_training_dataset

    with pytest.raises(ValueError, match="'condition'"):
        generate_training_dataset(_settings(plate, metadata_type_by=''))


def test_class_metadata_arriving_as_a_string_is_parsed(plate):
    """A settings CSV stores ``"[['c1'], ['c2']]"``.

    Iterating the string gave one class per character -- '[', '[', "'", 'c',
    '1' -- so the first thing the old code did with the user's configuration
    was look up ``png_df['condition']`` for a class named '['.
    """
    from spacr.io import generate_training_dataset

    train, _ = generate_training_dataset(
        _settings(plate, class_metadata="[['c1'], ['c2']]"))
    assert sorted(os.listdir(train)) == ['c1', 'c2']
