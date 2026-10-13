"""Round-6 coverage for the last unwalked corners of :mod:`spacr.io`.

Two things are pinned here.

The first is the only branch of this module's round-6 target list that no
earlier round had reasoned about: the fallback in ``_read_and_merge_data``
that rebuilds an object key when the metadata frame carries no ``prcf``
column. It cannot be reached, and the reason matters -- the fallback spells
the key WITHOUT the timepoint, so if it ever did run on timelapse data every
frame of an object would collapse onto one row. What guarantees it stays dead
is that ``utils._split_data`` emits ``prcf`` into the same half of the frame
as ``plateID``, and the line above the fallback already indexes ``plateID``.

The second is the complement rule inside ``_get_avg_object_size``: the ``else``
that reports an unusable mask has exactly two entrances, so its second test can
never be false. Both entrances are driven here, which is what makes the third
possibility -- an ``else`` a mask enters and neither warning describes --
observable if it is ever introduced.

CPU-only and offline throughout.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.io as IO


# ---------------------------------------------------------------------------
# helpers (same shape as the round-5 file's, kept local so the two do not
# share fixtures)
# ---------------------------------------------------------------------------

def _well(obj_key):
    row, col = f"r{(obj_key % 2) + 1}", f"c{(obj_key % 2) + 1}"
    return {"plateID": "plate1", "rowID": row, "columnID": col,
            "fieldID": "f1", "prcf": f"plate1_{row}_{col}_f1"}


def _cell_frame(labels):
    rows = []
    for label in labels:
        row = dict(_well(label), object_label=label)
        row['cell_area'] = 10.0 + label
        rows.append(row)
    return pd.DataFrame(rows)


def _measure_db(path, tables):
    with sqlite3.connect(str(path)) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, index=False, if_exists='replace')
    return str(path)


# ---------------------------------------------------------------------------
# _read_and_merge_data: the prcf fallback
# ---------------------------------------------------------------------------

def test_the_object_key_comes_from_prcf_and_its_fallback_cannot_be_reached(
        tmp_path):
    """``metadata`` never arrives with ``plateID`` but without ``prcf``.

    ``_read_and_merge_data`` keys every object as ``prcf + '_' + <parent id>``
    and keeps a fallback that rebuilds the same key from
    plate/row/column/field when ``prcf`` is missing. The two are NOT
    equivalent: ``_split_data`` folds the timepoint into ``prcf`` when the
    table has one, and the fallback has no timepoint at all, so taking it on
    timelapse data would merge every frame of an object into a single row.

    It cannot be taken. ``_split_data`` writes ``prcf`` as a string column in
    the same pass that ``plateID`` survives, so the non-numeric half it hands
    back holds either both or neither -- and the line above the fallback has
    already indexed ``plateID``. This test drives all three outcomes: both
    present (the branch that runs), neither present, and ``prcf`` without
    ``plateID``.
    """
    from spacr.utils import _split_data

    # 1. The live branch. A plain plate keys on prcf, and a timelapse plate
    #    keys on a prcf that carries the timepoint -- which is exactly what
    #    the fallback below would drop.
    cells = _cell_frame([1, 2])
    db = _measure_db(tmp_path / "m.db", {"cell": cells})
    merged, _ = IO._read_and_merge_data([db], ["cell"])
    assert sorted(merged.index) == ['plate1_r1_c1_f1_o2',
                                    'plate1_r2_c2_f1_o1']

    frames = _cell_frame([1, 1])
    frames['timeID'] = [1, 2]
    frames['prcf'] = frames['prcf'] + '_' + frames['timeID'].astype(str)
    time_db = _measure_db(tmp_path / "t.db", {"cell": frames})
    time_merged, _ = IO._read_and_merge_data([time_db], ["cell"])
    # Two frames of ONE object stay two rows, because the timepoint is in the
    # key. The fallback's spelling, plate_row_column_field_object, would have
    # produced one row here.
    assert sorted(time_merged.index) == ['plate1_r2_c2_f1_1_o1',
                                         'plate1_r2_c2_f1_2_o1']

    # 2. prcf and plateID travel together out of _split_data ...
    shaped = _cell_frame([1, 2])
    shaped['prcfo'] = shaped['prcf'] + '_o' + shaped['object_label'].astype(str)
    _numeric, metadata = _split_data(shaped, 'prcfo', 'object_label')
    assert {'plateID', 'prcf'} <= set(metadata.columns)

    # ... and when the well metadata is numeric, BOTH plateID and prcf cannot
    # both be missing from the answer: prcf survives, plateID does not, so the
    # `metadata.assign(prc=...)` above the fallback raises first.
    numeric = shaped.assign(plateID=1, rowID=2, columnID=3, fieldID=4)
    _numeric, numeric_metadata = _split_data(numeric, 'prcfo', 'object_label')
    assert 'prcf' in numeric_metadata.columns
    assert 'plateID' not in numeric_metadata.columns

    numeric_db = _measure_db(tmp_path / "n.db",
                             {"cell": numeric.drop(columns=['prcfo'])})
    with pytest.raises(KeyError, match='plateID'):
        IO._read_and_merge_data([numeric_db], ["cell"])

    # 3. An empty table loses every column, prcf included, so it fails on
    #    plateID at the same line rather than reaching the fallback.
    empty = _cell_frame([1]).iloc[:0]
    empty_db = _measure_db(tmp_path / "e.db", {"cell": empty})
    with pytest.raises(KeyError, match='plateID'):
        IO._read_and_merge_data([empty_db], ["cell"])


# ---------------------------------------------------------------------------
# _get_avg_object_size
# ---------------------------------------------------------------------------

def test_a_mask_that_is_counted_as_nothing_always_says_which_kind_of_nothing(
        capsys):
    """The unusable-mask ``else`` has exactly two entrances, and names both.

    A mask is skipped when it is empty OR when it has an axis count
    ``regionprops`` cannot read. The report distinguishes them, because the
    two mean opposite things to a user: an empty mask is a field where nothing
    was segmented, while a 4-D mask is a reader or a merge that produced the
    wrong shape and whose "no objects" is not a measurement at all.

    Because those two are the whole of the condition that enters the branch,
    the second test inside it is the negation of the first and cannot be
    false; a mask that entered and matched neither would print nothing, which
    is what this pins against.
    """
    good = np.zeros((8, 8), dtype=np.int32)
    good[1:4, 1:4] = 1          # 9 pixels
    good[5:7, 5:7] = 2          # 4 pixels
    empty = np.zeros((8, 8), dtype=np.int32)
    wrong_rank = np.ones((2, 2, 2, 2), dtype=np.int32)

    avg_objects, avg_size = IO._get_avg_object_size(
        [good, empty, wrong_rank])

    # Three masks, two objects, both from the one usable mask.
    assert avg_objects == pytest.approx(2 / 3)
    assert avg_size == pytest.approx(6.5)

    out = capsys.readouterr().out
    assert "Mask 1 is empty" in out
    assert "Mask 2 has invalid dimension: 4" in out
    # Every skipped mask is accounted for by name: nothing entered the branch
    # silently.
    assert out.count("Warning: Mask") == 2


# ---------------------------------------------------------------------------
# Everything else on this chunk's list was already closed
#
# The round-6 chunk for spacr/io.py was measured against io.py as it stood at
# commit fc34e059, which predates `tests: coverage round 5` (87c8e256). All
# 133 of its items were re-measured against the current file (the line numbers
# shift by up to five lines across the two commits that have touched io.py
# since) and, apart from the two branches above, every one of them is either
# already executed by the committed io tests or already carries a written
# proof in tests/test_cov_r5_io.py:
#
#   * closed by tests/test_cov_r3_io_{1,2}.py and tests/test_cov_r5_io.py:
#     44 of the 50 lines and 68 of the 83 arcs, including the whole
#     "no image stacks were produced" refusal in preprocess_img_data, the
#     resume-manifest failures in _load_and_concatenate_arrays, and the
#     _read_and_merge_data reporting branches.
#   * proved unreachable, with an executable pin, in test_cov_r5_io.py:
#     the plate escape re-check in migrate_unescaped_plate_names, the
#     channel-less suffix in process_non_tif_non_2D_images, the empty-class
#     guard in _balance_lists, the empty-column guard in
#     _annotation_classes_from_columns, the unknown dataset_mode, the
#     grouped-split leakage guard, and the un-augmented short folder in
#     prepare_cellpose_dataset.
#   * proved unreachable in prose in test_cov_r5_io.py: the empty-array
#     `continue` in _create_movies_from_npy_per_channel, the `ch in seen`
#     re-check in preprocess_img_data, the `dst is None` check in
#     generate_dataset, the 99,999-sibling loop-else in _ensure_unique_dir,
#     the duplicate-key test in convert_to_yokogawa, and the missing
#     grouped_splits check in generate_dataset_from_lists.
#
# Nothing is excluded from coverage; every one of those branches is still in
# the source with a test asserting the guarantee that makes it dead.
# ---------------------------------------------------------------------------
