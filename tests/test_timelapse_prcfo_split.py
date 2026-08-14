"""``spacr.ml`` must be able to split a timelapse ``prcfo``.

``prcfo`` is ``plate_row_column_field_object`` on a plain screen and
``plate_row_column_field_TIME_object`` on a timelapse — that is what
:func:`spacr.utils._map_wells_png` writes onto ``png_list`` and what
:func:`spacr.utils._split_data` rebuilds as the index of
``io._read_and_merge_data``.

Three places in :mod:`spacr.ml` split it into exactly **five** names. The
question this file settles is whether that mis-assigns or raises: on pandas'
``__setitem__`` a six-column split against a five-name key raises
``ValueError: Columns must be same length as key``, so it **raises**, and it
raises at the very last statement of :func:`spacr.ml.ml_analysis` — after the
model has been fitted, thresholded, permutation-scored and similarity-scored.
The whole run is thrown away. (Had it not raised, the five names would have
been wrong too: the fifth token of a timelapse key is the timepoint, so
``objectID`` would have held ``'t1'`` and the object id would have been
dropped.)

Every ``prcfo`` asserted against here comes out of a real ``measurements.db``
written by ``utils.filepaths_to_database`` and
``utils._merge_and_save_to_database`` and read back through
``io._read_and_merge_data`` — never a literal, because a literal that happens
to match is how this survived.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io  # noqa: E402,F401
from spacr.io import TimelapseKeyMismatch  # noqa: E402
from spacr.ml import (_assign_prcfo_parts, ml_analysis, process_reads,  # noqa: E402
                      process_scores)
from spacr.utils import (_merge_and_save_to_database,  # noqa: E402
                         filepaths_to_database)

PLATE = "plate1"
WELLS = ("A01", "A02")          # -> r1_c1 (negative control), r1_c2 (positive)
FIELDS = (1, 2)
OBJECTS = (1, 2, 3)


@pytest.fixture(autouse=True)
def _no_blocking_show_and_clean_figs(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the code as it was
# ---------------------------------------------------------------------------

def _legacy_five_name_split(df, object_column='objectID'):
    """The pre-fix line from all three call sites, verbatim.

    ``df[[...five names...]] = df['prcfo'].str.split('_', expand=True)``.
    """
    df[['plateID', 'rowID', 'columnID', 'fieldID', object_column]] = \
        df['prcfo'].str.split('_', expand=True)
    return df


# ---------------------------------------------------------------------------
# databases — real writers only
# ---------------------------------------------------------------------------

def _crop_name(well, field, time, obj):
    if time is None:
        return f'{PLATE}_{well}_{field}_{obj}.png'
    return f'{PLATE}_{well}_{field}_{time}_{obj}.png'


def _stem(well, field, time):
    if time is None:
        return f'{PLATE}_{well}_{field}'
    return f'{PLATE}_{well}_{field}_{time}'


def _build(root, times=(1, 2, 3)):
    """Write a measurements.db with the real writers and return its path.

    ``times=(None,)`` means "not a timelapse": no timepoint in the crop names,
    none in the stack names, and therefore none in the schema or in ``prcfo``.

    The intensity values separate the two wells cleanly so
    :func:`ml_analysis` has something to learn, and three channel-3 features
    are written because ``filter_dataframe_features(channel_of_interest=3)``
    drops everything else and ``_calculate_similarity`` needs a covariance
    matrix with more than one column.
    """
    root = str(root)
    os.makedirs(os.path.join(root, 'measurements'), exist_ok=True)

    paths = [os.path.join(root, 'cell_png', _crop_name(w, f, t, o))
             for w in WELLS for f in FIELDS for t in times for o in OBJECTS]
    filepaths_to_database(paths, {'timelapse': times != (None,)}, root, 'cell')

    rng = np.random.default_rng(0)
    for index, well in enumerate(WELLS):
        base = 100.0 * (index + 1)
        for field in FIELDS:
            for time in times:
                _merge_and_save_to_database(
                    pd.DataFrame({
                        'label': list(OBJECTS),
                        'cell_area': [base + o for o in OBJECTS],
                    }),
                    pd.DataFrame({
                        'label': list(OBJECTS),
                        'cell_channel_3_mean_intensity':
                            [base + o + rng.normal(0, 1) for o in OBJECTS],
                        'cell_channel_3_max_intensity':
                            [2 * base + o + rng.normal(0, 1) for o in OBJECTS],
                        'cell_channel_3_median_intensity':
                            [1.5 * base + o + rng.normal(0, 1) for o in OBJECTS],
                    }),
                    'cell', root, _stem(well, field, time), 'exp',
                    timelapse=times != (None,))
    return os.path.join(root, 'measurements', 'measurements.db')


def _merged(db):
    """The per-object frame every ml.py entry point starts from, prcfo-indexed."""
    frame, _ = spacr.io._read_and_merge_data([db], ['cell'], verbose=False,
                                             nuclei_limit=True,
                                             pathogen_limit=True)
    return frame


def _fit(frame):
    return ml_analysis(frame, channel_of_interest=3, location_column='columnID',
                       positive_control='c2', negative_control='c1',
                       n_repeats=2, top_features=5, n_estimators=10,
                       model_type='logistic_regression', n_jobs=1,
                       remove_low_variance_features=False,
                       remove_highly_correlated_features=False,
                       split_by='cell',
                       verbose=False)


# ---------------------------------------------------------------------------
# the premise
# ---------------------------------------------------------------------------

def test_the_writers_produce_a_six_token_prcfo_on_a_timelapse(tmp_path):
    """2 wells x 2 fields x 3 frames x 3 objects = 36 rows, keys of 6 tokens."""
    frame = _merged(_build(tmp_path / 'tl'))
    assert len(frame) == 36
    assert frame.index.name == 'prcfo'
    assert frame.index[0] == 'plate1_r1_c1_f1_t1_o1'
    assert set(frame.index.str.count('_')) == {5}          # six tokens

    plain = _merged(_build(tmp_path / 'plain', times=(None,)))
    assert len(plain) == 12
    assert plain.index[0] == 'plate1_r1_c1_f1_o1'
    assert set(plain.index.str.count('_')) == {4}          # five tokens


# ---------------------------------------------------------------------------
# it raises — it does not mis-assign
# ---------------------------------------------------------------------------

def test_the_five_name_split_raises_on_a_real_timelapse_key(tmp_path):
    """Pinned: pandas refuses the assignment, so nothing is silently wrong.

    Six split columns against five names is ``ValueError: Columns must be same
    length as key``. Run against the keys of a real database, not a literal.
    """
    frame = _merged(_build(tmp_path / 'tl')).reset_index()
    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_five_name_split(frame.copy())


def test_ml_analysis_completed_the_model_and_then_threw_it_away(tmp_path):
    """The crash is at the last statement, after everything expensive is done.

    ``_calculate_similarity`` — the step immediately above the split — is what
    proves the run had gone all the way through: its columns are on the frame
    that the split then refuses. So the cost of this bug is a whole fit, not an
    early input-validation error.
    """
    frame = _merged(_build(tmp_path / 'tl'))
    output, _ = _fit(frame)
    scored = output[0]

    assert 'similarity_to_pos_euclidean' in scored.columns
    assert 'prediction_probability_class_1' in scored.columns
    assert len(scored) == 36

    # And that same completed frame is what the old line could not split.
    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_five_name_split(scored.copy(), object_column='object')


def test_ml_analysis_keeps_the_object_id_and_the_timepoint(tmp_path):
    """Every component lands where it belongs, and the timepoint is kept.

    The object id is the LAST token, not the fifth: reading the fifth would put
    ``'t1'`` in ``object`` and lose ``o1`` entirely.
    """
    frame = _merged(_build(tmp_path / 'tl'))
    scored = _fit(frame)[0][0]

    assert scored['plateID'].unique().tolist() == ['plate1']
    assert scored['rowID'].unique().tolist() == ['r1']
    assert sorted(scored['columnID'].unique()) == ['c1', 'c2']
    assert sorted(scored['fieldID'].unique()) == ['f1', 'f2']
    assert sorted(scored['object'].unique()) == ['o1', 'o2', 'o3']
    assert sorted(scored['timeID'].unique()) == ['t1', 't2', 't3']
    assert sorted(scored['prc'].unique()) == ['plate1_r1_c1', 'plate1_r1_c2']

    # The components must reassemble into the key they came from.
    rebuilt = (scored['plateID'] + '_' + scored['rowID'] + '_'
               + scored['columnID'] + '_' + scored['fieldID'] + '_'
               + scored['timeID'] + '_' + scored['object'])
    assert rebuilt.tolist() == scored['prcfo'].tolist()


def test_ml_analysis_on_a_plain_screen_is_unchanged(tmp_path):
    """The five-token path must come out exactly as the old line left it.

    Compared against the old line run on the same frame, not against a
    hand-written expectation of what the old line did. No timepoint column is
    invented for a database that has no timepoint.
    """
    frame = _merged(_build(tmp_path / 'plain', times=(None,)))
    scored = _fit(frame)[0][0]

    legacy = _legacy_five_name_split(scored[['prcfo']].copy(),
                                     object_column='object')
    for column in ('plateID', 'rowID', 'columnID', 'fieldID', 'object'):
        assert scored[column].tolist() == legacy[column].tolist()
    assert 'timeID' not in scored.columns
    assert 'time_id' not in scored.columns
    assert sorted(scored['object'].unique()) == ['o1', 'o2', 'o3']


# ---------------------------------------------------------------------------
# the other two call sites
# ---------------------------------------------------------------------------

def test_process_scores_on_a_timelapse_scores_frame(tmp_path):
    """A scores frame carrying only ``prcfo`` used to die in ``process_scores``.

    This is the shape ``perform_regression`` hands it: the per-object score
    table, keyed by ``prcfo``, with the well identifiers to be recovered from
    that key.
    """
    frame = _merged(_build(tmp_path / 'tl')).reset_index()
    scores = pd.DataFrame({'prcfo': frame['prcfo'],
                           'pred': np.linspace(0.0, 1.0, len(frame))})

    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_five_name_split(scores.copy())

    dependent_df, name = process_scores(scores.copy(), 'pred', plate=None,
                                        min_cell_count=1, agg_type='mean')
    assert name == 'pred'
    assert sorted(dependent_df['prc']) == ['plate1_r1_c1', 'plate1_r1_c2']
    assert dependent_df['cell_count'].tolist() == [18, 18]


def test_process_reads_reads_the_object_id_not_the_timepoint(tmp_path):
    """``process_reads`` recomputes the components, so a bad ``objectID`` is repaired.

    A counts CSV whose ``objectID`` was filled in by a positional guess over a
    timelapse crop name holds the *timepoint* — the same guess
    ``ml.interperate_vision_model`` already refuses to trust. Because the
    components are rebuilt from ``prcfo``, reading the CSV corrects it.
    """
    frame = _merged(_build(tmp_path / 'tl')).reset_index()
    counts = pd.DataFrame({
        'prcfo': frame['prcfo'],
        'objectID': frame['prcfo'].str.split('_').str[4],   # the wrong guess
        'grna': ['TGGT1_000001_1'] * len(frame),
        'count': [10] * len(frame),
    })
    assert sorted(counts['objectID'].unique()) == ['t1', 't2', 't3']

    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_five_name_split(counts.copy())

    independent_df = process_reads(counts.copy(), fraction_threshold=None,
                                   plate=None)
    assert sorted(independent_df['prc'].unique()) == ['plate1_r1_c1',
                                                      'plate1_r1_c2']
    # and the repaired components, read off the frame the helper wrote to
    repaired = _assign_prcfo_parts(counts.copy(), object_column='objectID')
    assert sorted(repaired['objectID'].unique()) == ['o1', 'o2', 'o3']
    assert sorted(repaired['timeID'].unique()) == ['t1', 't2', 't3']


# ---------------------------------------------------------------------------
# the helper's own contract
# ---------------------------------------------------------------------------

def test_a_frame_mixing_both_key_shapes_is_reported(tmp_path):
    """Five- and six-token keys in one frame is a TimelapseKeyMismatch.

    Both halves are real: two databases, one written as a timelapse and one
    not, concatenated the way ``generate_ml_scores`` concatenates its ``src``
    list. The fifth token then means the object id in some rows and the
    timepoint in others, and there is no answer that is right for both.
    """
    timelapse = _merged(_build(tmp_path / 'tl')).reset_index()
    plain = _merged(_build(tmp_path / 'plain', times=(None,))).reset_index()
    combined = pd.concat([timelapse[['prcfo']], plain[['prcfo']]],
                         ignore_index=True)

    with pytest.raises(TimelapseKeyMismatch,
                       match=r'12 key\(s\) without a timepoint and 36 with one'):
        _assign_prcfo_parts(combined)


def test_a_key_that_is_not_a_prcfo_at_all_is_named(tmp_path):
    """A ``prc`` handed in as a ``prcfo`` says so instead of splitting silently."""
    frame = _merged(_build(tmp_path / 'tl')).reset_index()
    wells = pd.DataFrame({'prcfo': frame['prc']})
    with pytest.raises(ValueError, match=r"found \[3\] token\(s\), e\.g\. 'plate1_r1_c1'"):
        _assign_prcfo_parts(wells)


def test_the_legacy_time_id_spelling_is_reused_not_duplicated(tmp_path):
    """A frame already carrying ``time_id`` keeps that spelling."""
    frame = _merged(_build(tmp_path / 'tl')).reset_index()
    frame = frame.rename(columns={'timeID': 'time_id'})
    frame['time_id'] = None

    result = _assign_prcfo_parts(frame, object_column='object')
    assert 'timeID' not in result.columns
    assert sorted(result['time_id'].unique()) == ['t1', 't2', 't3']
