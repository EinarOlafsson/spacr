"""Keys and joins in :mod:`spacr.timelapse` and :mod:`spacr.submodules`.

Two families of defect, both of which produce a plausible wrong number rather
than an error:

**Positional key splitting.** ``spacr.schema`` exists so that a ``prc`` /
``prcf`` / ``prcfo`` is taken apart by one parser, right to left. Three sites
were still splitting on ``'_'`` by position: the two ``summarize_per_well*``
functions of the calcium analysis, and the ``prc`` split in
``analyze_percent_positive``. A positional split has two failure modes and the
quiet one is the dangerous one — a five-token key that is not
plate/row/column/field/object (a timelapse ``prcf``, say) is *accepted* and
every column shifts, so ``cells_per_well`` counts timepoints.

**Unstated join cardinality.** Every ``.merge()`` in those two modules now
declares ``validate=``. The tests here pin the two decisions that are easy to
get wrong in the *other* direction: a join that is legitimately many-to-one
must not be declared one-to-one, because that turns ordinary data (a screen
run over two plates, a reads CSV that lists a gRNA twice) into a crash.

Everything runs on hand-built frames and tiny CSVs: no GPU, no network,
sub-second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pandas.errors import MergeError  # noqa: E402

from spacr import schema  # noqa: E402
import spacr.submodules as submodules  # noqa: E402
from spacr.timelapse import (  # noqa: E402
    _summarise_child_features_per_parent,
    summarize_per_well,
    summarize_per_well_inf_non_inf,
)


@pytest.fixture(autouse=True)
def _close_figs():
    """No figure may survive a test (Agg still accumulates them)."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the code as it was
# ---------------------------------------------------------------------------

def _legacy_explode(peak_details_df):
    """The two lines both ``summarize_per_well*`` functions used to run.

    Verbatim, so the damage below is the real damage and not a narration of it.
    """
    split_columns = peak_details_df['ID'].str.split('_', expand=True)
    peak_details_df[['plateID', 'rowID', 'columnID', 'fieldID',
                     'object_number']] = split_columns
    peak_details_df['well_ID'] = (peak_details_df['rowID'] + '_'
                                  + peak_details_df['columnID'])
    return peak_details_df


def _peaks(ids, infected=None):
    """A minimal peak-details frame: one peak per id.

    ``ID`` is object-typed explicitly so that the empty case is still a column
    of strings — the frame ``analyze_calcium_oscillations`` builds always is.
    """
    n = len(ids)
    return pd.DataFrame({
        'ID': pd.Series(list(ids), dtype=object),
        'time': [1.0] * n,
        'amplitude': [0.5] * n,
        'AUC': [1.0] * n,
        'infected': [0.0] * n if infected is None else list(infected),
    })


# ---------------------------------------------------------------------------
# summarize_per_well: the object key
# ---------------------------------------------------------------------------

def test_a_plate_id_with_an_underscore_no_longer_breaks_the_summary():
    """``exp_1_r1_c1_f1_o1`` has six tokens, and five columns were assigned."""
    ids = ['exp_1_r1_c1_f1_o1', 'exp_1_r1_c1_f1_o2', 'exp_1_r2_c1_f1_o1']

    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_explode(_peaks(ids))

    df = _peaks(ids)
    out = summarize_per_well(df)

    # The plate keeps its underscore; the well is still the well.
    assert list(df['plateID'].unique()) == ['exp_1']
    assert list(df['rowID']) == ['r1', 'r1', 'r2']
    assert list(df['object_number']) == ['o1', 'o2', 'o1']
    assert list(df['well_ID']) == ['r1_c1', 'r1_c1', 'r2_c1']
    assert list(out['well_ID']) == ['r1_c1', 'r2_c1']
    assert list(out['cells_per_well']) == [2, 1]
    assert list(out['peaks_per_well']) == [2, 1]


def test_a_timelapse_object_key_keeps_its_timepoint_out_of_the_identity():
    """``plate1_r1_c1_f1_t3_o7``: the timepoint is recognised, not counted."""
    ids = ['plate1_r1_c1_f1_t3_o7', 'plate1_r1_c1_f1_t4_o7',
           'plate1_r1_c1_f1_t3_o8']

    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_explode(_peaks(ids))

    df = _peaks(ids)
    out = summarize_per_well(df)

    assert list(df['fieldID'].unique()) == ['f1']
    assert list(df['object_number']) == ['o7', 'o7', 'o8']
    # Two cells in the well, three peaks between them — the object key
    # identifies a TRACK, so the same cell at t3 and t4 is one cell.
    assert list(out['cells_per_well']) == [2]
    assert list(out['peaks_per_well']) == [3]


def test_a_field_key_is_refused_instead_of_counting_timepoints_as_cells():
    """A ``prcf`` has five tokens too, and the old split took it silently."""
    ids = ['plate1_r1_c1_f1_t1', 'plate1_r1_c1_f1_t2', 'plate1_r1_c1_f1_t3']

    legacy = _legacy_explode(_peaks(ids))
    # This is the quiet failure. Nothing raises, the well even comes out
    # right — but the TIMEPOINT landed in 'object_number', which is what
    # cells_per_well counts, so three frames of one field are reported as
    # three cells and peaks_per_cell is divided by an invented number.
    assert list(legacy['well_ID'].unique()) == ['r1_c1']
    assert list(legacy['object_number']) == ['t1', 't2', 't3']
    assert legacy.groupby('well_ID')['object_number'].nunique().tolist() == [3]

    with pytest.raises(schema.KeyParseError) as excinfo:
        summarize_per_well(_peaks(ids))
    message = str(excinfo.value)
    assert 'summarize_per_well' in message
    assert 'plate1_r1_c1_f1_t1' in message


def test_the_legacy_object_spelling_without_the_o_prefix_still_parses():
    """A peak_details.csv from before the ``o`` prefix must still summarise."""
    df = _peaks(['plate1_r1_c1_f1_1', 'plate1_r1_c1_f1_2'])
    out = summarize_per_well(df)

    assert list(df['object_number']) == ['o1', 'o2']
    assert list(out['cells_per_well']) == [2]


def test_an_empty_peak_table_summarises_to_an_empty_frame():
    """No peaks is not an error. The positional split made it one.

    ``str.split`` on an empty column yields a frame with no columns at all, so
    assigning it to five keys raised ``ValueError: Columns must be same length
    as key`` — and, once that was gone, a list of nothing typed the identity
    columns float64 and building the well id blew up in numpy instead.
    """
    empty = _peaks([])

    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_explode(_peaks([]))

    out = summarize_per_well(empty)
    assert len(out) == 0
    assert {'well_ID', 'peaks_per_well', 'cells_per_well'} <= set(out.columns)


def test_the_identity_columns_are_written_by_position():
    """A repeated index must not scramble the identities.

    ``_read_and_merge_data`` and every ``pd.concat`` above this function can
    hand back a frame whose index repeats, and aligning on the label would
    then write one row's identity onto another's.
    """
    df = _peaks(['plate1_r1_c1_f1_o1', 'plate1_r2_c1_f1_o2'])
    df.index = [7, 7]

    summarize_per_well(df)

    assert list(df['rowID']) == ['r1', 'r2']
    assert list(df['object_number']) == ['o1', 'o2']


def test_summarize_per_well_inf_non_inf_reads_the_same_keys():
    """The infected/uninfected split parses its ids the same way."""
    ids = ['exp_1_r1_c1_f1_o1', 'exp_1_r1_c1_f1_o2']

    with pytest.raises(ValueError, match='Columns must be same length as key'):
        _legacy_explode(_peaks(ids))

    df = _peaks(ids, infected=[2.0, 0.0])
    out = summarize_per_well_inf_non_inf(df)

    assert list(df['plateID'].unique()) == ['exp_1']
    assert list(zip(out['well_ID'], out['infected_status'])) == [
        ('r1_c1', 'infected'), ('r1_c1', 'non_infected')]
    assert list(out['cells_per_well']) == [1, 1]


def test_summarize_per_well_inf_non_inf_refuses_a_field_key():
    """Same contract, same named error, from the other entry point."""
    ids = ['plate1_r1_c1_f1_t1', 'plate1_r1_c1_f1_t2']
    with pytest.raises(schema.KeyParseError,
                       match='summarize_per_well_inf_non_inf'):
        summarize_per_well_inf_non_inf(_peaks(ids))


# ---------------------------------------------------------------------------
# submodules._ensure_field_key: the field key is COMPOSED, not concatenated
# ---------------------------------------------------------------------------

def _field_frame(**overrides):
    """Two rows of one field at two timepoints."""
    frame = pd.DataFrame({
        'plateID': ['plate1', 'plate1'],
        'rowID': ['r1', 'r1'],
        'columnID': ['c1', 'c1'],
        'fieldID': ['f1', 'f1'],
        'timeID': ['t1', 't2'],
    })
    for key, value in overrides.items():
        frame[key] = value
    return frame


def test_the_canonical_timelapse_key_is_unchanged():
    """What the real writer produces must survive untouched."""
    built = submodules._ensure_field_key(_field_frame())
    assert built['prcf'].tolist() == ['plate1_r1_c1_f1_t1',
                                      'plate1_r1_c1_f1_t2']


def test_an_integer_timepoint_builds_a_key_schema_can_read_back():
    """A ``timeID`` of ``1`` used to build a key nothing could parse.

    Concatenation put the column in as it stood, so the key ended
    ``..._f1_1``. Read back, that trailing ``1`` is not a ``t<N>``, so
    :func:`spacr.schema.parse_prcf` sees a four-token key whose *field* is
    ``'1'`` and raises — the key was unreadable to every downstream parser
    while still looking perfectly plausible in the CSV.
    """
    frame = _field_frame(timeID=[1, 2])

    legacy = 'plate1_r1_c1_f1_' + frame['timeID'].astype(str)
    with pytest.raises(schema.KeyParseError):
        schema.parse_prcf(legacy.iloc[0])

    built = submodules._ensure_field_key(frame.copy())
    assert built['prcf'].tolist() == ['plate1_r1_c1_f1_t1',
                                      'plate1_r1_c1_f1_t2']
    assert [schema.parse_prcf(key).timeID for key in built['prcf']] == \
        ['t1', 't2']


def test_a_time_blind_key_written_the_old_way_is_still_repaired(capsys):
    """Repair-on-read must survive the switch to composed keys.

    The stored key here is the *concatenated* time-blind one, spelled with
    bare integer row/column/field ids — what an older run wrote and what the
    composed key no longer matches character for character.
    """
    frame = _field_frame(rowID=[1, 1], columnID=[1, 1], fieldID=[1, 1],
                         timeID=[1, 2], prcf=['plate1_1_1_1', 'plate1_1_1_1'])

    built = submodules._ensure_field_key(frame.copy(), source='the cell table')

    assert built['prcf'].tolist() == ['plate1_r1_c1_f1_t1',
                                      'plate1_r1_c1_f1_t2']
    assert 'Repaired 2 time-blind prcf value(s) in the cell table' \
        in capsys.readouterr().out


def test_a_prcf_from_another_source_is_left_alone():
    """Only a provably time-blind key is repaired; anything else is kept."""
    frame = _field_frame(prcf=['imported_key_a', 'imported_key_b'])
    built = submodules._ensure_field_key(frame.copy())
    assert built['prcf'].tolist() == ['imported_key_a', 'imported_key_b']


def test_a_table_that_needs_no_key_is_never_composed():
    """No time axis and a prcf already present: nothing is built, so nothing
    can fail. The plate id here would not compose, and must not have to."""
    frame = pd.DataFrame({
        'plateID': ['exp_1'], 'rowID': ['r1'], 'columnID': ['c1'],
        'fieldID': ['f1'], 'prcf': ['exp_1_r1_c1_f1'],
    })
    built = submodules._ensure_field_key(frame.copy())
    assert built['prcf'].tolist() == ['exp_1_r1_c1_f1']


def test_a_plate_id_carrying_the_separator_is_escaped_not_refused():
    """A composed key must be splittable again -- by escaping, not by refusing.

    This test used to pin the refusal, and instruction 100 replaced it with a
    reversible escape: ``exp_1`` is written ``exp%5F1``, so the key still has
    exactly one separator per component and ``parse_prcf`` gives the plate
    back character for character. A plate genuinely named ``exp_1`` is a
    reasonable thing for a user to have, and refusing it stopped both object
    assays on a table nothing was wrong with.
    """
    frame = _field_frame(plateID=['exp_1', 'exp_1'])
    built = submodules._ensure_field_key(frame.copy(), source="table 'pathogen'")

    assert built['prcf'].tolist() == ['exp%5F1_r1_c1_f1_t1',
                                      'exp%5F1_r1_c1_f1_t2']
    for key in built['prcf']:
        field = schema.parse_prcf(key)
        assert field.plateID == 'exp_1'
        assert field.prcf == key      # the escape survives a round trip


# ---------------------------------------------------------------------------
# join contracts
# ---------------------------------------------------------------------------

def test_child_features_are_summarised_one_row_per_parent():
    """The many_to_one join keeps one row per (frame, parent)."""
    overlaps = pd.DataFrame({
        'frame': [0, 0, 1],
        'track_id': [1, 1, 2],
        'nucleus_label': [10, 11, 10],
    })
    props = pd.DataFrame({
        'frame': [0, 0, 1],
        'nucleus_label': [10, 11, 10],
        'nucleus_area': [5.0, 7.0, 9.0],
    })

    out = _summarise_child_features_per_parent(
        overlaps_df=overlaps, child_props_df=props,
        parent_label_col='track_id', child_label_col='nucleus_label',
        count_col_name='n_nuclei')

    assert len(out) == 2
    assert out['n_nuclei'].tolist() == [2, 1]
    # '*area*' aggregates by sum, so the parent's two nuclei add up.
    assert out['nucleus_area'].tolist() == [12.0, 9.0]


def test_a_child_measured_twice_in_one_frame_is_caught_not_absorbed():
    """A duplicated regionprops row would count that child twice, silently."""
    overlaps = pd.DataFrame({
        'frame': [0, 0],
        'track_id': [1, 1],
        'nucleus_label': [10, 11],
    })
    props = pd.DataFrame({
        'frame': [0, 0, 0],
        'nucleus_label': [10, 10, 11],          # label 10 written twice
        'nucleus_area': [5.0, 5.0, 7.0],
    })

    with pytest.raises(MergeError):
        _summarise_child_features_per_parent(
            overlaps_df=overlaps, child_props_df=props,
            parent_label_col='track_id', child_label_col='nucleus_label',
            count_col_name='n_nuclei')


# --- compare_reads_to_scores: two plates share a row id --------------------

PC = 'TGGT1_220950_1'
NC = 'TGGT1_233460_4'
EMPIRICAL = {'r1': (90, 10), 'r2': (60, 40)}


def _reads_csv(path, plate):
    """Per-gRNA read counts for two rows of one column."""
    recs = []
    for i, row in enumerate(('r1', 'r2')):
        for grna, count in ((PC, 10 * (i + 1)), (NC, 10 * (2 - i))):
            recs.append({'plateID': plate, 'rowID': row, 'columnID': 'c3',
                         'grna_name': grna, 'count': count})
    pd.DataFrame(recs).to_csv(path, index=False)
    return str(path)


def _scores_csv(path, plate):
    """Per-object classifier calls for the same wells."""
    recs = []
    for i, row in enumerate(('r1', 'r2')):
        for label in (0, 1, 1):
            recs.append({'plateID': plate, 'rowID': row, 'columnID': 'c3',
                         'cv_predictions': label})
    pd.DataFrame(recs).to_csv(path, index=False)
    return str(path)


def test_two_plates_sharing_a_row_id_do_not_fan_out(tmp_path, monkeypatch):
    """The empirical join is many_to_one, and must stay that way.

    ``empirical_dict`` is keyed by plate ROW — the mixing ratio that row was
    seeded with — so on a two-plate run two wells legitimately match one key.
    Declaring that join one_to_one for strictness would raise MergeError on
    this, the ordinary multi-plate call the function is written for.

    The frame handed to ``display`` is the one the plots are drawn from, and
    it is the only externally visible form the join takes (the plot itself
    averages duplicate x values, so a fan-out would not show up there).
    """
    from spacr.submodules import compare_reads_to_scores

    shown = []
    monkeypatch.setattr(submodules, 'display', shown.append)

    reads = [_reads_csv(tmp_path / 'reads1.csv', 'plate1'),
             _reads_csv(tmp_path / 'reads2.csv', 'plate2')]
    scores = [_scores_csv(tmp_path / 'scores1.csv', 'plate1'),
              _scores_csv(tmp_path / 'scores2.csv', 'plate2')]

    figs = compare_reads_to_scores(reads, scores, empirical_dict=EMPIRICAL,
                                   save_paths=[None, None])

    assert len(figs) == 2
    merged = shown[-1]
    # 2 plates x 2 rows = 4 wells, each exactly once. A fan-out on either
    # join would give 8.
    assert sorted(merged['prc']) == ['plate1_r1_c3', 'plate1_r2_c3',
                                     'plate2_r1_c3', 'plate2_r2_c3']
    # Both plates' r1 wells picked up r1's empirical mixture.
    assert merged.loc[merged['rowID'] == 'r1', 'pc_fraction'].tolist() == \
        [0.9, 0.9]


# --- generate_score_heatmap: the left side may legitimately repeat ---------

def _heatmap_inputs(tmp_path, duplicate_well=False):
    """A one-model score folder, a reads CSV and a CV CSV for two wells."""
    rows = ('r1', 'r2')

    folder = tmp_path / 'models'
    (folder / 'modelA').mkdir(parents=True)
    pd.DataFrame([{'columnID': 'c3', 'rowID': row, 'pred': 0.25 * (i + 1)}
                  for i, row in enumerate(rows)]
                 ).to_csv(folder / 'modelA' / 'scores.csv', index=False)

    reads = [{'columnID': 'c3', 'rowID': row, 'grna_name': name,
              'count': count}
             for row in rows
             for name, count in (('sgA', 30), ('sgB', 70))]
    if duplicate_well:
        # The same gRNA counted twice for one well: two sequencing runs
        # appended to one CSV. Legitimate input, and it stays two rows.
        reads.append({'columnID': 'c3', 'rowID': 'r1', 'grna_name': 'sgA',
                      'count': 30})
    mixed = tmp_path / 'mixed.csv'
    pd.DataFrame(reads).to_csv(mixed, index=False)

    cv = tmp_path / 'cv.csv'
    pd.DataFrame([{'columnID': 'c3', 'rowID': row, 'pred_cv': 0.5}
                  for row in rows]).to_csv(cv, index=False)

    dst = tmp_path / 'out'
    dst.mkdir()
    return {
        'folders': [str(folder)], 'csv_name': 'scores.csv',
        'data_column': 'pred', 'csv': str(mixed), 'cv_csv': str(cv),
        'data_column_cv': 'pred_cv', 'plateID': 1, 'columnID': 'c3',
        'control_sgrnas': ['sgA', 'sgB'], 'fraction_grna': 'sgA',
        'cmap': 'coolwarm', 'dst': str(dst),
    }


def test_a_well_listed_twice_in_the_reads_csv_is_not_a_crash(tmp_path):
    """The score/CV joins are many_to_one on purpose.

    What must be unique is the SCORE side — one row per well, or the well is
    drawn twice in the heatmap. The reads side is the caller's CSV and may
    legitimately list a gRNA twice for a well; one_to_one there would turn
    that into a MergeError instead of two visible rows.
    """
    from spacr.submodules import generate_score_heatmap

    settings = _heatmap_inputs(tmp_path, duplicate_well=True)
    out = generate_score_heatmap(settings)

    assert isinstance(out, pd.DataFrame)
    # Two wells, one of them listed twice.
    assert len(out) == 3
    assert sorted(out['prc'].tolist()) == ['plate1_r1_c3', 'plate1_r1_c3',
                                           'plate1_r2_c3']
    assert 'modelA_pred' in out.columns
    assert np.isfinite(out['fraction']).all()


# ---------------------------------------------------------------------------
# timelapse._track_well_ids: naming a well must not be able to end a run
# ---------------------------------------------------------------------------
#
# The btrack track table composes 'wellID' out of the rowID/columnID
# utils._map_wells parsed, instead of re-splitting the npz batch name on '_'
# by position. schema.well_id is the right composer but it REFUSES two pairs
# _map_wells routinely produces, and calling it bare made an uncaught
# KeyParseError out of a batch the pipeline used to track without complaint.

def _well_ids(rows, columns, name='plate1_A01_1'):
    """Call the helper with a throwaway logger and return the ids."""
    import logging

    from spacr.timelapse import _track_well_ids

    return _track_well_ids(rows, columns, name,
                           logging.getLogger('test.trackwell'))


def test_a_row_and_column_pair_is_named_the_canonical_way():
    """The point of composing it: canonical spelling, whatever the file said."""
    assert _well_ids(['r1', 'r27'], ['c1', 'c13']) == ['A01', 'AA13']


def test_a_positional_well_keeps_its_own_number_instead_of_crashing():
    """``plate1_5_3`` is plate1, well 5, field 3.

    schema.parse_well passes a bare well number through into BOTH slots, so
    _map_wells hands back rowID == columnID == '5' and there is no 'A01'-style
    name to render. schema.well_id raises KeyParseError on that pair by
    design. The well is still '5' — which is exactly what the positional
    ``file_name.str.split('_').str[1]`` this replaced wrote — so that is what
    the track table has to say.
    """
    from spacr.utils import _map_wells

    _, row, column, _, _ = _map_wells('plate1_5_3')
    assert (row, column) == ('5', '5')
    with pytest.raises(schema.KeyParseError):
        schema.well_id(row, column)

    assert _well_ids([row], [column], name='plate1_5_3') == ['5']


@pytest.mark.parametrize('well, expected', [
    ('a1', 'A01'),          # lowercase
    ('A-01', 'A01'),        # separator inside the well
    (' A01 ', 'A01'),       # whitespace-padded
    ('AA13', 'AA13'),       # 1536-plate row
])
def test_the_well_id_agrees_with_the_row_and_column_beside_it(well, expected):
    """What composing it actually buys, as opposed to what was claimed.

    The old ``file_name.str.split('_').str[1]`` copied the file's spelling
    through while rowID/columnID beside it were canonicalised, so a track
    table could say wellID 'a1' next to rowID 'r1'/columnID 'c1'. Composing
    from the parsed pair makes the three agree character for character.

    (The claim that this also repairs a plate id containing an underscore is
    NOT true and is not asserted: schema.parse_field_stem splits a field stem
    left to right, so an underscored plate loses its tail to the well slot
    there too -- see the test below.)
    """
    from spacr.utils import _map_wells

    _, row, column, _, _ = _map_wells(f'plate1_{well}_3')
    assert _well_ids([row], [column], name=f'plate1_{well}_3') == [expected]


def test_an_underscored_plate_is_not_repaired_by_the_composition():
    """Pinned so the retracted claim cannot quietly come back.

    ``exp_plate1_A01_3`` has four tokens and parse_field_stem takes the first
    three, so 'plate1' lands in the WELL slot as a positional passthrough --
    which is also what token [1] of the old split said. The composition
    changes nothing here; it is not the fix for an underscored plate.
    """
    from spacr.utils import _map_wells

    plate, row, column, field, _ = _map_wells('exp_plate1_A01_3')
    assert (plate, row, column, field) == ('exp', 'plate1', 'plate1', 'f1')
    assert 'exp_plate1_A01_3'.split('_')[1] == 'plate1'      # the old answer
    assert _well_ids([row], [column], name='exp_plate1_A01_3') == ['plate1']


def test_a_name_map_wells_cannot_read_gives_the_error_sentinel_not_a_crash(
    caplog,
):
    """_map_wells does not raise for an unreadable name — it returns 'error'.

    Every identity column of the track table then carries that string, and
    ``('error', 'error')`` is an unprefixed equal pair, i.e. exactly what
    schema.is_positional_pair flags. wellID joins its siblings rather than
    ending the run, and the batch name is logged.
    """
    import logging

    from spacr.utils import _map_wells

    assert _map_wells('garbage') == ('error',) * 5
    with caplog.at_level(logging.WARNING, logger='test.trackwell'):
        assert _well_ids(['error'] * 3, ['error'] * 3, name='garbage') == \
            ['error'] * 3
    # Once for the batch, not once per track row.
    assert sum('garbage' in r.getMessage() for r in caplog.records) == 1


def test_a_malformed_well_is_refused_by_name_and_says_what_to_do():
    """Column 0 is not a well column, and this is not a positional pair.

    The only case left over. It still fails — a wellID of 'A00' would be a
    well name for a well that is on no plate — but it fails with the batch
    name and the offending pair in the message, which the bare
    ``cannot build a well name from column 'c0'`` does not carry.
    """
    with pytest.raises(schema.KeyParseError) as excinfo:
        _well_ids(['r1'], ['c0'], name='plate1_A00_1')
    message = str(excinfo.value)
    assert 'plate1_A00_1' in message
    assert "'c0'" in message
    assert 'Rename' in message


# ---------------------------------------------------------------------------
# submodules.translate_well_in_df: the refusal must describe the real writer
# ---------------------------------------------------------------------------

def test_a_short_renamed_tiff_is_refused_without_blaming_the_converters(
    tmp_path,
):
    """Both converters always write ``plate<N>_<well>_<vendor token>``.

    io.convert_to_yokogawa and io.convert_separate_files_to_yokogawa build
    every name from io._next_synthetic_yokogawa_well, which returns
    'plate<N>_<well>' — no input format makes either write a well-only name.
    The refusal here must not send the reader to look at a branch that does
    not exist.
    """
    import inspect

    from spacr import io as spacr_io

    # The claim the message makes, checked against the writers themselves.
    source = inspect.getsource(spacr_io.convert_to_yokogawa)
    assert '_next_synthetic_yokogawa_well' in source
    assert spacr_io._next_synthetic_yokogawa_well(set()) == 'plate1_A01'

    translate = _translate_well_in_df()
    csv = tmp_path / 'rename_log.csv'
    pd.DataFrame({'Renamed TIFF': ['A01_T0001F001L01C01.tif']}).to_csv(
        csv, index=False)

    with pytest.raises(schema.KeyParseError) as excinfo:
        translate(str(csv))
    message = str(excinfo.value)
    assert 'A01_T0001F001L01C01.tif' in message
    assert 'plate<N>_<well>_<vendor token>' in message
    # The retracted claim, in the spelling it was written in.
    assert 'well-only names it writes' not in message


def test_a_well_formed_rename_log_still_parses(tmp_path):
    """The refusal above must not have eaten the names that do parse."""
    translate = _translate_well_in_df()
    csv = tmp_path / 'rename_log.csv'
    pd.DataFrame({'Renamed TIFF': ['plate1_A01_T0001F001L01C01.tif',
                                   'exp_plate1_B02_T0001F001L01C01.tiff']}
                 ).to_csv(csv, index=False)

    out = translate(str(csv))
    assert out['plateID'].tolist() == ['plate1', 'exp_plate1']
    assert out['well'].tolist() == ['A01', 'B02']
    assert out['rowID'].tolist() == ['r1', 'r2']
    assert out['column_name'].tolist() == ['c1', 'c2']


def _translate_well_in_df():
    """Rebuild ``analyze_percent_positive``'s nested ``translate_well_in_df``.

    It closes over the ``schema`` module its parent imports, so the closure is
    rebuilt from ``co_freevars`` rather than assumed empty. Calling the parent
    instead would need a whole measurements.db.
    """
    import types

    code = next(c for c in submodules.analyze_percent_positive.__code__.co_consts
                if isinstance(c, types.CodeType)
                and c.co_name == 'translate_well_in_df')
    assert code.co_freevars == ('schema',)
    closure = (types.CellType(schema),)
    return types.FunctionType(
        code, submodules.analyze_percent_positive.__globals__,
        closure=closure)


# ---------------------------------------------------------------------------
# submodules._compose_field_keys: one compose per identity, one pass per row
# ---------------------------------------------------------------------------

def test_the_field_key_composer_calls_the_composer_once_per_identity(
    monkeypatch,
):
    """The cache is the point: a timelapse table is millions of rows."""
    calls = []
    real = schema.compose_prcf

    def _counting(*identity):
        calls.append(identity)
        return real(*identity)

    monkeypatch.setattr(submodules.schema, 'compose_prcf', _counting)

    frame = pd.DataFrame({
        'plateID': ['plate1'] * 6,
        'rowID': ['r1'] * 6,
        'columnID': ['c1'] * 6,
        'fieldID': ['f1', 'f1', 'f1', 'f2', 'f2', 'f2'],
    })
    keys = submodules._compose_field_keys(frame, None, 'the parasite table')

    assert keys.tolist() == ['plate1_r1_c1_f1'] * 3 + ['plate1_r1_c1_f2'] * 3
    assert len(calls) == 2                      # two fields, six rows
    # And the shared key string is shared, not rebuilt per row.
    assert keys.iloc[0] is keys.iloc[1]


def test_the_field_key_composer_reports_the_first_failure_deterministically():
    """The example in the message used to be drawn from a ``set``.

    Which identity it named therefore varied between runs of the same data.
    It is now the first failing row, in row order, and the count is still the
    number of DISTINCT failing identities.

    An underscored plate no longer fails -- it is escaped -- so the failing
    input here is the one identity that cannot be a key at all: an empty
    plate id. Two distinct spellings of empty, on three rows.
    """
    frame = pd.DataFrame({
        'plateID': ['plate1', '', '   ', ''],
        'rowID': ['r1'] * 4,
        'columnID': ['c1'] * 4,
        'fieldID': ['f1'] * 4,
    })

    with pytest.raises(schema.KeyParseError) as excinfo:
        submodules._compose_field_keys(frame, None, "table 'pathogen'")
    message = str(excinfo.value)
    assert '2 identity/identities' in message    # '' and '   ', not 3 rows
    assert "'plateID': ''" in message            # the FIRST failing row
    assert "'plateID': '   '" not in message
    assert "table 'pathogen'" in message


def test_the_field_key_composer_handles_an_empty_frame():
    """No rows is not a failure, and the result still aligns to the index."""
    frame = pd.DataFrame({'plateID': [], 'rowID': [], 'columnID': [],
                          'fieldID': []}, dtype=object)
    keys = submodules._compose_field_keys(frame, None, 'the parasite table')
    assert keys.empty
    assert keys.dtype == object
    assert keys.index.equals(frame.index)
