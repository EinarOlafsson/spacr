"""Pairing score and count files when only one of them names its plates.

The two inputs are joined on ``plateID_rowID_columnID``, so which plate a file
belongs to decides which wells meet. Guessing it from the filename is what this
resolution order replaced; these cover the two cases where one side has to be
narrowed to the other, and the case where the pair row's own position is the
only thing that says which plate a file is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ml


def write(frame, path):
    frame.to_csv(path, index=False)
    return str(path)


def score_frame(plates=('plate1',)):
    return pd.DataFrame({
        'plateID': list(plates),
        'rowID': ['r1'] * len(plates),
        'columnID': ['c1'] * len(plates),
        'pathogen_rate': np.linspace(0.4, 0.6, len(plates)),
    })


def count_frame(plates=('plate1', 'plate2')):
    return pd.DataFrame({
        'plateID': list(plates),
        'rowID': ['r1'] * len(plates),
        'columnID': ['c1'] * len(plates),
        'grna': [f'g{i + 1}' for i in range(len(plates))],
        'count': [10 * (i + 1) for i in range(len(plates))],
    })


def case_a_count_file_holding_more_plates_is_narrowed_to_the_score_file(
        tmp_path, capsys):
    """When the score names a subset, the count rows outside it are dropped.

    A consolidated count file is often paired with one plate's scores; keeping
    the other plates' rows would put wells in the join that the score side
    cannot match, and the pairing check would then read as a broken screen.
    """
    _seen = []
    score = write(score_frame(('plate1',)), tmp_path / 'scores.csv')
    count = write(count_frame(('plate1', 'plate2')), tmp_path / 'counts.csv')

    counts, scores, audit = ml.load_regression_input_pairs(
        [{'score': score, 'count': count}])

    _seen.append(keep(list(counts['plateID'].unique())))
    _seen.append(keep(list(scores['plateID'].unique())))
    _seen.append(keep(audit[0]['rule']))
    _seen.append(keep(audit[0]['plate']))
    _seen.append(('matched count rows to score-file plate subset' if 'matched count rows to score-file plate subset' in (capsys.readouterr().out) else ("missing", 'matched count rows to score-file plate subset')))
    return _seen


def case_a_count_file_of_several_plates_is_assigned_by_the_pair_row(tmp_path):
    """With the score naming no plate, the row's own position picks one.

    A count CSV written by the sequencing side carries every plate while the
    score file carries none; the pair row already says which plate the row is,
    and it is used only when the count file actually holds that plate.
    """
    _seen = []
    score = write(
        score_frame(('plate1',)).drop(columns=['plateID']),
        tmp_path / 'scores.csv')
    count = write(count_frame(('plate1', 'plate2')), tmp_path / 'counts.csv')

    counts, scores, audit = ml.load_regression_input_pairs(
        [{'score': score, 'count': count}])

    _seen.append(keep(list(counts['plateID'].unique())))
    _seen.append(keep(list(scores['plateID'].unique())))
    _seen.append(bool(
        audit[0]['rule'].startswith('assigned from pair row order')))
    _seen.append(('count file holds 2 plates' if 'count file holds 2 plates' in (audit[0]['rule']) else ("missing", 'count file holds 2 plates')))
    return _seen


def case_two_files_naming_different_plates_are_refused(tmp_path):
    """Neither side is a subset of the other, so nothing can narrow them.

    The three resolution rules all end in ONE plate identity for the row.
    When both files name plates and the sets are disjoint, choosing either
    one silently relabels a whole file's wells -- the join still succeeds,
    the row count still looks right, and every effect size is attached to the
    wrong plate. That is the exact failure the paired table exists to end,
    so it raises and names both sides rather than picking.
    """
    _seen = []
    score = write(score_frame(('plate1',)), tmp_path / 'scores.csv')
    count = write(count_frame(('plate3', 'plate4')), tmp_path / 'counts.csv')

    with pytest.raises(ValueError) as caught:
        ml.load_regression_input_pairs([{'score': score, 'count': count}])
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))

    message = str(caught.value)
    _seen.append(('conflicts' if 'conflicts' in (message) else ("missing", 'conflicts')))
    _seen.append(("'plate1'" if "'plate1'" in (message) else ("missing", "'plate1'")))
    _seen.append(bool(
        "'plate3'" in message and "'plate4'" in message))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_a_count_file_holding_more_plates_is_narrowed_to_the_score_file, [['plate1'], ['plate1'], 'matched count rows to score-file plate subset', 'plate1', 'matched count rows to score-file plate subset'], id="a_count_file_holding_more_plates_is_narrowed_to_the_score_file"),
    pytest.param(case_a_count_file_of_several_plates_is_assigned_by_the_pair_row, [['plate1'], ['plate1'], True, 'count file holds 2 plates'], id="a_count_file_of_several_plates_is_assigned_by_the_pair_row"),
    pytest.param(case_two_files_naming_different_plates_are_refused, ['ValueError', 'conflicts', "'plate1'", True], id="two_files_naming_different_plates_are_refused"),
])
def test_12_ml_inputs_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
