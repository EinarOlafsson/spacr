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


def test_a_count_file_holding_more_plates_is_narrowed_to_the_score_file(
        tmp_path, capsys):
    """When the score names a subset, the count rows outside it are dropped.

    A consolidated count file is often paired with one plate's scores; keeping
    the other plates' rows would put wells in the join that the score side
    cannot match, and the pairing check would then read as a broken screen.
    """
    score = write(score_frame(('plate1',)), tmp_path / 'scores.csv')
    count = write(count_frame(('plate1', 'plate2')), tmp_path / 'counts.csv')

    counts, scores, audit = ml.load_regression_input_pairs(
        [{'score': score, 'count': count}])

    assert list(counts['plateID'].unique()) == ['plate1']
    assert list(scores['plateID'].unique()) == ['plate1']
    assert audit[0]['rule'] == 'matched count rows to score-file plate subset'
    assert audit[0]['plate'] == 'plate1'
    assert 'matched count rows to score-file plate subset' in \
        capsys.readouterr().out


def test_a_count_file_of_several_plates_is_assigned_by_the_pair_row(tmp_path):
    """With the score naming no plate, the row's own position picks one.

    A count CSV written by the sequencing side carries every plate while the
    score file carries none; the pair row already says which plate the row is,
    and it is used only when the count file actually holds that plate.
    """
    score = write(
        score_frame(('plate1',)).drop(columns=['plateID']),
        tmp_path / 'scores.csv')
    count = write(count_frame(('plate1', 'plate2')), tmp_path / 'counts.csv')

    counts, scores, audit = ml.load_regression_input_pairs(
        [{'score': score, 'count': count}])

    assert list(counts['plateID'].unique()) == ['plate1']
    assert list(scores['plateID'].unique()) == ['plate1']
    assert audit[0]['rule'].startswith('assigned from pair row order')
    assert 'count file holds 2 plates' in audit[0]['rule']
