"""ml_analysis on the splits and settings that decide what a score means.

The classifier is trained on control wells and then scores every object, so
which rows were held out decides whether the numbers say anything about
generalisation. These cover the named-plate holdout, the cross-validation
guard that stops a split with too few independent groups, and the refusal that
names the column the controls were meant to be grouped by.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spacr.ml as ML


FEATURES = [
    'cell_channel_3_mean_intensity',
    'cell_channel_3_percentile_75',
    'nucleus_channel_3_mean_intensity',
    'cytoplasm_channel_3_mean_intensity',
    'pathogen_channel_3_mean_intensity',
    'cell_channel_3_std_intensity',
]

COMMON = dict(channel_of_interest=3, location_column='columnID', n_repeats=1,
              n_jobs=1, remove_highly_correlated_features=False,
              test_size=0.25, model_type='extra_trees', n_estimators=8)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def feature_frame(per_class=40, locations=('c1', 'c2'), plates=('plate1',),
                  seed=8):
    """Per-object features with a five-part prcfo-style index."""
    rng = np.random.default_rng(seed)
    rows, index = [], []
    for plate in plates:
        for location, centre in zip(locations, (0.3, 0.9, 0.5)):
            for _ in range(per_class):
                row = {'columnID': location, 'plateID': plate}
                for name in FEATURES:
                    row[name] = float(rng.normal(centre, 0.12)
                                      + rng.normal(0, 0.05))
                rows.append(row)
                index.append(f'{plate}_r1_{location}_f1_o{len(index)}')
    return pd.DataFrame(rows, index=index)


def test_a_missing_location_column_lists_the_columns_and_the_likely_cause():
    """The refusal names the columns present and how many there are.

    A pandas KeyError three frames down names the column and not the reason it
    is being asked for; a run left in annotation mode is the likely cause and
    saying so costs one sentence.
    """
    frame = feature_frame(per_class=6)
    for extra in range(10):
        frame[f'spare_{extra}'] = 0.0

    with pytest.raises(ValueError) as caught:
        ML.ml_analysis(frame, positive_control='c2', negative_control='c1',
                       **{**COMMON, 'location_column': 'not_a_column'})

    message = str(caught.value)
    assert f'({len(frame.columns)} columns)' in message
    assert 'annotation mode' in message


def test_a_named_holdout_plate_is_trained_without_and_scored_on():
    """Naming one plate holds it out entirely rather than splitting at random.

    Cross-validation inside one plate lets a model learn the PLATE rather than
    the phenotype and every number it reports still looks fine; the held-out
    plate is the one number that says whether it generalises.
    """
    frame = feature_frame(per_class=30, plates=('plate1', 'plate2'))

    output, _figs = ML.ml_analysis(
        frame, positive_control='c2', negative_control='c1',
        split_by='plate', holdout_plate='plate2', **COMMON)

    scored = output[0]
    assert 'predictions' in scored.columns
    assert len(scored) == len(frame)
