"""Nonfinite manual cuts cannot silently replace an accepted viability table."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.tabular import read_table, write_database
from tests.test_live_dead_viability import _signal_table

SOURCE = 'Manual viability thresholds must be finite numbers or automatic.'


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 'nan', 'Inf', '-Infinity'])
@pytest.mark.parametrize('position', [0, 1])
def test_parser_refuses_nonfinite_explicit_dead_and_live_cuts(value, position):
    values = [None, None]
    values[position] = value
    before = copy.deepcopy(values)
    with pytest.raises(ValueError) as error:
        measure._viability_manual(values)
    assert str(error.value) == SOURCE
    np.testing.assert_equal(values, before)


@pytest.mark.parametrize('values,expected', [
    (None, (None, None)), ('', (None, None)), ([], (None, None)),
    ([None, 'auto'], (None, None)), (['None', 'none'], (None, None)),
    ([0], (0., None)), ([-10, 20], (-10., 20.)),
    (['-0.5', '1e300'], (-.5, 1e300)), ('[5, 7]', (5., 7.)),
])
def test_valid_finite_and_automatic_settings_keep_existing_meaning(values, expected):
    before = copy.deepcopy(values)
    assert measure._viability_manual(values) == expected
    np.testing.assert_equal(values, before)


@pytest.fixture
def accepted_database(tmp_path):
    path = tmp_path / 'measurements.db'
    objects = _signal_table(n_live=30, n_dead=10)
    write_database(objects, path, 'nucleus', if_exists='replace', canonicalise=False)
    write_database(pd.DataFrame({'accepted_previous': [42]}), path, 'viability',
                   if_exists='replace', canonicalise=False)
    return path, objects


def settings(thresholds):
    return dict(channels=[0, 1, 2], viability_dead_channel=1,
                viability_live_channel=None, viability_thresholds=thresholds,
                plot=False)


def assert_refusal_keeps_database(path, config, expected):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original = copy.deepcopy(config)
    with pytest.raises(ValueError) as error:
        measure._classify_viability(str(path), config, plot=False)
    assert str(error.value) == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    np.testing.assert_equal(config, original)
    assert read_table(path, table='viability', report=None).accepted_previous.tolist() == [42]


@pytest.mark.parametrize('cut', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('role', ['dead', 'live'])
def test_actual_classifier_refuses_before_replacing_prior_outputs(accepted_database, cut, role):
    path, _ = accepted_database
    config = settings([cut, None] if role == 'dead' else [None, cut])
    if role == 'live':
        config.update(viability_dead_channel=None, viability_live_channel=2)
    assert_refusal_keeps_database(path, config, SOURCE)


def test_valid_finite_cut_still_classifies_declared_synthetic_signals(accepted_database):
    path, _ = accepted_database
    original = read_table(path, table='nucleus', report=None)
    table, _ = measure._classify_viability(str(path), settings([100, None]), plot=False)
    assert table.viability_state.value_counts().to_dict() == {'live': 30, 'dead': 10}
    assert table.dead_threshold.eq(100).all()
    pd.testing.assert_frame_equal(read_table(path, table='nucleus', report=None), original)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_installed_reviewed_language_at_classifier_refusal(accepted_database, language, monkeypatch):
    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-viability-finite-threshold.json'
    record = json.loads(review.read_text(encoding="utf-8"))['records'][0]
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    path, _ = accepted_database
    assert record['translation'] != SOURCE
    assert_refusal_keeps_database(path, settings(['NaN', None]), record['translation'])
