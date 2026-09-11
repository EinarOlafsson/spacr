"""Positive and deliberately corrupted native-curve counterparts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('profiler_example', Path(__file__).parents[1]/'profiler_example.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def curve():
    return dict(variable='input_a', values=[0., .5, 1.], predictions=[1.25, 3.25, 5.25],
                held={'input_b': .5}, scale='response')


def verify(value):
    return example.verify_curve(value, {'Intercept': 2., 'input_a': 4., 'input_b': -1.5},
                                variable='input_a', held={'input_b': .5}, points=3)


def test_every_value_checked_after_positive_counterpart():
    assert verify(curve())['maximum_absolute_error'] == 0
    broken = curve(); broken['predictions'][1] += .1
    with pytest.raises(ValueError, match='saved OLS'): verify(broken)


@pytest.mark.parametrize('key,value', [('variable','input_b'), ('scale','probability'),
                                      ('values',[0.,1.]), ('predictions',[1.25,5.25])])
def test_identity_and_length_after_positive_counterpart(key,value):
    assert verify(curve())['checked']
    broken = curve(); broken[key] = value
    with pytest.raises(ValueError, match='identity'): verify(broken)


def test_range_and_held_after_positive_counterpart():
    assert verify(curve())['checked']
    broken = curve(); broken['values'][1] = .4
    with pytest.raises(ValueError, match='range'): verify(broken)
    broken = curve(); broken['held']['input_b'] = .1
    with pytest.raises(ValueError, match='held input'): verify(broken)


def test_prepare_is_fitted_synthetic_and_never_overwrites(tmp_path):
    path = tmp_path/'example'; manifest = example.prepare(path)
    assert manifest['synthetic'] and manifest['rows'] == 121
    assert manifest['independent_coefficient_max_error'] < 1e-12
    assert manifest['observed_ranges'] == {'input_a':[0,1], 'input_b':[0,1]}
    assert np.isclose(manifest['coefficients']['input_a'], 4, atol=.05)
    before = {p.name:p.read_bytes() for p in path.iterdir()}
    with pytest.raises(FileExistsError): example.prepare(path)
    assert {p.name:p.read_bytes() for p in path.iterdir()} == before
