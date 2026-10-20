import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from spacr import attribution
from tests.test_generative_counterfactuals import _BothChannelEdit, _ReferenceScore


def _crops():
    return torch.stack([torch.full((8, 8), .2), torch.full((8, 8), .3)])[None]


def _target(value, representation):
    if representation == 'list':
        return [value]
    if representation == 'numpy':
        return np.asarray([value])
    return torch.tensor([value])


@pytest.mark.parametrize('representation', ['list', 'numpy', 'tensor'])
@pytest.mark.parametrize('value', [.5, -.5, 1.7, np.nan, np.inf])
def test_invalid_per_crop_class_refuses_before_model_or_generator_transfer(
        monkeypatch, representation, value):
    targets = _target(value, representation)
    before_targets = targets.clone() if isinstance(targets, torch.Tensor) else pickle.dumps(targets)
    model, generator, crops = _ReferenceScore(), _BothChannelEdit(), _crops()
    original = crops.clone()
    calls = []
    original_wrapper = attribution.ClassScoreModel

    def wrapped(*args, **kwargs):
        calls.append('wrapper')
        return original_wrapper(*args, **kwargs)

    original_to = generator.to

    def transfer(*args, **kwargs):
        calls.append('generator-transfer')
        return original_to(*args, **kwargs)

    monkeypatch.setattr(attribution, 'ClassScoreModel', wrapped)
    monkeypatch.setattr(generator, 'to', transfer)
    assert model.training and generator.training
    with pytest.raises(ValueError, match='is not a class index from 0 to 1'):
        attribution._counterfactual_sequences(model, generator, crops, targets=targets)
    assert not calls
    assert model.training and generator.training
    assert not model.seen
    assert torch.equal(crops, original)
    if isinstance(targets, torch.Tensor):
        torch.testing.assert_close(targets, before_targets, rtol=0, atol=0, equal_nan=True)
    else:
        assert pickle.dumps(targets) == before_targets


@pytest.mark.parametrize('kind', ['list', 'numpy', 'tensor', 'float-list', 'float-tensor',
                                  'bool-list', 'bool-numpy', 'bool-tensor'])
@pytest.mark.parametrize('goal', [0, 1])
def test_valid_scalar_representations_match_exact_public_sequence_oracle(kind, goal):
    if kind.startswith('float'):
        targets = [float(goal)] if kind == 'float-list' else torch.tensor([float(goal)])
    elif kind.startswith('bool'):
        targets = ([bool(goal)] if kind == 'bool-list' else
                   np.asarray([bool(goal)]) if kind == 'bool-numpy' else torch.tensor([bool(goal)]))
    else:
        targets = _target(goal, kind)
    root = Path(__file__).parents[1] / 'features/data/564_sequence_target_identity_2026-10-15'
    expected_rows = json.loads((root / 'valid-oracle.json').read_text(encoding="utf-8"))[f'class{goal}']
    crops = _crops()
    original = crops.clone()
    rows, frames = attribution._counterfactual_sequences(
        _ReferenceScore(), _BothChannelEdit(), crops, targets=targets, keep=1)
    np.testing.assert_equal(rows, expected_rows)
    with np.load(root / 'valid-oracle.npz') as expected:
        np.testing.assert_array_equal(frames, expected[f'class{goal}'])
    assert torch.equal(crops, original)


def test_default_target_arithmetic_matches_exact_public_sequence_oracle():
    root = Path(__file__).parents[1] / 'features/data/564_sequence_target_identity_2026-10-15'
    rows, frames = attribution._counterfactual_sequences(
        _ReferenceScore(), _BothChannelEdit(), _crops(), targets=None, keep=1)
    np.testing.assert_equal(rows, json.loads((root / 'valid-oracle.json').read_text(encoding="utf-8"))['default'])
    with np.load(root / 'valid-oracle.npz') as expected:
        np.testing.assert_array_equal(frames, expected['default'])
