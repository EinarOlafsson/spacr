import numpy as np
import pytest
import torch
import torch.nn as nn

from spacr import attribution


@pytest.mark.parametrize('target', [
    '0.5', '-0.5', 1.7, '1e-400', 'NaN', 'Inf', '-Infinity', '2',
])
def test_counterfactual_target_refuses_nonclass_numeric_values(target):
    with pytest.raises(ValueError, match='is not a class index from 0 to 1'):
        attribution._cf_target_index(target, [], 2)


@pytest.mark.parametrize('target,expected', [
    (0, 0), ('0', 0), (1.0, 1), ('1.0', 1), ('1e0', 1),
    (np.int64(1), 1), (np.float64(1.0), 1), (' 0 ', 0),
])
def test_existing_integral_target_representations_keep_their_class(target, expected):
    assert attribution._cf_target_index(target, [], 2) == expected


@pytest.mark.parametrize('name', ['0.5', 'NaN', '1e-400'])
def test_named_conditions_remain_exact_names_instead_of_numeric_classes(name):
    assert attribution._cf_target_index(name, ['control', name], 2) == 1


@pytest.mark.parametrize('target', ['0.5', '-0.5', '1e-400'])
def test_actual_report_refuses_fractional_target_before_training(monkeypatch, target):
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2)).eval()
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([0., 5.]))
    before_weights = [parameter.detach().clone() for parameter in model.parameters()]
    crops = torch.linspace(0., 1., 128).reshape(8, 1, 4, 4)
    before_crops = crops.clone()
    called = []

    def refuse_training(*args, **kwargs):
        called.append(True)
        raise AssertionError('Invalid class target reached the generator trainer')

    monkeypatch.setattr(attribution, '_train_counterfactual_generator', refuse_training)
    monkeypatch.setattr(attribution, '_train_counterfactual_diffusion', refuse_training)
    with pytest.raises(ValueError, match='is not a class index from 0 to 1'):
        attribution._counterfactual_report(model, crops, target=target, epochs=1)
    assert not called
    assert torch.equal(crops, before_crops)
    assert all(torch.equal(parameter, before)
               for parameter, before in zip(model.parameters(), before_weights))
