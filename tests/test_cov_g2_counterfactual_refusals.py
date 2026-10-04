"""Counterfactual reports refuse held-out sets they cannot morph."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spacr.attribution import _counterfactual_report


def _always_class_zero():
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([5.0, 0.0]))
    return model


def _split(n, seed=0):
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_test = min(n - 2, max(2, int(round(n * 0.25))))
    return order[:n_test].tolist(), order[n_test:].tolist()


def test_held_out_crops_from_unseen_conditions_are_refused():
    crops = torch.rand(8, 1, 4, 4)
    held_out, training = _split(8)
    conditions = [""] * 8
    for k, i in enumerate(training):
        conditions[i] = "ab"[k % 2]
    for i in held_out:
        conditions[i] = "unseen"
    with pytest.raises(ValueError, match="no held-out crop belongs"):
        _counterfactual_report(_always_class_zero(), crops, epochs=1,
                               conditions=conditions)


def test_a_target_every_held_out_crop_already_has_is_refused():
    crops = torch.rand(8, 1, 4, 4)
    with pytest.raises(ValueError, match="already in the target"):
        _counterfactual_report(_always_class_zero(), crops, epochs=1, target=0)
