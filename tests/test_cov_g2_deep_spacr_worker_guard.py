"""Every classifier loader that asks for workers goes through the RAM guard."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_item60_deep_spacr_coverage import (
    _cv_settings, _patch_cv_dependencies)


@pytest.fixture
def guarded(monkeypatch):
    import spacr.resource_log as resource_log

    calls = []

    def guard(task, workers, unit, **kwargs):
        calls.append((task, workers))
        return 0

    monkeypatch.setattr(resource_log, "_guard_workers", guard)
    return calls


def test_nested_inner_loaders_guard_their_workers(tmp_path, monkeypatch, guarded):
    import spacr.deep_spacr as ds

    layout = [{"inner": [(np.array([0]), np.array([1])),
                         (np.array([1]), np.array([0]))]}]
    _patch_cv_dependencies(monkeypatch, nested_layout=layout,
                           train_results=[(None, None), (None, None)])
    assert ds._cross_validate_model(
        _cv_settings(tmp_path, nested_cv_inner_folds=2, n_jobs=2), 2) is None
    assert guarded and all(call == ("classify", 2) for call in guarded)
