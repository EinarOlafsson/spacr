"""Every classifier loader that asks for workers goes through the RAM guard."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_cov_deep_spacr_apply_model import (  # noqa: F401
    _force_cpu, _save_model, _tar_settings, const_png_dir, const_png_tar)
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


def test_scoring_a_folder_guards_its_workers(const_png_dir, tmp_path, guarded):  # noqa: F811
    from spacr.deep_spacr import apply_model

    src, values = const_png_dir
    frame = apply_model(src, _save_model(tmp_path / "m.pth"), image_size=32,
                        batch_size=2, normalize=False, n_jobs=3)
    assert len(frame) == len(values)
    assert guarded == [("classify", 3)]


def test_scoring_a_tar_guards_its_workers(const_png_tar, tmp_path, guarded):  # noqa: F811
    from spacr.deep_spacr import apply_model_to_tar

    tar_path = const_png_tar[0] if isinstance(const_png_tar, tuple) else const_png_tar
    frame = apply_model_to_tar(_tar_settings(
        tar_path, _save_model(tmp_path / "m.pth"), n_jobs=2))
    assert len(frame)
    assert guarded == [("classify", 2)]


def test_nested_inner_loaders_guard_their_workers(tmp_path, monkeypatch, guarded):
    import spacr.deep_spacr as ds

    layout = [{"inner": [(np.array([0]), np.array([1])),
                         (np.array([1]), np.array([0]))]}]
    _patch_cv_dependencies(monkeypatch, nested_layout=layout,
                           train_results=[(None, None), (None, None)])
    assert ds._cross_validate_model(
        _cv_settings(tmp_path, nested_cv_inner_folds=2, n_jobs=2), 2) is None
    assert guarded and all(call == ("classify", 2) for call in guarded)
