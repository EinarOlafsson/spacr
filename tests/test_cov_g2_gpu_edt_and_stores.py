"""GPU distance transforms with a stand-in CuPy, and measurement-store retries."""
from __future__ import annotations

import sqlite3
import sys
import types

import numpy as np
import pytest
from scipy import ndimage as ndi

from spacr import object_distances as od


def _fake_cupy(monkeypatch, *, fail=False):
    cupy = types.ModuleType("cupy")
    cupy.asarray = np.asarray
    cupy.asnumpy = np.asarray
    cupy.float32 = np.float32
    cupyx = types.ModuleType("cupyx")
    scipy_mod = types.ModuleType("cupyx.scipy")
    ndimage = types.ModuleType("cupyx.scipy.ndimage")

    def edt(binary):
        if fail:
            raise RuntimeError("NVRTC compile error")
        return ndi.distance_transform_edt(binary)

    ndimage.distance_transform_edt = edt
    for name, module in (("cupy", cupy), ("cupyx", cupyx),
                         ("cupyx.scipy", scipy_mod),
                         ("cupyx.scipy.ndimage", ndimage)):
        monkeypatch.setitem(sys.modules, name, module)
    import spacr.measure as measure
    monkeypatch.setattr(measure, "_preload_matching_nvrtc", lambda: None)


def test_the_gpu_transform_matches_scipy_inside_its_block(monkeypatch):
    _fake_cupy(monkeypatch)
    binary = np.zeros((6, 6), bool)
    binary[1:5, 1:5] = True
    with od._gpu_distance_transforms():
        gpu = od._edt(binary, ndi.distance_transform_edt, {})
    assert od._EDT_ON_GPU["on"] is False
    np.testing.assert_allclose(gpu, ndi.distance_transform_edt(binary))


def test_a_failing_gpu_transform_falls_back_and_stays_off(monkeypatch):
    _fake_cupy(monkeypatch, fail=True)
    binary = np.ones((4, 4), bool)
    with od._gpu_distance_transforms():
        out = od._edt(binary, ndi.distance_transform_edt, {})
        assert od._EDT_ON_GPU["on"] is False
    assert out.dtype == np.float32


def test_a_locked_database_is_retried_then_answers(monkeypatch, tmp_path):
    from spacr import utils

    calls = []

    def flaky(db_path, table):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return {"ok": True}

    monkeypatch.setattr(utils, "_existing_measurement_identity_once", flaky)
    monkeypatch.setattr(utils.time, "sleep", lambda s: None)
    assert utils._existing_measurement_identity(str(tmp_path / "m.db"),
                                                "cell") == {"ok": True}

    def broken(db_path, table):
        raise sqlite3.OperationalError("no such table")

    monkeypatch.setattr(utils, "_existing_measurement_identity_once", broken)
    with pytest.raises(sqlite3.OperationalError):
        utils._existing_measurement_identity(str(tmp_path / "m.db"), "cell")


def test_an_optional_store_write_failure_is_reported(monkeypatch, capsys):
    import pandas as pd

    from spacr import utils

    def refuse(*a, **k):
        raise OSError("store offline")

    monkeypatch.setattr(utils, "write_database", refuse, raising=False)
    import spacr.tabular as tabular
    monkeypatch.setattr(tabular, "write_database", refuse)
    frame = pd.DataFrame({"a": [1]})
    utils._append_to_measurement_store("/store.duckdb", "cell", frame, False)
    assert "not written" in capsys.readouterr().out
    with pytest.raises(OSError):
        utils._append_to_measurement_store("/store.duckdb", "cell", frame, True)


def test_augmenting_no_images_does_nothing(tmp_path):
    from spacr.utils import augment_images

    assert augment_images([], str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []
