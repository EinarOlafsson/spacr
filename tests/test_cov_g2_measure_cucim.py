"""The cuCIM morphology path and its CUDA toolkit plumbing, without a GPU.

CuPy is stood in for by NumPy and cuCIM's ``regionprops_table`` by
scikit-image's, so the code that adapts one to the other runs here.
"""
from __future__ import annotations

import sys
import types
from importlib import metadata

import numpy as np
import pytest

from spacr import measure as M


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(M, "_CUCIM_STATE", {})


def _fake_cupy():
    return types.SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray)


def _mask():
    mask = np.zeros((20, 20), np.int32)
    mask[2:8, 2:8] = 3
    mask[10:18, 10:16] = 7
    return mask


def test_the_cupy_wheel_names_its_cuda_major(monkeypatch):
    def version(name):
        if name == "cupy-cuda12x":
            return "13.0"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", version)
    assert M._cupy_wheel_cuda_major() == 12


def test_the_runtime_headers_are_pinned_once(monkeypatch):
    finder = types.ModuleType("cuda.pathfinder")
    finder.find_nvidia_header_directory = lambda libname: f"/stock/{libname}"
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", finder)
    assert M._pin_cupy_cudart_headers("/cu12/include") is True
    pinned = finder.find_nvidia_header_directory
    assert pinned("cudart") == "/cu12/include"
    assert pinned("nvrtc") == "/stock/nvrtc"
    assert M._pin_cupy_cudart_headers("/cu12/include") is True
    assert finder.find_nvidia_header_directory is pinned
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", types.ModuleType("x"))
    assert M._pin_cupy_cudart_headers("/cu12/include") is False


def test_nvrtc_is_preloaded_only_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert M._preload_matching_nvrtc() is None


def _site(tmp_path, monkeypatch, *, headers=True):
    import site

    lib = tmp_path / "nvidia" / "cuda_nvrtc" / "lib"
    lib.mkdir(parents=True)
    (lib / "libnvrtc.so.12.alt.0").write_text("")
    (lib / "libnvrtc.so.12.0").write_text("")
    (lib / "libnvrtc.so.12.1").write_text("")
    include = tmp_path / "nvidia" / "cuda_runtime" / "include"
    include.mkdir(parents=True)
    if headers:
        (include / "cuda_fp16.h").write_text("")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(M, "_cupy_wheel_cuda_major", lambda: 12)
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(site, "getusersitepackages",
                        lambda: str(tmp_path / "user"))
    return lib, include


def test_the_matching_nvrtc_is_loaded_and_its_headers_pinned(
        tmp_path, monkeypatch):
    import ctypes

    lib, include = _site(tmp_path, monkeypatch)
    loaded, pinned = [], []

    def cdll(path, mode=0):
        if path.endswith(".12.0"):
            raise OSError("wrong architecture")
        loaded.append(path)

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(M, "_pin_cupy_cudart_headers", pinned.append)
    assert M._preload_matching_nvrtc() == str(lib / "libnvrtc.so.12.1")
    assert pinned == [str(include)]


def test_nvrtc_without_cupy_is_still_reported(tmp_path, monkeypatch):
    import builtins
    import ctypes

    lib, _include = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: None)
    real_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("no cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    assert M._preload_matching_nvrtc() == str(lib / "libnvrtc.so.12.0")


def test_no_loadable_nvrtc_returns_none(tmp_path, monkeypatch):
    import ctypes

    _site(tmp_path, monkeypatch, headers=False)

    def refuse(path, mode=0):
        raise OSError("no")

    monkeypatch.setattr(ctypes, "CDLL", refuse)
    assert M._preload_matching_nvrtc() is None


def _install_cucim(monkeypatch, table):
    cucim = types.ModuleType("cucim")
    skimage = types.ModuleType("cucim.skimage")
    measure = types.ModuleType("cucim.skimage.measure")
    measure.regionprops_table = table
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setitem(sys.modules, "cucim", cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.measure", measure)
    monkeypatch.setattr(M, "_preload_matching_nvrtc", lambda: None)


def test_the_gpu_table_matches_the_cpu_table(monkeypatch):
    from skimage.measure import regionprops_table

    _install_cucim(monkeypatch, regionprops_table)
    props = ["area", "area_filled", "eccentricity"]
    frame = M._cucim_morphology_table(_mask(), props)
    cpu = regionprops_table(_mask(), properties=props)
    assert list(frame.columns) == props
    np.testing.assert_allclose(frame["area"], cpu["area"])
    np.testing.assert_allclose(frame["area_filled"], cpu["area_filled"])


def test_the_label_column_is_returned_only_when_asked(monkeypatch):
    from skimage.measure import regionprops_table

    _install_cucim(monkeypatch, regionprops_table)
    frame = M._cucim_morphology_table(_mask(), ["label", "area"])
    assert frame["label"].tolist() == [3, 7]


def test_a_gpu_table_that_disagrees_on_labels_is_not_used(monkeypatch):
    def shifted(mask, properties):
        return {"label": np.array([2, 1]), "area": np.array([1.0, 2.0])}

    _install_cucim(monkeypatch, shifted)
    assert M._cucim_morphology_table(_mask(), ["area", "area_filled"]) is None


def test_a_gpu_table_missing_a_column_is_not_used(monkeypatch):
    def partial(mask, properties):
        return {"label": np.array([1, 2])}

    _install_cucim(monkeypatch, partial)
    assert M._cucim_morphology_table(_mask(), ["area"]) is None
