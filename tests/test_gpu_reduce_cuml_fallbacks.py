"""A cuML that imports but cannot work costs a CPU run, never a crash.

RAPIDS is an optional extra and is not installed here, so the GPU half of
:mod:`spacr.gpu_reduce` is driven by putting a fake ``cuml`` (and, where it
matters, a broken ``cupy``) in ``sys.modules``.
"""
from __future__ import annotations

import sys
import types

from spacr import gpu_reduce


class _Estimator:
    """Stands in for one of cuML's estimator classes."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


def _fake_cuml(broken=()):
    """A ``cuml`` module whose estimators record what they were asked for."""
    module = types.ModuleType("cuml")
    module.__version__ = "24.10"
    for attr in ("UMAP", "TSNE", "PCA", "DBSCAN", "KMeans"):
        def _make(attr=attr):
            def _build(**kwargs):
                if attr in broken:
                    raise RuntimeError(f"cuML {attr} needs a newer CUDA")
                return _Estimator(attr, **kwargs)
            return _build
        setattr(module, attr, _make())
    return module


def _cupy(devices):
    """A ``cupy`` reporting ``devices`` CUDA devices."""
    module = types.ModuleType("cupy")
    module.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(
        getDeviceCount=lambda: devices))
    return module


def test_make_reducer_uses_cuml_only_when_it_can_build(monkeypatch, caplog, subtests):
    """A working cuML builds the cuML estimator; one that raises falls back to the CPU class and says so."""
    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
    cases = [
        ("cuml_builds_its_estimator", (), "_Estimator", "cuml", ""),
        ("cuml_that_cannot_build_falls_back", ("UMAP",), "UMAP", "cpu",
         "cuML could not build a umap estimator; using the CPU implementation"),
    ]
    for case_id, broken, cls, backend, logged in cases:
        with subtests.test(msg=case_id):
            monkeypatch.setitem(sys.modules, "cuml", _fake_cuml(broken))
            caplog.clear()
            with caplog.at_level("INFO", logger="spacr.gpu_reduce"):
                estimator, got = gpu_reduce.make_reducer(
                    "umap", prefer_gpu=True, n_neighbors=15)
            assert (type(estimator).__name__, got) == (cls, backend)
            passed = (estimator.kwargs if backend == "cuml"
                      else {"n_neighbors": estimator.n_neighbors})
            assert passed == {"n_neighbors": 15}
            assert [r.getMessage() for r in caplog.records
                    if r.name == "spacr.gpu_reduce"] == ([logged] if logged else [])


def test_describe_names_what_is_missing(monkeypatch, subtests):
    """describe gives one exact line for disabled, device-less and working installs."""
    cases = [
        ("disabled_by_the_flag", "0", None, f"RAPIDS disabled by {gpu_reduce.ENV_FLAG}"),
        ("cupy_unimportable_is_no_device", None, None, "cuML 24.10 installed, no CUDA device"),
        ("cupy_with_zero_devices", None, _cupy(0), "cuML 24.10 installed, no CUDA device"),
        ("two_devices", None, _cupy(2), "cuML 24.10 on 2 device(s)"),
    ]
    for case_id, flag, cupy, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as patch:
                if flag is None:
                    patch.delenv(gpu_reduce.ENV_FLAG, raising=False)
                else:
                    patch.setenv(gpu_reduce.ENV_FLAG, flag)
                patch.setitem(sys.modules, "cuml", _fake_cuml())
                patch.setitem(sys.modules, "cupy", cupy)
                assert gpu_reduce.describe() == expected
