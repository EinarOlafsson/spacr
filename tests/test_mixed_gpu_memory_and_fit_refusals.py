"""Table-driven tests of the PyTorch mixed-model backend's memory probe and fit front door.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: :func:`spacr.mixed_gpu.available_memory` (every
source of a free-memory figure, and the zero that means "no budget can be
proved"), and :func:`spacr.mixed_gpu.fit_mixed_reml_torch` (the exact
refusals for designs it will not fit, and what a finished CPU fit reports).
Drivers, ``os.sysconf`` and ``psutil`` are replaced, so the CPU-only suite
never needs a GPU.
"""
from __future__ import annotations

import builtins
import importlib.util
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spacr import mixed_gpu as MG

GB = 1024 ** 3


def _block_torch(mp):
    """Hide PyTorch from both ``import`` and ``find_spec``, as an install without it does."""
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    def hidden(name, package=None):
        return None if name == "torch" else real_find_spec(name, package)

    mp.setattr(builtins, "__import__", no_torch)
    mp.setattr(importlib.util, "find_spec", hidden)


def _raise(exc):
    """A callable that raises ``exc`` whatever it is given."""
    def _call(*_args, **_kwargs):
        raise exc
    return _call


def _sysconf(pages, page_size):
    """An ``os.sysconf`` double answering the two names the probe asks for."""
    return {"SC_AVPHYS_PAGES": pages, "SC_PAGE_SIZE": page_size}.__getitem__


@pytest.fixture
def clustered():
    """A small two-level dataset: six wells of eight cells, one predictor."""
    rng = np.random.default_rng(0)
    wells = np.repeat([f"w{i}" for i in range(6)], 8)
    offsets = {f"w{i}": rng.normal(0, 0.5) for i in range(6)}
    predictor = rng.normal(size=wells.size)
    response = (2.0 + 1.5 * predictor
                + np.array([offsets[w] for w in wells])
                + rng.normal(0, 0.3, size=wells.size))
    design = pd.DataFrame({"intercept": np.ones(wells.size),
                           "predictor": predictor})
    return {"y": response, "X": design, "groups": wells}


def test_available_memory_reads_each_source_and_zero_when_none_answers(subtests, monkeypatch):
    """CUDA asks the driver; the CPU asks sysconf, then psutil; silence reads as 0."""
    import psutil
    import torch

    working = SimpleNamespace(mem_get_info=lambda: (3 * GB, 24 * GB))
    broken = SimpleNamespace(mem_get_info=_raise(
        RuntimeError("CUDA driver version is insufficient")))
    cases = [
        # (case id, device, torch.cuda double or "blocked", sysconf, psutil.virtual_memory, bytes)
        ("cuda free bytes come from the driver", "cuda", working, None, None, 3 * GB),
        ("an indexed cuda device asks the same driver", "cuda:1", working, None, None,
         3 * GB),
        ("a driver that raises has no budget", "cuda", broken, None, None, 0),
        ("no torch means no cuda budget", "cuda:0", "blocked", None, None, 0),
        ("cpu memory is free pages times page size", "cpu", None,
         _sysconf(1000, 4096), _raise(AssertionError("psutil must not be asked")),
         1000 * 4096),
        ("without SC_AVPHYS_PAGES psutil answers", "cpu", None,
         _raise(ValueError("unrecognized configuration name")),
         lambda: SimpleNamespace(available=7 * GB), 7 * GB),
        ("neither sysconf nor psutil answers", "cpu", None,
         _raise(OSError("no sysconf")), _raise(OSError("psutil cannot read memory")), 0),
        ("no torch does not matter to the cpu", "cpu", "blocked",
         _sysconf(5, 8192), None, 5 * 8192),
    ]
    for case_id, device, cuda, sysconf, vm, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            if cuda == "blocked":
                _block_torch(mp)
            elif cuda is not None:
                mp.setattr(torch, "cuda", cuda, raising=False)
            if sysconf is not None:
                mp.setattr(os, "sysconf", sysconf, raising=False)
            if vm is not None:
                mp.setattr(psutil, "virtual_memory", vm)
            got = MG.available_memory(device)
            assert type(got) is int
            assert got == expected

    with subtests.test(msg="the real cpu figure is the machine's own"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            pages = None
        free = MG.available_memory("cpu")
        if pages is not None:
            # Free pages drift between two reads; a few MB is the same answer.
            assert abs(free - pages) < 256 * 1024 ** 2
        else:
            assert 0 < free <= psutil.virtual_memory().total


def test_fit_mixed_reml_torch_refuses_designs_it_cannot_identify(subtests, clustered):
    """Each refused design raises ValueError with its exact, row-counting message."""
    y, X, groups = clustered["y"], clustered["X"], clustered["groups"]
    aliased = X.copy()
    aliased["copy_of_predictor"] = aliased["predictor"] * 2.0
    cases = [
        # (case id, y, X, groups, exact message prefix)
        ("response one row short", y[:-1], X, groups,
         "the response has 47 rows and the design has 48; each row of the "
         "design must carry its own response."),
        ("groups one row short", y, X, groups[:-1],
         "groups has 47 entries but the design has 48 rows; each row must "
         "carry its own cluster id."),
        ("no residual degrees of freedom", np.array([1.0, 2.0, 3.0]),
         pd.DataFrame(np.eye(3), columns=["a", "b", "c"]), ["w1", "w1", "w2"],
         "the fit has 3 rows and 3 fixed effects, so REML has 0 residual "
         "degrees of freedom and no residual variance to estimate. Reduce the "
         "fixed part, or fit more wells."),
        ("rank-deficient fixed part", y, aliased, groups,
         "regression_backend='torch': the fixed-effects design is rank 2 with "
         "3 columns, so its coefficients are not identified -- some terms are "
         "exact linear combinations of others"),
    ]
    for case_id, yy, XX, gg, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as caught:
                MG.fit_mixed_reml_torch(yy, XX, gg, device="cpu")
            assert str(caught.value).startswith(message)


def test_fit_mixed_reml_torch_names_columns_and_reports_its_cost(subtests, clustered, capsys):
    """A finished CPU fit names its coefficients, counts its df and prints one summary line."""
    y, X, groups = clustered["y"], clustered["X"], clustered["groups"]
    cases = [
        # (case id, design, coefficient names)
        ("a frame keeps its column names", X, ["intercept", "predictor"]),
        ("a bare 2-D array gets generated names", X.to_numpy(), ["x0", "x1"]),
        ("a bare 1-D array is one column", np.ones(len(y)), ["x0"]),
    ]
    for case_id, design, names in cases:
        with subtests.test(msg=case_id):
            fit = MG.fit_mixed_reml_torch(y, design, groups, device="cpu")
            assert list(fit.fe_params.index) == names
            assert (fit.n_obs, fit.k_fe) == (48, len(names))
            assert fit.df_resid == 48 - len(names)
            assert fit.summary_line() == (
                f"mixed fit: backend=torch device=cpu {fit.fit_seconds:.2f}s, "
                f"{fit.n_deviance_evals} deviance evaluations, "
                f"scale={fit.scale:.4g}, converged={fit.converged}")
            assert capsys.readouterr().out == ""

    with subtests.test(msg="verbose prints a trace line for each closure evaluation"):
        capsys.readouterr()
        fit = MG.fit_mixed_reml_torch(y, X, groups, device="cpu", verbose=True)
        lines = capsys.readouterr().out.splitlines()
        pattern = re.compile(r"^  deviance -?\d+\.\d{6} theta \[.+\]$")
        assert lines and all(pattern.match(line) for line in lines), lines
        assert 0 < len(lines) <= fit.n_deviance_evals
