"""The PyTorch mixed-model backend: what it refuses, and what it says.

The backend exists so a screen-sized REML fit finishes in under a second, and
its whole safety story is that it never silently substitutes something else --
not the CPU for a GPU that did not answer, not a rank-deficient design for an
identified one, not the first n grouping labels for the rows patsy actually
kept. Each of those refusals is driven here.
"""
from __future__ import annotations

import builtins
import os

import numpy as np
import pandas as pd
import pytest

from spacr import mixed_gpu as MG


def _block_torch(monkeypatch):
    """Hide PyTorch the way an environment without it does.

    ``torch_available`` asks ``find_spec`` rather than importing, so both the
    probe and the import have to fail together or the module is being asked
    about a state no installation is ever in.
    """
    import importlib.util

    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    def hidden(name, package=None):
        if name == "torch":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.setattr(importlib.util, "find_spec", hidden)


@pytest.fixture
def clustered():
    """A small two-level dataset: six wells, one predictor, one outcome."""
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


# ---------------------------------------------------------------------------
# what hardware answered
# ---------------------------------------------------------------------------

def test_free_memory_on_a_device_that_cannot_be_asked_reads_as_zero(
        monkeypatch):
    """Zero means "no budget can be proved", which blocks the large-fit check."""
    _block_torch(monkeypatch)
    assert MG.available_memory("cuda:0") == 0

    def no_sysconf(name):
        raise OSError("sysconf is not available here")

    monkeypatch.setattr(os, "sysconf", no_sysconf)
    assert MG.available_memory("cpu") == 0


def test_free_cpu_memory_is_pages_times_page_size():
    """The CPU figure is read off the machine, not estimated."""
    free = MG.available_memory("cpu")
    assert free > 0
    assert free == int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def test_free_cuda_memory_comes_from_the_driver(monkeypatch):
    """The CUDA figure is whatever ``mem_get_info`` reported, in bytes."""
    import torch

    class FakeCuda:
        @staticmethod
        def mem_get_info():
            return (3 * 1024 ** 3, 24 * 1024 ** 3)

    monkeypatch.setattr(torch, "cuda", FakeCuda(), raising=False)
    assert MG.available_memory("cuda") == 3 * 1024 ** 3


def test_a_driver_that_will_not_answer_reads_as_no_free_memory(monkeypatch):
    """A raising driver has not reported a budget."""
    import torch

    class BrokenCuda:
        @staticmethod
        def mem_get_info():
            raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(torch, "cuda", BrokenCuda(), raising=False)
    assert MG.available_memory("cuda") == 0


def test_no_torch_means_no_cuda_and_a_message_naming_the_install(monkeypatch):
    """The description is what a tooltip shows, so it names the fix."""
    _block_torch(monkeypatch)
    assert MG.torch_available() is False
    assert MG.cuda_available() is False
    assert MG.describe_device() == "torch is not installed (pip install torch)"


def test_a_cuda_probe_that_raises_is_read_as_no_cuda(monkeypatch):
    """A driver that errors when asked has not answered."""
    import torch

    class BrokenCuda:
        @staticmethod
        def is_available():
            raise RuntimeError("no NVIDIA driver on this system")

    monkeypatch.setattr(torch, "cuda", BrokenCuda(), raising=False)
    assert MG.cuda_available() is False


def test_torch_without_a_device_says_which_torch_is_installed(monkeypatch):
    """Naming the version is what tells an install from a driver problem."""
    import torch

    monkeypatch.setattr(MG, "cuda_available", lambda: False)
    text = MG.describe_device()
    assert torch.__version__ in text
    assert "no CUDA device answered" in text


def test_a_device_that_answered_is_described_by_name_and_size(monkeypatch):
    """The tooltip names the card and its memory, both read from torch."""
    import torch

    class Properties:
        total_memory = 24 * 1024 ** 3

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(index):
            return "GeForce RTX 3090"

        @staticmethod
        def get_device_properties(index):
            return Properties()

    monkeypatch.setattr(torch, "cuda", FakeCuda(), raising=False)
    text = MG.describe_device()
    assert "GeForce RTX 3090" in text
    assert "(24.0 GB)" in text


def test_asking_for_a_gpu_that_is_not_there_is_refused_not_downgraded(
        monkeypatch):
    """A GPU fit that quietly ran on the CPU is the slow run, reported as fast."""
    monkeypatch.setattr(MG, "cuda_available", lambda: False)
    with pytest.raises(MG.MixedBackendUnavailable) as excinfo:
        MG.resolve_device("cuda")
    assert "does not fall back to the CPU" in str(excinfo.value)
    assert str(MG.resolve_device("cpu")) == "cpu"


def test_without_torch_no_device_resolves_at_all(monkeypatch):
    """The message names both ways out: install torch, or use statsmodels."""
    _block_torch(monkeypatch)
    with pytest.raises(MG.MixedBackendUnavailable) as excinfo:
        MG.resolve_device("cpu")
    assert "pip install" in str(excinfo.value)
    assert "regression_backend='statsmodels'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------

def test_a_grouping_column_with_a_missing_value_is_refused():
    """A row with no cluster has no level, and dropping it silently would
    change the count of wells the fit reports."""
    with pytest.raises(ValueError, match="contains missing values"):
        MG._codes(["w1", None, "w2"])


def test_a_variance_component_with_a_missing_value_is_refused():
    """Named by component, because a screen defines several."""
    codes, levels = MG._codes(["w1", "w1", "w2"])
    with pytest.raises(ValueError, match="'guide' has missing values"):
        MG._nested_codes(codes, levels, ["g1", np.nan, "g2"], "guide")


def test_a_nested_component_labels_each_level_inside_its_own_group():
    """The same guide id in two wells is two levels, not one."""
    codes, levels = MG._codes(["w1", "w1", "w2", "w2"])
    inner_codes, names, owners = MG._nested_codes(
        codes, levels, ["g1", "g2", "g1", "g2"], "guide")
    assert len(set(inner_codes.tolist())) == 4
    assert names == ["guide[C(guide)[g1]]", "guide[C(guide)[g2]]",
                     "guide[C(guide)[g1]]", "guide[C(guide)[g2]]"]
    assert owners == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# the fit's refusals
# ---------------------------------------------------------------------------

def test_a_response_and_a_design_of_different_lengths_are_refused(clustered):
    """Each row of the design must carry its own response."""
    with pytest.raises(ValueError, match="must carry its own response"):
        MG.fit_mixed_reml_torch(clustered["y"][:-1], clustered["X"],
                                clustered["groups"], device="cpu")


def test_groups_shorter_than_the_design_are_refused(clustered):
    """Each row must carry its own cluster id; a short list would shift them."""
    with pytest.raises(ValueError, match="must carry its own cluster id"):
        MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                clustered["groups"][:-1], device="cpu")


def test_a_bare_array_design_gets_generated_column_names(clustered):
    """Without a frame there are no names, so ``x0``, ``x1`` are used."""
    fit = MG.fit_mixed_reml_torch(
        clustered["y"], clustered["X"].to_numpy(), clustered["groups"],
        device="cpu")
    assert list(fit.fe_params.index) == ["x0", "x1"]

    single = MG.fit_mixed_reml_torch(
        clustered["y"], np.ones(len(clustered["y"])), clustered["groups"],
        device="cpu")
    assert list(single.fe_params.index) == ["x0"]


def test_a_fit_with_no_residual_degrees_of_freedom_is_refused():
    """REML has no residual variance to estimate, so it says so."""
    y = np.array([1.0, 2.0, 3.0])
    X = pd.DataFrame(np.eye(3), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="residual degrees of freedom"):
        MG.fit_mixed_reml_torch(y, X, ["w1", "w1", "w2"], device="cpu")


def test_a_rank_deficient_fixed_part_is_named_rather_than_left_to_linalg(
        clustered):
    """MixedLM reports this three frames deep as a bare LinAlgError."""
    design = clustered["X"].copy()
    design["copy_of_predictor"] = design["predictor"] * 2.0
    with pytest.raises(ValueError, match="coefficients are not identified"):
        MG.fit_mixed_reml_torch(clustered["y"], design, clustered["groups"],
                                device="cpu")


def test_a_verbose_fit_prints_each_deviance_it_evaluated(clustered, capsys):
    """The trace is what a stuck optimiser is diagnosed from."""
    MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                            clustered["groups"], device="cpu", verbose=True)
    printed = capsys.readouterr().out
    assert "deviance" in printed and "theta" in printed


def test_a_finished_fit_reports_what_it_cost_in_one_line(clustered):
    """The run log gets the device, the time and whether it converged."""
    fit = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                  clustered["groups"], device="cpu")
    assert fit.df_resid == fit.n_obs - fit.k_fe == len(clustered["y"]) - 2
    line = fit.summary_line()
    assert line.startswith("mixed fit: backend=torch device=cpu")
    assert "deviance evaluations" in line
    assert f"converged={fit.converged}" in line


# ---------------------------------------------------------------------------
# the formula front door
# ---------------------------------------------------------------------------

def test_a_grouping_column_may_be_named_rather_than_passed(clustered):
    """``groups='well'`` is how every caller in spaCR spells it."""
    frame = clustered["X"].copy()
    frame["response"] = clustered["y"]
    frame["well"] = clustered["groups"]
    fit = MG.mixedlm_torch("response ~ predictor", frame, "well", device="cpu")
    assert fit.n_obs == len(frame)
    assert "predictor" in " ".join(str(i) for i in fit.fe_params.index)


def test_a_variance_component_naming_an_absent_column_is_refused(clustered):
    """A component built from a column that is not there cannot be built."""
    frame = clustered["X"].copy()
    frame["response"] = clustered["y"]
    frame["well"] = clustered["groups"]
    with pytest.raises(ValueError, match="which this frame does not have"):
        MG.mixedlm_torch("response ~ predictor", frame, "well",
                         vc_formula={"guide": "0 + C(guide_id)"},
                         device="cpu")


# ---------------------------------------------------------------------------
# the two backends, side by side
# ---------------------------------------------------------------------------

def test_the_two_backends_agree_on_a_simple_fit(clustered):
    """The comparison exists so a disagreement is a number, not an opinion."""
    report = MG.benchmark_against_statsmodels(
        clustered["y"], clustered["X"], clustered["groups"], device="cpu")
    assert report["device"] == "cpu"
    assert report["statsmodels_seconds"] > 0
    assert report["torch_seconds"] > 0
    assert report["max_abs_coefficient_difference"] < 1e-3
    assert report["scale_relative_difference"] < 1e-2
    assert report["speedup"] > 0


def test_the_two_backends_agree_with_a_variance_component(clustered):
    """The nested path builds a formula fit, which is the one screens use."""
    guides = np.tile([f"g{i}" for i in range(4)], len(clustered["y"]) // 4)
    report = MG.benchmark_against_statsmodels(
        clustered["y"], clustered["X"], clustered["groups"],
        vc={"guide": guides}, device="cpu")
    assert report["max_abs_coefficient_difference"] < 1e-2
    assert report["max_relative_variance_difference"] >= 0
    assert report["n_deviance_evals"] > 0
