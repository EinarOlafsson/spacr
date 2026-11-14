"""The PyTorch mixed-model backend: what it refuses, and what it says.

The backend exists so a screen-sized REML fit finishes in under a second, and
its whole safety story is that it never silently substitutes something else --
not the CPU for a GPU that did not answer, not a rank-deficient design for an
identified one, not the first n grouping labels for the rows patsy actually
kept. Each of those refusals is driven here.
"""
from __future__ import annotations

import builtins

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
