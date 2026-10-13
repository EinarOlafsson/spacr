"""A shared GPU running out of memory is a slower fit, not a failed run.

Reported 2026-08-21 from a real run:

    CUDACachingAllocator: memory allocation failed with OOM on device 0
    while trying to allocate 2587885568 bytes (free: 2000093184,
    total: 25295519744)

A 25 GB card with 2 GB free, because something else on the machine had the
rest.

`mixed_gpu._refuse_if_too_large` already checks free memory before building
the design, and IT CANNOT BE ENOUGH ON A SHARED DEVICE: it covers one
allocation, the optimiser makes others, and the free figure it read can be
stale by the time any of them run. A co-tenant that allocates between the
check and the fit turns a correct check into a wrong one.
"""
from __future__ import annotations

import pytest


class TestRecognisingIt:
    """Matched by NAME as well as type, because `torch.cuda.OutOfMemoryError`
    only exists once torch is imported, and deciding whether to import torch
    must not require importing torch."""

    def test_a_torch_style_error_is_recognised(self):
        from spacr.ml import _is_out_of_memory

        class OutOfMemoryError(Exception):
            pass

        assert _is_out_of_memory(OutOfMemoryError("CUDA OOM"))

    def test_a_plain_memory_error_counts(self):
        """`mixed_gpu` raises one deliberately when the design will not
        fit."""
        from spacr.ml import _is_out_of_memory

        assert _is_out_of_memory(MemoryError("needs a dense design"))

    def test_the_message_is_matched_when_the_type_is_not(self):
        from spacr.ml import _is_out_of_memory

        assert _is_out_of_memory(RuntimeError("CUDA error: out of memory"))

    @pytest.mark.parametrize("exc", [
        ValueError("design is rank deficient"),
        KeyError("grna"),
        ZeroDivisionError(),
    ])
    def test_an_unrelated_failure_is_not_swallowed(self, exc):
        """A fit that failed for a real reason must still raise. Falling
        back on every exception would turn a bug into a silently different
        model."""
        from spacr.ml import _is_out_of_memory

        assert not _is_out_of_memory(exc)


class TestTheFallbackIsWiredIn:

    def test_the_call_site_catches_and_falls_back(self):
        import inspect

        from spacr import ml

        # `perform_mixed_model` is where the backend is chosen and the
        # torch fit is called -- `fit_mixed_model` is the caller above it.
        body = inspect.getsource(ml.perform_mixed_model)
        assert "_is_out_of_memory" in body
        # It must RAISE anything else rather than fall back blindly.
        assert "raise" in body
        # And the fallback is the CPU model, not a different one.
        assert "MixedLM(y, X, groups=groups)" in body

    def test_it_says_what_happened_and_what_to_do(self):
        import inspect

        from spacr import ml

        body = inspect.getsource(ml.perform_mixed_model)
        assert "shared" in body
        assert "statsmodels (CPU)" in body


class TestTheFallbackActuallyRuns:
    """The class above proves the words are in the source. This one proves a
    fit comes back.

    IT HAS TO RUN WHERE THERE IS NO CARD, and that is the whole difficulty of
    testing this. `perform_mixed_model` asks `_require_backend` first, and
    that refuses `regression_backend='torch'` outright when no CUDA device
    answers -- deliberately, because a fit you asked to run on the GPU and
    that quietly ran on the CPU is the slow run you were avoiding, reported
    as the fast one. So a test that stubbed only the torch fit would reach
    the fallback on a machine with a card and never reach it on a hosted
    runner, which has none.

    Measured both ways on 2026-09-12 on the same working tree: with an RTX
    3090 visible, that call returns a statsmodels result; with
    `CUDA_VISIBLE_DEVICES=''` the identical call raises `ValueError: torch
    (GPU) needs a CUDA device and none was found`. A test written the naive
    way is green here and red on CI, and the tempting repair -- skipping it
    where there is no GPU, or where there is one -- would retire the guard on
    whichever machine was asked last.

    What is stubbed here is therefore the AVAILABILITY VERDICT and nothing
    else. The test says "a card answered", and then the real
    `perform_mixed_model`, the real `_is_out_of_memory` and the real
    statsmodels fit run underneath it. That path does not touch a device, so
    it is the same code on both machines -- which is why this carries no
    `gpu` marker and is skipped nowhere.
    """

    @staticmethod
    def _a_design_a_mixed_model_can_fit():
        """Eight groups of twelve rows with a random intercept worth finding.

        Fixed seed: a fallback that returned different numbers on different
        days would be the bug this file exists to catch, so the comparison
        below has to be against one fit and not a distribution of them.
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(8), 12)
        n = groups.size
        first = rng.normal(size=n)
        second = rng.normal(size=n)
        design = pd.DataFrame({"Intercept": np.ones(n),
                               "a": first, "b": second})
        response = (1.0 + 0.7 * first - 0.4 * second
                    + rng.normal(size=8)[groups]
                    + rng.normal(scale=0.5, size=n))
        return response, design, groups

    @pytest.fixture
    def a_card_that_answered(self, monkeypatch):
        """Let the torch branch be entered on a machine with no GPU.

        `_require_backend` reads `backend_status`, which this replaces, and
        nothing else about the branch is faked.
        """
        from spacr import ml

        monkeypatch.setattr(
            ml, "backend_status",
            lambda name, regression_type=None: {"enabled": True,
                                                "reason": ""})

    def test_the_cpu_model_comes_back_with_the_same_numbers(
            self, a_card_that_answered, monkeypatch):
        """The message promises the same model, the same numbers and only
        more time, so this holds it to the numbers and not to the type."""
        import numpy as np
        from statsmodels.regression.mixed_linear_model import MixedLM

        from spacr import ml, mixed_gpu

        class OutOfMemoryError(RuntimeError):
            """Shaped like torch's, which also subclasses RuntimeError."""

        def the_card_filled_up(*args, **kwargs):
            raise OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2.41 GiB")

        monkeypatch.setattr(mixed_gpu, "fit_mixed_reml_torch",
                            the_card_filled_up)

        y, X, groups = self._a_design_a_mixed_model_can_fit()
        fell_back = ml.perform_mixed_model(y, X, groups,
                                           regression_backend="torch")
        asked_for_the_cpu = MixedLM(y, X, groups=groups).fit()

        assert type(fell_back).__module__.startswith("statsmodels")
        assert np.allclose(np.asarray(fell_back.params, dtype=float),
                           np.asarray(asked_for_the_cpu.params, dtype=float))

    def test_a_failure_that_is_not_memory_is_not_quietly_refitted(
            self, a_card_that_answered, monkeypatch):
        """A bug in the torch fit must surface as that bug.

        Falling back on every exception would hand back a model that fitted,
        printed and plotted, so nothing downstream could tell that the run it
        describes never happened.
        """
        from spacr import ml, mixed_gpu

        def a_real_bug(*args, **kwargs):
            raise ValueError("the guide table lost its index")

        monkeypatch.setattr(mixed_gpu, "fit_mixed_reml_torch", a_real_bug)

        y, X, groups = self._a_design_a_mixed_model_can_fit()
        with pytest.raises(ValueError, match="lost its index"):
            ml.perform_mixed_model(y, X, groups, regression_backend="torch")


class TestThePackagingPromotion:
    """220: pyfixest, glum and gpytorch are core, gated where torch already
    gates."""

    @staticmethod
    def _core():
        import re
        from pathlib import Path

        text = Path("setup.py").read_text()
        block = re.search(r"^dependencies = \[(.*?)\n\]", text,
                          re.S | re.M).group(1)
        return block

    @pytest.mark.parametrize("package", ["pyfixest", "glum", "gpytorch"])
    def test_it_is_a_core_dependency(self, package):
        assert package in self._core()

    @pytest.mark.parametrize("package", ["pyfixest", "glum", "gpytorch"])
    def test_it_is_gated_at_the_floor_torch_already_imposes(self, package):
        """They add no interpreter constraint spaCR did not already have: a
        3.9 install does without them and still runs."""
        import re

        core = self._core()
        line = [l for l in core.splitlines() if package in l and "'" in l]
        assert line, package
        assert 'python_version >= "3.10"' in line[0], line[0]

    @pytest.mark.parametrize("package", ["numpyro", "pymer4"])
    def test_the_two_that_stay_extras_stay_extras(self, package):
        """numpyro drags 88 MB of jax and needs >=3.12; pymer4 needs R,
        which pip cannot install at any gate."""
        core = self._core()
        assert f"'{package}" not in core, f"{package} became core"
