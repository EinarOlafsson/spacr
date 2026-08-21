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
