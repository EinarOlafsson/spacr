"""Reporting the Torch CUDA allocator WITHOUT loading Torch.

The whole point of `_gpu_memory_mb` is in its docstring: report an
*already-initialised* allocator. It reads `sys.modules` rather than
importing, because a timing report that imported Torch would add
seconds to the very startup it is measuring -- and would make the
measurement a function of the measuring.

So every answer distinguishes three states that a naive version would
collapse into one number: Torch absent, Torch present but the allocator
never initialised, and Torch running on a device.
"""
from __future__ import annotations

import sys

import pytest

from spacr.qt import timing as T


class TestWithoutTorch:

    def test_no_torch_at_all_reports_nothing_known(self, monkeypatch):
        """None, not zero. Zero would claim the GPU was measured and idle."""
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        assert T._gpu_memory_mb() == {"allocated_mb": None,
                                      "peak_allocated_mb": None}

    def test_torch_without_a_cuda_attribute_reports_nothing_known(
            self, monkeypatch):
        """THE UNCOVERED ARM.

        A CPU-only Torch build has no `cuda` attribute at all. Asking it
        for one would raise; answering None says the figure is unknown,
        which is true.
        """
        import types

        monkeypatch.setitem(sys.modules, "torch",
                            types.ModuleType("torch"))
        assert T._gpu_memory_mb() == {"allocated_mb": None,
                                      "peak_allocated_mb": None}

    def test_torch_is_never_imported_to_answer(self, monkeypatch):
        """The measurement must not change what it measures."""
        import builtins

        real = builtins.__import__

        def refuse(name, *a, **k):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("the timing report imported torch")
            return real(name, *a, **k)

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", refuse)
        assert T._gpu_memory_mb()["allocated_mb"] is None


class TestWithTorchPresent:

    @staticmethod
    def _torch(cuda):
        import types

        module = types.ModuleType("torch")
        module.cuda = cuda
        return module

    def test_an_uninitialised_allocator_reports_zero_not_unknown(self,
                                                                 monkeypatch):
        """Zero is right here: Torch is there, CUDA simply has not been
        used yet, so nothing is allocated. That is a measurement."""
        class _Cuda:
            @staticmethod
            def is_initialized():
                return False

        monkeypatch.setitem(sys.modules, "torch", self._torch(_Cuda()))
        assert T._gpu_memory_mb() == {"allocated_mb": 0.0,
                                      "peak_allocated_mb": 0.0}

    def test_a_live_allocator_is_reported_in_megabytes(self, monkeypatch):
        class _Cuda:
            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def memory_allocated():
                return 512 * 1024 * 1024

            @staticmethod
            def max_memory_allocated():
                return 1024 * 1024 * 1024

        monkeypatch.setitem(sys.modules, "torch", self._torch(_Cuda()))
        out = T._gpu_memory_mb()
        assert out["allocated_mb"] == pytest.approx(512.0)
        assert out["peak_allocated_mb"] == pytest.approx(1024.0)

    def test_an_allocator_that_raises_reports_nothing_known(self,
                                                            monkeypatch):
        """A driver that has gone away mid-run must not take the report
        with it -- this is called while assembling a timing summary."""
        class _Cuda:
            @staticmethod
            def is_initialized():
                raise RuntimeError("CUDA driver has been unloaded")

        monkeypatch.setitem(sys.modules, "torch", self._torch(_Cuda()))
        assert T._gpu_memory_mb() == {"allocated_mb": None,
                                      "peak_allocated_mb": None}

    def test_the_three_states_are_distinguishable(self, monkeypatch):
        """Absent, idle and running must not collapse into one number.

        A report that answered 0.0 for "no Torch" would be claiming a
        measured, empty GPU on a machine that has none.
        """
        import types

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        absent = T._gpu_memory_mb()

        class _Idle:
            @staticmethod
            def is_initialized():
                return False

        monkeypatch.setitem(sys.modules, "torch", self._torch(_Idle()))
        idle = T._gpu_memory_mb()

        assert absent["allocated_mb"] is None
        assert idle["allocated_mb"] == 0.0
        assert absent != idle
