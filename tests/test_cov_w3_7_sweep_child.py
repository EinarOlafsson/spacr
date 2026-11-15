"""Import-time set-up of the contained-trial worker ``spacr.sweep_child``.

Importing the worker raises its OOM score and sets one-thread BLAS defaults
before NumPy can size its pools. What ``main`` writes for each trial is
tested in ``tests/test_sweep_child_trial_worker.py``.
"""
from __future__ import annotations

import os

import pytest

import spacr.sweep_child  # noqa: F401  (the import is what is under test)


def test_the_worker_volunteers_for_the_oom_killer():
    """Importing the child raises its own OOM score so the parent is not the target.

    The write happens at import; this reads back what the running interpreter
    now carries. Non-Linux kernels have no such file and the module tolerates
    that, so the assertion only applies where the file exists.
    """
    path = f"/proc/{os.getpid()}/oom_score_adj"
    if not os.path.exists(path):
        pytest.skip("no /proc/<pid>/oom_score_adj on this kernel")
    with open(path) as handle:
        assert int(handle.read().strip()) == 800


def test_blas_thread_limits_are_set_before_numpy_can_size_its_pool():
    """The module sets a one-thread default for every BLAS variable it names."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(variable), f"{variable} was left unset"
