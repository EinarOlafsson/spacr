"""Two decisions the torch backend makes that the ordinary fit never reaches.

The restart loop stops on the deviance rather than on a count, so a fit whose
deviance settles immediately -- which is every fixture in the suite -- never
reaches the end of ``range(_MAX_RESTARTS)``. The bound is what keeps a
pathological surface from restarting forever, so it is driven here with a
tolerance no movement can satisfy.

The CUDA synchronisation is the other one. It decides how long the fit is
reported to have taken, because an asynchronous CUDA queue is still draining
when ``perf_counter`` is read. It is driven from a DESCRIBED device rather
than a real one: the GPU on this machine drives the user's display, and the
question the code asks is ``torch_device.type``, which a device object can
answer without a driver behind it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import mixed_gpu as MG


class _CudaLabelledDevice(str):
    """A device that answers ``'cuda'`` to ``.type`` and allocates on the CPU.

    ``resolve_device`` returns something the tensors are built on AND something
    the synchronisation branch is chosen from. Subclassing ``str`` keeps torch
    able to place tensors (it parses the string, which reads ``'cpu'``) while
    ``.type`` reports what the fit is being told it is running on.
    """

    type = "cuda"


class _StoppedClock:
    """A ``perf_counter`` that moves only when a test moves it."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clustered():
    """Six wells, one predictor, one outcome -- a fit that converges quickly."""
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
# the restart loop is bounded
# ---------------------------------------------------------------------------

def test_a_deviance_that_never_settles_restarts_max_restarts_times_and_stops(
        clustered, monkeypatch):
    """The loop is capped, so an unsettling deviance ends rather than hangs.

    With a restart tolerance no pair of deviances can satisfy, the only exit
    is the end of ``range(_MAX_RESTARTS)``. Raising the cap must therefore buy
    more deviance evaluations and nothing else: the estimates are already at
    the optimum, so the extra passes must not move them.
    """
    monkeypatch.setattr(MG, "_RESTART_TOLERANCE", -1.0)

    monkeypatch.setattr(MG, "_MAX_RESTARTS", 2)
    two = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                  clustered["groups"], device="cpu",
                                  max_iter=20)
    monkeypatch.setattr(MG, "_MAX_RESTARTS", 6)
    six = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                  clustered["groups"], device="cpu",
                                  max_iter=20)

    assert six.n_deviance_evals > two.n_deviance_evals
    assert two.converged and six.converged
    assert np.allclose(two.fe_params.to_numpy(), six.fe_params.to_numpy(),
                       atol=1e-6)
    assert two.scale == pytest.approx(six.scale, rel=1e-6)


def test_a_settled_deviance_stops_before_the_restart_cap(clustered, monkeypatch):
    """The default fit exits on movement, not on the count, so it costs less.

    The same fit, forced past the tolerance so only the cap can stop it, is
    the control: it reaches the same estimates and pays more deviance
    evaluations for them.
    """
    settled = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                      clustered["groups"], device="cpu",
                                      max_iter=20)

    monkeypatch.setattr(MG, "_RESTART_TOLERANCE", -1.0)
    forced = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                     clustered["groups"], device="cpu",
                                     max_iter=20)

    assert settled.converged
    assert settled.n_deviance_evals < forced.n_deviance_evals
    assert np.allclose(settled.fe_params.to_numpy(),
                       forced.fe_params.to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# a CUDA fit waits for the device before it reports its time
# ---------------------------------------------------------------------------

def test_a_cuda_fit_synchronises_the_device_inside_the_time_it_reports(
        clustered, monkeypatch):
    """``fit_seconds`` covers the synchronisation, not just the queueing.

    CUDA work is queued, so a ``perf_counter`` read taken before the queue
    drains would report a fit that never happened yet. The stopped clock only
    advances inside the fake ``synchronize``, so the seconds that come back
    prove the wait happened between the two readings.
    """
    import torch

    clock = _StoppedClock()
    calls = []

    def fake_synchronize(*args, **kwargs):
        calls.append(args)
        clock.now += 7.5

    monkeypatch.setattr(MG, "time", clock)
    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)
    monkeypatch.setattr(
        MG, "resolve_device",
        lambda device=MG.GPU_DEVICE: _CudaLabelledDevice("cpu"))

    result = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                     clustered["groups"], device="cuda")

    assert len(calls) == 1
    assert result.fit_seconds == pytest.approx(7.5)
    # The described device changed nothing about the answer.
    assert result.converged
    assert result.fe_params["predictor"] == pytest.approx(1.5, abs=0.2)


def test_a_cpu_fit_never_touches_the_cuda_layer(clustered, monkeypatch):
    """A CPU device must not reach a synchronisation there is nothing to do."""
    import torch

    def refuse(*args, **kwargs):
        raise AssertionError("a CPU fit asked the CUDA layer to synchronise")

    monkeypatch.setattr(torch.cuda, "synchronize", refuse)

    result = MG.fit_mixed_reml_torch(clustered["y"], clustered["X"],
                                     clustered["groups"], device="cpu")

    assert result.device == "cpu"
    assert result.converged
