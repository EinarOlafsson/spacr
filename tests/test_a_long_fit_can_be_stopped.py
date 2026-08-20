"""Instruction 140 C — a mixed fit answers Stop.

`grep -n "cancellation" spacr/ml.py` returned NOTHING when 140 was filed, so
`worker.request_cancel` could not land inside a mixed fit however long it ran
— and on a real screen (823 guides over 610 wells) that fit is tens of minutes
to hours. What made it stoppable was `_force_stop`, which PARKS the thread and
gives the window back while the fit carries on in the background. That is
honest — it says so — but it is not stopping.

Two properties, and the second is the one a user notices a day later: pressing
Stop actually ends the fit, and the run folder afterwards says the run was
STOPPED rather than that it failed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
import statsmodels.formula.api as smf                            # noqa: E402

from spacr.cancellation import (                                 # noqa: E402
    CancellationToken, PipelineCancelled, installed_token,
)
from spacr.ml import _answering_stop                             # noqa: E402


def _model(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n),
                          "g": rng.integers(0, 20, size=n)})
    frame["y"] = frame["x"] * 0.3 + rng.normal(size=n)
    return smf.mixedlm("y ~ x", data=frame, groups=frame["g"])


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# -- it stops --------------------------------------------------------------

def test_a_fit_stops_when_the_user_asks_it_to():
    token = CancellationToken()
    token.cancel()
    with installed_token(token):
        with pytest.raises(PipelineCancelled):
            _answering_stop(_model()).fit()


def test_the_refusal_propagates_out_of_the_optimiser_unchanged():
    """An optimiser that swallowed it would leave the fit running silently."""
    token = CancellationToken(reason="the user pressed Stop")
    token.cancel()
    with installed_token(token):
        with pytest.raises(PipelineCancelled, match="pressed Stop"):
            _answering_stop(_model()).fit()


def test_a_fit_nobody_stops_returns_the_same_answer_as_before():
    plain = _model().fit()
    hooked = _answering_stop(_model()).fit()
    assert np.allclose(np.asarray(plain.params, dtype=float),
                       np.asarray(hooked.params, dtype=float))


def test_the_objective_is_a_finer_hook_than_the_callback_statsmodels_offers():
    """Which is the whole reason `fit(callback=)` was not used.

    statsmodels calls its callback once per optimizer ATTEMPT, not per
    iteration, so it offers a couple of chances to notice Stop in a fit that
    runs for an hour. The two are counted against each other on the SAME
    model rather than against a number somebody picked, because what matters
    is which hook is finer and not how many evaluations this small fixture
    happens to need.
    """
    evaluations, attempts = [], []
    model = _model()
    original = model.loglike
    model.loglike = lambda *a, **k: (evaluations.append(1),
                                     original(*a, **k))[1]
    _answering_stop(model).fit(callback=lambda params: attempts.append(1))

    assert len(evaluations) > len(attempts)
    assert len(evaluations) >= 5


def test_the_hook_is_on_the_instance_never_on_the_class():
    """Patching statsmodels would checkpoint every model in the process."""
    import statsmodels.regression.mixed_linear_model as mlm

    before = mlm.MixedLM.loglike
    _answering_stop(_model())
    assert mlm.MixedLM.loglike is before


# -- and the folder says so ------------------------------------------------

def test_a_stopped_run_is_recorded_as_cancelled_and_not_as_failed(tmp_path,
                                                                  monkeypatch):
    import json

    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path / "runs")
    with pytest.raises(PipelineCancelled):
        with run_journal.open_run("regression", {}) as run:
            raise PipelineCancelled("the user pressed Stop")

    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    # The traceback is KEPT: where a long fit was interrupted is exactly what
    # a user asks afterwards.
    assert "PipelineCancelled" in (manifest.get("traceback") or "")


def test_a_run_that_actually_broke_is_still_recorded_as_failed(tmp_path,
                                                              monkeypatch):
    import json

    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path / "runs")
    with pytest.raises(ValueError):
        with run_journal.open_run("regression", {}) as run:
            raise ValueError("the design is not identifiable")

    assert json.loads(
        (run.dir / "manifest.json").read_text())["status"] == "failed"


def test_the_folder_of_a_stopped_run_is_there_to_be_looked_at(tmp_path,
                                                             monkeypatch):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path / "runs")
    with pytest.raises(PipelineCancelled):
        with run_journal.open_run("regression", {"src": str(tmp_path)}) as run:
            raise PipelineCancelled("stopped")

    assert (run.dir / "manifest.json").is_file()
    assert (run.dir / "settings.json").is_file()
