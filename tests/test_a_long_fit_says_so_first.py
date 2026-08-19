"""The mixed model announces its cost before it spends it.

Measured on the maintainer's four-plate TSG101 screen: the same mixed model
took 26 SECONDS on `regression_backend='torch'` and was still inside scipy's
BFGS after twenty-five minutes on statsmodels. A CPU-bound fit that prints
nothing for half an hour is indistinguishable from a freeze, and it was
reported as one -- "i ran an ols then a mixed regression model and this hung
my computer twice so i had to restart it".

The default is NOT changed by any of this. statsmodels produced every
existing result; what was missing was the sentence.
"""
import pandas as pd
import pytest

from spacr.ml import _say_what_a_mixed_fit_will_cost


def test_the_slow_backend_says_it_is_the_slow_one(capsys):
    _say_what_a_mixed_fit_will_cost("statsmodels", pd.DataFrame({"a": range(7)}))

    printed = capsys.readouterr().out
    assert "mixed model" in printed
    assert "7 wells" in printed, "say how big the job is, not just that it is big"
    assert "prints nothing while it runs" in printed, (
        "the whole point: name the silence, so it is not read as a hang")


def test_the_fast_backend_says_nothing_at_all(capsys):
    _say_what_a_mixed_fit_will_cost("torch", pd.DataFrame({"a": range(7)}))

    assert capsys.readouterr().out == ""


def test_it_names_the_one_setting_that_removes_the_wait(capsys):
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("the alternative is only offered when it is usable")

    _say_what_a_mixed_fit_will_cost("statsmodels", None)

    printed = capsys.readouterr().out
    assert "regression_backend='torch'" in printed


def test_a_frame_it_cannot_measure_is_not_an_error(capsys):
    # Never let the announcement be the thing that breaks the fit.
    _say_what_a_mixed_fit_will_cost("statsmodels", object())

    assert "mixed model" in capsys.readouterr().out
