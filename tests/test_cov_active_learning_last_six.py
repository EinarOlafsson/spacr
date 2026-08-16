"""The last six uncovered lines in :mod:`spacr.active_learning`.

Instruction 60. The module was at 99% over its own tests -- these close it.
None of the four is unreachable, so none is a candidate for a pragma: each is
a real branch a real caller reaches, and each is asserted on its EFFECT rather
than merely executed. A test that runs a line without checking what it did
raises the number and finds nothing, which is the failure mode this
instruction warns about.

Worth recording: the standing coverage baseline listed this module at 37% with
650 uncovered statements. Measured over its own tests it is 99% with six. Both
numbers are real and they are not comparable -- a per-module figure only means
something alongside the test set it was taken over, which is the measurement
trap instruction 60 documents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
#  _read_only_uri -- the guard that makes a reader physically unable to write
# --------------------------------------------------------------------------- #

def test_a_read_only_uri_is_one_sqlite_refuses_writes_through():
    """Not cosmetic: this is what stops an analysis path from writing to a
    user's measurement database."""
    import sqlite3

    from spacr.active_learning import _read_only_uri

    uri = _read_only_uri("/tmp/example.db")
    assert uri.startswith("file:")
    assert uri.endswith("?mode=ro")

    # And prove the mode actually binds, rather than trusting the string.
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "ro.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE t (a INTEGER)")
    reader = sqlite3.connect(_read_only_uri(path), uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("INSERT INTO t VALUES (1)")
    finally:
        reader.close()


def test_a_windows_path_survives_the_uri_quoting():
    """Backslashes become forward slashes; the drive colon is not escaped."""
    from spacr.active_learning import _read_only_uri

    uri = _read_only_uri(r"C:\data\plate1\measurements.db")
    assert "\\" not in uri
    assert "C:/data/plate1/measurements.db" in uri


# --------------------------------------------------------------------------- #
#  _concentration -- the empty case
# --------------------------------------------------------------------------- #

def test_concentration_of_nothing_is_zero_not_a_division_error():
    """Zero labels is an ordinary state -- it is what the first round looks
    like before anyone has annotated -- so it must answer, not raise."""
    from spacr.active_learning import _concentration

    out = _concentration(pd.Series([], dtype=int))

    assert out == {"n": 0, "n_groups": 0, "top": None, "top_n": 0,
                   "top_share": 0.0, "hhi": 0.0, "effective_groups": 0.0}


def test_concentration_of_one_group_is_maximally_lopsided():
    """The contrast that gives the empty case its meaning: hhi is 1.0 when
    every label came from one group."""
    from spacr.active_learning import _concentration

    out = _concentration(pd.Series({"plate1": 40}))
    assert out["n"] == 40
    assert out["hhi"] == pytest.approx(1.0)
    assert out["top"] == "plate1"


# --------------------------------------------------------------------------- #
#  format_learning_curve -- the "no rounds yet" report
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_the_curve_report_says_so_when_there_are_no_rounds(empty):
    """A blank report is worse than a sentence saying what to do next: this
    is the state a user is in before their first retrain."""
    from spacr.active_learning import format_learning_curve

    text = format_learning_curve(empty)

    assert "Active-learning rounds" in text
    assert "No round recorded yet" in text
    # It must tell them where to go, not merely that nothing happened.
    assert "Annotate" in text
    # And not pretend to have a table.
    assert "held-out" not in text


def test_the_curve_report_draws_a_table_once_a_round_exists():
    """The other side of the same branch, so the empty case is not passing
    for the wrong reason."""
    from spacr.active_learning import format_learning_curve

    curve = pd.DataFrame({
        "round": [1], "n_labels": [10], "n_new_labels": [10],
        "n_holdout": [5], "holdout_accuracy": [0.8], "gain": [0.0],
        "per_class": [{"pos": 0.7, "neg": 0.9}],
    })
    text = format_learning_curve(curve)

    assert "No round recorded yet" not in text
    assert "held-out" in text


# --------------------------------------------------------------------------- #
#  _predict_proba -- an empty batch
# --------------------------------------------------------------------------- #

def test_predicting_on_an_empty_batch_returns_an_empty_matrix():
    """Shape, not just emptiness.

    A caller stacks this against other rounds' probabilities, so a (0,) or a
    (0, 1) would fail to concatenate later -- somewhere else entirely, which
    is exactly the kind of error this branch exists to prevent.
    """
    from spacr.active_learning import _predict_proba

    class _NeverCalled:
        def predict_proba(self, x):        # pragma: no cover - must not run
            raise AssertionError("an empty batch must not reach the model")

    out = _predict_proba(_NeverCalled(), np.empty((0, 4)), n_classes=3)

    assert out.shape == (0, 3)
    assert np.vstack([out, np.zeros((2, 3))]).shape == (2, 3)
