"""A regression that fails names its stage, its design and its exception.

Instruction 161, reported 2026-08-18: "merge works and the regression runs, but
failed for an unknown reason."

The message is its own bug whatever the underlying failure was: a run that got
far enough to fail knows the stage it reached, the shape of the design it had
built, and the exception -- and "failed" discards all three.
"""

import os

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import numpy as np
import pandas as pd
import pytest

from spacr.regression_failure import (FAILURE_FILENAME, describe_failure,
                                      write_failure_report)


def _frame():
    return pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c2", "p1_r2_c1"],
        "grna": ["g1", "g2", "g3"],
        "gene": ["A", "A", "B"],
        "y": [0.1, 0.2, 0.3],
    })


def test_the_report_names_the_stage_the_design_and_the_exception():
    try:
        raise np.linalg.LinAlgError("Singular matrix")
    except Exception as error:
        text = describe_failure(
            error, stage="fitting the gene level",
            settings={"regression_type": "mixed", "fdr_alpha": 0.05},
            frame=_frame(), include_traceback=False)

    assert "fitting the gene level" in text
    assert "rows in the design" in text and "3" in text
    assert "distinct wells" in text
    assert "LinAlgError" in text and "Singular matrix" in text
    assert "regression_type" in text


def test_a_known_failure_says_what_to_change():
    try:
        raise np.linalg.LinAlgError("Singular matrix")
    except Exception as error:
        text = describe_failure(error, include_traceback=False)
    assert "WHAT TO CHANGE" in text
    assert "singular" in text.lower()


def test_an_unknown_failure_admits_it_rather_than_guessing():
    """A wrong remedy is worse than none."""
    try:
        raise RuntimeError("something nobody has seen before")
    except Exception as error:
        text = describe_failure(error, include_traceback=False)
    assert "no recorded remedy" in text
    assert "guess" in text


def test_the_reporter_never_raises():
    """It must never replace the failure it is reporting."""
    class Awkward:
        @property
        def columns(self):
            raise ValueError("this frame refuses to describe itself")

        def __len__(self):
            raise ValueError("and refuses to be measured")

    try:
        raise ValueError("the real failure")
    except Exception as error:
        text = describe_failure(error, frame=Awkward(),
                                include_traceback=False)
    assert "the real failure" in text


def test_it_is_written_beside_the_run(tmp_path):
    try:
        raise MemoryError("out of memory")
    except Exception as error:
        path = write_failure_report(str(tmp_path), error, stage="fitting")

    assert path and os.path.basename(path) == FAILURE_FILENAME
    text = open(path).read()
    assert "fitting" in text
    # The remedy for this one is known and specific.
    assert "ols" in text
    # The traceback belongs in the FILE even though the console omits it.
    assert "Traceback" in text


def test_no_folder_is_not_a_crash(tmp_path):
    """A failure early enough to have no run folder is still reported."""
    try:
        raise ValueError("early")
    except Exception as error:
        assert write_failure_report("", error) is None


def test_perform_regression_re_raises_unchanged(monkeypatch):
    """The reporter ADDS to a failure; it must never replace one.

    A caller that handles a specific exception type has to keep seeing it.
    """
    from spacr import ml

    def boom(settings):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(ml, "_perform_regression", boom)
    with pytest.raises(np.linalg.LinAlgError):
        ml.perform_regression({"regression_type": "mixed"})


def test_the_stage_reaches_the_report(monkeypatch, capsys):
    from spacr import ml

    def boom(settings):
        settings["_regression_stage"] = "joining the counts"
        raise ValueError("no overlap")

    monkeypatch.setattr(ml, "_perform_regression", boom)
    with pytest.raises(ValueError):
        ml.perform_regression({"regression_type": "ols"})
    printed = capsys.readouterr().out
    assert "joining the counts" in printed
    assert "THE REGRESSION FAILED" in printed
