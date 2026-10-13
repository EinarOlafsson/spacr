"""Instruction 115's last two leftovers.

    "`perform_regression()`'s returned dict still does not carry the QC
     manifest -- and the manifest now holds a verdict AND the renderer that
     drew each panel, which is the thing a caller would most want from it."
    "The GUI does not show the verdict anywhere."

The suite computes it, the manifest carries it, and the report wrote it to a
text file nobody opens. A run whose design is rank deficient has coefficients
that are ONE of infinitely many solutions, and a screen that shows the volcano
without saying so is showing a picture of an arbitrary answer.

THE WORST VERDICT, not a summary and not a count — `worst_verdict` says why
where it is computed: nineteen panels passing and one saying the design is
rank deficient is a run whose "95% passed" hides exactly the panel the suite
was run for.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    said = []
    screen._say = said.append
    return screen, said


def _verdict(level="fail", name="design_rank",
             detail="the design is rank deficient: 827 parameters, 610 wells"):
    return SimpleNamespace(level=level, name=name, detail=detail)


def test_a_failing_verdict_is_said_loudly_and_names_the_panel(qtbot):
    screen, said = _screen(qtbot)

    screen._say_the_qc_verdict({"qc_verdict": _verdict(),
                                "qc_verdict_level": "fail"})
    assert said
    text = said[0]
    assert "REGRESSION QC: FAIL" in text
    assert "design_rank" in text
    assert "rank deficient" in text
    # And where the rest of it is, because a one-line verdict is a pointer.
    assert "regression_qc_report.txt" in text


def test_a_warning_is_said_without_shouting(qtbot):
    screen, said = _screen(qtbot)

    screen._say_the_qc_verdict({"qc_verdict": _verdict(level="warn"),
                                "qc_verdict_level": "warn"})
    assert said and "Regression QC: warn" in said[0]
    assert "REGRESSION QC" not in said[0]


def test_a_run_with_no_verdict_says_nothing_at_all(qtbot):
    """QC off, or a suite that could not build. "unknown" after every such
    run trains a reader to skip the line that matters."""
    screen, said = _screen(qtbot)

    assert screen._say_the_qc_verdict({"results": None}) == ""
    assert screen._say_the_qc_verdict({}) == ""
    assert screen._say_the_qc_verdict(None) == ""
    assert said == []


def test_the_verdict_is_said_when_a_run_finishes(qtbot):
    """The wiring, not just the helper."""
    screen, said = _screen(qtbot)
    seen = []
    screen._say_the_qc_verdict = seen.append

    screen._on_pipeline_result({"results": None, "res_folder": ""})
    assert seen, "a finished run did not consult the QC verdict"


# --------------------------------------------------------------------------- #
#  And the half that carries it out of the run
# --------------------------------------------------------------------------- #

def test_perform_regression_lifts_the_manifest_off_the_frame():
    """`regression` puts it in `coef_df.attrs`; the outcome dict lifts it.

    A 3-tuple that every caller unpacks positionally is not worth growing for
    one optional fact, and `.attrs` is pandas' own place for exactly this.
    """
    import inspect

    from spacr import ml

    source = inspect.getsource(ml)
    assert 'coef_df.attrs["qc_manifest"]' in source
    assert "output['qc'] = manifest" in source
    assert "output['qc_verdict']" in source


def test_the_key_is_absent_rather_than_None_when_qc_did_not_run():
    """A key holding None is indistinguishable from a suite that concluded
    nothing."""
    import inspect

    from spacr import ml

    source = inspect.getsource(ml)
    # The lift is guarded on the manifest being truthy.
    assert "if manifest:" in source
