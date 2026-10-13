"""Item 3 of instruction 322: the diagnostics button carries the verdict.

The suite's worst sheet is already computed and written --
``diagnostic_summary.csv`` carries a ``suite``/``verdict_level`` row --
so the button READS it. Recomputing would let the button and the panels
disagree about the same run.

``""`` is not a pass. A run that has not happened has no verdict, and a
green dot would claim it did and was fine.
"""
from __future__ import annotations

import csv
import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.regression import DIAGNOSTICS_DIRNAME, DiagnosticsOpener


def _summary(folder, level, detail="design: fail - it cannot identify"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "diagnostic_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["section", "metric", "value"])
        writer.writeheader()
        writer.writerow({"section": "design", "metric": "verdict_level",
                         "value": "pass"})
        writer.writerow({"section": "suite", "metric": "verdict_level",
                         "value": level})
        writer.writerow({"section": "suite", "metric": "verdict",
                         "value": detail})
    return path


@pytest.fixture
def opener(qtbot, tmp_path, monkeypatch):
    from spacr.qt.screens import regression as R
    from PySide6.QtWidgets import QWidget

    screen = QWidget()
    qtbot.addWidget(screen)
    monkeypatch.setattr(R, "project_path", lambda _screen: str(tmp_path))
    return DiagnosticsOpener(screen), tmp_path


def _diagnostics_folder(root):
    from spacr.qt.screens.regression import RESULTS_DIRNAME

    folder = os.path.join(str(root), RESULTS_DIRNAME, "run1",
                          DIAGNOSTICS_DIRNAME)
    os.makedirs(folder, exist_ok=True)
    return folder


@pytest.mark.parametrize("level", ["pass", "check", "fail"])
def test_the_verdict_is_read_from_the_summary(opener, level):
    """THE READ. Whatever the suite row says is what the button shows."""
    diagnostics, root = opener
    _summary(_diagnostics_folder(root), level)

    found, detail = diagnostics.verdict()

    assert found == level
    assert "cannot identify" in detail


def test_a_project_with_no_run_has_no_verdict(opener):
    """NOT A PASS. "Not measured" and "measured and fine" are different
    answers, and a green dot would collapse them."""
    diagnostics, _root = opener
    assert diagnostics.verdict() == ("", "")


def test_a_summary_that_cannot_be_read_leaves_it_unbadged(opener,
                                                          monkeypatch):
    """A summary that is missing or corrupt is not a verdict either. The
    button still opens the folder, which is where the panels are."""
    diagnostics, root = opener
    folder = _diagnostics_folder(root)
    with open(os.path.join(folder, "diagnostic_summary.csv"), "w",
              encoding="utf-8") as handle:
        handle.write("this is not a csv with the expected columns\n")

    level, _detail = diagnostics.verdict()

    assert level == ""


def test_the_worst_sheet_is_what_is_read_not_the_first(opener):
    """The suite row IS the worst, computed where the sheets are.

    Pinned because a reader that took the first per-section row would
    report "pass" for a run whose design cannot be identified -- the
    exact case the instruction says must not be averaged away.
    """
    diagnostics, root = opener
    _summary(_diagnostics_folder(root), "fail")

    assert diagnostics.verdict()[0] == "fail", (
        "the per-section 'pass' row above the suite row was read instead")


# ---------------------------------------------------------------------------
# The dot itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["pass", "check", "fail"])
def test_the_button_keeps_the_verdict_it_is_given(qtbot, level):
    from spacr.qt.widgets.fold_strip import FoldButton

    button = FoldButton("regression_diagnostics", "Diagnostics", "")
    qtbot.addWidget(button)

    button.set_verdict(level, "because of a reason")

    assert button._verdict == level
    assert button._verdict_detail == "because of a reason"
    assert button._verdict_ink(level).startswith("#")


def test_an_unknown_level_clears_the_badge_rather_than_inventing_one(qtbot):
    """A level the palette does not know must not draw a dot in some
    default colour -- that would be a verdict spaCR never reached."""
    from spacr.qt.widgets.fold_strip import FoldButton

    button = FoldButton("regression_diagnostics", "Diagnostics", "")
    qtbot.addWidget(button)

    button.set_verdict("fail")
    button.set_verdict("something_else")

    assert button._verdict == ""


def test_painting_an_unbadged_button_draws_no_dot(qtbot):
    """The clear path, exercised through a real paint."""
    from PySide6.QtGui import QPixmap
    from spacr.qt.widgets.fold_strip import FoldButton

    button = FoldButton("regression_diagnostics", "Diagnostics", "")
    qtbot.addWidget(button)
    button.resize(40, 40)

    button.set_verdict("")
    blank = QPixmap(button.size())
    button.render(blank)

    button.set_verdict("fail")
    badged = QPixmap(button.size())
    button.render(badged)

    assert blank.toImage() != badged.toImage(), (
        "the badge made no difference to what was painted")


def test_the_dot_uses_the_same_ink_as_the_panel_stamp(qtbot):
    """So a button and a panel cannot disagree about what "check" is."""
    from spacr.qt.widgets.fold_strip import FoldButton

    button = FoldButton("regression_diagnostics", "Diagnostics", "")
    qtbot.addWidget(button)

    from spacr.regression_qc import _VERDICT_INK

    for level in ("pass", "check", "fail"):
        if level in _VERDICT_INK:
            assert button._verdict_ink(level) == str(_VERDICT_INK[level])
