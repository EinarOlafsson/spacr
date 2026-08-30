"""Three loops in segmentation QC that must pass over an item.

An override that is not a number, a finding with nothing to add, a file that
is not a scorecard. In each case doing the work anyway would put something
into the QC output that a reader would act on -- a threshold of NaN, a blank
indented line, a path that cannot be read.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# _resolve — an override that is not a number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "  ", "loose", [], {}])
def test_an_override_that_is_not_a_number_leaves_the_default_alone(value):
    """Arc 423 -> 416: the loop goes round without writing the key.

    Thresholds come from a settings file or a panel, where a cleared box is a
    string and a slider that was never touched is None. Writing the value
    would put NaN or a string into a threshold that is later compared with
    ``>=`` -- which does not raise, it just silently stops flagging anything.
    """
    from spacr.seg_qc import QC_DEFAULTS, _resolve

    key = next(iter(QC_DEFAULTS))
    resolved = _resolve({key: value})

    assert resolved[key] == QC_DEFAULTS[key]


def test_an_override_that_is_a_number_replaces_the_default():
    """The taken side, so the skip above is visibly a decision."""
    from spacr.seg_qc import QC_DEFAULTS, _resolve

    key = next(iter(QC_DEFAULTS))
    resolved = _resolve({key: 42})

    assert resolved[key] == 42


def test_an_unknown_threshold_name_is_refused_by_name():
    """The raise above the loop: a typo in a settings key must not be silent.

    Silently ignoring it is what makes a user believe they have loosened a
    threshold when they have not.
    """
    from spacr.seg_qc import _resolve

    with pytest.raises(Exception) as excinfo:
        _resolve({"min_objects_typo": 3})

    assert "min_objects_typo" in str(excinfo.value)


# ---------------------------------------------------------------------------
# format_findings — a finding with nothing to add
# ---------------------------------------------------------------------------

def test_a_finding_with_no_detail_or_fix_prints_only_its_headline():
    """Arcs 1945 -> 1947 and 1947 -> 1943.

    Both extra lines are indented under the headline. Printing them empty
    gives a blank indented line, which reads as a detail that failed to render
    rather than as a finding that needed none.
    """
    from spacr.seg_qc import Finding, format_findings

    text = format_findings([Finding(severity="warn", kind="count",
                                    headline="12 fields have no objects")])

    lines = [line for line in text.splitlines() if line.strip()]
    assert lines == ["[WARN] 12 fields have no objects"]
    assert len(text.splitlines()) == len(lines)


def test_a_finding_with_a_detail_and_a_fix_prints_all_three():
    """The taken sides."""
    from spacr.seg_qc import Finding, format_findings

    text = format_findings([Finding(
        severity="fail", kind="count", headline="no objects",
        detail="every field in plate1 scored zero",
        fix="check the mask channel")])

    assert "[FAIL] no objects" in text
    assert "    every field in plate1 scored zero" in text
    assert "    -> check the mask channel" in text


def test_nothing_flagged_says_so():
    """The early return, so the tests above are reached deliberately."""
    from spacr.seg_qc import format_findings

    assert format_findings([]) == "Nothing flagged."


# ---------------------------------------------------------------------------
# find_scorecards — files that are not scorecards
# ---------------------------------------------------------------------------

def test_files_that_are_not_scorecards_are_passed_over(tmp_path):
    """Arc 2160 -> 2159.

    A QC folder holds figures and logs beside the scorecards. Returning a PNG
    as a scorecard path means the reader opens it as a CSV, which fails far
    away from here with a parse error about the wrong file.
    """
    from spacr.seg_qc import CARD_PREFIX, find_scorecards

    qc = tmp_path / "qc"
    qc.mkdir()
    real = qc / f"{CARD_PREFIX}cell.csv"
    real.write_text("field,objects\n")
    (qc / f"{CARD_PREFIX}cell.png").write_bytes(b"\x89PNG")
    (qc / "notes.csv").write_text("not a scorecard\n")
    (qc / "unrelated.txt").write_text("nothing\n")

    found = find_scorecards(str(tmp_path))

    assert [os.path.basename(p) for p in found] == [f"{CARD_PREFIX}cell.csv"]


def test_a_scorecard_is_found_whatever_the_case_of_its_extension(tmp_path):
    """The lowercase check, which is what makes a .CSV from Windows readable."""
    from spacr.seg_qc import CARD_PREFIX, find_scorecards

    qc = tmp_path / "qc"
    qc.mkdir()
    (qc / f"{CARD_PREFIX}nucleus.CSV").write_text("field,objects\n")

    found = find_scorecards(str(tmp_path))

    assert len(found) == 1
