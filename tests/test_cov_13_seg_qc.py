"""Segmentation QC when the mask, the card or the plate layout is not what it says.

A QC verdict is only worth having if it cannot be produced by accident. Every
path here is one where something is missing or damaged, and the rule is the
same throughout: refuse rather than default. A card with no verdict column
must not read as "ok"; a threshold that is not a number must not silently
become one; a plate with one row must not be reported as a gradient across
its halves; a mask holding NaN is not a label image.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from spacr import seg_qc
from spacr.seg_qc import (
    CARD_DIR,
    CARD_PREFIX,
    FLAG_EMPTY,
    FLAG_UNREADABLE,
    FieldQC,
    Finding,
    QCDigest,
    Scorecard,
    explain_flag,
    format_digest,
    format_findings,
    parse_field_name,
    score_field,
    thresholds_from_settings,
    write_scorecard,
)


# ---------------------------------------------------------------------------
# mask builders
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hit = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    sub = labels[y0:y1, x0:x1]
    sub[hit] = value
    return labels


def _grid_field(shape=(256, 256), radius=8, spacing=40, margin=30, count=None):
    labels = np.zeros(shape, np.int32)
    value = 0
    for cy in range(margin, shape[0] - margin, spacing):
        for cx in range(margin, shape[1] - margin, spacing):
            if count is not None and value >= count:
                return labels
            value += 1
            _disc(labels, cy, cx, radius, value)
    return labels


# ---------------------------------------------------------------------------
# thresholds that are not numbers
# ---------------------------------------------------------------------------

def test_a_threshold_that_is_not_a_number_is_ignored_rather_than_guessed():
    """A blank or mistyped ``seg_qc_*`` setting must fall back to the default.

    The settings dict is filled from GUI fields and from user CSVs, so a
    threshold arriving as ``""``, ``"none"`` or a list is ordinary. Coercing
    any of those to a number would move the line between pass and fail
    without saying so.
    """
    key = "seg_qc_border_fraction"

    assert thresholds_from_settings({key: "not a number"}) == {}
    assert thresholds_from_settings({key: [0.4]}) == {}
    assert thresholds_from_settings({key: None}) == {}
    assert thresholds_from_settings({key: float("nan")}) == {}
    # A real number still gets through, so the guard is not simply dropping
    # everything.
    assert thresholds_from_settings({key: "0.4"}) == {"border_fraction": 0.4}
    assert thresholds_from_settings({key: 0.4}) == {"border_fraction": 0.4}


# ---------------------------------------------------------------------------
# masks that are not label images
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask, expected", [
    (np.full((16, 16), np.nan, dtype=np.float32), "NaN or infinite"),
    (np.zeros((16, 16), dtype=np.complex64), "not a label type"),
    (np.array([[0, -1], [2, 3]], dtype=np.int32), "negative labels"),
])
def test_a_mask_that_is_not_a_label_image_is_flagged_and_says_why(mask,
                                                                  expected):
    """One corrupt field must not cost the plate its scorecard.

    Each of these is a real shape a mask arrives in: a float array a
    normalisation divided by zero, a complex array from the wrong reader, and
    a signed array where a downstream step wrote -1 for "unassigned". None of
    them can be counted, and each has to say which one it was.
    """
    qc = score_field(mask, object_type="cell", field="plate1_A01_f1")

    assert qc.flags == [FLAG_UNREADABLE]
    assert qc.severity == "fail"
    assert expected in qc.note
    assert qc.n_objects == 0


def test_a_foreground_too_small_to_hold_a_seed_is_not_called_fused():
    """The fusion cross-check needs seeds; with none it must claim nothing.

    A field holding one four-pixel speck has a foreground the distance
    transform cannot find a single peak in. Reporting zero seeds is the honest
    answer; anything else would compare a count against a number that was
    never computed.
    """
    speck = np.zeros((64, 64), np.int32)
    speck[30:32, 30:32] = 1

    assert seg_qc._split_seed_count(speck > 0, min_diameter=30.0) == 0


# ---------------------------------------------------------------------------
# the shapes a mask source arrives in
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# formatting a number nobody computed
# ---------------------------------------------------------------------------

def test_a_pixel_size_that_was_never_computed_prints_as_a_dash():
    """An empty field has no median diameter, and 0.0 px would be a claim.

    The scorecard is read by eye. A column of ``0.0`` invites the reader to
    conclude the objects were tiny rather than that there were none.
    """
    assert seg_qc._px(None) == "-"
    assert seg_qc._px(float("nan")) == "-"
    assert seg_qc._px(12.34) == "12.3"


def test_writing_a_scorecard_for_nothing_writes_no_file(tmp_path):
    """No fields is not an empty card; it is no card.

    An empty CSV beside the masks would be read back later as a plate that
    scored clean.
    """
    assert write_scorecard([], str(tmp_path), "cell") is None
    assert not (tmp_path / CARD_DIR).exists()


# ---------------------------------------------------------------------------
# how the flag guidance and the findings print
# ---------------------------------------------------------------------------

def test_a_flag_explanation_prints_as_its_flag_and_headline():
    """``str()`` on guidance is what a log line and a tooltip both use."""
    guidance = explain_flag(FLAG_EMPTY)

    assert str(guidance) == f"{guidance.flag}: {guidance.headline}"
    assert guidance.flag == FLAG_EMPTY


def test_a_finding_prints_its_severity_and_reads_as_one_paragraph():
    """The two renderings a finding has: a list line and a block of prose."""
    finding = Finding(severity="fail", kind="flag",
                      headline="plate1: 3 cell field(s) empty",
                      detail="Nothing was segmented there.",
                      fix="Check the diameter.")

    assert str(finding) == "[fail] plate1: 3 cell field(s) empty"
    assert finding.text() == ("plate1: 3 cell field(s) empty "
                              "Nothing was segmented there. "
                              "Check the diameter.")


def test_a_finding_with_nothing_but_a_headline_does_not_pad_it_with_spaces():
    """The joiner has to drop the empty parts, not join around them."""
    assert Finding(severity="warn", kind="flag",
                   headline="plate1: something").text() == "plate1: something"


def test_nothing_flagged_says_so_rather_than_printing_an_empty_report():
    """A blank report reads as a report that failed to run."""
    assert format_findings([]) == "Nothing flagged."


# ---------------------------------------------------------------------------
# plate addresses
# ---------------------------------------------------------------------------

def test_a_field_address_prints_the_well_when_it_has_one():
    """The address is what a finding names, so it has to degrade cleanly.

    A hand-assembled folder has no well in its file names, and printing
    ``plate1/`` for it would look like a well whose name went missing.
    """
    assert str(parse_field_name("plate1_A01_f1.npy")) == "plate1/A01"
    assert str(parse_field_name("just_a_name.npy")) == "just"
    assert str(parse_field_name("plateonly.npy")) == "plateonly"


def test_naming_no_wells_at_all_produces_no_list():
    """An empty list must not print as an empty pair of parentheses."""
    assert seg_qc._name_list([], max_named=3) == ""
    assert seg_qc._name_list(["", None], max_named=3) == ""
    assert seg_qc._name_list(["A01", "A02"], max_named=3) == "A01, A02"


def test_a_single_row_or_column_prints_as_itself_not_as_a_range():
    """``A-A`` and ``3-3`` are ranges of one, which read as a mistake."""
    assert seg_qc._row_range([]) == ""
    assert seg_qc._row_range(["c"]) == "C"
    assert seg_qc._row_range(["A", "B", "C"]) == "A-C"
    assert seg_qc._row_range(["A", "C"]) == "A, C"

    assert seg_qc._column_range([]) == ""
    assert seg_qc._column_range([3]) == "3"
    assert seg_qc._column_range([1, 2, 3]) == "1-3"
    assert seg_qc._column_range([1, 4]) == "1, 4"


# ---------------------------------------------------------------------------
# what the findings refuse to claim
# ---------------------------------------------------------------------------

def _qc(field, flags=(), n_objects=10, severity="ok", diameter=12.0,
        object_type="cell"):
    return FieldQC(field=field, object_type=object_type, n_objects=n_objects,
                   flags=list(flags), metrics={"median_diameter": diameter},
                   severity=severity)


def test_a_blank_field_name_does_not_print_an_empty_location():
    """A location phrase is useful only when there is a location to print."""
    finding = seg_qc._flag_findings([
        _qc("", flags=[FLAG_EMPTY], n_objects=0, severity="fail"),
    ], max_named=3)[0]

    assert finding.fields == ("",)
    assert finding.headline == (
        "1 cell field(s) on this project: no object at all in the field")
    assert "fields :" not in finding.headline


def test_a_plate_with_one_row_has_no_halves_to_compare():
    """Splitting one key into halves gives an empty half, not a comparison."""
    assert seg_qc._halves([]) == ([], [])
    assert seg_qc._halves(["A"]) == ([], [])
    assert seg_qc._halves(["A", "B", "C"]) == (["A"], ["B", "C"])

    one_row = {"A": [10.0] * 8}
    assert seg_qc._axis_step(one_row, ratio=2.0, min_fields=3) is None


def test_a_half_with_too_few_fields_is_not_compared():
    """One field's count is not a median, however far it is from the other."""
    thin = {"A": [1.0], "B": [100.0]}

    assert seg_qc._axis_step(thin, ratio=2.0, min_fields=3) is None


def test_a_half_that_is_simply_empty_is_not_reported_as_a_gradient():
    """A median of zero is an empty half, and the empty flag already said so.

    Dividing by it would give an infinite fold change and a headline claiming
    a ratio nobody can act on.
    """
    empty_half = {"A": [0.0, 0.0, 0.0], "B": [100.0, 110.0, 90.0]}

    assert seg_qc._axis_step(empty_half, ratio=2.0, min_fields=3) is None


# ---------------------------------------------------------------------------
# reading a card back off disk
# ---------------------------------------------------------------------------

def _write_card(path, rows, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = header or ["field", "object_type", "n_objects", "severity",
                        "flags", "note"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# looking for mask stacks and their dates
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------

def test_a_digest_prints_its_verdict_and_headline():
    """``str()`` on a digest is what a log line carries."""
    digest = QCDigest(root="/runs", verdict="fail", headline="plate1 is empty")

    assert str(digest) == "fail: plate1 is empty"


def test_a_source_that_is_not_a_path_contributes_no_project_root(tmp_path):
    """``settings['src']`` arrives as a list that can hold anything.

    A ``None`` or a number in it is a settings-file typo, and dropping it is
    what lets the real entries beside it still be searched.
    """
    roots = seg_qc.qc_roots([None, 17, str(tmp_path)])

    assert roots == (str(tmp_path),)


def test_a_digest_of_cards_with_no_verdict_and_no_findings_is_missing():
    """A card that scored no fields is not a card that passed.

    ``summarize_qc`` reports "empty" for a run that produced no fields, which
    is not a severity; with no findings either there is nothing to grade, and
    "missing" is the answer that sends the user to score the masks.
    """
    card = Scorecard(path="/runs/plate1/qc/segmentation_qc_cell.csv",
                     object_type="cell", field_qcs=[], summary={})
    assert card.verdict == "empty"

    digest = seg_qc._digest_from_cards("/runs", [card])

    assert digest.verdict == "missing"
    assert "Nothing has scored these masks" in digest.headline


def test_the_subhead_counts_the_failures_when_there_are_any():
    """The line under the verdict is a count, and it has to be the real one."""
    card = Scorecard(
        path="/runs/plate1/qc/segmentation_qc_cell.csv", object_type="cell",
        field_qcs=[_qc("plate1_A01_f1", flags=[FLAG_EMPTY], n_objects=0,
                       severity="fail"),
                   _qc("plate1_A02_f1", n_objects=50)],
        summary={"verdict": "fail", "n_fail": 1, "n_warn": 1})

    digest = seg_qc._digest_from_cards("/runs", [card])

    assert digest.verdict == "fail"
    assert "1 of 2 cell field(s) failed and 1 need a look" in digest.subhead


def test_a_card_whose_date_cannot_be_read_is_dated_zero(tmp_path, monkeypatch):
    """A card on a filesystem that will not stat is still readable content.

    Dating it zero means it is treated as older than its masks -- reported as
    out of date rather than as clean, which is the safe direction.
    """
    root = tmp_path / "plate1"
    _write_card(str(root / CARD_DIR / f"{CARD_PREFIX}cell.csv"), [
        {"field": "plate1_A01_f1", "object_type": "cell", "n_objects": "5",
         "severity": "ok", "flags": "", "note": ""},
    ])
    real_stat = os.stat
    seen: set = set()

    def refuse(path, *args, **kwargs):
        # The card is stat-ed once to check it is a file and again to date it.
        # Only the dating call fails, which is the race a card being rewritten
        # underneath the reader produces.
        if str(path).endswith(".csv"):
            if str(path) in seen:
                raise PermissionError(path)
            seen.add(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", refuse)

    digest = seg_qc.read_digest(str(root))

    assert [card.mtime for card in digest.scorecards] == [0.0]


def test_scoring_a_project_can_be_limited_to_one_object_type(tmp_path):
    """Re-scoring every stack on a plate is the expensive path.

    A user who asks about cells must not pay for the pathogen masks beside
    them.
    """
    root = tmp_path / "plate1"
    for object_type in ("cell", "pathogen"):
        folder = root / f"{object_type}_mask_stack"
        folder.mkdir(parents=True)
        np.save(folder / "plate1_A01_f1.npy", _grid_field().astype(np.uint16))

    digest = seg_qc.score_digest(str(root), object_types=["cell"], write=False)

    assert [card.object_type for card in digest.scorecards] == ["cell"]


def test_a_stack_folder_holding_no_masks_contributes_no_card(tmp_path):
    """An empty stack folder is what ``save=False`` leaves behind.

    A card with no rows would be read back later as a plate that scored
    clean, so nothing is recorded for it at all.
    """
    root = tmp_path / "plate1"
    (root / "cell_mask_stack").mkdir(parents=True)

    digest = seg_qc.score_digest(str(root), write=False)

    assert digest.scorecards == []
    assert digest.verdict == "missing"


def test_a_digest_printout_shows_a_card_that_could_not_be_read(tmp_path):
    """The printed digest is what the console shows and what a test reads.

    A card in error has no summary message, and printing an empty one would
    make the failure look like a clean line.
    """
    digest = QCDigest(
        root="/runs", verdict="error", headline="could not be read",
        scorecards=[Scorecard(path="/runs/qc/segmentation_qc_cell.csv",
                              object_type="cell",
                              error="segmentation_qc_cell.csv is not CSV",
                              stale=True)])

    printed = format_digest(digest)

    assert "cell: segmentation_qc_cell.csv is not CSV" in printed
    assert "OUT OF DATE" in printed
