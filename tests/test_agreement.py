"""Multi-annotator agreement — the statistics, checked against arithmetic.

Every κ in here is hand-computed in the test that asserts it, because a
κ implementation that is quietly wrong is indistinguishable from one that
is right until somebody publishes it.

The four traps this suite pins down:

* an unlabelled cell is an **abstention**, not a disagreement — it must
  not touch κ, and it must be counted separately;
* κ is **undefined** when the compared rows carry no variance, and must
  come back as ``nan`` with an explanation rather than as 1.0 or 0.0;
* 95 % raw agreement with κ ≈ 0 is the **prevalence paradox** and both
  numbers have to be reported;
* the module stays **dependency-light** — importing it must not drag in
  torch or cellpose.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spacr.agreement import (
    CONVENTION,
    PNG_KEY,
    PNG_TABLE,
    agreement_report,
    annotation_columns,
    cohens_kappa,
    confusion_matrix,
    disagreements,
    fleiss_kappa,
    format_agreement,
    interpret_kappa,
    kappa_detail,
    load_annotations,
    table_columns,
)


# ---------------------------------------------------------------------------
# Database fixtures — png_list exactly as spacr.utils.filepaths_to_database
# writes it, plus one INTEGER column per annotation pass (which is what
# spacr.qt.annotate_engine.ensure_annotation_column adds).
# ---------------------------------------------------------------------------

_META = ("png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
         "prcfo", "cell_id")


def make_db(path, annotators, rows):
    """Build a ``measurements.db`` with a ``png_list`` table.

    :param path: file to create.
    :param annotators: annotation column names.
    :param rows: list of per-row label tuples, aligned with ``annotators``.
        ``None`` means the annotator abstained (SQL NULL).
    :returns: the path, as a str.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    try:
        meta_sql = ", ".join(f'"{c}" TEXT' for c in _META)
        con.execute(f"CREATE TABLE {PNG_TABLE} ({meta_sql})")
        for name in annotators:
            con.execute(f'ALTER TABLE {PNG_TABLE} ADD COLUMN "{name}" INTEGER')
        placeholders = ", ".join("?" * (len(_META) + len(annotators)))
        payload = []
        for i, labels in enumerate(rows):
            crop = f"/data/plate1/cell_png/plate1_A01_1_{i}.png"
            payload.append((
                crop, os.path.basename(crop), "plate1", "r1", "c1", "f1",
                f"plate1_A01_1_o{i}", f"o{i}", *labels))
        con.executemany(
            f"INSERT INTO {PNG_TABLE} VALUES ({placeholders})", payload)
        con.commit()
    finally:
        con.close()
    return path


@pytest.fixture
def two_annotator_db(tmp_path):
    """50 rows whose 2×2 table is 20 / 5 / 10 / 15 — κ = 0.400 exactly."""
    rows = ([(1, 1)] * 20 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 15)
    return make_db(tmp_path / "run" / "measurements" / "measurements.db",
                   ["alice", "bob"], rows)


@pytest.fixture
def three_annotator_db(tmp_path):
    """4 fully-labelled rows (Fleiss' κ = 1/3) + 2 partial + 1 untouched."""
    rows = [
        (1, 1, 1),          # unanimous
        (1, 1, 2),          # disagreement
        (2, 2, 2),          # unanimous
        (1, 2, 2),          # disagreement
        (1, 1, None),       # partial, but the two who labelled agree
        (None, None, None),  # untouched
        (2, 1, None),       # partial, and the two who labelled disagree
    ]
    return make_db(tmp_path / "run" / "measurements" / "measurements.db",
                   ["alice", "bob", "carol"], rows)


def digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# Cohen's kappa against hand-computed values
# ---------------------------------------------------------------------------

def test_cohens_kappa_matches_the_hand_computed_value():
    """2×2 table  [[20, 5], [10, 15]]  over n = 50.

        p_o = (20 + 15) / 50                       = 0.70
        a marginals = 25/50, 25/50                 = 0.5, 0.5
        b marginals = 30/50, 20/50                 = 0.6, 0.4
        p_e = 0.5*0.6 + 0.5*0.4                    = 0.50
        κ   = (0.70 - 0.50) / (1 - 0.50)           = 0.400
    """
    a = [1] * 25 + [2] * 25
    b = [1] * 20 + [2] * 5 + [1] * 10 + [2] * 15
    detail = kappa_detail(a, b)
    assert detail.n_compared == 50
    assert detail.percent_agreement == pytest.approx(0.70)
    assert detail.expected_agreement == pytest.approx(0.50)
    assert detail.kappa == pytest.approx(0.400)
    assert cohens_kappa(a, b) == pytest.approx(0.400)
    assert detail.interpretation == "fair"      # Landis & Koch: 0.21-0.40
    assert detail.defined is True


def test_confusion_matrix_holds_the_hand_counted_cells():
    a = [1] * 25 + [2] * 25
    b = [1] * 20 + [2] * 5 + [1] * 10 + [2] * 15
    conf = confusion_matrix(a, b)
    assert list(conf.index) == [1, 2] and list(conf.columns) == [1, 2]
    assert conf.loc[1, 1] == 20
    assert conf.loc[1, 2] == 5
    assert conf.loc[2, 1] == 10
    assert conf.loc[2, 2] == 15
    assert int(conf.to_numpy().sum()) == 50


def test_perfect_agreement_is_exactly_one():
    a = [1, 2, 1, 2, 1, 2, 2, 1]
    detail = kappa_detail(a, list(a))
    assert detail.kappa == pytest.approx(1.0)
    assert detail.percent_agreement == pytest.approx(1.0)
    assert detail.n_disagree == 0
    assert detail.interpretation == "almost perfect"


def test_chance_level_agreement_is_zero():
    """[[25, 25], [25, 25]]: p_o = 0.5 and p_e = 0.5, so κ is exactly 0."""
    a = [1] * 50 + [2] * 50
    b = ([1] * 25 + [2] * 25) * 2
    detail = kappa_detail(a, b)
    assert detail.percent_agreement == pytest.approx(0.5)
    assert detail.expected_agreement == pytest.approx(0.5)
    assert detail.kappa == pytest.approx(0.0, abs=1e-12)
    assert detail.defined is True       # zero is a real answer here


def test_total_disagreement_with_varying_marginals_is_negative():
    a = [1, 1, 2, 2, 1, 2]
    b = [2, 2, 1, 1, 2, 1]
    assert cohens_kappa(a, b) < 0
    assert interpret_kappa(cohens_kappa(a, b)).startswith("poor")


def test_kappa_is_symmetric_in_its_arguments():
    a = [1, 1, 2, 2, 1, 2, 1, 1]
    b = [1, 2, 2, 1, 1, 2, 1, 2]
    assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(b, a))


def test_length_mismatch_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError, match="row-aligned"):
        cohens_kappa([1, 2, 1], [1, 2])


def test_labels_outside_the_declared_universe_raise():
    with pytest.raises(ValueError, match="not in the labels"):
        cohens_kappa([1, 2, 3], [1, 2, 3], labels=[1, 2])


def test_string_and_integer_labels_are_the_same_class():
    """A hand-edited TEXT column must still agree with an INTEGER one."""
    assert cohens_kappa(["1", "2", "1", "2"], [1, 2, 1, 2]) == pytest.approx(1.0)


def test_label_normalisation_covers_every_way_sqlite_hands_back_a_class():
    """One class must not split in two because of how it was stored.

    A BOOLEAN 1, an INTEGER 1, a REAL 1.0, a TEXT "1" and a BLOB b"1" are
    all the same annotation; ``NaT``/``None``/``""`` are all abstentions;
    and anything genuinely alien is passed through untouched rather than
    guessed at.
    """
    from spacr.agreement import _scalar_label

    assert _scalar_label(True) == 1
    assert _scalar_label(np.bool_(False)) == 0
    assert _scalar_label(2.0) == 2
    assert _scalar_label(np.int64(3)) == 3
    assert _scalar_label("2") == 2
    assert _scalar_label(" 2 ") == 2
    assert _scalar_label("2.0") == 2
    assert _scalar_label(b"2") == 2
    assert _scalar_label("hi") == "hi"
    # Non-integral values stay as they are — never silently rounded.
    assert _scalar_label(1.5) == pytest.approx(1.5)
    assert _scalar_label("1.5") == pytest.approx(1.5)
    # Abstentions
    for missing in (None, float("nan"), "", "   ", pd.NaT):
        assert _scalar_label(missing) is None
    # Something pandas cannot answer "is this NA?" about comes back as-is.
    alien = np.array([1, 2])
    assert _scalar_label(alien) is alien


def test_non_numeric_labels_work_too():
    a = ["pos", "neg", "pos", "neg", "pos", "neg"]
    b = ["pos", "neg", "neg", "neg", "pos", "pos"]
    detail = kappa_detail(a, b)
    assert list(detail.labels) == ["neg", "pos"]
    assert detail.n_compared == 6
    assert detail.percent_agreement == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# The no-variance trap: nan with an explanation, never 0.0 or 1.0
# ---------------------------------------------------------------------------

def test_no_variance_returns_nan_not_one():
    """Both annotators called all 40 rows class 1. p_e = 1, so κ = 0/0.

    This is the normal state of a screen where almost everything is
    negative. Reporting κ = 1.0 ("perfect!") would be a lie, and 0.0
    ("useless annotators!") would be a different lie.
    """
    detail = kappa_detail([1] * 40, [1] * 40)
    assert math.isnan(detail.kappa)
    assert math.isnan(cohens_kappa([1] * 40, [1] * 40))
    assert detail.defined is False
    assert detail.interpretation == "undefined"
    # ...and the raw agreement, which IS meaningful, is still reported.
    assert detail.percent_agreement == pytest.approx(1.0)
    assert detail.n_compared == 40
    note = detail.note.lower()
    assert "undefined" in note
    assert "no variance" in note or "denominator" in note
    assert "100" in note or "1.0" in note


def test_two_constant_annotators_who_never_agree_is_also_nan():
    detail = kappa_detail([1] * 10, [2] * 10)
    assert math.isnan(detail.kappa)
    assert detail.percent_agreement == pytest.approx(0.0)
    assert "one class" in detail.note


def test_one_constant_annotator_returns_nan_rather_than_a_misleading_zero():
    """When a rater never varies, κ is identically 0 whatever the data.

    a labels everything 1; b labels 38 of 40 the same way. Raw agreement
    is 95 %, and the textbook κ is exactly 0.0 — a number about a's
    behaviour, not about their agreement. We return nan and say why.
    """
    a = [1] * 40
    b = [1] * 38 + [2, 2]
    detail = kappa_detail(a, b, name_a="alice", name_b="bob")
    assert math.isnan(detail.kappa)
    assert detail.percent_agreement == pytest.approx(0.95)
    assert "degenerate" in detail.note
    assert "alice" in detail.note
    assert "identically 0" in detail.note


def test_no_overlapping_rows_is_nan_with_the_abstention_counts():
    """Two annotators who worked on disjoint halves cannot be compared."""
    a = [1, 2, 1, None, None, None]
    b = [None, None, None, 1, 2, 2]
    detail = kappa_detail(a, b)
    assert math.isnan(detail.kappa)
    assert detail.n_compared == 0
    assert detail.n_abstained == 6
    assert "both annotators committed" in detail.note


# ---------------------------------------------------------------------------
# Abstentions
# ---------------------------------------------------------------------------

def test_nulls_are_excluded_and_do_not_count_as_disagreement():
    """The same 8 shared rows, with and without a trail of NULLs.

    Adding rows only one annotator reached must not move κ by a hair —
    otherwise κ reports how far behind the slower annotator is.
    """
    a_core = [1, 1, 2, 2, 1, 2, 1, 2]
    b_core = [1, 2, 2, 2, 1, 2, 1, 1]
    baseline = kappa_detail(a_core, b_core)

    a_full = a_core + [1, 2, 1, None, None]
    b_full = b_core + [None, None, None, 1, 2]
    padded = kappa_detail(a_full, b_full)

    assert padded.kappa == pytest.approx(baseline.kappa)
    assert padded.n_compared == baseline.n_compared == 8
    assert padded.n_agree == baseline.n_agree
    assert padded.n_disagree == baseline.n_disagree == 2
    # The five one-sided rows are counted, not silently dropped...
    assert padded.n_abstained == 5
    # ...and they are NOT disagreements.
    assert padded.n_disagree == 2


def test_nan_empty_string_and_none_all_count_as_abstentions():
    a = [1, 2, None, float("nan"), "", 1]
    b = [1, 2, 1, 1, 1, 1]
    detail = kappa_detail(a, b)
    assert detail.n_compared == 3
    assert detail.n_abstained == 3
    assert detail.percent_agreement == pytest.approx(1.0)


def test_rows_neither_annotator_reached_are_counted_apart():
    detail = kappa_detail([1, 2, None, None], [1, 2, 1, None])
    assert detail.n_compared == 2
    assert detail.n_abstained == 1
    assert detail.n_neither == 1


def test_zero_is_a_real_class_unless_asked_otherwise():
    """The Qt annotator clears a label by writing NULL, so 0 is a class.

    Legacy Tk databases used 0 for "not looked at"; those callers pass
    ``missing_values=(0,)`` and get abstentions instead.
    """
    a = [0, 0, 1, 1, 0, 1]
    b = [0, 1, 1, 1, 0, 1]
    assert kappa_detail(a, b).n_compared == 6
    legacy = kappa_detail(a, b, missing_values=(0,))
    assert legacy.n_compared == 3
    assert legacy.n_abstained == 1
    assert legacy.n_neither == 2


# ---------------------------------------------------------------------------
# The prevalence paradox
# ---------------------------------------------------------------------------

def test_prevalence_paradox_reports_high_agreement_and_near_zero_kappa():
    """[[95, 3], [2, 0]] over n = 100.

        p_o = (95 + 0) / 100                        = 0.9500
        a marginals = 98/100, 2/100
        b marginals = 97/100, 3/100
        p_e = 0.98*0.97 + 0.02*0.03                 = 0.9512
        κ   = (0.9500 - 0.9512) / (1 - 0.9512)      = -0.0246

    95 % of the crops agree and κ is essentially zero. Both are true;
    reporting either alone misleads.
    """
    a = [1] * 98 + [2] * 2
    b = [1] * 95 + [2] * 3 + [1] * 2
    detail = kappa_detail(a, b)
    assert detail.percent_agreement == pytest.approx(0.95)
    assert detail.expected_agreement == pytest.approx(0.9512)
    assert detail.kappa == pytest.approx(-0.0246, abs=1e-4)
    assert abs(detail.kappa) < 0.05
    # The number alone would be read as "the annotators are useless".
    assert "prevalence paradox" in detail.note.lower()
    assert "95.0%" in detail.note


def test_prevalence_paradox_surfaces_in_the_report(tmp_path):
    rows = ([(1, 1)] * 95 + [(1, 2)] * 3 + [(2, 1)] * 2)
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["alice", "bob"], rows)
    report = agreement_report(db, ["alice", "bob"])
    assert report.percent_agreement == pytest.approx(0.95)
    assert report.overall_kappa == pytest.approx(-0.0246, abs=1e-4)
    assert any("prevalence paradox" in w.lower() for w in report.warnings)
    text = format_agreement(report)
    assert "95.0%" in text
    assert "-0.025" in text


# ---------------------------------------------------------------------------
# Interpretation bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kappa,band", [
    (-0.30, "poor (no better than chance)"),
    (0.00, "poor (no better than chance)"),
    (0.10, "slight"),
    (0.35, "fair"),
    (0.55, "moderate"),
    (0.75, "substantial"),
    (0.95, "almost perfect"),
    (1.00, "almost perfect"),
])
def test_landis_koch_bands(kappa, band):
    assert interpret_kappa(kappa) == band


def test_interpretation_of_an_undefined_kappa_is_undefined():
    assert interpret_kappa(float("nan")) == "undefined"
    assert interpret_kappa(None) == "undefined"
    assert interpret_kappa("not a number") == "undefined"


def test_a_kappa_above_the_top_band_still_gets_the_top_band():
    """κ cannot exceed 1, but a caller passing 2.0 gets a band, not a crash."""
    assert interpret_kappa(2.0) == "almost perfect"


def test_the_text_formatters_never_invent_a_number():
    from spacr.agreement import _fmt_kappa, _fmt_pct

    assert _fmt_kappa(0.5) == "+0.500"
    assert _fmt_kappa(float("nan")) == "undefined"
    assert _fmt_kappa("wat") == "undefined"
    assert _fmt_kappa(None) == "undefined"
    assert _fmt_pct(0.951) == "95.1%"
    assert _fmt_pct(float("nan")) == "n/a"
    assert _fmt_pct("wat") == "n/a"


def test_the_bands_are_labelled_as_a_convention():
    assert "convention" in CONVENTION.lower()
    assert "landis" in CONVENTION.lower()
    assert "not a law" in CONVENTION.lower()


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------

def test_fleiss_kappa_matches_a_hand_computed_value():
    """4 subjects, 2 annotators, counts [[2,0], [2,0], [0,2], [1,1]].

        p_1 = (2+2+0+1) / 8 = 5/8      p_2 = 3/8
        P_i = (Σ n_ij² - n) / (n(n-1)) = 1, 1, 1, 0
        P̄   = 3/4
        P̄e  = (5/8)² + (3/8)² = 34/64 = 0.53125
        κ   = (0.75 - 0.53125) / (1 - 0.53125) = 7/15 = 0.466667
    """
    matrix = [[2, 0], [2, 0], [0, 2], [1, 1]]
    assert fleiss_kappa(matrix) == pytest.approx(7 / 15)


def test_fleiss_kappa_matches_the_published_five_category_example():
    """Fleiss (1971) worked example: 10 subjects, 14 raters, κ = 0.2099."""
    matrix = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    assert fleiss_kappa(matrix) == pytest.approx(0.2099, abs=5e-4)


def test_fleiss_generalises_scotts_pi_and_so_differs_from_cohen():
    """Same two-rater data, two estimators — the difference is the point.

    Fleiss pools the raters into one marginal (Scott's π); Cohen gives
    each rater its own. Documented, and asserted so it stays documented.
    """
    matrix = [[2, 0], [2, 0], [0, 2], [1, 1]]
    a = [1, 1, 2, 1]
    b = [1, 1, 2, 2]
    assert fleiss_kappa(matrix) == pytest.approx(7 / 15)
    assert cohens_kappa(a, b) == pytest.approx(0.5)


def test_fleiss_perfect_agreement_is_one():
    assert fleiss_kappa([[3, 0], [0, 3], [3, 0], [0, 3]]) == pytest.approx(1.0)


def test_fleiss_without_variance_is_nan():
    """Every rater put every subject in one category — p_e = 1, κ = 0/0."""
    assert math.isnan(fleiss_kappa([[3, 0], [3, 0], [3, 0]]))


@pytest.mark.parametrize("bad,match", [
    ([[2, 0], [1, 0]], "same number of annotators"),
    ([[1, 0], [1, 0]], "at least 2 annotators"),
    ([[-1, 3], [2, 0]], "non-negative"),
    ([[1.5, 0.5], [1.0, 1.0]], "whole numbers"),
    ([1, 2, 3], "2-D"),
    ([], "2-D"),
])
def test_fleiss_rejects_malformed_matrices(bad, match):
    with pytest.raises(ValueError, match=match):
        fleiss_kappa(bad)


# ---------------------------------------------------------------------------
# Reading a real database
# ---------------------------------------------------------------------------

def test_table_columns_and_annotation_column_discovery(two_annotator_db):
    cols = table_columns(two_annotator_db)
    assert cols[0] == PNG_KEY
    assert "alice" in cols and "bob" in cols
    # Metadata columns are not annotation passes.
    assert annotation_columns(two_annotator_db) == ["alice", "bob"]


def test_annotation_column_discovery_ignores_wide_and_empty_columns(tmp_path):
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["alice", "unused", "score"],
                 [(1, None, i) for i in range(40)])
    found = annotation_columns(db)
    assert "alice" in found
    assert "unused" not in found, "an all-NULL column is not an annotator"
    assert "score" not in found, "40 distinct values is a measurement"


def test_missing_png_list_table_is_a_clear_error(tmp_path):
    path = tmp_path / "empty.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cell (prc TEXT)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="png_list"):
        table_columns(str(path))


def test_missing_database_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        table_columns(str(tmp_path / "nope.db"))
    with pytest.raises(ValueError, match="No database path"):
        table_columns("")


def test_load_annotations_normalises_nulls_to_none(three_annotator_db):
    df = load_annotations(three_annotator_db, ["alice", "carol"])
    assert list(df.columns) == [PNG_KEY, "alice", "carol"]
    assert len(df) == 7
    assert df["alice"].iloc[0] == 1
    assert df["carol"].iloc[4] is None
    assert df["alice"].iloc[5] is None
    # ints stay ints — a float 1.0 would not compare equal to a TEXT "1"
    assert isinstance(df["alice"].iloc[0], int)


def test_load_annotations_rejects_unknown_columns(two_annotator_db):
    with pytest.raises(ValueError, match="no column"):
        load_annotations(two_annotator_db, ["alice", "nobody"])


def test_a_png_list_without_the_key_column_is_a_clear_error(tmp_path):
    """Without png_path there is no way to line two annotators up row by row."""
    path = tmp_path / "keyless.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE png_list (alice INTEGER, bob INTEGER)")
    con.execute("INSERT INTO png_list VALUES (1, 1)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="no 'png_path' column"):
        load_annotations(str(path), ["alice", "bob"])


def test_report_on_two_annotators_uses_cohens_kappa(two_annotator_db):
    report = agreement_report(two_annotator_db, ["alice", "bob"])
    assert report.overall_method == "Cohen's κ"
    assert report.overall_kappa == pytest.approx(0.400)
    assert report.n_annotators == 2
    assert report.n_rows == 50
    assert report.n_complete == 50
    assert report.n_partial == 0
    assert report.n_unlabelled == 0
    assert report.n_disagreements == 15          # 5 + 10 off-diagonal
    assert report.percent_agreement == pytest.approx(0.70)
    assert len(report.pairs) == 1
    pair = report.pair("bob", "alice")           # order-insensitive lookup
    assert pair is not None and pair.kappa == pytest.approx(0.400)
    assert report.pair("alice", "nobody") is None
    assert report.convention == CONVENTION


def test_report_per_class_kappa_equals_the_overall_for_a_binary_problem(
        two_annotator_db):
    """One-vs-rest on a 2×2 table is the same table relabelled, so both
    per-class κ values must equal the overall κ. A cheap invariant that
    catches a transposed or mis-indexed per-class calculation."""
    report = agreement_report(two_annotator_db, ["alice", "bob"])
    per_class = report.per_class.set_index("label")
    assert list(per_class.index) == [1, 2]
    for label in (1, 2):
        assert per_class.loc[label, "kappa"] == pytest.approx(0.400)
    assert per_class.loc[1, "n_unanimous"] == 20
    assert per_class.loc[2, "n_unanimous"] == 15
    assert per_class.loc[1, "prevalence"] == pytest.approx(55 / 100)


def test_report_on_three_annotators_uses_fleiss(three_annotator_db):
    """The 4 fully-labelled rows are [[3,0], [2,1], [0,3], [1,2]]:

        p_1 = 6/12 = 0.5,  p_2 = 0.5
        P_i = 1, 1/3, 1, 1/3   ->  P̄ = 2/3
        P̄e  = 0.25 + 0.25      = 0.5
        κ   = (2/3 - 1/2) / (1 - 1/2) = 1/3
    """
    report = agreement_report(three_annotator_db, ["alice", "bob", "carol"])
    assert report.overall_method == "Fleiss' κ"
    assert report.overall_kappa == pytest.approx(1 / 3)
    assert report.n_rows == 7
    assert report.n_complete == 4
    assert report.n_partial == 2
    assert report.n_unlabelled == 1
    assert report.percent_agreement == pytest.approx(0.5)
    assert "Fleiss" in report.overall_note


def test_report_pairwise_kappas_are_each_hand_computable(three_annotator_db):
    """Each pair uses ITS OWN shared rows, which is why the three κ values
    differ and why n_compared differs from n_complete.

        alice vs bob:   6 shared rows, p_o = 4/6, p_e = 5/9   -> κ = 0.25
        alice vs carol: 4 shared rows, p_o = 1/2, p_e = 6/16  -> κ = 0.20
        bob   vs carol: 4 shared rows, p_o = 3/4, p_e = 1/2   -> κ = 0.50
    """
    report = agreement_report(three_annotator_db, ["alice", "bob", "carol"])
    assert len(report.pairs) == 3
    ab = report.pair("alice", "bob")
    ac = report.pair("alice", "carol")
    bc = report.pair("bob", "carol")
    assert (ab.n_compared, ab.n_abstained, ab.n_neither) == (6, 0, 1)
    assert (ac.n_compared, ac.n_abstained, ac.n_neither) == (4, 2, 1)
    assert (bc.n_compared, bc.n_abstained, bc.n_neither) == (4, 2, 1)
    assert ab.kappa == pytest.approx(0.25)
    assert ac.kappa == pytest.approx(0.20)
    assert bc.kappa == pytest.approx(0.50)


def test_report_counts_partial_rows_as_abstentions_not_disagreements(
        three_annotator_db):
    """Row 5 is (1, 1, NULL): two annotators agreed, one has not looked.

    It must land in n_partial and stay out of n_disagreements. Row 7 is
    (2, 1, NULL) — also partial, but the two who committed disagree, so
    it IS a disagreement.
    """
    report = agreement_report(three_annotator_db, ["alice", "bob", "carol"])
    assert report.n_partial == 2
    assert report.n_disagreements == 3        # rows 2, 4 and 7
    assert any("abstentions, not disagreements" in w for w in report.warnings)


def test_report_with_no_variance_says_so_instead_of_claiming_perfection(
        tmp_path):
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["alice", "bob"], [(1, 1)] * 40)
    report = agreement_report(db, ["alice", "bob"])
    assert math.isnan(report.overall_kappa)
    assert report.defined is False
    assert report.interpretation == "undefined"
    assert report.percent_agreement == pytest.approx(1.0)
    assert report.n_disagreements == 0
    text = format_agreement(report)
    assert "undefined" in text
    assert "100.0%" in text


def test_report_with_three_annotators_and_no_variance_is_nan(tmp_path):
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["a1", "a2", "a3"], [(2, 2, 2)] * 35)
    report = agreement_report(db, ["a1", "a2", "a3"])
    assert math.isnan(report.overall_kappa)
    assert "undefined" in report.overall_note
    assert "denominator" in report.overall_note


def test_report_with_three_annotators_and_no_shared_rows(tmp_path):
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["a1", "a2", "a3"],
                 [(1, 1, None), (2, 2, None), (1, 2, None), (2, 1, None)])
    report = agreement_report(db, ["a1", "a2", "a3"])
    assert report.n_complete == 0
    assert math.isnan(report.overall_kappa)
    assert math.isnan(report.percent_agreement)
    assert "nothing to score" in report.overall_note
    # The pairwise numbers still work — a1 vs a2 has 4 shared rows.
    assert report.pair("a1", "a2").n_compared == 4
    assert format_agreement(report)      # renders without blowing up


def test_report_needs_at_least_two_columns(two_annotator_db):
    with pytest.raises(ValueError, match="at least two"):
        agreement_report(two_annotator_db, ["alice"])
    with pytest.raises(ValueError, match="at least two"):
        agreement_report(two_annotator_db, ["alice", "alice"])
    with pytest.raises(ValueError, match="at least two"):
        agreement_report(two_annotator_db, [])


def test_report_rejects_a_column_that_is_not_there(two_annotator_db):
    with pytest.raises(ValueError, match="no column"):
        agreement_report(two_annotator_db, ["alice", "ghost"])


def test_report_honours_an_explicit_label_universe(two_annotator_db):
    report = agreement_report(two_annotator_db, ["alice", "bob"],
                              labels=[1, 2, 3])
    assert report.labels == [1, 2, 3]
    assert list(report.pairs[0].confusion.index) == [1, 2, 3]
    assert report.overall_kappa == pytest.approx(0.400)
    # Class 3 was never used by anybody: its one-vs-rest κ is undefined.
    per_class = report.per_class.set_index("label")
    assert math.isnan(per_class.loc[3, "kappa"])
    assert per_class.loc[3, "n_any"] == 0


def test_report_reads_the_database_read_only(two_annotator_db):
    before = digest(two_annotator_db)
    siblings = sorted(os.listdir(os.path.dirname(two_annotator_db)))
    agreement_report(two_annotator_db, ["alice", "bob"])
    disagreements(two_annotator_db, ["alice", "bob"])
    annotation_columns(two_annotator_db)
    assert digest(two_annotator_db) == before, "the database changed on disk"
    assert sorted(os.listdir(os.path.dirname(two_annotator_db))) == siblings, \
        "a -wal/-journal side file appeared — the open was not read-only"


def test_kappa_table_is_renderable(three_annotator_db):
    report = agreement_report(three_annotator_db, ["alice", "bob", "carol"])
    table = report.kappa_table()
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 3
    assert set(table.columns) >= {"annotator_a", "annotator_b", "kappa",
                                  "percent_agreement", "n_compared",
                                  "n_abstained", "interpretation"}


# ---------------------------------------------------------------------------
# Disagreement review
# ---------------------------------------------------------------------------

def test_disagreements_returns_exactly_the_differing_rows(three_annotator_db):
    """Rows 2 (1,1,2), 4 (1,2,2) and 7 (2,1,NULL) differ; nothing else does.

    Row 5 is (1, 1, NULL) — an abstention on top of agreement, so it must
    NOT be in the review queue.
    """
    rows = disagreements(three_annotator_db, ["alice", "bob", "carol"])
    assert list(rows.columns) == [PNG_KEY, "alice", "bob", "carol",
                                  "n_labelled", "n_classes"]
    assert len(rows) == 3
    assert [os.path.basename(p) for p in rows[PNG_KEY]] == [
        "plate1_A01_1_1.png", "plate1_A01_1_3.png", "plate1_A01_1_6.png"]
    assert list(rows["alice"]) == [1, 1, 2]
    assert list(rows["bob"]) == [1, 2, 1]
    assert list(rows["carol"]) == [2, 2, None]
    assert list(rows["n_labelled"]) == [3, 3, 2]
    assert list(rows["n_classes"]) == [2, 2, 2]


def test_disagreements_never_include_a_row_only_one_person_labelled(tmp_path):
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["alice", "bob"],
                 [(1, None), (None, 2), (1, 1), (None, None), (1, 2)])
    rows = disagreements(db, ["alice", "bob"])
    assert len(rows) == 1
    assert rows[PNG_KEY].iloc[0].endswith("_4.png")


def test_disagreements_complete_only_drops_partially_labelled_rows(
        three_annotator_db):
    rows = disagreements(three_annotator_db, ["alice", "bob", "carol"],
                         complete_only=True)
    assert len(rows) == 2
    assert list(rows["carol"]) == [2, 2]


def test_disagreements_limit_caps_the_review_queue(two_annotator_db):
    everything = disagreements(two_annotator_db, ["alice", "bob"])
    assert len(everything) == 15
    capped = disagreements(two_annotator_db, ["alice", "bob"], limit=4)
    assert len(capped) == 4
    assert list(capped[PNG_KEY]) == list(everything[PNG_KEY][:4])


def test_disagreements_on_a_database_with_none_returns_an_empty_frame(tmp_path):
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["alice", "bob"], [(1, 1), (2, 2), (1, 1)])
    rows = disagreements(db, ["alice", "bob"])
    assert len(rows) == 0
    assert list(rows.columns) == [PNG_KEY, "alice", "bob",
                                  "n_labelled", "n_classes"]


def test_disagreements_needs_two_columns(two_annotator_db):
    with pytest.raises(ValueError, match="at least two"):
        disagreements(two_annotator_db, ["alice"])


def test_disagreement_count_in_the_report_matches_the_review_list(
        three_annotator_db):
    report = agreement_report(three_annotator_db, ["alice", "bob", "carol"])
    rows = disagreements(three_annotator_db, ["alice", "bob", "carol"])
    assert report.n_disagreements == len(rows) == 3


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def test_format_agreement_shows_kappa_next_to_raw_agreement(two_annotator_db):
    text = format_agreement(agreement_report(two_annotator_db,
                                             ["alice", "bob"]))
    assert "alice" in text and "bob" in text
    assert "+0.400" in text
    assert "70.0%" in text          # raw agreement, next to κ
    assert "Confusion matrix" in text
    assert "Per class" in text
    assert CONVENTION in text
    assert not text.endswith("\n")


def test_format_agreement_of_three_annotators_lists_every_pair(
        three_annotator_db):
    text = format_agreement(agreement_report(three_annotator_db,
                                             ["alice", "bob", "carol"]))
    assert text.count("alice") >= 2
    assert "Fleiss" in text
    assert "+0.333" in text
    # A three-way report has no single confusion matrix to print.
    assert "Confusion matrix" not in text


def test_pair_agreement_str_is_readable(two_annotator_db):
    report = agreement_report(two_annotator_db, ["alice", "bob"])
    text = str(report.pairs[0])
    assert "alice vs bob" in text
    assert "+0.400" in text
    assert "70.0%" in text


def test_pair_str_says_undefined_when_it_is():
    assert "undefined" in str(kappa_detail([1] * 5, [1] * 5))


# ---------------------------------------------------------------------------
# Dependency weight
# ---------------------------------------------------------------------------

def test_importing_agreement_pulls_in_neither_torch_nor_cellpose():
    """The Qt screen must not pay a 4-second torch import to show a κ.

    Measured as "modules added by the import" rather than "torch is
    absent", so a conftest or sitecustomize that pre-imports torch for
    unrelated reasons cannot make this pass or fail by accident.
    """
    code = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import spacr.agreement\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    added = json.loads(proc.stdout.strip().splitlines()[-1])
    tops = {name.split(".")[0] for name in added}
    for heavy in ("torch", "cellpose", "torchvision", "cv2", "matplotlib"):
        assert heavy not in tops, (
            f"importing spacr.agreement dragged in {heavy!r}; "
            f"it must stay pandas/numpy/sqlite only")


def test_agreement_module_does_not_reference_torch_or_cellpose():
    import spacr.agreement as mod
    source = open(mod.__file__, encoding="utf-8").read().lower()
    assert "import torch" not in source
    assert "import cellpose" not in source


def test_report_scales_without_pandas_row_iteration(tmp_path):
    """A 5 000-row png_list must not take pathological time to score."""
    rows = [((i % 3) or None, (i % 4) or None) for i in range(5000)]
    db = make_db(tmp_path / "run" / "measurements" / "measurements.db",
                 ["alice", "bob"], rows)
    report = agreement_report(db, ["alice", "bob"])
    assert report.n_rows == 5000
    assert report.n_complete + report.n_partial + report.n_unlabelled == 5000
    assert np.isfinite(report.overall_kappa)
