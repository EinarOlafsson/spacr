"""Barcode-mapping QC and the target-driven threshold sweep.

The point of :mod:`spacr.sequencing_qc` is that a number nobody can defend
(2% of a well's reads, read off a histogram once) is replaced by a number
derived from a stated biological target. A test suite for it therefore has
to do more than call the functions: it has to build a library whose
gRNAs-per-well distribution is KNOWN, and then check that the derived
threshold really delivers that distribution, that the sweep is monotone
everywhere the maths says it must be, that collisions are counted the way
a person counting by hand would count them, and that the starved wells
identified are exactly the starved wells planted.

The end-to-end test goes through the project's synthetic sequencing
convention — a paired Illumina FASTQ carrying three barcodes (column,
gRNA, row) plus the three barcode reference tables — so the QC is fed by
the real read path rather than by a count table written by the test.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spacr.sequencing_qc as QC


# ---------------------------------------------------------------------------
# Library builders — the ground truth every assertion below is measured against
# ---------------------------------------------------------------------------

def make_library(n_rows=8, n_cols=12, real_per_well=4, n_grnas=240,
                 depth=20_000, junk_per_well=30, junk_share=0.06,
                 seed=0, starved_wells=(), starved_depth=200,
                 row_depth_factor=None):
    """Build a count table with a known number of real gRNAs per well.

    Each well gets exactly ``real_per_well`` guides splitting
    ``1 - junk_share`` of its reads, plus ``junk_per_well`` bleed-through
    guides splitting the rest — the shape a real pooled screen has, and
    the shape a threshold is supposed to cut between.

    :param starved_wells: ``(row, column)`` pairs to sequence at
        ``starved_depth`` instead of ``depth``.
    :param row_depth_factor: optional ``{row: factor}`` multiplier on the
        read depth, for planting a position effect.
    :returns: DataFrame in ``unique_combinations.csv`` shape.
    """
    rng = np.random.default_rng(seed)
    starved = {(str(r), str(c)) for r, c in starved_wells}
    factors = dict(row_depth_factor or {})
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            row_id, col_id = f"r{r + 1}", f"c{c + 1}"
            reads = starved_depth if (row_id, col_id) in starved else depth
            reads = int(reads * factors.get(row_id, 1.0))
            real = rng.choice(n_grnas, size=real_per_well, replace=False)
            weights = rng.dirichlet(np.full(real_per_well, 6.0))
            for guide, share in zip(real, weights):
                rows.append({"rowID": row_id, "columnID": col_id,
                             "grna_name": f"g{guide:04d}",
                             "count": int(round(reads * (1 - junk_share)
                                                * share))})
            pool = [g for g in rng.choice(n_grnas, size=junk_per_well,
                                          replace=False) if g not in real]
            junk_weights = rng.dirichlet(np.full(len(pool), 2.0))
            for guide, share in zip(pool, junk_weights):
                n = int(round(reads * junk_share * share))
                if n > 0:
                    rows.append({"rowID": row_id, "columnID": col_id,
                                 "grna_name": f"g{guide:04d}", "count": n})
    return pd.DataFrame(rows)


def hand_table(spec):
    """Build a count table straight from ``{(row, col): {grna: count}}``."""
    rows = [{"rowID": row, "columnID": col, "grna_name": grna, "count": count}
            for (row, col), guides in spec.items()
            for grna, count in guides.items()]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def library():
    """A 96-well library carrying exactly 4 real gRNAs per well."""
    return QC.load_count_table(make_library(real_per_well=4, seed=7))


# ---------------------------------------------------------------------------
# 1 — the derived threshold really achieves the stated target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("planted", [2, 3, 4, 6])
def test_derived_threshold_recovers_the_planted_grnas_per_well(planted):
    """State the target, get back the cutoff that delivers it.

    The library carries exactly ``planted`` real guides per well. Asking
    for that number must produce a threshold at which the median well
    holds exactly that many — not approximately, because the count is an
    integer and the derivation searches every attainable cutoff.
    """
    counts = QC.load_count_table(
        make_library(real_per_well=planted, seed=11 + planted))
    choice = QC.derive_threshold(counts, planted)

    assert choice.attainable
    assert choice.achieved == pytest.approx(float(planted))

    # And the threshold, applied as spacr.ml.process_reads applies it,
    # reproduces the same distribution independently of the machinery
    # that derived it.
    kept = counts[counts["fraction"] >= choice.threshold]
    per_well = kept.groupby("prc")["grna"].nunique()
    assert per_well.median() == pytest.approx(float(planted))
    # Every well keeps its real guides: nothing is lost to the cut.
    assert (per_well == planted).mean() > 0.9


def test_a_higher_target_needs_a_lower_threshold(library):
    """More guides per well can only be bought with a looser cutoff."""
    thresholds = [QC.derive_threshold(library, t).threshold
                  for t in (2, 3, 4, 5, 6)]
    assert thresholds == sorted(thresholds, reverse=True), thresholds


def test_the_derived_threshold_is_the_middle_of_its_plateau(library):
    """Both edges of the reported range give the same answer as the middle.

    That is what makes the middle worth quoting: the number is not
    balanced on the edge of an observation where a re-sequenced run would
    tip guides across it.
    """
    choice = QC.derive_threshold(library, 4)
    assert choice.interval_low < choice.threshold <= choice.interval_high
    fractions = QC.WellFractions(library)
    for edge in (choice.interval_low * (1 + 1e-9), choice.threshold,
                 choice.interval_high):
        assert fractions.statistic_at(edge, "median")[0] == \
            pytest.approx(choice.achieved)
    # ...and one step past the top edge, the answer changes. Otherwise
    # "plateau" would just mean "any old range".
    beyond = choice.interval_high * (1 + 1e-9)
    assert fractions.statistic_at(beyond, "median")[0] < choice.achieved


def test_a_target_no_threshold_can_reach_is_reported_not_faked():
    """A target above what the library holds must say so, not return a number."""
    counts = QC.load_count_table(
        make_library(n_rows=2, n_cols=2, real_per_well=2, junk_per_well=3,
                     n_grnas=20, seed=3))
    most = counts.groupby("prc")["grna"].nunique().median()
    choice = QC.derive_threshold(counts, most + 5)
    assert choice.attainable is False
    assert choice.achieved == pytest.approx(most)

    sweep = QC.threshold_sweep(counts, QC.sweep_grid(choice.threshold),
                               most + 5)
    text = QC.recommend_threshold(sweep, choice)
    assert "WARNING" in text
    assert "out of reach" in text


def test_a_target_between_two_steps_takes_whichever_lands_closer():
    """The median moves in steps, so most targets land between two of them.

    Ten identical wells, each holding one dominant guide at 97% and three
    at 1%. There are exactly two attainable answers: 4 gRNAs per well
    (cutoff at or below 1%) and 1 (anything above it). A target of 3.6
    must take the 4; a target of 1.4 must take the 1, even though 4 is
    the one that *meets* it; and an exact tie at 2.5 must take the 4,
    because a well short of its guides has lost power that no later step
    recovers.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", f"c{i + 1}"): {"g0": 970, "g1": 10, "g2": 10, "g3": 10}
        for i in range(10)}))
    attainable = {QC.derive_threshold(counts, t).achieved for t in (1, 4)}
    assert attainable == {1.0, 4.0}

    assert QC.derive_threshold(counts, 3.6).achieved == 4.0
    assert QC.derive_threshold(counts, 1.4).achieved == 1.0
    assert QC.derive_threshold(counts, 2.5).achieved == 4.0


def test_the_mean_statistic_is_offered_and_honoured():
    """`target_statistic='mean'` targets the mean, not the median."""
    counts = QC.load_count_table(make_library(real_per_well=4, seed=5))
    choice = QC.derive_threshold(counts, 4, statistic="mean")
    kept = counts[counts["fraction"] >= choice.threshold]
    per_well = kept.groupby("prc")["grna"].nunique()
    # Wells that lose every guide leave the numerator but stay in the
    # population the derivation used, so compare against that population.
    total = counts["prc"].nunique()
    assert per_well.sum() / total == pytest.approx(choice.achieved, abs=0.5)
    assert abs(choice.achieved - 4) <= 0.5


def test_the_target_must_be_positive_and_the_statistic_known(library):
    with pytest.raises(ValueError, match="must be positive"):
        QC.derive_threshold(library, 0)
    with pytest.raises(ValueError, match="must be positive"):
        QC.derive_threshold(library, -2)
    with pytest.raises(ValueError, match="median.*mean"):
        QC.derive_threshold(library, 4, statistic="mode")


# ---------------------------------------------------------------------------
# 2 — the sweep is monotone everywhere the maths says it must be
# ---------------------------------------------------------------------------

#: Columns that can only fall as the threshold rises. Each well's surviving
#: gRNA count is non-increasing in the cutoff, and the well population is
#: fixed, so every order statistic and every count over that population
#: inherits the property. A break here means the population is being
#: resized behind the statistic.
MONOTONE_COLUMNS = ("grnas_per_well", "wells_retained", "well_retention",
                    "wells_over_budget", "collision_rate", "n_calls",
                    "reads_retained")


def test_the_sweep_is_monotone_in_every_column_that_must_be(library):
    choice = QC.derive_threshold(library, 4)
    sweep = QC.threshold_sweep(library, QC.sweep_grid(choice.threshold,
                                                      span=16, points=60), 4)
    assert sweep["threshold"].is_monotonic_increasing
    for column in MONOTONE_COLUMNS:
        values = sweep[column].to_numpy(float)
        assert np.all(np.diff(values) <= 1e-12), (
            f"{column} rose with the threshold: "
            f"{values[np.argmax(np.diff(values) > 0)]}")


def test_the_retained_only_column_really_does_rise_as_wells_drop_out():
    """The retained-well statistic is the readable one, and is NOT monotone.

    It is in the table because "4 gRNAs in the wells I keep" is what a
    user reads; it is documented as non-monotone because dropping a
    thin well raises the median of what is left. Four wells, four
    thresholds, worked out by hand:

    ===========  =====================  ====  ====  ====  ====
    well         fractions              .05   .15   .35   .45
    ===========  =====================  ====  ====  ====  ====
    r1_c1        1.00                   1     1     1     1
    r1_c2        .40 .30 .30            3     3     1     0
    r2_c1        .50 .50                2     2     2     2
    r2_c2        .10 x10                10    0     0     0
    ===========  =====================  ====  ====  ====  ====

    Over all four wells the median goes 2.5, 1.5, 1.0, 0.5 — never up.
    Over the retained wells only it goes 2.5, 2.0, 1.0, **1.5**: losing
    ``r1_c2``'s single surviving guide at .45 leaves a shorter, denser
    list behind, and the median climbs.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 100},
        ("r1", "c2"): {"g1": 40, "g2": 30, "g3": 30},
        ("r2", "c1"): {"g1": 50, "g2": 50},
        ("r2", "c2"): {f"g{i}": 10 for i in range(1, 11)},
    }))
    sweep = QC.threshold_sweep(counts, [0.05, 0.15, 0.35, 0.45],
                               target_grnas_per_well=2)
    assert sweep["grnas_per_well"].tolist() == [2.5, 1.5, 1.0, 0.5]
    assert sweep["grnas_per_well_retained"].tolist() == [2.5, 2.0, 1.0, 1.5]
    assert np.all(np.diff(sweep["grnas_per_well"].to_numpy(float)) <= 0)
    assert np.any(np.diff(
        sweep["grnas_per_well_retained"].to_numpy(float)) > 0)


def test_the_sweep_can_report_the_mean_instead_of_the_median():
    """`target_statistic='mean'` changes the headline column, hand-checked.

    Four wells keeping 4, 2, 1 and 3 gRNAs at a cutoff of 0.10 (the same
    hand case as the collision test): the mean is 2.5, and so is the
    median, so the case is chosen with a fifth well that separates them —
    one holding a single guide drops the mean to 2.2 while the median
    falls to 2.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 40, "g2": 30, "g3": 20, "g4": 10},
        ("r1", "c2"): {"g1": 50, "g5": 50},
        ("r2", "c1"): {"g6": 90, "g7": 4, "g8": 3, "g9": 3},
        ("r2", "c2"): {"g1": 34, "g2": 33, "g3": 33},
        ("r2", "c3"): {"g1": 100},
    }))
    means = QC.threshold_sweep(counts, [0.10], 2, statistic="mean")
    medians = QC.threshold_sweep(counts, [0.10], 2, statistic="median")
    assert means.iloc[0]["grnas_per_well"] == pytest.approx(11 / 5)
    assert medians.iloc[0]["grnas_per_well"] == pytest.approx(2.0)
    assert means.iloc[0]["grnas_per_well_retained"] == pytest.approx(11 / 5)


def test_the_sweep_always_contains_the_derived_threshold(library):
    choice = QC.derive_threshold(library, 4)
    grid = QC.sweep_grid(choice.threshold)
    assert np.isclose(grid, choice.threshold).sum() == 1, (
        "the derived threshold must appear on the sweep exactly once — "
        "twice means a near-duplicate row the user cannot tell apart")


def test_sweep_grid_rejects_ranges_that_show_nothing():
    with pytest.raises(ValueError, match="must be positive"):
        QC.sweep_grid(0.0)
    with pytest.raises(ValueError, match="greater than 1"):
        QC.sweep_grid(0.02, span=1.0)
    with pytest.raises(ValueError, match="at least 3"):
        QC.sweep_grid(0.02, points=2)
    with pytest.raises(ValueError, match="empty or non-positive"):
        QC.sweep_grid(0.02, low=0.0)


def test_the_sweep_reaches_the_bleed_through_tail(tmp_path):
    """A sweep that stops above the junk cannot show the collision knee."""
    csv = tmp_path / "unique_combinations.csv"
    make_library(real_per_well=4, seed=9).to_csv(csv, index=True)
    out = QC.barcode_qc({"count_data": str(csv), "target_grnas_per_well": 4,
                         "plot": False, "save": False, "verbose": False})
    sweep = out["sweep"]
    assert sweep["collision_rate"].max() > 0.5, (
        "the sweep never reached a threshold where wells go over budget, so "
        "the user is shown no cost for relaxing the cutoff")
    assert sweep["collision_rate"].min() == 0.0


# ---------------------------------------------------------------------------
# 3 — collisions, counted against a hand-computed case
# ---------------------------------------------------------------------------

def test_barcode_collisions_match_a_hand_computed_case():
    """Six barcodes, three collisions at one substitution, worked out by eye.

    ::

        a  AAAACCCC
        b  TAAACCCC   a at position 0                    -> (a, b) = 1
        f  AAATCCCC   a at position 3                    -> (a, f) = 1
        e  AACCCCCC   a at positions 2 and 3             -> (a, e) = 2
        c  GGGGTTTT
        d  GGGGTTTT   the same sequence under two names  -> (c, d) = 0

    The remaining distances are ``(b, f) = 2`` (position 0 and position
    3), ``(b, e) = 3``, ``(e, f) = 2``, and 8 between anything in the
    A/C block and anything in the G/T one. So one substitution finds
    three pairs, and ``a`` is in two of them — which is what makes this a
    test of the neighbour search rather than of a single lookup.
    """
    reference = {
        "a": "AAAACCCC",
        "b": "TAAACCCC",
        "c": "GGGGTTTT",
        "d": "GGGGTTTT",
        "e": "AACCCCCC",
        "f": "AAATCCCC",
    }
    pairs = QC.barcode_collisions({"row": reference}, max_distance=1)
    found = {(r.name_a, r.name_b, r.distance) for r in pairs.itertuples()}
    assert found == {("a", "b", 1), ("a", "f", 1), ("c", "d", 0)}

    # Raising the distance brings in exactly the three pairs at 2 and
    # leaves (b, e) at 3 out.
    wider = QC.barcode_collisions({"row": reference}, max_distance=2)
    assert {(r.name_a, r.name_b, r.distance) for r in wider.itertuples()} == \
        found | {("a", "e", 2), ("b", "f", 2), ("e", "f", 2)}

    # At distance 0 only the outright duplicate survives.
    exact = QC.barcode_collisions({"row": reference}, max_distance=0)
    assert {(r.name_a, r.name_b) for r in exact.itertuples()} == {("c", "d")}

    with pytest.raises(ValueError, match="must not be negative"):
        QC.barcode_collisions({"row": reference}, max_distance=-1)


def test_collision_summary_counts_barcodes_at_risk_and_reads_at_risk():
    """The summary turns a list of pairs into an amount of data at risk."""
    reference = {"g1": "AAAACCCC", "g2": "TAAACCCC", "g3": "GGGGTTTT",
                 "g4": "CCCCAAAA"}
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 60, "g2": 20, "g3": 10, "g4": 10},
    }))
    pairs = QC.barcode_collisions({"grna": reference})
    summary = QC.collision_summary({"grna": reference}, pairs, counts)
    row = summary.iloc[0]
    assert row["n_barcodes"] == 4
    assert row["n_colliding_pairs"] == 1          # g1/g2 only
    assert row["n_barcodes_at_risk"] == 2
    assert row["collision_rate"] == pytest.approx(0.5)
    # g1 + g2 carry 80 of the 100 reads.
    assert row["reads_at_risk"] == pytest.approx(0.8)


def test_a_reference_with_no_near_neighbours_reports_no_collisions():
    clean = {"x": "AAAAAAAA", "y": "CCCCCCCC", "z": "GGGGGGGG"}
    pairs = QC.barcode_collisions({"row": clean})
    assert pairs.empty
    assert list(pairs.columns) == ["reference", "name_a", "name_b",
                                   "distance", "sequence_a", "sequence_b"]
    summary = QC.collision_summary({"row": clean}, pairs)
    assert summary.iloc[0]["collision_rate"] == 0.0


def test_collisions_read_the_projects_fasta_and_csv_references_alike(
        synth_barcodes):
    """The three barcode references ship as both FASTA and CSV; both parse.

    ``synth_barcodes`` is the project's synthetic sequencing fixture and
    writes each of the column / row / gRNA tables in both forms. A QC that
    could only read one of them would silently skip the collision panel
    for half the runs in this repository.
    """
    paths = synth_barcodes["paths"]
    for label in ("column", "row", "grna"):
        from_csv = QC._read_reference(paths[f"{label}_csv"])
        from_fasta = QC._read_reference(paths[f"{label}_fasta"])
        assert from_csv == from_fasta
        assert from_csv == {n: s.upper()
                            for n, s in synth_barcodes[
                                {"column": "columns", "row": "rows",
                                 "grna": "grnas"}[label]].items()}

    csv_pairs = QC.barcode_collisions({"grna": paths["grna_csv"]})
    fasta_pairs = QC.barcode_collisions({"grna": paths["grna_fasta"]})
    assert csv_pairs.equals(fasta_pairs)


def test_a_reference_missing_its_columns_is_refused(tmp_path):
    bad = tmp_path / "row.csv"
    bad.write_text("barcode,label\nAAAA,r1\n")
    with pytest.raises(ValueError, match="missing column"):
        QC.barcode_collisions({"row": str(bad)})


# ---------------------------------------------------------------------------
# 4 — the collision RATE in the sweep, hand-computed
# ---------------------------------------------------------------------------

def test_collision_rate_counts_exactly_the_wells_over_the_budget():
    """Four wells, a budget of 2, one threshold, counted by hand.

    At a cutoff of 0.10 each well keeps the guides holding at least 10% of
    its reads:

    ==========  ================================  ==============
    well        fractions                         kept at 0.10
    ==========  ================================  ==============
    r1_c1       .40 .30 .20 .10                   4  -> over
    r1_c2       .50 .50                           2  -> at budget
    r2_c1       .90 .04 .03 .03                   1  -> under
    r2_c2       .34 .33 .33                       3  -> over
    ==========  ================================  ==============

    So 2 of 4 wells are over a budget of 2, every well retains at least
    one guide, and 4 + 2 + 1 + 3 = 10 calls survive.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 40, "g2": 30, "g3": 20, "g4": 10},
        ("r1", "c2"): {"g1": 50, "g5": 50},
        ("r2", "c1"): {"g6": 90, "g7": 4, "g8": 3, "g9": 3},
        ("r2", "c2"): {"g1": 34, "g2": 33, "g3": 33},
    }))
    sweep = QC.threshold_sweep(counts, [0.10], target_grnas_per_well=2)
    row = sweep.iloc[0]
    assert row["wells_over_budget"] == 2
    assert row["collision_rate"] == pytest.approx(0.5)
    assert row["collision_rate_retained"] == pytest.approx(0.5)
    assert row["wells_retained"] == 4
    assert row["well_retention"] == pytest.approx(1.0)
    assert row["n_calls"] == 10
    assert row["grnas_per_well"] == pytest.approx(2.5)   # median of 4,2,1,3
    assert row["reads_retained"] == pytest.approx(
        (40 + 30 + 20 + 10 + 50 + 50 + 90 + 34 + 33 + 33) / 400)


def test_collision_rate_denominators_part_company_when_wells_drop_out():
    """Over all wells versus over retained wells is a real difference.

    At 0.60 only ``r2_c1`` keeps anything (its 0.90 guide), and it is not
    over a budget of 1. At 0.35 two wells keep guides and one of them
    (``r1_c1``, holding 0.40) is not over budget either, so both rates are
    zero; at 0.25 three wells survive and two are over budget — 2/4 of all
    wells but 2/3 of the retained ones.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 40, "g2": 30, "g3": 30},
        ("r1", "c2"): {"g1": 34, "g2": 33, "g3": 33},
        ("r2", "c1"): {"g6": 90, "g7": 10},
        ("r2", "c2"): {"g8": 20, "g9": 20, "g10": 20, "g11": 20, "g12": 20},
    }))
    sweep = QC.threshold_sweep(counts, [0.25, 0.60], target_grnas_per_well=1)
    loose, tight = sweep.iloc[0], sweep.iloc[1]
    assert loose["wells_retained"] == 3
    assert loose["wells_over_budget"] == 2
    assert loose["collision_rate"] == pytest.approx(0.5)
    assert loose["collision_rate_retained"] == pytest.approx(2 / 3)
    assert tight["wells_retained"] == 1
    assert tight["collision_rate"] == 0.0


# ---------------------------------------------------------------------------
# 5 — starved wells, identified exactly
# ---------------------------------------------------------------------------

def test_starved_wells_are_exactly_the_wells_planted_starved():
    planted = [("r1", "c1"), ("r3", "c5"), ("r8", "c12")]
    counts = QC.load_count_table(
        make_library(seed=2, starved_wells=planted, starved_depth=300,
                     depth=20_000))
    starved = QC.starved_wells(counts)
    expected = {f"plate1_{r}_{c}" for r, c in planted}
    assert set(starved["prc"]) == expected
    assert starved.attrs["cutoff"] == pytest.approx(0.1 * 20_000, rel=0.05)


def test_the_starvation_cut_is_strict_and_the_boundary_well_is_kept():
    """A well exactly on the cut is not starved — the comparison is `<`.

    Five wells at 1000, 1000, 1000, 100 and 99 reads: the median is 1000,
    so the derived cut is 100. The well holding exactly 100 sits on it and
    stays; the one holding 99 is the only one below it.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 1000},
        ("r1", "c2"): {"g1": 1000},
        ("r1", "c3"): {"g1": 1000},
        ("r1", "c4"): {"g1": 100},     # exactly 0.1 * median(1000)
        ("r1", "c5"): {"g1": 99},      # one read below it
    }))
    starved = QC.starved_wells(counts)
    assert starved.attrs["cutoff"] == pytest.approx(100.0)
    assert set(starved["prc"]) == {"plate1_r1_c5"}


def test_an_absolute_floor_overrides_the_derived_cut():
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 1000},
        ("r1", "c2"): {"g1": 500},
        ("r1", "c3"): {"g1": 400},
    }))
    assert set(QC.starved_wells(counts, min_reads=600)["prc"]) == {
        "plate1_r1_c2", "plate1_r1_c3"}
    assert QC.starved_wells(counts, min_reads=1).empty


def test_a_starvation_rule_that_marks_everything_or_nothing_is_refused():
    per_well = QC.reads_per_well(QC.load_count_table(
        hand_table({("r1", "c1"): {"g1": 10}})))
    with pytest.raises(ValueError, match="must not be negative"):
        QC.starvation_cutoff(per_well, min_reads=-1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        QC.starvation_cutoff(per_well, starved_read_fraction=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        QC.starvation_cutoff(per_well, starved_read_fraction=1.5)


def test_excluding_starved_wells_changes_the_derived_threshold(tmp_path):
    """Starved wells drag the target onto a cutoff the good wells never needed."""
    planted = [(f"r{r}", "c1") for r in range(1, 7)]
    frame = make_library(n_rows=8, n_cols=6, real_per_well=4, seed=4,
                         starved_wells=planted, starved_depth=60)
    csv = tmp_path / "unique_combinations.csv"
    frame.to_csv(csv, index=True)
    with_starved = QC.barcode_qc({
        "count_data": str(csv), "target_grnas_per_well": 4,
        "exclude_starved_wells": False, "plot": False, "save": False,
        "verbose": False})
    without = QC.barcode_qc({
        "count_data": str(csv), "target_grnas_per_well": 4,
        "exclude_starved_wells": True, "plot": False, "save": False,
        "verbose": False})
    assert len(without["starved"]) == len(planted)
    assert without["choice"].n_wells == with_starved["choice"].n_wells - len(planted)
    assert without["threshold"] != with_starved["threshold"]


# ---------------------------------------------------------------------------
# 6 — position effects and library depth
# ---------------------------------------------------------------------------

def test_a_planted_row_effect_is_flagged_and_the_rest_is_not():
    counts = QC.load_count_table(
        make_library(n_rows=6, n_cols=8, seed=1,
                     row_depth_factor={"r1": 0.3, "r6": 3.0}))
    effects = QC.position_effects(counts, ratio=2.0)
    flagged = effects[effects["flagged"]]
    assert set(zip(flagged["axis"], flagged["label"])) == {
        ("row", "r1"), ("row", "r6")}
    # Worst-first ordering: the flagged rows lead the table.
    assert bool(effects.iloc[0]["flagged"])
    # No column is flagged — the effect was planted on rows only.
    assert not effects[(effects["axis"] == "column") & effects["flagged"]].any(
        axis=None)


def test_position_effects_needs_a_ratio_above_one(library):
    with pytest.raises(ValueError, match="greater than 1"):
        QC.position_effects(library, ratio=1.0)


def test_library_depth_reports_dropout_against_the_designed_library():
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 10, "g2": 10},
        ("r1", "c2"): {"g3": 10},
    }))
    depth = QC.library_depth(counts, expected_grnas=[f"g{i}" for i in range(1, 6)])
    assert depth["n_grnas_observed"] == 3
    assert depth["n_grnas_expected"] == 5
    assert depth["dropout_fraction"] == pytest.approx(0.4)
    assert depth["dropped_grnas"] == ["g4", "g5"]
    # Three guides at 10 reads each: perfectly even.
    assert depth["gini"] == pytest.approx(0.0, abs=1e-12)
    assert depth["skew_ratio"] == pytest.approx(1.0)


def test_library_depth_without_a_reference_reports_no_dropout(library):
    depth = QC.library_depth(library)
    assert depth["n_grnas_expected"] is None
    assert depth["dropout_fraction"] is None
    assert 0.0 <= depth["gini"] < 1.0
    assert depth["top_decile_share"] >= 0.1


def test_gini_is_one_for_a_library_where_one_guide_took_everything():
    counts = QC.load_count_table(hand_table({
        ("r1", "c1"): {"g1": 1_000_000},
        ("r1", "c2"): {"g2": 1, "g3": 1, "g4": 1, "g5": 1},
    }))
    depth = QC.library_depth(counts)
    assert depth["gini"] > 0.75
    # A tenth of the guides hold essentially all the reads.
    assert depth["top_decile_share"] > 0.99


# ---------------------------------------------------------------------------
# 7 — unmapped reads
# ---------------------------------------------------------------------------

def test_unmapped_fraction_is_exact_when_the_count_table_is_supplied():
    qc = pd.DataFrame([{"read": 0, "column_sequence": 0, "columnID": 30,
                        "row_sequence": 0, "rowID": 10, "grna_sequence": 0,
                        "grna_name": 20, "total_reads": 1000}])
    counts = QC.load_count_table(hand_table({("r1", "c1"): {"g1": 940}}))
    out = QC.unmapped_read_fractions(qc, counts)
    assert out["total_reads"] == 1000
    assert out["per_field"] == {"columnID": 0.03, "rowID": 0.01,
                                "grna_name": 0.02}
    assert out["mapped_reads"] == 940
    assert out["unmapped_fraction"] == pytest.approx(0.06)
    # The bounds bracket the exact answer: at best every failure landed on
    # the same reads (3%), at worst they were disjoint (6%).
    assert out["unmapped_fraction_lower"] == pytest.approx(0.03)
    assert out["unmapped_fraction_upper"] == pytest.approx(0.06)
    assert (out["unmapped_fraction_lower"] - 1e-9 <= out["unmapped_fraction"]
            <= out["unmapped_fraction_upper"] + 1e-9)


def test_unmapped_fractions_sum_across_the_appended_qc_rows(tmp_path):
    """qc.csv accumulates one row per chunk; the totals are the sums."""
    path = tmp_path / "qc.csv"
    pd.DataFrame([
        {"columnID": 10, "rowID": 0, "grna_name": 5, "total_reads": 100},
        {"columnID": 30, "rowID": 20, "grna_name": 15, "total_reads": 300},
    ]).to_csv(path, index=False)
    out = QC.unmapped_read_fractions(str(path))
    assert out["total_reads"] == 400
    assert out["per_field"]["columnID"] == pytest.approx(0.1)
    assert out["per_field"]["rowID"] == pytest.approx(0.05)
    assert "unmapped_fraction" not in out


def test_several_qc_files_are_pooled_into_one_answer(tmp_path):
    """One qc.csv per sample; the run's unmapped share is over all of them."""
    paths = []
    for index, (bad, total) in enumerate(((10, 100), (90, 900))):
        path = tmp_path / f"qc{index}.csv"
        pd.DataFrame([{"columnID": bad, "rowID": 0, "grna_name": 0,
                       "total_reads": total}]).to_csv(path, index=False)
        paths.append(str(path))
    out = QC.unmapped_read_fractions(paths)
    assert out["total_reads"] == 1000
    assert out["per_field"]["columnID"] == pytest.approx(0.1)


def test_a_qc_file_that_is_not_a_qc_file_is_refused(tmp_path):
    path = tmp_path / "not_qc.csv"
    pd.DataFrame([{"a": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no 'total_reads'"):
        QC.unmapped_read_fractions(str(path))

    empty = tmp_path / "empty_qc.csv"
    pd.DataFrame([{"columnID": 0, "total_reads": 0}]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="zero total reads"):
        QC.unmapped_read_fractions(str(empty))


# ---------------------------------------------------------------------------
# 8 — loading and normalising count tables
# ---------------------------------------------------------------------------

def test_load_count_table_accepts_both_grna_spellings_and_computes_fractions():
    frame = pd.DataFrame([
        {"rowID": "r1", "columnID": "c1", "grna_name": "g1", "count": 30},
        {"rowID": "r1", "columnID": "c1", "grna_name": "g2", "count": 70},
    ])
    counts = QC.load_count_table(frame)
    assert set(QC.COUNT_COLUMNS).issubset(counts.columns)
    assert list(counts["fraction"]) == [0.3, 0.7]
    assert (counts["well_reads"] == 100).all()
    assert list(counts["prc"]) == ["plate1_r1_c1"] * 2

    renamed = frame.rename(columns={"grna_name": "grna", "rowID": "row",
                                    "columnID": "column"})
    assert QC.load_count_table(renamed)["fraction"].tolist() == [0.3, 0.7]


def test_several_plates_do_not_merge_their_wells(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    hand_table({("r1", "c1"): {"g1": 10}}).to_csv(a, index=False)
    hand_table({("r1", "c1"): {"g1": 20}}).to_csv(b, index=False)
    counts = QC.load_count_table([str(a), str(b)])
    assert sorted(counts["prc"]) == ["plate1_r1_c1", "plate2_r1_c1"]
    assert sorted(counts["count"]) == [10, 20]


def test_the_same_well_read_twice_has_its_reads_summed():
    """A resequenced lane adds reads to a well; it does not duplicate it."""
    frame = pd.DataFrame([
        {"plateID": "p1", "rowID": "r1", "columnID": "c1",
         "grna_name": "g1", "count": 30},
        {"plateID": "p1", "rowID": "r1", "columnID": "c1",
         "grna_name": "g1", "count": 70},
    ])
    counts = QC.load_count_table(frame)
    assert len(counts) == 1
    assert counts.iloc[0]["count"] == 100
    assert counts.iloc[0]["fraction"] == 1.0


def test_non_positive_and_unkeyed_rows_are_dropped_not_counted():
    frame = pd.DataFrame([
        {"rowID": "r1", "columnID": "c1", "grna_name": "g1", "count": 90},
        {"rowID": "r1", "columnID": "c1", "grna_name": "g2", "count": 0},
        {"rowID": "r1", "columnID": "c1", "grna_name": "g3", "count": -5},
        {"rowID": None, "columnID": "c1", "grna_name": "g4", "count": 10},
    ])
    counts = QC.load_count_table(frame)
    assert list(counts["grna"]) == ["g1"]
    assert counts.iloc[0]["fraction"] == 1.0


def test_a_table_missing_a_key_column_says_which_one(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"rowID": "r1", "count": 5}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columnID"):
        QC.load_count_table(str(path))
    with pytest.raises(ValueError, match="no sources"):
        QC.load_count_table([])
    with pytest.raises(ValueError, match="No usable rows"):
        QC.load_count_table(pd.DataFrame(
            [{"rowID": "r1", "columnID": "c1", "grna": "g", "count": 0}]))


def test_well_fractions_refuses_an_empty_population(library):
    with pytest.raises(ValueError, match="No wells left"):
        QC.WellFractions(library, wells=["nothing_like_a_well"])
    with pytest.raises(ValueError, match="median.*mean"):
        QC.WellFractions(library).statistic_at(0.1, "mode")
    with pytest.raises(ValueError, match="median.*mean"):
        QC.threshold_sweep(library, [0.1], 4, statistic="mode")


def test_column_names_are_matched_case_insensitively():
    """A hand-edited CSV writes RowID or Count; both still resolve."""
    frame = pd.DataFrame([{"RowID": "r1", "ColumnID": "c1",
                           "GRNA_Name": "g1", "Count": 40},
                          {"RowID": "r1", "ColumnID": "c1",
                           "GRNA_Name": "g2", "Count": 60}])
    counts = QC.load_count_table(frame)
    assert list(counts["fraction"]) == [0.4, 0.6]


def test_a_named_plate_is_used_for_the_first_unnamed_source(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    hand_table({("r1", "c1"): {"g1": 10}}).to_csv(a, index=False)
    hand_table({("r1", "c1"): {"g1": 20}}).to_csv(b, index=False)
    counts = QC.load_count_table([str(a), str(b)], plate="screenA")
    # The first nameless source takes the name; the second is numbered by
    # its position in count_data, so the name traces back to the file and
    # the two plates can never merge into one.
    assert sorted(counts["prc"]) == ["plate2_r1_c1", "screenA_r1_c1"]


def test_an_empty_frame_yields_empty_summaries_rather_than_a_crash():
    """The summary helpers are called on filtered frames and must survive one."""
    counts = QC.load_count_table(hand_table({("r1", "c1"): {"g1": 10}}))
    empty = counts.iloc[0:0]
    assert QC.reads_per_well(empty).empty
    assert QC.position_effects(empty).empty
    assert QC.starvation_cutoff(QC.reads_per_well(empty)) == 0.0
    depth = QC.library_depth(empty)
    assert depth["n_grnas_observed"] == 0
    assert np.isnan(depth["gini"])
    assert np.isnan(depth["skew_ratio"])


def test_gini_is_defined_at_the_edges_and_refuses_negative_reads():
    assert np.isnan(QC._gini(np.array([])))
    assert QC._gini(np.array([0.0, 0.0, 0.0])) == 0.0
    with pytest.raises(ValueError, match="undefined for negative"):
        QC._gini(np.array([1.0, -1.0]))


def test_a_fasta_reference_survives_blank_lines_and_wrapped_sequences(
        tmp_path):
    path = tmp_path / "row.fasta"
    path.write_text(">r1 first row barcode\nAAAA\nCCCC\n\n>r2\nGGGGTTTT\n")
    assert QC._read_reference(str(path)) == {"r1": "AAAACCCC",
                                             "r2": "GGGGTTTT"}


def test_barcodes_of_different_lengths_never_collide():
    """A shorter barcode shifts every field after it, so the regex rejects
    the read rather than mis-assigning it — comparing across lengths would
    invent a collision that cannot happen."""
    pairs = QC.barcode_collisions(
        {"row": {"short": "AAAA", "long": "AAAACCCC", "other": "AAAACCCG"}},
        max_distance=2)
    assert {(r.name_a, r.name_b) for r in pairs.itertuples()} == {
        ("long", "other")}


# ---------------------------------------------------------------------------
# 9 — the recommendation, in words
# ---------------------------------------------------------------------------

def test_the_recommendation_states_the_derived_number_and_what_it_buys(
        library):
    choice = QC.derive_threshold(library, 4)
    sweep = QC.threshold_sweep(library, QC.sweep_grid(choice.threshold,
                                                      span=32, points=40), 4)
    text = QC.recommend_threshold(sweep, choice)
    # The number the target translated to has to be legible, not implied.
    assert f"{choice.threshold:.4f}" in text
    assert f"{100 * choice.threshold:.2f}%" in text
    assert "Target: 4 gRNAs per well (median)" in text
    for phrase in ("gRNAs per well", "wells are retained", "Relaxing to",
                   "collision rate"):
        assert phrase in text, phrase
    assert "\n" in text


def test_the_recommendation_says_so_when_the_target_falls_between_steps():
    """A target of 2.5 cannot be hit exactly, and the text must not pretend."""
    counts = QC.load_count_table(hand_table({
        ("r1", f"c{i + 1}"): {"g0": 970, "g1": 10, "g2": 10, "g3": 10}
        for i in range(10)}))
    choice = QC.derive_threshold(counts, 2.5)
    sweep = QC.threshold_sweep(counts, QC.sweep_grid(choice.threshold), 2.5)
    text = QC.recommend_threshold(sweep, choice)
    assert "no cutoff hits 2.5 exactly" in text
    assert "moves in steps" in text


def test_the_recommendation_admits_when_tightening_would_change_nothing():
    """On a wide plateau the honest sentence is that nothing moves.

    Ten wells whose one real guide holds 97% of the reads: anywhere from
    just above 1% to 97% keeps exactly that guide. A sweep confined to the
    middle of that band has no row where tightening costs anything, and
    saying "tightening to X drops the median to 1.0" — the same 1.0 — would
    be noise dressed as advice.
    """
    counts = QC.load_count_table(hand_table({
        ("r1", f"c{i + 1}"): {"g0": 970, "g1": 10, "g2": 10, "g3": 10}
        for i in range(10)}))
    choice = QC.derive_threshold(counts, 1)
    sweep = QC.threshold_sweep(
        counts, QC.sweep_grid(choice.threshold, span=2.0), 1)
    assert sweep["grnas_per_well"].nunique() == 1     # all plateau
    text = QC.recommend_threshold(sweep, choice)
    assert "changes neither" in text
    assert "Relaxing to" in text


def test_the_recommendation_names_the_threshold_below_which_collisions_climb(
        library):
    choice = QC.derive_threshold(library, 4)
    sweep = QC.threshold_sweep(library, QC.sweep_grid(choice.threshold,
                                                      span=32, points=40), 4)
    text = QC.recommend_threshold(sweep, choice)
    assert "rises sharply" in text
    assert "floor worth defending" in text


# ---------------------------------------------------------------------------
# 10 — settings registration through the seam
# ---------------------------------------------------------------------------

def test_the_module_registers_its_defaults_through_the_seam():
    from spacr.settings import defaults_for, has_registered_defaults, tooltips

    assert has_registered_defaults(QC.APP_KEY)
    defaults = defaults_for(QC.APP_KEY)
    assert defaults["target_grnas_per_well"] == 5
    assert defaults["target_statistic"] == "median"
    # A fresh dict every call: one screen editing another's defaults is
    # what the seam exists to prevent.
    defaults["target_grnas_per_well"] = 99
    assert defaults_for(QC.APP_KEY)["target_grnas_per_well"] == 5
    # Every key the module introduces carries its own help text.
    for key in QC._EXPECTED_TYPES:
        assert key in tooltips, key


def test_registering_twice_is_a_no_op_rather_than_a_crash():
    """A module reload in a test session must not fail on its own key."""
    QC._register()
    QC._register()
    from spacr.settings import defaults_for
    assert defaults_for(QC.APP_KEY)["target_statistic"] == "median"


def test_the_defaults_factory_fills_in_place_without_overriding():
    filled = QC.barcode_qc_defaults({"target_grnas_per_well": 3})
    assert filled["target_grnas_per_well"] == 3
    assert filled["sweep_span"] == QC.DEFAULT_SWEEP_SPAN


# ---------------------------------------------------------------------------
# 11 — the whole entry point, including figures
# ---------------------------------------------------------------------------

def test_barcode_qc_writes_its_tables_figures_and_recommendation(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    csv = tmp_path / "unique_combinations.csv"
    make_library(real_per_well=4, seed=13).to_csv(csv, index=True)
    qc_csv = tmp_path / "qc.csv"
    pd.DataFrame([{"columnID": 100, "rowID": 50, "grna_name": 200,
                   "total_reads": 2_000_000}]).to_csv(qc_csv, index=False)

    out = QC.barcode_qc({"count_data": str(csv), "qc_data": str(qc_csv),
                         "target_grnas_per_well": 4, "verbose": False})
    dst = Path(out["dst"])
    assert dst.name == "barcode_qc"
    for name in ("threshold_sweep.csv", "reads_per_well.csv",
                 "starved_wells.csv", "position_effects.csv",
                 "threshold_recommendation.txt", "threshold_sweep.pdf",
                 "barcode_qc.pdf"):
        assert (dst / name).is_file(), name
    assert f"{out['threshold']:.4f}" in (
        dst / "threshold_recommendation.txt").read_text()
    assert out["choice"].achieved == pytest.approx(4.0)
    assert out["unmapped"]["total_reads"] == 2_000_000
    assert set(out) >= {"choice", "threshold", "sweep", "recommendation",
                        "per_well", "starved", "positions", "depth",
                        "collisions", "unmapped", "dst"}


def test_barcode_qc_can_be_asked_not_to_write_anything(tmp_path):
    csv = tmp_path / "unique_combinations.csv"
    make_library(n_rows=3, n_cols=4, seed=17).to_csv(csv, index=True)
    out = QC.barcode_qc({"count_data": str(csv), "target_grnas_per_well": 4,
                         "plot": False, "save": False, "verbose": False})
    assert not Path(out["dst"]).exists()
    assert out["threshold"] > 0


def test_barcode_qc_prints_the_recommendation_when_verbose(tmp_path, capsys):
    csv = tmp_path / "unique_combinations.csv"
    make_library(n_rows=3, n_cols=4, seed=19,
                 starved_wells=[("r1", "c1")]).to_csv(csv, index=True)
    QC.barcode_qc({"count_data": str(csv), "target_grnas_per_well": 4,
                   "plot": False, "save": False, "verbose": True})
    printed = capsys.readouterr().out
    assert "Derived abundance threshold" in printed
    assert "Starved wells" in printed


def test_barcode_qc_reads_a_list_of_count_tables_and_reports_every_panel(
        tmp_path, capsys):
    """Two plates, two barcode references, verbose on: every panel speaks.

    This is the shape a real run has — several count tables, the barcode
    CSVs beside them — and it is the only place the collision tables get
    written and the position / collision / unmapped lines get printed.
    """
    import matplotlib
    matplotlib.use("Agg")

    tables = []
    for index in (0, 1):
        path = tmp_path / f"plate{index}.csv"
        make_library(n_rows=4, n_cols=6, seed=31 + index,
                     starved_wells=[("r1", "c1")],
                     row_depth_factor={"r4": 0.25}).to_csv(path, index=True)
        tables.append(str(path))

    grna_csv = tmp_path / "grna.csv"
    # g0000/g0001 differ at one base, so the collision panel has content.
    pd.DataFrame({"name": [f"g{i:04d}" for i in range(3)],
                  "sequence": ["AAAACCCCGG", "TAAACCCCGG",
                               "GGGGTTTTAA"]}).to_csv(grna_csv, index=False)
    qc_csv = tmp_path / "qc.csv"
    pd.DataFrame([{"columnID": 1000, "rowID": 500, "grna_name": 2500,
                   "total_reads": 1_000_000}]).to_csv(qc_csv, index=False)

    out = QC.barcode_qc({"count_data": tables, "qc_data": str(qc_csv),
                         "grna_csv": str(grna_csv),
                         "target_grnas_per_well": 4, "verbose": True})
    printed = capsys.readouterr().out
    assert "Position effects flagged" in printed
    assert "Barcode collisions" in printed
    assert "Unmapped reads" in printed

    dst = Path(out["dst"])
    assert (dst / "barcode_collisions.csv").is_file()
    assert (dst / "collision_summary.csv").is_file()
    # Two plates, kept apart.
    assert set(out["per_well"]["plateID"]) == {"plate1", "plate2"}
    assert out["depth"]["n_grnas_expected"] == 3
    # Dropout is about the DESIGNED library: all three named guides were
    # seen, so nothing dropped out, however many extra guides the run
    # happens to carry.
    assert out["depth"]["dropout_fraction"] == pytest.approx(0.0)
    assert out["depth"]["dropped_grnas"] == []
    assert out["depth"]["n_grnas_observed"] > 3


def test_figures_are_drawn_even_when_nothing_is_written(tmp_path):
    """plot without save: the panels are built and returned, not saved."""
    import matplotlib
    matplotlib.use("Agg")

    csv = tmp_path / "unique_combinations.csv"
    make_library(n_rows=3, n_cols=4, seed=37).to_csv(csv, index=True)
    out = QC.barcode_qc({"count_data": str(csv), "target_grnas_per_well": 4,
                         "plot": True, "save": False, "verbose": False})
    assert not Path(out["dst"]).exists()
    assert QC._save_figure(object(), None, "anything") is None


def test_the_qc_panels_cope_with_nothing_to_draw():
    """No position effects and no qc.csv leaves two panels deliberately blank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = QC.load_count_table(hand_table({("r1", "c1"): {"g1": 10}}))
    empty_counts = counts.iloc[0:0]
    positions = QC.position_effects(empty_counts)
    assert positions.empty

    figure = QC.plot_barcode_qc(
        counts, per_well=QC.reads_per_well(counts),
        starved=QC.starved_wells(counts), positions=positions,
        depth=QC.library_depth(counts, expected_grnas=["g1", "g2"]),
        unmapped=None, dst=None)
    assert figure is not None
    plt.close(figure)


def test_the_choice_serialises_for_a_report():
    counts = QC.load_count_table(make_library(n_rows=2, n_cols=3, seed=23))
    payload = QC.derive_threshold(counts, 4).as_dict()
    assert set(payload) == {"threshold", "achieved", "target", "statistic",
                            "n_wells", "attainable", "n_candidates",
                            "interval_low", "interval_high"}
    assert isinstance(payload["threshold"], float)


# ---------------------------------------------------------------------------
# 12 — end to end, from an Illumina FASTQ pair with three barcodes
# ---------------------------------------------------------------------------

def _write_demo_reads(dst: Path, *, n_rows=4, n_cols=6, n_grnas=40,
                      real_per_well=3, reads_per_well=180, junk_share=0.09,
                      seed=0):
    """Write a paired Illumina FASTQ + the three barcode CSVs.

    Uses the project's synthetic read frame (``spacr.qt.synthetic``) so the
    shipped ``regex`` / ``target_sequence`` / ``offset_start`` /
    ``expected_end`` defaults recover every planted barcode. Each well
    carries exactly ``real_per_well`` guides splitting most of its reads,
    plus a bleed-through tail — the same shape as
    :func:`make_library`, but arriving through the real read path.

    :returns: ``(src, barcode_paths, real_per_well)``.
    """
    from spacr.qt.synthetic import (barcode_pool, generate_barcode_csv,
                                    synthetic_read, GRNA_LENGTH,
                                    WELL_BARCODE_LENGTH)

    dst.mkdir(parents=True, exist_ok=True)
    barcodes = dst / "barcodes"
    grnas = barcode_pool(n_grnas, GRNA_LENGTH, seed=seed)
    rows = barcode_pool(n_rows, WELL_BARCODE_LENGTH, seed=seed + 101)
    columns = barcode_pool(n_cols, WELL_BARCODE_LENGTH, seed=seed + 202)
    paths = {
        "grna_csv": str(generate_barcode_csv(
            barcodes / "grna.csv",
            [f"g{i:04d}" for i in range(n_grnas)], grnas)),
        "row_csv": str(generate_barcode_csv(
            barcodes / "row.csv", [f"r{i + 1}" for i in range(n_rows)], rows)),
        "column_csv": str(generate_barcode_csv(
            barcodes / "column.csv",
            [f"c{i + 1}" for i in range(n_cols)], columns)),
    }

    rng = np.random.default_rng(seed)
    quality = "I" * 150
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    index = 0
    r1_path = dst / "demo_R1_001.fastq.gz"
    r2_path = dst / "demo_R2_001.fastq.gz"
    with gzip.open(r1_path, "wt") as r1, gzip.open(r2_path, "wt") as r2:
        for row_bc in rows:
            for col_bc in columns:
                real = rng.choice(n_grnas, size=real_per_well, replace=False)
                junk = [g for g in range(n_grnas) if g not in real]
                n_junk = max(1, int(round(reads_per_well * junk_share)))
                picks = ([int(g) for g in real
                          for _ in range((reads_per_well - n_junk)
                                         // real_per_well)]
                         + [int(g) for g in rng.choice(junk, size=n_junk)])
                for pick in picks:
                    seq = synthetic_read(col_bc, grnas[pick], row_bc)
                    header = (f"@SIM:1:FC:1:1101:{index}:1")
                    r1.write(f"{header} 1:N:0:GCTTGCGC\n{seq}\n+\n{quality}\n")
                    rc = "".join(complement[b] for b in reversed(seq))
                    r2.write(f"{header} 2:N:0:GCTTGCGC\n{rc}\n+\n{quality}\n")
                    index += 1
    return str(dst), paths, real_per_well


@pytest.mark.integration
def test_end_to_end_from_fastq_the_derived_threshold_recovers_the_design(
        tmp_path):
    """FASTQ in, barcode QC out — and the target comes back as the answer.

    The reads are built so every well carries exactly three guides above a
    bleed-through tail. Mapping them produces the count table for real,
    and asking the QC for three gRNAs per well must hand back a threshold
    that separates the planted guides from the tail. Nothing in the chain
    is stubbed: this is the path a user's run takes.
    """
    import matplotlib
    matplotlib.use("Agg")
    from spacr.sequencing import generate_barecode_mapping

    src, paths, planted = _write_demo_reads(tmp_path / "run", seed=5)
    generate_barecode_mapping({
        "src": src, "mode": "paired", "single_direction": "R1",
        "save_h5": False, "chunk_size": 5000, "n_jobs": 1, "test": False,
        "fill_na": False, **paths,
    })

    combos = sorted(Path(src).rglob("unique_combinations.csv"))
    qc_files = sorted(Path(src).rglob("qc.csv"))
    assert combos and qc_files

    out = QC.barcode_qc({
        "count_data": str(combos[0]), "qc_data": str(qc_files[0]),
        "target_grnas_per_well": planted, "verbose": False,
        "plot": False, "save": False, **paths,
    })
    choice = out["choice"]
    assert choice.attainable
    assert choice.achieved == pytest.approx(float(planted))

    # The threshold really separates the design from the tail: applying it
    # leaves the planted number of guides in essentially every well.
    counts = out["counts"]
    kept = counts[counts["fraction"] >= choice.threshold]
    per_well = kept.groupby("prc")["grna"].nunique()
    assert per_well.median() == planted
    assert (per_well == planted).mean() > 0.8
    assert out["sweep"]["collision_rate"].max() > 0

    # The barcode references came through, so the collision and library
    # panels are real rather than skipped.
    assert not out["collision_summary"].empty
    assert out["depth"]["n_grnas_expected"] == 40
    # Every mapped read resolved all three barcodes: the references are
    # the ones the reads were built from.
    assert out["unmapped"]["unmapped_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert f"{choice.threshold:.4f}" in out["recommendation"]


# ---------------------------------------------------------------------------
# The pipeline call: generate_barecode_mapping QCs each sample as it lands
# ---------------------------------------------------------------------------
# Everything above proves `barcode_qc` answers the two questions a mapping
# run raises. Nothing asked it. A user had to know this module existed, find
# the sample folder, and type the paths -- and the questions ("did it work",
# "where does the threshold go") are ones nobody comes back for once the
# counts are on disk.

def _mapping_folder(tmp_path):
    """A src folder `parse_gz_files` sees one paired sample in."""
    src = tmp_path / "fastq"
    src.mkdir()
    for name in ("s1_R1_001.fastq.gz", "s1_R2_001.fastq.gz"):
        (src / name).write_bytes(b"")
    return src


def _mapping_settings(src, **over):
    from spacr.settings import set_default_generate_barecode_mapping

    settings = set_default_generate_barecode_mapping({"src": str(src)})
    settings.update(over)
    return settings


def _stub_reader(count_rows):
    """Stand in for the chunked read path: write the two CSVs and stop.

    The reads themselves are not what is under test here -- where the QC
    is called from is. This writes exactly the two files a real sample
    leaves behind, which is what the hook is handed.
    """
    def reader(**kwargs):
        pd.DataFrame(count_rows).to_csv(kwargs["unique_combinations_csv"],
                                        index=False)
        # The real qc.csv columns: one row per chunk, counting the reads
        # seen and the ones each barcode lookup could not resolve.
        pd.DataFrame([{"columnID": 5, "rowID": 4, "grna_name": 6,
                       "total_reads": 400}]).to_csv(
            kwargs["qc_csv_file"], index=False)
    return reader


COUNT_ROWS = {
    "plateID": ["plate1"] * 8,
    "rowID": ["A"] * 8,
    "columnID": ["01", "01", "02", "02", "03", "03", "04", "04"],
    "grna_name": ["g1", "g2", "g3", "g4", "g1", "g3", "g2", "g4"],
    "count": [90, 10, 80, 20, 70, 30, 60, 40],
}


def test_the_mapping_run_qcs_each_sample_it_finishes(tmp_path, monkeypatch):
    """The hook, at the end of the per-sample loop, on that sample's table."""
    import spacr.sequencing as SEQ

    src = _mapping_folder(tmp_path)
    monkeypatch.setattr(SEQ, "paired_read_chunked_processing",
                        _stub_reader(COUNT_ROWS))
    seen = {}
    real = QC.barcode_qc

    def watched(settings):
        seen["count_data"] = settings["count_data"]
        seen["dst"] = settings["dst"]
        seen["target"] = settings["target_grnas_per_well"]
        return real(settings)

    monkeypatch.setattr(QC, "barcode_qc", watched)

    SEQ.generate_barecode_mapping(_mapping_settings(
        src, barcode_qc=True, target_grnas_per_well=2, plot=False))

    dst = src / "s1_paired"
    assert (dst / "unique_combinations.csv").is_file(), "the run wrote nothing"
    assert seen.get("count_data") == str(dst / "unique_combinations.csv"), (
        "the QC was not run on the table this sample had just written")
    assert seen["dst"] == str(dst / "barcode_qc")
    assert seen["target"] == 2
    # ...and it left its answer beside the counts, which is the whole
    # point of running it here rather than telling the user to.
    assert os.path.isdir(seen["dst"])


def test_the_mapping_run_does_not_qc_unless_it_is_asked_to(tmp_path,
                                                           monkeypatch):
    """Opt-in: the QC pulls in plotting and statistics a read path must not.

    Off is also what a run that predates this module's existence expects,
    and `barcode_qc` is not in the mapping defaults, so the check has to
    survive a settings dict that has never heard of the key.
    """
    import spacr.sequencing as SEQ

    src = _mapping_folder(tmp_path)
    monkeypatch.setattr(SEQ, "paired_read_chunked_processing",
                        _stub_reader(COUNT_ROWS))
    called = []
    monkeypatch.setattr(QC, "barcode_qc",
                        lambda settings: called.append(settings))

    settings = _mapping_settings(src)
    assert "barcode_qc" not in settings
    SEQ.generate_barecode_mapping(settings)

    assert (src / "s1_paired" / "unique_combinations.csv").is_file()
    assert not called, "the QC ran on a settings dict that never asked for it"


def test_a_qc_failure_never_costs_the_run_its_counts(tmp_path, monkeypatch,
                                                     capsys):
    """The counts are on disk by the time the QC starts. They stay there.

    A missing barcode reference, a panel that cannot plot, an unreadable
    qc.csv -- every one of those is a reason to lose the report and none
    of them is a reason to lose a mapping run that may have taken hours.
    """
    import spacr.sequencing as SEQ

    src = _mapping_folder(tmp_path)
    monkeypatch.setattr(SEQ, "paired_read_chunked_processing",
                        _stub_reader(COUNT_ROWS))

    def explode(settings):
        raise RuntimeError("the QC could not plot")

    monkeypatch.setattr(QC, "barcode_qc", explode)

    SEQ.generate_barecode_mapping(_mapping_settings(
        src, barcode_qc=True, target_grnas_per_well=2))

    counts = src / "s1_paired" / "unique_combinations.csv"
    assert counts.is_file(), "a QC failure destroyed the run's own output"
    assert len(pd.read_csv(counts)) == 8
    printed = capsys.readouterr().out
    assert "barcode QC failed" in printed
    assert "the counts themselves were written" in printed.lower()
