"""What the barcode-QC report says when a piece of its input is not there.

Every function in :mod:`spacr.sequencing_qc` writes a number or a sentence a
user will quote: a collision rate that goes in a methods section, a
recommended abundance threshold, four panels that go in a figure. Each of
them has a branch for the case where the evidence behind one of those
statements is missing -- a plate whose mapping produced no reads, a sweep
evaluated at a single point, a starved-well table that came back from CSV
without its cutoff, a plate contributing no guides, a ``qc.csv`` read
without its count table.

Those branches are the ones that decide whether the report goes quiet or
says something it cannot support, and a quiet report is exactly what nobody
notices. So each test below drives BOTH sides: the input that produces the
statement, and the input that must withhold it.
"""
from __future__ import annotations

import io

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacr.sequencing_qc as QC

# ---------------------------------------------------------------------------
# One small plate, built so the ground truth of every assertion is known:
# 4 rows x 6 columns, three real guides per well, one well sequenced shallow.
# ---------------------------------------------------------------------------

_STARVED_WELL = ("r1", "c1")


def _counts(depth=6000, starved_depth=180, n_rows=4, n_cols=6, per_well=3):
    """A normalised count table with one deliberately starved well."""
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            row_id, col_id = f"r{r + 1}", f"c{c + 1}"
            reads = starved_depth if (row_id, col_id) == _STARVED_WELL else depth
            for k in range(per_well):
                rows.append({"plateID": "plate1", "rowID": row_id,
                             "columnID": col_id,
                             "grna": f"sg{(r * n_cols + c + k) % 12}",
                             "count": reads // (per_well + 1)})
            # bleed-through, so a threshold has something to cut against
            rows.append({"plateID": "plate1", "rowID": row_id,
                         "columnID": col_id, "grna": "junk0",
                         "count": max(1, reads // 60)})
    return QC.load_count_table(pd.DataFrame(rows))


def _panels(counts, *, per_well, starved, positions, depth, unmapped):
    """Draw the QC figure and read back what each panel actually carries."""
    fig = QC.plot_barcode_qc(counts, per_well=per_well, starved=starved,
                             positions=positions, depth=depth,
                             unmapped=unmapped)
    try:
        reads, _position, coverage, fate = fig.axes
        return {
            "reads_lines": len(reads.lines),
            "reads_legend": (None if reads.get_legend() is None
                             else reads.get_legend().get_texts()[0].get_text()),
            "coverage_lines": len(coverage.lines),
            "coverage_title": coverage.get_title(),
            "fate_bars": [t.get_text() for t in fate.get_xticklabels()],
            "fate_heights": [p.get_height() for p in fate.patches],
        }
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_barcode_qc, panel 2: physical plate order
# ---------------------------------------------------------------------------

def test_the_position_panel_sorts_numbered_labels_naturally():
    """Column 10 follows column 2 rather than sorting between 1 and 2."""
    counts = _counts(n_rows=1, n_cols=3)
    positions = pd.DataFrame({
        "axis": ["column", "column", "column"],
        "label": ["c10", "c2", "c1"],
        "ratio_to_plate": [1.0, 1.0, 1.0],
        "flagged": [False, False, False],
    })

    fig = QC.plot_barcode_qc(
        counts,
        per_well=QC.reads_per_well(counts),
        starved=QC.starved_wells(counts),
        positions=positions,
        depth=QC.library_depth(counts),
        unmapped=None,
    )
    try:
        assert [tick.get_text() for tick in fig.axes[1].get_xticklabels()] == [
            "c1", "c2", "c10"
        ]
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# collision_summary: reads at risk, when there are no reads to be at risk
# ---------------------------------------------------------------------------

def test_a_plate_with_no_mapped_reads_reports_no_reads_at_risk():
    """A collision rate is only alarming once it is priced in reads.

    ``collision_summary`` turns "two barcodes are one substitution apart"
    into "this share of your data is at risk", which is the number that
    decides whether a near-collision matters. The share is a division by
    the run's total reads, so a plate whose mapping produced nothing has
    no denominator: the column has to come back empty rather than as a
    reassuring 0.0, which a reader would take as "no reads affected"
    instead of "not measured". Both are checked here against the same
    reference, so the empty answer cannot be an artefact of a summary that
    never looked at the counts.
    """
    counts = _counts()
    references = {"grna": {"sg0": "ACGTAC", "sg1": "ACGTAA", "sg2": "TTGGCC"}}
    collisions = QC.barcode_collisions(references)
    assert list(zip(collisions["name_a"], collisions["name_b"])) == [("sg0", "sg1")]

    measured = QC.collision_summary(references, collisions, counts=counts)
    at_risk = float(measured.loc[0, "reads_at_risk"])
    sg01 = float(counts[counts["grna"].isin(["sg0", "sg1"])]["count"].sum())
    assert at_risk == pytest.approx(sg01 / counts["count"].sum())
    assert at_risk > 0.0

    # The same reference, against a plate the mapping run produced nothing
    # for -- a real shape, because a caller QC's one plate of a batch at a
    # time by filtering the pooled table.
    empty = counts[counts["plateID"] == "plate2"]
    assert empty.empty and float(empty["count"].sum()) == 0.0
    unmeasured = QC.collision_summary(references, collisions, counts=empty)
    assert unmeasured.loc[0, "reads_at_risk"] is None
    # ...and the parts that do not need reads are still reported.
    assert int(unmeasured.loc[0, "n_barcodes"]) == 3
    assert int(unmeasured.loc[0, "n_colliding_pairs"]) == 1
    assert unmeasured.loc[0, "collision_rate"] == pytest.approx(2 / 3)


def test_only_grna_collisions_are_priced_in_grna_counts():
    """A missing measurement is not evidence that no reads are at risk.

    The normalised count table prices guide abundance. It also names the row
    carried by each read, but ``collision_summary`` has no contract for using
    that column to price row or column barcodes. Reporting zero for them would
    turn "not measured" into the reassuring and unsupported "none affected".
    """
    references = {
        "grna": {"sg0": "AAAA", "sg1": "AAAT"},
        "row": {"r1": "CCCC", "r2": "CCCT"},
    }
    counts = pd.DataFrame({
        "rowID": ["r1", "r1"],
        "grna": ["sg0", "sg1"],
        "count": [2, 1],
    })
    collisions = QC.barcode_collisions(references)

    summary = QC.collision_summary(references, collisions, counts)
    by_reference = summary.set_index("reference")

    assert by_reference.loc["grna", "reads_at_risk"] == pytest.approx(1.0)
    assert pd.isna(by_reference.loc["row", "reads_at_risk"])
    assert int(by_reference.loc["row", "n_colliding_pairs"]) == 1
    assert by_reference.loc["row", "collision_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# recommend_threshold: the sentences a one-point sweep cannot support
# ---------------------------------------------------------------------------

def test_a_one_point_sweep_recommends_without_inventing_a_trade_off():
    """The recommendation must not quote a cost it never evaluated.

    Most of ``recommend_threshold`` is comparative: what relaxing the
    cutoff buys, what tightening it costs, and where the collision rate
    turns. Every one of those sentences is read off OTHER rows of the
    sweep. A user who evaluates the sweep at the derived point alone --
    the cheapest possible call, and the one a script makes when it only
    wants the headline -- has no other rows, and the report has to fall
    silent on all three rather than quote the derived row back as if it
    were a neighbour. The full sweep is derived from the same table in
    the same test, so the missing sentences are demonstrably missing
    rather than never implemented.
    """
    counts = _counts()
    choice = QC.derive_threshold(counts, 3)
    assert choice.attainable and choice.achieved == pytest.approx(3.0)

    grid = QC.sweep_grid(choice.threshold, span=8.0, points=25)
    full = QC.recommend_threshold(QC.threshold_sweep(counts, grid, 3), choice)
    assert "Relaxing to" in full
    assert "Tightening" in full
    assert "rises sharply" in full

    single = QC.threshold_sweep(counts, [choice.threshold], 3)
    assert len(single) == 1
    lone = QC.recommend_threshold(single, choice)
    assert "Relaxing to" not in lone
    assert "Tightening" not in lone
    assert "rises sharply" not in lone
    # What it still says: the derived number and what it buys.
    assert f"{choice.threshold:.4f}" in lone
    assert "gRNAs per well (median)" in lone
    assert "100% of wells are retained" in lone
    assert lone.splitlines()[0].startswith("Target: 3 gRNAs per well")


# ---------------------------------------------------------------------------
# plot_barcode_qc, panel 1: the starvation cut line
# ---------------------------------------------------------------------------

def test_a_starved_table_that_lost_its_cutoff_draws_no_cut_line():
    """A cut line drawn at the wrong place is worse than no cut line.

    Panel 1 marks the starvation cutoff on the read-depth histogram, and
    it takes that cutoff from ``starved.attrs['cutoff']`` -- a pandas
    attribute that does not survive a round trip through CSV. A run that
    is resumed from its written QC tables therefore arrives with the
    starved wells but not the number that defined them, and the panel
    must draw the histogram with no line rather than a line at zero
    labelled "starved below 0 reads", which would tell the reader every
    well passed. Both frames hold the same one starved well, so the
    difference is the cutoff and nothing else.
    """
    counts = _counts()
    per_well = QC.reads_per_well(counts)
    positions = QC.position_effects(counts)
    depth = QC.library_depth(counts)
    starved = QC.starved_wells(counts)
    assert len(starved) == 1
    assert starved.iloc[0]["rowID"] == _STARVED_WELL[0]
    assert starved.attrs["cutoff"] > 0

    marked = _panels(counts, per_well=per_well, starved=starved,
                     positions=positions, depth=depth, unmapped=None)
    assert marked["reads_lines"] == 1
    assert "starved below" in marked["reads_legend"]
    assert "1 well(s)" in marked["reads_legend"]

    # The same table, resumed from disk: same rows, no attrs.
    resumed = pd.read_csv(io.StringIO(starved.to_csv(index=False)))
    assert len(resumed) == len(starved)
    assert "cutoff" not in resumed.attrs
    plain = _panels(counts, per_well=per_well, starved=resumed,
                    positions=positions, depth=depth, unmapped=None)
    assert plain["reads_lines"] == 0
    assert plain["reads_legend"] is None


# ---------------------------------------------------------------------------
# plot_barcode_qc, panel 3: the Lorenz curve of library coverage
# ---------------------------------------------------------------------------

def test_a_plate_with_no_guides_leaves_the_coverage_panel_blank():
    """A Lorenz curve of nothing is a diagonal, and a diagonal means "even".

    Panel 3 draws cumulative read share against gRNA rank, and its
    reference line is the diagonal that perfect evenness would follow. If
    the depth summary carries no per-gRNA reads -- what
    ``library_depth`` returns for a plate the mapping run produced
    nothing for -- normalising by the total would divide by zero, and
    whatever came out would be plotted against that diagonal as though it
    were a measured library. The panel has to stay empty and unlabelled
    instead. The populated summary is drawn from the same counts in the
    same test, so the blank panel is a decision and not a dead plotting
    call.
    """
    counts = _counts()
    per_well = QC.reads_per_well(counts)
    positions = QC.position_effects(counts)
    starved = QC.starved_wells(counts)

    measured = QC.library_depth(counts)
    assert measured["n_grnas_observed"] == 13  # 12 real guides + junk0
    drawn = _panels(counts, per_well=per_well, starved=starved,
                    positions=positions, depth=measured, unmapped=None)
    assert drawn["coverage_lines"] == 2  # the curve and the parity diagonal
    assert "Library coverage" in drawn["coverage_title"]
    assert f"Gini {measured['gini']:.2f}" in drawn["coverage_title"]

    absent = QC.library_depth(counts[counts["plateID"] == "plate2"])
    assert absent["n_grnas_observed"] == 0
    assert len(absent["reads_per_grna"]) == 0
    blank = _panels(counts, per_well=per_well, starved=starved,
                    positions=positions, depth=absent, unmapped=None)
    assert blank["coverage_lines"] == 0
    assert blank["coverage_title"] == ""


# ---------------------------------------------------------------------------
# plot_barcode_qc, panel 4: the joint unmapped bar
# ---------------------------------------------------------------------------

def test_the_read_fate_panel_adds_the_joint_bar_only_with_a_count_table():
    """The "any field" bar is an exact number, not a summary of the others.

    Per-field unmapped fractions cannot be added up: a read lost by two
    fields is lost once. ``unmapped_read_fractions`` therefore reports the
    joint ``unmapped_fraction`` only when it is handed the count table,
    where the reads that survived ALL THREE lookups actually are. Panel 4
    has to follow that: with the count table it draws the joint bar
    alongside the fields; without it, only the fields -- because an "any
    field" bar the caller could not measure would be read as the run's
    true loss rate and used to decide whether the run is usable.
    """
    counts = _counts()
    per_well = QC.reads_per_well(counts)
    positions = QC.position_effects(counts)
    starved = QC.starved_wells(counts)
    depth = QC.library_depth(counts)
    total = float(counts["count"].sum()) * 1.25
    qc_table = pd.DataFrame({"total_reads": [total], "columnID": [total * 0.02],
                             "rowID": [total * 0.01],
                             "grna_name": [total * 0.05]})

    joint = QC.unmapped_read_fractions(qc_table, counts=counts)
    assert joint["unmapped_fraction"] == pytest.approx(0.2, abs=1e-6)
    with_joint = _panels(counts, per_well=per_well, starved=starved,
                         positions=positions, depth=depth, unmapped=joint)
    assert with_joint["fate_bars"] == ["columnID", "rowID", "grna_name",
                                       "any field"]
    assert with_joint["fate_heights"][-1] == pytest.approx(20.0, abs=1e-4)
    assert with_joint["fate_heights"][0] == pytest.approx(2.0, abs=1e-4)

    fields_only = QC.unmapped_read_fractions(qc_table)
    assert "unmapped_fraction" not in fields_only
    assert fields_only["unmapped_fraction_upper"] == pytest.approx(0.08)
    without_joint = _panels(counts, per_well=per_well, starved=starved,
                            positions=positions, depth=depth,
                            unmapped=fields_only)
    assert without_joint["fate_bars"] == ["columnID", "rowID", "grna_name"]
    assert without_joint["fate_heights"] == pytest.approx([2.0, 1.0, 5.0],
                                                          abs=1e-4)
