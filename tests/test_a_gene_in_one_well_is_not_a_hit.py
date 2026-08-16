"""A gene represented by a single well corroborates nothing.

FOUND BY RUNNING THE SCAN ON REAL DATA, which nobody had done: plate1 of the
tsg101 screen, 60,816 cells aggregated to 44 wells carrying a dominant guide,
156 measurements, 10 genes -- of which EIGHT had exactly one well.

The scan reported 78 measurements surviving the across-scan correction. 64 of
them, 82%, rested on a gene with one well, and the largest effects in the
table were all singletons at 12 to 22 residual standard deviations.

Permuting the gene labels on that same real data -- so that no effect can
exist by construction -- showed what those numbers were worth:

    singletons kept       65% of permuted scans produced an across-scan
                          "survivor", against the 5% the correction promises
    singletons dropped     2% (4 of 200)

THE MECHANISM, and it is why this is not a cosmetic filter. A gene in one
well has no within-gene replication: its "effect" IS that well's deviation
from the rest, carrying every well-level artefact there is -- edge position,
seeding density, focus, a bubble -- with nothing to separate any of them from
a phenotype. And because the measurement columns are strongly correlated, one
outlier well does not produce one false hit. It produces dozens of correlated
false hits at once, which is precisely the structure a correction that
assumes valid P values cannot rescue. No amount of across-scan correction
fixes an input whose per-gene tests are anti-conservative.

spaCR already refuses this one level down: hits.FLAG_SINGLE_GUIDE is "called
by one guide, so nothing corroborates it". This is the same rule at the well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.measurement_scan import (MIN_WELLS_PER_GENE, ScanRefused,
                                    scan_measurements)


def _screen(n_measurements=40, seed=0, singletons=8, replicated_wells=16):
    """A screen shaped like plate1: two well-replicated genes and a tail of
    genes seen once, over correlated measurements with one outlier well."""
    rng = np.random.default_rng(seed)
    genes = (["geneA"] * replicated_wells + ["geneB"] * replicated_wells
             + [f"one_{i}" for i in range(singletons)])
    n = len(genes)
    # Correlated measurements: one latent factor per well plus column noise,
    # which is what makes a single odd well move dozens of columns together.
    latent = rng.normal(size=n)
    latent[-1] += 6.0                      # the outlier well, a singleton
    data = {f"m{j}": latent * rng.uniform(0.6, 1.0) + rng.normal(0, 0.4, n)
            for j in range(n_measurements)}
    frame = pd.DataFrame(data)
    frame["gene"] = genes
    return frame


# --------------------------------------------------------------------------- #
#  The default drops them, and says so
# --------------------------------------------------------------------------- #

def test_single_well_genes_are_dropped_by_default():
    result = scan_measurements(_screen(), gene_column="gene",
                               block_columns=())

    assert result.genes_dropped, "the singleton genes were kept"
    assert all(name.startswith("one_") for name in result.genes_dropped)
    assert set(result.genes_dropped.values()) == {1}


def test_the_dropped_genes_are_named_not_merely_counted():
    """A gene missing from the result with no explanation reads as a gene
    with no effect, which is the opposite of what happened to it."""
    result = scan_measurements(_screen(singletons=3), gene_column="gene",
                               block_columns=())

    assert sorted(result.genes_dropped) == ["one_0", "one_1", "one_2"]


def test_no_replicated_gene_is_dropped():
    result = scan_measurements(_screen(), gene_column="gene",
                               block_columns=())

    assert "geneA" not in result.genes_dropped
    assert "geneB" not in result.genes_dropped


def test_the_default_is_two():
    assert MIN_WELLS_PER_GENE == 2


# --------------------------------------------------------------------------- #
#  The calibration this exists for
# --------------------------------------------------------------------------- #

def test_keeping_them_wrecks_the_across_scan_rate():
    """The measurement, as a test. Permuted labels mean no effect can exist,
    so a scan calling anything significant is a false positive."""
    rng = np.random.default_rng(11)
    frame = _screen(seed=5)

    kept_hits = dropped_hits = 0
    runs = 25
    for _ in range(runs):
        shuffled = frame.copy()
        shuffled["gene"] = rng.permutation(shuffled["gene"].to_numpy())
        with_singletons = scan_measurements(
            shuffled, gene_column="gene", block_columns=(),
            min_wells_per_gene=1)
        without = scan_measurements(
            shuffled, gene_column="gene", block_columns=())
        kept_hits += any(r.survives_across_scan for r in with_singletons.rows)
        dropped_hits += any(r.survives_across_scan for r in without.rows)

    assert dropped_hits < kept_hits, (
        f"dropping single-well genes did not improve the false-positive rate "
        f"({dropped_hits}/{runs} against {kept_hits}/{runs})")
    assert dropped_hits <= runs * 0.2, (
        f"{dropped_hits}/{runs} permuted scans still produced an across-scan "
        f"survivor; the correction promises about {runs * 0.05:.0f}")


def test_a_real_effect_still_survives():
    """So the filter cannot pass by returning nothing ever.

    The top gene is whichever of the two is NOT the baseline. With the
    singletons gone there are two levels left, one is the reference, and the
    contrast between them is symmetric -- so shifting geneA shows up as geneB
    moving the other way. That is the same finding, not a different one.
    """
    frame = _screen(seed=2)
    for column in [c for c in frame.columns if c.startswith("m")][:5]:
        frame.loc[frame.gene == "geneA", column] += 8.0

    result = scan_measurements(frame, gene_column="gene", block_columns=())

    survivors = result.surviving()
    assert survivors, "a planted 8-sigma effect was corrected away"
    assert survivors[0].top_gene in {"geneA", "geneB"}
    assert abs(survivors[0].effect_size) > 2.0, survivors[0].effect_size
    assert survivors[0].measurement.startswith("m")


# --------------------------------------------------------------------------- #
#  The edges
# --------------------------------------------------------------------------- #

def test_the_caller_can_keep_them_knowing_that():
    result = scan_measurements(_screen(), gene_column="gene",
                               block_columns=(), min_wells_per_gene=1)

    assert result.genes_dropped == {}


def test_a_control_is_never_dropped_for_being_thin():
    """A control is the BASELINE, not a candidate. Dropping it would move the
    baseline to whichever gene sorted first, and every effect in the table
    would then be measured from somewhere the caller did not choose."""
    frame = _screen(singletons=2)
    frame.loc[frame.index[-1], "gene"] = "nc"          # a lone control well

    result = scan_measurements(frame, gene_column="gene", block_columns=(),
                               control_genes=["nc"])

    assert "nc" not in result.genes_dropped


def test_it_refuses_rather_than_comparing_one_gene_with_itself():
    frame = pd.DataFrame({
        "gene": ["a", "a", "b", "c"],
        "m0": [1.0, 2.0, 3.0, 4.0],
        "m1": [4.0, 3.0, 2.0, 1.0],
    })

    with pytest.raises(ScanRefused) as caught:
        scan_measurements(frame, gene_column="gene", block_columns=())

    message = str(caught.value)
    assert "nothing corroborating it" in message
    assert "min_wells_per_gene" in message, "the refusal names no way out"
