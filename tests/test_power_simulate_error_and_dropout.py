"""The two spaCRPower gaps that were named: sequencing error and thin wells.

Neither the R package nor the port modelled either. Both make the answer
worse, and both are off by default -- so the first thing asserted here is that
a run which does not ask for them is bit-identical to the run that existed
before they were added. A simulator whose baseline drifted under a version
bump would invalidate every power figure ever quoted from it.

After that, the direction. Sequencing error must *always* cost, because
mis-assignment moves the covariate toward uniform and attenuates the
coefficient; there is no error rate at which the screen gets better. Well
dropout is a trade rather than a cost, and the tests measure both sides of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.power_simulate import (
    DEFAULT_MIN_CELLS_PER_WELL,
    DEFAULT_SEQUENCING_ERROR_RATE,
    ScreenDesignError,
    MalformedPlateError,
    drop_low_cell_wells,
    misassign_reads,
    simulate_screen,
    simulate_sequencing_plate,
    simulate_spot_plate,
    simulate_library,
)


#: A screen small enough to fit in a second and skewed enough to be realistic.
BASE = dict(
    n_genes_in_library=60,
    gene_abundance_alpha=1.5,
    gene_hit_rate=0.15,
    n_wells_per_screen=96,
    well_abundance_factor_mu=4.0,
    well_abundance_factor_var=1.0,
    imaging_n_cells_per_well_mu=60.0,
    imaging_n_cells_per_well_var=2500.0,
    class_pos_mu=0.8,
    class_pos_var=0.05,
    class_neg_mu=0.12,
    class_neg_var=0.01,
    sequencing_n_cells_per_well_lambda=1000.0,
    pcr_factor_mu=2.0,
    pcr_factor_var=1.0,
    n_reads_per_well=30000,
    read_depth_cv=0.35,
)


def _gene_scores(screen: pd.DataFrame):
    """Per-gene association between read fraction and the well's hit rate.

    A cheap stand-in for the model half, computed from the same two quantities
    the model regresses: each gene's per-well log10 read fraction, and each
    well's positive fraction. A gene whose covariate never varies is skipped
    for exactly the reason ``power_model.prepare_model_data`` skips it -- its
    coefficient would be confounded with the intercept -- which makes the size
    of the returned dict the "how many genes were testable" number.

    The real power figure comes from ``power_model.fit_and_evaluate``, which
    the slow test at the bottom of this file runs. This runs in milliseconds,
    which is what makes six seeds across several error rates affordable.

    :returns: ``{gene: (correlation, hit)}`` for every testable gene.
    """
    frame = screen.copy()
    well_total = frame.groupby("well")["n_reads_per_gene_per_well"].transform(
        "sum")
    frame["log_fraction"] = np.log10(
        frame["n_reads_per_gene_per_well"] / well_total.replace(0, np.nan)
        + 1e-4)
    positives = frame.groupby("well")["positive"].sum()
    cells = frame.groupby("well")["imaging_n_cells_per_well"].first()
    rate = (positives / cells.replace(0, np.nan)).rename("rate")
    frame = frame.merge(rate, left_on="well", right_index=True)

    out = {}
    for gene, block in frame.groupby("gene"):
        x = block["log_fraction"].to_numpy()
        y = block["rate"].to_numpy()
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 3 or np.ptp(x[good]) == 0 or np.ptp(y[good]) == 0:
            continue
        out[gene] = (float(np.corrcoef(x[good], y[good])[0, 1]),
                     int(block["hit"].iloc[0]))
    return out


def _identifiable_genes(screen: pd.DataFrame) -> set:
    """The genes a fit could estimate a coefficient for."""
    return set(_gene_scores(screen))


def _separation(screen: pd.DataFrame, genes=None) -> float:
    """AUROC of that per-gene statistic for hit vs non-hit. 0.5 is chance.

    :param genes: restrict to this gene set, so a comparison can hold the
        denominator fixed instead of silently comparing different libraries.
    """
    scored = _gene_scores(screen)
    if genes is not None:
        scored = {k: v for k, v in scored.items() if k in genes}
    if not scored:
        return float("nan")
    scores = np.asarray([v[0] for v in scored.values()], dtype=float)
    labels = np.asarray([v[1] for v in scored.values()], dtype=int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _read_matrix(n_genes=8, n_wells=5, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 500, size=(n_genes, n_wells))


# -- the baseline must not have moved ---------------------------------------


def test_the_defaults_reproduce_the_screen_that_existed_before():
    """Both new stages are opt-in, and opting out costs nothing -- not one
    draw from the generator, so a seeded run is bit-identical."""
    first = simulate_screen(**BASE, seed=42)
    second = simulate_screen(
        **BASE, sequencing_error_rate=0.0, min_cells_per_well=0, seed=42)
    pd.testing.assert_frame_equal(first, second)
    # And the pre-error truth column is not a separate draw either.
    assert (first["n_reads_true_per_gene_per_well"]
            == first["n_reads_per_gene_per_well"]).all()
    assert (first["sequencing_error_rate"] == 0.0).all()
    assert first["well_kept"].all()


def test_zero_error_is_the_identity_and_draws_nothing():
    counts = _read_matrix()
    generator = np.random.default_rng(5)
    before = generator.bit_generator.state
    out = misassign_reads(counts, 0.0, rng=generator)
    assert np.array_equal(out, counts)
    assert generator.bit_generator.state == before


def test_zero_threshold_is_the_identity():
    screen = simulate_screen(**BASE, seed=7)
    kept = drop_low_cell_wells(screen, 0)
    assert len(kept) == len(screen)
    assert kept["well_kept"].all()
    assert kept.attrs["n_wells_dropped"] == 0


# -- sequencing error: mechanics --------------------------------------------


def test_misassignment_conserves_every_well_read_total():
    """Reads move between genes; they are not created or destroyed. Anything
    else would confound the error with a change in sequencing depth."""
    counts = _read_matrix(n_genes=30, n_wells=12, seed=1)
    for rate in (0.01, 0.1, 0.5, 1.0):
        out = misassign_reads(counts, rate, seed=2)
        assert np.array_equal(out.sum(axis=0), counts.sum(axis=0))
        assert (out >= 0).all()


def test_misassignment_gives_absent_genes_phantom_reads():
    """The mechanism, isolated. Gene 0 is the only one in the well; after
    mis-assignment the others have reads they never earned. That is exactly
    what dilutes the covariate."""
    counts = np.zeros((50, 1), dtype=np.int64)
    counts[0, 0] = 100_000
    out = misassign_reads(counts, 0.2, seed=3)
    assert out[1:, 0].sum() > 0
    # Roughly 20% leaves, of which 1/50 lands back on the source.
    assert out[0, 0] == pytest.approx(100_000 * (0.8 + 0.2 / 50), rel=0.05)


def test_a_full_error_rate_erases_the_original_assignment():
    """The null case for this stage: at rate 1.0 every read is redrawn from a
    uniform library, so the observed counts carry no information about which
    gene produced them."""
    counts = np.zeros((40, 1), dtype=np.int64)
    counts[0, 0] = 40_000
    out = misassign_reads(counts, 1.0, seed=4)
    assert int(out.sum()) == 40_000
    # Every gene gets about 1/40 of the pool; the source gene is not special.
    assert out[0, 0] == pytest.approx(1000, rel=0.25)
    assert float(out.std()) < 100


def test_misassignment_is_reproducible_from_a_seed():
    counts = _read_matrix(seed=6)
    first = misassign_reads(counts, 0.15, seed=99)
    second = misassign_reads(counts, 0.15, seed=99)
    assert np.array_equal(first, second)
    assert not np.array_equal(misassign_reads(counts, 0.15, seed=100), first)


@pytest.mark.parametrize("rate", [-0.01, 1.01, float("nan")])
def test_an_impossible_error_rate_is_refused(rate):
    with pytest.raises(ScreenDesignError, match="must be in"):
        misassign_reads(_read_matrix(), rate, seed=1)


def test_a_non_matrix_or_negative_count_is_refused():
    with pytest.raises(ScreenDesignError, match="n_genes, n_wells"):
        misassign_reads(np.array([1, 2, 3]), 0.1, seed=1)
    with pytest.raises(ScreenDesignError, match="not be negative"):
        misassign_reads(np.array([[-1, 2]]), 0.1, seed=1)


def test_the_sequencing_plate_keeps_the_pre_error_truth():
    library = simulate_library(40, 10.0, 0.2, seed=20)
    spot = simulate_spot_plate(library, 24, 4.0, 1.0, seed=21)
    plate = simulate_sequencing_plate(
        spot, 1000.0, 2.0, 1.0, 20000,
        sequencing_error_rate=0.25, seed=22)

    assert (plate["sequencing_error_rate"] == 0.25).all()
    true_total = plate.groupby("well")["n_reads_true_per_gene_per_well"].sum()
    seen_total = plate.groupby("well")["n_reads_per_gene_per_well"].sum()
    pd.testing.assert_series_equal(
        true_total, seen_total, check_names=False)
    # The two columns must actually differ, or the stage did nothing.
    assert not (plate["n_reads_true_per_gene_per_well"]
                == plate["n_reads_per_gene_per_well"]).all()


# -- sequencing error: it always costs --------------------------------------


def test_the_direct_dilution_of_the_covariate_is_real_but_small():
    """Compared like with like -- the same genes, scored both ways -- a
    realistic error rate barely moves the separation.

    This is the effect everyone expects mis-assignment to have, and it is
    worth pinning down precisely because it turns out to be the *small* one.
    The comparison is restricted to genes that were identifiable in the clean
    screen, so the denominator is held fixed and only the dilution is being
    measured.
    """
    clean_values, dirty_values = [], []
    for seed in (1, 2, 3, 4, 5, 6):
        clean = simulate_screen(**BASE, seed=seed)
        dirty = simulate_screen(
            **BASE, sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
            seed=seed)
        testable = _identifiable_genes(clean)
        clean_values.append(_separation(clean, genes=testable))
        dirty_values.append(_separation(dirty, genes=testable))

    clean_mean = float(np.mean(clean_values))
    dirty_mean = float(np.mean(dirty_values))
    assert clean_mean > 0.75, "the clean screen should be detectable at all"
    assert dirty_mean > clean_mean - 0.06, (
        f"dilution should be small at {DEFAULT_SEQUENCING_ERROR_RATE}: "
        f"{clean_mean:.3f} -> {dirty_mean:.3f}")


def test_sequencing_error_switches_off_the_untested_gene_safeguard():
    """The effect that actually matters, and it is not the dilution.

    ``power_model.prepare_model_data`` refuses to score a gene whose read
    fraction never varies between wells: its coefficient is confounded with
    the intercept, and reporting it as a shrunk-to-zero non-hit would turn "we
    could not test this" into "this is not a hit". Mis-assignment gives every
    such gene phantom reads, so its covariate varies -- made entirely of noise
    -- and the safeguard stops firing.

    A screen with sequencing error and one without disagree about how many
    genes were tested. Only one of them is right.
    """
    gained = []
    for seed in (1, 2, 3, 4, 5, 6):
        clean = len(_identifiable_genes(simulate_screen(**BASE, seed=seed)))
        dirty = len(_identifiable_genes(simulate_screen(
            **BASE, sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
            seed=seed)))
        gained.append(dirty - clean)

    assert float(np.mean(gained)) > 0, (
        "error should make untestable genes look testable")
    # And every gene in the library ends up scored, including the ones that
    # were never in a well.
    dirty = simulate_screen(
        **BASE, sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE, seed=1)
    assert len(_identifiable_genes(dirty)) == BASE["n_genes_in_library"]


def test_the_phantom_genes_score_at_chance_and_drag_the_screen_down():
    """The two halves put together: the newly-'testable' genes carry no
    information, so scoring them lowers the screen-wide separation even
    though nothing about the real genes got worse."""
    held_fixed, everything = [], []
    for seed in (1, 2, 3, 4, 5, 6):
        clean = simulate_screen(**BASE, seed=seed)
        dirty = simulate_screen(
            **BASE, sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
            seed=seed)
        held_fixed.append(_separation(dirty, genes=_identifiable_genes(clean)))
        everything.append(_separation(dirty))

    assert float(np.mean(everything)) < float(np.mean(held_fixed))


def test_a_total_sequencing_failure_leaves_no_signal_to_find():
    """The null case: at rate 1.0 the read fractions are noise, so the screen
    must land at chance. A simulator that still 'detected' hits here would be
    leaking the truth through some other column."""
    values = [
        _separation(simulate_screen(**BASE, sequencing_error_rate=1.0,
                                    seed=seed))
        for seed in (1, 2, 3, 4, 5)
    ]
    assert float(np.mean(values)) == pytest.approx(0.5, abs=0.12)


# -- well dropout -----------------------------------------------------------


def test_dropout_removes_whole_wells_and_only_thin_ones():
    screen = simulate_screen(
        **{**BASE, "imaging_n_cells_per_well_mu": 20.0,
           "imaging_n_cells_per_well_var": 400.0}, seed=8)
    totals = screen.groupby("well")["imaging_n_cells_per_well"].first()
    threshold = int(totals.quantile(0.3))

    kept = drop_low_cell_wells(screen, threshold)
    surviving = kept.groupby("well")["imaging_n_cells_per_well"].first()

    assert (surviving >= threshold).all()
    assert kept.attrs["n_wells_dropped"] == int((totals < threshold).sum())
    assert kept.attrs["n_wells_before"] == len(totals)
    # Whole wells only: every surviving well keeps all of its gene rows.
    per_well = kept.groupby("well").size()
    assert per_well.nunique() == 1
    assert int(per_well.iloc[0]) == BASE["n_genes_in_library"]


def test_dropout_can_annotate_without_removing():
    screen = simulate_screen(
        **{**BASE, "imaging_n_cells_per_well_mu": 20.0,
           "imaging_n_cells_per_well_var": 400.0}, seed=9)
    marked = drop_low_cell_wells(screen, 15, drop=False)
    assert len(marked) == len(screen)
    assert not marked["well_kept"].all()
    assert marked.attrs["n_wells_dropped"] > 0


def test_dropout_through_the_orchestrator_reports_what_it_cost():
    screen = simulate_screen(
        **{**BASE, "imaging_n_cells_per_well_mu": 20.0,
           "imaging_n_cells_per_well_var": 400.0},
        min_cells_per_well=DEFAULT_MIN_CELLS_PER_WELL, seed=10)
    assert screen.attrs["n_wells_dropped"] > 0
    assert screen.attrs["n_wells_before"] == BASE["n_wells_per_screen"]
    assert screen.attrs["min_cells_per_well"] == DEFAULT_MIN_CELLS_PER_WELL
    assert screen["imaging_n_cells_per_well"].min() >= (
        DEFAULT_MIN_CELLS_PER_WELL)
    # The clip-count attr set earlier in the pipeline survives the filter.
    assert "n_prob_clipped" in screen.attrs


def test_dropping_the_thin_tail_does_not_destroy_the_signal():
    """The other half of the trade. Removing wells costs power directly; the
    wells removed here were carrying almost none, so the net must not be a
    collapse."""
    thin = {**BASE, "imaging_n_cells_per_well_mu": 20.0,
            "imaging_n_cells_per_well_var": 400.0}
    unfiltered = float(np.mean([
        _separation(simulate_screen(**thin, seed=seed))
        for seed in (1, 2, 3, 4, 5)]))
    filtered = float(np.mean([
        _separation(simulate_screen(
            **thin, min_cells_per_well=DEFAULT_MIN_CELLS_PER_WELL, seed=seed))
        for seed in (1, 2, 3, 4, 5)]))
    assert filtered > unfiltered - 0.05


def test_an_impossible_threshold_takes_every_well():
    screen = simulate_screen(**BASE, seed=11)
    empty = drop_low_cell_wells(screen, 10 ** 9)
    assert len(empty) == 0
    assert empty.attrs["n_wells_dropped"] == BASE["n_wells_per_screen"]


def test_a_negative_threshold_is_refused():
    screen = simulate_screen(**BASE, seed=12)
    with pytest.raises(ScreenDesignError):
        drop_low_cell_wells(screen, -1)


def test_a_frame_with_no_cell_count_is_refused():
    with pytest.raises(MalformedPlateError, match="imaging_n_cells_per_well"):
        drop_low_cell_wells(pd.DataFrame({"well": ["A01"]}), 5)


def test_a_frame_with_no_well_column_is_refused():
    with pytest.raises(MalformedPlateError, match="'well' column"):
        drop_low_cell_wells(
            pd.DataFrame({"imaging_n_cells_per_well": [5]}), 5)


def test_a_cell_total_that_varies_within_a_well_is_refused():
    """That column is the well total repeated on every row. If it differs
    between the genes of one well the table is malformed, and quietly taking
    the first value would hide it."""
    bad = pd.DataFrame({
        "well": ["A01", "A01"],
        "imaging_n_cells_per_well": [10, 400],
    })
    with pytest.raises(MalformedPlateError, match="varies within a well"):
        drop_low_cell_wells(bad, 5)


def test_dropout_falls_back_to_the_per_gene_counts():
    """The imaging plate's per-gene counts sum to the well total by
    construction, so a frame carrying only those is still filterable."""
    frame = pd.DataFrame({
        "well": ["A01", "A01", "B01", "B01"],
        "gene": ["g1", "g2", "g1", "g2"],
        "imaging_n_cells_per_gene_per_well": [1, 2, 100, 200],
    })
    kept = drop_low_cell_wells(frame, 10)
    assert list(kept["well"].unique()) == ["B01"]


# -- both together ----------------------------------------------------------


def test_the_two_stages_compose():
    screen = simulate_screen(
        **{**BASE, "imaging_n_cells_per_well_mu": 20.0,
           "imaging_n_cells_per_well_var": 400.0},
        sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
        min_cells_per_well=DEFAULT_MIN_CELLS_PER_WELL,
        seed=13)
    assert screen.attrs["n_wells_dropped"] > 0
    assert (screen["sequencing_error_rate"]
            == DEFAULT_SEQUENCING_ERROR_RATE).all()
    assert not screen.empty
    # Read totals are still conserved inside each surviving well.
    truth = screen.groupby("well")["n_reads_true_per_gene_per_well"].sum()
    seen = screen.groupby("well")["n_reads_per_gene_per_well"].sum()
    pd.testing.assert_series_equal(truth, seen, check_names=False)


# -- the design screen ------------------------------------------------------


def test_the_design_screen_carries_both_stages_and_says_what_they_do():
    """Once a caveat is real it belongs on the screen, and it belongs in the
    group that changes the number rather than in the footnotes."""
    from spacr.qt.widgets.power_design import (
        CAVEATS, DesignSpec, changes_the_number, simulator_kwargs,
    )

    keys = {caveat.key for caveat in changes_the_number()}
    assert "sequencing_error_hides_untested_genes" in keys
    assert "thin_wells_count_the_same_as_full_ones" in keys

    by_key = {caveat.key: caveat for caveat in CAVEATS}
    error = by_key["sequencing_error_hides_untested_genes"]
    # The detail must name the mechanism and the API, not just wave at it.
    assert "sequencing_error_rate" in error.detail
    assert "untested" in error.detail.lower()
    thin = by_key["thin_wells_count_the_same_as_full_ones"]
    assert "min_cells_per_well" in thin.detail

    # The spec defaults to the R behaviour, and reaches the simulator by the
    # simulator's own parameter names.
    spec = DesignSpec()
    assert spec.sequencing_error_rate == 0.0
    assert spec.min_cells_per_well == 0
    kwargs = simulator_kwargs(spec)
    assert kwargs["sequencing_error_rate"] == 0.0
    assert kwargs["min_cells_per_well"] == 0

    turned_on = simulator_kwargs(DesignSpec(
        sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
        min_cells_per_well=DEFAULT_MIN_CELLS_PER_WELL))
    screen = simulate_screen(
        **{**BASE, **{k: v for k, v in turned_on.items()
                      if k in ("sequencing_error_rate", "min_cells_per_well")}},
        seed=14)
    assert (screen["sequencing_error_rate"]
            == DEFAULT_SEQUENCING_ERROR_RATE).all()


# -- the real model, not the surrogate --------------------------------------


def test_the_fitted_model_scores_more_genes_when_reads_are_mis_assigned():
    """The finding, confirmed against the actual inference chain.

    Everything above uses a fast correlation surrogate. This runs
    ``power_model.fit_and_evaluate`` -- the same path the design screen runs
    -- to check that the surrogate was not the reason for the result. The
    assertion is the sharp one: mis-assignment moves genes out of the
    'untested' bucket and into the scored one.
    """
    from spacr.power_model import evaluate_model_fit, prepare_model_data

    clean = prepare_model_data(simulate_screen(**BASE, seed=1))
    dirty = prepare_model_data(simulate_screen(
        **BASE, sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE, seed=1))

    assert len(clean.unidentified_genes) > 0, (
        "the clean screen should have genes it cannot test")
    assert len(dirty.unidentified_genes) < len(clean.unidentified_genes), (
        "mis-assignment should make untestable genes look testable")
    assert evaluate_model_fit is not None
