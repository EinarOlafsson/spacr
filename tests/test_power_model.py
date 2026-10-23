"""Tests for :mod:`spacr.power_model` -- the inference half of the spaCRPower port.

The thing being defended against here is specific and has a name in this
codebase: a sparse-regression implementation that returns plausible
numbers while recovering nothing.  Every "it works" claim in this file is
therefore an assertion on a case whose answer is known in advance:

* :func:`test_signal_recovery_auroc_is_high` plants five strong hits and
  requires AUROC >= 0.9.  A fit that recovers nothing scores ~0.5 here
  and a fit whose sign convention is inverted scores ~0.
* :func:`test_null_screen_auroc_is_near_chance` removes the effect
  entirely -- hits are still *labelled*, they just do nothing -- and
  requires AUROC near 0.5 across four independent screens.  This is the
  test that catches a method that reports 0.95 on noise.
* :func:`test_evaluate_orientation_is_not_inverted` and
  :func:`test_signal_recovery_is_not_the_inverted_convention` pin the
  sign convention from both ends, so a test suite that would pass at
  both 0.95 and 0.05 cannot exist here.
* :func:`test_horseshoe_shrinks_non_hits_toward_zero` requires the
  non-hit coefficients to collapse relative to the hits, and
  :func:`test_null_screen_coefficients_collapse_to_numerical_zero`
  requires the spike at zero that only a sparsity-inducing prior gives.
  Both were checked against a mutant with the horseshoe replaced by a
  plain ridge; the second catches it by a factor of two hundred, the
  first does not catch it at all, and each says so.

The screen simulator used below is local to this file on purpose.  The
matching simulator, :mod:`spacr.power_simulate`, is a separate module;
these tests are written against the *column contract* the two halves
share, so they stay meaningful whether or not it is installed.
:func:`test_integration_with_power_simulate` closes that loop when it is.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr.power_model import (  # noqa: E402
    BACKENDS,
    EXPRESSION_PSEUDOCOUNT,
    ModelData,
    PowerFit,
    PowerFitError,
    available_backends,
    evaluate_model_fit,
    fit_and_evaluate,
    fit_model,
    gather_model_estimate,
    prepare_model_data,
    resolve_backend,
    scan_parameters,
)

# ADVI steps for the tests. 800 is enough for the signal case to reach
# AUROC 1.0 and for the ELBO to satisfy the convergence check; the whole
# file then runs in well under a minute on CPU.
FIT_STEPS = 800


# ---------------------------------------------------------------------------
# A screen simulator, local to the tests
# ---------------------------------------------------------------------------

def simulate_screen_frame(
    *,
    n_genes: int = 60,
    n_wells: int = 120,
    n_hits: int = 5,
    genes_per_well: int = 8,
    cells_per_well: int = 800,
    class_pos: float = 0.60,
    class_neg: float = 0.05,
    reads_per_well: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a pooled screen as a tidy ``(gene, well)`` table.

    A deliberately minimal stand-in for
    ``spacr.power_simulate.simulate_screen``: it emits the same columns
    :func:`spacr.power_model.prepare_model_data` documents as required,
    and nothing else.  Keeping it here means these tests exercise the
    *contract* between the two halves rather than one particular
    implementation of the other half.

    Each well gets a random subset of the library; the cells in the well
    are split between those genes; a cell is called positive with
    probability ``class_pos`` if its gene is a hit and ``class_neg`` if
    it is not; reads are apportioned in the same proportions as cells.

    :param n_genes: library size.
    :param n_wells: wells in the screen.
    :param n_hits: how many genes are true hits.
    :param genes_per_well: expected distinct genes landing in a well.
    :param cells_per_well: mean imaged cells per well.
    :param class_pos: probability a hit-gene cell is called positive
        (classifier sensitivity, times the real biology).
    :param class_neg: probability a non-hit cell is called positive
        (1 - specificity).  ``class_pos == class_neg`` makes hit status
        irrelevant to the readout, which is the null screen.
    :param reads_per_well: sequencing depth per well.
    :param seed: RNG seed; the table is a pure function of it.
    :returns: tidy ``DataFrame`` with one row per ``(gene, well)`` and
        columns ``gene``, ``well``, ``hit``,
        ``imaging_n_cells_per_gene_per_well``,
        ``imaging_n_cells_per_well``, ``positive``,
        ``n_reads_per_gene_per_well``.
    """
    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i:03d}" for i in range(n_genes)])
    hit = np.zeros(n_genes, dtype=int)
    hit[rng.choice(n_genes, size=n_hits, replace=False)] = 1

    p_in_well = genes_per_well / n_genes
    records = []
    for well in range(n_wells):
        present = rng.random(n_genes) < p_in_well
        if not present.any():
            # A well with nothing in it carries no contrast; force one gene in
            # so the simulator cannot accidentally produce a degenerate design
            # and make a real failure look like a simulator artefact.
            present[rng.integers(n_genes)] = True
        weights = np.zeros(n_genes)
        weights[present] = rng.gamma(3.0, 1.0, size=int(present.sum()))
        weights /= weights.sum()

        total_cells = int(rng.poisson(cells_per_well))
        cells = rng.multinomial(total_cells, weights)
        positive = rng.binomial(cells, np.where(hit == 1, class_pos, class_neg))
        reads = rng.multinomial(reads_per_well, weights)
        for g in range(n_genes):
            records.append(
                (
                    genes[g],
                    well,
                    int(hit[g]),
                    int(cells[g]),
                    total_cells,
                    int(positive[g]),
                    int(reads[g]),
                )
            )
    return pd.DataFrame(
        records,
        columns=[
            "gene",
            "well",
            "hit",
            "imaging_n_cells_per_gene_per_well",
            "imaging_n_cells_per_well",
            "positive",
            "n_reads_per_gene_per_well",
        ],
    )


@pytest.fixture(scope="module")
def signal_screen() -> pd.DataFrame:
    """A screen with five strong hits; the answer is known."""
    return simulate_screen_frame(seed=1)


@pytest.fixture(scope="module")
def signal_fit(signal_screen):
    """``(fit, estimate, evaluation)`` for :func:`signal_screen`."""
    return fit_and_evaluate(
        signal_screen, backend="torch", seed=0, n_steps=FIT_STEPS
    )


# ---------------------------------------------------------------------------
# 1. prepare_model_data
# ---------------------------------------------------------------------------

def test_prepare_model_data_matches_the_documented_formulas():
    """Npositive, Ntotal and log10expression are exactly what the R defines."""
    tidy = pd.DataFrame(
        {
            "well": ["w1", "w1", "w2", "w2"],
            "gene": ["A", "B", "A", "B"],
            "positive": [1, 0, 0, 9],
            "imaging_n_cells_per_gene_per_well": [50, 50, 10, 90],
            "n_reads_per_gene_per_well": [500, 500, 100, 900],
        }
    )
    md = prepare_model_data(tidy)

    assert list(md.wells) == ["w1", "w2"]
    assert list(md.genes) == ["A", "B"]
    # Npositive is the well total across genes -- NOT the first gene's count,
    # which is what the R's `sum(well_data$positive[1])` actually computes.
    assert list(md.Npositive) == [1, 9]
    assert list(md.Ntotal) == [100, 100]

    expected = np.log10(
        np.array([[0.5, 0.5], [0.1, 0.9]]) + EXPRESSION_PSEUDOCOUNT
    )
    np.testing.assert_allclose(md.log10expression, expected)


def test_prepare_model_data_prefers_the_per_well_cell_count_and_agrees_with_the_sum():
    """Both routes to Ntotal give the same design, because they must."""
    tidy = simulate_screen_frame(n_genes=8, n_wells=6, n_hits=2, seed=3)
    from_column = prepare_model_data(tidy)
    from_sum = prepare_model_data(tidy.drop(columns=["imaging_n_cells_per_well"]))
    np.testing.assert_array_equal(from_column.Ntotal, from_sum.Ntotal)
    np.testing.assert_allclose(
        from_column.log10expression, from_sum.log10expression
    )


def test_prepare_model_data_rejects_an_inconsistent_per_well_count():
    """A per-well total that varies inside its own well is a bad join, not a value to pick."""
    tidy = pd.DataFrame(
        {
            "well": ["w1", "w1", "w2", "w2"],
            "gene": ["A", "B", "A", "B"],
            "positive": [1, 0, 0, 9],
            "imaging_n_cells_per_well": [100, 250, 100, 100],
            "n_reads_per_gene_per_well": [500, 500, 100, 900],
        }
    )
    with pytest.raises(PowerFitError, match="varies between rows of the same"):
        prepare_model_data(tidy)


def test_prepare_model_data_drops_wells_with_no_imaged_cells():
    """R: filter(Ntotal > 0). log(0) is not an offset."""
    tidy = pd.DataFrame(
        {
            "well": ["w1", "w1", "w2", "w2", "w3", "w3"],
            "gene": ["A", "B", "A", "B", "A", "B"],
            "positive": [1, 0, 0, 0, 2, 3],
            "imaging_n_cells_per_gene_per_well": [50, 50, 0, 0, 40, 60],
            "n_reads_per_gene_per_well": [500, 500, 0, 0, 400, 600],
        }
    )
    md = prepare_model_data(tidy)
    assert list(md.wells) == ["w1", "w3"]
    assert md.dropped_wells == ("w2",)


def test_prepare_model_data_flags_a_gene_with_no_contrast_instead_of_calling_it_a_non_hit(caplog):
    """A constant covariate column is untested, not tested-and-negative."""
    tidy = simulate_screen_frame(n_genes=10, n_wells=20, n_hits=2, seed=5)
    tidy.loc[tidy["gene"] == "g000", "n_reads_per_gene_per_well"] = 0
    with caplog.at_level(logging.WARNING, logger="spacr.power_model"):
        md = prepare_model_data(tidy)
    assert "g000" in md.unidentified_genes
    assert "confounded with the intercept" in caplog.text

    fit = fit_model(md, backend="torch", seed=0, n_steps=100)
    estimate = gather_model_estimate(fit)
    row = estimate.loc[estimate["gene"] == "g000"].iloc[0]
    assert not row["identified"]
    # NaN, not 0.0. A shrunk-to-zero coefficient here would rank the gene
    # exactly where a confidently-tested non-hit ranks.
    assert np.isnan(row["mean"])
    assert np.isnan(row["q5"]) and np.isnan(row["q95"])


def test_prepare_model_data_zero_read_well_is_zero_fraction_not_nan():
    """0 reads out of 0 total is a fraction of 0 for every gene, not NaN."""
    tidy = pd.DataFrame(
        {
            "well": ["w1", "w1", "w2", "w2"],
            "gene": ["A", "B", "A", "B"],
            "positive": [1, 0, 3, 4],
            "imaging_n_cells_per_gene_per_well": [50, 50, 40, 60],
            "n_reads_per_gene_per_well": [500, 500, 0, 0],
        }
    )
    md = prepare_model_data(tidy)
    assert md.zero_read_wells == ("w2",)
    assert np.isfinite(md.log10expression).all()
    np.testing.assert_allclose(
        md.log10expression[1], np.log10(EXPRESSION_PSEUDOCOUNT)
    )


@pytest.mark.parametrize(
    "drop,message",
    [
        ("positive", "missing required column"),
        ("n_reads_per_gene_per_well", "missing required column"),
    ],
)
def test_prepare_model_data_names_the_missing_column(drop, message):
    """A missing column is named, not guessed at."""
    tidy = simulate_screen_frame(n_genes=4, n_wells=4, n_hits=1, seed=0)
    with pytest.raises(PowerFitError, match=message):
        prepare_model_data(tidy.drop(columns=[drop]))


def test_prepare_model_data_without_any_cell_count_refuses_to_guess():
    """No Ntotal means no Poisson offset, and no offset means wrong rates."""
    tidy = simulate_screen_frame(n_genes=4, n_wells=4, n_hits=1, seed=0)
    tidy = tidy.drop(
        columns=["imaging_n_cells_per_well", "imaging_n_cells_per_gene_per_well"]
    )
    with pytest.raises(PowerFitError, match="cannot be guessed"):
        prepare_model_data(tidy)


def test_prepare_model_data_rejects_duplicate_gene_well_pairs():
    """Two rows for one (well, gene) would be silently collapsed."""
    tidy = simulate_screen_frame(n_genes=3, n_wells=3, n_hits=1, seed=0)
    doubled = pd.concat([tidy, tidy.head(1)], ignore_index=True)
    with pytest.raises(PowerFitError, match="share a \\(well, gene\\) pair"):
        prepare_model_data(doubled)


def test_prepare_model_data_incomplete_grid_raises_then_fills_on_request():
    """A missing pair is ambiguous; the caller has to say which it means."""
    tidy = simulate_screen_frame(n_genes=4, n_wells=4, n_hits=1, seed=0)
    holed = tidy.drop(index=tidy.index[0]).reset_index(drop=True)
    with pytest.raises(PowerFitError, match="grid is incomplete"):
        prepare_model_data(holed)
    md = prepare_model_data(holed, fill_missing=True)
    assert md.n_wells == 4 and md.n_genes == 4


def test_prepare_model_data_rejects_more_positives_than_cells():
    """Positives cannot exceed the cells they were counted in."""
    tidy = pd.DataFrame(
        {
            "well": ["w1", "w1"],
            "gene": ["A", "B"],
            "positive": [80, 80],
            "imaging_n_cells_per_gene_per_well": [50, 50],
            "n_reads_per_gene_per_well": [500, 500],
        }
    )
    with pytest.raises(PowerFitError, match="More positives than cells"):
        prepare_model_data(tidy)


def test_prepare_model_data_rejects_negative_counts():
    """Counts are counts."""
    tidy = simulate_screen_frame(n_genes=3, n_wells=3, n_hits=1, seed=0)
    tidy.loc[0, "positive"] = -1
    with pytest.raises(PowerFitError, match="negative values"):
        prepare_model_data(tidy)


def test_model_data_to_frame_round_trips_the_design():
    """to_frame is for eyeballing; it must show the same numbers."""
    tidy = simulate_screen_frame(n_genes=5, n_wells=6, n_hits=1, seed=2)
    md = prepare_model_data(tidy)
    frame = md.to_frame()
    assert list(frame.columns) == ["well", "Npositive", "Ntotal"] + [
        str(g) for g in md.genes
    ]
    np.testing.assert_allclose(
        frame[[str(g) for g in md.genes]].to_numpy(), md.log10expression
    )


# ---------------------------------------------------------------------------
# 2. Backend selection -- never a silent substitution
# ---------------------------------------------------------------------------

def test_torch_backend_is_always_available():
    """spaCR depends on torch, so the default path always exists."""
    assert available_backends()["torch"] is True
    assert set(available_backends()) == set(BACKENDS)


def test_resolve_backend_auto_falls_back_to_torch_and_says_so(monkeypatch, caplog):
    """With no exact sampler installed, auto picks torch and logs the fact."""
    monkeypatch.setattr(
        "spacr.power_model._module_installed", lambda name: name == "torch"
    )
    with caplog.at_level(logging.INFO, logger="spacr.power_model"):
        assert resolve_backend("auto") == "torch"
    assert "resolved to 'torch'" in caplog.text


@pytest.mark.parametrize("installed", ["numpyro", "pymc"])
def test_resolve_backend_auto_prefers_an_exact_sampler_when_present(
    monkeypatch, caplog, installed
):
    """auto upgrades to NUTS when it can, and records which one."""
    monkeypatch.setattr(
        "spacr.power_model._module_installed",
        lambda name: name in {"torch", installed},
    )
    with caplog.at_level(logging.INFO, logger="spacr.power_model"):
        assert resolve_backend("auto") == installed
    assert installed in caplog.text


@pytest.mark.parametrize("name", ["numpyro", "pymc"])
def test_named_backend_is_never_silently_downgraded(monkeypatch, name):
    """Asking for pymc and getting ADVI would misdescribe the method used."""
    monkeypatch.setattr(
        "spacr.power_model._module_installed", lambda n: n == "torch"
    )
    with pytest.raises(PowerFitError, match="is not installed"):
        resolve_backend(name)


def test_unknown_backend_is_rejected():
    """A typo must not fall through to a default."""
    with pytest.raises(PowerFitError, match="unknown backend"):
        resolve_backend("cmdstanr")
    with pytest.raises(PowerFitError, match="must be a string"):
        resolve_backend(None)  # type: ignore[arg-type]


def test_fit_records_the_backend_that_actually_ran(signal_fit):
    """A result carries its own provenance."""
    fit = signal_fit[0]
    assert fit.backend == "torch"
    assert fit.requested_backend == "torch"
    assert fit.method == "advi"


# ---------------------------------------------------------------------------
# 3. The fit recovers signal -- the claim this module lives or dies on
# ---------------------------------------------------------------------------

def test_signal_recovery_auroc_is_high(signal_fit):
    """Five planted hits in a 60-gene library must be found.

    A fit that recovers nothing scores ~0.5 here; an inverted sign
    convention scores ~0.0.  Only a working fit clears 0.9.
    """
    _fit, _estimate, evaluation = signal_fit
    auroc = float(evaluation["model_auroc"].iloc[0])
    ap = float(evaluation["model_ap"].iloc[0])
    baseline = float(evaluation["ap_baseline"].iloc[0])
    assert auroc >= 0.9, f"AUROC {auroc:.3f} -- the fit is not finding the hits"
    # Average precision has to beat the prevalence, which is what a random
    # ranking gets. With 5 hits in 60 genes that floor is 0.083.
    assert ap > 4 * baseline, f"AP {ap:.3f} vs chance {baseline:.3f}"
    assert int(evaluation["n_hits"].iloc[0]) == 5
    assert evaluation["reason"].iloc[0] == ""


def test_signal_recovery_is_not_the_inverted_convention(signal_fit):
    """The hits are at the TOP of the ranking, one by one.

    :func:`test_signal_recovery_auroc_is_high` would already fail on an
    inverted convention, but only through the metric.  This asserts the
    raw coefficients directly: every hit's posterior mean must exceed
    every non-hit's.  A flipped sign anywhere in the chain reverses this.
    """
    _fit, estimate, _evaluation = signal_fit
    truth = simulate_screen_frame(seed=1)[["gene", "hit"]].drop_duplicates()
    merged = estimate.merge(truth, on="gene")
    hit_means = merged.loc[merged["hit"] == 1, "mean"]
    non_hit_means = merged.loc[merged["hit"] == 0, "mean"]
    assert hit_means.min() > non_hit_means.max(), (
        "a hit ranked below a non-hit: hits "
        f"[{hit_means.min():.4f}, {hit_means.max():.4f}] vs non-hits "
        f"[{non_hit_means.min():.4f}, {non_hit_means.max():.4f}]"
    )
    # And the direction is positive: more of a hit gene => more positive cells.
    assert hit_means.min() > 0


def test_null_screen_auroc_is_near_chance():
    """With hits labelled but inert, AUROC must sit at chance.

    ``class_pos == class_neg`` means hit status changes nothing about the
    imaging readout, so there is nothing to find.  A method that reports
    0.95 here is reading its own noise.  Four independent screens are
    averaged because a single AUROC with 15 positives has a null standard
    deviation near 0.09 -- one draw at 0.68 is noise, four averaging to
    0.68 is not.
    """
    aurocs = []
    for seed in range(4):
        screen = simulate_screen_frame(
            n_genes=80, n_hits=15, class_pos=0.05, class_neg=0.05,
            seed=100 + seed,
        )
        _fit, _estimate, evaluation = fit_and_evaluate(
            screen, backend="torch", seed=seed, n_steps=FIT_STEPS
        )
        aurocs.append(float(evaluation["model_auroc"].iloc[0]))

    mean_auroc = float(np.mean(aurocs))
    assert 0.35 <= mean_auroc <= 0.65, (
        f"null screens averaged AUROC {mean_auroc:.3f} over {aurocs} -- a "
        "screen with no effect must not look like a screen with one"
    )
    assert max(aurocs) < 0.9, (
        f"one null screen scored {max(aurocs):.3f}, which is the failure this "
        "test exists to catch"
    )


def test_null_screen_coefficients_collapse_to_numerical_zero():
    """No effect means no coefficient -- and this is the horseshoe-specific check.

    AUROC on pure noise is a coin flip and needs averaging to test.  The
    *magnitude* of the coefficients does not: with no signal the
    horseshoe drives every one of them to ~4e-4, three orders of
    magnitude below the ~0.18 the same model assigns a real hit.

    This is the assertion that distinguishes the regularized horseshoe
    from an ordinary shrinkage prior.  Replacing ``beta = z * tau *
    lambda_tilde`` with ``beta = z`` -- i.e. swapping the horseshoe for a
    plain ``Normal(0, 1)`` ridge -- was measured to leave the largest
    null coefficient at 0.083, two hundred times over this bar, while
    still passing :func:`test_horseshoe_shrinks_non_hits_toward_zero`.
    The spike at zero is what only the horseshoe gives you, and it only
    shows up when there is genuinely nothing to find.
    """
    screen = simulate_screen_frame(class_pos=0.05, class_neg=0.05, seed=1)
    _fit, estimate, _evaluation = fit_and_evaluate(
        screen, backend="torch", seed=0, n_steps=FIT_STEPS
    )
    largest_null = float(np.nanmax(np.abs(estimate["mean"].to_numpy())))
    assert largest_null < 0.01, (
        f"the largest coefficient on a null screen is {largest_null:.4g}; the "
        "prior is not shrinking a screen that contains nothing"
    )


def test_horseshoe_shrinks_non_hits_toward_zero(signal_fit):
    """Non-hit coefficients collapse relative to hits -- the prior's whole job.

    If this ratio is near 1 the model has degenerated into an
    unregularised Poisson regression, which with 60 genes and 120 wells
    would be badly overfit while still producing a number for every gene.

    Scope of the claim, stated because a test that oversells itself is
    worse than no test: this proves *shrinkage*, not specifically the
    *horseshoe*.  A plain ``Normal(0, 1)`` ridge on the same screen was
    measured at 4.5x, which clears this bar.  What separates the two is
    :func:`test_null_screen_coefficients_collapse_to_numerical_zero`, and
    that is where the horseshoe-specific claim is asserted.
    """
    _fit, estimate, _evaluation = signal_fit
    truth = simulate_screen_frame(seed=1)[["gene", "hit"]].drop_duplicates()
    merged = estimate.merge(truth, on="gene")
    hit_magnitude = merged.loc[merged["hit"] == 1, "mean"].abs().median()
    non_hit_magnitude = merged.loc[merged["hit"] == 0, "mean"].abs().median()
    assert hit_magnitude > 4 * non_hit_magnitude, (
        f"median |beta| is {hit_magnitude:.4f} for hits and "
        f"{non_hit_magnitude:.4f} for non-hits -- a ratio of "
        f"{hit_magnitude / max(non_hit_magnitude, 1e-12):.1f}x is not shrinkage"
    )
    # And it is the whole bulk that moves, not a lucky median: essentially
    # every non-hit has to sit below half a typical hit. An unregularised fit
    # scatters them across the same range as the hits.
    within = (
        merged.loc[merged["hit"] == 0, "mean"].abs() < 0.5 * hit_magnitude
    ).mean()
    assert within >= 0.9, f"only {within:.0%} of non-hits are shrunk to the spike"


def test_fit_is_deterministic_under_a_fixed_seed(signal_screen):
    """Same seed, same estimates -- to the last bit."""
    md = prepare_model_data(signal_screen)
    first = gather_model_estimate(
        fit_model(md, backend="torch", seed=11, n_steps=300)
    )
    second = gather_model_estimate(
        fit_model(md, backend="torch", seed=11, n_steps=300)
    )
    np.testing.assert_array_equal(
        first["mean"].to_numpy(), second["mean"].to_numpy()
    )
    np.testing.assert_array_equal(first["q5"].to_numpy(), second["q5"].to_numpy())


def test_a_different_seed_gives_a_different_fit(signal_screen):
    """The seed is actually wired in, not decoration.

    Without this, :func:`test_fit_is_deterministic_under_a_fixed_seed`
    would also pass on an implementation that ignores the seed entirely.
    """
    md = prepare_model_data(signal_screen)
    first = gather_model_estimate(fit_model(md, backend="torch", seed=11, n_steps=300))
    other = gather_model_estimate(fit_model(md, backend="torch", seed=12, n_steps=300))
    assert not np.array_equal(first["mean"].to_numpy(), other["mean"].to_numpy())


def test_fit_reports_the_units_beta_is_in(signal_screen):
    """A coefficient with an unstated scale is a number without a meaning."""
    md = prepare_model_data(signal_screen)
    plain = fit_model(md, backend="torch", seed=0, n_steps=100)
    scaled = fit_model(md, backend="torch", seed=0, n_steps=100, standardize=True)
    assert plain.diagnostics["beta_scale"] == "per unit log10expression"
    assert scaled.diagnostics["beta_scale"] == (
        "per standard deviation of log10expression"
    )


def test_standardized_fit_still_recovers_the_hits(signal_screen):
    """Rescaling the covariates must not break the recovery it exists to help.

    ``standardize=True`` divides each column by its SD so the horseshoe
    shrinks every gene on a common scale.  A back-transform applied in
    the wrong direction -- multiplying where it should divide -- reorders
    the genes by their covariate spread instead of by their effect, which
    still produces a full ranking and a plausible AUROC.
    """
    _fit, _estimate, evaluation = fit_and_evaluate(
        signal_screen, backend="torch", seed=0, n_steps=FIT_STEPS, standardize=True
    )
    assert float(evaluation["model_auroc"].iloc[0]) >= 0.9


def test_evaluate_accepts_a_boolean_hit_column():
    """Simulators write `hit` as 0/1 or as bool; both are ground truth."""
    truth = pd.DataFrame({"gene": list("ABCD"), "hit": [True, False, False, False]})
    estimate = pd.DataFrame({"gene": list("ABCD"), "mean": [2.0, 0.1, 0.0, -1.0]})
    result = evaluate_model_fit(truth, estimate)
    assert float(result["model_auroc"].iloc[0]) == pytest.approx(1.0)
    assert int(result["n_hits"].iloc[0]) == 1


def test_fit_survives_a_well_where_every_cell_is_positive():
    """The optimiser must not NaN out on the extreme end of the count range.

    A well with Npositive == Ntotal drives exp(eta) to the top of its
    range early in the optimisation, which is precisely where an ADVI
    implementation without gradient clipping takes one enormous step and
    poisons every parameter with NaN for the rest of the run.
    """
    screen = simulate_screen_frame(n_genes=20, n_wells=30, n_hits=3, seed=4)
    saturated = screen["well"] == 0
    screen.loc[saturated, "positive"] = screen.loc[
        saturated, "imaging_n_cells_per_gene_per_well"
    ]
    _fit, estimate, evaluation = fit_and_evaluate(
        screen, backend="torch", seed=0, n_steps=400
    )
    assert np.isfinite(estimate["mean"].to_numpy()).all()
    assert np.isfinite(float(evaluation["model_auroc"].iloc[0]))


def test_fit_refuses_a_design_it_cannot_identify():
    """One well cannot identify any gene effect; the posterior would be the prior."""
    md = ModelData(
        wells=np.array(["w1"]),
        genes=np.array(["A", "B"]),
        Npositive=np.array([3]),
        Ntotal=np.array([100]),
        log10expression=np.array([[-1.0, -2.0]]),
    )
    with pytest.raises(PowerFitError, match="single well cannot identify"):
        fit_model(md, backend="torch")


def test_fit_rejects_something_that_is_not_model_data():
    """The design has to come from prepare_model_data."""
    with pytest.raises(PowerFitError, match="needs the ModelData"):
        fit_model(pd.DataFrame({"a": [1]}), backend="torch")  # type: ignore[arg-type]


def test_non_convergence_is_reported_not_hidden(signal_screen, caplog):
    """Fifty ADVI steps is nowhere near converged, and it says so."""
    md = prepare_model_data(signal_screen)
    with caplog.at_level(logging.WARNING, logger="spacr.power_model"):
        fit = fit_model(md, backend="torch", seed=0, n_steps=50)
    assert fit.converged is False
    assert "did not converge" in caplog.text


def test_expected_hits_must_admit_some_sparsity():
    """p0 == D means "nothing is shrunk", which the horseshoe cannot express."""
    md = ModelData(
        wells=np.array(["w1", "w2"]),
        genes=np.array(["A", "B"]),
        Npositive=np.array([3, 4]),
        Ntotal=np.array([100, 100]),
        log10expression=np.array([[-1.0, -2.0], [-2.0, -1.0]]),
    )
    with pytest.raises(PowerFitError, match="expected_hits must be in"):
        fit_model(md, backend="torch", expected_hits=2)
    with pytest.raises(PowerFitError, match="scale_global must be"):
        fit_model(md, backend="torch", scale_global=0.0)


# ---------------------------------------------------------------------------
# 4. gather_model_estimate
# ---------------------------------------------------------------------------

def test_gather_model_estimate_summarises_the_draws():
    """mean/sd/q5/q95 are computed from the draws, and the R variable name is kept."""
    draws = np.stack([np.linspace(0.0, 1.0, 101), np.full(101, -2.0)], axis=1)
    fit = PowerFit(
        backend="torch",
        requested_backend="torch",
        method="advi",
        draws=draws,
        intercept_draws=np.full(101, -4.0),
        genes=np.array(["A", "B"]),
        converged=True,
    )
    estimate = gather_model_estimate(fit)
    assert list(estimate["gene"]) == ["A", "B"]
    assert list(estimate["variable"]) == [
        "b_log10expressionA",
        "b_log10expressionB",
    ]
    assert estimate["mean"].iloc[0] == pytest.approx(0.5)
    assert estimate["q5"].iloc[0] == pytest.approx(0.05, abs=1e-9)
    assert estimate["q95"].iloc[0] == pytest.approx(0.95, abs=1e-9)
    assert estimate["mean"].iloc[1] == pytest.approx(-2.0)
    assert estimate["prob_positive"].iloc[1] == pytest.approx(0.0)


def test_gather_model_estimate_rejects_a_mislabelled_draw_matrix():
    """More coefficients than genes means the labelling is off by an unknown amount."""
    fit = PowerFit(
        backend="torch",
        requested_backend="torch",
        method="advi",
        draws=np.zeros((10, 3)),
        intercept_draws=np.zeros(10),
        genes=np.array(["A", "B"]),
        converged=True,
    )
    with pytest.raises(PowerFitError, match="off by an unknown amount"):
        gather_model_estimate(fit)


def test_gather_model_estimate_rejects_a_non_fit():
    """It summarises a PowerFit, not whatever happens to have a .draws."""
    with pytest.raises(PowerFitError, match="needs a PowerFit"):
        gather_model_estimate({"draws": np.zeros((2, 2))})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. evaluate_model_fit -- the sign convention, pinned from both ends
# ---------------------------------------------------------------------------

def test_evaluate_orientation_is_not_inverted():
    """Higher `mean` must mean "more likely a hit".

    The R scores ``-mean`` for yardstick's first factor level, which is
    ``"no"``.  That is algebraically the same as scoring ``+mean`` for
    ``"yes"``, which is what this port does.  Both halves are asserted:
    the right orientation gives 1.0 and the inverted one gives 0.0, so a
    test suite that passes at both 0.95 and 0.05 is impossible here.
    """
    truth = pd.DataFrame({"gene": list("ABCDEF"), "hit": [1, 1, 0, 0, 0, 0]})
    correct = pd.DataFrame(
        {"gene": list("ABCDEF"), "mean": [2.0, 1.5, 0.1, 0.0, -0.2, -1.0]}
    )
    inverted = correct.assign(mean=lambda d: -d["mean"])

    right = evaluate_model_fit(truth, correct)
    wrong = evaluate_model_fit(truth, inverted)

    assert float(right["model_auroc"].iloc[0]) == pytest.approx(1.0)
    assert float(right["model_ap"].iloc[0]) == pytest.approx(1.0)
    # An inverted convention does not merely score badly -- it scores
    # 1 - AUROC, which for a mediocre real design would look plausible.
    assert float(wrong["model_auroc"].iloc[0]) == pytest.approx(0.0)
    assert float(right["model_auroc"].iloc[0]) + float(
        wrong["model_auroc"].iloc[0]
    ) == pytest.approx(1.0)


def test_evaluate_reports_the_chance_baseline_next_to_average_precision():
    """AP of 0.2 is excellent at 2 % prevalence and useless at 50 %."""
    truth = pd.DataFrame({"gene": list("ABCD"), "hit": [1, 0, 0, 0]})
    estimate = pd.DataFrame({"gene": list("ABCD"), "mean": [1.0, 0.0, 0.0, 0.0]})
    result = evaluate_model_fit(truth, estimate)
    assert float(result["ap_baseline"].iloc[0]) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "hits,fragment",
    [
        ([0, 0, 0, 0], "no gene is a hit"),
        ([1, 1, 1, 1], "every gene is a hit"),
    ],
)
def test_evaluate_degenerate_truth_is_nan_with_a_reason_never_half(hits, fragment):
    """AUROC is undefined with one class. 0.5 there would be a fabrication."""
    truth = pd.DataFrame({"gene": list("ABCD"), "hit": hits})
    estimate = pd.DataFrame({"gene": list("ABCD"), "mean": [1.0, 0.5, 0.0, -1.0]})
    result = evaluate_model_fit(truth, estimate)
    assert np.isnan(float(result["model_auroc"].iloc[0]))
    assert np.isnan(float(result["model_ap"].iloc[0]))
    assert fragment in result["reason"].iloc[0]


def test_evaluate_drops_unidentified_genes_and_counts_them():
    """A NaN coefficient is untested; it must not be scored as a low rank."""
    truth = pd.DataFrame({"gene": list("ABCD"), "hit": [1, 0, 0, 1]})
    estimate = pd.DataFrame(
        {"gene": list("ABCD"), "mean": [2.0, 0.0, np.nan, 1.0]}
    )
    result = evaluate_model_fit(truth, estimate)
    assert int(result["n_unidentified_dropped"].iloc[0]) == 1
    assert int(result["n_genes_scored"].iloc[0]) == 3
    assert float(result["model_auroc"].iloc[0]) == pytest.approx(1.0)


def test_evaluate_rejects_a_gene_with_two_truths():
    """Ground truth is one label per gene or it is not ground truth."""
    truth = pd.DataFrame({"gene": ["A", "A", "B"], "hit": [1, 0, 0]})
    estimate = pd.DataFrame({"gene": ["A", "B"], "mean": [1.0, 0.0]})
    with pytest.raises(PowerFitError, match="more than one hit status"):
        evaluate_model_fit(truth, estimate)


def test_evaluate_rejects_non_binary_truth():
    """"hit" is 0/1; anything else is a different variable."""
    truth = pd.DataFrame({"gene": ["A", "B"], "hit": ["yes", "no"]})
    estimate = pd.DataFrame({"gene": ["A", "B"], "mean": [1.0, 0.0]})
    with pytest.raises(PowerFitError, match="neither 0/1 nor"):
        evaluate_model_fit(truth, estimate)


def test_evaluate_names_a_missing_column():
    """No silent KeyError from three frames deep."""
    with pytest.raises(PowerFitError, match="missing column"):
        evaluate_model_fit(
            pd.DataFrame({"gene": ["A"]}), pd.DataFrame({"gene": ["A"], "mean": [1.0]})
        )


def test_evaluate_counts_genes_with_no_ground_truth(caplog):
    """An estimate with no truth row is dropped and counted, not scored as 0."""
    truth = pd.DataFrame({"gene": ["A", "B", "C"], "hit": [1, 0, 0]})
    estimate = pd.DataFrame(
        {"gene": ["A", "B", "C", "Z"], "mean": [2.0, 0.0, -1.0, 5.0]}
    )
    with caplog.at_level(logging.WARNING, logger="spacr.power_model"):
        result = evaluate_model_fit(truth, estimate)
    assert int(result["n_missing_truth"].iloc[0]) == 1
    assert int(result["n_genes_scored"].iloc[0]) == 3
    assert "no ground-truth hit status" in caplog.text


# ---------------------------------------------------------------------------
# 6. scan_parameters
# ---------------------------------------------------------------------------

def _scan_simulator(n_genes, n_wells, class_pos, seed=0):
    """Simulator with a scan-shaped signature, for the sweep tests."""
    return simulate_screen_frame(
        n_genes=n_genes,
        n_wells=n_wells,
        n_hits=3,
        class_pos=class_pos,
        class_neg=0.05,
        seed=seed,
    )


def test_scan_parameters_expands_the_grid_and_scores_every_point():
    """Two swept values times two replicates is four rows, all scored."""
    scan = scan_parameters(
        n_genes=20,
        n_wells=[30, 40],
        class_pos=0.5,
        n_replicates=2,
        simulate_fn=_scan_simulator,
        backend="torch",
        seed=3,
        fit_kwargs={"n_steps": 200},
    )
    assert len(scan) == 4
    assert sorted(scan["n_wells"].unique()) == [30, 40]
    assert sorted(scan["param_index"].unique()) == [1, 2]
    assert sorted(scan["replicate"].unique()) == [0, 1]
    assert (scan["status"] == "ok").all()
    assert scan["model_auroc"].notna().all()
    assert (scan["backend"] == "torch").all()
    # Held-fixed parameters travel with the row so the TSV is self-describing.
    assert (scan["n_genes"] == 20).all()
    # The seed reached the simulator through its declared `seed` argument, not
    # through a global.
    assert (scan["seed_channel"] == "seed").all()


def test_scan_parameters_replicates_differ_and_points_are_seeded_by_identity():
    """Replicates are independent screens, and a point's seed is its own."""
    scan = scan_parameters(
        n_genes=15,
        n_wells=[25, 35],
        class_pos=0.5,
        n_replicates=2,
        simulate_fn=_scan_simulator,
        backend="torch",
        seed=5,
        fit_kwargs={"n_steps": 150},
    )
    assert scan["seed_used"].nunique() == 4
    # Re-running only the second grid point must reproduce the seed it had
    # inside the full sweep -- which is only true if the seed is derived from
    # (seed, param_index, replicate) and not from iteration order.
    again = scan_parameters(
        n_genes=15,
        n_wells=[25, 35],
        class_pos=0.5,
        n_replicates=2,
        simulate_fn=_scan_simulator,
        backend="torch",
        seed=5,
        fit_kwargs={"n_steps": 150},
    )
    np.testing.assert_array_equal(
        scan["seed_used"].to_numpy(), again["seed_used"].to_numpy()
    )
    np.testing.assert_allclose(
        scan["model_auroc"].to_numpy(), again["model_auroc"].to_numpy()
    )


def test_scan_parameters_records_a_failed_point_as_failed_not_as_chance():
    """A point that blows up gets NaN and an error string, never 0.5.

    This is the single most consequential behaviour in the sweep: a
    backfilled 0.5 plots identically to "this design cannot find its
    hits", and the two conclusions have opposite consequences for
    whether you run the experiment.
    """
    def flaky(n_genes, n_wells, class_pos, seed=0):
        if n_wells == 30:
            raise RuntimeError("sequencing plate came back empty")
        return _scan_simulator(n_genes, n_wells, class_pos, seed=seed)

    with pytest.warns(RuntimeWarning, match="did not converge|failed"):
        scan = scan_parameters(
            n_genes=15,
            n_wells=[30, 40],
            class_pos=0.5,
            simulate_fn=flaky,
            backend="torch",
            seed=1,
            fit_kwargs={"n_steps": 150},
        )

    failed = scan.loc[scan["n_wells"] == 30].iloc[0]
    assert failed["status"] == "failed"
    assert np.isnan(failed["model_auroc"])
    assert np.isnan(failed["model_ap"])
    assert "sequencing plate came back empty" in failed["error"]

    survived = scan.loc[scan["n_wells"] == 40].iloc[0]
    assert survived["status"] == "ok"
    assert np.isfinite(survived["model_auroc"])


def test_scan_parameters_records_a_non_converged_fit_without_scoring_it():
    """A fit that did not converge has no posterior ordering to score."""
    with pytest.warns(RuntimeWarning):
        scan = scan_parameters(
            n_genes=15,
            n_wells=30,
            class_pos=0.5,
            simulate_fn=_scan_simulator,
            backend="torch",
            seed=1,
            fit_kwargs={"n_steps": 20},  # nowhere near converged
        )
    row = scan.iloc[0]
    assert row["status"] == "not_converged"
    assert row["converged"] is False or row["converged"] == False  # noqa: E712
    assert np.isnan(row["model_auroc"])
    assert "convergence" in row["error"]


def test_scan_parameters_on_error_raise_surfaces_the_exception():
    """For debugging one point, the sweep gets out of the way."""
    def broken(**kwargs):
        raise ValueError("deliberate")

    with pytest.raises(ValueError, match="deliberate"):
        scan_parameters(
            n_genes=10,
            n_wells=20,
            class_pos=0.5,
            simulate_fn=broken,
            backend="torch",
            on_error="raise",
        )


def test_scan_parameters_resumes_from_its_progress_file(tmp_path):
    """A killed sweep picks up where it stopped and does not redo finished work."""
    progress = tmp_path / "nested" / "scan.tsv"
    calls = []

    def counting(n_genes, n_wells, class_pos, seed=0):
        calls.append((n_wells, seed))
        return _scan_simulator(n_genes, n_wells, class_pos, seed=seed)

    kwargs = dict(
        n_genes=15,
        class_pos=0.5,
        simulate_fn=counting,
        backend="torch",
        seed=9,
        fit_kwargs={"n_steps": 150},
    )
    first = scan_parameters(n_wells=[25, 35], progress_file=str(progress), **kwargs)
    assert progress.exists()  # parent directory created for us
    assert len(calls) == 2
    assert len(first) == 2

    # Re-run the same grid: everything is already done, nothing is simulated.
    calls.clear()
    second = scan_parameters(n_wells=[25, 35], progress_file=str(progress), **kwargs)
    assert calls == []
    assert len(second) == 2
    np.testing.assert_allclose(
        first["model_auroc"].to_numpy(), second["model_auroc"].to_numpy()
    )
    # Resumed rows keep their string columns as strings, not as float NaN.
    assert second["error"].map(lambda v: isinstance(v, str)).all()

    # Extend the grid: only the new point is simulated, because the run key is
    # a digest of the parameter values and not of the row number.
    calls.clear()
    third = scan_parameters(
        n_wells=[25, 35, 45], progress_file=str(progress), **kwargs
    )
    assert [c[0] for c in calls] == [45]
    assert len(third) == 3
    assert sorted(third["n_wells"]) == [25, 35, 45]

    on_disk = pd.read_csv(progress, sep="\t")
    assert len(on_disk) == 3
    assert on_disk["run_key"].nunique() == 3


def test_scan_parameters_refuses_to_append_to_a_file_it_did_not_write(tmp_path):
    """Appending rows under someone else's header still parses, which is the danger."""
    progress = tmp_path / "scan.tsv"
    progress.write_text("alpha\tbeta\n1\t2\n", encoding="utf-8")
    with pytest.raises(PowerFitError, match="has columns"):
        scan_parameters(
            n_genes=10,
            n_wells=20,
            class_pos=0.5,
            simulate_fn=_scan_simulator,
            backend="torch",
            progress_file=str(progress),
        )


def test_scan_parameters_refuses_to_double_up_a_progress_file(tmp_path):
    """resume=False onto an existing file would duplicate every run key."""
    progress = tmp_path / "scan.tsv"
    kwargs = dict(
        n_genes=10,
        n_wells=20,
        class_pos=0.5,
        simulate_fn=_scan_simulator,
        backend="torch",
        fit_kwargs={"n_steps": 120},
    )
    scan_parameters(progress_file=str(progress), **kwargs)
    with pytest.raises(PowerFitError, match="resume=False"):
        scan_parameters(progress_file=str(progress), resume=False, **kwargs)


def test_scan_parameters_treats_a_string_parameter_as_one_value():
    """Sweeping over the characters of a word is the classic version of this bug."""
    seen = []

    def sim(n_genes, n_wells, class_pos, label, seed=0):
        seen.append(label)
        return _scan_simulator(n_genes, n_wells, class_pos, seed=seed)

    scan = scan_parameters(
        n_genes=10,
        n_wells=20,
        class_pos=0.5,
        label="pilot",
        simulate_fn=sim,
        backend="torch",
        fit_kwargs={"n_steps": 120},
    )
    assert len(scan) == 1
    assert seen == ["pilot"]


def test_scan_parameters_rejects_an_empty_or_impossible_grid():
    """A sweep with nothing in it is a mistake, not an empty result."""
    with pytest.raises(PowerFitError, match="no simulator parameters"):
        scan_parameters(simulate_fn=_scan_simulator, backend="torch")
    with pytest.raises(PowerFitError, match="empty sequence"):
        scan_parameters(
            n_wells=[], simulate_fn=_scan_simulator, backend="torch"
        )
    with pytest.raises(PowerFitError, match="on_error must be"):
        scan_parameters(
            n_wells=10, simulate_fn=_scan_simulator, backend="torch",
            on_error="ignore",
        )
    with pytest.raises(PowerFitError, match="n_replicates must be"):
        scan_parameters(
            n_wells=10, simulate_fn=_scan_simulator, backend="torch",
            n_replicates=0,
        )


def test_scan_parameters_says_which_simulator_it_wanted_when_there_is_none(monkeypatch):
    """A missing simulator half names itself rather than raising from an import."""
    import spacr.power_model as power_model

    def no_simulator():
        raise PowerFitError(
            "scan_parameters needs a screen simulator. "
            "spacr.power_simulate.simulate_screen could not be imported"
        )

    monkeypatch.setattr(power_model, "_default_simulator", no_simulator)
    with pytest.raises(PowerFitError, match="power_simulate"):
        scan_parameters(n_wells=10, backend="torch")


def test_scan_parameters_reports_the_seed_channel_for_a_simulator_without_a_seed():
    """A simulator seeded through a global says so, so the run stays auditable."""
    def unseeded(n_genes, n_wells, class_pos):
        return _scan_simulator(n_genes, n_wells, class_pos, seed=0)

    scan = scan_parameters(
        n_genes=10,
        n_wells=20,
        class_pos=0.5,
        simulate_fn=unseeded,
        backend="torch",
        fit_kwargs={"n_steps": 120},
    )
    assert scan["seed_channel"].iloc[0] == "numpy-global"


# ---------------------------------------------------------------------------
# 7. Integration with the simulator half, when it is present
# ---------------------------------------------------------------------------

def test_integration_with_power_simulate():
    """The two halves agree on the column contract.

    Skipped until :mod:`spacr.power_simulate` lands.  When it does, this
    fails loudly if its output does not carry the columns
    :func:`prepare_model_data` documents -- which is exactly the seam
    where a port like this silently breaks.
    """
    power_simulate = pytest.importorskip(
        "spacr.power_simulate", reason="the simulator half is not installed yet"
    )
    simulate = getattr(power_simulate, "simulate_screen", None)
    if simulate is None:
        pytest.skip("spacr.power_simulate has no simulate_screen")

    screen = simulate(
        n_genes_in_library=40,
        gene_abundance_alpha=200.0,
        gene_hit_rate=0.15,
        n_wells_per_screen=60,
        well_abundance_factor_mu=8.0,
        well_abundance_factor_var=2.0,
        imaging_n_cells_per_well_mu=800.0,
        imaging_n_cells_per_well_var=1600.0,
        class_pos_mu=0.9,
        class_pos_var=0.001,
        class_neg_mu=0.05,
        class_neg_var=0.001,
        sequencing_n_cells_per_well_lambda=500.0,
        pcr_factor_mu=0.1,
        pcr_factor_var=0.01,
        n_reads_per_well=20_000,
        seed=0,
    )
    # The seam: every column prepare_model_data documents as required has to
    # be here, spelled exactly this way.
    for column in (
        "gene",
        "well",
        "hit",
        "positive",
        "n_reads_per_gene_per_well",
        "imaging_n_cells_per_well",
    ):
        assert column in screen.columns, (
            f"spacr.power_simulate.simulate_screen no longer emits {column!r}; "
            "the two halves of the port have drifted apart"
        )

    model_data = prepare_model_data(screen)
    assert model_data.n_wells > 1
    assert model_data.n_genes == 40

    fit = fit_model(model_data, backend="torch", seed=0, n_steps=FIT_STEPS)
    estimate = gather_model_estimate(fit)
    assert len(estimate) == 40
    evaluation = evaluate_model_fit(screen, estimate)
    assert len(evaluation) == 1
    # A real screen simulated with a near-perfect classifier and 60 wells has
    # to be solvable. If this drops to chance the two halves disagree about
    # what the columns mean, which is the failure this test exists for.
    auroc = float(evaluation["model_auroc"].iloc[0])
    assert np.isfinite(auroc), evaluation["reason"].iloc[0]
    assert auroc >= 0.8, (
        f"AUROC {auroc:.3f} on the real simulator with a 0.9/0.05 classifier "
        "and 60 wells -- the inference half is not reading the simulator half "
        "correctly"
    )


@pytest.mark.parametrize("local_scale", [2.0, 1e200])
def test_numpyro_horseshoe_keeps_supported_large_scales_finite(monkeypatch, local_scale):
    jax = pytest.importorskip("jax")
    numpyro = pytest.importorskip("numpyro")
    import jax.numpy as jnp
    import torch
    from numpyro import handlers, infer
    from torch.nn import functional as F

    class ConditionedSampler:
        def __init__(self, model, **kwargs):
            self.model = model

        def run(self, key):
            log_scales = np.log([local_scale, 0.25, 4.0])

            def trace_at(scales):
                latents = {
                    "z": jnp.array([0.5, -0.5]),
                    "lam": jnp.array([jnp.exp(scales[0]), 2.0]),
                    "tau": jnp.exp(scales[1]),
                    "c2": jnp.exp(scales[2]),
                    "intercept": jnp.array(-2.0),
                }
                conditioned = handlers.condition(self.model, data=latents)
                return handlers.trace(handlers.seed(conditioned, rng_seed=key)).get_trace()

            self.trace = trace_at(jnp.asarray(log_scales))
            beta, gradient = jax.value_and_grad(
                lambda scales: trace_at(scales)["beta"]["value"][0]
            )(jnp.asarray(log_scales))
            theta = torch.tensor(log_scales, dtype=torch.float64, requires_grad=True)
            log_lam, log_tau, log_c2 = theta.unbind()
            expected = 0.5 * torch.exp(log_tau + log_lam - 0.5 * F.softplus(
                2.0 * log_tau + 2.0 * log_lam - log_c2))
            expected.backward()
            assert np.isfinite(np.asarray(beta)).all()
            assert np.isfinite(np.asarray(gradient)).all()
            np.testing.assert_allclose(beta, expected.detach().numpy(), rtol=1e-12)
            np.testing.assert_allclose(gradient, theta.grad.numpy(), rtol=1e-12, atol=1e-14)
            rate = np.asarray(self.trace["obs"]["fn"].rate)
            assert np.isfinite(rate).all() and (rate > 0).all()
            if local_scale == 2.0:
                direct = 0.5 * 0.25 * np.sqrt(4.0 * 2.0 ** 2 / (4.0 + 0.25 ** 2 * 2.0 ** 2))
                np.testing.assert_allclose(beta, direct, rtol=1e-12)
            else:
                np.testing.assert_allclose(beta, 0.5 * np.sqrt(4.0), rtol=1e-12)

        def get_samples(self):
            return {name: jnp.repeat(self.trace[name]["value"][None], 2, axis=0)
                    for name in ("beta", "intercept")}

        def get_extra_fields(self, **kwargs):
            return {"diverging": np.zeros(2, dtype=bool)}

    monkeypatch.setattr(infer, "NUTS", lambda model, **kwargs: model)
    monkeypatch.setattr(infer, "MCMC", ConditionedSampler)
    data = ModelData(
        wells=np.array(["w1", "w2", "w3"]), genes=np.array(["A", "B"]),
        Npositive=np.array([1, 3, 2]), Ntotal=np.array([10, 10, 10]),
        log10expression=np.array([[-1.0, -0.1], [-0.1, -1.0], [-0.5, -0.5]]))
    fit = fit_model(data, backend="numpyro", n_chains=1)
    assert np.isfinite(fit.draws).all()


@pytest.mark.parametrize("sampler_raises", [False, True])
def test_numpyro_preserves_half_student_priors_and_scoped_double_precision(
        monkeypatch, sampler_raises):
    jax = pytest.importorskip("jax")
    numpyro = pytest.importorskip("numpyro")
    import jax.numpy as jnp
    from numpyro import handlers
    from numpyro import infer
    try:
        from jax import enable_x64
    except ImportError:
        from jax.experimental import enable_x64

    class CheckedSampler:
        def __init__(self, model, **kwargs):
            self.model = model

        def run(self, key):
            assert jax.config.x64_enabled
            self.trace = handlers.trace(handlers.seed(self.model, rng_seed=key)).get_trace()
            assert self.trace["obs"]["fn"].rate.dtype == jnp.float64
            for name in ("lam", "tau"):
                distribution = self.trace[name]["fn"]
                assert bool(jnp.all(distribution.support(jnp.array([1.0, 2.0]))))
                assert not bool(jnp.any(distribution.support(jnp.array([-1.0]))))
                values = jnp.array([1.0, 2.0])
                folded = (distribution.base_dist
                          if isinstance(distribution, numpyro.distributions.ExpandedDistribution)
                          else distribution)
                assert isinstance(folded, numpyro.distributions.FoldedDistribution)
                assert isinstance(folded.base_dist, numpyro.distributions.StudentT)
                assert float(folded.base_dist.df) == (10.0 if name == "lam" else 1.0)
                expected = folded.base_dist.log_prob(values) + jnp.log(2.0)
                actual = distribution.log_prob(values)
                np.testing.assert_allclose(actual, expected)
                assert not np.allclose(actual[0], actual[1])
            if sampler_raises:
                raise RuntimeError("sampler failed")

        def get_samples(self):
            return {name: jnp.repeat(self.trace[name]["value"][None], 4, axis=0)
                    for name in ("beta", "intercept")}

        def get_extra_fields(self, **kwargs):
            return {"diverging": np.zeros(4, dtype=bool)}

    def checked_kernel(model, **kwargs):
        assert kwargs["target_accept_prob"] == 0.87
        return model

    monkeypatch.setattr(infer, "NUTS", checked_kernel)
    monkeypatch.setattr(infer, "MCMC", CheckedSampler)
    data = ModelData(
        wells=np.array(["w1", "w2", "w3"]), genes=np.array(["A", "B"]),
        Npositive=np.array([1, 3, 2]), Ntotal=np.array([10, 10, 10]),
        log10expression=np.array([[-1.0, -0.1], [-0.1, -1.0], [-0.5, -0.5]]))
    with enable_x64(False):
        if sampler_raises:
            with pytest.raises(RuntimeError, match="sampler failed"):
                fit_model(data, backend="numpyro", n_chains=1, target_accept=0.87)
        else:
            fit = fit_model(data, backend="numpyro", n_chains=1, target_accept=0.87)
            assert fit.draws.dtype == np.float64
            assert fit.converged
        assert not jax.config.x64_enabled


@pytest.mark.parametrize('weighted', [False, True])
def test_numpyro_repeated_well_likelihood_and_gradient_match_single_observations(monkeypatch, weighted):
    jax = pytest.importorskip('jax')
    numpyro = pytest.importorskip('numpyro')
    import jax.numpy as jnp
    import torch
    from numpyro import handlers, infer

    repeated = np.array([0, 0, 0, 1, 1])
    weights = np.array([1/3, 1/3, 1/3, 1/2, 1/2]) if weighted else None
    design = np.array([[0.2, 0.8], [0.7, 0.3]])
    counts, exposure = np.array([2, 4]), np.array([10, 12])

    class ConditionedSampler:
        def __init__(self, model, **kwargs):
            self.model = model

        def run(self, key):
            def trace_at(theta):
                latents = {'z': theta[:2], 'lam': jnp.ones(2),
                           'tau': jnp.array(0.25), 'c2': jnp.array(4.0),
                           'intercept': theta[2]}
                return handlers.trace(handlers.seed(
                    handlers.condition(self.model, data=latents), rng_seed=key)).get_trace()

            def loglik(theta):
                site = trace_at(theta)['obs']
                return jnp.sum(site['fn'].log_prob(site['value']) * site['scale'])

            theta = np.array([0.4, -0.2, -1.2])
            value, gradient = jax.value_and_grad(loglik)(jnp.asarray(theta))
            expected_theta = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            beta = expected_theta[:2] * (0.25 * np.sqrt(4 / (4 + 0.25**2)))
            indices = np.arange(2) if weighted else repeated
            eta = expected_theta[2] + torch.tensor(design[indices] - design[repeated].mean(axis=0)) @ beta + torch.log(
                torch.tensor(exposure[indices], dtype=torch.float64))
            expected = torch.distributions.Poisson(torch.exp(eta)).log_prob(
                torch.tensor(counts[indices], dtype=torch.float64)).sum()
            expected.backward()
            np.testing.assert_allclose(value, expected.detach().numpy(), rtol=1e-12)
            np.testing.assert_allclose(gradient, expected_theta.grad.numpy(), rtol=1e-12, atol=1e-14)
            self.trace = trace_at(jnp.asarray(theta))
            np.testing.assert_array_equal(self.trace['obs']['value'], counts[repeated])

        def get_samples(self):
            return {name: jnp.repeat(self.trace[name]['value'][None], 2, axis=0)
                    for name in ('beta', 'intercept')}

        def get_extra_fields(self, **kwargs):
            return {'diverging': np.zeros(2, dtype=bool)}

    monkeypatch.setattr(infer, 'NUTS', lambda model, **kwargs: model)
    monkeypatch.setattr(infer, 'MCMC', ConditionedSampler)
    data = ModelData(wells=np.array(['a']*3 + ['b']*2), genes=np.array(['A', 'B']),
                     Npositive=counts[repeated], Ntotal=exposure[repeated],
                     log10expression=design[repeated], likelihood_weights=weights,
                     likelihood_well_ids=tuple(repeated) if weighted else None)
    fit_model(data, backend='numpyro', standardize=False, n_chains=1)


@pytest.mark.parametrize('weights', [[1.0], [1.0, 0.0], [1.0, np.nan], [[1.0], [1.0]]])
def test_likelihood_weights_refuse_invalid_or_misaligned_rows(weights):
    data = ModelData(wells=np.array(['a', 'b']), genes=np.array(['A']),
                     Npositive=np.array([1, 2]), Ntotal=np.array([10, 10]),
                     log10expression=np.array([[0.2], [0.8]]), likelihood_weights=weights)
    with pytest.raises(ValueError, match='likelihood_weights.*aligned'):
        fit_model(data, backend='torch')


def test_pymc_repeated_well_loglik_and_gradient_match_one_contribution(monkeypatch):
    pm = pytest.importorskip('pymc')
    import torch

    class InspectedModel(Exception):
        pass

    def inspect_model(**kwargs):
        model = pm.modelcontext(None)
        point = model.initial_point()
        likelihood = model.compile_logp(vars=model.observed_RVs + model.potentials)
        z = np.array([0.4, -0.2])
        point['z'] = z
        point['intercept'] = np.array(-1.2)
        point['lam_log__'] = np.zeros(2)
        point['tau_log__'] = np.array(np.log(0.25))
        point['c2_log__'] = np.array(np.log(4.0))
        theta = torch.tensor([0.4, -0.2, -1.2], dtype=torch.float64, requires_grad=True)
        beta = theta[:2] * (0.25 * np.sqrt(4 / (4 + 0.25**2)))
        design = np.array([[0.2, 0.8], [0.7, 0.3]])
        centre = design[[0, 0, 0, 1, 1]].mean(axis=0)
        eta = theta[2] + torch.tensor(design - centre) @ beta + torch.log(torch.tensor([10., 12.], dtype=torch.float64))
        expected = torch.distributions.Poisson(torch.exp(eta)).log_prob(torch.tensor([2, 4], dtype=torch.float64)).sum()
        expected.backward()
        np.testing.assert_allclose(likelihood(point), expected.detach().numpy(), rtol=1e-7)
        gradients = []
        for name, index in [('z', 0), ('z', 1), ('intercept', None)]:
            values = []
            for delta in [-1e-5, 1e-5]:
                changed = {key: np.array(value, copy=True) for key, value in point.items()}
                if index is None:
                    changed[name] += delta
                else:
                    changed[name][index] += delta
                values.append(likelihood(changed))
            gradients.append((values[1] - values[0]) / 2e-5)
        np.testing.assert_allclose(gradients, theta.grad.numpy(), rtol=1e-6, atol=1e-7)
        raise InspectedModel

    monkeypatch.setattr(pm, 'sample', inspect_model)
    data = ModelData(wells=np.array(['a']*3 + ['b']*2), genes=np.array(['A', 'B']),
                     Npositive=np.array([2, 2, 2, 4, 4]), Ntotal=np.array([10, 10, 10, 12, 12]),
                     log10expression=np.array([[0.2, 0.8]]*3 + [[0.7, 0.3]]*2),
                     likelihood_weights=np.array([1/3]*3 + [1/2]*2),
                     likelihood_well_ids=(0, 0, 0, 1, 1))
    with pytest.raises(InspectedModel):
        fit_model(data, backend='pymc', standardize=False, n_chains=1)


def test_torch_weighted_repeated_well_loglik_and_gradient_equal_one_factor():
    import torch
    from spacr.power_model import _torch_poisson_log_likelihood

    eta = torch.tensor([[0.3, -0.2], [0.1, 0.7]], dtype=torch.float64, requires_grad=True)
    repeated = [0, 0, 0, 1, 1]
    counts = torch.tensor([2., 4.], dtype=torch.float64)
    weights = torch.tensor([1/3]*3 + [1/2]*2, dtype=torch.float64)
    actual = _torch_poisson_log_likelihood(eta[:, repeated], counts[repeated], weights)
    expected = (counts * eta - torch.exp(eta)).sum(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), eta, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), eta)[0])


def _calibration_design(repeat=False, padded=False):
    from dataclasses import replace

    data = ModelData(wells=np.array(['w1', 'w2']), genes=np.array(['a', 'b']),
                     Npositive=np.array([2, 4]), Ntotal=np.array([10, 12]),
                     log10expression=np.array([[.2, .8], [.7, .3]]))
    if repeat:
        positions = [0, 0, 0, 1, 1]
        data = replace(data, wells=data.wells[positions], Npositive=data.Npositive[positions],
                       Ntotal=data.Ntotal[positions], log10expression=data.log10expression[positions],
                       likelihood_weights=np.array([1/3]*3 + [1/2]*2),
                       likelihood_well_ids=(('s1', 'plate1', 'A01'),)*3 + (('s2', 'plate1', 'A01'),)*2)
    if padded:
        data = replace(data, genes=np.array(['a', 'b', 'ignored_intercept']),
                       log10expression=np.column_stack([data.log10expression, np.ones(data.n_wells)]))
    return data


@pytest.mark.parametrize('repeat,padded', [(False, False), (False, True), (True, False), (True, True)])
def test_prior_calibration_is_invariant_to_inverse_duplicates_and_constant_padding(repeat, padded):
    from spacr.power_model import _prior_calibration_counts, _horseshoe_global_scale, _prepare_design

    data = _calibration_design(repeat, padded)
    predictors, observations = _prior_calibration_counts(data)
    assert (predictors, observations) == (2, 2)
    assert data.n_wells == (5 if repeat else 2)
    assert data.n_genes == (3 if padded else 2)
    assert _horseshoe_global_scale(predictors, observations, None, None) == pytest.approx(1/np.sqrt(2))
    transformed, _, _, identified = _prepare_design(data, False)
    assert identified.tolist() == ([True, True, False] if padded else [True, True])
    if padded:
        np.testing.assert_array_equal(transformed[:, -1], 0)


@pytest.mark.parametrize('change', ['ids_length', 'missing_id', 'wrong_weights', 'different_design', 'different_count', 'different_exposure', 'no_ids'])
def test_independent_well_calibration_refuses_ambiguous_or_nonduplicate_rows(change):
    from dataclasses import replace
    from spacr.power_model import _prior_calibration_counts

    data = _calibration_design(repeat=True)
    if change == 'ids_length':
        data = replace(data, likelihood_well_ids=data.likelihood_well_ids[:-1])
    elif change == 'missing_id':
        data = replace(data, likelihood_well_ids=(None,) + data.likelihood_well_ids[1:])
    elif change == 'wrong_weights':
        data = replace(data, likelihood_weights=np.ones(5))
    elif change == 'no_ids':
        data = replace(data, likelihood_well_ids=None)
    else:
        field = {'different_design':'log10expression', 'different_count':'Npositive', 'different_exposure':'Ntotal'}[change]
        array = getattr(data, field).copy()
        array[0] += 1
        data = replace(data, **{field:array})
    with pytest.raises(PowerFitError, match='likelihood_well_ids|inverse row-multiplicity'):
        _prior_calibration_counts(data)


@pytest.mark.parametrize("with_ids", [False, True])
def test_arbitrary_weights_do_not_implicitly_set_independent_sample_size(with_ids):
    from dataclasses import replace
    from spacr.power_model import _prior_calibration_counts

    data = replace(_calibration_design(), likelihood_weights=np.array([.2, .8]))
    if with_ids:
        data = replace(data, likelihood_well_ids=(("s1", "plate1", "A01"), ("s2", "plate1", "A01")))
    with pytest.raises(PowerFitError, match="explicit scale_global|inverse row-multiplicity"):
        _prior_calibration_counts(data)
    assert _prior_calibration_counts(data, scale_global=.03) == (2, None)


@pytest.mark.parametrize('genes,expected_hits', [(0, None), (1, None), (2, 2.0), (2, 0.0)])
def test_horseshoe_calibration_rejects_undefined_active_predictor_sparsity(genes, expected_hits, monkeypatch):
    from dataclasses import replace

    data = _calibration_design()
    if genes == 0:
        data = replace(data, log10expression=np.ones((2, 2)))
    elif genes == 1:
        data = replace(data, log10expression=np.column_stack([data.log10expression[:, 0], np.ones(2)]))
    probes = []

    def unavailable(name):
        probes.append(name)
        return False

    monkeypatch.setattr('spacr.power_model._module_installed', unavailable)
    with pytest.raises(PowerFitError, match='identified|expected_hits|sparsity'):
        fit_model(data, backend='numpyro', expected_hits=expected_hits)
    assert probes == []


def test_valid_horseshoe_calibration_still_refuses_the_requested_missing_backend(monkeypatch):
    monkeypatch.setattr('spacr.power_model._module_installed', lambda name: False)
    with pytest.raises(PowerFitError, match="backend='numpyro'.*numpyro is not installed"):
        fit_model(_calibration_design(), backend='numpyro', expected_hits=1.0)


def test_uniform_inverse_replication_preserves_centered_design_likelihood_and_prior():
    from dataclasses import replace
    from spacr.power_model import _prior_calibration_counts, _prepare_design, _observation_weights

    original = _calibration_design()
    positions = np.repeat(np.arange(2), 3)
    repeated = replace(original, wells=original.wells[positions],
                       log10expression=original.log10expression[positions],
                       Npositive=original.Npositive[positions], Ntotal=original.Ntotal[positions],
                       likelihood_weights=np.full(6, 1/3), likelihood_well_ids=tuple(positions))
    x, means, scales, _ = _prepare_design(original, True)
    repeated_x, repeated_means, repeated_scales, _ = _prepare_design(repeated, True)
    np.testing.assert_allclose(repeated_x, x[positions])
    np.testing.assert_allclose(repeated_means, means)
    np.testing.assert_allclose(repeated_scales, scales)
    beta = np.array([.4, -.2])
    eta = -1.2 + x @ beta + np.log(original.Ntotal)
    repeated_eta = -1.2 + repeated_x @ beta + np.log(repeated.Ntotal)
    expected = np.sum(original.Npositive*eta - np.exp(eta))
    actual = np.sum((repeated.Npositive*repeated_eta - np.exp(repeated_eta))*_observation_weights(repeated))
    assert actual == pytest.approx(expected)
    assert _prior_calibration_counts(original) == _prior_calibration_counts(repeated) == (2, 2)


@pytest.mark.parametrize('expected_hits,scale_global,expected', [(.5, None, 1/np.sqrt(2)), (None, .03, .03)])
def test_one_active_predictor_requires_an_explicit_valid_sparsity_or_scale(expected_hits, scale_global, expected):
    from spacr.power_model import _horseshoe_global_scale

    assert _horseshoe_global_scale(1, 2, expected_hits, scale_global) == pytest.approx(expected)


@pytest.mark.parametrize("identity", [["s1", "plate1", "A01"],
                                      {"screen": "s1", "well": "A01"}])
def test_unhashable_physical_well_identity_refuses_before_backend_selection(identity, monkeypatch):
    from dataclasses import replace

    data = _calibration_design(repeat=True)
    data = replace(data, likelihood_well_ids=(identity,) + data.likelihood_well_ids[1:])
    probes = []

    def unavailable(name):
        probes.append(name)
        return False

    monkeypatch.setattr("spacr.power_model._module_installed", unavailable)
    with pytest.raises(PowerFitError, match="hashable physical-well identities") as error:
        fit_model(data, backend="numpyro")
    assert isinstance(error.value.__cause__, TypeError)
    assert probes == []


@pytest.mark.parametrize("theta", [
    [.4, -.6, -.8, .5, -4., -.5, -3.],
    [-.7, .3, 1.2, -1.5, -1., 1.4, -1.2],
    [.05, 1.1, -2., 2., .4, 3., .2],
])
def test_numpyro_marginal_target_and_outputs_ignore_unidentified_coordinates(
        monkeypatch, theta):
    """Unused z/lam factors normalize to one, including their support Jacobian.

    Their analytical removal leaves the active marginal target unchanged.
    Capture actual model closures and evaluate fixed latents without running
    NUTS; padded outputs must still retain full gene order and NaN masking.
    Poisson residuals have no separate residual-scale parameter.
    """
    jax = pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    import jax.numpy as jnp
    from dataclasses import replace
    from numpyro import infer
    from numpyro.infer.util import get_transforms, log_density, potential_energy
    from scipy.integrate import quad
    from scipy.stats import norm, t
    from spacr.power_model import _prepare_design, _prior_calibration_counts

    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        from jax.experimental import enable_x64

    def parameters(value):
        return dict(z=value[:2], lam=jnp.exp(value[2:4]),
                    tau=jnp.exp(value[4]), c2=jnp.exp(value[5]), intercept=value[6])

    models, traces = [], []

    class ConditionedOnly:
        def __init__(self, model, **kwargs):
            self.model = model

        def run(self, key):
            models.append(self.model)
            _, self.trace = log_density(self.model, (), {}, parameters(jnp.asarray(theta)))
            traces.append(self.trace)

        def get_samples(self):
            return {name: jnp.repeat(self.trace[name]["value"][None], 2, axis=0)
                    for name in ("beta", "intercept")}

        def get_extra_fields(self, **kwargs):
            return {"diverging": np.zeros(2, dtype=bool)}

    monkeypatch.setattr(infer, "NUTS", lambda model, **kwargs: model)
    monkeypatch.setattr(infer, "MCMC", ConditionedOnly)
    active = _calibration_design(repeat=True)
    padded = replace(
        active,
        genes=np.array(["Intercept", "a", "ignored_middle", "b", "ignored_last"]),
        log10expression=np.column_stack([
            np.ones(active.n_wells), active.log10expression[:, 0],
            np.full(active.n_wells, 4.), active.log10expression[:, 1],
            np.full(active.n_wells, -2.)]),
    )
    assert _prior_calibration_counts(active) == _prior_calibration_counts(padded) == (2, 2)
    active_fit = fit_model(active, backend="numpyro", standardize=True)
    padded_fit = fit_model(padded, backend="numpyro", standardize=True)
    active_model, padded_model = models
    active_trace, padded_trace = traces
    assert padded_trace["z"]["value"].shape == padded_trace["lam"]["value"].shape == (2,)
    assert padded_trace["intercept"]["value"].shape == ()
    np.testing.assert_array_equal(padded_fit.genes, padded.genes)
    np.testing.assert_array_equal(padded_fit.draws[:, [1, 3]], active_fit.draws)
    assert np.isnan(padded_fit.draws[:, [0, 2, 4]]).all()
    assert padded_fit.draws.shape == (2, 5)
    assert padded_fit.draws.dtype == np.float64
    np.testing.assert_array_equal(padded_fit.intercept_draws, active_fit.intercept_draws)
    assert padded_fit.diagnostics["n_unidentified"] == 3
    assert padded_fit.diagnostics["tau0"] == active_fit.diagnostics["tau0"]

    with enable_x64():
        x, means, scales, identified = _prepare_design(padded, True)
        active_x, active_means, active_scales, _ = _prepare_design(active, True)
        np.testing.assert_array_equal(x[:, [1, 3]], active_x)
        np.testing.assert_array_equal(x[:, [0, 2, 4]], 0)
        np.testing.assert_array_equal(means[[1, 3]], active_means)
        np.testing.assert_array_equal(scales[[1, 3]], active_scales)
        np.testing.assert_array_equal(padded_fit.diagnostics["column_means"], means)
        np.testing.assert_array_equal(padded_fit.diagnostics["column_scales"], scales)
        assert padded_fit.diagnostics["beta_scale"] == active_fit.diagnostics["beta_scale"]
        assert identified.tolist() == [False, True, False, True, False]

        def density(model, value):
            return log_density(model, (), {}, parameters(value))[0]

        def energy(model, value):
            constrained = parameters(value)
            transforms = get_transforms(model, (), {}, constrained)
            values = {key: transforms[key].inv(item)
                      for key, item in constrained.items()}
            return potential_energy(model, (), {}, values)

        theta = jnp.asarray(theta)
        for target in (density, energy):
            padded_value, padded_gradient = jax.value_and_grad(
                lambda value: target(padded_model, value))(theta)
            active_value, active_gradient = jax.value_and_grad(
                lambda value: target(active_model, value))(theta)
            assert np.isfinite(np.asarray(padded_gradient)).all()
            np.testing.assert_allclose(padded_value, active_value, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(padded_gradient, active_gradient, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(padded_trace["beta"]["value"])[[1, 3]],
                                   active_trace["beta"]["value"], rtol=1e-12, atol=1e-12)
        full_obs, active_obs = padded_trace["obs"], active_trace["obs"]
        np.testing.assert_array_equal(full_obs["value"], active.Npositive)
        np.testing.assert_array_equal(full_obs["scale"], active.likelihood_weights)
        np.testing.assert_allclose(full_obs["fn"].rate, active_obs["fn"].rate,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(full_obs["value"] - full_obs["fn"].rate,
                                   active_obs["value"] - active_obs["fn"].rate,
                                   rtol=1e-12, atol=1e-12)

    assert quad(norm.pdf, -np.inf, np.inf)[0] == pytest.approx(1, abs=1e-10)
    assert quad(lambda value: 2*t.pdf(value, df=10), 0, np.inf)[0] == pytest.approx(1, abs=1e-10)
