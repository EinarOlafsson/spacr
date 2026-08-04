"""The design arithmetic behind the Power / Design screen.

No Qt and no fitting here — this is the half of the screen that decides
*which* library call to make and *how* to read the answer, and it is tested
without a QApplication so a failure points at the arithmetic rather than at
the widget. The screen's end of the contract (that it really does make that
call, and really does render what comes back) is in
``tests/qt/test_power_screen.py``.

The properties that matter most:

* the defaults are the real *T. gondii* screen's fitted values, because a
  power analysis that opens on invented numbers teaches the wrong scale;
* a replicate that failed or did not converge counts against the design, and
  the aggregation cannot be refactored into a ``dropna()`` without this
  suite going red;
* a withheld metric stays withheld — nothing here ever backfills 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.power_design import (
    CAVEATS,
    CELLS_MULTIPLIERS,
    POWER_CURVE_COLUMNS,
    DesignSpec,
    cells_grid,
    changes_the_number,
    estimate_runtime_s,
    plain_sentence,
    power_curve,
    simulator_kwargs,
    wells_grid,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scan(values, statuses, aurocs, aps=None, column="imaging_n_cells_per_well_mu"):
    """A synthetic `scan_parameters` frame, one row per (value, replicate)."""
    rows = []
    for value, status_block, auroc_block in zip(values, statuses, aurocs):
        ap_block = (aps[values.index(value)] if aps is not None
                    else [a for a in auroc_block])
        for status, auroc, average_precision in zip(
                status_block, auroc_block, ap_block):
            rows.append({
                column: float(value),
                "status": status,
                "model_auroc": auroc,
                "model_ap": average_precision,
                "ap_baseline": 0.1,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the defaults
# ---------------------------------------------------------------------------

def test_the_defaults_are_the_real_screen_not_round_numbers():
    """Every shipped default traces to the *T. gondii* screen it was fitted to.

    ``proposals/SIM_PORT_PLAN.md`` §1 records the screen: 452 genes at ~4
    guides, 4 x 384 wells, ~4.6 constructs per well, ~123 cells imaged per
    well with variance ~8000, a MaxViT classifier at 0.80 / 0.12, ~2.5 % of
    genes true hits, ~3e4 reads per well, library skew alpha 0.6. A default
    that is a round number instead is a default somebody invented, and the
    first number a new user sees would then be uncheckable.
    """
    spec = DesignSpec()
    assert spec.n_genes == 452
    assert spec.n_grnas_per_gene == 4
    assert (spec.wells_per_plate, spec.n_plates) == (384, 4)
    assert spec.n_wells == 1536
    assert spec.constructs_per_well == pytest.approx(4.6)
    assert spec.cells_per_well == pytest.approx(123.0)
    assert spec.cells_per_well_var == pytest.approx(8000.0)
    assert spec.background_positive_rate == pytest.approx(0.12)
    assert spec.hit_positive_rate == pytest.approx(0.80, abs=1e-3)
    assert spec.hit_rate == pytest.approx(0.025)
    assert spec.reads_per_well == pytest.approx(30000.0)
    assert spec.gene_abundance_alpha == pytest.approx(0.6)
    assert spec.read_depth_cv == pytest.approx(0.35)
    # ~11 hit genes, which is the figure the vignette infers.
    assert spec.expected_hits == pytest.approx(11.3, abs=0.1)


def test_the_translation_to_simulator_arguments_is_exactly_the_library_s_names():
    """`simulator_kwargs` speaks the simulator's own parameter names.

    Every key here is a parameter of
    :func:`spacr.power_simulate.simulate_screen`, and none of the simulator's
    required parameters is missing — otherwise the screen would look like it
    was configuring the simulation while quietly leaving half of it at
    whatever the library happened to default to.
    """
    import inspect

    from spacr.power_simulate import simulate_screen

    signature = inspect.signature(simulate_screen)
    accepted = set(signature.parameters)
    produced = simulator_kwargs(DesignSpec())

    assert set(produced) <= accepted, sorted(set(produced) - accepted)
    required = {
        name for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert required <= set(produced), sorted(required - set(produced))


def test_guides_reach_the_library_size_only_when_scoring_per_guide():
    """Scoring per gene, guides are pooled before the model sees them.

    This is what spaCRPower and the real analysis do, and pretending
    otherwise would let a user "improve" their power by typing a bigger
    guide count into a model that has no guide-efficiency term.
    """
    per_gene = DesignSpec(n_genes=100, n_grnas_per_gene=4, score_per="gene")
    per_guide = per_gene.with_values(score_per="guide")
    assert per_gene.n_library_units == 100
    assert per_guide.n_library_units == 400
    assert simulator_kwargs(per_gene)["n_genes_in_library"] == 100
    assert simulator_kwargs(per_guide)["n_genes_in_library"] == 400
    # And the guide count is invisible to the simulation in gene mode.
    assert (simulator_kwargs(per_gene)
            == simulator_kwargs(per_gene.with_values(n_grnas_per_gene=9)))


def test_the_effect_size_is_a_fold_change_on_the_per_cell_positive_rate():
    """`effect_fold` multiplies the background rate, and is capped below 1.

    The cap is not a clamp on the user's request: a beta with mean exactly 1
    exists only at zero variance, and
    :func:`spacr.power_simulate.rbeta_mean_variance` refuses to substitute a
    distribution the caller did not ask for. Capping just below keeps the
    requested spread meaningful.
    """
    spec = DesignSpec(background_positive_rate=0.2, effect_fold=3.0)
    assert spec.hit_positive_rate == pytest.approx(0.6)
    assert simulator_kwargs(spec)["class_pos_mu"] == pytest.approx(0.6)
    assert simulator_kwargs(spec)["class_neg_mu"] == pytest.approx(0.2)
    saturated = DesignSpec(background_positive_rate=0.5, effect_fold=40.0)
    assert saturated.hit_positive_rate < 1.0


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_a_workable_design_has_no_complaints():
    assert DesignSpec().validate() == []


@pytest.mark.parametrize("changes, fragment", [
    ({"n_plates": 0, "wells_per_plate": 1}, "single well"),
    ({"hit_rate": 1e-6}, "expects"),
    ({"effect_fold": 0.5}, "LESS likely"),
    ({"n_replicates": 0}, "at least 1 replicate"),
    ({"detection_auroc": 0.0}, "AUROC"),
    ({"score_per": "sgRNA"}, "score_per"),
    ({"class_neg_var": 0.5}, "Bernoulli bound"),
])
def test_an_impossible_design_is_refused_before_the_run_starts(changes, fragment):
    """Bad designs are named, not discovered as 27 identical failures.

    ``scan_parameters`` records a raising point as ``status="failed"`` and
    carries on, so a design the library cannot simulate at all would come
    back as a full grid of failures — which on the curve is indistinguishable
    from a design with no power.
    """
    problems = DesignSpec(**changes).validate()
    assert problems, f"{changes} was accepted"
    assert any(fragment in problem for problem in problems), problems


def test_an_effect_below_one_is_refused_because_the_model_scores_one_direction():
    """A protective knockout is not a small positive effect, it is a negative one.

    ``evaluate_model_fit`` scores ``beta`` with higher meaning more hit-like;
    an effect that lowers the positive rate produces negative coefficients
    for the true hits and reports near-chance. Silently accepting it would
    have the screen answer "you have no power" for a design that has plenty
    of power in the other direction.
    """
    problems = DesignSpec(effect_fold=0.5).validate()
    assert any("LESS likely" in p for p in problems), problems


# ---------------------------------------------------------------------------
# the sweep grids
# ---------------------------------------------------------------------------

def test_the_users_own_design_is_always_a_point_on_its_own_curve():
    """The grid brackets what was typed rather than being fixed.

    The whole question is "is MY design enough". A fixed grid would answer it
    for somebody else's design and leave the user interpolating — and
    :func:`plain_sentence` refuses to interpolate, on purpose.
    """
    for cells in (7.0, 123.0, 400.0, 1.0):
        spec = DesignSpec(cells_per_well=cells)
        assert float(round(cells)) in cells_grid(spec)
    for wells_per_plate, plates in ((96, 1), (384, 4), (1536, 2)):
        spec = DesignSpec(wells_per_plate=wells_per_plate, n_plates=plates)
        assert float(spec.n_wells) in wells_grid(spec)


def test_the_grids_are_sorted_unique_and_never_degenerate():
    """Duplicates would fit the same design twice and plot a vertical step."""
    for spec in (DesignSpec(), DesignSpec(cells_per_well=1.0),
                 DesignSpec(cells_per_well=2.0),
                 DesignSpec(wells_per_plate=96, n_plates=1)):
        for grid in (cells_grid(spec), wells_grid(spec)):
            assert grid == sorted(grid)
            assert len(grid) == len(set(grid))
            assert all(value >= 1 for value in grid)
    # A one-well-per-plate design still cannot produce a one-well fit, which
    # `fit_model` refuses outright.
    assert min(wells_grid(DesignSpec(wells_per_plate=96, n_plates=1))) >= 2


def test_the_runtime_estimate_grows_with_the_work():
    """Not a promise, but it has to move the right way.

    An estimate that ignores library size or replicate count would tell a
    user that a five-minute sweep and a fifty-minute sweep cost the same.
    """
    base = DesignSpec()
    assert estimate_runtime_s(base) > 0
    assert (estimate_runtime_s(base.with_values(n_replicates=9))
            > estimate_runtime_s(base))
    assert (estimate_runtime_s(base.with_values(n_genes=4000))
            > estimate_runtime_s(base))
    assert (estimate_runtime_s(base.with_values(n_plates=16))
            > estimate_runtime_s(base))


# ---------------------------------------------------------------------------
# the power curve
# ---------------------------------------------------------------------------

def test_power_is_the_fraction_of_replicates_over_the_bar():
    scan = _scan(
        values=[10.0, 100.0],
        statuses=[["ok", "ok", "ok"], ["ok", "ok", "ok"]],
        aurocs=[[0.55, 0.61, 0.90], [0.95, 0.99, 0.81]],
    )
    curve = power_curve(scan, "imaging_n_cells_per_well_mu", 0.80)
    assert list(curve.columns) == list(POWER_CURVE_COLUMNS)
    assert list(curve["value"]) == [10.0, 100.0]
    assert list(curve["n_detected"]) == [1, 3]
    assert curve["power"].tolist() == [pytest.approx(1 / 3), pytest.approx(1.0)]


def test_a_replicate_that_never_fit_counts_against_the_design():
    """The denominator is every replicate, not the ones that worked.

    This is the flattering-number trap: thin designs do not score badly, they
    stop fitting, so computing the power over the surviving replicates
    reports the power of the runs that were easy enough to fit. It is also
    the assertion that stops this from being "simplified" into a dropna.
    """
    scan = _scan(
        values=[10.0],
        statuses=[["ok", "not_converged", "failed", "ok"]],
        aurocs=[[0.95, np.nan, np.nan, 0.50]],
    )
    curve = power_curve(scan, "imaging_n_cells_per_well_mu", 0.80).iloc[0]
    assert curve["n_replicates"] == 4
    assert curve["n_ok"] == 2
    assert curve["n_not_converged"] == 1
    assert curve["n_failed"] == 1
    assert curve["n_detected"] == 1
    assert curve["power"] == pytest.approx(0.25)
    # ...and the honest-when-it-worked number is over the two that worked.
    assert curve["mean_auroc"] == pytest.approx(0.725)


def test_a_point_where_nothing_converged_is_nan_not_one_half():
    """A design that cannot be fit is not a design sitting at chance.

    Backfilling 0.5 here would draw the two as the same curve, and they lead
    to opposite decisions: one says buy more cells, the other says the
    analysis is broken.
    """
    scan = _scan(
        values=[5.0],
        statuses=[["not_converged", "not_converged"]],
        aurocs=[[np.nan, np.nan]],
    )
    curve = power_curve(scan, "imaging_n_cells_per_well_mu", 0.80).iloc[0]
    assert curve["power"] == 0.0
    assert np.isnan(curve["mean_auroc"])
    assert np.isnan(curve["mean_ap"])
    assert curve["n_not_converged"] == 2


def test_the_curve_preserves_a_monotone_ordering_it_is_given():
    """Aggregation cannot invert or flatten the ordering of its input.

    The simulation decides whether power rises with cells per well; this
    function must not be the thing that decides it. Given replicate scores
    that rise, the curve rises.
    """
    values = [8.0, 16.0, 32.0, 64.0, 128.0]
    aurocs = [[0.50, 0.52, 0.55], [0.70, 0.72, 0.60], [0.81, 0.79, 0.82],
              [0.90, 0.85, 0.88], [0.99, 0.97, 0.98]]
    scan = _scan(values, [["ok"] * 3] * 5, aurocs)
    power = power_curve(scan, "imaging_n_cells_per_well_mu", 0.80)["power"]
    assert list(power) == sorted(power)
    assert power.iloc[0] == 0.0 and power.iloc[-1] == 1.0


def test_an_empty_scan_gives_an_empty_curve_rather_than_an_exception():
    empty = power_curve(pd.DataFrame(), "imaging_n_cells_per_well_mu", 0.8)
    assert len(empty) == 0
    assert list(empty.columns) == list(POWER_CURVE_COLUMNS)
    with pytest.raises(KeyError):
        power_curve(_scan([1.0], [["ok"]], [[0.9]]), "no_such_column", 0.8)


def test_a_frame_without_a_status_column_is_refused_not_assumed_successful():
    """Missing ``status`` must not be read as "every replicate succeeded".

    That default would turn a mis-wired call into a design that looks better
    than it is, which is the single direction this module must never fail in.
    """
    scan = _scan([1.0], [["ok", "ok"]], [[0.9, 0.2]]).drop(columns=["status"])
    with pytest.raises(KeyError, match="status"):
        power_curve(scan, "imaging_n_cells_per_well_mu", 0.8)
    no_metric = _scan([1.0], [["ok"]], [[0.9]]).drop(columns=["model_auroc"])
    with pytest.raises(KeyError, match="model_auroc"):
        power_curve(no_metric, "imaging_n_cells_per_well_mu", 0.8)


# ---------------------------------------------------------------------------
# the sentence
# ---------------------------------------------------------------------------

def test_the_sentence_answers_for_the_design_that_was_typed():
    """It reads the user's own point off the curve, not the best point.

    Reporting the best grid point would answer a question nobody asked and
    would read as a promise about the design on the form.
    """
    spec = DesignSpec(cells_per_well=64.0, n_replicates=4, detection_auroc=0.8)
    curve = power_curve(
        _scan([32.0, 64.0, 128.0],
              [["ok"] * 4] * 3,
              [[0.99] * 4, [0.9, 0.95, 0.5, 0.4], [0.99] * 4]),
        "imaging_n_cells_per_well_mu", 0.8)
    sentence = plain_sentence(spec, curve)
    assert "64 cells per well" in sentence
    assert "50% of simulations" in sentence
    assert "2 of 4" in sentence


def test_the_sentence_says_how_many_replicates_produced_no_fit():
    spec = DesignSpec(cells_per_well=10.0, n_replicates=3)
    curve = power_curve(
        _scan([10.0], [["ok", "not_converged", "failed"]],
              [[0.99, np.nan, np.nan]]),
        "imaging_n_cells_per_well_mu", 0.8)
    sentence = plain_sentence(spec, curve)
    assert "2 of those 3 did not produce a usable fit" in sentence
    assert "non-detections" in sentence


def test_the_sentence_refuses_to_interpolate_a_point_that_was_not_simulated():
    """A power read between grid points is a number the simulation never made."""
    spec = DesignSpec(cells_per_well=77.0)
    curve = power_curve(_scan([32.0, 128.0], [["ok"]] * 2, [[0.99], [0.99]]),
                        "imaging_n_cells_per_well_mu", 0.8)
    assert "did not include 77" in plain_sentence(spec, curve)
    assert plain_sentence(spec, None).startswith("No run yet")


# ---------------------------------------------------------------------------
# the caveats
# ---------------------------------------------------------------------------

def test_the_caveats_that_change_the_number_are_a_named_subset():
    """The split is the panel's whole job.

    A list where the COM-Poisson third moment carries the same weight as
    "the R version overstates power" is a list that gets skimmed.
    """
    high = changes_the_number()
    assert 0 < len(high) < len(CAVEATS)
    assert all(caveat.changes_the_number for caveat in high)
    keys = {caveat.key for caveat in high}
    assert "even_split_overstates_power" in keys
    assert "failed_replicates_count_against" in keys
    assert "advi_not_nuts" in keys


def test_every_caveat_is_a_complete_record():
    keys = [caveat.key for caveat in CAVEATS]
    assert len(keys) == len(set(keys))
    for caveat in CAVEATS:
        assert caveat.headline.strip() and caveat.detail.strip()
        assert len(caveat.detail) > len(caveat.headline)


def test_the_even_split_caveat_says_which_way_the_error_goes():
    """"Differs from the R" is useless; "the R overstates power" is the point.

    A screener who reads a caveat that only says the two implementations
    differ has learned nothing actionable. The direction is what tells them
    whether a published R-derived power figure was optimistic.
    """
    caveat = next(c for c in CAVEATS if c.key == "even_split_overstates_power")
    assert "OVERSTATES" in caveat.headline
    assert "abundance" in caveat.headline.lower()
    assert "optimistic" in caveat.detail


def test_the_default_imaging_split_is_the_one_that_does_not_overstate_power():
    """spaCR ships the honest split; 'uniform' exists only for R parity."""
    assert DesignSpec().imaging_split == "abundance"
    assert simulator_kwargs(DesignSpec())["imaging_split"] == "abundance"


def test_the_multipliers_bracket_the_design_on_both_sides():
    """A curve that only goes up cannot show a design is already past the knee."""
    assert min(CELLS_MULTIPLIERS) < 1.0 < max(CELLS_MULTIPLIERS)
    assert 1.0 in CELLS_MULTIPLIERS
