"""The design arithmetic behind the Power / Design screen — no Qt in here.

:mod:`spacr.power_simulate` and :mod:`spacr.power_model` answer the question
*"given these sixteen parameters, how well does the model recover the hits?"*.
The question a screener actually asks is *"how many cells per well, and how
many wells, do I need to detect an effect of size X?"*. This module is the
translation between the two, and it is deliberately separate from the widget
so the translation can be tested without a QApplication — the same split
``pca_model`` / ``pca`` and ``pivot_spec`` / ``tabulate`` already use.

What lives here:

* :class:`DesignSpec` — the experiment as a screener describes it (genes,
  guides, wells, plates, effect size, prevalence, depth), plus the
  real-screen values for everything the screener does not want to name.
* :func:`simulator_kwargs` — that description as the keyword arguments
  :func:`spacr.power_model.scan_parameters` takes. **This is the only place
  the translation happens.** The screen never builds simulator arguments of
  its own and never computes a metric of its own; it calls the library and
  renders what comes back.
* :func:`cells_grid` / :func:`wells_grid` — the two sweep axes, bracketing
  whatever the user typed rather than a fixed grid, so the curve always has
  the user's own design on it.
* :func:`power_curve` — replicates collapsed into a detection probability.
* :func:`plain_sentence` — the one sentence the whole screen exists to print.
* :data:`CAVEATS` — the port's documented departures from spaCRPower, as
  data, so the screen can render the ones that change the number next to the
  number instead of leaving them in a docstring nobody opens.

Where the defaults come from
----------------------------
Every default in :class:`DesignSpec` is the fitted value for the real
*T. gondii* screen this simulator was built against, as recorded in
``proposals/SIM_PORT_PLAN.md`` §1 and §10 and in the port's own docstrings:
452 genes at ~4 guides each, spotted into 4 x 384 wells at ~4.6 constructs
per well, ~123 cells imaged per well (var ~8000), a MaxViT classifier at
0.80 / 0.12, ~2.5 % of genes true hits, ~3e4 reads per well. That makes the
screen open on a working example rather than on filler, and it means the
first number a user sees is one they can check against a screen that has
actually been run.

What "detection" means here, exactly
------------------------------------
A power analysis needs a yes/no per simulated screen before it can report a
percentage. The model does not emit a yes/no — it emits a *ranking* of genes,
scored by AUROC and average precision. So the screen asks the user for the bar
(``detection_auroc``, default 0.80) and reports the fraction of simulated
screens that clear it. Two things about that fraction are load-bearing:

1. **A replicate whose fit failed or did not converge counts as a
   non-detection**, not as a missing value. Dropping it would raise the
   reported power by removing exactly the runs where the design was too thin
   to fit — which is the failure mode the analysis is supposed to find.
2. **The mean AUROC is reported beside the power**, over the replicates that
   did converge, with the count of those that did not. A power of 0/5 with
   five non-converged fits and a power of 0/5 with five converged fits at
   AUROC 0.52 are different findings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CAVEATS",
    "CELLS_MULTIPLIERS",
    "Caveat",
    "DesignSpec",
    "PLATE_FORMATS",
    "WELLS_MULTIPLIERS",
    "cells_grid",
    "changes_the_number",
    "estimate_runtime_s",
    "plain_sentence",
    "power_curve",
    "simulator_kwargs",
    "wells_grid",
]


#: Wells per plate the screen offers. 384 is what the real screen used; 1536
#: is offered because spaCR's own converter handles that format.
PLATE_FORMATS: Tuple[int, ...] = (96, 384, 1536)

#: The cells-per-well sweep, as multiples of the user's own value. Centred on
#: 1.0 so the design being asked about is always a point on its own curve —
#: a fixed grid would answer a question about somebody else's design, and the
#: user would then have to interpolate the one number they came for.
CELLS_MULTIPLIERS: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0)

#: The wells sweep, as multiples of the user's own total. Coarser than the
#: cells axis because wells are bought in plates, not in ones.
WELLS_MULTIPLIERS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

#: Rough seconds per fit, for the "this will take about N minutes" estimate.
#: Measured on the real-screen design (1536 wells x 452 genes) with the torch
#: ADVI backend on CPU: ~5 s. Scaled by design size in
#: :func:`estimate_runtime_s`. It is an order-of-magnitude figure and is
#: labelled as one on screen — a wrong estimate that is honest about being an
#: estimate is worth much more than no estimate at all before a 5-minute run.
SECONDS_PER_FIT_REFERENCE: float = 5.0

#: The design the reference timing was measured on.
_REFERENCE_CELLS_TIMES_WELLS: float = 1536.0 * 452.0


# ---------------------------------------------------------------------------
# The caveats, as data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Caveat:
    """One documented departure or limitation, ready to render.

    :ivar key: stable identifier, for tests and for the export.
    :ivar headline: the one line that goes on screen next to the number.
    :ivar detail: the paragraph behind it.
    :ivar changes_the_number: whether believing the opposite would change the
        power the screen reports. Only these are rendered next to the
        headline sentence; the rest go in the list below it. The distinction
        is the whole point of the panel — a caveat list where everything is
        equally urgent is read as a disclaimer and skipped.
    """

    key: str
    headline: str
    detail: str
    changes_the_number: bool


#: Every caveat the screen knows about, most consequential first.
#:
#: Sourced from the port's own docstrings —
#: :mod:`spacr.power_simulate`'s "Deviations from the R package, and why" and
#: :mod:`spacr.power_model`'s "What ADVI costs you versus NUTS" — plus the two
#: reporting rules this module itself imposes. They are duplicated here rather
#: than scraped out of ``__doc__`` because a docstring is prose for a reader
#: who already opened the file, and this is a label for someone who never
#: will.
CAVEATS: Tuple[Caveat, ...] = (
    Caveat(
        key="even_split_overstates_power",
        headline=(
            "spaCRPower splits a well's cells evenly between the genes in "
            "it, which OVERSTATES power. This run splits by abundance."
        ),
        detail=(
            "The R original passes a 0/1 in-well indicator as the "
            "multinomial probability, so a gene that is 2 % of the well and "
            "a gene that is 40 % of it get the same number of imaged cells. "
            "Real libraries are skewed (this one at Dirichlet alpha 0.6), so "
            "an even split understates how few cells a rare genotype "
            "actually contributes, and a hit you can only see in a handful "
            "of cells is the hit you lose. Power computed the R way is "
            "therefore optimistic relative to the numbers on this screen. "
            "Switch imaging_split to 'uniform' in "
            "spacr.power_simulate.simulate_imaging_plate to reproduce the R."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="failed_replicates_count_against",
        headline=(
            "A replicate whose fit failed or did not converge is counted as "
            "a non-detection, never dropped and never scored 0.5."
        ),
        detail=(
            "Thin designs do not merely score badly, they stop fitting: at "
            "very few cells per well the ADVI optimisation does not meet its "
            "convergence criterion and the library withholds the metrics. "
            "Dropping those replicates would compute the power over exactly "
            "the runs that worked, which is the most flattering possible "
            "denominator. Backfilling them at 0.5 would draw a design that "
            "cannot be fit as a design sitting at chance. The count of "
            "non-converged and failed replicates is shown beside every point "
            "so the two readings stay distinguishable."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="advi_not_nuts",
        headline=(
            "The default backend is mean-field ADVI, not NUTS: the ranking "
            "is trustworthy, the intervals are not."
        ),
        detail=(
            "spaCRPower fits with brms + cmdstanr (full NUTS). There is no "
            "pip-installable equivalent, so the default here is variational "
            "inference in torch. Mean-field VI underestimates posterior "
            "variance, has no R-hat and can land in a local optimum. What it "
            "does get right is the ORDER of the per-gene coefficients — and "
            "AUROC and average precision depend on nothing else, which is "
            "why it is defensible for a power analysis and not defensible "
            "for quoting a per-gene credible interval. Install numpyro or "
            "pymc and pick that backend when the interval is the deliverable."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="r_simulate_fit_path_does_not_run",
        headline=(
            "These numbers cannot be reproduced from spaCRPower as "
            "published: its simulate-to-fit path does not run."
        ),
        detail=(
            "R/fit_model.R reads well_data$imaging_n_cells_per_well as the "
            "Poisson offset, but simulate_imaging_plate never emits that "
            "column — R's $ partial matching finds three columns sharing the "
            "prefix, returns NULL, and the offset silently vanishes. This "
            "port emits the column for real, as the realised per-well cell "
            "total. So the model being scored here is the model the R "
            "package describes, not the model the R package runs."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="unidentified_genes_are_untested",
        headline=(
            "Genes whose read fraction never varies between wells are "
            "untested, not non-hits, and are dropped from the metrics."
        ),
        detail=(
            "A gene that lands in every well, or in none, has a constant "
            "covariate column; its coefficient is perfectly confounded with "
            "the intercept and carries no information. The horseshoe would "
            "happily shrink it to zero, which reads exactly like 'tested, "
            "not a hit'. The library reports those genes as NaN and counts "
            "them separately. A design where most of the library is "
            "unidentified has not been shown to lack power — it has not been "
            "tested at all, and the fix is more wells, not more cells."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="sequencing_error_hides_untested_genes",
        headline=(
            "Sequencing error is OFF by default. Turning it on barely dilutes "
            "the effect — but it stops the untested-gene check from firing."
        ),
        detail=(
            "Neither spaCRPower nor this port modelled mis-assigned barcode "
            "reads. Simulating them says the direct dilution is small: on the "
            "452-gene reference design, 0.5 % mis-assignment moves the "
            "hit/non-hit separation from 0.799 to 0.794 among genes that were "
            "testable to begin with. The large effect is elsewhere. A gene "
            "that landed in every well or in none has a constant read "
            "fraction and is reported as UNTESTED rather than as a non-hit — "
            "and phantom reads give every such gene a covariate that varies, "
            "so it is scored, at chance, on noise. On that design 0.5 % error "
            "takes the scored library from 317 genes to all 452 and drops the "
            "screen-wide separation from 0.799 to 0.723: fourteen times the "
            "dilution, entirely from a safeguard switching off. Turn it on "
            "with set_spec(DesignSpec(sequencing_error_rate=0.005)), or pass "
            "sequencing_error_rate to spacr.power_simulate.simulate_screen."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="thin_wells_count_the_same_as_full_ones",
        headline=(
            "Well dropout is OFF by default: a well with three imaged cells "
            "enters the fit next to a well with four hundred."
        ),
        detail=(
            "Its positive fraction can only be 0, 1/3, 2/3 or 1, and its "
            "standard error is several times the whole gap between the "
            "classifier's hit-cell and background rates. The Poisson offset "
            "stops such a well dominating the scale of the fit; it does not "
            "stop its read-fraction covariate being paired with a response "
            "that is almost pure noise. Dropping thin wells is what an "
            "analyst does by hand, and it costs wells, so which way the trade "
            "comes out depends on how long the thin tail is — which is worth "
            "simulating rather than arguing about. Turn it on with "
            "set_spec(DesignSpec(min_cells_per_well=25)), or pass "
            "min_cells_per_well to spacr.power_simulate.simulate_screen. Note "
            "that the cells per well on this form is a MEAN: at 123 with a "
            "variance of 8000 a real fraction of wells lands under 25."
        ),
        changes_the_number=True,
    ),
    Caveat(
        key="reads_per_well_is_per_well",
        headline=(
            "'Reads per well' is per well. spaCRPower divided its read "
            "budget by the number of GENES, giving ~284 where a real screen "
            "has ~30 000."
        ),
        detail=(
            "Upstream computes round(n_reads_total / nrow(well_data)) inside "
            "a group_by(well), so nrow is the library size. With 452 genes "
            "and n_reads_total = 128318 that is 284 reads per well. The "
            "parameter is also documented as the screen total in one "
            "vignette and as the per-well geometric mean in another. This "
            "port takes an unambiguous per-well figure and derives the total."
        ),
        changes_the_number=False,
    ),
    Caveat(
        key="no_guide_level_model",
        headline=(
            "There is no guide-efficiency layer: scoring per gene, more "
            "guides per gene buys nothing in this model."
        ),
        detail=(
            "The simulator's library unit is whatever gets its own read "
            "count. Scored per gene — which is what the real analysis does, "
            "and what spaCRPower modelled — the guides are pooled before the "
            "model sees them, so guides-per-gene changes no number on this "
            "screen. Score per guide to simulate each construct separately, "
            "and note that this then assumes every guide of a hit gene is "
            "itself a hit: it prices the DILUTION of a bigger library, not "
            "the insurance of having several guides. A real guide-efficiency "
            "term is not in the port."
        ),
        changes_the_number=False,
    ),
    Caveat(
        key="com_poisson_replaced",
        headline=(
            "The COM-Poisson dispersion knob is replaced by a mean/variance "
            "pair; third and higher moments differ from the R."
        ),
        detail=(
            "COMPoissonReg::rcmp is called but not declared in the R "
            "package's DESCRIPTION and has no maintained Python equivalent "
            "inside spaCR's dependency set. sample_count_mean_variance "
            "dispatches to negative-binomial / Poisson / binomial, which "
            "spans the same over-, equi- and under-dispersed range in closed "
            "form. Nothing upstream used the higher moments."
        ),
        changes_the_number=False,
    ),
    Caveat(
        key="abundance_clipping",
        headline=(
            "If a gene's abundance times the well factor exceeds 1 it is "
            "clipped, and the realised constructs per well then sits BELOW "
            "the number you asked for."
        ),
        detail=(
            "That product is used as a Bernoulli probability and nothing "
            "constrains it to [0, 1]; at alpha 0.6 the most abundant gene "
            "saturates every well. R's rbinom returns NA there and carries "
            "on. This port clips and warns, and the screen reports the clip "
            "count — the answer is still usable, but not for the "
            "constructs-per-well figure written on it. Raise the library "
            "evenness (alpha) or lower constructs per well."
        ),
        changes_the_number=False,
    ),
)


def changes_the_number() -> Tuple[Caveat, ...]:
    """The caveats that belong next to the headline number.

    :returns: the subset of :data:`CAVEATS` whose ``changes_the_number`` is
        set, in declaration order.
    """
    return tuple(c for c in CAVEATS if c.changes_the_number)


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignSpec:
    """A screen as its designer describes it, with real-screen defaults.

    The first block is what the screen puts on the form. The second block is
    everything the simulator needs that a screener does not want to name;
    they are fields rather than constants so a test — or a later advanced
    panel — can vary them, and so the exported record of a run carries every
    number that went into it.

    :ivar n_genes: genes in the library.
    :ivar n_grnas_per_gene: guides per gene. Only reaches the simulation
        when ``score_per`` is ``"guide"``; see the ``no_guide_level_model``
        caveat.
    :ivar score_per: ``"gene"`` (guides pooled, what the real analysis and
        spaCRPower do) or ``"guide"`` (each construct is its own library
        unit).
    :ivar cells_per_well: mean cells imaged per well — the microscope time.
    :ivar wells_per_plate: plate format.
    :ivar n_plates: plates in the screen.
    :ivar constructs_per_well: mean library units spotted into each well.
        This is ``well_abundance_factor_mu``, and it means the same number
        of constructs per well at any library size because gene abundances
        sum to 1.
    :ivar background_positive_rate: probability a non-hit cell is called
        positive — the classifier's false-positive rate.
    :ivar effect_fold: how many times more often a hit-genotype cell is
        called positive. The effect size. 0.80 / 0.12 = 6.667 in the real
        screen.
    :ivar hit_rate: fraction of library units that are true hits.
    :ivar reads_per_well: mean sequencing reads per well.
    :ivar gene_abundance_alpha: Dirichlet concentration on library
        abundance; small is skewed.
    :ivar cells_per_well_var: variance of the imaged cell count per well.
    :ivar class_pos_var: variance of the hit-cell positive rate.
    :ivar class_neg_var: variance of the background positive rate.
    :ivar well_abundance_var: variance of the per-well abundance factor.
    :ivar sequencing_cells_per_well: cells per gene per well contributing
        DNA — far more than are imaged, because sequencing sees the whole
        well.
    :ivar pcr_factor_mu: log-scale mean of the per-well amplification.
    :ivar pcr_factor_var: log-scale variance of it.
    :ivar read_depth_cv: coefficient of variation of depth between wells.
    :ivar sequencing_error_rate: probability a barcode read is credited to
        the wrong gene. ``0.0`` reproduces spaCRPower and every power figure
        this screen has ever printed; see the
        ``sequencing_error_hides_untested_genes`` caveat for why the number
        that moves is not the one you would expect.
    :ivar min_cells_per_well: wells with fewer imaged cells than this are
        dropped before the fit. ``0`` keeps every well, which is what both
        packages did; see the ``thin_wells_count_the_same_as_full_ones``
        caveat.
    :ivar imaging_split: ``"abundance"`` or ``"uniform"``; see the
        ``even_split_overstates_power`` caveat.
    :ivar n_replicates: simulated screens per grid point.
    :ivar detection_auroc: the AUROC a replicate must reach to count as a
        detection.
    :ivar seed: master seed. Every number on the screen is reproducible
        from this plus the fields above.
    :ivar backend: inference backend, passed straight to
        :func:`spacr.power_model.scan_parameters`.
    """

    # -- what the form asks for ------------------------------------------
    n_genes: int = 452
    n_grnas_per_gene: int = 4
    score_per: str = "gene"
    cells_per_well: float = 123.0
    wells_per_plate: int = 384
    n_plates: int = 4
    constructs_per_well: float = 4.6
    background_positive_rate: float = 0.12
    # 0.80 / 0.12 to three decimals — the form carries three, so this is the
    # value a freshly-opened screen reads back and `DesignSpec() ==
    # PowerScreen().spec()` holds. The rounding puts the hit-cell rate at
    # 0.80004 rather than 0.80; expressing a pair of rates as their ratio
    # cannot do better, and 4e-5 of a probability changes no power.
    effect_fold: float = 6.667
    hit_rate: float = 0.025
    reads_per_well: float = 30000.0

    # -- held at the real screen's fitted values --------------------------
    gene_abundance_alpha: float = 0.6
    cells_per_well_var: float = 8000.0
    class_pos_var: float = 0.10
    class_neg_var: float = 0.01
    well_abundance_var: float = 1.0
    sequencing_cells_per_well: float = 1000.0
    pcr_factor_mu: float = 2.0
    pcr_factor_var: float = 1.0
    read_depth_cv: float = 0.35
    # Both default to the R behaviour rather than to the realistic value. A
    # simulator whose baseline moved under a version bump would make every
    # power figure already quoted from this screen wrong, and "the number
    # changed because spaCR got more honest" is indistinguishable from "the
    # number changed because something broke" to the person reading it.
    sequencing_error_rate: float = 0.0
    min_cells_per_well: int = 0
    imaging_split: str = "abundance"

    # -- how the sweep is run ---------------------------------------------
    n_replicates: int = 3
    detection_auroc: float = 0.80
    seed: int = 0
    backend: str = "torch"

    # -- derived -----------------------------------------------------------

    @property
    def n_wells(self) -> int:
        """Total wells in the screen."""
        return int(self.wells_per_plate) * int(self.n_plates)

    @property
    def n_library_units(self) -> int:
        """Rows of the library the model actually estimates a coefficient for.

        Genes when guides are pooled, constructs when they are not. This is
        what reaches ``n_genes_in_library``, and it is the ``p`` in the
        ``p >> n`` the horseshoe prior exists to handle.
        """
        if self.score_per == "guide":
            return int(self.n_genes) * max(1, int(self.n_grnas_per_gene))
        return int(self.n_genes)

    @property
    def hit_positive_rate(self) -> float:
        """Probability a hit-genotype cell is called positive.

        The effect size applied to the background rate. Capped just below 1
        because a rate of exactly 1 is only a realisable beta at zero
        variance, and silently swapping the requested spread for zero is the
        kind of substitution :func:`spacr.power_simulate.rbeta_mean_variance`
        refuses to make.
        """
        return float(min(0.999, self.background_positive_rate * self.effect_fold))

    @property
    def expected_hits(self) -> float:
        """Expected number of true hits in the library."""
        return float(self.n_library_units) * float(self.hit_rate)

    def validate(self) -> List[str]:
        """Every reason this design cannot be simulated, in plain words.

        Called before the run starts. The library would raise on each of
        these too — but it raises inside ``scan_parameters``, which records
        the point as ``status="failed"`` and carries on, so a design that is
        simply impossible would come back as a full grid of failures rather
        than as one sentence saying which box is wrong.

        :returns: list of problems; empty means the design will run.
        """
        problems: List[str] = []
        if int(self.n_genes) < 2:
            problems.append("The library needs at least 2 genes.")
        if int(self.n_grnas_per_gene) < 1:
            problems.append("There has to be at least 1 guide per gene.")
        if self.score_per not in ("gene", "guide"):
            problems.append(
                f"score_per must be 'gene' or 'guide', not {self.score_per!r}.")
        if self.imaging_split not in ("abundance", "uniform"):
            problems.append(
                f"imaging_split must be 'abundance' or 'uniform', not "
                f"{self.imaging_split!r}.")
        if self.n_wells < 2:
            problems.append(
                "A single well cannot identify any gene effect — with one "
                "observation the intercept explains the data exactly.")
        if not 0.0 < float(self.cells_per_well):
            problems.append("Cells imaged per well has to be more than 0.")
        if not 0.0 < float(self.constructs_per_well):
            problems.append("Constructs per well has to be more than 0.")
        if not 0.0 <= float(self.hit_rate) <= 1.0:
            problems.append("Hit prevalence is a fraction between 0 and 1.")
        if float(self.hit_rate) * float(self.n_library_units) < 1.0:
            problems.append(
                f"A prevalence of {self.hit_rate:g} over "
                f"{self.n_library_units} library units expects "
                f"{self.expected_hits:.2f} hits. With no hits in the library "
                "AUROC is undefined and the sweep reports nothing, which is "
                "not the same finding as no power.")
        if float(self.effect_fold) < 1.0:
            problems.append(
                "An effect size below 1 means a knockout makes cells LESS "
                "likely to be called positive. The model scores evidence in "
                "the other direction, so it would report near-chance rather "
                "than a detection. Use a fold of 1 or more.")
        for name, mean, var in (
            ("the background positive rate", self.background_positive_rate,
             self.class_neg_var),
            ("the hit-cell positive rate", self.hit_positive_rate,
             self.class_pos_var),
        ):
            mean = float(mean)
            if not 0.0 <= mean <= 1.0:
                problems.append(f"{name.capitalize()} has to be in [0, 1].")
                continue
            ceiling = mean * (1.0 - mean)
            if float(var) >= ceiling and float(var) > 0.0:
                problems.append(
                    f"{name.capitalize()} is {mean:.3f}, so its variance "
                    f"cannot reach {ceiling:.4f} (the Bernoulli bound). "
                    f"{float(var):g} is too spread out — lower the variance "
                    "or move the rate away from 0 and 1.")
        if float(self.reads_per_well) < 0.0:
            problems.append("Reads per well cannot be negative.")
        if int(self.n_replicates) < 1:
            problems.append("A sweep needs at least 1 replicate per point.")
        if not 0.0 < float(self.detection_auroc) <= 1.0:
            problems.append(
                "The detection threshold is an AUROC, so it lies in (0, 1].")
        return problems

    def with_values(self, **changes: Any) -> "DesignSpec":
        """Return a copy with ``changes`` applied. Thin wrapper on ``replace``."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Design -> library call
# ---------------------------------------------------------------------------

def simulator_kwargs(spec: DesignSpec) -> Dict[str, Any]:
    """The design as keyword arguments for the simulator, all held fixed.

    The single translation point between the form and
    :func:`spacr.power_simulate.simulate_screen`. A sweep is produced by
    overwriting ONE of these keys with a list — see :func:`cells_grid` and
    :func:`wells_grid` — and handing the whole dict to
    :func:`spacr.power_model.scan_parameters`, which is what makes a run
    from this screen identical to the same run typed at a Python prompt.

    :param spec: the design.
    :returns: keyword arguments for the simulator. Keys are the simulator's
        own parameter names, spelling included.
    """
    return {
        "n_genes_in_library": int(spec.n_library_units),
        "gene_abundance_alpha": float(spec.gene_abundance_alpha),
        "gene_hit_rate": float(spec.hit_rate),
        "n_wells_per_screen": int(spec.n_wells),
        "well_abundance_factor_mu": float(spec.constructs_per_well),
        "well_abundance_factor_var": float(spec.well_abundance_var),
        "imaging_n_cells_per_well_mu": float(spec.cells_per_well),
        "imaging_n_cells_per_well_var": float(spec.cells_per_well_var),
        "class_pos_mu": float(spec.hit_positive_rate),
        "class_pos_var": float(spec.class_pos_var),
        "class_neg_mu": float(spec.background_positive_rate),
        "class_neg_var": float(spec.class_neg_var),
        "sequencing_n_cells_per_well_lambda": float(spec.sequencing_cells_per_well),
        "pcr_factor_mu": float(spec.pcr_factor_mu),
        "pcr_factor_var": float(spec.pcr_factor_var),
        "n_reads_per_well": float(spec.reads_per_well),
        "read_depth_cv": float(spec.read_depth_cv),
        "sequencing_error_rate": float(spec.sequencing_error_rate),
        "min_cells_per_well": int(spec.min_cells_per_well),
        "imaging_split": str(spec.imaging_split),
    }


def _grid(centre: float, multipliers: Sequence[float], *,
          minimum: int = 1) -> List[float]:
    """Round ``centre * multipliers`` to whole units, dedupe, sort.

    :param centre: the user's own value, which must survive onto the grid.
    :param multipliers: fractions and multiples of it.
    :param minimum: floor for a rounded value.
    :returns: sorted unique grid, always containing ``round(centre)``.
    """
    values = {max(minimum, int(round(float(centre) * float(m))))
              for m in multipliers}
    values.add(max(minimum, int(round(float(centre)))))
    return [float(v) for v in sorted(values)]


def cells_grid(spec: DesignSpec) -> List[float]:
    """The cells-per-well sweep for ``spec``.

    :param spec: the design.
    :returns: sorted cells-per-well values, including the design's own.
    """
    return _grid(spec.cells_per_well, CELLS_MULTIPLIERS)


def wells_grid(spec: DesignSpec) -> List[float]:
    """The wells sweep for ``spec``.

    Floored at 2 wells rather than 1: a one-well fit is refused by
    :func:`spacr.power_model.fit_model`, and a grid point that can only ever
    be an error is not a data point.

    :param spec: the design.
    :returns: sorted well counts, including the design's own total.
    """
    return _grid(spec.n_wells, WELLS_MULTIPLIERS, minimum=2)


def estimate_runtime_s(spec: DesignSpec) -> float:
    """Very rough seconds for the two sweeps, for a "this will take…" line.

    Scales :data:`SECONDS_PER_FIT_REFERENCE` by design size relative to the
    real screen it was measured on, then multiplies by the number of fits.
    Deliberately crude: it exists so a user is not surprised by a five-minute
    wait, and the screen labels it as an estimate.

    :param spec: the design.
    :returns: estimated wall-clock seconds.
    """
    total = 0.0
    for wells in wells_grid(spec):
        total += (wells * spec.n_library_units) / _REFERENCE_CELLS_TIMES_WELLS
    total += len(cells_grid(spec)) * (
        (spec.n_wells * spec.n_library_units) / _REFERENCE_CELLS_TIMES_WELLS)
    return float(max(1.0, total * SECONDS_PER_FIT_REFERENCE
                     * max(1, int(spec.n_replicates))))


# ---------------------------------------------------------------------------
# Scan output -> power curve
# ---------------------------------------------------------------------------

#: Columns :func:`power_curve` returns, in order.
POWER_CURVE_COLUMNS: Tuple[str, ...] = (
    "value",
    "n_replicates",
    "n_ok",
    "n_not_converged",
    "n_failed",
    "n_detected",
    "power",
    "mean_auroc",
    "mean_ap",
    "ap_baseline",
)


def power_curve(scan: pd.DataFrame, sweep_column: str,
                detection_auroc: float) -> pd.DataFrame:
    """Collapse a :func:`~spacr.power_model.scan_parameters` frame into a curve.

    One row per swept value. ``power`` is the fraction of that point's
    replicates whose fit both succeeded and reached ``detection_auroc``:

    * the denominator is EVERY replicate at the point, including the ones
      that failed or did not converge;
    * ``mean_auroc`` and ``mean_ap`` average only the replicates that did
      converge, so they answer "how well did it do when it worked" — and
      ``n_not_converged`` / ``n_failed`` sit beside them so that question is
      never mistaken for the first one.

    :param scan: the frame ``scan_parameters`` returned. Needs the swept
        column plus ``status``, ``model_auroc``, ``model_ap`` and
        ``ap_baseline``.
    :param sweep_column: the simulator parameter that was swept.
    :param detection_auroc: the bar a replicate must clear to count.
    :returns: ``pandas.DataFrame`` with :data:`POWER_CURVE_COLUMNS`, sorted
        by ``value``. Empty input gives an empty frame with those columns
        rather than a KeyError.
    :raises KeyError: if ``sweep_column`` or any of ``status``,
        ``model_auroc``, ``model_ap``, ``ap_baseline`` is missing. Named
        rather than defaulted: a frame without ``status`` would otherwise be
        summarised as if every replicate had succeeded, which turns a
        mis-wired call into a design that looks better than it is.
    """
    if not isinstance(scan, pd.DataFrame) or len(scan) == 0:
        return pd.DataFrame(columns=list(POWER_CURVE_COLUMNS))
    needed = (sweep_column, "status", "model_auroc", "model_ap", "ap_baseline")
    missing = [name for name in needed if name not in scan.columns]
    if missing:
        raise KeyError(
            f"the scan result is missing {missing}; it has "
            f"{sorted(map(str, scan.columns))}")

    threshold = float(detection_auroc)
    rows: List[Dict[str, Any]] = []
    for value, block in scan.groupby(sweep_column, sort=True):
        status = block["status"].astype(str)
        auroc = pd.to_numeric(block["model_auroc"], errors="coerce")
        average_precision = pd.to_numeric(block["model_ap"], errors="coerce")
        baseline = pd.to_numeric(block["ap_baseline"], errors="coerce")

        ok = (status == "ok").to_numpy()
        # A detection needs BOTH an ok status and a score over the bar. The
        # `fillna(-inf)` is not cosmetic: a NaN comparison is False in numpy
        # too, but writing it down is what stops a later refactor from
        # "tidying" this into a dropna() and quietly changing the denominator.
        detected = ok & (auroc.fillna(-np.inf).to_numpy() >= threshold)
        rows.append({
            "value": float(value),
            "n_replicates": int(len(block)),
            "n_ok": int(ok.sum()),
            "n_not_converged": int((status == "not_converged").sum()),
            "n_failed": int((status == "failed").sum()),
            "n_detected": int(detected.sum()),
            "power": float(detected.sum()) / float(len(block)),
            "mean_auroc": float(auroc[ok].mean()) if ok.any() else float("nan"),
            "mean_ap": (float(average_precision[ok].mean()) if ok.any()
                        else float("nan")),
            "ap_baseline": (float(baseline[ok].mean()) if ok.any()
                            else float("nan")),
        })
    return pd.DataFrame(rows, columns=list(POWER_CURVE_COLUMNS))


def _fraction_words(detected: int, total: int) -> str:
    """``"4 of 5"``, for a sentence that has to survive being read aloud."""
    return f"{int(detected)} of {int(total)}"


def plain_sentence(spec: DesignSpec, cells: Optional[pd.DataFrame],
                   wells: Optional[pd.DataFrame] = None) -> str:
    """The one sentence the screen exists to print.

    Reads the user's OWN design off the curve — the point at
    ``spec.cells_per_well`` — rather than the best point on it, because the
    question was "is my design enough", not "what is the best design in this
    grid". If the run did not reach that point, says so instead of
    interpolating.

    :param spec: the design the sweep was run for.
    :param cells: the cells-per-well curve from :func:`power_curve`.
    :param wells: the wells curve, used only for the follow-on clause about
        what a different well count would buy.
    :returns: one or two sentences of plain English.
    """
    if cells is None or len(cells) == 0:
        return "No run yet — set the design and press Run."

    target = float(round(float(spec.cells_per_well)))
    row = cells.loc[cells["value"] == target]
    if len(row) == 0:
        # Not interpolated. A power curve read between its own points is a
        # number the simulation never produced.
        return (f"The sweep did not include {target:g} cells per well, so "
                "there is no simulated answer for this exact design.")
    row = row.iloc[0]

    effect = f"{float(spec.effect_fold):.2g}-fold"
    rates = (f"{float(spec.hit_positive_rate):.2f} vs "
             f"{float(spec.background_positive_rate):.2f} positive-call rate")
    power_pc = 100.0 * float(row["power"])
    sentence = (
        f"At {target:g} cells per well and "
        f"{float(spec.constructs_per_well):g} constructs per well, "
        f"{spec.n_wells} wells detect a {effect} effect ({rates}) in "
        f"{power_pc:.0f}% of simulations — "
        f"{_fraction_words(row['n_detected'], row['n_replicates'])} "
        f"replicates reached AUROC {float(spec.detection_auroc):.2f}."
    )

    withheld = int(row["n_not_converged"]) + int(row["n_failed"])
    if withheld:
        sentence += (
            f" {withheld} of those {int(row['n_replicates'])} did not produce "
            "a usable fit and count as non-detections, not as missing data.")
    if np.isfinite(float(row["mean_auroc"])):
        sentence += (
            f" Mean AUROC over the fits that converged: "
            f"{float(row['mean_auroc']):.2f}"
            f" (average precision {float(row['mean_ap']):.2f} against a "
            f"prevalence baseline of {float(row['ap_baseline']):.3f}).")

    if wells is not None and len(wells) > 1:
        best = wells.loc[wells["power"].idxmax()]
        if float(best["power"]) > float(row["power"]):
            sentence += (
                f" Going to {int(best['value'])} wells would take that to "
                f"{100.0 * float(best['power']):.0f}%.")
    return sentence
