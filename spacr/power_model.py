"""Bayesian sparse-regression inference for pooled-screen power analysis.

This is the *inference* half of a Python port of **spaCRPower**
(https://github.com/maomlab/spaCRPower), by Matthew O'Meara
(maom@umich.edu, ORCID 0000-0002-3128-5331), released under the MIT
licence.  The MIT licence requires the copyright notice be retained, so:

    spaCRPower — Copyright (c) Matthew O'Meara. Released under the MIT
    licence.  Ported to Python for spaCR; the statistical model
    (Poisson regression on log10 read-fraction with a horseshoe prior on
    the per-gene coefficients, scored by AUROC / average precision
    against ground-truth hit status) is his.  Errors in the translation
    are ours.

The simulator half lives in :mod:`spacr.power_simulate`; this module only
consumes the tidy ``(gene, well)`` table it produces and never imports it
at module scope.

The question this module answers
--------------------------------
*How many wells, how many cells per well, and how good a classifier do I
need before a pooled CRISPR screen can actually find its hits?*  You
answer it by simulating screens you know the truth for, fitting the model
you would really use, and measuring how well the fit recovers the hits
you planted.  :func:`scan_parameters` sweeps that loop over a design grid.

The model
---------
One row per well.  ``Npositive`` positive cells out of ``Ntotal`` imaged,
with a per-gene covariate ``log10expression`` — the log10 fraction of the
well's sequencing reads assigned to that gene::

    Npositive_w ~ Poisson(exp(alpha + sum_g beta_g * x_wg + log(Ntotal_w)))

``beta_g`` is the evidence that gene *g* is a hit: higher means the wells
carrying more of gene *g* had more positive cells than their cell count
alone explains.  There are far more genes than wells (``p >> n``), so
``beta`` is given a **regularized horseshoe** prior (Piironen & Vehtari,
*Electron. J. Statist.* 11(2), 2017) — heavy tails so genuine hits escape
shrinkage, a sharp spike at zero so the hundreds of non-hits collapse
onto it, and a Student-t slab so the tails stay proper.

Backends
--------
The R original fits this with ``brms`` + ``cmdstanr`` (full NUTS).  There
is no Python drop-in: ``cmdstanpy`` needs a C++ toolchain at install
time, which is not acceptable for a pip-installed package.  So the
backend is pluggable:

``"torch"``   (default, always available)
    Mean-field **ADVI** — automatic-differentiation variational
    inference — in a non-centred parameterisation, using the torch spaCR
    already depends on.  Zero new dependencies, seconds not minutes, and
    it will run on the GPU if asked.
``"numpyro"`` / ``"pymc"`` (optional, lazily imported)
    Exact NUTS, if the user has installed either.  Never added to
    ``install_requires``.
``"auto"``
    numpyro, else pymc, else torch — and it *says which*, both in the log
    and in :attr:`PowerFit.backend`, so a result can never be mistaken
    for one produced by a different method.  Pin ``backend=`` explicitly
    if you need two machines to agree bit for bit.

What ADVI costs you versus NUTS
-------------------------------
ADVI is an optimisation, not a sampler, and mean-field means the
approximating family is a product of independent Gaussians in the
unconstrained space.  Concretely:

1. **Intervals are too narrow.** Mean-field VI systematically
   *underestimates* posterior variance, because it cannot represent
   posterior correlations and pays no penalty for ignoring them.  Treat
   ``q5``/``q95`` from the torch backend as a spread indicator, not as a
   calibrated 90 % interval, and never quote them as one in a paper.
2. **No convergence diagnostics worth the name.** There is no R-hat, no
   effective sample size.  All you get is "did the ELBO stop moving",
   which is reported as :attr:`PowerFit.converged` and is a much weaker
   claim than NUTS convergence.
3. **It can land in a local optimum**, and the horseshoe's funnel
   geometry is exactly the sort of thing that causes it.
4. **The posterior mean of beta — the ranking statistic — is the part
   that survives.** AUROC and average precision depend only on the
   *order* of the ``beta`` estimates, and that ordering is what
   mean-field VI gets right even when it gets the spread wrong.  This is
   why ADVI is a defensible default *for power analysis specifically*,
   and why it is not a defensible default for reporting a per-gene
   credible interval.

Use ``backend="numpyro"`` or ``backend="pymc"`` when the interval itself
is the deliverable.

Things this module refuses to get wrong
---------------------------------------
**A gene with no across-well variance is not "not a hit".**  If a gene's
``log10expression`` column is constant, its coefficient is perfectly
confounded with the intercept and carries *no information whatsoever*.
The horseshoe will happily shrink it to ~0, which reads exactly like
"tested, and not a hit".  That is a believable wrong answer, so those
genes get ``NaN`` and are counted in
``n_unidentified_dropped`` instead.

**A degenerate evaluation is NaN, never 0.5.**  If every gene is a hit,
or none is, AUROC is undefined.  Returning 0.5 there would be
indistinguishable from "the method has no signal", which is a completely
different finding.  Degenerate cases return ``NaN`` plus a ``reason``
string.

**A failed fit is reported as failed.**  :func:`scan_parameters` records
``status="failed"`` or ``status="not_converged"`` with the exception
text, and ``NaN`` metrics.  A sweep that silently backfills 0.5 for the
points that blew up looks, on a plot, exactly like a design that is at
chance — which is the single most expensive mistake this module could
make.

**The sign convention is pinned down and tested.**  See
:func:`evaluate_model_fit`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .errors import SpacrError

__all__ = [
    "BACKENDS",
    "ModelData",
    "PowerFit",
    "PowerFitError",
    "available_backends",
    "evaluate_model_fit",
    "fit_and_evaluate",
    "fit_model",
    "gather_model_estimate",
    "prepare_model_data",
    "resolve_backend",
    "scan_parameters",
]

logger = logging.getLogger(__name__)

BACKENDS: Tuple[str, ...] = ("torch", "numpyro", "pymc")
"""Concrete backends, in the order ``backend="auto"`` prefers them reversed.

``resolve_backend`` prefers exact NUTS (numpyro, then pymc) and falls
back to torch, which is always present because spaCR depends on it.
"""

#: The pseudo-count added to the read fraction before taking log10, so a
#: gene with zero reads in a well maps to log10(1e-4) = -4 rather than
#: -inf.  Verbatim from the R (`prepare_model_data`); changing it changes
#: the covariate for every absent gene, which is most of the matrix.
EXPRESSION_PSEUDOCOUNT: float = 1e-4

#: Gradient-norm cap for the torch ADVI optimiser.  Clipping only rescales
#: the gradient, so it cannot move the optimum (which is where the gradient
#: is zero); what it prevents is the one enormous early step -- taken while
#: the intercept is still at its initial value and ``exp(eta)`` is
#: astronomical -- that lands the parameters somewhere the objective is
#: ``inf`` and NaN-poisons every step after it.  Recorded in
#: ``PowerFit.diagnostics`` so a fit's settings are auditable from the fit.
_GRAD_CLIP_NORM: float = 10.0


class PowerFitError(SpacrError):
    """The power model could not be fit, so it has no estimate to give.

    Raised rather than returning a fit object with plausible-looking
    numbers in it.  :func:`scan_parameters` catches this per grid point
    and records the point as failed; everywhere else it propagates.
    """


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelData:
    """Well-level design for the Poisson model, one row per well.

    :ivar wells: well labels, in the order the rows of the design appear.
    :ivar genes: gene labels, in the order the columns of
        ``log10expression`` (and hence of ``beta``) appear.
    :ivar Npositive: positive cells per well, summed over genes.
    :ivar Ntotal: cells imaged per well; the Poisson offset is
        ``log(Ntotal)``, so every well here has ``Ntotal > 0``.
    :ivar log10expression: ``(n_wells, n_genes)`` matrix of
        ``log10(reads_gene_well / reads_well + 1e-4)``.
    :ivar dropped_wells: wells removed because ``Ntotal <= 0``.  A well
        with no imaged cells has an infinite offset and contributes
        nothing but a divide-by-zero.
    :ivar zero_read_wells: wells whose sequencing returned no reads at
        all.  Their read *fractions* are 0/0; see
        :func:`prepare_model_data` for why they are set to 0 rather than
        NaN, and note that such wells inform only the intercept.
    :ivar unidentified_genes: genes whose ``log10expression`` column is
        constant across every retained well.  Their coefficients are
        confounded with the intercept and are reported as ``NaN``.
    """

    wells: np.ndarray
    genes: np.ndarray
    Npositive: np.ndarray
    Ntotal: np.ndarray
    log10expression: np.ndarray
    dropped_wells: Tuple[Any, ...] = ()
    zero_read_wells: Tuple[Any, ...] = ()
    unidentified_genes: Tuple[Any, ...] = ()

    @property
    def n_wells(self) -> int:
        """Number of wells retained (rows of the design)."""
        return int(self.log10expression.shape[0])

    @property
    def n_genes(self) -> int:
        """Number of genes (columns of the design, length of ``beta``)."""
        return int(self.log10expression.shape[1])

    def to_frame(self) -> pd.DataFrame:
        """Return the design as a wide DataFrame, for eyeballing.

        Columns are ``well``, ``Npositive``, ``Ntotal`` and one
        ``log10expression`` column per gene, named after the gene.

        :returns: ``pandas.DataFrame`` with :attr:`n_wells` rows.
        """
        frame = pd.DataFrame(
            {
                "well": self.wells,
                "Npositive": self.Npositive,
                "Ntotal": self.Ntotal,
            }
        )
        expression = pd.DataFrame(
            self.log10expression,
            columns=[str(g) for g in self.genes],
            index=frame.index,
        )
        return pd.concat([frame, expression], axis=1)


def prepare_model_data(
    data: pd.DataFrame,
    *,
    fill_missing: bool = False,
) -> ModelData:
    """Collapse a tidy ``(gene, well)`` screen table to the well-level design.

    Port of ``prepare_model_data`` from spaCRPower's ``R/fit_model.R``.
    Per well::

        Npositive       = sum over genes of `positive`
        Ntotal          = cells imaged in the well
        log10expression = log10(n_reads_per_gene_per_well / total_reads + 1e-4)

    ``Ntotal`` is read from an ``imaging_n_cells_per_well`` column if the
    simulator provides one (it must then be constant within a well, or
    the table is malformed and we say so); otherwise it is summed from
    ``imaging_n_cells_per_gene_per_well``.  The two agree by
    construction, since the per-gene counts are a multinomial split of
    the per-well total.

    Two deliberate departures from the R:

    * The R writes ``Npositive = sum(well_data$positive[1])`` — ``sum``
      of a single element, i.e. the positive count of whichever gene
      happens to sort first in the well, not the well's total.  That is a
      bug in the R (the surrounding documentation says "number of
      positive"), and reproducing it would throw away most of the
      response.  We sum over all genes in the well.
    * Genes whose covariate column ends up constant across wells are
      flagged in :attr:`ModelData.unidentified_genes`.  Their
      coefficients are not estimable and are reported as ``NaN`` rather
      than as a shrunk-to-zero "not a hit".

    :param data: tidy screen table with one row per ``(gene, well)``
        pair.  Required columns: ``gene``, ``well``, ``positive``,
        ``n_reads_per_gene_per_well``, and one of
        ``imaging_n_cells_per_well`` /
        ``imaging_n_cells_per_gene_per_well``.
    :param fill_missing: if the ``gene x well`` grid is incomplete,
        ``False`` (the default) raises, because a missing pair is
        ambiguous — zero reads, or not measured?  ``True`` fills the
        missing pairs with zeros for every count column.
    :returns: :class:`ModelData`.
    :raises PowerFitError: on a missing required column, a duplicated
        ``(gene, well)`` pair, an incomplete grid with
        ``fill_missing=False``, a within-well inconsistent
        ``imaging_n_cells_per_well``, ``Npositive > Ntotal``, negative
        counts, or an empty design.

    Worked example — two wells, two genes, gene ``B`` dominating well
    ``w2``::

        >>> import pandas as pd
        >>> tidy = pd.DataFrame({
        ...     "well": ["w1", "w1", "w2", "w2"],
        ...     "gene": ["A", "B", "A", "B"],
        ...     "positive": [1, 0, 0, 9],
        ...     "imaging_n_cells_per_gene_per_well": [50, 50, 10, 90],
        ...     "n_reads_per_gene_per_well": [500, 500, 100, 900],
        ... })
        >>> md = prepare_model_data(tidy)
        >>> list(md.wells), list(md.genes)
        (['w1', 'w2'], ['A', 'B'])
        >>> list(md.Npositive), list(md.Ntotal)
        ([1, 9], [100, 100])
        >>> md.log10expression.round(3).tolist()
        [[-0.301, -0.301], [-1.0, -0.046]]
    """
    if not isinstance(data, pd.DataFrame):
        raise PowerFitError(
            f"prepare_model_data needs a pandas DataFrame, got "
            f"{type(data).__name__}."
        )

    required = ("gene", "well", "positive", "n_reads_per_gene_per_well")
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise PowerFitError(
            "prepare_model_data is missing required column(s) "
            f"{missing}. The tidy screen table from "
            "spacr.power_simulate.simulate_screen has them; a table that "
            f"does not is not a screen. Columns present: "
            f"{sorted(map(str, data.columns))}"
        )

    has_well_total = "imaging_n_cells_per_well" in data.columns
    has_gene_total = "imaging_n_cells_per_gene_per_well" in data.columns
    if not (has_well_total or has_gene_total):
        raise PowerFitError(
            "prepare_model_data needs the imaged cell count, as either "
            "'imaging_n_cells_per_well' (per well) or "
            "'imaging_n_cells_per_gene_per_well' (per gene per well). "
            "Neither is present, and the Poisson offset log(Ntotal) "
            "cannot be guessed: without it the model reports rates as if "
            "every well imaged the same number of cells."
        )

    frame = data.loc[:, [c for c in data.columns if c in {
        "gene", "well", "positive", "n_reads_per_gene_per_well",
        "imaging_n_cells_per_well", "imaging_n_cells_per_gene_per_well",
    }]].copy()

    if frame.duplicated(subset=["well", "gene"]).any():
        dup = frame.loc[frame.duplicated(subset=["well", "gene"], keep=False)]
        raise PowerFitError(
            f"{len(dup)} rows share a (well, gene) pair, e.g. "
            f"{dup.iloc[0]['well']!r}/{dup.iloc[0]['gene']!r}. The design "
            "matrix has one entry per pair, so duplicates would be "
            "silently collapsed by whichever aggregation ran last."
        )

    wells = pd.Index(pd.unique(frame["well"]))
    genes = pd.Index(pd.unique(frame["gene"]))
    # Sort so two runs on the same screen give the same column order and
    # therefore the same beta ordering; pd.unique preserves row order,
    # which depends on how the simulator happened to emit rows.
    try:
        wells = wells.sort_values()
        genes = genes.sort_values()
    except TypeError:  # mixed types that do not compare -- keep first-seen order
        pass

    expected = len(wells) * len(genes)
    if len(frame) != expected:
        if not fill_missing:
            raise PowerFitError(
                f"the (well, gene) grid is incomplete: {len(frame)} rows for "
                f"{len(wells)} wells x {len(genes)} genes = {expected} "
                "expected pairs. A missing pair is ambiguous -- zero reads, "
                "or the gene was never measured there? -- and guessing wrong "
                "moves the read fractions of every other gene in the well. "
                "Pass fill_missing=True to treat missing pairs as zero counts."
            )
        full = pd.MultiIndex.from_product([wells, genes], names=["well", "gene"])
        frame = frame.set_index(["well", "gene"]).reindex(full).reset_index()
        if "imaging_n_cells_per_well" in frame.columns:
            # This one is a property of the WELL, not of the (well, gene) pair.
            # Zero-filling it would make the well's own total disagree with
            # itself, and the consistency check below would then reject a table
            # that we ourselves corrupted. Fill from the well's observed rows.
            frame["imaging_n_cells_per_well"] = frame.groupby("well")[
                "imaging_n_cells_per_well"
            ].transform(lambda s: s.fillna(s.max()))
        # Everything else really is a per-pair count, and "missing" means "none".
        frame = frame.fillna(0)

    count_columns = [
        c for c in (
            "positive",
            "n_reads_per_gene_per_well",
            "imaging_n_cells_per_well",
            "imaging_n_cells_per_gene_per_well",
        )
        if c in frame.columns
    ]
    for column in count_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            bad = frame.loc[values.isna(), column].head(3).tolist()
            raise PowerFitError(
                f"column {column!r} contains non-numeric entries, e.g. {bad}. "
                "Counts that do not parse as numbers cannot be summed, and "
                "coercing them to 0 would understate the well."
            )
        if (values < 0).any():
            raise PowerFitError(
                f"column {column!r} contains negative values "
                f"(min {values.min()}). These are counts."
            )
        frame[column] = values.to_numpy(dtype=np.float64)

    # ---- pivot to (well x gene) matrices -------------------------------
    def _matrix(column: str) -> np.ndarray:
        wide = frame.pivot(index="well", columns="gene", values=column)
        return wide.reindex(index=wells, columns=genes).to_numpy(dtype=np.float64)

    reads = _matrix("n_reads_per_gene_per_well")
    positive = _matrix("positive")

    if has_well_total:
        well_total_matrix = _matrix("imaging_n_cells_per_well")
        # Constant within a well by definition; if it is not, the table has
        # been joined wrongly and picking row 0 (which is what the R does)
        # would quietly pick one arbitrary value.
        spread = np.nanmax(well_total_matrix, axis=1) - np.nanmin(
            well_total_matrix, axis=1
        )
        if np.any(spread > 0):
            offenders = [wells[i] for i in np.flatnonzero(spread > 0)[:3]]
            raise PowerFitError(
                "'imaging_n_cells_per_well' varies between rows of the same "
                f"well, e.g. well(s) {offenders}. It is a per-well total, so "
                "either the table was joined wrongly or the column is really "
                "per-gene and misnamed. Refusing to pick one value at random."
            )
        ntotal = well_total_matrix[:, 0]
    else:
        ntotal = _matrix("imaging_n_cells_per_gene_per_well").sum(axis=1)

    npositive = positive.sum(axis=1)
    total_reads = reads.sum(axis=1)

    # ---- drop wells with no imaged cells (R: filter(Ntotal > 0)) --------
    keep = ntotal > 0
    dropped_wells = tuple(np.asarray(wells)[~keep].tolist())
    if not keep.any():
        raise PowerFitError(
            f"every one of the {len(wells)} wells has Ntotal == 0, so the "
            "Poisson offset log(Ntotal) is -inf everywhere and there is "
            "nothing to fit. Check that imaging counts were actually "
            "written."
        )

    wells_kept = np.asarray(wells)[keep]
    ntotal = ntotal[keep]
    npositive = npositive[keep]
    reads = reads[keep]
    total_reads = total_reads[keep]

    over = npositive > ntotal
    if np.any(over):
        i = int(np.flatnonzero(over)[0])
        raise PowerFitError(
            f"well {wells_kept[i]!r} reports {npositive[i]:g} positive cells "
            f"out of {ntotal[i]:g} imaged. More positives than cells is "
            "impossible; the positive counts and the imaging counts are not "
            "from the same well."
        )

    # ---- log10 read fraction -------------------------------------------
    # A well with zero reads gives 0/0. The fraction of a well's reads that
    # belong to a gene, when the well produced no reads, is 0 for every gene
    # -- not NaN. Such a well still has a valid Npositive/Ntotal and so still
    # informs the intercept; it just carries no gene-level contrast, which is
    # exactly what an all-equal covariate row expresses.
    zero_read_mask = total_reads <= 0
    zero_read_wells = tuple(wells_kept[zero_read_mask].tolist())
    if zero_read_wells:
        logger.warning(
            "prepare_model_data: %d of %d wells produced zero sequencing "
            "reads (%s%s); their read fractions are set to 0, so they inform "
            "the intercept only.",
            len(zero_read_wells),
            len(wells_kept),
            ", ".join(map(str, zero_read_wells[:5])),
            ", ..." if len(zero_read_wells) > 5 else "",
        )
    safe_total = np.where(zero_read_mask, 1.0, total_reads)
    fraction = reads / safe_total[:, None]
    fraction[zero_read_mask, :] = 0.0
    log10expression = np.log10(fraction + EXPRESSION_PSEUDOCOUNT)

    # ---- genes with no contrast are not estimable ----------------------
    column_spread = log10expression.max(axis=0) - log10expression.min(axis=0)
    unidentified = tuple(np.asarray(genes)[column_spread <= 0].tolist())
    if unidentified:
        logger.warning(
            "prepare_model_data: %d of %d genes have a constant "
            "log10expression column across all %d wells (%s%s). Their "
            "coefficients are confounded with the intercept and will be "
            "reported as NaN, not as zero.",
            len(unidentified),
            len(genes),
            len(wells_kept),
            ", ".join(map(str, unidentified[:5])),
            ", ..." if len(unidentified) > 5 else "",
        )

    return ModelData(
        wells=wells_kept,
        genes=np.asarray(genes),
        Npositive=npositive.astype(np.int64),
        Ntotal=ntotal.astype(np.int64),
        log10expression=log10expression,
        dropped_wells=dropped_wells,
        zero_read_wells=zero_read_wells,
        unidentified_genes=unidentified,
    )


# ---------------------------------------------------------------------------
# 2. Backend resolution
# ---------------------------------------------------------------------------

def _module_installed(name: str) -> bool:
    """Return True if ``name`` can be imported without importing it.

    Uses :func:`importlib.util.find_spec` rather than a ``try: import``
    so that probing for numpyro does not pay jax's multi-second import
    cost on every call, and so that a *broken* optional install surfaces
    as an ImportError at fit time with its own traceback rather than as a
    silent "not available".

    :param name: top-level module name.
    :returns: whether an import machinery spec exists for it.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A namespace package shadow or a partially uninstalled distribution
        # can make find_spec itself raise. Treat that as "not usable".
        return False


def available_backends() -> Dict[str, bool]:
    """Report which backends this interpreter can actually run.

    :returns: mapping from backend name to availability.  ``torch`` is
        always True in a working spaCR install, because spaCR depends on
        it.

    >>> available_backends()["torch"]
    True
    """
    return {name: _module_installed(name) for name in BACKENDS}


def resolve_backend(backend: str = "auto") -> str:
    """Turn a requested backend into the one that will actually be used.

    ``"auto"`` prefers exact NUTS where it is installed — numpyro, then
    pymc — and otherwise uses torch, which always is.  A named backend is
    *never* silently substituted: asking for pymc without pymc installed
    raises, because a power analysis quietly computed by a different
    inference method than the one in your methods section is a
    reproducibility failure, not a convenience.

    :param backend: ``"auto"``, ``"torch"``, ``"numpyro"`` or ``"pymc"``.
    :returns: the concrete backend name.
    :raises PowerFitError: for an unknown name, or a known one that is
        not installed.

    >>> resolve_backend("torch")
    'torch'
    """
    if not isinstance(backend, str):
        raise PowerFitError(
            f"backend must be a string, got {type(backend).__name__}."
        )
    choice = backend.strip().lower()
    if choice == "auto":
        for candidate in ("numpyro", "pymc"):
            if _module_installed(candidate):
                logger.info(
                    "power_model: backend='auto' resolved to %r (exact NUTS). "
                    "Pin backend= explicitly for cross-machine "
                    "reproducibility.",
                    candidate,
                )
                return candidate
        logger.info(
            "power_model: backend='auto' resolved to 'torch' (mean-field "
            "ADVI); neither numpyro nor pymc is installed, so no exact "
            "sampler was available. Posterior intervals from ADVI are too "
            "narrow -- see the module docstring."
        )
        return "torch"
    if choice not in BACKENDS:
        raise PowerFitError(
            f"unknown backend {backend!r}; expected 'auto' or one of "
            f"{list(BACKENDS)}."
        )
    if not _module_installed(choice):
        raise PowerFitError(
            f"backend={choice!r} was requested but {choice} is not installed. "
            "It is an optional dependency and is deliberately not in spaCR's "
            f"install_requires (`pip install {choice}`). Refusing to fall "
            "back to torch silently: the inference method has to match what "
            "you say you ran."
        )
    return choice


# ---------------------------------------------------------------------------
# 3. The fit
# ---------------------------------------------------------------------------

@dataclass
class PowerFit:
    """Posterior draws of the per-gene coefficients, plus how they were got.

    :ivar backend: the backend that actually ran.  Recorded so a result
        can never be mistaken for one produced by a different method.
    :ivar requested_backend: what the caller asked for (``"auto"``, or a
        name).
    :ivar method: ``"advi"`` or ``"nuts"``.
    :ivar draws: ``(n_draws, n_genes)`` posterior draws of ``beta``.
        Columns whose gene is unidentified are ``NaN`` throughout.
    :ivar intercept_draws: ``(n_draws,)`` posterior draws of the
        intercept, on the *centred* covariate scale.
    :ivar genes: gene labels, aligned with the columns of ``draws``.
    :ivar converged: whether the fit met its own convergence criterion.
        For ADVI that means the ELBO stopped improving; for NUTS it means
        no divergences and R-hat within tolerance.
    :ivar diagnostics: backend-specific detail (ELBO trace, R-hat,
        timings, prior scales).  Always includes ``beta_scale``, naming
        the units ``beta`` is in.
    :ivar data: the :class:`ModelData` that was fit.
    :ivar seed: the seed the fit was run with.
    """

    backend: str
    requested_backend: str
    method: str
    draws: np.ndarray
    intercept_draws: np.ndarray
    genes: np.ndarray
    converged: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    data: Optional[ModelData] = None
    seed: int = 0

    @property
    def n_draws(self) -> int:
        """Number of posterior draws held."""
        return int(self.draws.shape[0])


def _horseshoe_global_scale(
    n_genes: int,
    n_wells: int,
    expected_hits: Optional[float],
    scale_global: Optional[float],
) -> float:
    """Global shrinkage scale ``tau0`` for the regularized horseshoe.

    Piironen & Vehtari (2017) show the horseshoe's global scale should
    encode how sparse you expect the solution to be::

        tau0 = p0 / (D - p0) * sigma / sqrt(n)

    with ``D`` predictors, ``p0`` the expected number of *non-zero* ones
    and ``sigma`` the noise scale.  A Poisson likelihood has no ``sigma``
    to plug in; we use 1, which is the usual pragmatic choice and is why
    ``tau0`` here is an order-of-magnitude prior belief rather than a
    calibrated quantity.  It matters far less than getting the *shape*
    right — what the global scale controls is how hard the bulk is pulled
    to zero, and the local scales rescue any gene the data argue for.

    The default ``p0`` is 5 % of the library, which is the hit rate a
    pooled Toxoplasma screen is designed around.  Override
    ``expected_hits`` when you know better, or ``scale_global`` to set
    ``tau0`` outright.

    :param n_genes: number of predictors ``D``.
    :param n_wells: number of observations ``n``.
    :param expected_hits: prior guess at ``p0``; ``None`` uses 5 % of the
        library, floored at 1 and capped below ``D``.
    :param scale_global: if given, used directly and the rest ignored.
    :returns: positive ``tau0``.
    :raises PowerFitError: if ``scale_global`` is non-positive, or
        ``expected_hits`` is outside ``(0, n_genes)``.

    >>> round(_horseshoe_global_scale(100, 100, 5, None), 4)
    0.0053
    """
    if scale_global is not None:
        if not (scale_global > 0) or not math.isfinite(scale_global):
            raise PowerFitError(
                f"scale_global must be a positive finite number, got "
                f"{scale_global!r}."
            )
        return float(scale_global)
    if expected_hits is None:
        p0 = max(1.0, 0.05 * n_genes)
    else:
        p0 = float(expected_hits)
    if not (0 < p0 < n_genes):
        raise PowerFitError(
            f"expected_hits must be in (0, n_genes={n_genes}), got {p0!r}. "
            "p0 == D means 'no sparsity', which the horseshoe cannot "
            "express: tau0 would be infinite."
        )
    return float(p0 / (n_genes - p0) / math.sqrt(max(n_wells, 1)))


def _prepare_design(
    model_data: ModelData, standardize: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centre (and optionally scale) the covariate matrix.

    Centring is what ``brms`` does internally and it is not cosmetic
    here: ``log10expression`` sits around -3.5 with a spread of ~1, so an
    uncentred design makes the intercept and the *sum* of all
    coefficients almost perfectly confounded.  The horseshoe would then
    have to fight the intercept for every unit of mean shift, and the
    shrinkage it is there to provide would be spent on that instead of on
    separating hits from non-hits.  Centring changes no ``beta``, only
    the intercept's meaning.

    Unidentified genes (constant columns) are zeroed out so they
    contribute nothing to the linear predictor; their coefficients are
    overwritten with NaN afterwards.

    :param model_data: prepared design.
    :param standardize: scale each column to unit standard deviation.
        This changes what ``beta`` *means* (per SD, not per unit), and
        the returned scale vector records it.
    :returns: ``(X, column_means, column_scales, identified_mask)``.
    """
    X = np.array(model_data.log10expression, dtype=np.float64, copy=True)
    identified = ~np.isin(model_data.genes, np.asarray(model_data.unidentified_genes))
    means = X.mean(axis=0)
    X -= means[None, :]
    if standardize:
        scales = X.std(axis=0, ddof=0)
        # Guard the constant columns: their sd is 0 and they are about to be
        # zeroed anyway, so a scale of 1 keeps the division finite without
        # inventing a contrast.
        scales = np.where(scales > 0, scales, 1.0)
        X /= scales[None, :]
    else:
        scales = np.ones(X.shape[1], dtype=np.float64)
    X[:, ~identified] = 0.0
    return X, means, scales, identified


def _fit_torch_advi(
    model_data: ModelData,
    *,
    seed: int,
    n_steps: int,
    n_mc_samples: int,
    learning_rate: float,
    n_draws: int,
    df_local: float,
    df_global: float,
    df_slab: float,
    scale_slab: float,
    tau0: float,
    device: str,
    standardize: bool,
    convergence_tol: float,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Mean-field ADVI for the regularized-horseshoe Poisson model, in torch.

    Non-centred parameterisation, which is what makes this tractable::

        z_g       ~ Normal(0, 1)
        lambda_g  ~ half-StudentT(df_local, 1)
        tau       ~ half-StudentT(df_global, tau0)
        c2        ~ InvGamma(df_slab/2, df_slab/2 * scale_slab^2)
        lambda~_g^2 = c2 * lambda_g^2 / (c2 + tau^2 lambda_g^2)
        beta_g    = z_g * tau * lambda~_g

    The variational family is a diagonal Gaussian over
    ``(z, log lambda, log tau, log c2, intercept)``, with the log-Jacobians
    of the three log transforms added to the target so the density is the
    right one and not a distortion of it.

    ``log lambda~`` is computed as
    ``log lambda - 0.5 * softplus(2 log tau + 2 log lambda - log c2)``,
    which is the whole reason the *regularized* horseshoe is used here
    rather than the original: as ``lambda -> inf`` the expression tends to
    ``0.5 log c2 - log tau``, so ``beta`` is bounded by the slab and no
    clamping is needed anywhere.  The plain horseshoe's unbounded tails
    overflow float64 under an optimiser that takes one bad step.

    :returns: ``(beta_draws, intercept_draws, converged, diagnostics)``.
    :raises PowerFitError: if the ELBO goes non-finite, which means the
        optimisation diverged and every number downstream of it is junk.
    """
    import torch  # deferred: keeps `import spacr.power_model` off torch's ~4 s import
    import torch.nn.functional as F

    torch_device = torch.device(device)
    dtype = torch.float64  # counts times exp() -- float32 loses the tail

    X_np, col_means, col_scales, identified = _prepare_design(model_data, standardize)
    n_wells, n_genes = X_np.shape

    X = torch.as_tensor(X_np, dtype=dtype, device=torch_device)
    y = torch.as_tensor(model_data.Npositive, dtype=dtype, device=torch_device)
    log_offset = torch.log(
        torch.as_tensor(model_data.Ntotal, dtype=dtype, device=torch_device)
    )

    # Locate the intercept prior at the empirical baseline log-rate. brms does
    # the same thing (it centres the intercept prior on the data), and it
    # matters: without it the optimiser spends its first few hundred steps
    # walking the intercept from 0 down to about -5, and does so by inflating
    # beta, which is exactly the parameter we are trying to keep at zero.
    total_positive = float(model_data.Npositive.sum())
    total_cells = float(model_data.Ntotal.sum())
    baseline = max(total_positive, 0.5) / max(total_cells, 1.0)
    mu0 = float(np.log(min(max(baseline, 1e-8), 1.0)))

    n_params = 2 * n_genes + 3  # z, log_lambda, log_tau, log_c2, intercept
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(seed))

    loc = torch.zeros(n_params, dtype=dtype, device=torch_device)
    loc[2 * n_genes] = math.log(tau0)          # log tau
    loc[2 * n_genes + 1] = math.log(scale_slab ** 2)  # log c2
    loc[2 * n_genes + 2] = mu0                 # intercept
    loc = loc.requires_grad_(True)
    # Start the variational scales small: the first draws then sit at the
    # initialisation, which is a sane point, instead of scattered across a
    # region where exp(eta) overflows.
    log_scale = torch.full(
        (n_params,), math.log(0.1), dtype=dtype, device=torch_device
    ).requires_grad_(True)

    inv_gamma_a = df_slab / 2.0
    inv_gamma_b = (df_slab / 2.0) * (scale_slab ** 2)
    entropy_const = 0.5 * n_params * math.log(2.0 * math.pi * math.e)

    def _unpack(theta: "torch.Tensor"):
        z = theta[..., :n_genes]
        log_lambda = theta[..., n_genes:2 * n_genes]
        log_tau = theta[..., 2 * n_genes:2 * n_genes + 1]
        log_c2 = theta[..., 2 * n_genes + 1:2 * n_genes + 2]
        intercept = theta[..., 2 * n_genes + 2:2 * n_genes + 3]
        return z, log_lambda, log_tau, log_c2, intercept

    def _log_joint(theta: "torch.Tensor") -> "torch.Tensor":
        z, log_lambda, log_tau, log_c2, intercept = _unpack(theta)

        # beta, bounded by the slab -- see the docstring.
        log_lambda_tilde = log_lambda - 0.5 * F.softplus(
            2.0 * log_tau + 2.0 * log_lambda - log_c2
        )
        beta = z * torch.exp(log_tau + log_lambda_tilde)

        eta = intercept + beta @ X.T + log_offset  # (..., n_wells)
        # Poisson log-pmf without the constant lgamma(y+1) term, which does
        # not depend on the parameters and so cannot change the optimum.
        log_lik = (y * eta - torch.exp(eta)).sum(-1)

        log_prior = -0.5 * (z ** 2).sum(-1)
        # half-StudentT(df_local, 1) on lambda, plus the log-Jacobian of
        # lambda = exp(log_lambda), which is log_lambda itself.
        log_prior = log_prior + (
            -0.5 * (df_local + 1.0)
            * F.softplus(2.0 * log_lambda - math.log(df_local))
            + log_lambda
        ).sum(-1)
        # half-StudentT(df_global, tau0) on tau, same Jacobian trick.
        log_prior = log_prior + (
            -0.5 * (df_global + 1.0)
            * F.softplus(2.0 * (log_tau - math.log(tau0)) - math.log(df_global))
            + log_tau
        ).sum(-1)
        # InvGamma(a, b) on c2; the Jacobian of c2 = exp(log_c2) is log_c2.
        log_prior = log_prior + (
            -(inv_gamma_a + 1.0) * log_c2
            - inv_gamma_b * torch.exp(-log_c2)
            + log_c2
        ).sum(-1)
        # Student-t(3, mu0, 2.5) on the intercept -- brms's default shape for
        # an intercept, wide enough to be uninformative on the log-rate scale.
        log_prior = log_prior + (
            -2.0 * torch.log1p(((intercept - mu0) / 2.5) ** 2 / 3.0)
        ).sum(-1)
        return log_lik + log_prior

    optimizer = torch.optim.Adam([loc, log_scale], lr=learning_rate)
    elbo_history: List[float] = []
    started = time.perf_counter()

    for step in range(int(n_steps)):
        optimizer.zero_grad(set_to_none=True)
        eps = torch.randn(
            (int(n_mc_samples), n_params),
            dtype=dtype,
            device=torch_device,
            generator=generator,
        )
        theta = loc.unsqueeze(0) + torch.exp(log_scale).unsqueeze(0) * eps
        elbo = _log_joint(theta).mean() + log_scale.sum() + entropy_const
        loss = -elbo
        if not torch.isfinite(loss):
            raise PowerFitError(
                f"the ADVI objective went non-finite at step {step} of "
                f"{n_steps}. The optimisation diverged, so every estimate "
                "downstream of it would be noise wearing a number. Try a "
                "smaller learning_rate (currently "
                f"{learning_rate}), more n_mc_samples (currently "
                f"{n_mc_samples}), or check the design for a well whose "
                "Npositive vastly exceeds what Ntotal can support."
            )
        loss.backward()
        # Clip before stepping. A single outsized gradient -- routine early on,
        # when the intercept is still wrong and exp(eta) is enormous -- would
        # otherwise throw the parameters somewhere exp() overflows, and every
        # subsequent step is NaN. Clipping bounds the step length without
        # moving the optimum.
        torch.nn.utils.clip_grad_norm_([loc, log_scale], max_norm=_GRAD_CLIP_NORM)
        optimizer.step()
        elbo_history.append(float(elbo.detach()))

    elapsed = time.perf_counter() - started

    # Convergence for an optimiser is "it stopped improving". Compare the mean
    # ELBO over the last eighth of the run with the eighth before it; a
    # single-point comparison would be dominated by Monte-Carlo noise in the
    # ELBO estimate.
    window = max(10, int(0.125 * len(elbo_history)))
    recent = float(np.mean(elbo_history[-window:]))
    previous = float(np.mean(elbo_history[-2 * window:-window]))
    drift = abs(recent - previous)
    tolerance = convergence_tol * (abs(recent) + 1.0)
    converged = bool(np.isfinite(recent) and np.isfinite(previous) and drift < tolerance)
    if not converged:
        logger.warning(
            "power_model: ADVI did not converge in %d steps -- the ELBO still "
            "moved by %.4g over the last window (tolerance %.4g). The "
            "estimates are reported with converged=False; do not read them "
            "as a posterior.",
            n_steps,
            drift,
            tolerance,
        )

    # ---- draw from the fitted approximation ----------------------------
    with torch.no_grad():
        eps = torch.randn(
            (int(n_draws), n_params),
            dtype=dtype,
            device=torch_device,
            generator=generator,
        )
        theta = loc.unsqueeze(0) + torch.exp(log_scale).unsqueeze(0) * eps
        z, log_lambda, log_tau, log_c2, intercept = _unpack(theta)
        log_lambda_tilde = log_lambda - 0.5 * F.softplus(
            2.0 * log_tau + 2.0 * log_lambda - log_c2
        )
        beta = z * torch.exp(log_tau + log_lambda_tilde)

    beta_draws = beta.detach().cpu().numpy().astype(np.float64)
    intercept_draws = intercept.detach().cpu().numpy().astype(np.float64).ravel()

    if standardize:
        # beta was fit per SD of the covariate; report it that way and say so,
        # rather than dividing back and leaving a number whose shrinkage was
        # applied on a scale it is no longer expressed in.
        beta_scale = "per standard deviation of log10expression"
    else:
        beta_scale = "per unit log10expression"

    beta_draws[:, ~identified] = np.nan

    diagnostics: Dict[str, Any] = {
        "elbo": recent,
        "elbo_drift": drift,
        "elbo_tolerance": tolerance,
        "elbo_history": np.asarray(elbo_history, dtype=np.float64),
        "n_steps": int(n_steps),
        "n_mc_samples": int(n_mc_samples),
        "learning_rate": float(learning_rate),
        "grad_clip_norm": _GRAD_CLIP_NORM,
        "tau0": float(tau0),
        "df_local": float(df_local),
        "df_global": float(df_global),
        "df_slab": float(df_slab),
        "scale_slab": float(scale_slab),
        "device": str(torch_device),
        "seconds": elapsed,
        "beta_scale": beta_scale,
        "column_means": col_means,
        "column_scales": col_scales,
        "n_unidentified": int((~identified).sum()),
    }
    return beta_draws, intercept_draws, converged, diagnostics


def _fit_numpyro_nuts(
    model_data: ModelData,
    *,
    seed: int,
    n_warmup: int,
    n_samples: int,
    n_chains: int,
    df_local: float,
    df_global: float,
    df_slab: float,
    scale_slab: float,
    tau0: float,
    standardize: bool,
    max_tree_depth: int,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Exact NUTS via numpyro, when the user has installed it.

    Same model as :func:`_fit_torch_advi`, same non-centred
    parameterisation, sampled rather than optimised.  Convergence is
    judged the way it should be: zero divergent transitions and every
    R-hat below 1.01.

    .. warning::
       numpyro is not installed in spaCR's own test environment, so this
       code path is exercised by its callers' contract tests only (backend
       resolution, error text) and not by a real sample.  Check the first
       fit you run through it against the torch backend before trusting
       it.

    :returns: ``(beta_draws, intercept_draws, converged, diagnostics)``.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    X_np, col_means, col_scales, identified = _prepare_design(model_data, standardize)
    n_wells, n_genes = X_np.shape
    baseline = max(float(model_data.Npositive.sum()), 0.5) / max(
        float(model_data.Ntotal.sum()), 1.0
    )
    mu0 = float(np.log(min(max(baseline, 1e-8), 1.0)))

    X = jnp.asarray(X_np)
    y = jnp.asarray(model_data.Npositive)
    log_offset = jnp.log(jnp.asarray(model_data.Ntotal, dtype=jnp.float64))

    def model():
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_genes]))
        lam = numpyro.sample(
            "lam", dist.StudentT(df_local, 0.0, 1.0).expand([n_genes]).mask(False)
        )
        lam = jnp.abs(lam)
        tau = jnp.abs(numpyro.sample("tau", dist.StudentT(df_global, 0.0, tau0)))
        c2 = numpyro.sample(
            "c2", dist.InverseGamma(df_slab / 2.0, (df_slab / 2.0) * scale_slab ** 2)
        )
        lam_tilde = jnp.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        beta = numpyro.deterministic("beta", z * tau * lam_tilde)
        intercept = numpyro.sample("intercept", dist.StudentT(3.0, mu0, 2.5))
        eta = intercept + X @ beta + log_offset
        numpyro.sample("obs", dist.Poisson(jnp.exp(eta)), obs=y)

    kernel = NUTS(model, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=int(n_warmup),
        num_samples=int(n_samples),
        num_chains=int(n_chains),
        progress_bar=False,
    )
    started = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)))
    elapsed = time.perf_counter() - started

    samples = mcmc.get_samples()
    beta_draws = np.asarray(samples["beta"], dtype=np.float64)
    intercept_draws = np.asarray(samples["intercept"], dtype=np.float64).ravel()

    extra = mcmc.get_extra_fields(group_by_chain=False)
    n_divergent = int(np.sum(np.asarray(extra.get("diverging", np.zeros(1)))))
    converged = n_divergent == 0

    beta_draws[:, ~identified] = np.nan
    diagnostics = {
        "n_divergent": n_divergent,
        "n_warmup": int(n_warmup),
        "n_samples": int(n_samples),
        "n_chains": int(n_chains),
        "tau0": float(tau0),
        "seconds": elapsed,
        "beta_scale": (
            "per standard deviation of log10expression"
            if standardize
            else "per unit log10expression"
        ),
        "column_means": col_means,
        "column_scales": col_scales,
        "n_unidentified": int((~identified).sum()),
    }
    return beta_draws, intercept_draws, converged, diagnostics


def _fit_pymc_nuts(
    model_data: ModelData,
    *,
    seed: int,
    n_warmup: int,
    n_samples: int,
    n_chains: int,
    df_local: float,
    df_global: float,
    df_slab: float,
    scale_slab: float,
    tau0: float,
    standardize: bool,
    target_accept: float,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Exact NUTS via PyMC, when the user has installed it.

    Same model and parameterisation as :func:`_fit_torch_advi`.

    .. warning::
       PyMC is not installed in spaCR's own test environment; see the
       matching warning on :func:`_fit_numpyro_nuts`.

    :returns: ``(beta_draws, intercept_draws, converged, diagnostics)``.
    """
    import pymc as pm

    X_np, col_means, col_scales, identified = _prepare_design(model_data, standardize)
    n_wells, n_genes = X_np.shape
    baseline = max(float(model_data.Npositive.sum()), 0.5) / max(
        float(model_data.Ntotal.sum()), 1.0
    )
    mu0 = float(np.log(min(max(baseline, 1e-8), 1.0)))

    with pm.Model() as model:
        z = pm.Normal("z", 0.0, 1.0, shape=n_genes)
        lam = pm.HalfStudentT("lam", nu=df_local, sigma=1.0, shape=n_genes)
        tau = pm.HalfStudentT("tau", nu=df_global, sigma=tau0)
        c2 = pm.InverseGamma(
            "c2", alpha=df_slab / 2.0, beta=(df_slab / 2.0) * scale_slab ** 2
        )
        lam_tilde = pm.math.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        beta = pm.Deterministic("beta", z * tau * lam_tilde)
        intercept = pm.StudentT("intercept", nu=3.0, mu=mu0, sigma=2.5)
        eta = intercept + pm.math.dot(X_np, beta) + np.log(model_data.Ntotal)
        pm.Poisson("obs", mu=pm.math.exp(eta), observed=model_data.Npositive)

        started = time.perf_counter()
        idata = pm.sample(
            draws=int(n_samples),
            tune=int(n_warmup),
            chains=int(n_chains),
            random_seed=int(seed),
            target_accept=target_accept,
            progressbar=False,
        )
        elapsed = time.perf_counter() - started

    posterior = idata.posterior
    beta_draws = (
        posterior["beta"].stack(draw=("chain", "draw")).transpose("draw", ...).to_numpy()
    ).astype(np.float64)
    intercept_draws = (
        posterior["intercept"].stack(draw=("chain", "draw")).to_numpy()
    ).astype(np.float64).ravel()

    n_divergent = int(idata.sample_stats["diverging"].to_numpy().sum())
    converged = n_divergent == 0

    beta_draws[:, ~identified] = np.nan
    diagnostics = {
        "n_divergent": n_divergent,
        "n_warmup": int(n_warmup),
        "n_samples": int(n_samples),
        "n_chains": int(n_chains),
        "tau0": float(tau0),
        "seconds": elapsed,
        "beta_scale": (
            "per standard deviation of log10expression"
            if standardize
            else "per unit log10expression"
        ),
        "column_means": col_means,
        "column_scales": col_scales,
        "n_unidentified": int((~identified).sum()),
    }
    return beta_draws, intercept_draws, converged, diagnostics


def fit_model(
    model_data: ModelData,
    *,
    backend: str = "auto",
    seed: int = 0,
    standardize: bool = False,
    expected_hits: Optional[float] = None,
    scale_global: Optional[float] = None,
    df_local: float = 10.0,
    df_global: float = 1.0,
    df_slab: float = 4.0,
    scale_slab: float = 2.0,
    n_draws: int = 1000,
    n_steps: int = 3000,
    n_mc_samples: int = 8,
    learning_rate: float = 0.05,
    convergence_tol: float = 1e-3,
    device: str = "cpu",
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 4,
    max_tree_depth: int = 12,
    target_accept: float = 0.9,
) -> PowerFit:
    """Fit the horseshoe Poisson model and return posterior draws of ``beta``.

    Port of ``compile_model`` + ``fit_model`` from spaCRPower's
    ``R/fit_model.R``, which run ``brms::brm(Npositive ~ 1 +
    log10expression + offset(log(Ntotal)), family = poisson, prior =
    horseshoe(df = 10))``.  ``df_local=10`` here is that ``df = 10``.

    :param model_data: output of :func:`prepare_model_data`.
    :param backend: ``"auto"``, ``"torch"``, ``"numpyro"`` or ``"pymc"``.
        See :func:`resolve_backend`; the choice is recorded on the result.
    :param seed: RNG seed.  The same seed and backend give the same
        estimates.
    :param standardize: scale each covariate column to unit SD before
        fitting, so the horseshoe shrinks every gene on a common scale.
        Changes the units of ``beta`` (recorded in
        ``diagnostics["beta_scale"]``).  Default ``False``, matching the R.
    :param expected_hits: prior guess at the number of true hits, used
        for the horseshoe's global scale.  Default: 5 % of the library.
    :param scale_global: set the horseshoe global scale ``tau0``
        outright, ignoring ``expected_hits``.
    :param df_local: degrees of freedom of the local shrinkage
        half-Student-t.  ``brms``'s ``horseshoe(df=...)``.
    :param df_global: degrees of freedom of the global half-Student-t;
        1 is a half-Cauchy, the standard choice.
    :param df_slab: degrees of freedom of the regularising slab.
    :param scale_slab: scale of the regularising slab; caps how large a
        coefficient the prior will entertain.
    :param n_draws: posterior draws to keep (torch backend; the NUTS
        backends keep ``n_samples * n_chains``).
    :param n_steps: ADVI optimisation steps (torch backend).
    :param n_mc_samples: Monte-Carlo samples per ELBO gradient (torch).
    :param learning_rate: Adam learning rate (torch).
    :param convergence_tol: relative ELBO drift below which the ADVI fit
        is called converged (torch).
    :param device: torch device; ``"cpu"`` by default because the problem
        is small and CPU float64 is bit-reproducible across machines.
    :param n_warmup: NUTS warmup iterations (numpyro/pymc).
    :param n_samples: NUTS post-warmup draws per chain (numpyro/pymc).
    :param n_chains: NUTS chains (numpyro/pymc).
    :param max_tree_depth: NUTS max tree depth (numpyro); the R uses 12.
    :param target_accept: NUTS target acceptance (pymc).
    :returns: :class:`PowerFit`.
    :raises PowerFitError: on an unusable design, an unavailable named
        backend, or a diverged optimisation.

    >>> import numpy as np
    >>> md = ModelData(
    ...     wells=np.array(["w1", "w2"]),
    ...     genes=np.array(["A", "B"]),
    ...     Npositive=np.array([1, 9]),
    ...     Ntotal=np.array([100, 100]),
    ...     log10expression=np.array([[-0.3, -0.3], [-1.0, -0.05]]),
    ... )
    >>> fit = fit_model(md, backend="torch", n_steps=50, n_draws=16, seed=0)
    >>> fit.backend, fit.draws.shape
    ('torch', (16, 2))
    """
    if not isinstance(model_data, ModelData):
        raise PowerFitError(
            "fit_model needs the ModelData returned by prepare_model_data, "
            f"got {type(model_data).__name__}."
        )
    if model_data.n_wells == 0 or model_data.n_genes == 0:
        raise PowerFitError(
            f"nothing to fit: the design is {model_data.n_wells} wells x "
            f"{model_data.n_genes} genes."
        )
    if model_data.n_wells < 2:
        raise PowerFitError(
            "a single well cannot identify any gene effect: with one "
            "observation the intercept explains the data exactly and every "
            "beta is pure prior. Refusing to return a posterior that is "
            "just the prior wearing the name of an estimate."
        )

    resolved = resolve_backend(backend)
    tau0 = _horseshoe_global_scale(
        n_genes=model_data.n_genes,
        n_wells=model_data.n_wells,
        expected_hits=expected_hits,
        scale_global=scale_global,
    )

    if resolved == "torch":
        draws, intercept_draws, converged, diagnostics = _fit_torch_advi(
            model_data,
            seed=seed,
            n_steps=n_steps,
            n_mc_samples=n_mc_samples,
            learning_rate=learning_rate,
            n_draws=n_draws,
            df_local=df_local,
            df_global=df_global,
            df_slab=df_slab,
            scale_slab=scale_slab,
            tau0=tau0,
            device=device,
            standardize=standardize,
            convergence_tol=convergence_tol,
        )
        method = "advi"
    elif resolved == "numpyro":
        draws, intercept_draws, converged, diagnostics = _fit_numpyro_nuts(
            model_data,
            seed=seed,
            n_warmup=n_warmup,
            n_samples=n_samples,
            n_chains=n_chains,
            df_local=df_local,
            df_global=df_global,
            df_slab=df_slab,
            scale_slab=scale_slab,
            tau0=tau0,
            standardize=standardize,
            max_tree_depth=max_tree_depth,
        )
        method = "nuts"
    else:  # pymc -- resolve_backend has already rejected anything else
        draws, intercept_draws, converged, diagnostics = _fit_pymc_nuts(
            model_data,
            seed=seed,
            n_warmup=n_warmup,
            n_samples=n_samples,
            n_chains=n_chains,
            df_local=df_local,
            df_global=df_global,
            df_slab=df_slab,
            scale_slab=scale_slab,
            tau0=tau0,
            standardize=standardize,
            target_accept=target_accept,
        )
        method = "nuts"

    return PowerFit(
        backend=resolved,
        requested_backend=backend,
        method=method,
        draws=draws,
        intercept_draws=intercept_draws,
        genes=np.asarray(model_data.genes),
        converged=converged,
        diagnostics=diagnostics,
        data=model_data,
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# 4. Summaries and evaluation
# ---------------------------------------------------------------------------

def gather_model_estimate(fit: PowerFit) -> pd.DataFrame:
    """Summarise the per-gene coefficients, one row per gene.

    Port of ``gather_model_estimate``, which runs
    ``posterior::summarize_draws()`` and keeps the ``b_log10expression*``
    variables.  The ``variable`` column reproduces that R naming so the
    two implementations' outputs can be joined and compared directly.

    **Higher ``mean`` means stronger evidence the gene is a hit.**  Wells
    carrying more of a hit gene have more positive cells than their
    imaged cell count alone accounts for, which is a positive coefficient
    on log read fraction.

    :param fit: result of :func:`fit_model`.
    :returns: ``pandas.DataFrame`` with columns ``gene``, ``variable``,
        ``mean``, ``sd``, ``q5``, ``q95``, ``prob_positive`` (posterior
        probability that ``beta > 0``) and ``identified``.  Unidentified
        genes carry ``NaN`` in every numeric column and
        ``identified=False``.
    :raises PowerFitError: if ``fit`` is not a :class:`PowerFit`.

    >>> import numpy as np
    >>> fit = PowerFit(
    ...     backend="torch", requested_backend="torch", method="advi",
    ...     draws=np.array([[1.0, 0.0], [3.0, 0.0]]),
    ...     intercept_draws=np.array([-4.0, -4.0]),
    ...     genes=np.array(["A", "B"]), converged=True)
    >>> gather_model_estimate(fit)[["gene", "mean"]].to_dict("records")
    [{'gene': 'A', 'mean': 2.0}, {'gene': 'B', 'mean': 0.0}]
    """
    if not isinstance(fit, PowerFit):
        raise PowerFitError(
            f"gather_model_estimate needs a PowerFit, got {type(fit).__name__}."
        )
    draws = np.asarray(fit.draws, dtype=np.float64)
    if draws.ndim != 2:
        raise PowerFitError(
            f"fit.draws must be (n_draws, n_genes), got shape {draws.shape}."
        )
    if draws.shape[1] != len(fit.genes):
        raise PowerFitError(
            f"fit.draws has {draws.shape[1]} columns but there are "
            f"{len(fit.genes)} genes; the coefficient labelling would be "
            "off by an unknown amount."
        )

    identified = ~np.all(np.isnan(draws), axis=0)
    with warnings.catch_warnings():
        # All-NaN columns are the unidentified genes and are expected; numpy's
        # "Mean of empty slice" is noise here, not news.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(draws, axis=0)
        sd = np.nanstd(draws, axis=0, ddof=1) if draws.shape[0] > 1 else np.zeros(
            draws.shape[1]
        )
        q5 = np.nanquantile(draws, 0.05, axis=0)
        q95 = np.nanquantile(draws, 0.95, axis=0)
        prob_positive = np.nanmean(draws > 0, axis=0)

    mean = np.where(identified, mean, np.nan)
    sd = np.where(identified, sd, np.nan)
    q5 = np.where(identified, q5, np.nan)
    q95 = np.where(identified, q95, np.nan)
    prob_positive = np.where(identified, prob_positive, np.nan)

    genes = np.asarray(fit.genes)
    return pd.DataFrame(
        {
            "gene": genes,
            "variable": [f"b_log10expression{g}" for g in genes],
            "mean": mean,
            "sd": sd,
            "q5": q5,
            "q95": q95,
            "prob_positive": prob_positive,
            "identified": identified,
        }
    )


def evaluate_model_fit(
    data: pd.DataFrame,
    model_estimate: pd.DataFrame,
) -> pd.DataFrame:
    """Score the fit against ground-truth hit status: AUROC and average precision.

    Port of ``evaluate_model_fit``.

    **The sign convention, spelled out, because it is the easiest thing
    in this port to get backwards and the easiest to not notice.**  The R
    computes ``mean_inv = -mean``, recodes ``hit`` as a factor with
    levels ``c("no", "yes")``, and hands both to ``yardstick``.
    ``yardstick``'s default ``event_level`` is ``"first"``, so the event
    it scores is ``"no"`` — the *non*-hits — and the score it scores them
    with is ``-mean``.  Scoring ``-m`` for the event "not a hit" is
    identical to scoring ``+m`` for the event "is a hit": both ROC curves
    are the same curve.  So here we do the equivalent and much more
    legible thing, ``roc_auc_score(y_true=hit, y_score=mean)``, and
    :func:`spacr.power_model.evaluate_model_fit`'s tests pin the
    orientation on a case whose answer is known — strong planted hits
    must score near 1, not near 0.  A flipped convention returns
    ``1 - AUROC``, which for a mediocre design looks entirely plausible.

    Average precision uses scikit-learn's step-wise definition
    (``sum_n (R_n - R_{n-1}) P_n``), the same estimator ``yardstick``
    uses, so the two are comparable.

    :param data: any frame carrying ground truth, with columns ``gene``
        and ``hit`` (0/1 or boolean).  The tidy screen table works
        directly; ``hit`` must be constant within a gene.
    :param model_estimate: output of :func:`gather_model_estimate`, or
        any frame with ``gene`` and ``mean``.
    :returns: one-row ``pandas.DataFrame`` with ``model_ap``,
        ``model_auroc``, ``ap_baseline`` (the prevalence, which is what
        average precision would be at chance), ``n_genes_scored``,
        ``n_hits``, ``n_non_hits``, ``n_unidentified_dropped``,
        ``n_missing_truth`` and ``reason``.  Degenerate cases give
        ``NaN`` metrics and a non-empty ``reason``; they never give 0.5.
    :raises PowerFitError: on missing columns, or a gene whose ``hit``
        status is inconsistent between rows.

    >>> import pandas as pd
    >>> truth = pd.DataFrame({"gene": ["A", "B", "C", "D"],
    ...                       "hit": [1, 0, 0, 0]})
    >>> est = pd.DataFrame({"gene": ["A", "B", "C", "D"],
    ...                     "mean": [2.0, 0.1, 0.0, -0.1]})
    >>> float(evaluate_model_fit(truth, est)["model_auroc"].iloc[0])
    1.0
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    for name, frame, needed in (
        ("data", data, ("gene", "hit")),
        ("model_estimate", model_estimate, ("gene", "mean")),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise PowerFitError(
                f"evaluate_model_fit needs {name} as a DataFrame, got "
                f"{type(frame).__name__}."
            )
        absent = [c for c in needed if c not in frame.columns]
        if absent:
            raise PowerFitError(
                f"evaluate_model_fit: {name} is missing column(s) {absent}. "
                f"Present: {sorted(map(str, frame.columns))}"
            )

    truth = data.loc[:, ["gene", "hit"]].drop_duplicates()
    conflicting = truth["gene"][truth["gene"].duplicated()].unique()
    if len(conflicting):
        raise PowerFitError(
            f"{len(conflicting)} gene(s) carry more than one hit status, e.g. "
            f"{list(conflicting[:3])}. Ground truth has to be one label per "
            "gene; scoring against an ambiguous truth produces a number that "
            "means nothing."
        )

    merged = model_estimate.loc[:, ["gene", "mean"]].merge(
        truth, on="gene", how="left", validate="one_to_one"
    )

    n_estimates = len(merged)
    missing_truth = merged["hit"].isna()
    n_missing_truth = int(missing_truth.sum())
    unidentified = merged["mean"].isna() & ~missing_truth
    n_unidentified = int(unidentified.sum())

    scored = merged.loc[~missing_truth & ~merged["mean"].isna()]
    y_true = pd.to_numeric(scored["hit"], errors="coerce").to_numpy()
    if np.isnan(y_true).any():
        raise PowerFitError(
            "the 'hit' column contains values that are neither 0/1 nor "
            "boolean. Ground truth has to be binary."
        )
    y_true = (y_true > 0).astype(int)
    y_score = scored["mean"].to_numpy(dtype=np.float64)

    n_hits = int(y_true.sum())
    n_non_hits = int(len(y_true) - n_hits)

    reason = ""
    model_ap = float("nan")
    model_auroc = float("nan")
    ap_baseline = float("nan")

    if n_missing_truth:
        logger.warning(
            "evaluate_model_fit: %d of %d estimated genes have no ground-truth "
            "hit status and were dropped from the metrics.",
            n_missing_truth,
            n_estimates,
        )
    if n_unidentified:
        logger.warning(
            "evaluate_model_fit: %d of %d genes had no estimable coefficient "
            "(constant covariate column) and were dropped. They are not "
            "'not hits' -- they are untested.",
            n_unidentified,
            n_estimates,
        )

    if len(y_true) == 0:
        reason = (
            "no gene had both an estimate and a ground-truth label, so there "
            "is nothing to score"
        )
    elif n_hits == 0:
        reason = (
            "no gene is a hit, so AUROC and average precision are undefined "
            "(there is no positive class to rank). This is NOT chance "
            "performance -- reporting 0.5 here would be a fabrication"
        )
    elif n_non_hits == 0:
        reason = (
            "every gene is a hit, so AUROC and average precision are "
            "undefined (there is no negative class to rank against)"
        )
    elif not np.isfinite(y_score).all():
        reason = (
            "the coefficient estimates contain non-finite values, so the "
            "ranking they imply is undefined"
        )
    else:
        # Orientation: y_score = posterior mean of beta, higher = more
        # hit-like. See the docstring for why this is the R's `-mean` scored
        # against event level "no".
        model_auroc = float(roc_auc_score(y_true, y_score))
        model_ap = float(average_precision_score(y_true, y_score))
        ap_baseline = float(n_hits / len(y_true))

    return pd.DataFrame(
        [
            {
                "model_ap": model_ap,
                "model_auroc": model_auroc,
                "ap_baseline": ap_baseline,
                "n_genes_scored": int(len(y_true)),
                "n_hits": n_hits,
                "n_non_hits": n_non_hits,
                "n_unidentified_dropped": n_unidentified,
                "n_missing_truth": n_missing_truth,
                "reason": reason,
            }
        ]
    )


def fit_and_evaluate(
    data: pd.DataFrame,
    *,
    fill_missing: bool = False,
    **fit_kwargs: Any,
) -> Tuple[PowerFit, pd.DataFrame, pd.DataFrame]:
    """Run the whole inference chain on one simulated screen.

    ``prepare_model_data`` -> ``fit_model`` -> ``gather_model_estimate``
    -> ``evaluate_model_fit``, which is the sequence every vignette in
    the R package runs and the sequence :func:`scan_parameters` runs at
    each grid point.

    :param data: tidy screen table (see :func:`prepare_model_data`).
    :param fill_missing: passed to :func:`prepare_model_data`.
    :param fit_kwargs: passed to :func:`fit_model`.
    :returns: ``(fit, model_estimate, model_evaluation)``.
    """
    model_data = prepare_model_data(data, fill_missing=fill_missing)
    fit = fit_model(model_data, **fit_kwargs)
    estimate = gather_model_estimate(fit)
    evaluation = evaluate_model_fit(data, estimate)
    return fit, estimate, evaluation


# ---------------------------------------------------------------------------
# 5. Parameter scan
# ---------------------------------------------------------------------------

_SCAN_RESULT_COLUMNS: Tuple[str, ...] = (
    "param_index",
    "replicate",
    "run_key",
    "backend",
    "method",
    "status",
    "converged",
    "model_ap",
    "model_auroc",
    "ap_baseline",
    "n_genes_scored",
    "n_hits",
    "n_non_hits",
    "n_unidentified_dropped",
    "n_wells_fit",
    "seed_used",
    "seed_channel",
    "elapsed_s",
    "reason",
    "error",
)


def _is_scalar(value: Any) -> bool:
    """Whether a scan parameter is a single value rather than a sweep.

    Strings and bytes are sequences but are obviously single values here;
    treating ``"cellpose"`` as a four-way sweep over characters is the
    classic version of this bug.

    :param value: candidate parameter value.
    :returns: True if it should be held fixed across the grid.
    """
    if isinstance(value, (str, bytes)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return not isinstance(value, (list, tuple, set, frozenset, range, pd.Series))


def _run_key(params: Mapping[str, Any], replicate: int, seed: int, backend: str) -> str:
    """Stable identifier for one grid point, for resuming a killed sweep.

    Keyed on the parameter *values*, not on the row index, so a sweep
    resumed after the grid was extended or reordered still recognises the
    points it already did.  Keying on ``param_index`` alone -- the
    obvious choice -- silently re-labels every completed row the moment
    somebody adds a value to one of the swept lists.

    :param params: the parameter values for this point.
    :param replicate: replicate index within the point.
    :param seed: the sweep's master seed.
    :param backend: resolved backend; a point run under a different
        inference method is a different point.
    :returns: 16-hex-character digest.
    """
    payload = json.dumps(
        {
            "params": {k: _jsonable(v) for k, v in sorted(params.items())},
            "replicate": int(replicate),
            "seed": int(seed),
            "backend": str(backend),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    """Coerce a parameter value into something ``json.dumps`` accepts.

    numpy scalars are the common case (a grid built with
    ``np.linspace``); they are not JSON-serialisable but their Python
    equivalents are, and the digest must not depend on which of the two
    the caller happened to pass.

    :param value: parameter value.
    :returns: a JSON-serialisable equivalent.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in np.asarray(value).ravel().tolist()]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def _default_simulator() -> Callable[..., pd.DataFrame]:
    """Return :func:`spacr.power_simulate.simulate_screen`, lazily.

    Deferred to call time so that importing this module works whether or
    not the simulator half is present, and so that a missing simulator
    produces a message about the simulator rather than an ImportError
    from the middle of ``import spacr``.

    :returns: the simulator callable.
    :raises PowerFitError: if :mod:`spacr.power_simulate` cannot be
        imported.
    """
    try:
        from .power_simulate import simulate_screen
    except ImportError as exc:
        raise PowerFitError(
            "scan_parameters needs a screen simulator. "
            "spacr.power_simulate.simulate_screen could not be imported "
            f"({exc}). Pass simulate_fn= to use your own, or check that the "
            "simulator half of the power-analysis port is installed."
        ) from exc
    return simulate_screen


def _call_simulator(
    simulate_fn: Callable[..., pd.DataFrame],
    params: Mapping[str, Any],
    seed: int,
) -> Tuple[pd.DataFrame, str]:
    """Call the simulator with ``params``, routing the seed however it accepts one.

    The simulator is written by a different module and may name its
    entropy argument ``seed``, ``rng`` or ``random_state``.  We inspect
    the signature rather than guess.  If it accepts none of them and has
    no ``**kwargs``, we fall back to seeding numpy's legacy global RNG,
    and *say so* in the returned channel name -- which lands in the scan
    output, so a run whose reproducibility rests on a global seed is
    identifiable after the fact rather than assumed.

    :param simulate_fn: the simulator.
    :param params: parameters for this grid point.
    :param seed: seed for this grid point.
    :returns: ``(tidy_screen_table, seed_channel)``.
    """
    try:
        signature = inspect.signature(simulate_fn)
        accepted = set(signature.parameters)
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
    except (TypeError, ValueError):  # builtins and C callables have no signature
        accepted, has_var_kw = set(), True

    kwargs = dict(params)
    if "seed" in accepted or has_var_kw:
        kwargs["seed"] = int(seed)
        channel = "seed"
    elif "rng" in accepted:
        kwargs["rng"] = np.random.default_rng(int(seed))
        channel = "rng"
    elif "random_state" in accepted:
        kwargs["random_state"] = int(seed)
        channel = "random_state"
    else:
        np.random.seed(int(seed) % (2 ** 32))
        channel = "numpy-global"
    return simulate_fn(**kwargs), channel


def scan_parameters(
    *,
    progress_file: Optional[str] = None,
    resume: bool = True,
    on_error: str = "record",
    backend: str = "auto",
    seed: int = 0,
    n_replicates: int = 1,
    verbose: bool = False,
    simulate_fn: Optional[Callable[..., pd.DataFrame]] = None,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    **parameters: Any,
) -> pd.DataFrame:
    """Sweep a design grid: simulate, fit, score, at every combination.

    Port of ``scan_parameters`` from spaCRPower's
    ``R/scan_parameters.R``.  Every keyword argument that is not one of
    the named options above is a simulator parameter; pass a scalar to
    hold it fixed and a sequence to sweep it.  The grid is the Cartesian
    product, in the order the sweeps were given (matching R's
    ``tidyr::expand_grid``), with ``param_index`` numbered from 1 as in
    the R.

    **A point that fails is reported as failed.**  A fit that raises
    lands as ``status="failed"`` with the exception text in ``error``; a
    fit that runs but does not meet its convergence criterion lands as
    ``status="not_converged"``.  Both carry ``NaN`` metrics.  This is the
    whole point: a sweep that backfilled 0.5 for its broken points would
    plot as a design sitting at chance, and "this design cannot find its
    hits" and "this fit crashed" are conclusions with opposite
    consequences.

    **Resuming.**  With ``progress_file``, each completed row is appended
    to a TSV immediately, so a killed sweep loses at most the point it
    was on.  Re-running with ``resume=True`` (the default) reads the file
    and skips points whose ``run_key`` is already there.  The key is a
    digest of the parameter *values*, the replicate, the seed and the
    backend -- not the row number -- so extending or reordering the grid
    does not invalidate finished work.

    :param progress_file: TSV to append completed rows to; created, with
        its parent directory, if absent.
    :param resume: skip points already present in ``progress_file``.  With
        ``resume=False`` and a non-empty progress file, this raises rather
        than appending a second copy of every point.
    :param on_error: ``"record"`` (default) logs the failure, writes the
        row and continues; ``"raise"`` re-raises, for debugging one point.
    :param backend: inference backend, see :func:`resolve_backend`.
    :param seed: master seed.  Each point's seed is derived from
        ``(seed, param_index, replicate)``, so a point's data does not
        depend on how many points ran before it.
    :param n_replicates: independent simulated screens per grid point.
        One screen at one setting is a single draw from a noisy process;
        the sweep is much easier to read with three or five.
    :param verbose: log each point's parameters and score as it runs.
    :param simulate_fn: screen simulator; defaults to
        :func:`spacr.power_simulate.simulate_screen`.
    :param fit_kwargs: extra keyword arguments for :func:`fit_model`.
    :param parameters: simulator parameters, scalar or sequence.
    :returns: ``pandas.DataFrame``, one row per (grid point, replicate),
        with every parameter column plus the columns in
        ``_SCAN_RESULT_COLUMNS``.  Rows loaded from a resumed progress
        file are included.
    :raises PowerFitError: if no parameters were given, ``on_error`` is
        not recognised, ``n_replicates < 1``, the progress file exists
        with a different column layout (appending misaligned rows to a
        TSV is a silent data-corruption bug, so we stop instead), or the
        progress file is non-empty and ``resume=False``.

    Example -- sweep the number of wells, three replicates each, resumable.
    Every keyword here is forwarded verbatim to the simulator, so the full
    set is whatever :func:`spacr.power_simulate.simulate_screen` accepts;
    the ones below are the design knobs a power analysis usually turns::

        scores = scan_parameters(
            n_genes_in_library=200,
            gene_hit_rate=0.05,
            n_wells_per_screen=[24, 48, 96],   # the sweep
            class_pos_mu=0.99,
            class_neg_mu=0.01,
            ...,                               # remaining simulator arguments
            n_replicates=3,
            progress_file="scan/wells.tsv",
        )
        solvable = scores.query("status == 'ok' and model_auroc >= 0.9")
    """
    if not parameters:
        raise PowerFitError(
            "scan_parameters was given no simulator parameters, so there is "
            "no grid to scan."
        )
    if on_error not in {"record", "raise"}:
        raise PowerFitError(
            f"on_error must be 'record' or 'raise', got {on_error!r}."
        )
    if int(n_replicates) < 1:
        raise PowerFitError(
            f"n_replicates must be at least 1, got {n_replicates!r}."
        )

    resolved_backend = resolve_backend(backend)
    simulator = simulate_fn if simulate_fn is not None else _default_simulator()
    fit_options = dict(fit_kwargs or {})
    fit_options.pop("backend", None)  # the sweep's backend wins; one method per sweep

    names = list(parameters)
    grids = [
        [parameters[n]] if _is_scalar(parameters[n]) else list(parameters[n])
        for n in names
    ]
    for name, values in zip(names, grids):
        if len(values) == 0:
            raise PowerFitError(
                f"parameter {name!r} was swept over an empty sequence, so the "
                "grid is empty."
            )
    combinations = list(itertools.product(*grids))
    n_points = len(combinations) * int(n_replicates)
    logger.info(
        "power_model.scan_parameters: %d grid point(s) x %d replicate(s) = "
        "%d fits, backend=%r",
        len(combinations),
        int(n_replicates),
        n_points,
        resolved_backend,
    )

    result_columns = list(names) + list(_SCAN_RESULT_COLUMNS)

    # ---- progress file ---------------------------------------------------
    done: Dict[str, Dict[str, Any]] = {}
    path: Optional[Path] = None
    if progress_file is not None:
        path = Path(progress_file)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("power_model: created progress directory %s", path.parent)
        if path.exists() and path.stat().st_size > 0:
            existing = pd.read_csv(path, sep="\t")
            if list(existing.columns) != result_columns:
                raise PowerFitError(
                    f"progress file {path} has columns {list(existing.columns)} "
                    f"but this sweep produces {result_columns}. Appending rows "
                    "to a TSV whose header says something else writes values "
                    "under the wrong names and the file still parses, which is "
                    "the worst possible outcome. Point progress_file at a new "
                    "path, or delete the old one."
                )
            if not resume:
                raise PowerFitError(
                    f"progress file {path} already holds {len(existing)} row(s) "
                    "and resume=False. Appending would put two rows with the "
                    "same run_key in one file, and every later read of it -- a "
                    "groupby, a mean, a plot -- would double-count the points "
                    "that were run twice. Delete the file, or point "
                    "progress_file somewhere new."
                )
            # Round-tripping through TSV turns empty strings into NaN, which
            # would then compare unequal to the "" that a fresh row carries and
            # would print as 'nan' in the error column of a perfectly fine run.
            for column in ("run_key", "backend", "method", "status",
                           "seed_channel", "reason", "error"):
                if column in existing.columns:
                    existing[column] = existing[column].fillna("").astype(str)
            if "run_key" in existing.columns:
                for record in existing.to_dict("records"):
                    done[str(record["run_key"])] = record
                logger.info(
                    "power_model: resuming from %s -- %d completed row(s) "
                    "will be skipped.",
                    path,
                    len(done),
                )

    rows: List[Dict[str, Any]] = []

    for point_index, combination in enumerate(combinations, start=1):
        point_params = dict(zip(names, combination))
        for replicate in range(int(n_replicates)):
            key = _run_key(point_params, replicate, seed, resolved_backend)
            if key in done:
                rows.append(done[key])
                continue

            # Seed derived from the point's identity, not from iteration order,
            # so re-running a single point reproduces the screen it produced
            # inside the full sweep.
            point_seed = int(
                np.random.SeedSequence(
                    [int(seed), int(point_index), int(replicate)]
                ).generate_state(1)[0]
            )

            if verbose:
                logger.info(
                    "power_model: point %d/%d replicate %d: %s",
                    point_index,
                    len(combinations),
                    replicate,
                    ", ".join(f"{k}={v!r}" for k, v in point_params.items()),
                )

            row: Dict[str, Any] = dict(point_params)
            row.update(
                {
                    "param_index": point_index,
                    "replicate": replicate,
                    "run_key": key,
                    "backend": resolved_backend,
                    "method": "",
                    "status": "failed",
                    "converged": False,
                    "model_ap": float("nan"),
                    "model_auroc": float("nan"),
                    "ap_baseline": float("nan"),
                    "n_genes_scored": 0,
                    "n_hits": 0,
                    "n_non_hits": 0,
                    "n_unidentified_dropped": 0,
                    "n_wells_fit": 0,
                    "seed_used": point_seed,
                    "seed_channel": "",
                    "elapsed_s": float("nan"),
                    "reason": "",
                    "error": "",
                }
            )

            started = time.perf_counter()
            try:
                screen, channel = _call_simulator(simulator, point_params, point_seed)
                row["seed_channel"] = channel
                fit, _estimate, evaluation = fit_and_evaluate(
                    screen, backend=resolved_backend, seed=point_seed, **fit_options
                )
                metrics = evaluation.iloc[0]
                row.update(
                    {
                        "method": fit.method,
                        "converged": bool(fit.converged),
                        "ap_baseline": float(metrics["ap_baseline"]),
                        "n_genes_scored": int(metrics["n_genes_scored"]),
                        "n_hits": int(metrics["n_hits"]),
                        "n_non_hits": int(metrics["n_non_hits"]),
                        "n_unidentified_dropped": int(
                            metrics["n_unidentified_dropped"]
                        ),
                        "n_wells_fit": int(
                            fit.data.n_wells if fit.data is not None else 0
                        ),
                        "reason": str(metrics["reason"]),
                    }
                )
                if not fit.converged:
                    # Metrics stay NaN. A non-converged fit's coefficient
                    # ordering is not a posterior ordering, and scoring it
                    # would put a number on the plot that the fit does not
                    # support.
                    row["status"] = "not_converged"
                    row["error"] = (
                        "the fit did not meet its convergence criterion; "
                        "metrics withheld"
                    )
                    logger.warning(
                        "power_model: point %d replicate %d did not converge; "
                        "recorded as not_converged with NaN metrics.",
                        point_index,
                        replicate,
                    )
                else:
                    row["status"] = "ok"
                    row["model_ap"] = float(metrics["model_ap"])
                    row["model_auroc"] = float(metrics["model_auroc"])
            except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised if asked
                if on_error == "raise":
                    raise
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "power_model: point %d replicate %d FAILED (%s: %s). "
                    "Recorded with NaN metrics -- it is not a chance-level "
                    "result.",
                    point_index,
                    replicate,
                    type(exc).__name__,
                    exc,
                )
            finally:
                row["elapsed_s"] = time.perf_counter() - started

            if verbose:
                logger.info(
                    "power_model: point %d replicate %d -> status=%s "
                    "model_ap=%s model_auroc=%s",
                    point_index,
                    replicate,
                    row["status"],
                    row["model_ap"],
                    row["model_auroc"],
                )

            rows.append(row)
            if path is not None:
                # Append one complete row at a time and close the handle, so a
                # kill -9 between points leaves a file whose last line is whole.
                pd.DataFrame([row], columns=result_columns).to_csv(
                    path,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not (path.exists() and path.stat().st_size > 0),
                )

    frame = pd.DataFrame(rows, columns=result_columns)
    n_failed = int((frame["status"] == "failed").sum())
    n_unconverged = int((frame["status"] == "not_converged").sum())
    if n_failed or n_unconverged:
        warnings.warn(
            f"scan_parameters: {n_failed} of {len(frame)} point(s) failed and "
            f"{n_unconverged} did not converge. Their model_ap/model_auroc are "
            "NaN, not 0.5 -- filter on status before plotting or the gaps will "
            "read as chance-level performance.",
            RuntimeWarning,
            stacklevel=2,
        )
    return frame
