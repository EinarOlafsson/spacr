"""Sweep the DEPENDENT variable: which measurement has genes with an effect.

Instruction 122 part 3, in the maintainer's words:

    "doing a sweep on these screen data of which measurements have genes with
     an effect size -- so instead of a parameter search, a search for which
     measurement has genes with clear effect sizes (one or several)"

A parameter sweep holds the data fixed and varies the settings. This holds the
model fixed and varies the RESPONSE: one linear model per measurement, over
the same wells and the same guides, and a table saying which measurements the
guides moved.

WHY IT IS NOT JUST A LOOP
-------------------------
Because of what the loop does to the statistics.

**1. A measurement scan is a multiple-testing problem ACROSS measurements.**
spaCR measures hundreds of features per object. Scan 500 and some will look
clear by chance -- and they will look *exactly* as convincing as the real
ones, because the FDR each run reports was computed WITHIN that measurement
and knows nothing about the other 499. So every row here carries **two**
numbers:

* ``within_run_q`` -- the correction the single-measurement analysis would
  have reported, across the genes of that one measurement;
* ``across_scan_q`` -- the correction across the scan, which is the one that
  answers the question the user actually asked.

A measurement that survives the first and fails the second is the single most
important thing this module can say, and the easiest for it to hide. Both are
in :meth:`ScanResult.frame`, on the same row, always.

The across-scan stage is fed one P value per measurement, and that P value is
**Simes' global-null P** for the measurement -- the minimum Benjamini-Hochberg
adjusted P over its genes. That is the right summary rather than the raw
minimum: the raw minimum of 23 gene tests is not a P value, and using it would
smuggle the within-measurement multiplicity past the second stage.

**2. The measurements are heavily CORRELATED.** Area, perimeter and
equivalent-diameter are one thing measured three ways. So the effective number
of independent tests is far below the column count, and a naive Bonferroni
over 500 correlated columns is too harsh in the other direction.

The default across-scan correction is therefore **Benjamini-Hochberg**, which
is valid under the positive dependence measurement columns actually have and
is not made absurd by it, and :attr:`ScanResult.effective_n_tests` reports the
Li & Ji (2005) estimate of how many independent tests the scanned columns
amount to. ``across_scan_method='bonferroni_effective'`` uses that estimate as
the divisor for callers who want family-wise control without the naive
column-count penalty.

**3. RANK BY EFFECT SIZE, NOT BY P VALUE.** With two screens' worth of wells a
trivial effect is significant. ``effect_size`` is the gene's coefficient in
units of the model's **residual** standard deviation -- a Cohen's d. Residual
and not total, because a blocking factor's whole job is to remove a nuisance
source of spread, and dividing by the total would make blocking on it look
pointless.

Every correction comes from :mod:`spacr.multiple_testing`. There is no second
implementation here, so the GUI dropdown, the settings validator and this scan
cannot mean different things by one word.

Dependencies: numpy, scipy.stats and pandas. Deliberately no ``spacr.ml``
import -- this is the pure logic, it must be importable without torch, and the
GUI is not its business.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import schema
from .multiple_testing import adjust_p_values, canonical_method

__all__ = [
    "ScanRefused",
    "MeasurementEffect",
    "ScanResult",
    "scan_measurements",
    "effective_number_of_tests",
    "simes_p_value",
    "DEFAULT_BLOCK_COLUMNS",
]

#: What the scan blocks on unless told otherwise.
#:
#: The screen, because that is what instruction 122's first two parts were
#: for: two screens stacked into one frame differ from each other for reasons
#: that have nothing to do with the guides, and leaving that in the residual
#: inflates every standard error in the scan. A block that is absent, or that
#: holds one value, is simply dropped -- a single-screen project is not a
#: special case, it is the same model with one fewer term.
DEFAULT_BLOCK_COLUMNS: Tuple[str, ...] = (schema.SCREEN_KEY,)

#: Wells a gene needs before its effect is estimable at all.
#:
#: A GENE IN ONE WELL HAS NOTHING CORROBORATING IT. Its "effect" is that
#: single well's deviation from the rest, which carries every well-level
#: artefact there is -- edge position, seeding density, focus, a bubble --
#: and no way to tell any of them from a phenotype. spaCR already refuses
#: this one level down: :data:`spacr.hits.FLAG_SINGLE_GUIDE` is "called by
#: one guide, so nothing corroborates it". This is the same rule at the well.
#:
#: It is not a cosmetic filter. MEASURED on plate1 of the tsg101 screen,
#: 44 wells over 10 genes of which 8 had a single well, gene labels permuted
#: so no effect can exist:
#:
#:     singletons kept      65% of permuted scans produced an "across-scan
#:                          survivor" -- against the 5% the correction promises
#:     singletons dropped    0%
#:
#: One outlier well does not produce one false hit. The measurement columns
#: are strongly correlated, so it produces DOZENS of correlated false hits at
#: once, which is precisely the structure a correction assuming valid P values
#: cannot rescue. On that screen's true labels, 64 of the 78 surviving
#: measurements -- 82% -- rested on a gene with one well.
MIN_WELLS_PER_GENE = 2

#: Columns that are the DESIGN, not the response.
#:
#: Regressing the guides on the row index is not an analysis, and neither is
#: regressing them on the well's own guide count. Non-numeric columns are
#: excluded automatically; this closed list is for the numeric ones that would
#: otherwise sail through.
_DESIGN_COLUMNS: Tuple[str, ...] = (
    'count', 'counts', 'total_count', 'well_count', 'object_count',
    'cell_count', 'n', 'n_wells', 'n_cells', 'size',
    'fraction', 'gene_fraction', 'grna_fraction',
    'index', 'level_0',
)

_IDENTITY_COLUMNS: Tuple[str, ...] = (
    schema.SCREEN_KEY, schema.PLATE_KEY, schema.ROW_KEY, schema.COLUMN_KEY,
    schema.FIELD_KEY, schema.TIME_KEY, schema.CHANNEL_KEY, schema.SLICE_KEY,
    schema.OBJECT_LABEL_KEY, schema.OBJECT_TYPE_KEY,
    schema.PRC_KEY, schema.PRCF_KEY, schema.PRCFO_KEY,
    'source_database', 'prcfo', 'objectID', 'well', 'condition',
)


class ScanRefused(ValueError):
    """The scan would not have meant what its output claimed."""


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MeasurementEffect:
    """One scanned measurement: its best gene, and both corrections."""

    measurement: str
    #: Wells the model was fitted on. Not constant across a merged frame --
    #: databases from different spaCR versions have different columns filled.
    n_wells: int
    #: Gene terms in the design (the baseline / control level is not one).
    n_genes: int
    #: The gene with the largest absolute standardised effect.
    top_gene: str
    #: That gene's coefficient in units of the residual standard deviation.
    #: **This is the primary sort**, and it is signed-magnitude: the table is
    #: ranked on ``abs``, and ``coefficient`` keeps the direction.
    effect_size: float
    #: The same effect in the measurement's own units.
    coefficient: float
    #: The top gene's raw two-sided P value. Reported, never ranked on.
    p_value: float
    #: The measurement's best within-run adjusted P, across its own genes.
    within_run_q: float
    #: How many genes the within-run correction called at ``alpha``.
    within_run_hits: int
    #: Simes' global-null P for this measurement -- what the across-scan
    #: stage was actually given.
    measurement_p: float
    #: The across-scan adjusted P. Never smaller than ``within_run_q`` under
    #: the default methods, which is the whole point of computing it.
    across_scan_q: float
    survives_within_run: bool
    survives_across_scan: bool


@dataclass(frozen=True)
class ScanResult:
    """Every measurement the scan looked at, and how it was corrected."""

    rows: Tuple[MeasurementEffect, ...]
    #: ``{measurement: why it could not be scanned}``. Named rather than
    #: dropped: a measurement missing from the table without explanation reads
    #: as a measurement with no effect.
    skipped: Mapping[str, str]
    block_columns: Tuple[str, ...]
    gene_column: str
    control_genes: Tuple[str, ...]
    within_run_method: str
    across_scan_method: str
    alpha: float
    #: Li & Ji (2005) estimate of how many independent tests the scanned
    #: columns amount to. ``nan`` when it could not be estimated.
    effective_n_tests: float
    #: ``{gene: wells it had}`` for genes dropped as having too few wells to
    #: corroborate anything. Named rather than silently filtered: a gene
    #: missing from the result with no explanation reads as a gene with no
    #: effect, which is the opposite of what happened to it.
    genes_dropped: Mapping[str, int] = field(default_factory=dict)

    @property
    def n_measurements_scanned(self) -> int:
        return len(self.rows)

    def surviving(self) -> Tuple[MeasurementEffect, ...]:
        """The measurements that survive the ACROSS-SCAN correction.

        Not the within-run one. A caller that shows the within-run survivors
        as "the hits" has rebuilt the exact trap this module exists to close.
        """
        return tuple(row for row in self.rows if row.survives_across_scan)

    def frame(self):
        """The result table, ranked by effect size, largest first."""
        import pandas as pd

        table = pd.DataFrame([vars(row) for row in self.rows])
        if table.empty:
            return pd.DataFrame(columns=[
                'measurement', 'n_wells', 'n_genes', 'top_gene', 'effect_size',
                'coefficient', 'p_value', 'within_run_q', 'within_run_hits',
                'measurement_p', 'across_scan_q', 'survives_within_run',
                'survives_across_scan'])
        order = table['effect_size'].abs().sort_values(
            ascending=False, kind='stable').index
        return table.loc[order].reset_index(drop=True)

    def describe(self) -> str:
        """What a user has to be told before they read the table."""
        genes = self.rows[0].n_genes if self.rows else 0
        lines = [f"{self.n_measurements_scanned} measurement(s) scanned "
                 f"against {genes} gene term(s)"]
        if self.block_columns:
            lines.append("  blocked on " + ", ".join(self.block_columns))
        lines.append(
            f"  within-run correction across genes: {self.within_run_method}")
        lines.append(
            f"  across-scan correction across measurements: "
            f"{self.across_scan_method}")
        if math.isfinite(self.effective_n_tests):
            lines.append(
                f"  effective number of independent tests: "
                f"{self.effective_n_tests:.1f} of "
                f"{self.n_measurements_scanned} columns (measurements are "
                f"correlated, so the column count overstates the family)")
        survivors = self.surviving()
        lines.append(
            f"  {len(survivors)} survive the across-scan correction at "
            f"alpha={self.alpha:g}; "
            f"{sum(1 for row in self.rows if row.survives_within_run)} would "
            f"have been reported by a per-measurement run")
        if self.skipped:
            lines.append(
                f"  {len(self.skipped)} not scanned: "
                + ", ".join(f"{name} ({why})"
                            for name, why in sorted(self.skipped.items())))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  The two statistics that are not in multiple_testing
# --------------------------------------------------------------------------- #

def simes_p_value(p_values) -> float:
    """Simes' global-null P value for one family of tests.

    ``min_i (m * p_(i) / i)`` over the sorted P values -- the probability of
    seeing a family this extreme when none of its members is real. Identical
    to the minimum Benjamini-Hochberg adjusted P, and valid under the positive
    dependence that gene terms in one design actually have.

    This is what one measurement contributes to the across-scan stage. The raw
    minimum of 23 gene tests is not a P value and would smuggle the
    within-measurement multiplicity past the second correction.
    """
    values = np.asarray(p_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    ordered = np.sort(values)
    ranks = np.arange(1, ordered.size + 1, dtype=float)
    # No cap is needed: the last term is m * p_(m) / m == p_(m) <= 1, so the
    # minimum is never above 1 and a clip here would be code no input reaches.
    return float(np.min(ordered * ordered.size / ranks))


def effective_number_of_tests(matrix) -> float:
    """Li & Ji (2005) effective number of independent tests.

    ``M_eff = sum_i f(|lambda_i|)`` over the eigenvalues of the correlation
    matrix, with ``f(x) = I(x >= 1) + (x - floor(x))``: an eigenvalue of 3
    counts as one independent test, an eigenvalue of 0 counts as none.

    Why it is reported. Twelve columns that are four things measured three
    ways are not twelve tests, and correcting as though they were is the
    mirror image of not correcting at all -- it hides real effects instead of
    inventing them. This number is what lets a reader see which situation they
    are in, and it is the divisor ``'bonferroni_effective'`` uses.

    :param matrix: ``(n_wells, n_measurements)`` array. Columns with no
        variance are dropped; NaNs are handled pairwise-complete.
    :returns: the estimate, or ``nan`` when it cannot be computed.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        return float('nan')
    if data.shape[1] == 1:
        return 1.0
    import pandas as pd

    correlation = pd.DataFrame(data).corr().to_numpy()
    # Drop a column on its OWN diagonal, not on whether its row is clean. A
    # column with no variance has an undefined correlation with everything,
    # and testing the row would throw away every column it touches -- which
    # is every column -- for one constant neighbour.
    keep = np.isfinite(np.diag(correlation))
    correlation = correlation[np.ix_(keep, keep)]
    if correlation.size == 0:
        return float('nan')
    if correlation.shape[0] == 1:
        return 1.0
    # Two columns can still have no wells in common (pandas correlates
    # pairwise-complete), leaving a NaN between two perfectly good columns.
    # Read that as uncorrelated: it counts them as separate tests, which is
    # the conservative direction for a correction.
    correlation = np.nan_to_num(correlation, nan=0.0)
    eigenvalues = np.abs(np.linalg.eigvalsh(correlation))
    total = float(sum((1.0 if value >= 1.0 else 0.0)
                      + (value - math.floor(value))
                      for value in eigenvalues))
    return max(1.0, min(total, float(correlation.shape[0])))


# --------------------------------------------------------------------------- #
#  The design
# --------------------------------------------------------------------------- #

def _measurement_columns(frame, gene_column: str,
                         guide_column: Optional[str]) -> Tuple[str, ...]:
    """Numeric columns that are a response rather than part of the design."""
    excluded = {gene_column, guide_column}
    excluded.update(_IDENTITY_COLUMNS)
    excluded.update(_DESIGN_COLUMNS)
    out = []
    for name in frame.columns:
        if name in excluded or str(name).lower() in excluded:
            continue
        if not _is_numeric(frame[name]):
            continue
        out.append(name)
    return tuple(out)


def _is_numeric(series) -> bool:
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_bool_dtype(series))


def _dummy_block(values) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Treatment-coded dummies for one categorical column, first level out.

    A column with one level yields **no** columns rather than a constant one.
    That is not a special case to guard: a block that never varies carries no
    information, and including it would make the design rank-deficient and the
    standard errors meaningless.
    """
    levels = sorted({str(value) for value in values})
    if len(levels) < 2:
        return np.zeros((len(values), 0)), ()
    text = np.asarray([str(value) for value in values])
    columns = [(text == level).astype(float) for level in levels[1:]]
    return np.column_stack(columns), tuple(levels[1:])


def _build_design(frame, gene_column: str, block_columns: Sequence[str],
                  control_genes: Sequence[str]):
    """Return ``(X, gene_term_names, gene_term_slice, used_blocks)``.

    The baseline is the **controls** when the caller names any, because a
    screen's effect size is "against the controls" and not "against whichever
    gene sorted first" -- and a scan that does not know its controls reports
    the control gene itself as the hit whenever the rest of the library moves
    together.
    """
    n = len(frame)
    parts = [np.ones((n, 1))]
    used_blocks = []
    for column in block_columns:
        if column not in frame.columns:
            continue
        block, levels = _dummy_block(frame[column].to_numpy())
        if not levels:
            continue
        parts.append(block)
        used_blocks.append(column)

    genes = np.asarray([str(value) for value in frame[gene_column]])
    controls = {str(value) for value in control_genes}
    levels = sorted(set(genes))
    if len(levels) < 2:
        raise ScanRefused(
            f"{gene_column!r} holds {len(levels)} distinct value(s). With one "
            f"gene there is nothing to contrast it against -- the gene term "
            f"IS the intercept, and every effect size would be zero.")
    terms = [level for level in levels if level not in controls]
    if controls and not terms:
        raise ScanRefused(
            f"every value of {gene_column!r} is a control, so there is no "
            f"gene left to measure an effect for.")
    if len(terms) == len(levels):
        # No controls named, or none of the named ones are in this frame.
        # The first level becomes the baseline -- an arbitrary choice, which
        # is exactly why naming the controls matters.
        terms = levels[1:]
    gene_block = np.column_stack([(genes == term).astype(float)
                                  for term in terms])
    start = sum(part.shape[1] for part in parts)
    parts.append(gene_block)
    design = np.column_stack(parts)
    return (design, tuple(terms), slice(start, start + len(terms)),
            tuple(used_blocks))


def _fit_columns(design: np.ndarray, responses: np.ndarray,
                 gene_slice: slice):
    """OLS of every column of ``responses`` on one shared ``design``.

    One decomposition for the whole block. The design does not change between
    measurements -- only the response does -- so fitting them one at a time
    through statsmodels would repeat the same factorisation hundreds of times
    for an identical answer.

    :returns: ``(betas, standard_errors, residual_sd, df)``, gene terms only.
    """
    n, p = design.shape
    rank = int(np.linalg.matrix_rank(design))
    df = n - rank
    if df <= 0:
        return None
    gram_inverse = np.linalg.pinv(design.T @ design)
    coefficients = gram_inverse @ (design.T @ responses)
    residuals = responses - design @ coefficients
    sigma2 = (residuals ** 2).sum(axis=0) / df
    diagonal = np.clip(np.diag(gram_inverse), 0.0, None)
    standard_errors = np.sqrt(np.outer(diagonal, sigma2))
    return (coefficients[gene_slice], standard_errors[gene_slice],
            np.sqrt(sigma2), df)


# --------------------------------------------------------------------------- #
#  The scan
# --------------------------------------------------------------------------- #

def scan_measurements(frame,
                      *,
                      measurements: Optional[Iterable[str]] = None,
                      gene_column: str = 'gene',
                      guide_column: Optional[str] = 'grna',
                      block_columns: Sequence[str] = DEFAULT_BLOCK_COLUMNS,
                      control_genes: Sequence[str] = (),
                      within_run_method: str = 'fdr_bh',
                      across_scan_method: str = 'fdr_bh',
                      min_wells_per_gene: int = MIN_WELLS_PER_GENE,
                      alpha: float = 0.05) -> ScanResult:
    """Fit every measurement against the guides and rank them by effect size.

    :param frame: one row per **well**, carrying the gene assignment, any
        blocking factors, and the measurement columns to scan.
    :param measurements: the columns to scan. Default: every numeric column
        that is not identity or design (see :data:`_DESIGN_COLUMNS`).
    :param gene_column: the independent variable.
    :param guide_column: excluded from the scanned columns when present.
    :param block_columns: fixed effects fitted alongside the genes and not
        reported. Default :data:`DEFAULT_BLOCK_COLUMNS` -- the screen. Absent
        or single-valued blocks are dropped.
    :param control_genes: the baseline the effects are measured from.
    :param within_run_method: correction across the genes of ONE measurement,
        i.e. what a single-measurement analysis would have reported.
    :param across_scan_method: correction across the measurements. Any
        :mod:`spacr.multiple_testing` method, plus
        ``'bonferroni_effective'`` -- Bonferroni divided by
        :func:`effective_number_of_tests` rather than by the column count.
    :param min_wells_per_gene: genes with fewer wells than this are dropped
        before fitting, and named in :attr:`ScanResult.genes_dropped`. See
        :data:`MIN_WELLS_PER_GENE` -- a gene in one well has nothing
        corroborating it, and on real data keeping them made the across-scan
        correction fire on 65% of scans over permuted labels instead of 5%.
        Pass ``1`` to keep them, knowing that.
    :param alpha: the level both corrections target.
    :returns: the :class:`ScanResult`.
    :raises ScanRefused: when there is no gene column, fewer than two genes,
        or nothing numeric to scan.
    """
    import pandas as pd

    if gene_column not in frame.columns:
        raise ScanRefused(
            f"no {gene_column!r} column: a measurement scan with no "
            f"independent variable is not an empty result, it is a caller "
            f"that has not passed the screen in.")
    within_run_method = canonical_method(within_run_method)
    effective_bonferroni = (
        str(across_scan_method).strip().lower() == 'bonferroni_effective')
    if not effective_bonferroni:
        across_scan_method = canonical_method(across_scan_method)
    else:
        across_scan_method = 'bonferroni_effective'

    usable = frame[frame[gene_column].notna()].reset_index(drop=True)

    # DROP THE GENES NOTHING CORROBORATES, AND SAY WHICH.
    #
    # Named rather than quietly filtered: a gene missing from the result with
    # no explanation reads as a gene with no effect, which is the opposite of
    # what happened to it.
    genes_dropped: Dict[str, int] = {}
    if min_wells_per_gene > 1 and len(usable):
        per_gene = usable[gene_column].astype(str).value_counts()
        thin = per_gene[per_gene < int(min_wells_per_gene)]
        controls = {str(gene) for gene in control_genes}
        # A control is the BASELINE, not a candidate. Dropping it for being
        # thin would silently move the baseline to whichever gene sorted
        # first, and every effect in the table would then be measured from
        # somewhere the caller did not choose.
        thin = thin[[gene for gene in thin.index if gene not in controls]]
        if len(thin):
            genes_dropped = {str(g): int(n) for g, n in thin.items()}
            usable = usable[~usable[gene_column].astype(str)
                            .isin(genes_dropped)].reset_index(drop=True)
            kept = usable[gene_column].astype(str).nunique()
            if kept < 2:
                raise ScanRefused(
                    f"only {kept} gene(s) have at least {min_wells_per_gene} "
                    f"wells, so there is nothing to compare. Dropped: "
                    f"{sorted(genes_dropped)}. A gene in one well has nothing "
                    f"corroborating it -- its effect is that well's own "
                    f"deviation, artefacts included. Lower "
                    f"min_wells_per_gene to keep them, knowing that.")

    if measurements is None:
        candidates = _measurement_columns(usable, gene_column, guide_column)
    else:
        candidates = tuple(measurements)
        missing = [name for name in candidates if name not in usable.columns]
        if missing:
            raise ScanRefused(f"no such measurement column(s): {missing}")
    if not candidates:
        raise ScanRefused(
            "no numeric measurement columns to scan. Identity and design "
            "columns are excluded on purpose -- regressing the guides on the "
            "row index is not an analysis.")

    design, gene_terms, gene_slice, used_blocks = _build_design(
        usable, gene_column, block_columns, control_genes)

    skipped: Dict[str, str] = {}
    kept: list = []
    for name in candidates:
        column = pd.to_numeric(usable[name], errors='coerce').to_numpy(float)
        present = np.isfinite(column)
        # Only the "there is nothing here at all" case is decided here.
        # Whether the wells that ARE present can carry the design is
        # _fit_columns' question, because it is the one that knows the rank
        # of the sub-design after the empty gene levels have dropped out.
        if present.sum() < 3:
            skipped[name] = 'too few wells with a value'
            continue
        if float(np.nanstd(column[present])) <= 0.0:
            skipped[name] = 'no variance'
            continue
        kept.append((name, column, present))

    if not kept:
        return ScanResult((), skipped, used_blocks, gene_column,
                          tuple(control_genes), within_run_method,
                          across_scan_method, float(alpha), float('nan'),
                          genes_dropped)

    # Group by which wells are present, so the common case -- every
    # measurement complete -- costs one matrix factorisation, not hundreds.
    groups: Dict[bytes, list] = {}
    for name, column, present in kept:
        groups.setdefault(present.tobytes(), []).append((name, column, present))

    records: list = []
    for members in groups.values():
        present = members[0][2]
        sub_design = design[present]
        responses = np.column_stack([column[present] for _n, column, _p in members])
        fitted = _fit_columns(sub_design, responses, gene_slice)
        if fitted is None:
            for name, _column, _present in members:
                skipped[name] = 'not enough wells left for the design'
            continue
        betas, errors, residual_sd, df = fitted
        from scipy import stats

        for index, (name, column, _present) in enumerate(members):
            beta = betas[:, index]
            error = errors[:, index]
            # A residual of essentially nothing is not a huge effect, it is a
            # column the design already IS -- a per-well aggregate that turns
            # out to be the guide assignment, say. Dividing by it gives an
            # effect size of 1e8 that sits at the top of the table for ever
            # and is not a measurement of anything. Judged RELATIVE to the
            # response's own spread, because "small" has no absolute meaning
            # for a measurement whose units the scan does not know.
            scale = float(np.nanstd(column[present]))
            estimable = residual_sd[index] > 1e-8 * max(scale, 1e-300)
            with np.errstate(divide='ignore', invalid='ignore'):
                t_values = np.where(error > 0, beta / error, np.nan)
                effects = np.where(estimable, beta / residual_sd[index],
                                   np.nan)
            p_values = 2.0 * stats.t.sf(np.abs(t_values), df)
            p_values = np.where(np.isfinite(t_values), p_values, np.nan)
            if not np.isfinite(effects).any():
                skipped[name] = 'no estimable gene effect'
                continue
            top = int(np.nanargmax(np.abs(effects)))
            adjusted, rejected = adjust_p_values(
                p_values, method=within_run_method, alpha=alpha)
            finite = np.isfinite(adjusted)
            records.append({
                'measurement': name,
                'n_wells': int(present.sum()),
                'n_genes': len(gene_terms),
                'top_gene': gene_terms[top],
                'effect_size': float(effects[top]),
                'coefficient': float(beta[top]),
                'p_value': float(p_values[top]),
                'within_run_q': (float(np.min(adjusted[finite]))
                                 if finite.any() else float('nan')),
                'within_run_hits': int(rejected.sum()),
                'measurement_p': simes_p_value(p_values),
            })

    if not records:
        return ScanResult((), skipped, used_blocks, gene_column,
                          tuple(control_genes), within_run_method,
                          across_scan_method, float(alpha), float('nan'),
                          genes_dropped)

    scanned = [record['measurement'] for record in records]
    matrix = np.column_stack([
        pd.to_numeric(usable[name], errors='coerce').to_numpy(float)
        for name in scanned])
    m_effective = effective_number_of_tests(matrix)

    family = np.asarray([record['measurement_p'] for record in records],
                        dtype=float)
    if effective_bonferroni:
        divisor = m_effective if math.isfinite(m_effective) else len(family)
        across = np.clip(family * divisor, 0.0, 1.0)
        across_rejected = across < alpha
    else:
        across, across_rejected = adjust_p_values(
            family, method=across_scan_method, alpha=alpha)

    rows = tuple(
        MeasurementEffect(
            measurement=record['measurement'],
            n_wells=record['n_wells'],
            n_genes=record['n_genes'],
            top_gene=record['top_gene'],
            effect_size=record['effect_size'],
            coefficient=record['coefficient'],
            p_value=record['p_value'],
            within_run_q=record['within_run_q'],
            within_run_hits=record['within_run_hits'],
            measurement_p=record['measurement_p'],
            across_scan_q=float(across[index]),
            survives_within_run=bool(record['within_run_hits'] > 0),
            survives_across_scan=bool(across_rejected[index]),
        )
        for index, record in enumerate(records)
    )
    return ScanResult(rows, skipped, used_blocks, gene_column,
                      tuple(control_genes), within_run_method,
                      across_scan_method, float(alpha), float(m_effective),
                      genes_dropped)
