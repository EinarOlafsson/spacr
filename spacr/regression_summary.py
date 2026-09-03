"""spaCR's own summary of a regression run — one shape for every mode.

WHY THIS EXISTS. ``regression_results.summary_text`` relays the *statsmodels*
summary, verbatim and deliberately: the point of asking for the statsmodels
summary is to get the statsmodels summary, and a re-implementation would
differ from every textbook a reader compares it against. That is correct, and
it means the tab has exactly one source — so most of
:data:`spacr.regression_spec.REGRESSION_TYPES` got nothing because statsmodels
does not write a summary for them, and ``inference='nonparametric'`` got

    No summary: this run came back without a fitted model, so there is none
    to summarise.

which is TRUE AND USELESS. The permutation path is a within-plate marginal
test per guide: no design matrix, no coefficient covariance, no statsmodels
object to ask. The run still produced results, and those results have
properties worth reporting — just not the ones statsmodels prints.

THE CONTRACT IS THE DELIVERABLE, not the statistics. Every field named in
:data:`CONTRACT` is present in every mode, and each one is either COMPUTED or
explicitly NOT APPLICABLE WITH A REASON. A blank is not allowed, and neither
is a zero standing in for "we did not check" — that is the failure this module
exists to prevent, in miniature. :class:`SummaryField` refuses to be built any
other way, so the rule is enforced by the type rather than by review.

A NONPARAMETRIC RUN LISTS ITS ASSUMPTIONS AS "NOT ASSUMED", NEVER AS BLANKS.
That is the POINT of choosing it, and a summary that left them empty would
make the safer method look like the less informative one — which is precisely
the mistake a reader would then make in a methods section.

NOTHING HERE IS NEW STATISTICS. It is a collector:

* :func:`spacr.trial_metrics.fit_quality`, ``.residual_diagnostics``,
  ``.design_diagnostics``, ``.calibration`` and ``.control_recovery`` already
  read every scalar off the fitted model, cheaply, and are what a sweep row
  is built from;
* :func:`spacr.regression_qc.residual_normality` is the normality verdict the
  QC panel draws, so the picture and the prose cannot disagree;
* :func:`spacr.regression_qc.context_from_model` recovers leverage and the
  standardised residual from a model that kept its own design;
* :func:`spacr.multiple_testing.critical_p_value` is the exact BH threshold;
* :data:`spacr.qt.widgets.sweep_runs.PREFERRED_COLUMNS` names what a run is
  COMPARED by, and :data:`COMPARISON_FIELDS` maps every one of those columns
  onto the field that reports it. The columns a run is compared by should be
  the columns its summary reports; if they disagree, one of them is wrong.

WHERE IT GOES. The run folder, under :data:`spacr.ml.SUMMARY_FILENAME` — the
file :func:`spacr.qt.widgets.regression_results.find_summary_file` already
reads back, so re-opening a run from disk shows this summary
with no GUI change. The statsmodels text, where there is one, is appended
VERBATIM at the end rather than replaced.
"""

from __future__ import annotations

import os
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CONTRACT",
    "COMPARISON_FIELDS",
    "NOT_A_FIT_PROPERTY",
    "RunSummary",
    "SummaryField",
    "SummarySection",
    "build_run_summary",
    "format_run_summary",
    "write_run_summary",
]

# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

#: How a field was filled. Three states, because two are not enough: a
#: permutation test does not FAIL to check equal variance, it declines to
#: assume it, and printing that as "not applicable" reads like a gap in the
#: method rather than a property of it.
COMPUTED = "computed"
NOT_APPLICABLE = "not applicable"
NOT_ASSUMED = "not assumed"

#: The sections, in the order they are printed, with their headings.
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("fitted", "WHAT WAS FITTED"),
    ("design", "THE DESIGN"),
    ("fit_quality", "FIT QUALITY"),
    ("assumptions", "ASSUMPTIONS, EACH WITH ITS TEST AND ITS VERDICT"),
    ("call", "THE CALL"),
    ("excluded", "WHAT WAS EXCLUDED, AND WHY"),
)

#: EVERY FIELD OF EVERY SECTION. This is the contract: whatever the mode,
#: :func:`build_run_summary` returns exactly these names, each answered.
#: ``tests/test_every_regression_mode_gets_a_summary.py`` iterates the whole
#: product of :data:`spacr.regression_spec.REGRESSION_TYPES` and both
#: inferences against it, which is the point of the item.
CONTRACT: Dict[str, Tuple[str, ...]] = {
    "fitted": (
        "regression_type", "hyperparameters", "inference", "analysis_mode",
        "backend", "level", "dependent_variable", "analysis_unit", "agg_type",
        "transform", "formula", "plate_position",
    ),
    "design": (
        "n_wells", "n_guides", "n_genes", "n_cells", "n_rows_fitted",
        "n_observations", "n_parameters", "design_rank",
        "wells_per_parameter", "identifiable",
    ),
    "fit_quality": (
        "r_squared", "r_squared_adj", "pseudo_r_squared", "log_likelihood",
        "aic", "bic", "residual_se", "selection_frequency", "permutations",
        "finest_p", "n_at_finest_p", "blocking",
    ),
    "assumptions": (
        "equal_variance", "normality", "independence", "influence",
        "multicollinearity",
    ),
    "call": (
        "multiple_testing_method", "fdr_alpha", "n_tested", "n_called",
        "critical_p", "genomic_inflation", "effect_size_cut",
        "positive_rank", "positive_percentile",
    ),
    "excluded": (
        "min_cell_count", "exclude_grnas", "fraction_threshold", "missing_metadata",
        "rows_not_fitted", "untested_coefficients", "below_effect_size",
    ),
}

#: The human label for each field, keyed by ``(section, name)``.
LABELS: Dict[Tuple[str, str], str] = {
    ("fitted", "regression_type"): "regression type",
    ("fitted", "hyperparameters"): "hyperparameters",
    ("fitted", "inference"): "inference",
    ("fitted", "analysis_mode"): "analysis mode",
    ("fitted", "backend"): "backend",
    ("fitted", "level"): "level(s) fitted",
    ("fitted", "dependent_variable"): "response",
    ("fitted", "analysis_unit"): "analysis unit",
    ("fitted", "agg_type"): "well aggregation",
    ("fitted", "transform"): "response transform",
    ("fitted", "formula"): "formula fitted",
    ("fitted", "plate_position"): "plate position",
    ("design", "n_wells"): "wells",
    ("design", "n_guides"): "guides",
    ("design", "n_genes"): "genes",
    ("design", "n_cells"): "cells",
    ("design", "n_rows_fitted"): "rows in the fitted table",
    ("design", "n_observations"): "observations fitted",
    ("design", "n_parameters"): "parameters estimated",
    ("design", "design_rank"): "rank of the design",
    ("design", "wells_per_parameter"): "observations per parameter",
    ("design", "identifiable"): "identifiable",
    ("fit_quality", "r_squared"): "R2",
    ("fit_quality", "r_squared_adj"): "adjusted R2",
    ("fit_quality", "pseudo_r_squared"): "pseudo-R2",
    ("fit_quality", "log_likelihood"): "log-likelihood",
    ("fit_quality", "aic"): "AIC",
    ("fit_quality", "bic"): "BIC",
    ("fit_quality", "residual_se"): "residual standard error",
    ("fit_quality", "selection_frequency"): "bootstrap selection",
    ("fit_quality", "permutations"): "permutations run",
    ("fit_quality", "finest_p"): "finest P expressible",
    ("fit_quality", "n_at_finest_p"): "tests at that floor",
    ("fit_quality", "blocking"): "blocking",
    ("assumptions", "equal_variance"): "equal variance",
    ("assumptions", "normality"): "normality of residuals",
    ("assumptions", "independence"): "independence / clustering",
    ("assumptions", "influence"): "influence and leverage",
    ("assumptions", "multicollinearity"): "multicollinearity",
    ("call", "multiple_testing_method"): "correction",
    ("call", "fdr_alpha"): "alpha",
    ("call", "n_tested"): "coefficients tested",
    ("call", "n_called"): "coefficients called",
    ("call", "critical_p"): "critical raw P",
    ("call", "genomic_inflation"): "genomic inflation",
    ("call", "effect_size_cut"): "effect-size cut",
    ("call", "positive_rank"): "positive control rank",
    ("call", "positive_percentile"): "positive control percentile",
    ("excluded", "min_cell_count"): "min_cell_count",
    ("excluded", "exclude_grnas"): "pre-fraction exclusions",
    ("excluded", "fraction_threshold"): "fraction_threshold",
    ("excluded", "missing_metadata"): "unpaired / missing metadata",
    ("excluded", "rows_not_fitted"): "rows not fitted",
    ("excluded", "untested_coefficients"): "rows that are not tests",
    ("excluded", "below_effect_size"): "called, then cut for width",
}

#: ``PREFERRED_COLUMNS`` name -> the summary field that reports it.
#:
#: THE COLUMNS A RUN IS COMPARED BY SHOULD BE THE COLUMNS ITS SUMMARY REPORTS.
#: :data:`spacr.qt.widgets.sweep_runs.PREFERRED_COLUMNS` is the Runs tab's
#: ordering, and a column there that no field answers is either a comparison
#: nobody can justify or a summary with a hole in it. A test walks that tuple
#: against this mapping and :data:`NOT_A_FIT_PROPERTY`, so the two cannot
#: drift apart in silence.
COMPARISON_FIELDS: Dict[str, str] = {
    "dependent_variable": "dependent_variable",
    "regression_type": "regression_type",
    "regression_backend": "backend",
    "inference": "inference",
    "guide_permutations": "permutations",
    "analysis_unit": "analysis_unit",
    "agg_type": "agg_type",
    "transform": "transform",
    "multiple_testing_method": "multiple_testing_method",
    "fdr_alpha": "fdr_alpha",
    "fraction_threshold": "fraction_threshold",
    "min_cell_count": "min_cell_count",
    "n_wells": "n_wells",
    "n_guides": "n_guides",
    "n_cells": "n_cells",
    "n_rows_fitted": "n_rows_fitted",
    "n_results": "n_tested",
    "n_below_alpha": "n_called",
    "positive_rank": "positive_rank",
    "positive_percentile": "positive_percentile",
    "r_squared": "r_squared",
    "genomic_inflation": "genomic_inflation",
}

#: Runs-tab columns that are properties of the BOOKKEEPING rather than of the
#: fit, so a summary of the fit correctly has no field for them. Named
#: explicitly instead of left out, because "no field for it" and "nobody
#: thought about it" look identical in a mapping.
NOT_A_FIT_PROPERTY: Tuple[str, ...] = (
    "loaded", "run", "source", "trial_id", "status", "seconds", "error_type",
)

#: Backends that report a coefficient but no usable frequentist P value, so
#: training R2 is not a fit statistic for them and they are ranked by
#: bootstrap selection frequency. Kept as a local tuple rather than imported
#: from :mod:`spacr.regression_spec` only for the two extra penalised names;
#: the shared list is read below and this adds to it.
_EXTRA_PENALISED = ("ridge", "hinge", "horseshoe")

#: Where the summary is written when :mod:`spacr.ml` cannot be imported to
#: ask. ml is the writer of record; this is the fallback so the module stays
#: importable on its own.
_FALLBACK_SUMMARY_FILENAME = "model_summary.txt"

#: Total line width the values are wrapped at.
_WIDTH = 88


# ---------------------------------------------------------------------------
# The field types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryField:
    """One reported field: a value, or a reason there is none.

    A BLANK IS NOT REPRESENTABLE. Exactly one of ``value`` and ``reason`` may
    be given, and neither may be empty, so "we did not check" cannot be
    written as an empty string and a zero cannot stand in for it either. That
    is the whole failure this module is about, enforced where it cannot be
    forgotten.

    :param name: stable machine name; unique within its section.
    :param label: what the printed line calls it.
    :param value: the answer, already formatted for a human.
    :param reason: why there is no answer, as a sentence.
    :param kind: :data:`COMPUTED`, :data:`NOT_APPLICABLE` or
        :data:`NOT_ASSUMED`. Defaults from which of the two above was given.
    :raises ValueError: if both or neither are given, or one is blank.
    """

    name: str
    label: str
    value: Optional[str] = None
    reason: Optional[str] = None
    kind: str = ""

    def __post_init__(self):
        has_value = self.value is not None
        has_reason = self.reason is not None
        if has_value == has_reason:
            raise ValueError(
                f"field {self.name!r} must have exactly one of a value and a "
                f"reason; got value={self.value!r} reason={self.reason!r}")
        text = self.value if has_value else self.reason
        if not str(text).strip():
            raise ValueError(
                f"field {self.name!r} was given an empty "
                f"{'value' if has_value else 'reason'}; a blank is the thing "
                f"this type exists to refuse")
        if not self.kind:
            object.__setattr__(
                self, "kind", COMPUTED if has_value else NOT_APPLICABLE)
        if self.kind not in (COMPUTED, NOT_APPLICABLE, NOT_ASSUMED):
            raise ValueError(f"field {self.name!r}: unknown kind {self.kind!r}")
        if self.kind is not COMPUTED and has_value and not has_reason:
            raise ValueError(
                f"field {self.name!r} is {self.kind} but carries a value; the "
                f"reason is what a reader needs")

    @property
    def answered(self) -> bool:
        """True when the field carries a number rather than a reason."""
        return self.kind == COMPUTED

    @property
    def text(self) -> str:
        """The right-hand side of the printed line, prefix included."""
        if self.kind == COMPUTED:
            return str(self.value)
        prefix = "NOT ASSUMED" if self.kind == NOT_ASSUMED else "not applicable"
        return f"{prefix} — {self.reason}"


@dataclass
class SummarySection:
    """One headed block of fields, in contract order.

    :param name: key into :data:`CONTRACT`.
    :param title: the printed heading.
    :param fields: the fields, one per name in ``CONTRACT[name]``.
    """

    name: str
    title: str
    fields: List[SummaryField] = field(default_factory=list)

    def get(self, name: str) -> Optional[SummaryField]:
        """The field called ``name``, or ``None``.

        :param name: stable field name to look up in this section.
        """
        for one in self.fields:
            if one.name == name:
                return one
        return None


@dataclass
class RunSummary:
    """Everything spaCR knows about one finished run.

    :param sections: the six sections of :data:`SECTIONS`, in order.
    :param warnings: sentences printed ABOVE everything else. The
        identifiability warning lives here, which is where it already is on
        the Summary tab and where it has to stay: statsmodels prints a full
        table of standard errors regardless, and it looks exactly like a
        summary of a well-posed fit.
    :param verbatim: the statsmodels text summary, appended unchanged, or
        ``None``.
    :param verbatim_note: what the verbatim block is, or why there is none.
    :param recommendations: evidence-backed changes derived from values in
        this summary, in display order.
    """

    sections: List[SummarySection] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verbatim: Optional[str] = None
    verbatim_note: str = ""
    #: What the run's own numbers say to change (225). Derived by
    #: `spacr.run_recommendations.recommend` from measured values, never
    #: composed -- each one is a rule with a threshold that fired.
    recommendations: List[Any] = field(default_factory=list)

    def section(self, name: str) -> Optional[SummarySection]:
        """The section called ``name``, or ``None``.

        :param name: contract section name to look up.
        """
        for one in self.sections:
            if one.name == name:
                return one
        return None

    def field(self, name: str) -> Optional[SummaryField]:
        """The first field called ``name`` in any section, or ``None``.

        :param name: stable field name to search for across sections.
        """
        for one in self.sections:
            found = one.get(name)
            if found is not None:
                return found
        return None

    def missing(self) -> List[str]:
        """Contract names this summary failed to answer, as ``section.name``.

        Empty on every well-formed summary. It is the assertion the
        whole-product test makes, expressed once here so the test says what it
        means rather than re-deriving the contract.
        """
        out: List[str] = []
        for key, names in CONTRACT.items():
            block = self.section(key)
            if block is None:
                out.extend(f"{key}.{name}" for name in names)
                continue
            for name in names:
                if block.get(name) is None:
                    out.append(f"{key}.{name}")
        return out

    def text(self) -> str:
        """The whole summary as it is written to disk."""
        return format_run_summary(self)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


#: Column width the labels are padded to. MEASURED FROM :data:`LABELS`
#: rather than chosen, so adding a longer label cannot silently produce a
#: column that runs into its own values -- which is what "observations the
#: estimator saw" did on the first real run.
_LABEL_WIDTH = max(len(one) for one in LABELS.values()) + 2


def _raw(value) -> Optional[str]:
    """``str(value)`` unless it is missing.

    NOT :func:`_clean`. ``'none'`` is a REAL answer for
    ``multiple_testing_method`` -- it is one of the thirteen entries in
    :data:`spacr.multiple_testing.METHODS` and it is spaCR's own default --
    and reading it as absent is how the first real run of this module reported
    "neither the results table nor the settings record which correction was
    applied" about a run that recorded it twice.
    """
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def _clean(value) -> Optional[str]:
    """``str(value)`` unless it is one of the many spellings of absent."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan", "null"):
        return None
    return text


def _count(value) -> Optional[int]:
    """``value`` as a plain int, or ``None`` when it is not a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def _number(value, digits: int = 4) -> Optional[str]:
    """A float formatted for a human, or ``None`` when there is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    if number and (abs(number) < 1e-3 or abs(number) >= 1e6):
        return f"{number:.{digits}g}"
    return f"{number:,.{digits}g}"


def _setting(settings: Optional[Mapping[str, Any]], key: str, default=None):
    if not isinstance(settings, Mapping):
        return default
    value = settings.get(key, default)
    return default if value is None else value


def _column(frame, *names) -> Optional[str]:
    if not isinstance(frame, pd.DataFrame):
        return None
    for name in names:
        if name in frame.columns:
            return name
    return None


def _floats(frame, column) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or not column:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _one_value(frame, column):
    """The single value of ``column``, when every row agrees on one."""
    if not isinstance(frame, pd.DataFrame) or column not in getattr(
            frame, "columns", ()):
        return None
    values = frame[column].dropna().unique()
    return values[0] if len(values) == 1 else None


# ---------------------------------------------------------------------------
# What kind of run is this
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    """Everything the builders read, resolved once."""

    res_folder: Optional[str]
    model: Any
    settings: Mapping[str, Any]
    coef_df: Optional[pd.DataFrame]
    regression_type: Optional[str]
    nonparametric: bool
    penalised: bool
    data: Optional[pd.DataFrame]
    data_note: str
    metrics: Dict[str, Any]
    #: WHAT THE SECTIONS ABOVE ACTUALLY WROTE. The assumption builders
    #: measure a number, format it into a sentence and would otherwise throw
    #: the number away -- so the recommendations at the end would have had to
    #: recompute it, and a recommendation that disagrees with the line it
    #: cites is worse than no recommendation. Each builder deposits here as
    #: it goes, and `recommend` reads only this.
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _is_nonparametric(settings, coef_df) -> bool:
    """Whether this run was the permutation test.

    THREE SIGNALS, because no one of them is present on every path.
    ``inference='auto'`` is resolved into ``analysis_mode`` and leaves
    ``inference`` reading ``'auto'``, a settings CSV from before the
    ``inference`` key existed carries only ``analysis_mode``, and a table
    re-read from disk has neither — but a permutation table always carries the
    ``permutations`` column :func:`spacr.guide_permutation.analyse_long_guide_table`
    puts on every row.
    """
    mode = str(_setting(settings, "analysis_mode", "") or "").strip().lower()
    if mode == "guide_permutation":
        return True
    inference = str(_setting(settings, "inference", "") or "").strip().lower()
    if inference == "nonparametric":
        return True
    return _column(coef_df, "permutations", "permutation_p_value") is not None


def _penalised_types() -> Tuple[str, ...]:
    try:
        from .regression_spec import NO_P_VALUE_TYPES
    except Exception:                                            # noqa: BLE001
        NO_P_VALUE_TYPES = ("lasso", "elasticnet", "group_lasso")
    return tuple(NO_P_VALUE_TYPES) + _EXTRA_PENALISED


def _read_fitted_table(res_folder) -> Tuple[Optional[pd.DataFrame], str]:
    """The run's own ``regression_data.csv``, and where it came from.

    THIS IS THE TABLE THAT REACHED THE FIT, written by
    :func:`spacr.ml.perform_regression` before any model is built, so it is
    the honest source for wells, guides, genes and cells — the four counts the
    Runs tab compares runs by. Reading it is collecting; counting the inputs
    again would be re-deriving, and the two can differ by every row the merge
    dropped.
    """
    if not res_folder:
        return None, ("no run folder was given, so regression_data.csv could "
                      "not be read")
    path = os.path.join(str(res_folder), "regression_data.csv")
    if not os.path.isfile(path):
        return None, f"{path} does not exist"
    try:
        return pd.read_csv(path), f"read from {path}"
    except Exception as error:                                   # noqa: BLE001
        return None, f"{path} could not be read ({type(error).__name__}: {error})"


def _collect_metrics(model, coef_df, settings) -> Dict[str, Any]:
    """Every scalar :mod:`spacr.trial_metrics` can read off this run.

    ONE CALL PER BLOCK, each guarded on its own, exactly as
    :func:`spacr.trial_metrics.summarise_trial` does — a family with no
    R-squared must still contribute its control recovery. Nothing is
    recomputed that the fit already knows.
    """
    out: Dict[str, Any] = {}
    try:
        from . import trial_metrics
    except Exception:                                            # noqa: BLE001
        return out
    frame = coef_df if isinstance(coef_df, pd.DataFrame) else pd.DataFrame()
    for block in (
        lambda: trial_metrics.fit_quality(model),
        lambda: trial_metrics.residual_diagnostics(model),
        lambda: trial_metrics.design_diagnostics(model),
        lambda: trial_metrics.calibration(frame),
        lambda: trial_metrics.control_recovery(frame, settings or {}),
    ):
        try:
            out.update(block())
        except Exception:                                        # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# The identifiability warning, which stays at the top
# ---------------------------------------------------------------------------

#: What the run says when a fit is not identifiable, repeated here so it is
#: the FIRST thing in the written summary.
#:
#: The same sentence is on the Summary tab
#: (``regression_results.UNIDENTIFIABLE_WARNING``) and it cannot be imported
#: from there: that module imports PySide6 at the top, and a headless run
#: writing its summary must not need a GUI toolkit.
UNIDENTIFIABLE_WARNING = (
    "THIS FIT IS NOT IDENTIFIABLE: {observations:,} analysed observations are "
    "being used to estimate {parameters:,} parameters.\n"
    "Every standard error and P value below is one arbitrary solution out of "
    "infinitely many; refitting the same data can give different numbers and "
    "neither set is wrong.\n"
    "Set inference='nonparametric' to test each guide as a marginal "
    "association, reshuffling wells only between wells of the same plate, "
    "which stays valid at any width."
)


def _warnings(run: "_Run") -> List[str]:
    """Sentences that go ABOVE the summary, most serious first."""
    out: List[str] = []
    parameters = _count(run.metrics.get("n_parameters"))
    observations = _count(run.metrics.get("n_observations"))
    if observations is None:
        observations = _count(getattr(run.model, "nobs", None))
    rank = _count(run.metrics.get("design_rank"))
    if parameters and observations is not None:
        deficient = rank is not None and rank < parameters
        if observations <= parameters or deficient:
            out.append(UNIDENTIFIABLE_WARNING.format(
                observations=observations, parameters=parameters))
            if deficient:
                out.append(
                    f"The design has rank {rank:,} against {parameters:,} "
                    f"columns, so {parameters - rank:,} direction(s) of the "
                    f"coefficient vector are not determined by the data at "
                    f"all.")
    return out


# ---------------------------------------------------------------------------
# Section builders. One per section of the contract.
# ---------------------------------------------------------------------------


def _fitted_section(run: "_Run") -> List[SummaryField]:
    settings, out = run.settings, []

    def add(name, value=None, reason=None, kind=""):
        """Append a labelled field to the fitted-model section."""
        out.append(SummaryField(name, LABELS[("fitted", name)],
                                value=value, reason=reason, kind=kind))

    kind = _clean(run.regression_type) or _clean(
        _setting(settings, "regression_type"))
    if run.nonparametric:
        # SAID, because the settings still carry one and a reader will
        # otherwise believe it. The permutation path never reaches
        # `regression_model`: it is a marginal test, so no family is fitted
        # and the setting has no effect on the numbers below.
        add("regression_type",
            value=(f"none — the permutation path fits no model, so the "
                   f"regression_type in the settings ({kind}) was not read "
                   f"and did not affect this run" if kind else
                   "none — the permutation path fits no model, so no "
                   "regression family was chosen"))
    elif kind:
        add("regression_type", value=kind)
    else:
        add("regression_type",
            reason="this run recorded no regression_type, so the family it "
                   "fitted cannot be named from what it wrote")

    # Report only hyperparameters the selected regression family reads. The
    # shared family table also drives the disabled-setting rules and tooltip
    # text, keeping the summary aligned with the interface and preventing an
    # ignored value (such as alpha for OLS) from being presented as fitted.
    if run.nonparametric:
        add("hyperparameters",
            value="none — the permutation path fits no model, so there is no "
                  "hyperparameter to read")
    elif kind:
        add("hyperparameters", **_hyperparameter_report(kind, settings, run))
    else:
        add("hyperparameters",
            reason="this run recorded no regression_type, so the "
                   "hyperparameters its family reads cannot be listed")

    inference = str(_setting(settings, "inference", "") or "").strip().lower()
    mode = str(_setting(settings, "analysis_mode", "") or "").strip().lower()
    if run.nonparametric:
        chosen = ("inference='auto' resolved to it" if inference == "auto"
                  else f"inference={inference or 'nonparametric'!r}")
        add("inference",
            value=f"nonparametric — each guide tested as a marginal "
                  f"association by Freedman-Lane permutation, wells "
                  f"reshuffled only between wells of the same plate "
                  f"({chosen})")
    else:
        chosen = ("inference='auto' resolved to it" if inference == "auto"
                  else f"inference={inference or 'parametric'!r}")
        add("inference",
            value=f"parametric — every coefficient fitted simultaneously in "
                  f"one design ({chosen})")
    add("analysis_mode",
        value=mode or ("guide_permutation" if run.nonparametric
                       else "regression"))

    backend = _clean(_setting(settings, "regression_backend")) or "statsmodels"
    label = backend
    try:
        from .regression_spec import REGRESSION_BACKENDS
        label = REGRESSION_BACKENDS.get(backend, {}).get("label", backend)
    except Exception:                                            # noqa: BLE001
        pass
    # NOT "statsmodels (statsmodels (CPU))". The spec's label already
    # CONTAINS the backend's name -- it is "pyfixest (CPU)", not "(CPU)" --
    # so bracketing it after the name said everything twice.
    add("backend", value=label if label.startswith(backend) else
        (f"{backend} ({label})" if label != backend else backend))

    asked = _clean(_setting(settings, "level")) or "both"
    got = None
    if _column(run.coef_df, "level"):
        got = ", ".join(sorted({str(v) for v in
                                run.coef_df["level"].dropna().unique()}))
    add("level", value=f"{asked}" + (f" (rows written for: {got})" if got
                                     else ""))

    response = _clean(_setting(settings, "dependent_variable"))
    if response:
        add("dependent_variable", value=response)
    else:
        add("dependent_variable",
            reason="the run recorded no dependent_variable in its settings")

    add("analysis_unit",
        value=str(_setting(settings, "analysis_unit", "well") or "well"))
    agg = _setting(settings, "agg_type")
    add("agg_type",
        value=str(agg) if agg is not None
        else "none — one row per object, not per well")
    transform = _setting(settings, "transform")
    if transform is not None and str(transform).lower() == "beta":
        # NAMED HERE, not left in the run log. The logit squeeze moves a
        # well sitting at exactly 0 or 1, and a reader comparing the fitted
        # response to their own column has to be told that happened.
        from .ml import BETA_SQUEEZE_NOTE
        add("transform", value=BETA_SQUEEZE_NOTE)
    else:
        add("transform",
            value=str(transform) if transform is not None
            else "none — the response was fitted as measured")

    add("formula", **_formula(run))

    if run.nonparametric:
        block = _clean(_one_value(run.coef_df, "block_column")) or str(
            _setting(settings, "guide_permutation_block", "plateID"))
        nuisance = _clean(_one_value(run.coef_df, "nuisance_columns"))
        add("plate_position",
            value=f"residualised out as part of the block design on {block}"
                  + (f" plus {nuisance}" if nuisance else "")
                  + "; it is not a fitted coefficient here")
    elif not bool(_setting(settings, "model_plate_position", True)):
        add("plate_position",
            value="OUT of the model entirely (model_plate_position=False)")
    elif bool(_setting(settings, "random_row_column_effects", False)):
        add("plate_position",
            value="in, as VARIANCE COMPONENTS on rowID and columnID "
                  "(random_row_column_effects=True)")
    else:
        add("plate_position",
            value="in, as FIXED effects on rowID and columnID")
    return out


def _formula(run: "_Run") -> Dict[str, str]:
    """The formula actually fitted, read off the fit where there is one."""
    if run.nonparametric:
        block = _clean(_one_value(run.coef_df, "block_column")) or str(
            _setting(run.settings, "guide_permutation_block", "plateID"))
        return {"value": (
            f"no design matrix — each guide's well fraction and the response "
            f"are both residualised on the {block} block design, and the "
            f"residual correlation is compared against its own permutation "
            f"null (Freedman-Lane)")}
    inner = getattr(run.model, "model", None)
    formula = _clean(getattr(inner, "formula", None))
    if formula:
        return {"value": f"{formula}   (read off the fit)"}
    try:
        from .ml import prepare_formula

        level = str(_setting(run.settings, "level", "grna") or "grna")
        if level == "both":
            level = "grna"
        # THE RESPONSE THE ESTIMATOR SAW, not the one the user typed.
        # `transform='log'` fits `log_pred`, and a formula naming `pred`
        # describes a fit nobody ran -- the statsmodels block below this one
        # says `Dep. Variable: log_pred` in the same file.
        response = (_raw(getattr(inner, "endog_names", None))
                    or _clean(_setting(run.settings, "dependent_variable"))
                    or "y")
        rebuilt = prepare_formula(
            response,
            random_row_column_effects=bool(
                _setting(run.settings, "random_row_column_effects", False)),
            level=level,
            model_plate_position=bool(
                _setting(run.settings, "model_plate_position", True)))
        return {"value": f"{rebuilt}   (rebuilt from the settings by "
                         f"prepare_formula; this backend keeps no formula)"}
    except Exception as error:                                   # noqa: BLE001
        return {"reason": (
            f"this backend kept no formula and one could not be rebuilt from "
            f"the settings ({type(error).__name__}: {error})")}


def _design_section(run: "_Run") -> List[SummaryField]:
    out = []

    def add(name, value=None, reason=None, kind=""):
        """Append a labelled field to the fitted-design section."""
        out.append(SummaryField(name, LABELS[("design", name)],
                                value=value, reason=reason, kind=kind))

    frame, note = run.data, run.data_note
    counts = _design_counts(frame)
    for name, column, unit in (("n_wells", "prc", "well"),
                               ("n_guides", "grna", "guide"),
                               ("n_genes", "gene", "gene")):
        number = counts.get(name)
        if number is None:
            add(name, reason=f"the fitted table has no {column!r} column "
                             f"({note})")
        else:
            add(name, value=f"{number:,} distinct {unit}(s)")
    cells = counts.get("n_cells")
    if cells is None:
        add("n_cells",
            reason=f"the fitted table has no 'cell_count' column, so the "
                   f"objects behind the wells cannot be counted ({note})")
    else:
        add("n_cells",
            value=f"{cells:,} objects, summed over the distinct wells")
    rows = counts.get("n_rows_fitted")
    if rows is None:
        add("n_rows_fitted", reason=note)
    else:
        add("n_rows_fitted", value=f"{rows:,} rows in regression_data.csv")

    observations = _count(run.metrics.get("n_observations"))
    if observations is None:
        observations = _count(getattr(run.model, "nobs", None))
    if observations is not None:
        add("n_observations", value=f"{observations:,}")
    elif run.nonparametric:
        wells = counts.get("n_wells")
        add("n_observations",
            value=f"{wells:,} wells entered every guide's test"
            if wells is not None else
            "one row per well; the count is the well count above")
    else:
        add("n_observations",
            reason="this estimator reports neither nobs nor residuals, so "
                   "what it saw cannot be read back off it")

    parameters = _count(run.metrics.get("n_parameters"))
    rank = _count(run.metrics.get("design_rank"))
    if run.nonparametric:
        reason = ("the permutation test estimates no joint parameter vector: "
                  "each guide is its own marginal test, so there is no "
                  "design matrix to count columns of or take the rank of")
        add("n_parameters", reason=reason)
        add("design_rank", reason=reason)
        add("wells_per_parameter", reason=reason)
        add("identifiable",
            value="yes, by construction — a marginal test of one guide "
                  "against a block design cannot be rank deficient in the "
                  "way a simultaneous fit of every guide can")
        return out
    if parameters is None:
        reason = ("this backend does not keep the design matrix it was "
                  "fitted with, so its width and rank cannot be recovered")
        add("n_parameters", reason=reason)
        add("design_rank", reason=reason)
        add("wells_per_parameter", reason=reason)
        add("identifiable", reason=reason)
        return out
    add("n_parameters", value=f"{parameters:,}")
    if rank is None:
        add("design_rank", reason="the fit reported no rank and the design "
                                  "could not be decomposed to find one")
    else:
        add("design_rank", value=f"{rank:,} of {parameters:,} columns")
    ratio = _number(run.metrics.get("wells_per_parameter"), 3)
    if ratio is None:
        add("wells_per_parameter",
            reason="there are no parameters to divide the observations by")
    else:
        add("wells_per_parameter", value=ratio)
    if "design_identifiable" in run.metrics:
        ok = bool(run.metrics["design_identifiable"])
        lost = _count(run.metrics.get("non_identifiable_directions")) or 0
        add("identifiable",
            value="yes" if ok else
                  f"NO — {lost:,} direction(s) of the coefficient vector are "
                  f"not determined by the data; the standard errors below "
                  f"come from a pseudo-inverse")
    else:
        add("identifiable",
            reason="the rank and the residual degrees of freedom could not "
                   "both be read off this fit, and identifiability is the "
                   "comparison between them")
    return out


#: Retention below this percentage receives an explicit warning in the run
#: summary. The threshold distinguishes severe filtering from routine removal
#: of low-abundance rows.
_LOW_RETENTION_PERCENT = 5.0

#: Pairing below this percentage of the smaller input receives an explicit
#: warning. The two inputs may have different coverage, so unmatched wells on
#: the larger side do not make an otherwise complete join appear incomplete.
_LOW_PAIRING_PERCENT = 50.0


def _design_counts(frame) -> Dict[str, Optional[int]]:
    """Wells, guides, genes, cells and rows from the run's fitted table.

    ``n_cells`` is summed over the DISTINCT WELLS. ``regression_data.csv`` is
    one row per (well, guide), so ``cell_count`` repeats once per guide in the
    well and summing the column would multiply the screen's objects by its
    guides-per-well — about eight times over on a real screen.
    """
    out: Dict[str, Optional[int]] = {
        "n_wells": None, "n_guides": None, "n_genes": None,
        "n_cells": None, "n_rows_fitted": None,
    }
    if not isinstance(frame, pd.DataFrame):
        return out
    out["n_rows_fitted"] = int(len(frame))
    for name, column in (("n_wells", "prc"), ("n_guides", "grna"),
                         ("n_genes", "gene")):
        if column in frame.columns:
            out[name] = int(frame[column].nunique())
    if "cell_count" in frame.columns:
        wells = frame.drop_duplicates(subset=["prc"]) \
            if "prc" in frame.columns else frame
        total = pd.to_numeric(wells["cell_count"], errors="coerce").sum()
        if np.isfinite(total):
            out["n_cells"] = int(total)
    return out


#: Families fitted by maximum likelihood through a link, whose R2 does not
#: exist and whose comparable quantity is a PSEUDO-R2 on a different scale.
_GLM_FAMILY = ("glm", "poisson", "quasi_binomial", "beta", "logit", "probit")

#: Families that minimise something other than the sum of squares, so the
#: fraction-of-variance-explained identity R2 rests on does not hold for them
#: and statsmodels reports none. Named, so the summary can say WHICH of the
#: several reasons applies rather than "this run reports no R2".
_NO_R2_FAMILY = {
    "rlm": ("a robust fit minimises Huber's loss, not the sum of squares, so "
            "the fraction-of-variance identity R2 rests on does not hold and "
            "statsmodels reports none"),
    "huber": ("a Huber fit minimises a clipped loss, not the sum of squares, "
              "so the fraction-of-variance identity R2 rests on does not hold "
              "and statsmodels reports none"),
    "quantile": ("quantile regression fits a conditional QUANTILE by "
                 "minimising a check loss; there is no conditional mean to "
                 "take the explained variance of. statsmodels reports a "
                 "pseudo-R2 for it instead, on its own scale"),
    "mixed": ("a mixed model splits the variance between fixed effects and "
              "random effects, so there is no single R2: statsmodels reports "
              "none and the marginal / conditional pair a reader may expect "
              "would need the variance components, not a fitted value"),
}


def _no_residual_reason(run: "_Run") -> str:
    """Why a residual-based test could not be run on this fit."""
    if run.model is None:
        return ("this run handed over no fitted model, so there are no "
                "residuals to test")
    resid = np.asarray(getattr(run.model, "resid", []), dtype=object)
    try:
        values = np.asarray(getattr(run.model, "resid", []), dtype=float)
        finite = int(np.isfinite(values).sum())
    except (TypeError, ValueError):
        return (f"{type(run.model).__name__} exposes no residual vector, so "
                f"the residual tests cannot be run on it")
    del resid
    if finite < 8:
        return (f"the test needs at least 8 finite residuals and this fit has "
                f"{finite}")
    return (f"{type(run.model).__name__} kept no design matrix beside its "
            f"residuals, or the design is singular, so the test could not be "
            f"formed")


def _fit_quality_section(run: "_Run") -> List[SummaryField]:
    out, metrics = [], run.metrics

    def add(name, value=None, reason=None, kind=""):
        """Append a labelled field to the fit-quality section."""
        out.append(SummaryField(name, LABELS[("fit_quality", name)],
                                value=value, reason=reason, kind=kind))

    kind = (run.regression_type or "").strip().lower()
    r2 = _number(metrics.get("r_squared"), 4)
    if r2 is not None:
        add("r_squared", value=r2)
    elif run.nonparametric:
        add("r_squared",
            reason="R2 DOES NOT EXIST for this run. The permutation test "
                   "fits no model, so there is no fitted value, no residual "
                   "and no explained variance to divide. What it reports "
                   "instead is the permutation resolution below")
    elif run.penalised:
        add("r_squared",
            reason="R2 on the training data of a penalised fit is NOT a fit "
                   "statistic: the penalty deliberately trades fit for "
                   "stability, so the number measures how little was "
                   "penalised. The bootstrap selection frequency below is "
                   "what these types are ranked by")
    elif kind in _GLM_FAMILY:
        add("r_squared",
            reason="this family is fitted by maximum likelihood through a "
                   "link, not by least squares, so it has no R2. The "
                   "pseudo-R2 below is the comparable quantity and is NOT on "
                   "the same scale")
    elif kind in _NO_R2_FAMILY:
        add("r_squared", reason=_NO_R2_FAMILY[kind])
    elif run.model is None:
        add("r_squared",
            reason="this run handed over no fitted model, so there is no R2 "
                   "to read off one. Re-opened from disk, the coefficient "
                   "table survives and the fit does not")
    else:
        add("r_squared",
            reason=f"{type(run.model).__name__} reports no R2, and spaCR does "
                   f"not compute one for it: a number this module invented "
                   f"would not be the one any other tool prints")

    adjusted = _number(metrics.get("r_squared_adj"), 4)
    if adjusted is not None:
        add("r_squared_adj", value=adjusted)
    elif r2 is not None:
        add("r_squared_adj",
            reason="this estimator reports an R2 but no degrees-of-freedom "
                   "adjusted one")
    else:
        add("r_squared_adj",
            reason="there is no R2 to adjust — see the line above")

    add("pseudo_r_squared", **_pseudo_r_squared(run))

    for name, key in (("log_likelihood", "log_likelihood"),
                      ("aic", "aic"), ("bic", "bic")):
        value = _number(metrics.get(key), 6)
        if value is not None:
            add(name, value=value)
        elif run.nonparametric:
            add(name, reason="there is no likelihood: a permutation test "
                             "assumes no distribution to have one under")
        else:
            add(name, reason=f"{LABELS[('fit_quality', name)]} is not "
                             f"reported by this estimator")
    se = _number(metrics.get("residual_se"), 4)
    if se is not None:
        add("residual_se", value=se)
    else:
        add("residual_se", reason=_no_residual_reason(run))

    add("selection_frequency", **_selection_frequency(run))
    add("permutations", **_permutations(run))
    add("finest_p", **_finest_p(run))
    add("n_at_finest_p", **_n_at_finest_p(run))
    add("blocking", **_blocking(run))
    return out


def _pseudo_r_squared(run: "_Run") -> Dict[str, str]:
    """McFadden's pseudo-R2, NAMED as pseudo so it is not read as an OLS R2."""
    model = run.model
    if model is None:
        return {"reason": ("there is no fitted model, so there is no "
                           "likelihood to form a pseudo-R2 from")}
    value = None
    getter = getattr(model, "pseudo_rsquared", None)
    if callable(getter):
        for kwargs in ({"kind": "mcf"}, {}):
            try:
                value = float(getter(**kwargs))
                break
            except Exception:                                    # noqa: BLE001
                value = None
    if value is None:
        try:
            value = float(getattr(model, "prsquared"))
        except Exception:                                        # noqa: BLE001
            value = None
    if value is None:
        try:
            llf = float(getattr(model, "llf"))
            llnull = float(getattr(model, "llnull"))
            value = 1.0 - llf / llnull if llnull else None
        except Exception:                                        # noqa: BLE001
            value = None
    text = _number(value, 4)
    if text is None:
        if _number(run.metrics.get("r_squared"), 4) is not None:
            return {"reason": (
                "this fit is least-squares and reports a real R2 above; a "
                "pseudo-R2 is the substitute for the families that have none, "
                "not a second statistic to quote beside one")}
        return {"reason": (
            "this estimator reports neither a pseudo-R2 nor the null "
            "log-likelihood one is formed from")}
    return {"value": (
        f"{text} (McFadden). THIS IS A PSEUDO-R2: it is a likelihood ratio "
        f"against the intercept-only model, not a fraction of variance "
        f"explained, and it must not be compared against an OLS R2")}


def _selection_frequency(run: "_Run") -> Dict[str, str]:
    column = _column(run.coef_df, "selection_frequency", "selection_freq")
    if column is None:
        if run.penalised:
            return {"reason": (
                "this penalised fit wrote no selection_frequency column, so "
                "the stability of its selections cannot be reported")}
        return {"reason": (
            "this family is ranked by its P values; bootstrap selection "
            "frequency is what the penalised families are ranked by instead")}
    values = _floats(run.coef_df, column)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"reason": "the selection_frequency column holds no finite "
                          "value"}
    threshold = float(_setting(run.settings, "lasso_selection_threshold", 0.6)
                      or 0.6)
    n_boot = _count(_setting(run.settings, "lasso_n_boot", 200)) or 200
    above = int((values >= threshold).sum())
    return {"value": (
        f"median {np.median(values):.3g} over {values.size:,} coefficients; "
        f"{above:,} at or above the selection threshold of {threshold:g}, "
        f"from {n_boot:,} bootstrap resamples")}


def _permutation_count(run: "_Run") -> Optional[int]:
    if not run.nonparametric:
        return None
    number = _count(_one_value(run.coef_df, "permutations"))
    if number is None:
        number = _count(_setting(run.settings, "guide_permutations", 200000))
    return number


def _permutations(run: "_Run") -> Dict[str, str]:
    if not run.nonparametric:
        return {"reason": (
            "no permutation null was drawn: this run fits a model and reads "
            "its P values off the estimator's own sampling distribution")}
    number = _permutation_count(run)
    if number is None:
        return {"reason": ("the results table records no permutation count "
                           "and the settings name none")}
    return {"value": f"{number:,}"}


def _finest_p(run: "_Run") -> Dict[str, str]:
    if not run.nonparametric:
        return {"reason": (
            "a P value from a fitted model is continuous, so it has no "
            "coarsest expressible step")}
    number = _permutation_count(run)
    if number is None:
        return {"reason": ("without the permutation count there is no "
                           "1/(n+1) to report")}
    floor = 1.0 / (number + 1.0)
    return {"value": (
        f"{floor:.3g} = 1/({number:,}+1). NO P VALUE BELOW THIS IS "
        f"EXPRESSIBLE — {number:,} permutations cannot report p < "
        f"{floor:.0e}, however strong the effect")}


def _n_at_finest_p(run: "_Run") -> Dict[str, str]:
    if not run.nonparametric:
        return {"reason": "there is no permutation floor for tests to sit on"}
    number = _permutation_count(run)
    column = _column(run.coef_df, "permutation_p_value", "p_value")
    if number is None or column is None:
        return {"reason": ("the results table carries no permutation P value "
                           "to compare against the floor")}
    values = _floats(run.coef_df, column)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"reason": "the P value column holds no finite value"}
    floor = 1.0 / (number + 1.0)
    at_floor = int((values <= floor * (1.0 + 1e-9)).sum())
    return {"value": (
        f"{at_floor:,} of {values.size:,} tests are AT the floor, so their "
        f"evidence is censored: raise guide_permutations to separate them")}


def _blocking(run: "_Run") -> Dict[str, str]:
    if run.nonparametric:
        block = _clean(_one_value(run.coef_df, "block_column")) or str(
            _setting(run.settings, "guide_permutation_block", "plateID"))
        blocks = None
        if isinstance(run.data, pd.DataFrame) and block in run.data.columns:
            blocks = int(run.data[block].nunique())
        return {"value": (
            f"residuals are permuted WITHIN {block}"
            + (f" ({blocks:,} block(s))" if blocks is not None else "")
            + ", so a plate-wide difference cannot become a guide effect")}
    if (run.regression_type or "").strip().lower() == "mixed":
        return {"value": ("plateID as a random intercept; wells within a "
                          "plate are modelled as correlated")}
    if not bool(_setting(run.settings, "model_plate_position", True)):
        return {"value": ("none — plate position is out of the model, so "
                          "nothing absorbs a positional difference")}
    return {"value": ("none — this is a fixed-effects fit; position is "
                      "modelled by rowID and columnID terms rather than by "
                      "blocking")}


def _assumptions_section(run: "_Run") -> List[SummaryField]:
    out = []

    def add(name, value=None, reason=None, kind=""):
        """Append a labelled field to the assumptions section."""
        out.append(SummaryField(name, LABELS[("assumptions", name)],
                                value=value, reason=reason, kind=kind))

    for name, builder in (("equal_variance", _equal_variance),
                          ("normality", _normality),
                          ("independence", _independence),
                          ("influence", _influence),
                          ("multicollinearity", _multicollinearity)):
        add(name, **builder(run))
    return out


def _verdict(p_value: float, alpha: float = 0.05) -> str:
    return ("REJECTED at %g" % alpha) if p_value < alpha \
        else "not rejected at %g" % alpha


def _equal_variance(run: "_Run") -> Dict[str, str]:
    if run.nonparametric:
        return {"kind": NOT_ASSUMED, "reason": (
            "the null is built by permuting residuals within plate blocks, "
            "so no variance model is assumed. A funnel in the response cannot "
            "make a permutation P value too small")}
    bp = run.metrics.get("breusch_pagan_p")
    white = run.metrics.get("white_p")
    if bp is None and white is None:
        return {"reason": _no_residual_reason(run)}
    # ONE VERDICT PER TEST. Reporting the smaller of the two under a single
    # "REJECTED" is how a run where Breusch-Pagan says 0.645 and White says
    # 0.045 comes to read as though both agreed.
    parts, rejected = [], False
    for name, value in (("Breusch-Pagan", bp), ("White", white)):
        if value is None:
            continue
        value = float(value)
        rejected = rejected or value < 0.05
        parts.append(f"{name} p = {value:.3g} ({_verdict(value)})")
    if white is None:
        parts.append("White's test was not run: it squares every column pair, "
                     "so it is attempted only on designs of at most 30 columns "
                     "and skipped when singular")
    tail = (" — at least one test rejects equal variance, so the standard "
            "errors here are the wrong width; cov_type='HC3', or weighting "
            "wells by cell count, is the fix"
            if rejected else
            " — consistent with equal variance")
    return {"value": "; ".join(parts) + tail}


def _normality(run: "_Run") -> Dict[str, str]:
    if run.nonparametric:
        return {"kind": NOT_ASSUMED, "reason": (
            "nothing here is normal by assumption: the reference distribution "
            "is the permutation null of this screen's own residuals, whatever "
            "shape they have")}
    if run.model is None:
        return {"reason": _no_residual_reason(run)}
    try:
        from .regression_qc import residual_normality

        resid = np.asarray(getattr(run.model, "resid", []), dtype=float)
        shape = residual_normality(resid)
    except Exception as error:                                   # noqa: BLE001
        return {"reason": (f"the residual shape could not be measured "
                           f"({type(error).__name__}: {error})")}
    if shape["n"] < 3:
        return {"reason": shape["test"]}
    run.diagnostics.update(
        normality_p=shape["normality_p"], residual_skew=shape["skew"],
        excess_kurtosis=shape["excess_kurtosis"])
    head = (f"skew {shape['skew']:+.2f}, excess kurtosis "
            f"{shape['excess_kurtosis']:+.2f} over {shape['n']:,} residuals")
    p_value = shape["normality_p"]
    if not np.isfinite(p_value):
        return {"value": f"{head}; {shape['test']}, so there is no P value"}
    return {"value": (f"{head}; {shape['test']} p = {p_value:.3g} "
                      f"({_verdict(p_value)})")}


def _independence(run: "_Run") -> Dict[str, str]:
    if run.nonparametric:
        return {"kind": NOT_ASSUMED, "reason": (
            "wells are treated as exchangeable only WITHIN a block, which is "
            "weaker than independence and is what the blocking buys; nothing "
            "assumes independent errors across plates")}
    kind = (run.regression_type or "").strip().lower()
    clustering = (
        "plateID is a random intercept, so within-plate correlation is "
        "modelled rather than assumed away" if kind == "mixed" else
        "wells within a plate share reagents, imaging and handling; this "
        "fixed-effects fit ASSUMES independent errors and models position "
        "with rowID / columnID terms rather than a plate random effect — "
        "regression_type='mixed' is the fit that relaxes it")
    dw = run.metrics.get("durbin_watson")
    if dw is None:
        return {"value": f"{clustering}. Durbin-Watson: "
                         f"{_no_residual_reason(run)}"}
    dw = float(dw)
    reading = ("no first-order autocorrelation" if 1.5 <= dw <= 2.5 else
               "positive autocorrelation in row order" if dw < 1.5 else
               "negative autocorrelation in row order")
    return {"value": (f"Durbin-Watson = {dw:.3g} (2 is none) — {reading}. "
                      f"{clustering}")}


def _influence(run: "_Run") -> Dict[str, str]:
    if run.nonparametric:
        return {"kind": NOT_ASSUMED, "reason": (
            "no observation has leverage on a coefficient that is never "
            "fitted; one extreme well changes a marginal correlation and its "
            "permutation null together, which is what makes the test robust "
            "to it")}
    if run.model is None:
        return {"reason": _no_residual_reason(run)}
    try:
        from .regression_qc import PanelUnavailable, context_from_model, \
            cooks_distance

        ctx = context_from_model(run.model,
                                 regression_type=run.regression_type)
    except Exception as error:                                   # noqa: BLE001
        return {"reason": (
            f"leverage could not be recovered from this fit "
            f"({type(error).__name__}: {error})")}
    del PanelUnavailable
    leverage = np.asarray(ctx.leverage, dtype=float)
    finite = leverage[np.isfinite(leverage)]
    if not finite.size:
        return {"reason": ("this fit exposes no hat-matrix diagonal, so "
                           "leverage and Cook's distance are undefined")}
    guide = 2.0 * ctx.p / ctx.n if ctx.n else float("nan")
    high = int((finite > guide).sum()) if np.isfinite(guide) else 0
    run.diagnostics["max_leverage"] = float(finite.max())
    parts = [f"max leverage {finite.max():.3g} against the 2p/n guide of "
             f"{guide:.3g} ({high:,} above it)"]
    if ctx.standardisation_available:
        cooks = cooks_distance(ctx.std_resid, leverage, ctx.p)
        cooks = cooks[np.isfinite(cooks)]
        if cooks.size:
            run.diagnostics["max_cooks_distance"] = float(cooks.max())
            rule = 4.0 / ctx.n if ctx.n else float("nan")
            parts.append(f"max Cook's D {cooks.max():.3g}, {int((cooks > rule).sum()):,} "
                         f"observation(s) above the 4/n rule of {rule:.3g}")
    else:
        reason = getattr(ctx.standardisation, "reason", "") or "unknown"
        parts.append(f"Cook's distance needs a studentised residual and this "
                     f"model class has none ({reason})")
    return {"value": "; ".join(parts)}


def _multicollinearity(run: "_Run") -> Dict[str, str]:
    if run.nonparametric:
        return {"kind": NOT_ASSUMED, "reason": (
            "the guides are never in one design together, so none of them can "
            "inflate another's variance. Two guides that co-occur still give "
            "correlated TESTS, which the shared block design and the common "
            "permutation null already account for")}
    parts = []
    vif = run.metrics.get("max_vif")
    if vif is not None:
        above = _count(run.metrics.get("n_vif_above_10")) or 0
        parts.append(f"max VIF {float(vif):.3g}, {above:,} above 10")
    else:
        parts.append("VIF is not defined here: it is read off the standard "
                     "errors of a full-rank fit with an intercept, and this "
                     "fit is not one")
    # THE BANDS ARE FOR THE SCALED NUMBER, AND ONLY FOR IT.
    # `model.condition_number` is what statsmodels prints, and it is UNSCALED:
    # it is dominated by the units of the columns, so a predictor measured in
    # cells rather than thousands of cells moves it by 1000 with no change in
    # the science. Belsley-Kuh-Welsch's 30 / 100 / 1000 apply to the
    # column-scaled one. Reading the bands off the unscaled number reported
    # "severe collinearity" for a full-rank design with a max VIF of 1.38 on
    # the first real run of this module.
    scaled = None
    inner = getattr(run.model, "model", None)
    exog = getattr(inner, "exog", None) if inner is not None else None
    if exog is not None:
        try:
            from .regression_qc import condition_number

            scaled = float(condition_number(np.asarray(exog, dtype=float))[0])
        except Exception:                                        # noqa: BLE001
            scaled = None
    if scaled is not None:
        try:
            from .regression_qc import condition_verdict

            reading = condition_verdict(scaled)
        except Exception:                                        # noqa: BLE001
            reading = "no reading available"
        parts.append(f"scaled condition number {scaled:.4g} — {reading}")
    else:
        raw_condition = run.metrics.get("condition_number")
        if raw_condition is not None:
            parts.append(
                f"condition number {float(raw_condition):.4g}, UNSCALED as "
                f"statsmodels prints it; the 30 / 100 / 1000 bands apply to "
                f"the column-scaled number and this design could not be "
                f"scaled to give one")
        else:
            parts.append("this estimator reports no condition number")
    pairs = _count(run.metrics.get("n_collinear_pairs"))
    if pairs is not None:
        worst = _number(run.metrics.get("max_abs_predictor_correlation"), 3)
        parts.append(f"{pairs:,} predictor pair(s) correlated at |r| >= 0.95"
                     + (f" (worst {worst})" if worst else ""))
    return {"value": "; ".join(parts)}


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def _tested_mask(run: "_Run") -> Optional[np.ndarray]:
    """Which rows of ``coef_df`` are hypotheses, by the run's own rule.

    THE SAME LINE :func:`spacr.ml._call_level_hits` DREW. The family is the
    guide and gene coefficients: the intercept and the row/column nuisance
    terms are covariates, a row with no P value is not a test, and a mixed
    fit's variance components and BLUPs are neither. Read through
    :func:`spacr.hits.tested_family` so there is one statement of it — a
    second regex is how the summary and the correction come to describe
    different experiments.
    """
    frame = run.coef_df
    feature = _column(frame, "feature")
    if feature is None:
        return None
    try:
        from .hits import tested_family

        mask = np.asarray(tested_family(frame[feature]), dtype=bool)
    except Exception:                                            # noqa: BLE001
        return None
    p_column = _column(frame, "p_value", "permutation_p_value")
    if p_column is not None:
        mask &= frame[p_column].notna().to_numpy()
    if "term_type" in frame.columns:
        mask &= frame["term_type"].eq("fixed").to_numpy()
    return mask


def _cut(run: "_Run") -> Tuple[float, str]:
    """``(alpha, kind)`` a coefficient was actually CALLED at."""
    alpha = float(_setting(run.settings, "fdr_alpha", 0.05) or 0.05)
    cut_alpha = float(_setting(run.settings, "p_threshold_alpha", alpha)
                      or alpha)
    kind = str(_setting(run.settings, "p_threshold_kind", "adjusted")
               or "adjusted").strip().lower()
    return cut_alpha, kind


def _hit_mask(run: "_Run") -> Tuple[Optional[np.ndarray], str]:
    """Which rows this run CALLED, and the sentence saying how."""
    frame = run.coef_df
    if not isinstance(frame, pd.DataFrame) or not len(frame):
        return None, "there is no coefficient table to count"
    cut_alpha, kind = _cut(run)
    if "significant" in frame.columns:
        mask = frame["significant"].astype(bool).to_numpy()
        note = f"corrected P below {cut_alpha:g}"
        if "passes_effect_size" in frame.columns:
            mask = mask & frame["passes_effect_size"].astype(bool).to_numpy()
            note += " AND at least as wide as the effect-size cut"
        return mask, note
    column = "p_value" if kind == "raw" else "q_value"
    if column not in frame.columns:
        selection = _column(frame, "selection_frequency", "selection_freq")
        if selection is not None:
            threshold = float(
                _setting(run.settings, "lasso_selection_threshold", 0.6) or 0.6)
            values = _floats(frame, selection)
            coefficients = _floats(frame, _column(frame, "coefficient") or "")
            mask = np.isfinite(values) & (values >= threshold)
            if coefficients.size == mask.size:
                mask &= coefficients != 0
            return mask, (f"selected in at least {threshold:g} of the "
                          f"bootstrap resamples with a non-zero coefficient — "
                          f"NOT a P value")
        return None, (f"the table carries no {column!r} column, so nothing "
                      f"can be counted as called")
    values = _floats(frame, column)
    mask = np.isfinite(values) & (values < cut_alpha)
    if kind == "raw":
        return mask, f"raw P below {cut_alpha:g}"
    corrected = (_method(run) or "").strip().lower() not in ("", "none")
    return mask, (f"q value below {cut_alpha:g}"
                  + ("" if corrected else
                     " (with no correction applied, that q value IS the raw "
                     "P)"))


def _call_section(run: "_Run") -> List[SummaryField]:
    out, frame = [], run.coef_df

    def add(name, value=None, reason=None, kind=""):
        """Append a labelled field to the hit-calling section."""
        out.append(SummaryField(name, LABELS[("call", name)],
                                value=value, reason=reason, kind=kind))

    method = _method(run)
    if run.penalised and _column(frame, "q_value") is None:
        add("multiple_testing_method",
            value=f"none — {run.regression_type} has no valid frequentist P "
                  f"value (the one attached to a penalised coefficient is "
                  f"OLS-style and ignores the penalty), so nothing was "
                  f"corrected; features are selected by bootstrap frequency")
    elif method == "none":
        add("multiple_testing_method",
            value="none — NO correction was applied, so the q values ARE the "
                  "raw P values. Over a screen-sized family that is a false "
                  "call for every 1/alpha tests from noise alone")
    elif method:
        label = method
        try:
            from .multiple_testing import method_label

            label = method_label(method)
        except Exception:                                        # noqa: BLE001
            pass
        add("multiple_testing_method",
            value=f"{method}" + (f" ({label})" if label != method else ""))
    else:
        add("multiple_testing_method",
            reason="neither the results table nor the settings record which "
                   "correction was applied")

    cut_alpha, kind = _cut(run)
    alpha = float(_setting(run.settings, "fdr_alpha", 0.05) or 0.05)
    text = f"{alpha:g} (fdr_alpha, the level the correction targets)"
    if kind == "raw" or cut_alpha != alpha:
        text += (f"; hits were CALLED on the {kind} P at {cut_alpha:g}"
                 + (", NOT corrected for multiple testing" if kind == "raw"
                    else ""))
    add("fdr_alpha", value=text)

    tested = _tested_mask(run)
    if tested is None:
        add("n_tested",
            reason="the coefficient table has no 'feature' column, so the "
                   "tested family cannot be separated from the covariates")
    else:
        rows = len(frame) if isinstance(frame, pd.DataFrame) else 0
        others = rows - int(tested.sum())
        add("n_tested",
            value=f"{int(tested.sum()):,} of {rows:,} rows are hypotheses"
                  + (f"; the other {others:,} are covariates (the intercept "
                     f"and the row / column terms), which are fitted so the "
                     f"real effects are estimated cleanly and are not "
                     f"themselves tested" if others else
                     " — every row of this table is a test"))

    hits, how = _hit_mask(run)
    if hits is None:
        add("n_called", reason=how)
    else:
        add("n_called", value=f"{int(hits.sum()):,} — {how}")

    add("critical_p", **_critical_p(run, tested))

    inflation = _number(run.metrics.get("genomic_inflation"), 4)
    if inflation is None:
        add("genomic_inflation",
            reason="genomic inflation needs at least ten P values in (0, 1] "
                   "and this table does not have them")
    else:
        value = float(run.metrics["genomic_inflation"])
        reading = ("DEFLATED: the tests are conservative; the null is not "
                   "flat and the P values are larger than the evidence "
                   "warrants" if value < 0.9 else
                   "consistent with a calibrated null" if value < 1.1 else
                   "INFLATED: part of what this run called is artefact rather "
                   "than signal")
        add("genomic_inflation", value=f"{inflation} — {reading}")

    add("effect_size_cut", **_effect_size_cut(run))

    for name, key in (("positive_rank", "positive_control_rank"),
                      ("positive_percentile", "positive_control_percentile")):
        value = run.metrics.get(key)
        if value is None:
            named = _clean(_setting(run.settings, "positive_control"))
            add(name, reason=(
                f"the positive control {named!r} has no coefficient in this "
                f"table" if named else
                "no positive_control was named in the settings, so there is "
                "no yardstick to rank against"))
        elif name == "positive_rank":
            total = _count(run.metrics.get("n_ranked"))
            add(name, value=f"{int(value):,}"
                + (f" of {total:,} by raw P" if total else " by raw P"))
        else:
            add(name, value=f"{float(value):.3g} (0 is the top of the list)")
    return out


def _method(run: "_Run") -> Optional[str]:
    """Which correction this run applied, as the run itself recorded it.

    The table first, the settings second: ``_call_level_hits`` writes the
    CANONICAL spelling into every row, so the table is what the q values were
    actually made with and the settings are only what was asked for.
    """
    return (_raw(_one_value(run.coef_df, "multiple_testing_method"))
            or _raw(_setting(run.settings, "multiple_testing_method")))


def _critical_p(run: "_Run", tested) -> Dict[str, str]:
    """The exact largest RAW P the correction calls at this alpha."""
    frame = run.coef_df
    column = _column(frame, "p_value", "permutation_p_value")
    if column is None or tested is None:
        return {"reason": ("the raw P values of the tested family are not on "
                           "this table, so the threshold cannot be located")}
    values = _floats(frame, column)[tested]
    values = values[np.isfinite(values)]
    if not values.size:
        return {"reason": "the tested family carries no finite raw P value"}
    method = _method(run) or "fdr_bh"
    alpha = float(_setting(run.settings, "fdr_alpha", 0.05) or 0.05)
    try:
        from .multiple_testing import critical_p_value

        threshold = critical_p_value(values, method=method, alpha=alpha)
    except Exception as error:                                   # noqa: BLE001
        return {"reason": (f"the critical P could not be computed "
                           f"({type(error).__name__}: {error})")}
    if threshold is None:
        return {"value": (
            f"none — no test was called at alpha={alpha:g} by the {method!r} "
            f"correction, so there is no rank k and no threshold. Drawing one "
            f"at alpha itself would claim a line the procedure never reached")}
    return {"value": (
        f"{threshold:.4g} — every test with a raw P at or below this was "
        f"called by {method} at alpha={alpha:g}. It is NOT alpha, and it is "
        f"far below it")}


def _effect_size_cut(run: "_Run") -> Dict[str, str]:
    frame = run.coef_df
    rule = _clean(_one_value(frame, "effect_size_rule"))
    threshold = _one_value(frame, "effect_size_threshold")
    number = None
    try:
        number = float(threshold)
    except (TypeError, ValueError):
        number = None
    if number is not None and np.isfinite(number) and number:
        return {"value": f"|coefficient| >= {number:.4g}"
                         + (f" — {rule}" if rule else "")}
    if rule:
        return {"value": f"none — {rule}"}
    if _clean(_setting(run.settings, "controls")) is None:
        return {"value": ("none — no control gRNAs were named, so there is "
                          "nothing to calibrate a width against and a hit is "
                          "the corrected P value alone")}
    return {"reason": ("this table records no effect-size threshold and no "
                       "rule, so what width was required cannot be said")}


# ---------------------------------------------------------------------------
# What was excluded
# ---------------------------------------------------------------------------

#: Said once, for the three filters whose drop counts the run prints and does
#: not record. NAMED rather than left as a blank, because "0 rows removed" and
#: "nobody counted" are opposite findings and a summary must not spell them
#: the same way.
def _exclusion_count(settings, key):
    """What a filter recorded dropping, or ``None`` when nothing recorded it.

    ``None`` IS NOT ZERO and the caller must not spell them the same way: "no
    row was dropped" and "nobody counted" are opposite findings, and a summary
    that reported the second as the first would understate what the fit was
    given.
    """
    try:
        recorded = _setting(settings, "_regression_exclusions") or {}
        value = recorded.get(key)
    except Exception:                                            # noqa: BLE001
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_NOT_RECORDED = ("the run printed how many it removed and did not record it, "
                 "so the count is in the console log and not in any file "
                 "this summary can read")


def _excluded_section(run: "_Run") -> List[SummaryField]:
    out, settings = [], run.settings

    def add(name, value=None, reason=None, kind=""):
        """Append a labelled field to the exclusions section."""
        out.append(SummaryField(name, LABELS[("excluded", name)],
                                value=value, reason=reason, kind=kind))

    minimum = _count(_setting(settings, "min_cell_count"))
    if minimum is None:
        add("min_cell_count",
            value="not set, so no well was dropped for having too few objects")
    else:
        add("min_cell_count",
            value=f"wells with fewer than {minimum:,} objects were dropped "
                  f"before aggregation; {_NOT_RECORDED}")

    # A KNOWN CONTAMINANT MUST LEAVE BEFORE THE FRACTION DENOMINATOR. Merely
    # echoing the setting cannot establish that it matched anything, and a
    # misspelling that removed zero rows is exactly the failure this audit is
    # meant to expose. ``process_reads`` records both the resolved guide names
    # and unmatched requests at the raw-count boundary, before well totals.
    requested = _setting(settings, "exclude_grnas")
    if not requested:
        add("exclude_grnas",
            value="not set, so no guide was removed before well totals")
    else:
        recorded = _setting(settings, "_regression_exclusions") or {}
        dropped = _exclusion_count(settings, "exclude_grnas")
        outof = _exclusion_count(settings, "exclude_grnas_of")
        resolved = [str(value) for value in
                    (recorded.get("exclude_grnas_guides") or [])]
        unmatched = [str(value) for value in
                     (recorded.get("exclude_grnas_unmatched") or [])]
        if dropped is None:
            add("exclude_grnas",
                value=(f"{requested!r} was requested, but this run predates "
                       "the raw-count exclusion audit; whether it matched "
                       "and left before well totals cannot be verified"))
        else:
            denominator = (f" of {outof:,}" if outof is not None else "")
            guide_text = (", ".join(resolved[:8])
                          + (" ..." if len(resolved) > 8 else ""))
            value = (f"{dropped:,}{denominator} raw count rows spanning "
                     f"{len(resolved):,} guide(s) were removed before well "
                     f"totals and fractions")
            if guide_text:
                value += f": {guide_text}"
            if unmatched:
                missing = (", ".join(unmatched[:8])
                           + (" ..." if len(unmatched) > 8 else ""))
                value += (f" -- warning: {len(unmatched):,} requested "
                          f"identifier(s) matched nothing: {missing}")
            add("exclude_grnas", value=value)

    fraction = _setting(settings, "fraction_threshold")
    if fraction is None:
        add("fraction_threshold",
            value="not set, so no gRNA row was dropped for a low well "
                  "fraction")
    else:
        # RECORDED SINCE 2026-08-19. `ml.process_reads` takes a `record=` dict
        # and accumulates what it dropped, per plate, into
        # `settings['_regression_exclusions']` -- so this is a number now
        # rather than an admission. The admission is kept for the runs that
        # predate the recorder, because "0 removed" and "nobody counted" are
        # opposite findings and must not be spelled the same way.
        dropped = _exclusion_count(settings, "fraction_threshold")
        outof = _exclusion_count(settings, "fraction_threshold_of")
        if dropped is None:
            add("fraction_threshold",
                value=f"gRNA rows below a well fraction of "
                      f"{float(fraction):g} were dropped; {_NOT_RECORDED}")
        elif outof:
            # Report both counts and percentage because the percentage makes
            # severe filtering directly comparable across datasets.
            retained = outof - dropped
            share = 100.0 * retained / outof if outof else 0.0
            flag = (
                f" -- warning: fewer than {_LOW_RETENTION_PERCENT:g}% of "
                "gRNA rows were retained. "
                "Review fraction_threshold and its calibration diagnostics "
                "before interpreting the regression results."
                if share < _LOW_RETENTION_PERCENT else ""
            )
            add("fraction_threshold",
                value=f"{dropped:,} of {outof:,} gRNA rows were below a well "
                      f"fraction of {float(fraction):g} and were dropped, "
                      f"leaving {retained:,} ({share:.1f}% retained){flag}")
        else:
            add("fraction_threshold",
                value=f"{dropped:,} gRNA rows were below a well fraction of "
                      f"{float(fraction):g} and were dropped")
    # Pairing counts are recorded at the score/count join and persist in the
    # settings saved with the run.
    paired = _exclusion_count(settings, "wells_paired")
    if paired is None:
        add("missing_metadata",
            value=f"the score and count tables are joined on the well, so a "
                  f"well present in only one of them takes no part in the "
                  f"fit; {_NOT_RECORDED}")
    else:
        lost_counts = _exclusion_count(settings, "wells_unpaired_counts") or 0
        lost_scores = _exclusion_count(settings, "wells_unpaired_scores") or 0
        smaller_input = paired + min(lost_counts, lost_scores)
        share = 100.0 * paired / smaller_input if smaller_input else 100.0
        flag = (
            "" if share >= _LOW_PAIRING_PERCENT else
            " -- warning: fewer than half of the wells in the smaller input "
            "had a matching identifier. Verify plate and well identifiers "
            "before interpreting the regression results."
        )
        paired_label = "well" if paired == 1 else "wells"
        paired_verb = "was" if paired == 1 else "were"
        count_label = "well" if lost_counts == 1 else "wells"
        score_label = "well" if lost_scores == 1 else "wells"
        add("missing_metadata",
            value=f"{paired:,} {paired_label} {paired_verb} matched by well "
                  f"identifier; {lost_counts:,} count-table {count_label} "
                  f"and {lost_scores:,} score-table {score_label} had no "
                  f"match and were excluded before fitting ({share:.0f}% "
                  f"of wells in the smaller input paired){flag}")

    counts = _design_counts(run.data)
    rows = counts.get("n_rows_fitted")
    observations = _count(run.metrics.get("n_observations"))
    if observations is None:
        observations = _count(getattr(run.model, "nobs", None))
    if rows is None:
        add("rows_not_fitted", reason=run.data_note)
    elif run.nonparametric:
        wells = counts.get("n_wells")
        add("rows_not_fitted",
            value=f"{rows:,} (well, guide) rows became {wells:,} wells; the "
                  f"permutation test is a test about wells, so every guide's "
                  f"test runs over all of them"
            if wells is not None else
            f"{rows:,} rows reached the permutation test")
    elif observations is None:
        add("rows_not_fitted",
            value=f"{rows:,} rows reached the fit; what the estimator kept of "
                  f"them cannot be read back off it")
    else:
        dropped = rows - observations
        add("rows_not_fitted",
            value=f"{dropped:,} — {rows:,} rows in regression_data.csv "
                  f"against {observations:,} observations in the fit "
                  f"({'aggregation to wells and non-finite rows' if dropped > 0 else 'nothing was dropped'})")

    tested = _tested_mask(run)
    if tested is None or not isinstance(run.coef_df, pd.DataFrame):
        add("untested_coefficients",
            reason="the coefficient table has no 'feature' column, so its "
                   "rows cannot be split into tests and covariates")
    else:
        others = int((~tested).sum())
        add("untested_coefficients",
            value=f"{others:,} of {len(run.coef_df):,} rows are not "
                  f"hypotheses — the intercept, the rowID / columnID "
                  f"covariates, and any row without a P value (a mixed fit's "
                  f"variance components and its guide BLUPs). Every one keeps "
                  f"its row in results.csv" if others else
                  f"none — all {len(run.coef_df):,} rows of this table are "
                  f"tests")
    add("below_effect_size", **_below_effect_size(run))
    return out


def _below_effect_size(run: "_Run") -> Dict[str, str]:
    frame = run.coef_df
    if not isinstance(frame, pd.DataFrame) or not len(frame):
        return {"reason": "there is no coefficient table to count"}
    if "passes_effect_size" in frame.columns:
        called = frame["significant"].astype(bool).to_numpy() \
            if "significant" in frame.columns \
            else np.ones(len(frame), dtype=bool)
        wide = frame["passes_effect_size"].astype(bool).to_numpy()
        cut = int((called & ~wide).sum())
        return {"value": f"{cut:,} passed the correction and were then cut "
                         f"for being narrower than the effect-size threshold"}
    threshold = _one_value(frame, "effect_size_threshold")
    try:
        number = float(threshold)
    except (TypeError, ValueError):
        number = None
    coefficients = _floats(frame, _column(frame, "coefficient") or "")
    cut_alpha, kind = _cut(run)
    column = "p_value" if kind == "raw" else "q_value"
    if number is None or not np.isfinite(number) or not number:
        return {"value": "0 — there was no effect-size cut to fail"}
    if column not in frame.columns or coefficients.size != len(frame):
        return {"reason": (f"the table has no {column!r} beside its "
                           f"coefficients, so what the width cut removed "
                           f"cannot be recounted")}
    called = _floats(frame, column) < cut_alpha
    cut = int((called & (np.abs(coefficients) < abs(number))).sum())
    return {"value": f"{cut:,} passed the correction and were then cut for "
                     f"being narrower than {abs(number):.4g}"}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_BUILDERS = {
    "fitted": _fitted_section,
    "design": _design_section,
    "fit_quality": _fit_quality_section,
    "assumptions": _assumptions_section,
    "call": _call_section,
    "excluded": _excluded_section,
}


def build_run_summary(*, model=None, settings=None, coef_df=None,
                      regression_type=None, res_folder=None) -> RunSummary:
    """spaCR's own summary of one run, in the same shape for every mode.

    THE CONTRACT IS THE RETURN VALUE. Whatever the regression type and
    whichever inference was used, the result carries every name in
    :data:`CONTRACT` and each one is answered — with a number, or with a
    sentence saying why this mode cannot have one.
    :meth:`RunSummary.missing` is empty on every well-formed summary, and a
    builder that raises is backfilled with the exception rather than allowed
    to leave a hole: reporting is not worth losing a field over, and a hole is
    the failure this module exists to prevent.

    :param model: the fitted model, or ``None`` — which is the ordinary case
        for the permutation path and for the sklearn-backed penalised fits,
        not an error.
    :param settings: the run's settings dict.
    :param coef_df: the corrected coefficient table the run wrote.
    :param regression_type: the family actually fitted, when the caller knows
        it (``regression_type=None`` is auto-selected during the run, so the
        settings may still say ``None``).
    :param res_folder: the run folder. Read for ``regression_data.csv``, which
        is the table that reached the fit and therefore the honest source for
        the well / guide / gene / cell counts.
    :returns: :class:`RunSummary`.
    """
    settings = settings if isinstance(settings, Mapping) else {}
    frame = coef_df if isinstance(coef_df, pd.DataFrame) else None
    kind = _clean(regression_type) or _clean(settings.get("regression_type"))
    data, note = _read_fitted_table(res_folder)
    run = _Run(
        res_folder=res_folder, model=model, settings=settings, coef_df=frame,
        regression_type=kind,
        nonparametric=_is_nonparametric(settings, frame),
        penalised=(kind or "").strip().lower() in _penalised_types(),
        data=data, data_note=note,
        metrics=_collect_metrics(model, frame, settings),
    )

    sections: List[SummarySection] = []
    for name, title in SECTIONS:
        section = SummarySection(name=name, title=title)
        try:
            section.fields = list(_BUILDERS[name](run))
        except Exception as error:                               # noqa: BLE001
            section.fields = []
            note_error = (f"this section could not be built "
                          f"({type(error).__name__}: {error})")
            for field_name in CONTRACT[name]:
                section.fields.append(SummaryField(
                    field_name, LABELS[(name, field_name)],
                    reason=note_error))
        # THE BACKFILL IS THE CONTRACT'S LAST LINE OF DEFENCE. A builder that
        # returns early, or a field added to CONTRACT and not yet to its
        # builder, would otherwise ship a summary with a silent hole -- which
        # is exactly the failure mode this item is about.
        present = {one.name for one in section.fields}
        for field_name in CONTRACT[name]:
            if field_name not in present:
                section.fields.append(SummaryField(
                    field_name, LABELS[(name, field_name)],
                    reason="this build of spaCR did not fill the field in; "
                           "that is a defect in the summary, not a property "
                           "of the run"))
        order = {one: index for index, one in enumerate(CONTRACT[name])}
        section.fields.sort(key=lambda one: order.get(one.name, len(order)))
        sections.append(section)

    verbatim, verbatim_note = _verbatim(run)
    return RunSummary(sections=sections, warnings=_warnings(run),
                      verbatim=verbatim, verbatim_note=verbatim_note,
                      recommendations=_recommendations(run))


def _recommendations(run: "_Run") -> List[Any]:
    """What to change, derived from what the sections above just measured.

    AFTER THE SECTIONS, NEVER BESIDE THEM. The builders deposit into
    ``run.diagnostics`` as they format their lines, so by the time this runs
    every number here is one that appears on screen. Recomputing them would
    have been fewer lines and would have allowed a recommendation to cite a
    figure the summary does not show -- which is the one failure a
    recommendations section cannot recover from, because the reader has no
    way to tell which of the two is wrong.

    ``run.metrics`` supplies what ``trial_metrics`` already measured
    (Durbin-Watson, VIF); ``run.diagnostics`` supplies what the assumption
    builders measured themselves. The deposits win where both have a key:
    they are the ones that were written down.
    """
    if run.nonparametric:
        # A permutation run assumes none of this, and the sections above say
        # so five times over. Recommending a fix for an assumption that was
        # never made is how the section loses the reader.
        return []
    measured: Dict[str, Any] = dict(run.metrics)
    measured.update(run.diagnostics)
    measured.setdefault("median_wells_per_guide", _median_wells(run))
    try:
        from .run_recommendations import recommend

        return list(recommend(measured, settings=run.settings))
    except Exception:                                            # noqa: BLE001
        # A summary is worth more than its last section: a run that reached
        # here has numbers worth reading, and losing them to a failure in the
        # advice would be the wrong trade.
        return []


def _median_wells(run: "_Run") -> Optional[float]:
    """The median number of wells the surviving guides are in.

    READ OFF THE FITTED TABLE, which is the table that reached the fit --
    so it reflects `fraction_threshold` as applied, not as configured.
    """
    data = run.data
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None
    guide = next((c for c in ("grna", "grna_name", "gRNA")
                  if c in data.columns), None)
    well = next((c for c in ("prc", "well", "wellID")
                 if c in data.columns), None)
    if guide is None or well is None:
        return None
    try:
        return float(data.groupby(guide)[well].nunique().median())
    except Exception:                                            # noqa: BLE001
        return None


def _verbatim(run: "_Run") -> Tuple[Optional[str], str]:
    """The statsmodels summary, unchanged, and what it is.

    VERBATIM AND AT THE END, never instead of the above. The point of asking
    for the statsmodels summary is to get the statsmodels summary; this module
    adds a summary for the modes that have none and takes nothing away from
    the two that do.
    """
    model = run.model
    if model is None:
        if run.nonparametric:
            return None, ("There is no statsmodels summary: the permutation "
                          "test fits no model. Everything above is what this "
                          "run can be summarised by.")
        return None, ("There is no statsmodels summary: this run handed over "
                      "no fitted model.")
    summary = getattr(model, "summary", None)
    if not callable(summary):
        return None, (
            f"There is no statsmodels summary: this backend "
            f"({type(model).__name__}) is not a statsmodels fit. The "
            f"sklearn-backed types report coefficients without standard "
            f"errors, which is why they are ranked by bootstrap selection "
            f"frequency instead.")
    try:
        text = str(summary())
    except Exception as error:                                   # noqa: BLE001
        return None, (f"statsmodels could not render its summary for this fit "
                      f"({type(error).__name__}: {error}).")
    return text, "The statsmodels summary, unchanged:"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _wrap(label: str, text: str) -> List[str]:
    lead = f"  {label:<{_LABEL_WIDTH}}"
    body = textwrap.wrap(text, width=_WIDTH - len(lead)) or [""]
    lines = [lead + body[0]]
    lines.extend(" " * len(lead) + more for more in body[1:])
    return lines



#: Headline fields shown first in the written summary: fitted method, design
#: size, call count, threshold, and positive-control rank.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("fitted", "regression type"),
    ("fitted", "inference"),
    ("design", "wells"),
    ("call", "coefficients called"),
    ("call", "alpha"),
    ("call", "positive control rank"),
)

#: How a field says it does not apply. Nine of these in a row, with a
#: paragraph each, is the bulk of what makes a permutation summary hard to
#: scan -- and they all say the same thing: this run fitted no model.
_NOT_APPLICABLE = "not applicable"


def headline(summary: "RunSummary") -> List["SummaryField"]:
    """The six lines a reader can stop after.

    NOT A SUMMARY OF THE SUMMARY. Every line here is one of the fields below,
    quoted verbatim -- so there is no second place for a number to be wrong,
    and nothing can drift between the headline and the section it came from.
    """
    out: List[SummaryField] = []
    for section_name, field_name in HEADLINE:
        section = summary.section(section_name)
        if section is None:
            continue
        for one in section.fields:
            if one.label == field_name:
                out.append(one)
                break
    return out


def _split_not_applicable(fields):
    """``(shown, deferred)`` -- the fields that say something, and the rest.

    A field whose text begins "not applicable" is deferred: named in one line
    where it stood, with its full explanation kept under a heading of its own.
    NOTHING IS DELETED -- the reasoning is what makes a permutation run
    legible at all, and a summary that dropped it to be shorter would have
    traded away the thing worth reading.
    """
    shown, deferred = [], []
    for one in fields:
        text = str(one.text or "").strip().lower()
        (deferred if text.startswith(_NOT_APPLICABLE) else shown).append(one)
    return shown, deferred

def format_run_summary(summary: RunSummary) -> str:
    """Render a :class:`RunSummary` as the text written to the run folder.

    THE WARNING GOES FIRST, where the Summary tab already puts it: a
    rank-deficient fit prints a full table of standard errors regardless, and
    it looks exactly like a summary of a well-posed one.

    :param summary: what :func:`build_run_summary` returned.
    :returns: the whole summary as text.
    """
    lines: List[str] = ["spaCR RUN SUMMARY", "=" * 17, ""]
    for warning in summary.warnings:
        for paragraph in str(warning).splitlines():
            lines.extend(textwrap.wrap(paragraph, width=_WIDTH) or [""])
        lines.append("")

    # THE ANSWER FIRST, and every line of it quoted verbatim from below.
    top = headline(summary)
    if top:
        lines.append("THE ANSWER")
        lines.append("-" * len("THE ANSWER"))
        for one in top:
            lines.extend(_wrap(one.label, one.text))
        lines.append("")
        lines.append("Everything below is how that was arrived at.")
        lines.append("")

    postponed: List[Tuple[str, "SummaryField"]] = []
    for section in summary.sections:
        lines.append(section.title)
        lines.append("-" * len(section.title))
        shown, deferred = _split_not_applicable(section.fields)
        for one in shown:
            lines.extend(_wrap(one.label, one.text))
        if deferred:
            # ONE LINE WHERE NINE STOOD, naming them, with every word of the
            # explanation still in the file under its own heading.
            names = ", ".join(one.label for one in deferred)
            lines.extend(_wrap(
                "not applicable here",
                f"{len(deferred)} field(s) — {names}. Why each does not apply "
                f"is under NOT APPLICABLE, AND WHY at the end."))
            postponed.extend((section.title, one) for one in deferred)
        lines.append("")

    if postponed:
        lines.append("NOT APPLICABLE, AND WHY")
        lines.append("-" * len("NOT APPLICABLE, AND WHY"))
        # ONE EXPLANATION PER REASON, NOT PER FIELD. Measured on the
        # maintainer's own run: eleven deferred fields carry SIX distinct
        # explanations, two of them printed three times each -- six paragraphs
        # where two would do, and that repetition is most of what "not very
        # accessable" was about. The fields sharing a reason are named
        # together and the reason is given once.
        grouped: "OrderedDict[str, List[str]]" = OrderedDict()
        for _title, one in postponed:
            grouped.setdefault(str(one.text), []).append(one.label)
        for text, labels in grouped.items():
            joined = ", ".join(labels)
            if len(joined) <= _LABEL_WIDTH:
                lines.extend(_wrap(joined, text))
                continue
            # A JOINED LABEL LONGER THAN THE COLUMN gets its own line, or the
            # explanation is squeezed into whatever is left and comes out one
            # word wide.
            lines.extend(textwrap.wrap(joined + ":", width=_WIDTH - 2,
                                       initial_indent="  ",
                                       subsequent_indent="  ") or [""])
            lines.extend(textwrap.wrap(text, width=_WIDTH - 6,
                                       initial_indent="      ",
                                       subsequent_indent="      ") or [""])
        lines.append("")
    note = summary.verbatim_note or ""
    if note:
        lines.append("THE STATSMODELS SUMMARY")
        lines.append("-" * len("THE STATSMODELS SUMMARY"))
        lines.extend(textwrap.wrap(note, width=_WIDTH) or [""])
        lines.append("")
    if summary.verbatim:
        lines.append(summary.verbatim)
        lines.append("")

    # LAST, BECAUSE IT IS WHAT TO DO NEXT. Everything above says what was
    # found; this says what to change, and a reader who stops early has
    # still read the findings.
    #
    # It is printed even when empty: an absent section reads as a bug, and
    # "every check passed" is a result worth stating rather than implying by
    # silence.
    try:
        from .run_recommendations import format_recommendations

        lines.append(format_recommendations(
            list(summary.recommendations or [])))
        lines.append("")
    except Exception:                                        # noqa: BLE001
        pass
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Writing it with the run
# ---------------------------------------------------------------------------


def _summary_filename() -> str:
    """The name a run's summary is written under.

    Asked of :mod:`spacr.ml`, which is the module that owns the name and the
    list of older ones the readers accept -- one vocabulary, so a rename there
    cannot leave this writing a file nobody looks for.
    """
    try:
        from .ml import SUMMARY_FILENAME

        return str(SUMMARY_FILENAME)
    except Exception:                                            # noqa: BLE001
        return _FALLBACK_SUMMARY_FILENAME


def model_identity_line(regression_type, settings=None, model=None,
                        backend=None) -> str:
    """Return a compact description of the model used for a figure.

    Parameters
    ----------
    regression_type : str or None
        Regression family. When omitted, the value is read from ``settings``.
    settings : mapping, optional
        Run settings used to resolve the backend and relevant hyperparameters.
    model : object, optional
        Fitted estimator. A cross-validated ``alpha_`` value is reported when
        the estimator provides one.
    backend : str or None, optional
        Explicit backend name, overriding the value in ``settings``.

    Returns
    -------
    str
        Model family, backend, and applicable hyperparameters, or an empty
        string when the regression family is unknown.

    Notes
    -----
    Related regression families can produce identical coefficients while
    differing in uncertainty estimates. Displaying the model identity keeps
    visually similar figures attributable to the fit that generated them.
    """
    kind = _clean(regression_type) or _clean(_setting(settings or {},
                                                      "regression_type"))
    if not kind:
        return ""
    parts = [f"model: {kind}"]
    where = _clean(backend) or _clean(_setting(settings or {}, "backend"))
    if where:
        parts.append(f"backend: {where}")

    class _Holder:                       # what _hyperparameter_report reads
        nonparametric = False

    holder = _Holder()
    holder.model = model
    knobs = _hyperparameter_report(kind, settings or {}, holder)
    said = knobs.get("value") or knobs.get("reason") or ""
    if said and not said.startswith("none"):
        parts.append(said)
    return " · ".join(parts)


def _hyperparameter_report(kind, settings, run) -> dict:
    """Describe the hyperparameters used by a fitted regression.

    The report includes only settings consumed by the selected regression
    family. When cross-validation selected ``alpha``, the fitted value is
    reported instead of the input value ``"auto"`` whenever it is available.
    """
    try:
        from .regression_spec import REGRESSION_SETTINGS_USED
    except Exception:                                        # noqa: BLE001
        return {"reason": "the family table could not be read, so the "
                          "hyperparameters this type uses cannot be listed"}
    wanted = sorted(REGRESSION_SETTINGS_USED.get(str(kind).lower(), ()))
    # `cov_type` is not a hyperparameter of the fit -- it is how the standard
    # errors are computed afterwards -- and it has its own line already.
    wanted = [name for name in wanted if name != "cov_type"]
    if not wanted:
        return {"value": f"none — {kind} reads no hyperparameter"}

    # THE MODEL KNOWS. `_find_best_alpha` returns the fitted RidgeCV /
    # LassoCV / ElasticNetCV itself, and those carry the alpha they chose as
    # `alpha_` -- so the value that won is on the object the run already
    # holds, and does not need recording separately.
    chosen = getattr(getattr(run, "model", None), "alpha_", None)
    try:
        chosen = None if chosen is None else float(chosen)
    except (TypeError, ValueError):
        chosen = None
    parts = []
    for name in wanted:
        value = _setting(settings, name, None)
        if name == "alpha" and str(value).strip().lower() in ("auto", "none",
                                                              ""):
            if chosen is not None:
                parts.append(f"alpha={chosen:.6g} (cross-validated, not given)")
            else:
                parts.append("alpha=auto — cross-validated, and this run did "
                             "not record the value that won")
            continue
        if value is None or str(value).strip() == "":
            parts.append(f"{name}=(not recorded)")
        else:
            parts.append(f"{name}={value}")
    return {"value": ", ".join(parts)}


def write_run_summary(res_folder, *, model=None, settings=None, coef_df=None,
                      regression_type=None) -> Optional[str]:
    """Write this run's spaCR summary into its own folder, and return the path.

    CALLED ON EVERY RUN, for every supported regression type and
    both inferences. It writes the file
    :func:`spacr.qt.widgets.regression_results.find_summary_file` already
    reads back, so a run re-opened from disk shows this summary with no GUI
    change (the design taught the panel to read the run folder;
    the design is about there being something worth reading in it).

    THE STATSMODELS SUMMARY IS PRESERVED. ``ols`` and ``beta`` runs have
    already written the statsmodels text into this same file by the time this
    is called; it is rendered again from the model where it can be, and
    otherwise read back off the file, and either way it is appended UNCHANGED
    at the end rather than replaced. The whole text is built in memory first,
    so a failure here leaves the existing file exactly as it was.

    :param res_folder: the run folder, beside ``results.csv``.
    :param model: the fitted model, or ``None``.
    :param settings: the run's settings dict.
    :param coef_df: the corrected coefficient table.
    :param regression_type: the family actually fitted.
    :returns: the path written, or ``None`` when there was no folder to write
        into.
    """
    if not res_folder:
        return None
    folder = os.path.abspath(os.path.expanduser(os.fspath(res_folder)))
    summary = build_run_summary(model=model, settings=settings,
                                coef_df=coef_df,
                                regression_type=regression_type,
                                res_folder=folder)
    path = os.path.join(folder, _summary_filename())
    if summary.verbatim is None and os.path.isfile(path):
        # THE RUN'S OWN STATSMODELS TEXT, RECOVERED RATHER THAN LOST. It was
        # written into this path minutes ago by `save_summary_to_file`, and a
        # model that cannot render a second time (or was not handed over)
        # would otherwise silently drop it on the way past.
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                existing = handle.read().strip()
        except OSError:
            existing = ""
        if existing and not existing.startswith("spaCR RUN SUMMARY"):
            summary.verbatim = existing
            summary.verbatim_note = (
                "The statsmodels summary this run wrote, unchanged:")
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(format_run_summary(summary))
    return path
