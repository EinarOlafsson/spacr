"""Compare measurements across annotated gene groups and remaining cells.

Comparisons can use cells, wells, or plates as observations. Statistical test
selection is delegated to :mod:`spacr.sp_stats`, and each result records its
normality, equal-variance, and sample-size evidence. Cell-level tests treat
individual cells as observations and do not account for cells that share a
well; use the well level when wells are the experimental units.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# THE HOUSE STYLE (136). `figures.style` imports matplotlib
# only inside its own functions, so naming it here costs
# nothing at import time.
from .figures.style import figure_style, theme_target
from .style_base import SHARED_CHOICES, FigureStyle

#: Explanations for the controls in the measurement-comparison panel.
#: Entries are keyed by control field rather than by the displayed heading.
HEADING_HELP: dict = {
    "measurement": (
        "Select the measurement to plot. Choices are offered from columns "
        "available in the attached tables. An optional arithmetic operator can combine it "
        "with a second measurement; the resulting expression is recorded in "
        "the plot, legend, and saved settings."),
    "level": (
        "Define what one datapoint represents. 'cell' retains individual "
        "objects; 'well' averages measurements within each well; 'plate' "
        "averages within each plate. Screens are usually randomised at the "
        "well level, so using cells as independent observations can produce "
        "pseudoreplication."),
    "plot": (
        "Choose how the same values are displayed: box, violin, bar, jitter, "
        "or jitter over a box. This setting changes nothing about the values "
        "or statistical test; it changes only their visualization."),
    "show": (
        "Restrict the drawing to one class without changing the statistical "
        "comparison. Reported statistics continue to describe the whole "
        "comparison rather than only the visible class."),
    "compare": (
        "Select the reference population: none, other objects in the same "
        "well, control wells, or other wells on the plate. The selected "
        "contrast determines the comparison and its p-value."),
}

LEVELS: Tuple[Tuple[str, str], ...] = (
    ("cell", "one row per cell — the most rows and the fewest independent "
             "units, so a p-value here is about cells and not about wells"),
    ("well", "one row per well — the unit the screen randomises, and the "
             "honest default for a well-level design"),
    ("plate", "one row per plate — very few rows, and the only level that "
              "removes plate as a confounder entirely"),
)

#: Supported plot types and their user-facing descriptions.
PLOTS: Tuple[Tuple[str, str], ...] = (
    ("jitter_box", "jitter over box — every point, and the summary behind it"),
    ("box", "box"),
    ("jitter", "jitter — every point"),
    ("violin", "violin — the shape of the distribution"),
    ("bar", "bar — mean and error, and the least informative of the five"),
)

#: Label assigned to observations that do not belong to a named gene group.
REST = "the rest"

#: Comparison-group definitions and the confounding each one removes.
#:
#: ``within_well`` controls both plate and well effects because the two groups
#: share acquisition and treatment conditions. ``against_controls`` compares
#: annotated cells with named control wells but does not remove plate effects.
#: ``against_other_wells`` is the broadest comparison and remains exposed to
#: well and plate effects.
#:
#: The empty value preserves the default comparison against every unannotated
#: cell in the loaded data.
CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("", "everything else",
     "the annotated cells against every other cell loaded, wherever it is. "
     "Removes nothing, and mixes wells, controls and plates into one "
     "comparison group."),
    ("within_well", "within the well",
     "the annotated cells against the rest of the SAME well. Removes plate "
     "and well batch entirely: both sides were seeded, treated, stained and "
     "imaged together."),
    ("against_controls", "against the controls",
     "the annotated cells against the cells in the CONTROL wells. Removes "
     "nothing by itself -- it asks whether the gene moves the measurement "
     "away from where an untargeted well sits. The controls have to be "
     "named."),
    ("against_other_wells", "against every other well",
     "the annotated cells against every cell in every OTHER well. The widest "
     "comparison and the most exposed to well batch, because the two sides "
     "share no well."),
)

#: Column names that identify one well, in preference order. ``prc`` is the
#: joined plate/row/column key `process_reads` writes; the three columns are
#: what an object table straight out of a database carries.
WELL_KEY_COLUMNS: Tuple[str, ...] = ("plateID", "rowID", "columnID")


def contrast_note(contrast: str) -> str:
    """Describe the comparison group and confounding for a contrast.

    Parameters
    ----------
    contrast : str
        Contrast value from :data:`CONTRASTS`.

    Returns
    -------
    str
        User-facing explanation of the selected contrast. Unknown values
        return an empty string so older saved runs remain readable.
    """
    for value, _label, why in CONTRASTS:
        if value == str(contrast or ""):
            return why
    return ""


def well_labels(objects: "pd.DataFrame") -> Optional["pd.Series"]:
    """Resolve one well label for each object row.

    Parameters
    ----------
    objects : pandas.DataFrame
        Object table containing ``montage_well``, ``prc``, or the plate, row,
        and column identifiers in :data:`WELL_KEY_COLUMNS`.

    Returns
    -------
    pandas.Series or None
        Well labels aligned to ``objects``. ``montage_well`` is preferred so
        labels match montage captions, followed by ``prc`` and the composite
        plate/row/column key. ``None`` indicates that well identity cannot be
        recovered from the table.
    """
    columns = getattr(objects, "columns", ())
    for single in ("montage_well", "prc"):
        if single in columns:
            return objects[single].astype(str)
    if all(c in columns for c in WELL_KEY_COLUMNS):
        parts = [objects[c].astype(str) for c in WELL_KEY_COLUMNS]
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + "_" + part
        return joined
    return None


def wells_of(objects: "pd.DataFrame",
             groups: Dict[str, Sequence[Any]]) -> Dict[str, Tuple[str, ...]]:
    """List the observed wells represented by each annotated group.

    Parameters
    ----------
    objects : pandas.DataFrame
        Object rows indexed by the values stored in ``groups``.
    groups : dict of str to sequence
        Group names mapped to object-index values.

    Returns
    -------
    dict of str to tuple of str
        Observed well labels in first-occurrence order. Groups with no matching
        rows and wells absent from the object table are omitted.

    Notes
    -----
    Wells are derived from the annotated object rows rather than count data,
    so the result contains only wells represented in the montage.
    """
    wells = well_labels(objects)
    if wells is None:
        return {}
    out: Dict[str, Tuple[str, ...]] = {}
    for name, members in (groups or {}).items():
        picked = objects.index.isin(list(members))
        if not picked.any():
            continue
        seen = wells[picked].astype(str)
        out[str(name)] = tuple(dict.fromkeys(seen.tolist()))
    return out


def control_wells(counts: "pd.DataFrame", typed, *,
                  guide_column: str = "grna",
                  gene_column: str = "gene") -> Tuple[str, ...]:
    """Resolve the wells occupied by named controls in count data.

    Parameters
    ----------
    counts : pandas.DataFrame
        Per-well count table with guide identifiers and recoverable well
        labels.
    typed : str or sequence of str
        Control gene or guide names. Names are resolved through
        :mod:`spacr.control_names`, including supported prefixes.
    guide_column : str, default 'grna'
        Column containing guide identifiers.
    gene_column : str, default 'gene'
        Column containing gene identifiers, when available.

    Returns
    -------
    tuple of str
        Matching well labels in first-occurrence order. An empty tuple is
        returned when the table lacks required columns or no control matches.
    """
    from .control_names import resolve_controls, rows_for

    wells = well_labels(counts)
    if wells is None or guide_column not in getattr(counts, "columns", ()):
        return ()
    guides = counts[guide_column]
    genes = counts[gene_column] if gene_column in counts.columns else None
    names = [str(g) for g in pd.Series(guides).astype(str).unique()]
    # ONE NAME IS A NAME, NOT FOUR LETTERS. `resolve_controls` iterates its
    # argument, so a bare string arrives as a sequence of characters and
    # every one of them resolves to nothing. The control field hands over a
    # list; a caller with one control in hand should not have to know that.
    wanted = [typed] if isinstance(typed, str) else list(typed or ())
    found: List[str] = []
    for spec in resolve_controls(wanted, names=names) or ():
        mask, _said = rows_for(spec.typed, guides, genes, names=names)
        picked = np.asarray(mask, dtype=bool)
        if picked.any():
            found.extend(wells[picked].astype(str).tolist())
    return tuple(dict.fromkeys(found))

#: Arithmetic operators for combining two measurements. The expression, such
#: as ``pathogen_area / cell_area``, becomes the result's display name.
OPERATORS: Tuple[Tuple[str, str], ...] = (
    ("", "on its own"),
    ("+", "plus"),
    ("-", "minus"),
    ("*", "multiplied by"),
    ("/", "divided by"),
)


def combine(objects: "pd.DataFrame", first: str, operator: str,
            second: str) -> Tuple["pd.Series", str, int]:
    """Combine one or two measurement columns.

    :param objects: table containing the measurement columns.
    :param first: name of the first measurement column.
    :param operator: one of ``''``, ``'+'``, ``'-'``, ``'*'``, or ``'/'``.
        An empty string returns ``first`` unchanged.
    :param second: name of the second measurement column. Required when
        ``operator`` is not empty.
    :returns: ``(values, name, dropped)``. ``name`` is the displayed
        expression and ``dropped`` counts zero or non-finite denominators.
    :raises KeyError: if a requested measurement column is absent.
    :raises ValueError: if ``operator`` is unsupported.

    Division by zero or a non-finite denominator produces a missing value;
    it is never converted to zero or infinity.
    """
    left = pd.to_numeric(objects[first], errors="coerce")
    if not operator:
        return left, str(first), 0
    right = pd.to_numeric(objects[second], errors="coerce")
    name = f"{first} {operator} {second}"
    if operator == "+":
        return left + right, name, 0
    if operator == "-":
        return left - right, name, 0
    if operator == "*":
        return left * right, name, 0
    if operator == "/":
        zero = (right == 0) | ~np.isfinite(right)
        values = left.divide(right.where(~zero))
        return values, name, int(zero.sum())
    raise ValueError(
        f"{operator!r} is not one of "
        f"{', '.join(repr(o) for o, _l in OPERATORS if o)}")


@dataclass
class Comparison:
    """Store one grouped measurement comparison.

    :param measurement: label for the measured value or arithmetic expression.
    :param level: observation level used to build ``frame``.
    :param frame: long-form table containing ``group``, ``value``, and the
        observation identifier when available.
    :param statistics: statistical-test records added by
        :func:`with_statistics`.
    :param note: warning or interpretation text that accompanies the result.
    """

    measurement: str
    level: str
    frame: pd.DataFrame            # long: group, value, and the unit id
    statistics: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def groups(self) -> Tuple[str, ...]:
        """Return group names in their first-occurrence order."""
        if not len(self.frame):
            return ()
        return tuple(self.frame["group"].astype(str).unique())

    def counts(self) -> Dict[str, int]:
        """Return the number of observations in each group."""
        if not len(self.frame):
            return {}
        return {str(k): int(v) for k, v in
                self.frame.groupby("group")["value"].size().items()}


def _unit_columns(level: str) -> Tuple[str, ...]:
    """Which columns identify one observation at ``level``."""
    if level == "plate":
        return ("plateID",)
    if level == "well":
        return ("plateID", "rowID", "columnID")
    return ()


def build(objects: pd.DataFrame, measurement: str, *,
          groups: Dict[str, Sequence[Any]],
          level: str = "well",
          value_column: Optional[str] = None,
          operator: str = "",
          second: str = "",
          contrast: str = "",
          wells: Optional[Sequence[str]] = None,
          controls: Optional[Sequence[str]] = None) -> Comparison:
    """Build the long-form table used for plotting and statistical testing.

    :param objects: per-object rows containing the requested measurements.
        Well-level comparisons also require ``plateID``, ``rowID``, and
        ``columnID``; plate-level comparisons require ``plateID``.
    :param measurement: measurement column and default display label.
    :param groups: mapping of group names to object-index values. Objects not
        listed in any group are assigned to :data:`REST`. If an object occurs
        in more than one group, the last matching mapping entry wins.
    :param level: ``'cell'``, ``'well'``, or ``'plate'``. The default is
        ``'well'``.
    :param value_column: source column to read instead of ``measurement``.
    :param operator: optional arithmetic operator from :data:`OPERATORS`.
    :param second: second measurement column used with ``operator``.
    :param contrast: which comparison group to build, from :data:`CONTRASTS`.
        The default ``''`` keeps every unannotated row, wherever it came
        from. The other three restrict it to the same well, to the control
        wells, or to every other well; each is recorded in
        :attr:`Comparison.note` together with what it removes.
    :param wells: the wells of the annotation to INCLUDE. ``None`` includes
        all of them. A well left out is dropped from BOTH sides -- it is not
        promoted into the comparison group, because a well excluded for
        being bad is no more usable as a comparison than as an annotation.
    :param controls: the wells the controls occupy, for the
        ``'against_controls'`` contrast. :func:`control_wells` resolves them
        from the count data.
    :returns: comparison data and any explanation of rows that could not be
        used. Missing measurement or identity columns produce an empty
        comparison with the reason in :attr:`Comparison.note`.

    Well- and plate-level values are means within each observation and group.
    A well containing both a named and an unassigned cell therefore contributes
    one row to each group.
    """
    column = str(value_column or measurement)
    wanted = [column] + ([str(second)] if operator and second else [])
    missing = [c for c in wanted if c not in getattr(objects, "columns", ())]
    if missing:
        return Comparison(
            measurement=measurement, level=level,
            frame=pd.DataFrame(columns=["group", "value"]),
            note=f"{', '.join(repr(c) for c in missing)} is not a column "
                 f"on these objects")

    dropped = 0
    if operator and second:
        values, measurement, dropped = combine(objects, column,
                                               str(operator), str(second))
    else:
        values = pd.to_numeric(objects[column], errors="coerce")
    labels = pd.Series(REST, index=objects.index, dtype=object)
    for name, members in (groups or {}).items():
        picked = objects.index.isin(list(members))
        labels[picked] = str(name)

    # ------------------------------------------------------- the contrast
    # WHICH ROWS ARE "the rest" IS THE QUESTION (187 B). The same annotated
    # cells against the rest of their own well, against the controls, and
    # against every other well are three different experiments, and the
    # default -- everything else, wherever it is -- is the one that mixes
    # all three together.
    contrast = str(contrast or "")
    chosen_note = ""
    keep = pd.Series(True, index=objects.index)
    if contrast or wells is not None:
        where = well_labels(objects)
        if where is None:
            return Comparison(
                measurement=measurement, level=level,
                frame=pd.DataFrame(columns=["group", "value"]),
                note=("these object rows do not say which well they came "
                      "from: none of 'montage_well', 'prc' or "
                      f"{list(WELL_KEY_COLUMNS)} is on them, so a well-based "
                      "contrast cannot be built"))
        annotated = labels.astype(str) != REST
        theirs = set(where[annotated].astype(str))

        # THE CHOSEN WELLS FIRST, because every contrast below is defined
        # against the annotation that is actually being used.
        if wells is not None:
            wanted = {str(w) for w in wells}
            left_out = theirs - wanted
            keep &= ~where.astype(str).isin(left_out)
            annotated = annotated & where.astype(str).isin(wanted)
            if left_out:
                chosen_note = (
                    f"{len(left_out)} annotated well(s) left out: "
                    f"{', '.join(sorted(left_out))}")

        rest = ~annotated
        if contrast == "within_well":
            mine = set(where[annotated].astype(str))
            keep &= annotated | (rest & where.astype(str).isin(mine))
        elif contrast == "against_other_wells":
            # `theirs`, NOT `mine`: a well excluded from the annotation is
            # excluded, full stop. Letting it back in as "another well"
            # would put the very rows the user threw out on the other side
            # of the comparison.
            keep &= annotated | (rest & ~where.astype(str).isin(theirs))
        elif contrast == "against_controls":
            named = {str(w) for w in (controls or ())}
            if not named:
                return Comparison(
                    measurement=measurement, level=level,
                    frame=pd.DataFrame(columns=["group", "value"]),
                    note=("a comparison against the controls needs the "
                          "controls named: no control well was resolved, so "
                          "there is nothing on the other side"))
            keep &= annotated | (rest & where.astype(str).isin(named))
        elif contrast:
            raise ValueError(
                f"{contrast!r} is not one of "
                f"{', '.join(repr(c) for c, _l, _w in CONTRASTS)}")

    if not bool(keep.all()):
        labels = labels[keep]
        values = values[keep]

    work = pd.DataFrame({"group": labels, "value": values})
    keys = [c for c in _unit_columns(level) if c in objects.columns]
    if level != "cell" and not keys:
        return Comparison(
            measurement=measurement, level=level,
            frame=pd.DataFrame(columns=["group", "value"]),
            note=(f"a {level}-level comparison needs "
                  f"{', '.join(_unit_columns(level))} on the object rows, "
                  f"and they are not there"))
    for key in keys:
        # `.loc[work.index]` because the contrast may have dropped rows:
        # assigning the FULL column onto a shorter frame aligns on the index
        # and leaves NaN wherever the two disagree, which then joins every
        # such row into one well named "nan".
        work[key] = objects[key].astype(str).loc[work.index]
    work = work.dropna(subset=["value"])

    # NOTHING LEFT IS AN ANSWER, NOT A CRASH. Every other empty case in this
    # function returns a Comparison carrying the reason, and this one used to
    # raise instead: on an empty frame `agg("_".join, axis=1)` hands back an
    # empty DATAFRAME of the key columns rather than a Series, and assigning
    # that to one column is a ValueError -- "Cannot set a DataFrame with
    # multiple columns to the single column unit". It reached the user as a
    # traceback out of the Cells tab's Compare dialog, from a measurement
    # that simply held no numbers.
    if not len(work):
        return Comparison(
            measurement=measurement, level=level,
            frame=pd.DataFrame(columns=["group", "value"]),
            note=(f"no object carries a numeric {measurement!r}: every value "
                  f"is missing or could not be read as a number"))

    if level == "cell":
        work["unit"] = work.index.astype(str)
    else:
        # ONE ROW PER (unit, group), not per unit: see the docstring.
        work["unit"] = work[keys].agg("_".join, axis=1)
        work = (work.groupby(["unit", "group"], as_index=False)["value"]
                .mean())

    note = ""
    # THE CONTRAST IS NAMED FIRST, ahead of every other caveat, because it
    # decides what the p-value below is a p-value ABOUT. A number from
    # "within the well" and a number from "against every other well" are not
    # comparable, and nothing else on the panel distinguishes them.
    said = contrast_note(contrast) if contrast else ""
    if said:
        label = next((l for c, l, _w in CONTRASTS if c == contrast), contrast)
        note = f"{label}: {said}"
    if chosen_note:
        note = ((note + " · ") if note else "") + chosen_note
    if dropped:
        # SAID, ALWAYS. A comparison quietly computed on fewer rows than the
        # user thinks is the kind of result that survives review and is
        # wrong.
        note = ((note + " · ") if note else "") + (
            f"{dropped:,} row(s) left out: the denominator was zero or "
            f"missing, which has no value rather than an extreme one")
    if level == "cell":
        units = int(work["unit"].nunique())
        note = ((note + " · ") if note else "") + (
            f"one row per CELL: {units:,} rows, and every cell from one "
            f"well shares that well's treatment — so these rows are not "
            f"independent and the p-value is about cells, not wells")
    return Comparison(measurement=measurement, level=level, frame=work,
                      note=note)


def with_statistics(comparison: Comparison) -> Comparison:
    """Run the applicable statistical test and attach its evidence.

    :param comparison: grouped observations produced by :func:`build`.
    :returns: the same comparison object with ``statistics`` replaced.

    :func:`spacr.sp_stats.perform_statistical_tests` selects among Student's
    t-test, Welch's t-test, Mann-Whitney U, one-way ANOVA, Welch's ANOVA, and
    Kruskal-Wallis from the group count and assumption checks. Each result is
    supplemented with the observation level, measurement, normality result,
    equal-variance result, and sample size per group. Comparisons with fewer
    than two groups receive no statistical records.
    """
    if len(comparison.frame) < 2 or len(comparison.groups) < 2:
        comparison.statistics = []
        return comparison
    try:
        from .sp_stats import (perform_levene_test, perform_normality_tests,
                               perform_statistical_tests)

        rows = perform_statistical_tests(comparison.frame, "group", ["value"])
        normal = perform_normality_tests(comparison.frame, "group", ["value"])
        levene = perform_levene_test(comparison.frame, "group", "value")
    except Exception as exc:                                 # noqa: BLE001
        comparison.statistics = [{
            "Test Name": "not testable",
            "Why This Test": f"the statistics engine refused: {exc}"}]
        return comparison

    for row in rows or []:
        row.setdefault("Level", comparison.level)
        row["Measurement"] = comparison.measurement
        # THE ASSUMPTION CHECKS TRAVEL WITH THE RESULT. "where the variance
        # and normality and n is noted and the correct test chosen" -- a test
        # name without the checks that produced it cannot be reported.
        row["Normality"] = _summarise(normal)
        row["Equal variance"] = _summarise(levene)
        row["n per group"] = "; ".join(
            f"{k}={v}" for k, v in comparison.counts().items())
    comparison.statistics = list(rows or [])
    return comparison


#: What an assumption check is worth saying, in the order a reader wants it.
#: THE VERDICT FIRST: `sp_stats` already writes a sentence ("departs from
#: normal ... p = 0.00326 < 0.05, Bonferroni"), and repeating the statistic,
#: the column name and the row count beside it buries the one part that
#: decided the test.
_CHECK_KEYS = ("Verdict", "Test Name", "p-value")


def _summarise(result) -> str:
    """One readable line from whatever shape an assumption check came back as.

    The engine returns a list of per-group dicts with a dozen fields each,
    and dumping them makes a line no one reads -- measured at 400 characters
    for a two-group normality check. This keeps the verdict and the number
    behind it.
    """
    if result is None:
        return "not computed"
    if isinstance(result, pd.DataFrame):
        result = result.to_dict("records")
    if isinstance(result, dict):
        parts = [f"{result[k]}" for k in _CHECK_KEYS if result.get(k) not in
                 (None, "")]
        return " · ".join(parts) if parts else "; ".join(
            f"{k}={v}" for k, v in result.items())
    if isinstance(result, (list, tuple)):
        lines = [_summarise(one) for one in result]
        return " | ".join(x for x in lines if x) or "not computed"
    return str(result)


# --------------------------------------------------------------------------- #
# Drawing                                                                      #
# --------------------------------------------------------------------------- #


def plot(comparison: Comparison, path: Optional[str] = None, *,
         kind: str = "jitter_box", title: str = ""):
    """Draw a grouped measurement comparison.

    :param comparison: grouped observations, optionally with statistics.
    :param path: output path for an additional saved copy, or ``None`` to keep
        the figure in memory only.
    :param kind: one of ``'jitter_box'``, ``'box'``, ``'jitter'``,
        ``'violin'``, or ``'bar'``.
    :param title: custom title. By default the title reports the observation
        level, selected test, and P value when available.
    :returns: a Matplotlib figure, or ``None`` when no finite values can be
        drawn.

    Named groups use the spaCR highlight palette and :data:`REST` is grey. A
    bar plot warns on the figure when its smallest group has eight or fewer
    observations, because individual points show those data more clearly.
    """
    import matplotlib.pyplot as plt

    from .gene_measurement_sweep import HOUSE, _readable

    if not len(comparison.frame) or not comparison.groups:
        return None
    order = [g for g in comparison.groups if g != REST] + (
        [REST] if REST in comparison.groups else [])
    series = [comparison.frame.loc[comparison.frame["group"] == g, "value"]
              .to_numpy(dtype=float) for g in order]
    series = [s[np.isfinite(s)] for s in series]
    if not any(len(s) for s in series):
        return None

    # GREY IS THE REST, COLOUR IS THE CLAIM. More than one gene gets the
    # palette in its fixed order rather than a colormap, so the same gene is
    # the same colour in every panel of a figure.
    highlight = [HOUSE.BLUE, HOUSE.RUST, HOUSE.GREEN, HOUSE.PURPLE,
                 HOUSE.OCHRE, HOUSE.NAVY]
    colours = [HOUSE.GREY if g == REST else highlight[i % len(highlight)]
               for i, g in enumerate(order)]

    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:
    # rcParams reach an artist when it is CREATED, so a
    # context opened after `plt.subplots` would leave the
    # spines, ticks and labels at the caller's globals.
    with figure_style(theme_target()):
        figure, axes = plt.subplots(figsize=(1.5 + 1.15 * len(order), 3.6))
        positions = np.arange(len(order))
        rng = np.random.default_rng(0)

        if kind in ("box", "jitter_box"):
            drawn = axes.boxplot(series, positions=positions, widths=0.55,
                                 patch_artist=True, showfliers=False,
                                 medianprops={"color": HOUSE.GREY_DARK,
                                              "linewidth": HOUSE.DATA})
            for patch, colour in zip(drawn["boxes"], colours):
                patch.set_facecolor(colour)
                patch.set_alpha(0.25 if kind == "jitter_box" else 0.8)
                patch.set_edgecolor(colour)
                patch.set_linewidth(HOUSE.SPINE)
            for part in ("whiskers", "caps"):
                for line in drawn[part]:
                    line.set_color(HOUSE.GREY_DARK)
                    line.set_linewidth(HOUSE.SPINE)
        if kind == "violin":
            alive = [(i, s) for i, s in enumerate(series) if len(s) > 1]
            if alive:
                drawn = axes.violinplot([s for _i, s in alive],
                                        positions=[i for i, _s in alive],
                                        widths=0.7, showextrema=False)
                for body, (i, _s) in zip(drawn["bodies"], alive):
                    body.set_facecolor(colours[i])
                    body.set_alpha(0.55)
                    body.set_edgecolor("none")
        if kind == "bar":
            means = [float(np.mean(s)) if len(s) else np.nan for s in series]
            errors = [float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
                      for s in series]
            axes.bar(positions, means, yerr=errors, width=0.6, color=colours,
                     linewidth=0, error_kw={"ecolor": HOUSE.GREY_DARK,
                                            "elinewidth": HOUSE.SPINE,
                                            "capsize": 2.5})
        if kind in ("jitter", "jitter_box"):
            for i, (values, colour) in enumerate(zip(series, colours)):
                if not len(values):
                    continue
                spread = rng.uniform(-0.14, 0.14, len(values))
                axes.scatter(np.full(len(values), i) + spread, values, s=9,
                             color=colour, edgecolor="none", zorder=3,
                             rasterized=len(values) > 2000)

        axes.set_xticks(positions)
        axes.set_xticklabels(
            [f"{g}\n(n={len(s)})" for g, s in zip(order, series)],
            fontsize=HOUSE.TICK)
        axes.set_ylabel(str(comparison.measurement))
        axes.set_xlabel("")

        smallest = min((len(s) for s in series if len(s)), default=0)
        if kind == "bar" and 0 < smallest <= 8:
            # Preserve the requested bar plot but identify when the sample is so
            # small that showing individual observations would be more informative.
            axes.text(0.02, 0.98,
                      f"n = {smallest} in the smallest group: individual points "
                      f"say more than a bar here",
                      transform=axes.transAxes, fontsize=HOUSE.NOTE,
                      color=HOUSE.RUST, va="top")

        stat = (comparison.statistics or [{}])[0]
        name = str(stat.get("Test Name", "") or "")
        p_value = stat.get("p-value")
        caption = f"{comparison.level} level"
        if name:
            caption += f" · {name}"
        if isinstance(p_value, float) and np.isfinite(p_value):
            caption += f" · p = {p_value:.3g}"
        axes.set_title(title or caption, fontsize=HOUSE.LABEL)

        _readable(figure, axes)
        figure.tight_layout()
        if path:
            # 108 point 6: the one writer, so a comparison saved from this
            # panel is in the format every other kept figure is in.
            from .plot import save_figure

            save_figure(figure, path, bbox_inches="tight")
        return figure


# --------------------------------------------------------------------------- #
# Saving                                                                       #
# --------------------------------------------------------------------------- #


def save(comparison: Comparison, folder: str, *, kind: str = "jitter_box",
         settings: Optional[Dict[str, Any]] = None,
         images: Optional[Dict[str, Sequence[Any]]] = None,
         title: str = "") -> Dict[str, str]:
    """Save a comparison and its supporting data in one folder.

    :param comparison: grouped observations, optionally with statistics.
    :param folder: destination directory, created when necessary.
    :param kind: plot type accepted by :func:`plot`.
    :param settings: regression settings to include in ``settings.json``.
    :param images: mapping from well identifiers to image arrays. Images are
        written under ``cells/<well>/``; missing images are allowed.
    :param title: optional title passed to :func:`plot`.
    :returns: mapping from artifact type to each successfully written path.

    The folder always receives ``settings.json``. When available, it also
    receives PDF and PNG figures, ``data.csv``, ``statistics.csv``, and the
    supplied cell images. The returned mapping lists only artifacts that were
    written successfully.
    """
    import json
    import os

    os.makedirs(folder, exist_ok=True)
    written: Dict[str, str] = {}

    figure = plot(comparison, kind=kind, title=title)
    if figure is not None:
        import matplotlib.pyplot as plt

        try:
            from .plot import save_figure

            for suffix in ("pdf", "png"):
                path = os.path.join(folder, f"comparison.{suffix}")
                try:
                    # BOTH FORMATS ON PURPOSE (the folder is a deliverable),
                    # so `fmt` is the loop's; everything else `save_figure`
                    # does -- the DPI rule and the repaint for paper -- is
                    # gained.
                    written[suffix] = save_figure(
                        figure, path, fmt=suffix, bbox_inches="tight")
                except Exception:                            # noqa: BLE001
                    continue
        finally:
            # ``save`` returns paths rather than the figure, so there is no
            # caller that can own this pyplot registration.  Keeping it open
            # accumulated one figure per saved comparison in long-lived GUI
            # and batched-test processes.
            plt.close(figure)

    if len(comparison.frame):
        path = os.path.join(folder, "data.csv")
        # THE DATA BEHIND THE GRAPH, which is the frame that was PLOTTED --
        # not the objects it came from. A reader checking the figure needs
        # the numbers the figure drew.
        comparison.frame.to_csv(path, index=False)
        written["data"] = path

    if comparison.statistics:
        path = os.path.join(folder, "statistics.csv")
        pd.DataFrame(comparison.statistics).to_csv(path, index=False)
        written["statistics"] = path

    record = {
        "measurement": comparison.measurement,
        "level": comparison.level,
        "level_means": dict(LEVELS).get(comparison.level, ""),
        "plot": kind,
        "groups": list(comparison.groups),
        "n_per_group": comparison.counts(),
        "note": comparison.note,
    }
    if settings:
        # THE SETTINGS THAT GENERATED THE REGRESSION AND THE GRAPH, together
        # under separate keys, so it is never ambiguous which produced which.
        record["regression_settings"] = {
            str(k): _plain(v) for k, v in dict(settings).items()}
    path = os.path.join(folder, "settings.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
    written["settings"] = path

    for well, crops in (images or {}).items():
        if crops is None or not len(crops):
            continue
        here = os.path.join(folder, "cells", str(well))
        os.makedirs(here, exist_ok=True)
        for index, crop in enumerate(crops):
            try:
                from PIL import Image

                data = np.asarray(crop)
                if data.ndim == 2:
                    data = np.repeat(data[:, :, None], 3, axis=2)
                Image.fromarray(data.astype("uint8")[:, :, :3]).save(
                    os.path.join(here, f"{index:04d}.png"))
            except Exception:                                # noqa: BLE001
                # A crop that will not encode costs one image, never the
                # save: the figure and the numbers are the point.
                continue
        written.setdefault("cells", os.path.join(folder, "cells"))

    return written


def _plain(value):
    """A settings value JSON can hold, without losing what it was."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# 187 A: every measurement in the database, not only the ones on png_list
# ---------------------------------------------------------------------------

#: Columns an object's integer label can arrive under, in the order
#: :func:`object_identity` reads them. The same order
#: :data:`spacr.cell_montage._LABEL_COLUMNS` uses, because a row that the
#: crop source identified one way and this module identified another way is
#: two objects as far as the join is concerned.
LABEL_COLUMNS: Tuple[str, ...] = (
    "object_label", "label", "cell_id", "nucleus_id", "pathogen_id",
    "cytoplasm_id")


def object_identity(frame: "pd.DataFrame") -> Optional["pd.Series"]:
    """Resolve a stable plate/field/object identity for each row.

    Parameters
    ----------
    frame : pandas.DataFrame
        Object table containing ``prcfo`` or enough component columns to build
        it from a field key and an object label.

    Returns
    -------
    pandas.Series or None
        String identities aligned to ``frame``. ``None`` indicates that no
        supported object label or field key is available.

    Notes
    -----
    The ``prcfo`` key combines plate, row, column, field, and object label and
    matches the identity written by :func:`spacr.io._read_and_join_tables`.
    """
    columns = getattr(frame, "columns", ())
    if "prcfo" in columns:
        return frame["prcfo"].astype(str)
    label = next((c for c in LABEL_COLUMNS if c in columns), "")
    if not label:
        return None
    if "prcf" in columns:
        head = frame["prcf"].astype(str)
    elif "prc" in columns and "fieldID" in columns:
        head = frame["prc"].astype(str) + "_" + frame["fieldID"].astype(str)
    elif all(c in columns for c in FIELD_KEY):
        # THE FOUR COLUMNS ARE THE FIELD KEY. `prcf` is nothing but
        # plateID_rowID_columnID_fieldID pasted together -- `plate1_r5_c1_f16`
        # -- and `png_list` carries the four without ever carrying the paste.
        # So the crop table, which is the object table the Compare panel
        # starts from, had "no object identity" and the join refused it:
        # "these object rows carry no object identity (prcfo, or prcf and an
        # object label)". Every morphological measurement in the screen was
        # unreachable from that panel because of a column that was not
        # written rather than data that was not there.
        head = frame[FIELD_KEY[0]].astype(str)
        for column in FIELD_KEY[1:]:
            head = head + "_" + frame[column].astype(str)
    else:
        return None
    return head + "_" + _label_text(frame[label])


#: The columns `prcf` is built from, in the order it pastes them.
FIELD_KEY: Tuple[str, ...] = ("plateID", "rowID", "columnID", "fieldID")


def _label_text(values: "pd.Series") -> "pd.Series":
    """Object labels as the text `prcfo` uses, whichever spelling arrived.

    The object tables store the label as an integer; `png_list` stores the
    SAME object as `cell_id` in the prcfo spelling -- `o2`, not `2`. Pasting
    those two into an identity gives `..._f17_o2` on one side and `..._f17_2`
    on the other, so every row would match nothing and the join would come
    back empty while reporting no trouble at all.
    """
    import pandas as pd

    text = values.astype(str).str.strip()
    leading_letter = text.str.match(r"^[A-Za-z]")
    if not bool(leading_letter.any()):
        return text
    from .utils import object_label_from_png_id

    return pd.Series(object_label_from_png_id(values),
                     index=values.index).astype("Int64").astype(str)


def measurements_are_joined(objects: "pd.DataFrame") -> bool:
    """Return whether object rows contain joined morphology measurements.

    Parameters
    ----------
    objects : pandas.DataFrame
        Object table to inspect.

    Returns
    -------
    bool
        ``True`` when a non-identifier column uses a cell, nucleus, pathogen,
        or cytoplasm measurement prefix.
    """
    columns = {str(c) for c in getattr(objects, "columns", ())}
    return any(c.startswith(("cell_", "nucleus_", "pathogen_", "cytoplasm_"))
               and c not in ("cell_id", "nucleus_id", "pathogen_id",
                             "cytoplasm_id")
               for c in columns)


#: Columns that come across a merge even though they are not numeric.
#:
#: THE gRNA ANNOTATION IS THE GROUPING THE WHOLE SCREEN EXISTS TO SERVE, and
#: it is a string -- so the numeric filter that keeps a merge from dragging
#: every path and label across was also dropping the one column the compare
#: panel needs to offer a group at all. Named rather than "every object
#: column", because a merge that brought paths and filenames over would put
#: hundreds of useless entries in the measurement chooser.
ANNOTATION_COLUMNS: frozenset = frozenset({
    "grna", "grna_name", "gene", "gene_name", "condition", "prediction",
    "predicted_class", "annotation", "class",
    # FROM `png_list`, and the reason it is worth joining at all: the score
    # and the call are there and in no object table. `png_path` stays out --
    # a path is not a measurement and it would be offered in the comparison
    # chooser.
    "pred", "score", "test", "cv_predicted_class",
})


#: The object tables, without the crop table. `_read_and_join_tables`
#: defaults to these PLUS `png_list`, so leaving it out has to be said
#: explicitly.
OBJECT_TABLES: Tuple[str, ...] = ("cell", "cytoplasm", "nucleus", "pathogen")


def join_measurements(objects: "pd.DataFrame",
                      databases: Sequence[str],
                      *, keep_uninfected: bool = True,
                      png_list: bool = True
                      ) -> Tuple["pd.DataFrame", str]:
    """Join morphology measurements onto montage object rows.

    Parameters
    ----------
    objects : pandas.DataFrame
        Montage object table. Its index is preserved because group membership
        is expressed with these index values.
    databases : sequence of path-like
        ``measurements.db`` files containing object measurement tables.
    keep_uninfected : bool, default True
        Preserve cells without a pathogen row when reading joined tables.
    png_list : bool, default True
        Join the crop table as well as the object tables. It carries the
        classification score and the crop path, neither of which is in any
        object table -- which is why it is on by default and why the panel
        offers it at all.

        THE ARGUMENT EXISTS SO THE BOX CAN MEAN SOMETHING. It was a
        checkbox that was created, laid out and never read: a control the
        user can change that changes nothing, which is the failure this
        codebase writes comments about and shipped anyway.

    Returns
    -------
    frame : pandas.DataFrame
        Original rows widened with new numeric measurement columns. Existing
        columns are never replaced.
    note : str
        Empty after a clean join; otherwise a user-facing explanation of files
        or rows that could not be read or matched.

    Notes
    -----
    Object identities are matched with :func:`object_identity`. Recoverable
    read and matching failures are reported in ``note`` so callers can still
    use measurements already present on ``objects``.
    """
    from .io import _read_and_join_tables

    mine = object_identity(objects)
    if mine is None:
        return objects, ("these object rows carry no object identity "
                         "(prcfo, or prcf and an object label), so the "
                         "measurement tables cannot be joined onto them")

    frames, troubles = [], []
    for path in databases or ():
        try:
            wide = _read_and_join_tables(
                str(path), keep_uninfected=keep_uninfected,
                require_crops=False,
                table_names=None if png_list else list(OBJECT_TABLES))
        except Exception as exc:                             # noqa: BLE001
            troubles.append(f"{path}: {exc}")
            continue
        if wide is None or not len(wide):
            continue
        theirs = object_identity(wide)
        if theirs is None:
            troubles.append(f"{path}: the joined tables name no object")
            continue
        frames.append(wide.assign(_prcfo=theirs.values))
    if not frames:
        said = "no measurement table could be read"
        return objects, f"{said}: {'; '.join(troubles)}" if troubles else said

    wide = pd.concat(frames, ignore_index=True) if len(frames) > 1 \
        else frames[0]
    wide = wide.drop_duplicates(subset=["_prcfo"], keep="first")
    # ONLY WHAT IS NEW. A column present on both sides is already the value
    # the montage selected on, and replacing it here would move the cells
    # under the user's feet.
    have = set(map(str, objects.columns))
    fresh = [c for c in wide.columns
             if str(c) not in have and str(c) != "_prcfo"
             and (pd.api.types.is_numeric_dtype(wide[c])
                  or str(c) in ANNOTATION_COLUMNS)]
    if not fresh:
        return objects, ("the measurement tables add no column these rows "
                         "do not already have")

    lookup = wide.set_index("_prcfo")[fresh]
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    added = lookup.reindex(mine.values)

    # ASSIGNED BY POSITION, not joined. `_all_objects` concatenates one frame
    # per plan, so the index can repeat -- and a join on a repeated label is
    # a cartesian product of the matching rows on both sides, which turns
    # 20,000 cells into 40,000 and every count on the panel with it.
    # ALL AT ONCE, not column by column. Inserting them in a loop makes a
    # new block per column, and a measurement table brings hundreds -- which
    # is O(n^2) copying and a PerformanceWarning per column, hundreds of
    # identical lines in the user's terminal for one merge.
    #
    # RESET BOTH INDEXES FIRST, and that is the positional contract above,
    # not tidying: `concat(axis=1)` ALIGNS ON THE INDEX, so with a repeated
    # label -- which `_all_objects` produces, one frame per plan -- it would
    # do exactly the cartesian product the loop existed to avoid. Two clean
    # RangeIndexes make the concatenation positional, and the real index goes
    # back on afterwards.
    #
    # `reset_index` rather than `to_numpy`, so each column keeps its own
    # dtype. One array for the block would cast an integer count to float
    # because some other column beside it is float.
    out = pd.concat(
        [objects.reset_index(drop=True),
         added[fresh].reset_index(drop=True)], axis=1)
    out.index = objects.index

    matched = int(added[fresh[0]].notna().sum())
    note = ""
    if not matched and len(objects):
        # A SILENT ZERO IS THE REAL FAULT (instruction 203). There is no run
        # where NONE of the objects have a measurement, so zero matches is
        # a join key that does not line up -- and a merge that matched
        # nothing looks exactly like a merge that worked on an empty column
        # once it reaches a panel that draws it.
        #
        # THE ORIGINAL ROWS GO BACK, not the widened ones. Handing on a
        # frame of all-NaN measurement columns is how the empty plot gets
        # drawn; returning what the caller already had leaves them exactly
        # where they were, with a sentence saying why.
        return objects, (
            f"THE MERGE MATCHED NOTHING: none of {len(objects):,} object "
            f"row(s) were found in the measurement tables, so no measurement "
            f"was joined. This is a join-key problem rather than an empty "
            f"screen — the object identities on the two sides do not line "
            f"up. Nothing was changed."
            + (" · " + "; ".join(troubles) if troubles else ""))
    if matched < len(objects):
        # SAID, ALWAYS -- the same rule the dropped denominators follow. A
        # measurement that is missing on a third of the cells produces a
        # comparison on a third fewer cells, and the panel has to be able to
        # say so rather than quietly shrinking.
        note = (f"{len(objects) - matched:,} of {len(objects):,} object "
                f"row(s) found no match in the measurement tables and carry "
                f"no joined measurement")
    if troubles:
        note = ((note + " · ") if note else "") + "; ".join(troubles)
    return out, note


# ---------------------------------------------------------------------------
# 108 points 1 and 2: a second style on the shared base, and the contract
# ---------------------------------------------------------------------------

@dataclass
class ComparisonStyle(FigureStyle):
    """Appearance settings for grouped measurement comparisons.

    Shared axes, typography, grid, legend, page, and background settings are
    inherited from :class:`spacr.style_base.FigureStyle`. This class adds only
    comparison-specific choices such as plot kind, group filtering, jitter,
    and count labels.

    Parameters
    ----------
    x_label : str
        Horizontal-axis label; empty leaves the renderer's category labels.
    y_label : str
        Vertical-axis label; empty preserves the measurement name.
    title : str
        Optional title drawn above the comparison.
    x_scale : str
        Matplotlib horizontal scale inherited for house-style portability.
    y_scale : str
        Matplotlib vertical scale applied after the values are drawn.
    x_lim : tuple of float, optional
        Explicit horizontal limits, or ``None`` for data-derived limits.
    y_lim : tuple of float, optional
        Explicit vertical limits, or ``None`` for data-derived limits.
    invert_x : bool
        Reverse the horizontal axis after applying limits.
    invert_y : bool
        Reverse the vertical axis after applying limits.
    font_family : str
        Font-family value retained when styles move between figure types; the
        current comparison renderer does not apply a family override.
    font_size : float
        General text size retained for style portability; comparison text uses
        the specific title, label, and tick sizes below.
    title_font_size : float
        Title size in points.
    label_font_size : float
        Axis-label size in points.
    tick_font_size : float
        Group and value tick-label size in points.
    font_weight : str
        Weight applied to the optional title.
    figure_width : float
        Live and saved figure width in inches.
    figure_height : float
        Live and saved figure height in inches.
    dpi : int
        Resolution in dots per inch for raster exports.
    grid : bool
        Whether to show the selected grid lines.
    grid_axis : str
        Grid direction: ``"x"``, ``"y"``, ``"both"``, or ``"none"``.
    grid_color : str
        Matplotlib-compatible grid-line color.
    grid_width : float
        Grid-line width in points.
    hide_top_right_spines : bool
        Remove the top and right frame lines when true.
    legend : bool
        Retained for cross-style compatibility; comparison plots currently
        create no legend, so this value has no visual effect here.
    legend_location : str
        Retained legend position; unused while comparisons have no legend.
    background_color : str
        Matplotlib-compatible page and axes background, or ``"none"``.
    transparent : bool
        Export raster/vector backgrounds transparently when supported.
    kind : str
        Plot geometry from :data:`PLOTS`: box, jitter, violin, bar, or the
        default jitter-over-box combination.
    only : str
        Draw only one named group; stored statistics continue to describe the
        complete comparison.
    rest_color : str
        Color of the :data:`REST` group; empty uses the house grey.
    marker_size : float
        Scatter-marker area in squared typographic points.
    jitter_width : float
        Maximum random horizontal displacement on either side of a category.
    show_counts : bool
        Append each group's observation count to its tick label.
    """

    #: Which of :data:`PLOTS` to draw.
    kind: str = "jitter_box"
    #: Draw only this group, or ``''`` for all of them. A filter on the
    #: DRAW: the statistics still describe the whole comparison, because a
    #: test computed on one of two groups is not a comparison at all.
    only: str = ""
    #: The colour of the unannotated side. Grey, because grey is the default
    #: ink for data and colour is an argument -- the one rule the house style
    #: keeps above all others.
    rest_color: str = ""
    #: Point size for the jitter, and the half-width it is spread over.
    marker_size: float = 9.0
    jitter_width: float = 0.14
    #: Draw the n under each group's tick label.
    show_counts: bool = True

    CHOICES = dict(SHARED_CHOICES, kind=tuple(k for k, _label in PLOTS))


def render_comparison(comparison: Comparison, style: "ComparisonStyle" = None,
                      *, figure=None, save_path=None):
    """Render a grouped measurement comparison.

    Parameters
    ----------
    comparison : Comparison
        Grouped observations returned by :func:`build`.
    style : ComparisonStyle, optional
        Plot appearance. The default style is used when omitted.
    figure : matplotlib.figure.Figure, optional
        Existing figure to clear and redraw. Reusing a figure preserves the
        live-canvas object while restyling.
    save_path : path-like, optional
        Also write the completed figure through spaCR's export pipeline.

    Returns
    -------
    figure, axes
        Rendered Matplotlib objects, or ``(None, None)`` when no finite
        observations can be drawn.
    """
    import matplotlib.pyplot as plt

    from .figures.style import figure_style, theme_target
    from .gene_measurement_sweep import HOUSE
    from .style_base import apply_page, write

    style = style or ComparisonStyle()
    if comparison is None or not len(comparison.frame) or not comparison.groups:
        return None, None

    showing = comparison
    if style.only:
        from dataclasses import replace as _replace

        frame = showing.frame
        showing = _replace(
            showing, frame=frame[frame["group"].astype(str) == style.only])

    order = [g for g in showing.groups if g != REST] + (
        [REST] if REST in showing.groups else [])
    series = [showing.frame.loc[showing.frame["group"] == g, "value"]
              .to_numpy(dtype=float) for g in order]
    series = [s[np.isfinite(s)] for s in series]
    if not any(len(s) for s in series):
        return None, None

    highlight = [HOUSE.BLUE, HOUSE.RUST, HOUSE.GREEN, HOUSE.PURPLE,
                 HOUSE.OCHRE, HOUSE.NAVY]
    rest = style.rest_color or HOUSE.GREY
    colours = [rest if g == REST else highlight[i % len(highlight)]
               for i, g in enumerate(order)]

    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS, and that is why the
    # `figure=` branch is inside the context too: an axes added to an
    # existing figure creates its own spines and ticks at that moment.
    with figure_style(theme_target()):
        if figure is None:
            figure, axes = plt.subplots(
                figsize=(style.figure_width, style.figure_height))
        else:
            figure.clear()
            axes = figure.add_subplot(111)
        positions = np.arange(len(order))
        rng = np.random.default_rng(0)
        kind = str(style.kind or "jitter_box")

        if kind in ("box", "jitter_box"):
            drawn = axes.boxplot(series, positions=positions, widths=0.55,
                                 patch_artist=True, showfliers=False)
            for patch, colour in zip(drawn["boxes"], colours):
                patch.set_facecolor(colour)
                patch.set_alpha(0.30 if kind == "jitter_box" else 0.85)
                patch.set_edgecolor(HOUSE.GREY_DARK)
        if kind == "violin":
            alive = [(i, s) for i, s in enumerate(series) if len(s)]
            drawn = axes.violinplot([s for _i, s in alive],
                                    positions=[i for i, _s in alive],
                                    widths=0.7, showextrema=False)
            for body, (i, _s) in zip(drawn["bodies"], alive):
                body.set_facecolor(colours[i])
                body.set_alpha(0.55)
                body.set_edgecolor("none")
        if kind == "bar":
            means = [float(np.mean(s)) if len(s) else np.nan for s in series]
            errors = [float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
                      for s in series]
            axes.bar(positions, means, yerr=errors, width=0.6, color=colours,
                     linewidth=0)
        if kind in ("jitter", "jitter_box"):
            for i, (values, colour) in enumerate(zip(series, colours)):
                if not len(values):
                    continue
                spread = rng.uniform(-style.jitter_width, style.jitter_width,
                                     len(values))
                axes.scatter(np.full(len(values), i) + spread, values,
                             s=style.marker_size, color=colour,
                             edgecolor="none", zorder=3,
                             rasterized=len(values) > 2000)

        axes.set_xticks(positions)
        axes.set_xticklabels(
            [f"{g}\n(n={len(s)})" if style.show_counts else str(g)
             for g, s in zip(order, series)])
        # THE DEFAULT LABEL IS THE MEASUREMENT'S NAME, and a style that names
        # one wins -- `apply_page` sets it after this and only when it is not
        # blank, so "leave it alone" and "set it to nothing" stay different.
        axes.set_ylabel(str(comparison.measurement))
        apply_page(figure, axes, style)

    if save_path:
        write(figure, save_path, style)
    return figure, axes
