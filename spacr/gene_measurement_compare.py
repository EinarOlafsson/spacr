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

from .figures.style import figure_style, theme_target
from .object_roles import ORGANELLE_ROLES
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
        An empty string returns ``first`` after numeric conversion,
        with unparseable values made missing.
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
    frame: pd.DataFrame
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
    :param wells: the wells of the annotation to include. ``None`` includes
        all of them. A well left out is dropped from both sides -- it is not
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
        work[key] = objects[key].astype(str).loc[work.index]
    work = work.dropna(subset=["value"])

    if not len(work):
        return Comparison(
            measurement=measurement, level=level,
            frame=pd.DataFrame(columns=["group", "value"]),
            note=(f"no object carries a numeric {measurement!r}: every value "
                  f"is missing or could not be read as a number"))

    if level == "cell":
        work["unit"] = work.index.astype(str)
    else:
        work["unit"] = work[keys].agg("_".join, axis=1)
        work = (work.groupby(["unit", "group"], as_index=False)["value"]
                .mean())

    note = ""
    said = contrast_note(contrast) if contrast else ""
    if said:
        label = next((l for c, l, _w in CONTRASTS if c == contrast), contrast)
        note = f"{label}: {said}"
    if chosen_note:
        note = ((note + " · ") if note else "") + chosen_note
    if dropped:
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

    highlight = [HOUSE.BLUE, HOUSE.RUST, HOUSE.GREEN, HOUSE.PURPLE,
                 HOUSE.OCHRE, HOUSE.NAVY]
    colours = [HOUSE.GREY if g == REST else highlight[i % len(highlight)]
               for i, g in enumerate(order)]

    with figure_style(theme_target()):
        figure, axes = plt.subplots(figsize=(1.5 + 1.15 * len(order), 3.6))
        from .figures.bundle import _register_figure_data
        _register_figure_data(figure, comparison.frame, x="group", y="value", kind=str(kind), order=[str(g) for g in order])
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
            from .plot import save_figure

            save_figure(figure, path, bbox_inches="tight")
        return figure




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
                    written[suffix] = save_figure(
                        figure, path, fmt=suffix, bbox_inches="tight")
                except Exception:                            # noqa: BLE001
                    continue
        finally:
            plt.close(figure)

    if len(comparison.frame):
        path = os.path.join(folder, "data.csv")
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
    "pred", "score", "test", "cv_predicted_class",
})


#: The object tables, without the crop table. `_read_and_join_tables`
#: defaults to these PLUS `png_list`, so leaving it out has to be said
#: explicitly. The organelle slots are named for the same reason they are
#: named in :data:`spacr.run_compare.OBJECT_TABLES`: a comparison that cannot
#: see an organelle table cannot report that one run measured organelles and
#: another did not.
OBJECT_TABLES: Tuple[str, ...] = (
    "cell", "cytoplasm", "nucleus", "pathogen", *ORGANELLE_ROLES,
)


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
        object table, so it is joined by default. The corresponding panel
        checkbox sets this argument.

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

    out = pd.concat(
        [objects.reset_index(drop=True),
         added[fresh].reset_index(drop=True)], axis=1)
    out.index = objects.index

    matched = int(added[fresh[0]].notna().sum())
    note = ""
    if not matched and len(objects):
        return objects, (
            f"THE MERGE MATCHED NOTHING: none of {len(objects):,} object "
            f"row(s) were found in the measurement tables, so no measurement "
            f"was joined. This is a join-key problem rather than an empty "
            f"screen — the object identities on the two sides do not line "
            f"up. Nothing was changed."
            + (" · " + "; ".join(troubles) if troubles else ""))
    if matched < len(objects):
        note = (f"{len(objects) - matched:,} of {len(objects):,} object "
                f"row(s) found no match in the measurement tables and carry "
                f"no joined measurement")
    if troubles:
        note = ((note + " · ") if note else "") + "; ".join(troubles)
    return out, note



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

    with figure_style(theme_target()):
        if figure is None:
            figure, axes = plt.subplots(
                figsize=(style.figure_width, style.figure_height))
        else:
            figure.clear()
            axes = figure.add_subplot(111)
        from .figures.bundle import _register_figure_data
        _register_figure_data(figure, showing.frame, x="group", y="value", kind=str(style.kind or "jitter_box"), order=[str(g) for g in order])
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
        axes.set_ylabel(str(comparison.measurement))
        apply_page(figure, axes, style)

    if save_path:
        write(figure, save_path, style)
    return figure, axes


#: The constant that makes a median absolute deviation estimate a normal
#: standard deviation, so a robust z of 1 means what a z of 1 means.
MAD_TO_SD: float = 1.4826


def robust_z_to_controls(cells: pd.DataFrame, features: Sequence[str], *,
                         control: pd.Series, group: str = "well"
                         ) -> pd.DataFrame:
    """Each feature as a robust z against the control cells of its own group.

    :param cells: one row per cell.
    :param features: the numeric columns to normalise.
    :param control: a boolean Series over ``cells``; True marks a control
        (non-targeting) cell.
    :param group: the column whose groups are normalised separately, the
        well in an optical pooled screen.
    :returns: a frame of the same index and ``features`` columns:
        ``(x - median_control) / (1.4826 * MAD_control)`` within the group.

    THE CONTROLS SET THE SCALE, as in Funk et al. 2022: each well's
    non-targeting cells are that well's reference, so a well-to-well shift
    in staining moves the reference with it rather than reading as a
    phenotype. A group whose controls have zero spread, or no controls at
    all, is NaN rather than infinite.
    """
    control = pd.Series(control, index=cells.index).fillna(False).astype(bool)
    values = cells.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(np.nan, index=cells.index, columns=list(features))
    for _key, rows in cells.groupby(group, sort=False).groups.items():
        rows = pd.Index(rows)
        reference = values.loc[rows[control.loc[rows].to_numpy()]]
        if reference.empty:
            continue
        centre = reference.median()
        spread = (reference - centre).abs().median() * MAD_TO_SD
        spread = spread.where(spread > 0)
        out.loc[rows] = ((values.loc[rows] - centre) / spread).to_numpy()
    return out


def mitotic_calls(cells: pd.DataFrame, *, dna: str, area: str,
                  group: str = "well", dna_z: float = 3.0,
                  area_z: float = 0.0) -> pd.Series:
    """Mitotic cells, called from nuclear DNA intensity and nuclear area.

    :param cells: one row per cell.
    :param dna: the nuclear mean DNA (DAPI) intensity column.
    :param area: the nuclear area column.
    :param group: normalise within this column's groups.
    :param dna_z: the robust z above which the DNA mean must sit.
    :param area_z: the robust z the nuclear area must not exceed.
    :returns: a boolean Series over ``cells``.

    A mitotic nucleus is condensed chromatin: its DNA is packed into a
    smaller area, so its MEAN DNA intensity is far above an interphase
    nucleus's while its area is at or below the median. Both are robust z
    scores against every cell of the group, so the rule needs no absolute
    intensity and travels between plates. It is a deliberately simple rule:
    Funk et al. 2022 trained a classifier on DNA and tubulin features, which
    this does not reproduce, and a cell in early prophase or in telophase
    is missed.
    """
    every = pd.Series(True, index=cells.index)
    z = robust_z_to_controls(cells, [dna, area], control=every, group=group)
    return (z[dna] > float(dna_z)) & (z[area] <= float(area_z))


def aggregate_guides(cells: pd.DataFrame, features: Sequence[str], *,
                     guide: str = "guide", gene: str = "gene",
                     mitotic: Optional[str] = None) -> pd.DataFrame:
    """One row per guide: its gene, cell count and per-feature medians.

    :param cells: one row per cell, already normalised if wanted.
    :param features: the columns to summarise.
    :param guide: the guide column.
    :param gene: the gene column.
    :param mitotic: an optional boolean column; its mean becomes
        ``mitotic_fraction`` and its sum ``mitotic_cells``.
    :returns: ``guide, gene, n_cells, [mitotic_cells, mitotic_fraction,]
        <features>``.
    """
    grouped = cells.groupby([guide, gene], sort=True, observed=True)
    out = grouped[list(features)].median()
    out.insert(0, "n_cells", grouped.size())
    if mitotic is not None:
        flags = cells[mitotic].astype(bool)
        out.insert(1, "mitotic_cells",
                   flags.groupby([cells[guide], cells[gene]]).sum())
        out.insert(2, "mitotic_fraction",
                   out["mitotic_cells"] / out["n_cells"])
    return out.reset_index().rename(columns={guide: "guide", gene: "gene"})


#: How a gene profile is reduced from its cells (``ops_aggregate_level``).
AGGREGATE_LEVELS: Tuple[str, ...] = ("guides", "cells")

#: How non-targeting guides become profile rows (``ops_nt_rows``).
NT_ROW_MODES: Tuple[str, ...] = ("pseudo", "barcode")


def aggregate_genes(guides: pd.DataFrame, features: Sequence[str], *,
                    level: str = "guides",
                    cells: Optional[pd.DataFrame] = None,
                    gene: str = "gene",
                    min_cells: Optional[int] = None) -> pd.DataFrame:
    """One row per gene: the median of its guides' medians, or of its cells.

    :param guides: the table :func:`aggregate_guides` returns.
    :param features: the feature columns to reduce.
    :param level: ``'guides'`` (the median of the gene's guide medians) or
        ``'cells'`` (the median over every cell of the gene).
    :param cells: the cell table the guides were reduced from; ``level=
        'cells'`` needs it.
    :param gene: the gene column of ``cells``.
    :param min_cells: when set, genes with this many cells or fewer are
        left out.
    :returns: ``gene, n_guides, n_cells, [mitotic_cells, mitotic_index,]
        <features>``. The mitotic index pools the gene's cells -- mitotic
        cells over all cells -- which is how the published per-gene
        ``mitotic_index`` is defined.
    :raises ValueError: for an unknown ``level``, or ``level='cells'``
        without ``cells``.

    THE CELL LEVEL FOLLOWS BRIEFLOW. Brieflow (Di Bernardo et al., Nat
    Commun 2026, doi:10.1038/s41467-026-73643-7; cheeseman-lab/brieflow,
    MIT licence) reduces a gene as the median of all of its cells
    (``lib.aggregate.aggregate`` by ``gene_symbol_0``). A guide with many
    cells then weighs more than one with few, where the guide level gives
    every guide one vote. On plate A of Funk et al. 2022 the cell level
    raised mean retrieval mAP from 0.0602 to 0.0639 in TVN space.
    """
    level = str(level or "guides").strip().lower()
    if level not in AGGREGATE_LEVELS:
        raise ValueError(f"level must be one of {list(AGGREGATE_LEVELS)}; "
                         f"got {level!r}")
    grouped = guides.groupby("gene", sort=True)
    if level == "cells":
        if cells is None:
            raise ValueError("level='cells' needs the cell table")
        by_cell = cells.groupby(gene, sort=True)
        out = by_cell[list(features)].median()
        out.index.name = "gene"
        out.insert(0, "n_guides", grouped.size().reindex(out.index)
                   .fillna(0).astype(int))
        out.insert(1, "n_cells", by_cell.size())
    else:
        out = grouped[list(features)].median()
        out.insert(0, "n_guides", grouped.size())
        out.insert(1, "n_cells", grouped["n_cells"].sum())
    if "mitotic_cells" in guides:
        mitotic = grouped["mitotic_cells"].sum().reindex(out.index).fillna(0)
        out.insert(2, "mitotic_cells", mitotic.astype(int))
        out.insert(3, "mitotic_index", out["mitotic_cells"] / out["n_cells"])
    if min_cells is not None:
        out = out[out["n_cells"] > int(min_cells)]
    return out.reset_index()


def nt_row_names(guides: Sequence, genes: Sequence, *, rows: str = "pseudo",
                 control_prefix: str = "nontargeting",
                 nt_guides: Optional[Sequence] = None, group_size: int = 4,
                 seed: int = 0, drop_incomplete: bool = True,
                 pseudo_prefix: str = "NTpseudo_") -> np.ndarray:
    """The gene name each cell (or guide) gets when controls become rows.

    :param guides: the guide of every row.
    :param genes: the gene of every row; a name starting with
        ``control_prefix`` (any case) marks a non-targeting guide.
    :param rows: ``'pseudo'`` groups the non-targeting guides at random into
        pseudo-genes of ``group_size``, named ``NTpseudo_00``,
        ``NTpseudo_01``, ...; ``'barcode'`` makes every non-targeting guide
        its own row, named ``<control_prefix>_<guide>``.
    :param control_prefix: the control gene prefix.
    :param nt_guides: the non-targeting guides the pseudo-genes are drawn
        from; by default those among ``guides``. Pass the full set when only
        part of the cells is named, so every part gets the same groups.
    :param group_size: guides per pseudo-gene.
    :param seed: the permutation seed.
    :param drop_incomplete: name the guides of a last, smaller pseudo-gene
        ``nontargeting_drop`` so they can be left out.
    :param pseudo_prefix: the pseudo-gene name before its two-digit number.
    :returns: the gene names; targeting rows keep their gene.
    :raises ValueError: for an unknown ``rows``.

    The pseudo-genes are drawn as in the improved Funk plate A analysis:
    the sorted unique non-targeting guides are permuted with
    ``numpy.random.default_rng(seed)`` and guide ``i`` goes to pseudo-gene
    ``rank_i // group_size``, where ``rank_i`` is its position in the
    permutation. A pseudo-gene is processed exactly as a gene, so it is the
    null a gene's statistic is compared against. One row per barcode is
    Brieflow's layout (Di Bernardo et al., Nat Commun 2026).
    """
    rows = str(rows or "pseudo").strip().lower()
    if rows not in NT_ROW_MODES:
        raise ValueError(f"rows must be one of {list(NT_ROW_MODES)}; got {rows!r}")
    guides = np.asarray(guides).astype(str)
    names = np.asarray(genes).astype(str).astype(object)
    control = pd.Series(names).str.lower().str.startswith(
        control_prefix.lower()).to_numpy()
    if rows == "barcode":
        names[control] = [f"{control_prefix}_{g}" for g in guides[control]]
        return names
    pool = guides[control] if nt_guides is None else np.asarray(nt_guides).astype(str)
    universe = np.array(sorted(set(pool)))
    rank = np.argsort(np.random.default_rng(int(seed)).permutation(len(universe)))
    mapping = {g: f"{pseudo_prefix}{k // int(group_size):02d}"
               for g, k in zip(universe, rank)}
    if drop_incomplete:
        sizes = pd.Series(mapping).value_counts()
        mapping = {g: p for g, p in mapping.items()
                   if sizes[p] == int(group_size)}
    names[control] = [mapping.get(g, "nontargeting_drop") for g in guides[control]]
    return names


_GENE_PVALUE_NULLS: Tuple[str, ...] = ("calibrated", "cell_bootstrap")


def _calibrated_null(control_values: pd.Series, control_guides: pd.DataFrame,
                     feature: str) -> Optional[Dict[str, float]]:
    """The normal null of one feature from non-targeting cells and guides.

    :param control_values: the feature over the non-targeting cells.
    :param control_guides: one row per non-targeting guide, with
        ``n_cells`` and the feature's guide median.
    :param feature: the feature column of ``control_guides``.
    :returns: ``centre``, ``cell_variance`` and ``between_variance``, or
        None when the null cannot be estimated.

    The cell variance is the squared robust standard deviation of the
    control cells (1.4826 times their MAD). A median of ``n`` cells then
    varies by ``pi / 2 * cell_variance / n``. The between-guide variance is
    what the non-targeting guide medians vary by beyond that, floored at
    zero.
    """
    cells = pd.to_numeric(control_values, errors="coerce").to_numpy(float)
    cells = cells[np.isfinite(cells)]
    medians = pd.to_numeric(control_guides[feature],
                            errors="coerce").to_numpy(float)
    sizes = control_guides["n_cells"].to_numpy(float)
    finite = np.isfinite(medians) & (sizes > 0)
    medians, sizes = medians[finite], sizes[finite]
    if cells.size == 0 or medians.size < 2:
        return None
    cell_variance = float(
        (MAD_TO_SD * np.median(np.abs(cells - np.median(cells)))) ** 2)
    sampling = np.pi / 2 * cell_variance / sizes
    between = max(0.0, float(np.var(medians, ddof=1) - sampling.mean()))
    if cell_variance + between <= 0:
        return None
    return {"centre": float(np.median(medians)),
            "cell_variance": cell_variance,
            "between_variance": between}


def bootstrap_gene_pvalues(control_values: pd.DataFrame,
                           guides: pd.DataFrame, features: Sequence[str], *,
                           n_boot: int = 1000, seed: int = 0,
                           max_cells: int = 2000,
                           null: str = "calibrated",
                           control_guides: Optional[pd.DataFrame] = None
                           ) -> pd.DataFrame:
    """Two-sided P values per gene against non-targeting controls.

    :param control_values: one row per non-targeting cell, ``features``
        columns, on the same scale as the guide medians.
    :param guides: :func:`aggregate_guides` output for the genes to test.
    :param features: the features to test.
    :param n_boot: simulated or bootstrap replicates; P values resolve to
        ``1 / (n_boot + 1)``.
    :param seed: the random seed, so a rerun gives the same table.
    :param max_cells: the cell bootstrap caps a guide's draw at this many
        cells. The median of a larger draw barely moves, and the cost is
        linear in it. The calibrated null does not use it.
    :param null: ``'calibrated'`` or ``'cell_bootstrap'``.
    :param control_guides: :func:`aggregate_guides` output for the
        non-targeting guides. The calibrated null needs it.
    :returns: ``gene`` plus ``<feature>_pvalue`` and ``<feature>_fdr``
        (Benjamini-Hochberg across genes) per feature. With the calibrated
        null, ``attrs["null"]`` holds each feature's centre, cell variance
        and between-guide variance.
    :raises ValueError: for an unknown ``null``, or for the calibrated null
        without ``control_guides``.

    BOTH NULLS REPEAT THE GENE'S OWN SAMPLING. The gene statistic is the
    median of its guide medians. Each null draws one value per guide of the
    gene, sized like that guide, and takes their median.

    THE CALIBRATED NULL IS THE DEFAULT. Each simulated guide median is drawn
    from a normal distribution centred on the median of the non-targeting
    guides. Its variance is ``between + pi / 2 * cell / n``. Here ``n`` is
    the guide's cell count and ``cell`` is the squared robust standard
    deviation of the control cells. The ``between`` term is the variance
    the non-targeting guide medians show beyond cell sampling. Cells of one
    guide share wells, fields and clonal patches, so in an optical pooled
    screen that term is large.

    THE CELL BOOTSTRAP IGNORES THE BETWEEN-GUIDE VARIANCE. It draws as many
    non-targeting cells as each guide has and takes their median, which is
    the construction of Funk et al. 2022. Its null therefore holds only
    cell-sampling noise, and its P values are too small. On one plate of the
    Funk et al. 2022 screen it called 4,053 of 5,049 genes and a third of
    the non-targeting guides hits. It is kept only to rerun analyses made
    before spaCR 1.5.1.8.
    """
    from .multiple_testing import _multipletests

    null = str(null or "calibrated").strip().lower()
    if null not in _GENE_PVALUE_NULLS:
        raise ValueError(f"null must be one of {list(_GENE_PVALUE_NULLS)}; "
                         f"got {null!r}")
    if null == "calibrated" and control_guides is None:
        raise ValueError(
            "the calibrated null needs control_guides, the guide medians of "
            "the non-targeting guides; pass null='cell_bootstrap' for the "
            "earlier test")
    rng = np.random.default_rng(int(seed))
    replicates = int(n_boot)
    params: Dict[str, Optional[Dict[str, float]]] = {}
    pool: Dict[str, np.ndarray] = {}
    if null == "calibrated":
        params = {feature: _calibrated_null(control_values[feature],
                                            control_guides, feature)
                  for feature in features}
    else:
        pool = {feature: pd.to_numeric(control_values[feature],
                                       errors="coerce").dropna().to_numpy()
                for feature in features}
    rows = []
    for gene, members in guides.groupby("gene", sort=True):
        row: Dict[str, Any] = {"gene": gene}
        if null == "calibrated":
            for feature in features:
                column = members[feature]
                values = (column.to_numpy()
                          if pd.api.types.is_float_dtype(column)
                          else pd.to_numeric(column, errors="coerce")
                          .to_numpy(float))
                sizes = members["n_cells"].to_numpy(float)
                keep = np.isfinite(values)
                fit = params[feature]
                if fit is None or not keep.any():
                    row[f"{feature}_pvalue"] = np.nan
                    continue
                centre = fit["centre"]
                observed = float(np.median(values[keep]))
                spread = np.sqrt(fit["between_variance"] + np.pi / 2
                                 * fit["cell_variance"] / sizes[keep])
                simulated = np.median(
                    centre + rng.standard_normal((replicates, spread.size))
                    * spread, axis=1)
                extreme = np.abs(simulated - centre) >= abs(observed - centre)
                row[f"{feature}_pvalue"] = float(
                    (extreme.sum() + 1) / (replicates + 1))
            rows.append(row)
            continue
        sizes = np.minimum(members["n_cells"].to_numpy(int), int(max_cells))
        for feature in features:
            values = pool[feature]
            observed = float(np.nanmedian(members[feature]))
            if values.size == 0 or not np.isfinite(observed):
                row[f"{feature}_pvalue"] = np.nan
                continue
            centre = float(np.median(values))
            null_medians = np.empty(replicates)
            per_guide = np.empty((replicates, sizes.size))
            for index, size in enumerate(sizes):
                draws = rng.choice(values, size=(replicates, max(1, size)))
                per_guide[:, index] = np.median(draws, axis=1)
            null_medians[:] = np.median(per_guide, axis=1)
            extreme = np.abs(null_medians - centre) >= abs(observed - centre)
            row[f"{feature}_pvalue"] = float(
                (extreme.sum() + 1) / (replicates + 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    for feature in features:
        column = f"{feature}_pvalue"
        if column not in out:
            continue
        fdr = pd.Series(np.nan, index=out.index)
        finite = out[column].notna()
        if finite.any():
            fdr[finite] = _multipletests(out.loc[finite, column].to_numpy(),
                                         "fdr_bh")[1]
        out[f"{feature}_fdr"] = fdr
    if null == "calibrated":
        out.attrs["null"] = params
    return out


#: Name prefixes that mark a non-targeting profile row, besides the control
#: prefix itself: the pseudo-genes of :func:`nt_row_names`.
_PSEUDO_PREFIXES: Tuple[str, ...] = ("ntpseudo",)


def _control_rows(names: Sequence, control_prefix: str = "nontargeting"
                  ) -> np.ndarray:
    """True for every name that marks a non-targeting row (any case).

    :param names: gene names.
    :param control_prefix: the control gene prefix.
    :returns: a boolean array.
    """
    prefixes = (control_prefix.lower(), *_PSEUDO_PREFIXES)
    return pd.Series(np.asarray(names, dtype=object)).astype(str).str.lower(
    ).str.startswith(prefixes).to_numpy()


class _RowEmbedder:
    """Standardise the gene rows and keep the PCs holding ``variance``.

    The transform is fitted on the gene rows and kept, so guide rows of the
    same feature space land in the same embedding. It is the embedding of
    Brieflow's ``funk_benchmark.py``: columns with 5 % or more non-finite
    values are dropped, the rest median-imputed, standardised (columns with
    no spread dropped) and reduced by PCA to ``variance``.
    """

    def __init__(self, values, variance: float = 0.99, seed: int = 0):
        """Fit the embedding.

        :param values: the gene rows, one row per gene.
        :param variance: the share of variance the PCA keeps.
        :param seed: the PCA seed.
        """
        from sklearn.decomposition import PCA

        x = np.asarray(values, float)
        x = np.where(np.isfinite(x), x, np.nan)
        self.keep = np.isnan(x).mean(axis=0) < 0.05
        x = x[:, self.keep]
        self.median = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.median, x)
        spread = x.std(axis=0)
        self.ok = spread > 0
        self.mean, self.sd = x[:, self.ok].mean(axis=0), spread[self.ok]
        self.pca = PCA(n_components=variance, random_state=seed)
        self.rows = self.pca.fit_transform((x[:, self.ok] - self.mean) / self.sd)

    def __call__(self, values) -> np.ndarray:
        """Embed other rows with the fitted transform.

        :param values: rows of the same feature columns.
        :returns: the embedded rows.
        """
        x = np.asarray(values, float)[:, self.keep]
        x = np.where(np.isfinite(x), x, self.median)
        return self.pca.transform((x[:, self.ok] - self.mean) / self.sd)


def contrastive_weighting(genes: pd.DataFrame, guides: pd.DataFrame,
                          features: Sequence[str], *,
                          control_prefix: str = "nontargeting",
                          min_cells: Optional[int] = 10,
                          variance: float = 0.99, seed: int = 0,
                          drop_zero: bool = False
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Weight gene profiles by how far each direction rises above guide noise.

    :param genes: one row per gene with ``gene``, ``n_cells`` and the
        ``features``; non-targeting rows (names starting with
        ``control_prefix`` or ``NTpseudo``) are kept as rows but not fitted.
    :param guides: one row per guide with ``guide``, ``gene``, ``n_cells``
        and the same ``features``. The non-targeting guides give the noise.
    :param features: the profile columns.
    :param control_prefix: the control gene prefix.
    :param min_cells: genes with this many cells or fewer are left out
        first; None keeps every gene.
    :param variance: the share of variance the embedding keeps.
    :param seed: the PCA seed.
    :param drop_zero: leave out the directions whose weight is 0. The
        cosine of two rows is the same either way.
    :returns: ``(gene_rows, guide_rows, info)``: ``gene, n_cells, w_0 ...``
        and ``guide, gene, n_cells, w_0 ...`` in the weighted space, and
        ``lambda``, ``weights``, ``directions_kept``, ``kbar`` and
        ``nt_guides``.
    :raises ValueError: when fewer than two directions have lambda above 1.

    The gene rows are embedded (standardised, PCA 99 %) and the guide rows
    put in the same space. The noise covariance is the Ledoit-Wolf
    covariance (Ledoit & Wolf 2004) of the centred non-targeting guide rows,
    divided by ``kbar``, the mean number of guides per targeting gene, since
    a gene profile pools that many guides. The generalised eigenproblem
    ``Sigma_genes v = lambda Sigma_noise v`` gives directions ordered by
    signal over noise; each is weighted by ``max((lambda - 1) / lambda,
    0)``, a Wiener weight. This is a contrastive eigenproblem in the spirit
    of contrastive PCA (Abid et al. 2018).

    IT REFUSES A DEGENERATE SPACE. On profiles already shrunk toward the
    controls (:func:`shrink_gene_profiles`) no direction rises above the
    noise, every weight is 0 and every row the zero vector; a retrieval
    score then ranks ties first and reports AP = 1. Do not stack the two.
    """
    from scipy.linalg import eigh
    from sklearn.covariance import LedoitWolf

    if min_cells is not None:
        genes = genes[genes["n_cells"] > int(min_cells)]
    genes = genes.reset_index(drop=True)
    names = genes["gene"].to_numpy()
    embed = _RowEmbedder(genes[list(features)].to_numpy(float),
                         variance=variance, seed=seed)
    targeting = ~_control_rows(names, control_prefix)
    keep = guides["gene"].isin(set(names)) | _control_rows(guides["gene"],
                                                           control_prefix)
    guides = guides[keep.to_numpy()].reset_index(drop=True)
    guide_rows = embed(guides[list(features)].to_numpy(float))
    guide_nt = _control_rows(guides["gene"], control_prefix)
    kbar = float(pd.Series(guides["gene"].to_numpy()[~guide_nt])
                 .value_counts().mean())
    noise = guide_rows[guide_nt] - guide_rows[guide_nt].mean(0)
    noise = LedoitWolf().fit(noise).covariance_ / kbar
    lam, vectors = eigh(np.cov(embed.rows[targeting], rowvar=False), noise)
    lam, vectors = lam[::-1], vectors[:, ::-1]
    weights = np.clip((lam - 1) / lam, 0, None)
    above = int((lam > 1).sum())
    if above < 2:
        raise ValueError(
            f"contrastive weighting refused: only {above} direction(s) have "
            "lambda above 1, so the weighted profiles would be degenerate; "
            "profiles shrunk toward the controls should not be weighted")
    if drop_zero:
        kept = weights > 0
        basis = vectors[:, kept] * weights[kept]
        z_genes, z_guides = embed.rows @ basis, guide_rows @ basis
    else:
        z_genes = (embed.rows @ vectors) * weights
        z_guides = (guide_rows @ vectors) * weights
    columns = [f"w_{j}" for j in range(z_genes.shape[1])]
    out_genes = pd.DataFrame(z_genes, columns=columns)
    out_genes.insert(0, "gene", names)
    out_genes.insert(1, "n_cells", genes["n_cells"].to_numpy())
    out_guides = pd.DataFrame(z_guides, columns=columns)
    out_guides.insert(0, "guide", guides["guide"].to_numpy())
    out_guides.insert(1, "gene", guides["gene"].to_numpy())
    out_guides.insert(2, "n_cells", guides["n_cells"].to_numpy())
    info = {"lambda": lam, "weights": weights, "directions_kept": above,
            "kbar": kbar, "nt_guides": int(guide_nt.sum()),
            "genes": int(targeting.sum())}
    return out_genes, out_guides, info


#: Gene-profile options applied after :func:`contrastive_weighting`
#: (``ops_guide_efficacy``).
GUIDE_EFFICACY_MODES: Tuple[str, ...] = ("jacks_blend",)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Every row scaled to length 1 (a zero row stays zero).

    :param values: a matrix.
    :returns: the scaled matrix.
    """
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True),
                            1e-12, None)


def _rank_one_efficacy(rows: np.ndarray, weights: np.ndarray, *,
                       iters: int, emax: float
                       ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fit ``x_g = e_g m`` to one gene's guide rows.

    :param rows: the gene's guide rows.
    :param weights: each guide's weight (its cell count).
    :param iters: the most alternations run.
    :param emax: the cap on an efficacy before rescaling.
    :returns: ``(m, e, iterations)``: the profile ``sum w e x / sum w e^2``,
        the efficacies and the alternations run.
    """
    e = np.ones(len(rows))
    done = 0
    for done in range(1, int(iters) + 1):
        den = (weights * e ** 2).sum()
        if den <= 0:
            e = np.ones(len(rows))
            den = weights.sum()
        m = (weights * e) @ rows / den
        mm = m @ m
        if mm <= 0:
            break
        e_new = np.clip(rows @ m / mm, 0, emax)
        e_new = (e_new / max(e_new.max(), 1e-12) if e_new.max() > 0
                 else np.ones(len(rows)))
        if np.allclose(e_new, e, atol=1e-6):
            e = e_new
            break
        e = e_new
    m = (weights * e) @ rows / max((weights * e ** 2).sum(), 1e-12)
    return m, e, done


def guide_efficacy_profiles(genes: pd.DataFrame, guides: pd.DataFrame, *,
                            columns: Optional[Sequence[str]] = None,
                            iters: int = 20, emax: float = 1.0
                            ) -> Tuple[pd.DataFrame, pd.Series]:
    """Gene profiles that weight each guide by how well it knocks down.

    :param genes: the gene rows :func:`contrastive_weighting` returns
        (``gene``, ``n_cells``, ``w_*``); with the cell level of
        :func:`aggregate_genes` they are the all-cell gene medians.
    :param guides: its guide rows (``guide``, ``gene``, ``n_cells``,
        ``w_*``).
    :param columns: the profile columns; by default the ``w_*`` columns.
    :param iters: the most alternations per gene (``ops_guide_efficacy_iters``).
    :param emax: the cap on an efficacy before rescaling
        (``ops_guide_efficacy_emax``).
    :returns: ``(rows, efficacy)``: one row per gene, ``gene, n_cells`` and
        the columns of ``Z/|Z| + M/|M|``, whose cosine is the similarity;
        and the efficacy of every guide, indexed by guide.
    :raises ValueError: when the weighted space has fewer than two
        directions, as :func:`contrastive_weighting` refuses.

    THE GUIDE MODEL IS JACKS' RANK-1 MODEL (Allen et al. 2019, Genome Res,
    doi:10.1101/gr.238923.118), with the profile features in the role of
    JACKS' screens: each guide row is ``x_g = e_g m + noise``, the noise
    variance proportional to ``1 / n_cells(g)``. Starting from ``e = 1``,
    ``m = sum w e x / sum w e^2`` (``w`` the cell counts) and ``e =
    clip(<x, m> / <m, m>, 0, emax)`` alternate, ``e`` rescaled so its
    maximum is 1, for at most ``iters`` rounds or until ``e`` changes by
    less than 1e-6. The efficacy-weighted profile ``M`` is then blended
    with the all-cell gene profile ``Z`` as unit vectors. A gene row
    without guide rows keeps ``Z``. The method uses no reference labels.
    """
    if columns is None:
        columns = [c for c in genes.columns if str(c).startswith("w_")]
    columns = list(columns)
    z = genes[columns].to_numpy(float)
    if (np.abs(z).sum(axis=0) > 0).sum() < 2:
        raise ValueError(
            "guide efficacy refused: the weighted profiles have fewer than "
            "two directions; run contrastive_weighting first")
    zg = guides[columns].to_numpy(float)
    counts = guides["n_cells"].to_numpy(float)
    members = pd.Series(guides["gene"].to_numpy())
    members = members.groupby(members).indices
    m_rows = z.copy()
    efficacy = pd.Series(np.nan, index=guides["guide"].to_numpy(), dtype=float)
    for i, name in enumerate(genes["gene"].to_numpy()):
        index = members.get(name)
        if index is None or len(index) == 0:
            continue
        m_rows[i], e, _done = _rank_one_efficacy(
            zg[index], counts[index], iters=iters, emax=emax)
        efficacy.iloc[index] = e
    rows = pd.DataFrame(_unit_rows(z) + _unit_rows(m_rows), columns=columns)
    rows.insert(0, "gene", genes["gene"].to_numpy())
    rows.insert(1, "n_cells", genes["n_cells"].to_numpy())
    return rows, efficacy


#: The cell-cycle phases of :func:`cell_cycle_phases`, in profile order.
CELL_CYCLE_PHASES: Tuple[str, ...] = ("G1", "SG2", "M")

#: What ``ops_profile_cell_cycle`` may be besides empty.
CELL_CYCLE_PROFILES: Tuple[str, ...] = ("concat", *CELL_CYCLE_PHASES)


def g1_mode(log_dna: np.ndarray, *, bins: int = 120,
            smooth: float = 2.0) -> float:
    """The G1 peak of a log2 DNA-content distribution.

    :param log_dna: log2 integrated nuclear DNA intensity of interphase
        cells.
    :param bins: histogram bins between the 0.5th and 99.5th percentiles.
    :param smooth: the Gaussian smoothing of the histogram, in bins.
    :returns: the centre of the highest bin of the smoothed histogram.
    """
    from scipy.ndimage import gaussian_filter1d

    low, high = np.percentile(log_dna, [0.5, 99.5])
    counts, edges = np.histogram(log_dna, bins=int(bins), range=(low, high))
    smoothed = gaussian_filter1d(counts.astype(float), float(smooth))
    return float(edges[np.argmax(smoothed)] + (edges[1] - edges[0]) / 2)


def cell_cycle_phases(cells: pd.DataFrame, *, dna: str, control: pd.Series,
                      mitotic: str = "mitotic", group: str = "well",
                      split_log2: float = 0.30,
                      mode_cells: Optional[pd.Series] = None) -> pd.DataFrame:
    """Each cell's cell-cycle phase: G1, S/G2 or M.

    :param cells: one row per cell.
    :param dna: the integrated nuclear DNA (DAPI) intensity column,
        flat-field corrected.
    :param control: True for a non-targeting cell.
    :param mitotic: the boolean mitotic-call column (the DNA rule of
        :func:`mitotic_calls` or a trained classifier).
    :param group: the column whose groups (wells) find their own G1 peak.
    :param split_log2: interphase cells this far above the G1 peak, in log2
        units, are S/G2 (``ops_profile_g1_sg2_split_log2``; 0.30 is about
        1.23 times G1 DNA, the dip of the non-targeting histogram on Funk
        plate A).
    :param mode_cells: the cells the G1 peak is found on; by default the
        non-targeting interphase cells.
    :returns: ``phase`` (``'G1'``, ``'SG2'`` or ``'M'``) and
        ``dna_log2_rel`` (log2 DNA minus the group's G1 peak), on the index
        of ``cells``.

    The G1 peak is the mode of the smoothed histogram of log2 DNA content
    (:func:`g1_mode`), found per well on the non-targeting interphase cells,
    so a knockout that piles cells up in S/G2 does not move it.
    """
    control = pd.Series(control, index=cells.index).fillna(False).astype(bool)
    flags = cells[mitotic].astype(bool).to_numpy()
    if mode_cells is None:
        mode_cells = control & ~flags
    mode_cells = pd.Series(mode_cells, index=cells.index).fillna(False
                                                                 ).astype(bool)
    log_dna = np.log2(np.clip(pd.to_numeric(cells[dna], errors="coerce")
                              .to_numpy(float), 1, None))
    relative = np.full(len(cells), np.nan)
    for _key, rows in cells.groupby(group, sort=False).indices.items():
        anchor = rows[mode_cells.to_numpy()[rows]]
        if len(anchor) == 0:
            continue
        relative[rows] = log_dna[rows] - g1_mode(log_dna[anchor])
    phase = np.where(flags, "M", np.where(relative < float(split_log2),
                                          "G1", "SG2"))
    return pd.DataFrame({"phase": phase,
                         "dna_log2_rel": relative.astype(np.float32)},
                        index=cells.index)


def phase_profiles(cells: pd.DataFrame, features: Sequence[str], *,
                   control: pd.Series, phase: str = "phase",
                   group: str = "well", guide: str = "guide",
                   gene: str = "gene",
                   phases: Sequence[str] = CELL_CYCLE_PHASES,
                   drop_genes: Sequence[str] = ("nontargeting_drop",)
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Guide and gene profiles resolved by cell-cycle phase.

    :param cells: one row per cell with ``phase``, ``group``, ``guide``,
        ``gene`` and the ``features`` (TVN components in the published
        analysis).
    :param features: the profile columns.
    :param control: True for a non-targeting cell.
    :param phase: the phase column (:func:`cell_cycle_phases`).
    :param group: each group (well) is centred on its own controls.
    :param guide: the guide column.
    :param gene: the gene column.
    :param phases: the phases profiled, in column order.
    :param drop_genes: gene names left out of both tables after the
        centring (the guides :func:`nt_row_names` could not group).
    :returns: ``(guides, genes)``. The guide table has ``guide, gene`` and
        per phase ``<phase>_n_cells`` and ``<phase>_<feature>``; a guide
        with no cell in a phase has NaN there. The gene table has ``gene``
        and the ``<phase>_<feature>`` columns: the median of its guides.

    Within each well and phase, every cell is first centred on the median of
    the well's non-targeting cells of the same phase. A profile then shows
    what the knockout changes within a phase, not that it moves cells
    between phases (the phase fractions say that). The concatenated profile
    ``[G1 | S/G2 | M]`` keeps the three apart. A gene's cells therefore count
    in the phase they are in.
    """
    control = pd.Series(control, index=cells.index).fillna(False
                                                           ).astype(bool).to_numpy()
    values = cells[list(features)].to_numpy()
    centred = values.copy()
    stage = cells[phase].astype(str).to_numpy()
    wells = cells[group].to_numpy()
    for well in pd.unique(wells):
        for name in phases:
            rows = (wells == well) & (stage == name)
            reference = rows & control
            if reference.any():
                centred[rows] -= np.median(values[reference], axis=0)
    parts = []
    for name in phases:
        rows = stage == name
        columns = [f"{name}_{c}" for c in features]
        frame = pd.DataFrame(centred[rows], columns=columns)
        frame["guide"] = cells[guide].to_numpy()[rows]
        frame["gene"] = cells[gene].to_numpy()[rows]
        grouped = frame.groupby(["guide", "gene"], sort=True)
        table = grouped[columns].median()
        table.insert(0, f"{name}_n_cells", grouped.size())
        parts.append(table)
    guides = pd.concat(parts, axis=1).reset_index()
    guides = guides[~guides["gene"].isin(list(drop_genes))].reset_index(drop=True)
    columns = [f"{name}_{c}" for name in phases for c in features]
    genes = guides.groupby("gene", sort=True)[columns].median().reset_index()
    return guides, genes


def phase_columns(table: pd.DataFrame, profile: str) -> List[str]:
    """The profile columns ``ops_profile_cell_cycle`` keeps.

    :param table: a :func:`phase_profiles` table.
    :param profile: ``'concat'`` for every phase, or one phase name.
    :returns: the ``<phase>_<feature>`` columns of the chosen phases.
    :raises ValueError: for an unknown profile.
    """
    profile = str(profile).strip()
    if profile not in CELL_CYCLE_PROFILES:
        raise ValueError(f"the cell-cycle profile must be one of "
                         f"{list(CELL_CYCLE_PROFILES)}; got {profile!r}")
    chosen = CELL_CYCLE_PHASES if profile == "concat" else (profile,)
    return [c for c in table.columns
            if any(str(c).startswith(f"{p}_") for p in chosen)
            and not str(c).endswith("_n_cells")]

