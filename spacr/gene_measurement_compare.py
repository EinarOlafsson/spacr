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

#: Supported observation levels and the meaning of one result-table row.
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
          second: str = "") -> Comparison:
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
        work[key] = objects[key].astype(str)
    work = work.dropna(subset=["value"])

    if level == "cell":
        work["unit"] = work.index.astype(str)
    else:
        # ONE ROW PER (unit, group), not per unit: see the docstring.
        work["unit"] = work[keys].agg("_".join, axis=1)
        work = (work.groupby(["unit", "group"], as_index=False)["value"]
                .mean())

    note = ""
    if dropped:
        # SAID, ALWAYS. A comparison quietly computed on fewer rows than the
        # user thinks is the kind of result that survives review and is
        # wrong.
        note = (f"{dropped:,} row(s) left out: the denominator was zero or "
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
        figure.savefig(path, dpi=200, bbox_inches="tight")
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
        for suffix in ("pdf", "png"):
            path = os.path.join(folder, f"comparison.{suffix}")
            try:
                figure.savefig(path, dpi=300, bbox_inches="tight")
                written[suffix] = path
            except Exception:                                # noqa: BLE001
                continue

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
