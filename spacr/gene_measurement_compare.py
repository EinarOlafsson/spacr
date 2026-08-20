"""Compare a measurement between annotated genes and the rest.

Instruction 177 F, asked for 2026-08-19:

    "there should be the opertunity to show the any measurement comparing the
    values for the cells that have been annotated with a gene vs the rest,
    this should also work for several annotated genes at a time (as the cell
    picker has logic for doing this). the option to do this on the cell, well,
    and plate level should be available. the graphing options should be
    bar/box/jitter/jitter-box/ and violin plot. ther e should also be the
    ability to do statistics. where the variance and normality and n is noted
    and the correct test chosen. then the option to save should be available"

THREE THINGS THIS MODULE DOES NOT DO, EACH ON PURPOSE.

It does not decide which cells carry which gene. `spacr.cell_montage` already
does -- rank, attributed, assigned, multivariate -- and a second answer here
would be a second definition of "this cell is a 220950 cell".

It does not choose the test. `spacr.sp_stats.perform_statistical_tests` does,
from the normality and equal-variance checks, and it already reports n per
group, the effect size and WHY THAT TEST. Writing a second chooser would be
the one-vocabulary failure (145) in its most dangerous form: two functions
that answer "which test" differently, in a figure legend.

It does not invent a palette. `gene_measurement_sweep.HOUSE` carries the
apicomplexan-genomics one, and the rule that everything is grey except the
claim applies here exactly: the rest of the screen is grey and the annotated
genes are the argument.

THE LEVEL CHANGES WHAT AN OBSERVATION IS, and that is the whole reason it is
a setting rather than a detail. A per-CELL test on 60,000 cells drawn from 12
wells has 60,000 rows and 12 independent units; the p it produces is not
about the genes. Every result carries the level it was computed at so a
reader can see which one they are looking at.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The levels an observation can be, and what one row means at each.
LEVELS: Tuple[Tuple[str, str], ...] = (
    ("cell", "one row per cell — the most rows and the fewest independent "
             "units, so a p-value here is about cells and not about wells"),
    ("well", "one row per well — the unit the screen randomises, and the "
             "honest default for a well-level design"),
    ("plate", "one row per plate — very few rows, and the only level that "
              "removes plate as a confounder entirely"),
)

#: How the comparison can be drawn.
PLOTS: Tuple[Tuple[str, str], ...] = (
    ("jitter_box", "jitter over box — every point, and the summary behind it"),
    ("box", "box"),
    ("jitter", "jitter — every point"),
    ("violin", "violin — the shape of the distribution"),
    ("bar", "bar — mean and error, and the least informative of the five"),
)

#: What the rest of the screen is called in the plot and in the table.
REST = "the rest"


@dataclass
class Comparison:
    """One measurement compared between groups, at one level."""

    measurement: str
    level: str
    frame: pd.DataFrame            # long: group, value, and the unit id
    statistics: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def groups(self) -> Tuple[str, ...]:
        if not len(self.frame):
            return ()
        return tuple(self.frame["group"].astype(str).unique())

    def counts(self) -> Dict[str, int]:
        """n per group -- the number the legend has to state."""
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
          value_column: Optional[str] = None) -> Comparison:
    """Assemble the long frame this comparison is drawn and tested from.

    :param objects: per-object rows, carrying ``measurement`` and whichever
        identity columns ``level`` needs.
    :param groups: ``{name: the object index values that carry it}`` -- what
        the cell picker decided. Everything not named lands in
        :data:`REST`.
    :param level: one of :data:`LEVELS`.
    :param value_column: the column to read; ``measurement`` by default.

    :returns: a :class:`Comparison`. AGGREGATION IS THE MEAN at well and
        plate level, and it is the mean of the cells IN THAT GROUP -- a well
        holding both a 220950 cell and an unpicked one contributes a row to
        each, because the alternative is to decide the well belongs to
        whichever group happens to be larger.
    """
    column = str(value_column or measurement)
    if column not in getattr(objects, "columns", ()):
        return Comparison(measurement=measurement, level=level,
                          frame=pd.DataFrame(columns=["group", "value"]),
                          note=f"{column!r} is not a column on these objects")

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
    if level == "cell":
        units = int(work["unit"].nunique())
        note = (f"one row per CELL: {units:,} rows, and every cell from one "
                f"well shares that well's treatment — so these rows are not "
                f"independent and the p-value is about cells, not wells")
    return Comparison(measurement=measurement, level=level, frame=work,
                      note=note)


def with_statistics(comparison: Comparison) -> Comparison:
    """Run the comparison spaCR's own engine chooses, and record why.

    NOT CALLED `test`. A public function of that name is collected by pytest
    the moment any test module imports it, and it then fails asking for a
    `comparison` fixture -- an error in the library's own name, reported
    against a file that is not a test.

    `sp_stats.perform_statistical_tests` picks between Student's t, Welch's t,
    Mann-Whitney U, one-way ANOVA, Welch's ANOVA and Kruskal-Wallis from the
    normality and equal-variance checks, and reports n per group, the effect
    size and its own reason. NOTHING HERE SECOND-GUESSES IT: a second chooser
    would answer "which test" differently in a figure legend, which is the
    one-vocabulary failure at its most dangerous.
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
    """Draw the comparison. Returns the Figure, or ``None`` if there is none.

    THE APICOMPLEXAN IDIOM, from `gene_measurement_sweep.HOUSE`: the rest of
    the screen is GREY and the annotated genes are the argument. And for
    n <= 8 the skill is explicit that a bar chart is not done in these
    papers -- so `bar` is offered because it was asked for, and it says on
    the panel when the group is small enough that dots would be honest.
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
        # SAID, NOT OVERRIDDEN. The user asked for a bar and gets one; the
        # skill's rule -- "a bar for n = 3 is not done in these papers" -- is
        # reported so the choice is informed rather than silently corrected.
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
    """Write everything behind this comparison into ONE folder.

    Asked for in full: "this should save the graph as pdf and png, the data
    that underlies the graph, the settings that generated the regression and
    the graph, the images of cells if available in well folders, and the
    statistics. all of this in one folder upon saving."

    ONE FOLDER, because the point of saving is to be able to come back to it
    -- a figure in one place, its numbers in another and the settings that
    produced both in a third is how a result becomes unreproducible six
    months later. The PDF is the one to put in a paper; the PNG is the one to
    paste into a message.

    :param images: ``{well: [array, ...]}`` -- written under ``cells/<well>/``
        where they exist. Absent or empty is normal and not an error: a
        comparison run on a screen whose crops are not to hand is still a
        comparison.
    :returns: ``{what: path}`` for everything actually written.
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
