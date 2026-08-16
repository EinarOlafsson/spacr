"""Every regression panel, drawn in the house style.

One function per panel. Each takes an Axes and a results frame, draws into
it, and returns a :class:`Panel` record saying what it drew and what the
reader has to be told — so the sheet can caption it and the GUI can list it
without a second copy of that knowledge.

WHY THE PANELS ARE FUNCTIONS AND NOT A CLASS HIERARCHY: a panel is a pure
drawing step with no state worth keeping. Making one is `panel(ax, frame)`,
and the registry below is the only place that knows the full set.

THE RULE EVERY PANEL FOLLOWS, from the skill: everything is grey except what
the sentence is about. The volcano's non-significant guides are grey; the
called ones carry the colour. A panel that colours everything has no claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .style import (ROLES, TYPE_SCALE, Palette, annotate, descriptor,
                    panel_letter, reference_line, rotate_ticks, text_legend)


@dataclass
class Panel:
    """What a panel drew, and what a reader needs told about it."""

    key: str
    #: Two to four lower-case words. Not a sentence.
    title: str
    #: One sentence for the figure legend. States the test, the n and the
    #: convention where a statistic was drawn -- the skill is explicit that a
    #: bare p is not acceptable.
    caption: str = ""
    #: False when the data for this panel was not present. The sheet skips
    #: it and says so rather than drawing an empty frame.
    drawn: bool = True
    #: Why it could not be drawn, when it could not.
    reason: str = ""
    #: Columns the panel needed. Used by the registry to answer "what can I
    #: draw from this table" without drawing anything.
    needs: Sequence[str] = field(default_factory=tuple)


def _column(frame, *names) -> Optional[str]:
    """The first of ``names`` this frame has."""
    for name in names:
        if name in frame.columns:
            return name
    return None


def effect_column(frame) -> Optional[str]:
    """The fitted effect column, whatever this backend called it.

    spaCR writes ``coefficient``; a statsmodels summary carries ``coef``.
    Both are the same quantity and both have to plot.
    """
    return _column(frame, "coefficient", "coef", "effect", "estimate")


def p_column(frame) -> Optional[str]:
    """The raw p-value column, whatever this backend calls it."""
    return _column(frame, "p_value", "p", "pvalue", "P>|z|", "P>|t|")


def q_column(frame) -> Optional[str]:
    """The multiple-testing-corrected p, if the fit produced one.

    Its ABSENCE is meaningful and is not an error: the penalised backends
    have no p-value to correct, and a run asked for ``multiple_testing_method
    = 'none'`` deliberately has none either. A panel that finds no q falls
    back to the raw p and says which one it drew.
    """
    return _column(frame, "q_value", "adjusted_p_value", "fdr", "qval")


def statistic_column(frame) -> Optional[str]:
    """The test statistic, which differs by backend.

    OLS and RLM report ``t value``; GLM, Poisson and the mixed model report
    ``z value``; the penalised backends report neither and give a selection
    frequency instead. Every one of those tables has to display.
    """
    return _column(frame, "t_value", "z_value", "t value", "z value",
                   "statistic", "selection_frequency")


def tested(frame) -> np.ndarray:
    """Rows that are hypotheses, via the repo's single statement of it."""
    from ..hits import tested_family

    if "feature" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return tested_family(frame["feature"])


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    return np.where(np.isfinite(array), array, np.nan)


def label_series(frame):
    """A readable name per row, with no holes.

    `gene` is NaN on the per-guide rows and `grna` is NaN on the per-gene
    rows, so either column alone labels half the volcano "nan" -- which is
    what a first pass at this actually drew. Coalesce them, and fall back to
    the design term with its boilerplate stripped.
    """
    import pandas as pd

    parts = [frame[name] for name in ("gene", "grna")
             if name in frame.columns]
    if parts:
        combined = parts[0].astype("object")
        for extra in parts[1:]:
            combined = combined.where(combined.notna(), extra)
        if "feature" in frame.columns:
            combined = combined.where(combined.notna(), frame["feature"])
        return combined.astype(str).str.replace(
            r"^(gene_)?fraction:(gene|grna)\[|\]$", "", regex=True)
    if "feature" in frame.columns:
        return frame["feature"].astype(str).str.replace(
            r"^(gene_)?fraction:(gene|grna)\[|\]$", "", regex=True)
    return pd.Series([str(i) for i in range(len(frame))], index=frame.index)


# --------------------------------------------------------------------------- #
#  The result
# --------------------------------------------------------------------------- #

def volcano(ax, frame, *, alpha=0.05, effect_threshold=None,
            highlight=None, label_top=8) -> Panel:
    """Effect against significance. The panel the screen exists to produce.

    Grey for everything that was not called, GREEN up, RUST down -- the
    skill's rule for differential expression, which is structurally the same
    question. A handful of the strongest are labelled; labelling all of them
    is a word cloud.
    """
    effect, p = effect_column(frame), p_column(frame)
    if effect is None or p is None:
        return Panel("volcano", "volcano", drawn=False,
                     reason="no effect or p-value column",
                     needs=("coefficient", "p_value"))

    keep = tested(frame)
    sub = frame.loc[keep]
    x = _finite(sub[effect])
    raw = _finite(sub[p])
    smallest = np.nanmin(raw[raw > 0]) if np.any(raw > 0) else 1e-300
    y = -np.log10(np.clip(raw, smallest * 1e-3, 1.0))

    q = q_column(frame)
    called = _finite(sub[q]) <= alpha if q else raw <= alpha
    called = np.nan_to_num(called, nan=False).astype(bool)
    if effect_threshold:
        called &= np.abs(x) >= abs(effect_threshold)

    up = called & (x > 0)
    down = called & (x < 0)
    rest = ~called

    ax.scatter(x[rest], y[rest], s=4.0, c=ROLES["data"], linewidths=0,
               rasterized=True, zorder=1)
    ax.scatter(x[up], y[up], s=7.0, c=ROLES["up"], linewidths=0, zorder=2)
    ax.scatter(x[down], y[down], s=7.0, c=ROLES["down"], linewidths=0, zorder=2)

    reference_line(ax, y=-np.log10(alpha), label=f"q = {alpha:g}" if q
                   else f"p = {alpha:g}")
    if effect_threshold:
        for sign in (-1, 1):
            reference_line(ax, x=sign * abs(effect_threshold))

    names = label_series(sub)
    if label_top:
        # Label the strongest, and only where a label would not land on
        # another one. A volcano with every hit labelled is a word cloud.
        order = np.argsort(-np.nan_to_num(np.where(called, y, 0.0), nan=0.0))
        placed = []
        for index in order[: label_top * 3]:
            if not called[index] or len(placed) >= label_top:
                continue
            name = names.iloc[index]
            if not name or name.lower() == "nan":
                continue
            if any(abs(x[index] - px) < 0.42 and abs(y[index] - py) < 0.9
                   for px, py in placed):
                continue
            placed.append((x[index], y[index]))
            ax.annotate(name, (x[index], y[index]),
                        fontsize=TYPE_SCALE["annotation"],
                        xytext=(3, 2), textcoords="offset points",
                        color=ax.xaxis.label.get_color())

    if highlight is not None:
        mask = names.astype(str) == str(highlight)
        if mask.any():
            ax.scatter(x[mask.to_numpy()], y[mask.to_numpy()], s=44,
                       facecolors="none", edgecolors=ROLES["highlight"],
                       linewidths=1.4, zorder=3)

    ax.set_xlabel("effect size")
    ax.set_ylabel("$-$log$_{10}$ " + ("q" if q else "p"))
    text_legend(ax, [(f"{int(up.sum())} up", ROLES["up"]),
                     (f"{int(down.sum())} down", ROLES["down"]),
                     (f"{int(rest.sum())} not called", ROLES["data"])])
    return Panel("volcano", "volcano",
                 caption=(f"Effect size against significance for "
                          f"{int(keep.sum())} tested coefficients. "
                          f"{int(called.sum())} called at "
                          f"{'BH q' if q else 'p'} ≤ {alpha:g}"
                          + (f" and |effect| ≥ {abs(effect_threshold):g}"
                             if effect_threshold else "")
                          + ". Nuisance terms are excluded."),
                 needs=(effect, p))


def effect_rank(ax, frame, *, alpha=0.05, top=14) -> Panel:
    """Every gene ranked by effect, as a dot with its interval.

    The skill's rule for n <= 8: individual points with a mean line, never a
    bar. Here the same logic one level up -- a coefficient with a confidence
    interval is a point with a range, and a bar chart of coefficients hides
    the uncertainty that decides whether to believe any of them.
    """
    effect = effect_column(frame)
    if effect is None:
        return Panel("effect_rank", "effect by rank", drawn=False,
                     reason="no effect column", needs=("coefficient",))
    keep = tested(frame)
    sub = frame.loc[keep].copy()
    sub["_abs"] = sub[effect].abs()
    sub = sub.sort_values("_abs", ascending=False).head(top).iloc[::-1]

    names = label_series(sub)
    y = np.arange(len(sub))
    x = _finite(sub[effect])

    q = q_column(frame)
    called = (_finite(sub[q]) <= alpha) if q else np.zeros(len(sub), bool)
    called = np.nan_to_num(called, nan=False).astype(bool)
    colours = np.where(called, np.where(x > 0, ROLES["up"], ROLES["down"]),
                       ROLES["data"])

    se = _column(frame, "std_err", "std err", "se", "bse")
    if se:
        half = 1.96 * _finite(sub[se])
        ax.hlines(y, x - half, x + half, colors=colours,
                  linewidth=1.0, zorder=1)
    ax.scatter(x, y, s=14, c=colours, linewidths=0, zorder=2)
    reference_line(ax, x=0.0)

    # NAMES INSIDE THE PANEL, not on the axis. A y-tick label is drawn
    # outside the axes, so a long gene id in one cell of a sheet reaches into
    # the cell to its left -- which is what the first pass did to panel A.
    # Inside, each name sits against its own dot and cannot collide with a
    # neighbouring panel at any width.
    ax.set_yticks([])
    span = float(np.nanmax(np.abs(x))) or 1.0
    for row, (value, name) in enumerate(zip(x, names)):
        ax.text(0.04 * span if value < 0 else -0.04 * span, row,
                str(name)[:20],
                ha="left" if value < 0 else "right", va="center",
                fontsize=TYPE_SCALE["annotation"],
                color=ax.xaxis.label.get_color())
    ax.set_ylim(-0.8, len(sub) - 0.2)
    ax.set_xlim(-1.35 * span, 1.35 * span)
    ax.set_xlabel("effect size")
    ax.set_ylabel("")
    return Panel("effect_rank", "strongest effects",
                 caption=(f"The {len(sub)} coefficients with the largest "
                          f"absolute effect"
                          + (", with 95% intervals" if se else "")
                          + ". Coloured where called; grey otherwise."),
                 needs=(effect,))


def p_histogram(ax, frame, *, bins=40) -> Panel:
    """The single most informative check that a correction means anything.

    Under the null p is uniform. Flat with a spike at zero is a screen with
    real hits; a slope, a hump in the middle or a spike at one is a model
    that is wrong, and no amount of FDR fixes it.
    """
    p = p_column(frame)
    if p is None:
        return Panel("p_histogram", "p-value distribution", drawn=False,
                     reason="this backend reports no p-value",
                     needs=("p_value",))
    values = _finite(frame.loc[tested(frame), p])
    values = values[np.isfinite(values)]
    if not values.size:
        return Panel("p_histogram", "p-value distribution", drawn=False,
                     reason="no finite p-values", needs=(p,))

    ax.hist(values, bins=bins, range=(0, 1), color=ROLES["fill"],
            edgecolor="none")
    expected = values.size / bins
    reference_line(ax, y=expected, label="uniform")
    ax.set_xlabel("p")
    ax.set_ylabel("coefficients")

    # The shape, stated. A reader should not have to judge flatness by eye.
    upper = float(np.mean(values > 0.5)) * 2.0
    annotate(ax, f"n = {values.size}\nπ₀ ≈ {min(upper, 1):.2f}")
    return Panel("p_histogram", "p-value distribution",
                 caption=("Raw p-values for the tested coefficients. Under "
                          "the null these are uniform; the dashed line is "
                          "that expectation. π₀ estimates the "
                          "fraction of true nulls from the upper half."),
                 needs=(p,))


def qq_plot(ax, frame) -> Panel:
    """Observed against expected p, and the inflation factor.

    A screen whose points leave the diagonal early has either real signal or
    a mis-specified model, and lambda says which is more likely.
    """
    p = p_column(frame)
    if p is None:
        return Panel("qq", "q-q of p-values", drawn=False,
                     reason="this backend reports no p-value",
                     needs=("p_value",))
    values = _finite(frame.loc[tested(frame), p])
    values = np.sort(values[np.isfinite(values) & (values > 0)])
    if values.size < 3:
        return Panel("qq", "q-q of p-values", drawn=False,
                     reason="too few p-values", needs=(p,))

    n = values.size
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(values)
    ax.scatter(expected, observed, s=3.5, c=ROLES["data"], linewidths=0,
               rasterized=True)
    limit = float(max(expected.max(), observed.max()))
    ax.plot([0, limit], [0, limit], color=ROLES["reference"],
            lw=0.6, ls=(0, (4, 3)), zorder=0)

    from scipy.stats import chi2

    chi = chi2.isf(values, 1)
    lam = float(np.median(chi) / chi2.ppf(0.5, 1))
    annotate(ax, f"λ = {lam:.2f}")
    ax.set_xlabel("expected $-$log$_{10}$ p")
    ax.set_ylabel("observed $-$log$_{10}$ p")
    return Panel("qq", "q-q of p-values",
                 caption=(f"Observed against expected p-values on the "
                          f"−log₁₀ scale. λ = {lam:.2f}; "
                          f"a value near 1 means the null is calibrated, and "
                          f"a large one means inflation rather than signal."),
                 needs=(p,))


def control_separation(ax, frame) -> Panel:
    """The assay window: do the controls actually separate?

    Individual points with a mean line, never a bar -- the skill is explicit,
    and a bar for a handful of controls hides the spread that decides whether
    the screen worked at all.
    """
    effect = effect_column(frame)
    condition = _column(frame, "condition", "control", "class")
    if effect is None or condition is None:
        return Panel("controls", "control separation", drawn=False,
                     reason="no condition column", needs=("condition",))

    names = {"nc": "negative", "pc": "positive", "control": "control",
             "other": "screen"}
    colours = {"negative": ROLES["control_negative"],
               "positive": ROLES["control_positive"],
               "control": Palette.GREY_DARK, "screen": ROLES["data"]}
    groups, labels, shades = [], [], []
    for key, label in names.items():
        rows = frame[frame[condition].astype(str) == key]
        if len(rows):
            groups.append(_finite(rows[effect]))
            labels.append(label)
            shades.append(colours[label])
    if len(groups) < 2:
        return Panel("controls", "control separation", drawn=False,
                     reason="fewer than two conditions present",
                     needs=(condition,))

    rng = np.random.default_rng(0)
    for index, (values, colour) in enumerate(zip(groups, shades)):
        values = values[np.isfinite(values)]
        jitter = rng.uniform(-0.09, 0.09, values.size)
        ax.scatter(np.full(values.size, index) + jitter, values, s=5,
                   c=colour, linewidths=0, alpha=.75, zorder=1)
        ax.hlines(np.median(values), index - 0.22, index + 0.22,
                  colors=colour, linewidth=1.4, zorder=2)
    reference_line(ax, y=0.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("effect size")
    annotate(ax, "  ".join(f"{label} n={len(g)}"
                           for label, g in zip(labels, groups)))
    return Panel("controls", "control separation",
                 caption=("Effect size by control class. Points are "
                          "individual coefficients, the bar is the median. "
                          "The screen works only if the positive controls "
                          "separate from the negative ones."),
                 needs=(effect, condition))


def guide_agreement(ax, frame) -> Panel:
    """Per gene: do its own guides push the same way?

    The one thing a volcano structurally cannot show. A gene called by one
    guide out of six is the commonest way a pooled screen makes a confident
    artefact, and it is the same dot as a gene whose guides agree.
    """
    from ..hits import gene_of

    effect = effect_column(frame)
    if effect is None or "feature" not in frame.columns:
        return Panel("agreement", "guide agreement", drawn=False,
                     reason="no per-guide coefficients",
                     needs=("feature", "coefficient"))
    sub = frame.loc[tested(frame)].copy()
    sub["_gene"] = [gene_of(value) for value in sub["feature"]]
    sub = sub[sub["_gene"].notna()]
    if not len(sub):
        return Panel("agreement", "guide agreement", drawn=False,
                     reason="no gene could be parsed from the terms",
                     needs=("feature",))

    grouped = sub.groupby("_gene")[effect]
    counts = grouped.size()
    concordance = grouped.apply(
        lambda values: max((values > 0).mean(), (values < 0).mean()))
    size = grouped.apply(lambda values: values.abs().mean())

    multi = counts >= 2
    # JITTERED, because guides-per-gene is an integer and agreement is a
    # small set of fractions: without it several hundred genes stack into a
    # dozen dots and the panel looks like it has no data in it.
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.22, 0.22, len(counts))
    ax.scatter(counts[multi] + jitter[multi.to_numpy()],
               concordance[multi], s=7, c=ROLES["data"], linewidths=0,
               alpha=.55, zorder=1, rasterized=True)
    if (~multi).any():
        ax.scatter(counts[~multi] + jitter[(~multi).to_numpy()],
                   concordance[~multi], s=11, c=ROLES["down"],
                   linewidths=0, zorder=2)
    reference_line(ax, y=0.5, label="chance")
    ax.set_xlabel("guides per gene")
    ax.set_ylabel("fraction agreeing in sign")
    ax.set_ylim(0.28, 1.08)
    single = int((~multi).sum())
    entries = [(f"{len(counts)} genes", ROLES["data"])]
    if single:
        entries.append((f"{single} on a single guide", ROLES["down"]))
    text_legend(ax, entries, y=0.22)
    return Panel("agreement", "guide agreement",
                 caption=("For each gene, how many of its guides agree in "
                          "direction. Genes carried by a single guide are "
                          "highlighted: nothing corroborates them, and they "
                          "are indistinguishable from agreement on a "
                          "volcano."),
                 needs=("feature", effect))


def effect_distribution(ax, frame) -> Panel:
    """Where the screen's effects sit, and where the controls sit in them."""
    effect = effect_column(frame)
    if effect is None:
        return Panel("effect_distribution", "effect distribution", drawn=False,
                     reason="no effect column", needs=("coefficient",))
    values = _finite(frame.loc[tested(frame), effect])
    values = values[np.isfinite(values)]
    if not values.size:
        return Panel("effect_distribution", "effect distribution", drawn=False,
                     reason="no finite effects", needs=(effect,))

    ax.hist(values, bins=50, color=ROLES["fill"], edgecolor="none")
    reference_line(ax, x=0.0)
    mad = float(np.median(np.abs(values - np.median(values))) * 1.4826)
    for sign in (-1, 1):
        reference_line(ax, x=sign * 3 * mad,
                       label="3σ" if sign > 0 else "")
    ax.set_xlabel("effect size")
    ax.set_ylabel("coefficients")
    annotate(ax, f"n = {values.size}\nσ (MAD) = {mad:.3g}")
    return Panel("effect_distribution", "effect distribution",
                 caption=("Distribution of fitted effects for the tested "
                          "coefficients. σ is a MAD-based estimate, "
                          "which the outliers a screen is looking for do not "
                          "inflate; the dashed lines are ±3σ."),
                 needs=(effect,))


#: The catalog. The GUI lists this, the sheet lays it out, and a panel that
#: is not here does not exist as far as either is concerned -- which is the
#: point: one place knows the full set.
REGISTRY: Dict[str, Callable] = {
    "volcano": volcano,
    "effect_rank": effect_rank,
    "effect_distribution": effect_distribution,
    "controls": control_separation,
    "agreement": guide_agreement,
    "p_histogram": p_histogram,
    "qq": qq_plot,
}

#: Reading order for the sheet: result first, then what it rests on, then
#: whether the model was entitled to say it. The skill's rule -- reading
#: order matches the argument.
SHEET_ORDER = ("volcano", "effect_rank", "effect_distribution", "controls",
               "agreement", "p_histogram", "qq")


def available(frame) -> Dict[str, bool]:
    """Which panels this table can support, without drawing anything."""
    import matplotlib.pyplot as plt

    answer = {}
    figure = plt.figure()
    try:
        for key in SHEET_ORDER:
            ax = figure.add_subplot(111)
            try:
                answer[key] = bool(REGISTRY[key](ax, frame).drawn)
            except Exception:
                answer[key] = False
            figure.clear()
    finally:
        plt.close(figure)
    return answer


__all__ = ["Panel", "REGISTRY", "SHEET_ORDER", "available", "effect_column",
           "p_column", "q_column", "statistic_column", "volcano",
           "effect_rank", "effect_distribution", "control_separation",
           "guide_agreement", "p_histogram", "qq_plot"]
