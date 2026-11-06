"""Publication panels for plate assays: statistics, drawing, saving and figures.

Used by :func:`spacr.submodules.analyze_recruitment` and
:func:`spacr.submodules.analyze_replication` (``panels`` / ``figures``
settings), and usable on any table with one row per object.

**Experimental unit.** Objects are averaged per well
(``plateID``, ``rowID``, ``columnID``), and well means are averaged per
replicate when a replicate column is given (:func:`aggregate_to_unit`). Tests
run on these unit values, never on individual cells, so the n of a test is the
number of wells or replicates.

**Test selection for bar panels** (:func:`select_and_run_test`, one test per
panel):

* Shapiro-Wilk normality per group (groups with fewer than 3 values or no
  spread count as normal) and Levene equal variance (median centred, the
  Brown-Forsythe variant);
* every group normal and equal variance: one-way ANOVA, post hoc Tukey HSD;
* every group normal, unequal variance: Welch ANOVA, post hoc Games-Howell;
* any group non-normal: Kruskal-Wallis, post hoc Dunn test with Holm
  correction.

**Proportion panels** (:func:`proportion_tests`): object counts per bin,
pooled over replicates, compared with a chi-square test of independence or
the Fisher-Freeman-Halton exact test (Monte Carlo over tables with the
observed margins; the Fisher exact test for a 2 x 2 table). The omnibus test
covers every group; each group is also compared with the reference group
(Holm correction). Cramer V is the effect size. These tests treat every object
as independent, so with thousands of objects they flag small shifts; the
replicate-level bar panel drawn next to them is the conservative test.

**Panel layout on disk** (:func:`save_panel`): one folder per panel,
``<out_dir>/<Pnn>_<name>/`` holding ``<Pnn>_<name>.pdf`` and ``.png``,
``data_replicates.csv`` (the plotted unit values), ``stats_test.csv``,
``stats_assumptions.csv``, ``stats_posthoc.csv`` and ``stats_descriptive.csv``
for bar panels, and ``contingency_table_counts.csv`` for stacked panels.

**Aligned order.** Every panel takes ``order`` (the x-axis order) from
:func:`condition_order`, so a condition sits at the same place in every panel
of a figure, and ``labels`` (raw value -> display label; underscores become
spaces when no label is given). ``annotations`` draws aligned text rows under
the x axis, for example a +/- table of what each condition contains.

Pure functions plus matplotlib; nothing here opens a window.
"""
from __future__ import annotations

import itertools
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .tabular import write_table

__all__ = [
    "ALPHA", "stars", "fmt_p", "condition_order", "display_label",
    "select_and_run_test", "proportion_tests", "aggregate_to_unit",
    "fractions_per_unit", "bar_spec", "stacked_spec", "draw_bar_panel",
    "draw_stacked_panel", "draw_panel", "save_panel", "assemble_figure",
    "panel_folder_name", "STYLE", "shade", "style_axes", "outlier_policy", "mark_outliers",
    "parse_outlier_method", "OUTLIER_FENCES", "condition_parts", "string_parts", "axis_layout",
    "select_bracket_pairs", "pack_brackets",
]

ALPHA = 0.05
WELL_KEYS: Tuple[str, ...] = ("plateID", "rowID", "columnID")
_DEFAULT_COLORS = ["#4C72B0", "#2A9D8F", "#8C8C8C", "#264653", "#7FA7C9", "#A8DADC",
                   "#5E548E", "#B5838D"]
STYLE = dict(
    palette=_DEFAULT_COLORS,
    bar_alpha=0.6,
    edge_shade=0.6,
    edge_lw=0.8,
    bar_width=0.7,
    dot_size=22.0,
    dot_alpha=0.65,
    dot_shade=0.55,
    dot_edge="white",
    dot_edge_lw=0.35,
    jitter=0.12,
    err_color="#333333",
    err_lw=0.8,
    capsize=2.0,
    bracket_lw=0.7,
    grid_alpha=0.3,
    grid_lw=0.5,
    stacked_cmap="viridis",
    stacked_edge="white",
    stacked_lw=0.5,
    show_test=False,
    show_title=False,
    annot_pt=7.0,
    annot_line_pt=9.0,
)


def shade(color, factor):
    """``color`` darkened to ``factor`` (0 black, 1 unchanged), as an RGB tuple.

    :param color: any matplotlib colour specification (name, hex string or
        RGB(A) tuple).
    :param factor: brightness multiplier applied to each RGB channel; results
        are clipped to [0, 1].
    :returns: ``(r, g, b)`` tuple of floats in [0, 1].
    """
    from matplotlib.colors import to_rgb
    return tuple(max(0.0, min(1.0, c * factor)) for c in to_rgb(color))


def style_axes(ax):
    """Light horizontal grid behind the data, top and right spines removed.

    :param ax: matplotlib ``Axes`` styled in place.
    """
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=STYLE["grid_alpha"], linewidth=STYLE["grid_lw"])
    ax.xaxis.grid(False)
    _despine(ax)


_ANNOT_LINE_PT = 11.0


OUTLIER_FENCES = {"mad": 3.5, "iqr": 1.5}
_OUTLIER_POLICY = {"method": None}


def parse_outlier_method(method):
    """``'mad'`` / ``'iqr'`` (default fences 3.5 and 1.5) or ``'mad:3'`` -> (name, fence); None -> None.

    :param method: ``'mad'``, ``'iqr'``, ``'<name>:<fence>'``, ``True``
        (same as ``'mad'``), or ``None`` / ``False`` / ``'none'`` / ``''`` for
        no outlier marking.
    :returns: ``(name, fence)`` with ``fence`` a float, or ``None``.
    :raises ValueError: if the method name is not ``'mad'`` or ``'iqr'``.
    """
    if method in (None, False, "none", ""):
        return None
    if method is True:
        method = "mad"
    name, _, fence = str(method).partition(":")
    name = name.strip().lower()
    if name not in OUTLIER_FENCES:
        raise ValueError(f"outlier method must be 'mad' or 'iqr' (optionally 'mad:3.0'), got {method!r}")
    return name, float(fence) if fence else OUTLIER_FENCES[name]


class outlier_policy:
    """Context manager: default outlier method of :func:`bar_spec` inside the block.

    ``with outlier_policy('mad'): ...`` makes every bar panel built inside
    mark and leave out points beyond the robust fence of their group.

    :param method: outlier method made the default inside the block, in any
        form accepted by :func:`parse_outlier_method` (``None`` turns marking
        off); the previous default is restored on exit.
    """

    def __init__(self, method):
        self.method = method

    def __enter__(self):
        self.previous = _OUTLIER_POLICY["method"]
        _OUTLIER_POLICY["method"] = self.method
        return self

    def __exit__(self, *exc):
        _OUTLIER_POLICY["method"] = self.previous
        return False


def mark_outliers(data: pd.DataFrame, value: str, group, method) -> pd.Series:
    """True for points beyond the robust fence of their group.

    ``mad``: ``|x - median| > fence * 1.4826 * MAD``; ``iqr``: outside
    ``[Q1 - fence * IQR, Q3 + fence * IQR]``. Groups with fewer than 4 points
    or no spread (MAD or IQR 0) keep every point.

    :param data: table with one row per point.
    :param value: name of the numeric column tested; non-numeric and
        infinite values are ignored.
    :param group: column or list of columns defining the groups.
    :param method: outlier method in any form accepted by
        :func:`parse_outlier_method`; ``None`` marks nothing.
    :returns: boolean Series aligned to ``data.index``.
    :raises ValueError: if ``method`` names an unknown outlier method.
    """
    parsed = parse_outlier_method(method)
    out = pd.Series(False, index=data.index)
    if parsed is None or not len(data):
        return out
    name, fence = parsed
    v = pd.to_numeric(data[value], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keys = [group] if isinstance(group, str) else list(group)
    for _, idx in data.groupby(keys, observed=True, dropna=False).groups.items():
        x = v.loc[idx].dropna()
        if len(x) < 4:
            continue
        if name == "mad":
            med = x.median()
            mad = (x - med).abs().median()
            if not mad > 0:
                continue
            flag = (x - med).abs() > fence * 1.4826 * mad
        else:
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            if not iqr > 0:
                continue
            flag = (x < q1 - fence * iqr) | (x > q3 + fence * iqr)
        out.loc[flag.index[flag.to_numpy()]] = True
    return out


def outlier_table(data, group, order, method, by=None):
    """Points and removed outliers per group (and per ``by`` level).

    :param data: table with the ``group`` column and a boolean ``outlier``
        column (see :func:`mark_outliers`).
    :param group: column of ``data`` with the condition.
    :param order: conditions of the panel; not used to filter or sort the rows.
    :param method: outlier method used, in any form accepted by
        :func:`parse_outlier_method`; recorded in the ``method`` and ``fence``
        columns.
    :returns: one row per group with ``n_points``, ``n_removed``, ``method``
        and ``fence``.
    """
    parsed = parse_outlier_method(method)
    keys = ([by] if by else []) + [group]
    t = data.groupby(keys, observed=True).agg(n_points=("outlier", "size"),
                                             n_removed=("outlier", "sum")).reset_index()
    t["n_removed"] = t["n_removed"].astype(int)
    t["method"] = parsed[0] if parsed else ""
    t["fence"] = parsed[1] if parsed else np.nan
    return t


def stars(p) -> str:
    """Significance stars for a p value ('ns' at or above 0.05, 'n/a' if missing).

    :param p: the p value; anything that is not a finite number gives ``'n/a'``.
    :returns: ``'****'`` (p < 1e-4), ``'***'``, ``'**'``, ``'*'`` (p < 0.05) or ``'ns'``.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    return ("****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2
            else "*" if p < ALPHA else "ns")


def fmt_p(p) -> str:
    """Format a p value for a panel title.

    :param p: the p value; anything that is not a finite number gives ``'p = n/a'``.
    :returns: ``'p < 1e-4'`` or ``'p = '`` followed by three significant digits.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "p = n/a"
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 1e-4" if p < 1e-4 else f"p = {p:.3g}"


def _natural_key(value):
    """Sort key that orders embedded numbers by value, so ``c2`` precedes ``c10``.

    :param value: any condition value; it is compared as text.
    :returns: the text split into lower-cased words and integers.
    """
    text = str(value)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def condition_order(values: Iterable, preferred: Optional[Sequence] = None) -> List:
    """Return one x-axis order for a set of condition values.

    Values listed in ``preferred`` come first, in the order given (entries not
    present in ``values`` are skipped); the remaining values follow, sorted
    naturally (``r2`` before ``r10``). Missing values are dropped. Using the
    same ``preferred`` list for every panel keeps the conditions aligned.

    :param values: condition values, for example a column; duplicates are kept once.
    :param preferred: values to place first, in this order; matched as given or
        by their text. ``None`` sorts every value naturally.
    :returns: the ordered list of distinct values.
    """
    present = []
    seen = set()
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            continue
        if v not in seen:
            seen.add(v)
            present.append(v)
    out = []
    by_str = {str(v): v for v in present}
    for p in preferred or []:
        match = p if p in seen else by_str.get(str(p))
        if match is not None and match not in out:
            out.append(match)
    rest = sorted((v for v in present if v not in out), key=_natural_key)
    return out + rest


def display_label(value, labels: Optional[Mapping] = None) -> str:
    """Display text of a raw value: ``labels[value]`` or the value with spaces for underscores.

    :param value: the raw condition value.
    :param labels: raw value -> display label; a value is looked up as given and
        then by its text. ``None`` uses no labels.
    :returns: the text to show.
    """
    if labels:
        if value in labels:
            return str(labels[value])
        if str(value) in labels:
            return str(labels[str(value)])
    return str(value).replace("_", " ")


def panel_folder_name(panel_id: str, name: str) -> str:
    """Folder / file stem of a panel, ``<panel_id>_<name>`` with unsafe characters replaced.

    :param panel_id: panel id, for example ``'P01'`` or ``'C03'``.
    :param name: panel name; runs of characters other than letters, digits and
        ``._+-`` become one underscore.
    :returns: the stem, or ``panel_id`` alone when nothing of ``name`` is left.
    """
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")
    return f"{panel_id}_{safe}" if safe else str(panel_id)


MINUS = "−"


def condition_parts(values: Mapping, factors: Optional[Mapping] = None) -> List[Tuple[str, str]]:
    """Split a condition into (variable, factor) parts.

    :param values: {variable (slot or column): value} of the condition.
    :param factors: ``condition_factors`` - {value: [factor, ...]}; a value
        listed there contributes one part per factor (variable ``<var>#<i>``
        for the i-th factor, so backgrounds and genotypes form separate
        variables), any other value is one part of its own variable.
    """
    out = []
    for var, v in values.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        fs = (factors or {}).get(v, (factors or {}).get(str(v)))
        if fs:
            fs = [fs] if isinstance(fs, str) else list(fs)
            out += [(f"{var}#{i}", str(f)) for i, f in enumerate(fs)]
        else:
            out.append((str(var), str(v)))
    return out


def string_parts(condition, sep="_") -> List[Tuple[str, str]]:
    """Fallback decomposition: the tokens of the condition string, by position.

    :param condition: condition name; converted to text and split on ``sep``.
    :returns: ``[('token0', first token), ('token1', second token), ...]``.
    """
    return [(f"token{i}", t) for i, t in enumerate(str(condition).split(sep))]


def axis_layout(order: Sequence, parts: Mapping, style: str = "auto",
                factor_order: Optional[Sequence] = None, labels: Optional[Mapping] = None) -> dict:
    """How to label the x axis of one graph.

    :param order: conditions in x-axis order.
    :param parts: {condition: [(variable, factor), ...]} (see :func:`condition_parts`).
    :param style: ``'auto'`` (labels when at most one variable varies along
        this axis, else the matrix), ``'matrix'`` or ``'labels'``.
    :param factor_order: matrix rows in display order (raw or display names);
        other rows follow by variable, then first appearance.
    :param labels: raw value -> display label, as in :func:`display_label`;
        ``None`` uses no labels.
    :returns: ``{'matrix': bool, 'annotations': {row: {condition: '+'/'-'}},
        'labels': {condition: text}, 'constant': [display texts shared by every
        condition], 'varying': [variables]}``.
    """
    order = list(order)
    show = lambda f: display_label(f, labels)
    variables = list(dict.fromkeys(v for g in order for v, _ in parts.get(g, [])))
    per_var = {v: [dict(parts.get(g, [])).get(v) for g in order] for v in variables}
    varying = [v for v in variables if len(set(per_var[v])) > 1]
    constant = [show(per_var[v][0]) for v in variables if v not in varying and per_var[v][0] is not None]
    use_matrix = style == "matrix" or (style == "auto" and len(varying) > 1)
    if style == "labels":
        use_matrix = False
    out = dict(matrix=use_matrix, annotations={}, labels={}, constant=constant, varying=varying)
    if not use_matrix:
        for g in order:
            d = dict(parts.get(g, []))
            txt = [show(d[v]) for v in (varying or variables) if d.get(v) is not None]
            out["labels"][g] = " ".join(txt) if (varying and txt) else display_label(g, labels)
        if style == "labels":
            out["labels"] = {g: display_label(g, labels) for g in order}
        return out
    rows = []
    for v in varying:
        for f in dict.fromkeys(x for x in per_var[v] if x is not None):
            rows.append((v, f))
    if factor_order:
        rank = {str(x): i for i, x in enumerate(factor_order)}
        key = lambda r: (rank.get(r[1], rank.get(show(r[1]), len(rank) + varying.index(r[0]))),
                         rows.index(r))
        rows = sorted(rows, key=key)
    for v, f in rows:
        out["annotations"][f"{show(f)}:"] = {g: "+" if dict(parts.get(g, [])).get(v) == f else MINUS
                                             for g in order}
    return out


def _empty_test(test, n_groups, reason):
    """One ``test`` row for a comparison that was not run.

    :param test: the value of the ``test`` column, e.g. ``"not tested"``.
    :param n_groups: how many groups had data.
    :param reason: why no test was run, written to ``note``.
    :returns: the row as a dict, every statistic NaN.
    """
    return dict(test=test, statistic=np.nan, df1=np.nan, df2=np.nan, p_value=np.nan,
                effect_size_name="", effect_size=np.nan, all_groups_normal=np.nan,
                equal_variance=np.nan, significance="n/a", n_groups=n_groups, note=reason)


def _eta_squared(groups):
    """Share of the total sum of squares that lies between the groups.

    :param groups: one array of values per group.
    :returns: eta squared, or NaN when every value is the same.
    """
    values = np.concatenate(groups)
    grand = values.mean()
    total = float(((values - grand) ** 2).sum())
    between = float(sum(len(x) * (x.mean() - grand) ** 2 for x in groups))
    return between / total if total > 0 else np.nan


def _studentized_pairs(labels, groups, *, pooled):
    """Every pair's t statistic and studentized-range p value.

    ``pooled`` gives Tukey HSD (the ANOVA's pooled error on ``N - k``
    degrees of freedom); otherwise Games-Howell (each pair's own variances
    and Welch-Satterthwaite degrees of freedom).

    :param labels: group labels, in the order pairs are formed.
    :param groups: one array of values per label.
    :param pooled: True for Tukey HSD, False for Games-Howell.
    :returns: one ``(label_1, label_2, mean_1, mean_2, t, p)`` per pair.
    """
    from scipy.stats import studentized_range

    k = len(groups)
    n = np.array([len(x) for x in groups], dtype=float)
    means = np.array([x.mean() for x in groups])
    if pooled:
        dof = float(n.sum() - k)
        mse = float(sum(((x - x.mean()) ** 2).sum() for x in groups)) / dof
        var_n = mse / n
    else:
        var_n = np.array([x.var(ddof=1) for x in groups]) / n
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        pair_var = var_n[i] + var_n[j]
        t = (means[i] - means[j]) / np.sqrt(pair_var)
        pair_dof = dof if pooled else pair_var ** 2 / (
            var_n[i] ** 2 / (n[i] - 1) + var_n[j] ** 2 / (n[j] - 1))
        p = float(np.clip(studentized_range.sf(np.sqrt(2) * abs(t), k, pair_dof), 0, 1))
        rows.append((labels[i], labels[j], float(means[i]), float(means[j]), float(t), p))
    return rows


def select_and_run_test(df: pd.DataFrame, value: str, group: str, order: Sequence,
                        alpha: float = ALPHA) -> Dict[str, pd.DataFrame]:
    """Choose and run one omnibus test plus its post hoc test.

    :param df: one row per experimental unit (replicate or well).
    :param value: numeric column compared between groups.
    :param group: column naming the group of each row.
    :param order: groups to compare, in plotting order. Groups without data
        are reported with ``n = 0`` and left out of the test.
    :param alpha: significance level of the Shapiro-Wilk and Levene checks
        that choose the test.
    :returns: dict of DataFrames - ``test`` (one row: test, statistic, df1,
        df2, p_value, effect_size_name, effect_size, all_groups_normal,
        equal_variance, significance), ``assumptions`` (Shapiro-Wilk per group
        and Levene), ``posthoc`` (every pair: group_1, group_2, mean_1, mean_2,
        statistic, p_adj, posthoc, significance) and ``descriptive`` (n, mean,
        std, median, sem per group in ``order``).
    """
    from scipy import stats

    order = list(order)
    data = df[df[group].isin(order)][[group, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    desc = (data.groupby(group)[value].agg(["size", "mean", "std", "median"])
            .reindex(order).rename(columns={"size": "n"}))
    desc["n"] = desc["n"].fillna(0).astype(int)
    desc["sem"] = desc["std"] / np.sqrt(desc["n"].where(desc["n"] > 0))
    desc = desc.rename_axis(group).reset_index()

    used = [g for g in order if (data[group] == g).sum() > 0]
    groups = [data.loc[data[group] == g, value].to_numpy(float) for g in used]
    post_cols = ["group_1", "group_2", "mean_1", "mean_2", "statistic", "p_adj",
                 "posthoc", "significance"]
    empty_post = pd.DataFrame(columns=post_cols)

    assum = []
    normal_all = True
    for g, x in zip(used, groups):
        if len(x) >= 3 and np.ptp(x) > 0:
            w, p = stats.shapiro(x)
        else:
            w, p = np.nan, np.nan
        normal = (not np.isfinite(p)) or p >= alpha
        normal_all = normal_all and normal
        assum.append(dict(check="Shapiro-Wilk normality", group=g, n=len(x), statistic=w,
                          p_value=p, passed=normal))
    if len(groups) >= 2 and all(len(x) >= 2 for x in groups):
        with np.errstate(all="ignore"):
            lev_w, lev_p = stats.levene(*groups, center="median")
        equal_var = (not np.isfinite(lev_p)) or lev_p >= alpha
    else:
        lev_w, lev_p, equal_var = np.nan, np.nan, True
    assum.append(dict(check="Levene equal variance (median centred)", group="all",
                      n=len(data), statistic=lev_w, p_value=lev_p, passed=equal_var))
    assum = pd.DataFrame(assum)

    if len(groups) < 2:
        summary = _empty_test("not tested", len(used), "fewer than two groups with data")
        return dict(test=pd.DataFrame([summary]), assumptions=assum, posthoc=empty_post,
                    descriptive=desc)
    if all(np.ptp(x) == 0 for x in groups) and len({float(x[0]) for x in groups}) == 1:
        summary = _empty_test("not tested", len(used), "no variation (every value equal)")
        return dict(test=pd.DataFrame([summary]), assumptions=assum, posthoc=empty_post,
                    descriptive=desc)
    if all(len(x) < 2 for x in groups):
        summary = _empty_test("not tested", len(used), "one value per group")
        return dict(test=pd.DataFrame([summary]), assumptions=assum, posthoc=empty_post,
                    descriptive=desc)

    means = {g: float(np.mean(x)) for g, x in zip(used, groups)}
    post = empty_post
    note = ""
    sub = data[data[group].isin(used)].copy()
    sub[group] = sub[group].astype(str)
    key = {str(g): g for g in used}
    choice = ("anova" if normal_all and equal_var else "welch" if normal_all else "kruskal")
    if choice == "welch" and any(len(x) < 2 or np.ptp(x) == 0 for x in groups):
        choice = "kruskal"
        note = "Welch ANOVA undefined (a group has no variance); Kruskal-Wallis used"
    labelled = sorted(zip((str(g) for g in used), groups))
    try:
        if choice in ("anova", "welch"):
            if choice == "anova":
                test, posthoc_name = "one-way ANOVA", "Tukey HSD"
                stat, p = stats.f_oneway(*groups)
                df1, df2 = float(len(used) - 1), float(len(data) - len(used))
            else:
                from .figures.stats import _welch_anova
                test, posthoc_name = "Welch ANOVA", "Games-Howell"
                stat, p = _welch_anova(groups)
                k = len(groups)
                n_g = np.array([len(x) for x in groups], dtype=float)
                w = n_g / np.array([x.var(ddof=1) for x in groups])
                lam = float((((1 - w / w.sum()) ** 2) / (n_g - 1)).sum())
                df1, df2 = float(k - 1), float((k ** 2 - 1) / (3 * lam))
            stat, p = float(stat), float(p)
            eff_name, eff = "eta_squared", _eta_squared(groups)
            pairs = _studentized_pairs([lab for lab, _ in labelled],
                                       [x for _, x in labelled],
                                       pooled=choice == "anova")
            post = pd.DataFrame(
                [dict(group_1=key[a_], group_2=key[b_], mean_1=m1, mean_2=m2,
                      statistic=t, p_adj=pp, posthoc=posthoc_name)
                 for a_, b_, m1, m2, t, pp in pairs])
        else:
            import scikit_posthocs as sp
            test = "Kruskal-Wallis"
            stat, p = stats.kruskal(*groups)
            stat, p = float(stat), float(p)
            df1, df2 = float(len(used) - 1), np.nan
            n = len(sub)
            eff_name = "epsilon_squared"
            eff = (stat - len(used) + 1) / (n - len(used)) if n > len(used) else np.nan
            dunn = sp.posthoc_dunn(sub, val_col=value, group_col=group, p_adjust="holm")
            rows = []
            for a_, b_ in itertools.combinations(used, 2):
                rows.append(dict(group_1=a_, group_2=b_, mean_1=means[a_], mean_2=means[b_],
                                 statistic=np.nan, p_adj=float(dunn.loc[str(a_), str(b_)]),
                                 posthoc="Dunn test (Holm)"))
            post = pd.DataFrame(rows, columns=post_cols[:-1])
    except Exception as exc:
        summary = _empty_test("not tested", len(used), f"test failed: {exc}")
        summary.update(all_groups_normal=normal_all, equal_variance=equal_var)
        return dict(test=pd.DataFrame([summary]), assumptions=assum, posthoc=empty_post,
                    descriptive=desc)

    post = post.reset_index(drop=True)
    post["significance"] = post["p_adj"].map(stars)
    summary = dict(test=test, statistic=stat, df1=df1, df2=df2, p_value=p,
                   effect_size_name=eff_name, effect_size=eff,
                   all_groups_normal=bool(normal_all), equal_variance=bool(equal_var),
                   significance=stars(p), n_groups=len(used), note=note)
    return dict(test=pd.DataFrame([summary]), assumptions=assum, posthoc=post,
                descriptive=desc)


def _cramers_v(table):
    """Cramér's V of a contingency table, from its uncorrected chi-square.

    :param table: counts, conditions by bins.
    :returns: V between 0 and 1, or NaN for an empty or one-row/one-column table.
    """
    from scipy import stats
    n = table.sum()
    k = min(table.shape) - 1
    if not n or k <= 0:
        return np.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (n * k)))


def _clean_table(table):
    """Drop the all-zero columns and rows of a contingency table.

    :param table: counts, conditions by bins.
    :returns: the remaining counts as a float array.
    """
    table = np.asarray(table, dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    table = table[table.sum(axis=1) > 0, :]
    return table


def _nan_result(test, table):
    """A proportion-test result for a table too small to test.

    :param test: the name of the test that could not run.
    :param table: the cleaned counts, used only for ``n``.
    :returns: the result dict with NaN statistics.
    """
    return dict(test=test, statistic=np.nan, dof=np.nan, p_value=np.nan, cramers_v=np.nan,
                n=int(np.nansum(table)) if np.size(table) else 0, n_resamples=0)


def _chi2_test(table, **_):
    """Chi-square test of independence without continuity correction.

    :param table: counts, conditions by bins; empty rows and columns are dropped.
    :returns: test name, statistic, degrees of freedom, p value, Cramér's V and
        ``n``, or a NaN result when fewer than two rows or columns remain.
    """
    from scipy import stats
    table = _clean_table(table)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _nan_result("chi-square test of independence", table)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return dict(test="chi-square test of independence", statistic=float(chi2), dof=int(dof),
                p_value=float(p), cramers_v=_cramers_v(table), n=int(table.sum()),
                n_resamples=0)


def _fisher_test(table, n_resamples=20000, seed=0):
    """Fisher's exact test, or its Freeman-Halton extension beyond 2 x 2.

    A 2 x 2 table uses :func:`scipy.stats.fisher_exact`. A larger one draws
    ``n_resamples`` tables with the observed margins and reports the share at
    least as improbable as the observed one (Monte Carlo p value).

    :param table: counts, conditions by bins; empty rows and columns are dropped.
    :param n_resamples: Monte Carlo tables drawn for a table larger than 2 x 2.
    :param seed: seed for those draws, so the p value is reproducible.
    :returns: the result dict, as :func:`_chi2_test` returns it.
    """
    from scipy import stats
    table = _clean_table(table).astype(int)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _nan_result("Fisher-Freeman-Halton exact test (Monte Carlo)", table)
    if table.shape == (2, 2):
        res = stats.fisher_exact(table)
        return dict(test="Fisher exact test", statistic=float(res.statistic), dof=np.nan,
                    p_value=float(res.pvalue), cramers_v=_cramers_v(table),
                    n=int(table.sum()), n_resamples=0)
    dist = stats.random_table(table.sum(axis=1), table.sum(axis=0), seed=seed)
    observed = dist.logpmf(table)
    sims = dist.rvs(size=n_resamples)
    logp = np.asarray(dist.logpmf(sims))
    p = (1 + np.sum(logp <= observed + 1e-7)) / (n_resamples + 1)
    return dict(test="Fisher-Freeman-Halton exact test (Monte Carlo)",
                statistic=float(observed), dof=np.nan, p_value=float(p),
                cramers_v=_cramers_v(table), n=int(table.sum()), n_resamples=int(n_resamples))


def proportion_tests(counts: pd.DataFrame, order: Sequence, reference=None,
                     method: str = "chi2", n_resamples: int = 20000,
                     seed: int = 0) -> Tuple[dict, pd.DataFrame]:
    """Compare bin proportions between groups.

    :param counts: contingency table, one row per group and one column per
        bin (object counts).
    :param order: groups to test (rows of ``counts``), in plotting order.
    :param reference: group every other group is compared with; default
        ``order[0]``.
    :param method: ``'chi2'`` (chi-square test of independence) or
        ``'fisher'`` (Fisher-Freeman-Halton exact test by Monte Carlo with
        ``n_resamples`` tables and ``seed``; the Fisher exact test for 2 x 2).
    :param n_resamples: Monte Carlo tables drawn for the Fisher-Freeman-Halton test.
    :param seed: random seed of the Monte Carlo tables.
    :returns: ``(omnibus, pairwise)`` - a dict for the test over every group
        in ``order`` (test, statistic, dof, p_value, cramers_v, n,
        significance) and a DataFrame with one row per group against the
        reference, with Holm-corrected ``p_holm`` and ``significance``.
    """
    from statsmodels.stats.multitest import multipletests

    if method not in ("chi2", "fisher"):
        raise ValueError(f"method must be 'chi2' or 'fisher', got {method!r}")
    order = [g for g in order if g in counts.index]
    if reference is None:
        reference = order[0] if order else None
    fn = _chi2_test if method == "chi2" else _fisher_test
    kw = dict(n_resamples=n_resamples, seed=seed)
    omni = fn(counts.loc[order].to_numpy(), **kw) if order else _nan_result(method, [])
    omni.update(comparison=" vs ".join(map(str, order)), scope="omnibus",
                significance=stars(omni["p_value"]))
    rows = []
    for g in order:
        if g == reference or reference not in counts.index:
            continue
        r = fn(counts.loc[[reference, g]].to_numpy(), **kw)
        r.update(comparison=f"{reference} vs {g}", group_1=reference, group_2=g,
                 scope=f"pairwise vs {reference}")
        rows.append(r)
    pair = pd.DataFrame(rows)
    if len(pair):
        p = pair["p_value"].to_numpy(float)
        ok = np.isfinite(p)
        holm = np.full(len(p), np.nan)
        if ok.any():
            holm[ok] = multipletests(p[ok], method="holm")[1]
        pair["p_holm"] = holm
        pair["significance"] = pair["p_holm"].map(stars)
    else:
        pair = pd.DataFrame(columns=["test", "statistic", "dof", "p_value", "cramers_v", "n",
                                     "comparison", "group_1", "group_2", "scope", "p_holm",
                                     "significance"])
    return omni, pair


def aggregate_to_unit(df: pd.DataFrame, value, keys: Sequence[str],
                      well_keys: Sequence[str] = WELL_KEYS,
                      replicate_column: Optional[str] = None) -> pd.DataFrame:
    """Average object values to the experimental unit.

    Objects -> well mean (per ``keys`` + ``well_keys``) -> replicate mean of the
    well means (per ``keys`` + ``replicate_column``) when ``replicate_column``
    is given; otherwise the unit is the well.

    :param df: object-level table, one row per object.
    :param value: column name or list of column names to average.
    :param keys: grouping columns kept in the output (e.g. the panel group).
    :param well_keys: columns that identify one well.
    :param replicate_column: column naming the biological replicate; ``None``
        makes the well the unit.
    :returns: one row per unit with ``keys``, the unit columns, the averaged
        ``value`` column(s), ``n_objects`` and (replicate unit) ``n_wells``.
    """
    values = [value] if isinstance(value, str) else list(value)
    keys = list(dict.fromkeys(keys))
    for col in list(keys) + list(well_keys) + values + ([replicate_column] if replicate_column else []):
        if col not in df.columns:
            raise KeyError(f"aggregate_to_unit: column '{col}' is not in the table")
    data = df.copy()
    for v in values:
        data[v] = pd.to_numeric(data[v], errors="coerce").replace([np.inf, -np.inf], np.nan)
    well_group = list(dict.fromkeys(keys + ([replicate_column] if replicate_column else [])
                                    + list(well_keys)))
    wells = data.groupby(well_group, observed=True, dropna=False)
    well = wells[values].mean()
    well["n_objects"] = wells.size()
    well = well.reset_index()
    if not replicate_column:
        return well
    rep_group = list(dict.fromkeys(keys + [replicate_column]))
    reps = well.groupby(rep_group, observed=True, dropna=False)
    out = reps[values].mean()
    out["n_wells"] = reps.size()
    out["n_objects"] = reps["n_objects"].sum()
    return out.reset_index()


def fractions_per_unit(df: pd.DataFrame, bin_column: str, bins: Sequence, keys: Sequence[str],
                       well_keys: Sequence[str] = WELL_KEYS,
                       replicate_column: Optional[str] = None) -> pd.DataFrame:
    """Fraction of objects in each bin per unit (replicate if given, else well).

    :param df: object-level table, one row per object.
    :param bin_column: column holding the bin of each object.
    :param bins: bins to report, in legend order; bins without objects get 0.
    :param keys: grouping columns kept in the output (e.g. the panel group).
    :param well_keys: columns that identify one well; the unit when
        ``replicate_column`` is ``None``.
    :param replicate_column: column naming the biological replicate, or ``None``.
    :returns: one row per unit with ``keys``, unit columns, ``n_objects`` and one
        column per bin (fractions summing to 1).
    """
    unit = list(dict.fromkeys(list(keys) + ([replicate_column] if replicate_column
                                            else list(well_keys))))
    counts = pd.crosstab([df[c] for c in unit], df[bin_column])
    counts = counts.reindex(columns=list(bins), fill_value=0)
    n = counts.sum(axis=1)
    frac = counts.div(n.replace(0, np.nan), axis=0)
    frac.columns = list(bins)
    frac.insert(0, "n_objects", n)
    return frac.reset_index()


def bar_spec(data: pd.DataFrame, value: str, group: str, order: Sequence, title: str = "",
             ylabel: Optional[str] = None, labels: Optional[Mapping] = None,
             annotations: Optional[Mapping] = None, ymin=None, refline=None,
             palette=None, extra_tables: Optional[Mapping[str, pd.DataFrame]] = None,
             outliers="policy") -> dict:
    """Run the bar-panel test on unit data and return a panel spec for drawing/saving.

    :param data: one row per experimental unit, from :func:`aggregate_to_unit`.
    :param value: numeric column drawn as bars.
    :param group: column naming the group of each row.
    :param order: groups on the x axis, in order.
    :param title: panel title; the test result is added below it.
    :param ylabel: y-axis label; ``None`` uses ``value`` with spaces for underscores.
    :param labels: raw group value -> x tick label.
    :param annotations: row name -> {group: text}, aligned text rows under the x axis.
    :param ymin: lower y limit; ``None`` chooses it from the data.
    :param refline: y value of a dashed reference line, or ``None``.
    :param palette: group -> colour, or a list of colours in ``order``; ``None``
        uses the default colours.
    :param extra_tables: file stem -> table, written as more CSVs by :func:`save_panel`.
    :param outliers: ``'mad'``, ``'iqr'`` (see :func:`mark_outliers`), None,
        or ``'policy'`` (default) to use the active :class:`outlier_policy`.
        Outliers stay in ``data`` with ``outlier = True`` and are left out of
        the drawing and the test; ``stats['outliers']`` counts them per group.
    :returns: a dict spec with ``kind`` ``'bar'`` and the :func:`select_and_run_test`
        result under ``stats``.
    """
    method = _OUTLIER_POLICY["method"] if outliers == "policy" else outliers
    used = data
    if parse_outlier_method(method) is not None:
        data = data.copy()
        data["outlier"] = mark_outliers(data, value, group, method).to_numpy()
        used = data[~data["outlier"]]
    stats = select_and_run_test(used, value, group, order)
    if "outlier" in data.columns:
        stats["outliers"] = outlier_table(data, group, order, method)
    return dict(kind="bar", data=data, value=value, group=group, order=list(order),
                title=title, ylabel=ylabel or value.replace("_", " "), labels=labels,
                annotations=annotations, ymin=ymin, refline=refline, palette=palette,
                stats=stats, extra_tables=dict(extra_tables or {}),
                outlier_method=method if "outlier" in data.columns else None)


def stacked_spec(fractions: pd.DataFrame, counts: pd.DataFrame, group: str, order: Sequence,
                 bins: Sequence, bin_labels: Optional[Sequence[str]] = None,
                 method: str = "chi2", reference=None, title: str = "",
                 labels: Optional[Mapping] = None, annotations: Optional[Mapping] = None,
                 legend_title: str = "", ylabel: str = "fraction of objects (mean ± SD)",
                 n_resamples: int = 20000, seed: int = 0, colors=None,
                 extra_tables: Optional[Mapping[str, pd.DataFrame]] = None) -> dict:
    """Run the proportion test and return a stacked-panel spec for drawing/saving.

    :param fractions: one row per unit with one column per bin, from
        :func:`fractions_per_unit`.
    :param counts: object counts, one row per group and one column per bin.
    :param group: column naming the group of each row of ``fractions``.
    :param order: groups on the x axis, in order.
    :param bins: bins stacked from the bottom, in order.
    :param bin_labels: legend text per bin, in ``bins`` order; ``None`` uses the bins.
    :param method: ``'chi2'`` or ``'fisher'``, as in :func:`proportion_tests`.
    :param reference: group every other group is compared with; default ``order[0]``.
    :param title: panel title; the omnibus result is added below it.
    :param labels: raw group value -> x tick label.
    :param annotations: row name -> {group: text}, aligned text rows under the x axis.
    :param legend_title: title of the bin legend.
    :param ylabel: y-axis label.
    :param n_resamples: Monte Carlo tables for the Fisher-Freeman-Halton test.
    :param seed: random seed of the Monte Carlo tables.
    :param colors: one colour per bin; ``None`` uses viridis.
    :param extra_tables: file stem -> table, written as more CSVs by :func:`save_panel`.
    :returns: a dict spec with ``kind`` ``'stacked'``, the ``omnibus`` and
        ``pairwise`` results and the reindexed ``counts``.
    """
    order = [g for g in order]
    counts = counts.reindex(index=order, columns=list(bins), fill_value=0)
    omni, pair = proportion_tests(counts, order, reference=reference, method=method,
                                  n_resamples=n_resamples, seed=seed)
    method_label = "chi-square" if method == "chi2" else (
        "Fisher exact" if omni.get("test") == "Fisher exact test" else "Fisher-Freeman-Halton")
    return dict(kind="stacked", data=fractions, counts=counts, group=group, order=order,
                bins=list(bins), bin_labels=list(bin_labels) if bin_labels else [str(b) for b in bins],
                method=method, method_label=method_label, omnibus=omni, pairwise=pair,
                title=title, labels=labels, annotations=annotations, legend_title=legend_title,
                ylabel=ylabel, colors=colors, extra_tables=dict(extra_tables or {}))


def _despine(ax):
    """Hide the top and right spines of ``ax``."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw_annotations(ax, order, annotations, fontsize=None, line_pt=None, row_names=True):
    """Draw aligned text rows under the x axis; return the number of rows drawn."""
    if not annotations:
        return 0
    fontsize = STYLE["annot_pt"] if fontsize is None else fontsize
    line_pt = STYLE["annot_line_pt"] if line_pt is None else line_pt
    rows = list(annotations.items())
    for k, (row_name, mapping) in enumerate(rows):
        dy = -(k + 1) * line_pt
        if row_names:
            ax.annotate(str(row_name), xy=(0, 0), xycoords="axes fraction", xytext=(-6, dy),
                        textcoords="offset points", ha="right", va="top", fontsize=fontsize,
                        annotation_clip=False)
        mapping = mapping or {}
        for i, g in enumerate(order):
            text = mapping.get(g, mapping.get(str(g), ""))
            ax.annotate(str(text), xy=(i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, dy), textcoords="offset points", ha="center", va="top",
                        fontsize=fontsize, annotation_clip=False)
    return len(rows)


def _set_xticks(ax, order, labels, annotations, rotation=None, matrix=False):
    """Label the categories along the x axis, with annotation rows beneath.

    :param ax: the panel's axes.
    :param order: the categories, in plotting order.
    :param labels: display names for the categories, or None.
    :param annotations: rows of per-category text drawn under the axis, or None.
    :param rotation: tick label rotation in degrees; when None the labels are
        rotated by 35 degrees only when they would otherwise touch.
    :param matrix: draw only the annotation rows (the +/- matrix) and no
        tick labels.
    """
    n_rows = _draw_annotations(ax, order, annotations)
    text = ["" for g in order] if matrix else [display_label(g, labels) for g in order]
    if rotation is None:
        longest = max((len(t) for t in text), default=0)
        rotation = 0 if longest * len(order) <= 24 and longest <= 10 else 35
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(text, rotation=rotation, ha="right" if rotation else "center",
                       fontsize=8)
    if n_rows and not matrix:
        ax.tick_params(axis="x", pad=n_rows * STYLE["annot_line_pt"] + 4)
    ax.set_xlim(-0.6, len(order) - 0.4)


def _differing(parts, a, b):
    da, db = dict(parts.get(a, [])), dict(parts.get(b, []))
    return sum(da.get(v) != db.get(v) for v in set(da) | set(db))


def select_bracket_pairs(posthoc: pd.DataFrame, order: Sequence, parts: Optional[Mapping] = None,
                         mode="one_factor", max_n: int = 12):
    """Post hoc pairs that get a bracket, and how many qualifying pairs were left out.

    ``'one_factor'``: significant pairs that differ in exactly one factor of
    ``parts`` (same host with another strain, same strain in another host);
    every significant pair when the conditions have no factor structure (at
    most one variable). ``'all'``: every significant pair. A list of pairs:
    those pairs, significant or not. Beyond ``max_n`` the pairs with the
    smallest p values are kept.

    :param posthoc: post hoc table with ``group_1``, ``group_2``, ``p_adj``
        and ``significance`` columns; ``None`` or empty gives no pairs.
    :param order: conditions shown on the axis; pairs with a group outside it
        are dropped.
    :returns: (DataFrame of chosen pairs, number of qualifying pairs not drawn).
    """
    order = list(order)
    if posthoc is None or not len(posthoc):
        return pd.DataFrame(columns=["group_1", "group_2", "p_adj", "significance"]), 0
    post = posthoc[posthoc["group_1"].isin(order) & posthoc["group_2"].isin(order)]
    if isinstance(mode, (list, tuple)):
        wanted = {frozenset(map(str, p)) for p in mode}
        chosen = post[np.array([frozenset((str(a), str(b))) in wanted
                                for a, b in zip(post["group_1"], post["group_2"])], dtype=bool)]
    else:
        chosen = post[post["p_adj"] < ALPHA]
        n_vars = len({v for g in order for v, _ in (parts or {}).get(g, [])})
        if mode == "one_factor" and parts and n_vars > 1:
            chosen = chosen[np.array([_differing(parts, a, b) == 1
                                      for a, b in zip(chosen["group_1"], chosen["group_2"])], dtype=bool)]
    hidden = max(len(chosen) - int(max_n), 0)
    if hidden:
        chosen = chosen.nsmallest(int(max_n), "p_adj")
    return chosen, hidden


def pack_brackets(pairs: pd.DataFrame, order: Sequence) -> List[int]:
    """Stack level of every bracket: shortest spans lowest, brackets on one level never overlap.

    :param pairs: chosen pairs with ``group_1`` and ``group_2`` columns, as
        returned by :func:`select_bracket_pairs`.
    :param order: conditions in x-axis order; every pair's groups must be in it.
    :returns: stack level (0 = lowest) of each row of ``pairs``, in row order.
    """
    order = list(order)
    spans = [(min(order.index(a), order.index(b)), max(order.index(a), order.index(b)))
             for a, b in zip(pairs["group_1"], pairs["group_2"])]
    levels = [0] * len(spans)
    taken: List[List[Tuple[int, int]]] = []
    for k in sorted(range(len(spans)), key=lambda q: (spans[q][1] - spans[q][0], spans[q][0])):
        lo, hi = spans[k]
        lv = 0
        while lv < len(taken) and any(not (hi < a or lo > b) for a, b in taken[lv]):
            lv += 1
        if lv == len(taken):
            taken.append([])
        taken[lv].append((lo, hi))
        levels[k] = lv
    return levels


def draw_bar_panel(ax, data: pd.DataFrame, value: str, group: str, order: Sequence,
                   stats: Optional[Dict[str, pd.DataFrame]] = None, title: str = "",
                   ylabel: Optional[str] = None, labels: Optional[Mapping] = None,
                   annotations: Optional[Mapping] = None, ymin=None, refline=None,
                   palette=None, show_pairs: str = "significant", max_brackets: int = 12,
                   seed: int = 0, show_test=None, matrix=False, parts: Optional[Mapping] = None,
                   bracket_pairs="one_factor"):
    """Bar of the mean of unit values, SD error bar, one dot per unit, post hoc brackets.

    :param ax: matplotlib axes to draw on.
    :param data: one row per experimental unit.
    :param value: numeric column drawn as bars.
    :param group: column naming the group of each row.
    :param order: groups on the x axis, in order.
    :param stats: output of :func:`select_and_run_test`; the omnibus test goes
        in the title and post hoc results become brackets.
    :param show_pairs: ``'significant'`` (default), ``'all'`` or ``'none'``.
    :param bracket_pairs: which pairs get a bracket (:func:`select_bracket_pairs`):
        ``'one_factor'`` (significant pairs that differ in one factor of
        ``parts``; all significant pairs without factor structure), ``'all'``
        or an explicit list of pairs. At most ``max_brackets`` are drawn (the
        smallest p values); every pair stays in ``stats_posthoc.csv``.
    :param ymin: lower y limit (``None``: 0, or the data minimum if negative).
    :param title: panel title; the test result is added below it.
    :param ylabel: y-axis label; ``None`` uses ``value`` with spaces for underscores.
    :param labels: raw group value -> x tick label.
    :param annotations: row name -> {group: text}, aligned text rows under the x axis.
    :param refline: y value of a dashed reference line, or ``None``.
    :param palette: group -> colour, or a list of colours in ``order``.
    :param max_brackets: most post hoc brackets drawn.
    :param seed: random seed of the horizontal jitter of the dots.
    :returns: ``ax``.
    """
    order = list(order)
    rng = np.random.default_rng(seed)
    if "outlier" in data.columns:
        data = data[~data["outlier"].astype(bool)]
    vals = pd.to_numeric(data[value], errors="coerce") if len(data) else pd.Series(dtype=float)
    d = data.assign(_v=vals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_v"])
    means = d.groupby(group)["_v"].mean().reindex(order)
    sds = d.groupby(group)["_v"].std().reindex(order).fillna(0)
    if palette is None:
        palette = {g: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, g in enumerate(order)}
    elif not isinstance(palette, Mapping):
        palette = {g: palette[i % len(palette)] for i, g in enumerate(order)}
    x = np.arange(len(order))
    if ymin is None:
        lo = float(np.nanmin(d["_v"])) if len(d) else 0.0
        hi = float(np.nanmax(d["_v"])) if len(d) else 1.0
        ymin = 0.0 if lo >= 0 else lo - 0.05 * max(hi - lo, abs(lo), 1e-9)
    base = ymin if ymin > 0 else 0.0
    heights = means.fillna(base).to_numpy(float) - base
    from matplotlib.colors import to_rgba
    fills = [palette.get(g, "#888888") for g in order]
    ax.bar(x, heights, bottom=base, width=STYLE["bar_width"],
           color=[to_rgba(c, STYLE["bar_alpha"]) for c in fills],
           edgecolor=[shade(c, STYLE["edge_shade"]) for c in fills], linewidth=STYLE["edge_lw"])
    ax.errorbar(x, means.to_numpy(float), yerr=sds.to_numpy(float), fmt="none",
                ecolor=STYLE["err_color"], elinewidth=STYLE["err_lw"], capsize=STYLE["capsize"],
                zorder=4)
    for i, g in enumerate(order):
        y = d.loc[d[group] == g, "_v"].to_numpy(float)
        if len(y):
            ax.scatter(i + rng.uniform(-STYLE["jitter"], STYLE["jitter"], len(y)), y,
                       s=STYLE["dot_size"], color=shade(fills[i], STYLE["dot_shade"]),
                       alpha=STYLE["dot_alpha"], edgecolors=STYLE["dot_edge"],
                       linewidths=STYLE["dot_edge_lw"], zorder=3)
    if refline is not None:
        ax.axhline(refline, ls="--", lw=0.8, color="grey")
    top = np.nanmax(np.r_[(means + sds).to_numpy(float), d["_v"].to_numpy(float), ymin + 1e-9])
    span = top - ymin if top > ymin else 1.0
    step = span * 0.08
    level = top + step * 0.6
    summary = None
    if stats is not None:
        summary = stats["test"].iloc[0].to_dict()
        post = stats["posthoc"]
        if show_pairs != "none" and len(post):
            mode = "all" if show_pairs == "all" and bracket_pairs == "one_factor" else bracket_pairs
            chosen, _ = select_bracket_pairs(post, order, parts, mode, max_brackets)
            levels = pack_brackets(chosen, order)
            for (_, r), k in zip(chosen.iterrows(), levels):
                i, j = order.index(r["group_1"]), order.index(r["group_2"])
                y = level + k * step * 1.1
                ax.plot([i, i, j, j], [y, y + step * 0.25, y + step * 0.25, y],
                        lw=STYLE["bracket_lw"], color="black")
                ax.text((i + j) / 2, y + step * 0.3, r["significance"], ha="center",
                        va="bottom", fontsize=8)
            if levels:
                level += (max(levels) + 1) * step * 1.1
    ax.set_ylim(ymin, level + step * 0.6)
    show_test = STYLE["show_test"] if show_test is None else show_test
    head = title if STYLE["show_title"] else ""
    if summary is not None and show_test:
        if summary['test'] == "not tested":
            line = f"not tested ({summary.get('note') or 'too few units'})"
        else:
            line = f"{summary['test']}: {fmt_p(summary['p_value'])} ({summary['significance']})"
        head = f"{head}\n{line}" if head else line
    if head:
        ax.set_title(head, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or value.replace("_", " "), fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    _set_xticks(ax, order, labels, annotations, matrix=matrix)
    style_axes(ax)
    return ax


def _bin_colors(n, colors=None):
    """Colours for ``n`` stacked bins.

    :param n: number of bins.
    :param colors: colours given by the caller, returned as a list when set.
    :returns: the given colours, or ``n`` evenly spaced viridis colours.
    """
    if colors is not None:
        return list(colors)
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(STYLE["stacked_cmap"])
    return [cmap(v) for v in np.linspace(0.05, 0.95, max(n, 1))]


def draw_stacked_panel(ax, fractions: pd.DataFrame, group: str, order: Sequence,
                       bins: Sequence, bin_labels: Optional[Sequence[str]] = None,
                       omnibus: Optional[dict] = None, pairwise: Optional[pd.DataFrame] = None,
                       method_label: str = "", title: str = "",
                       labels: Optional[Mapping] = None, annotations: Optional[Mapping] = None,
                       colors=None, ylabel: str = "fraction of objects (mean ± SD)",
                       legend: bool = True, legend_title: str = "", show_test=None,
                       matrix=False):
    """Stacked mean fractions over units, SD bar at the top of each segment, stars vs reference.

    :param ax: matplotlib axes to draw on.
    :param fractions: output of :func:`fractions_per_unit` (one row per unit).
    :param group: column naming the group of each row.
    :param order: groups on the x axis, in order.
    :param bins: bins stacked from the bottom, in order.
    :param bin_labels: legend text per bin; ``None`` uses the bins.
    :param omnibus: omnibus result of :func:`proportion_tests`, shown in the title.
    :param pairwise: pairwise results of :func:`proportion_tests`, drawn as stars.
    :param method_label: test name shown in the title.
    :param title: panel title.
    :param labels: raw group value -> x tick label.
    :param annotations: row name -> {group: text}, aligned text rows under the x axis.
    :param colors: one colour per bin; ``None`` uses viridis.
    :param ylabel: y-axis label.
    :param legend: draw the bin legend beside the axes.
    :param legend_title: title of the bin legend.
    :returns: ``ax``.
    """
    order = list(order)
    bins = list(bins)
    bin_labels = list(bin_labels) if bin_labels else [str(b) for b in bins]
    colors = _bin_colors(len(bins), colors)
    means = fractions.groupby(group)[bins].mean().reindex(order).fillna(0)
    sds = fractions.groupby(group)[bins].std().reindex(order).fillna(0)
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for i, b in enumerate(bins):
        ax.bar(x, means[b].to_numpy(float), bottom=bottom, color=colors[i % len(colors)],
               width=STYLE["bar_width"], label=bin_labels[i], edgecolor=STYLE["stacked_edge"],
               linewidth=STYLE["stacked_lw"])
        ax.errorbar(x, bottom + means[b].to_numpy(float), yerr=sds[b].to_numpy(float),
                    fmt="none", ecolor=STYLE["err_color"], elinewidth=STYLE["err_lw"],
                    capsize=STYLE["capsize"])
        bottom += means[b].to_numpy(float)
    if pairwise is not None and len(pairwise):
        for _, r in pairwise.iterrows():
            if r["group_2"] in order:
                ax.text(order.index(r["group_2"]), 1.075, r["significance"], ha="center",
                        va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    show_test = STYLE["show_test"] if show_test is None else show_test
    head = title if STYLE["show_title"] else ""
    if omnibus is not None and show_test:
        cv = omnibus.get("cramers_v", np.nan)
        cv_txt = f", Cramér V = {cv:.2f}" if np.isfinite(cv) else ""
        line = f"{method_label} omnibus: {fmt_p(omnibus.get('p_value'))}{cv_txt}"
        head = f"{head}\n{line}" if head else line
    if head:
        ax.set_title(head, fontsize=9)
    _set_xticks(ax, order, labels, annotations, matrix=matrix)
    if legend:
        ax.legend(title=legend_title or None, fontsize=7, title_fontsize=7, frameon=False,
                  bbox_to_anchor=(1.0, 1.0), loc="upper left")
    style_axes(ax)
    return ax


def draw_panel(ax, spec: dict, legend: bool = True):
    """Draw a spec made by :func:`bar_spec` or :func:`stacked_spec` on ``ax``.

    :param ax: matplotlib axes to draw on.
    :param spec: the panel spec.
    :param legend: draw the bin legend of a stacked panel.
    :returns: ``ax``.
    """
    if spec["kind"] == "bar":
        return draw_bar_panel(ax, spec["data"], spec["value"], spec["group"], spec["order"],
                              stats=spec["stats"], title=spec.get("title", ""),
                              ylabel=spec.get("ylabel"), labels=spec.get("labels"),
                              annotations=spec.get("annotations"), ymin=spec.get("ymin"),
                              refline=spec.get("refline"), palette=spec.get("palette"),
                              matrix=bool(spec.get("matrix")), parts=spec.get("parts"),
                              bracket_pairs=spec.get("bracket_pairs", "one_factor"))
    return draw_stacked_panel(ax, spec["data"], spec["group"], spec["order"], spec["bins"],
                              bin_labels=spec.get("bin_labels"), omnibus=spec["omnibus"],
                              pairwise=spec["pairwise"], method_label=spec.get("method_label", ""),
                              title=spec.get("title", ""), labels=spec.get("labels"),
                              annotations=spec.get("annotations"), colors=spec.get("colors"),
                              ylabel=spec.get("ylabel", "fraction of objects (mean ± SD)"),
                              legend=legend, legend_title=spec.get("legend_title", ""),
                              matrix=bool(spec.get("matrix")))


def _panel_size(spec):
    """Figure size of one panel, in inches.

    :param spec: the panel spec; its ``order``, ``kind`` and ``annotations`` set
        the size.
    :returns: ``(width, height)``, wider for more categories and for a stacked
        panel's legend, taller for each annotation row.
    """
    n = max(len(spec["order"]), 1)
    width = 1.6 + 0.55 * n + (1.8 if spec["kind"] == "stacked" else 0)
    n_rows = len(spec.get("annotations") or {})
    return max(width, 3.0), 4.0 + 0.16 * n_rows


def save_panel(out_dir: str, panel_id: str, name: str, spec: dict, dpi: int = 300) -> str:
    """Write one panel folder: graph (PDF + PNG), plotted data and statistics CSVs.

    :param out_dir: folder that receives the panel folder.
    :param panel_id: panel id, for example ``'P01'``.
    :param name: panel name, made safe by :func:`panel_folder_name`.
    :param spec: the panel spec from :func:`bar_spec` or :func:`stacked_spec`;
        its ``folder`` key is set to the folder written.
    :param dpi: resolution of the PNG.
    :returns: the panel folder path ``<out_dir>/<panel_id>_<name>``.
    """
    import matplotlib.pyplot as plt

    stem = panel_folder_name(panel_id, name)
    folder = os.path.join(out_dir, stem)
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=_panel_size(spec))
    try:
        draw_panel(ax, spec)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(folder, f"{stem}.{ext}"), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    write_table(spec["data"], os.path.join(folder, "data_replicates.csv"))
    if spec["kind"] == "bar":
        st = spec["stats"]
        write_table(st["test"], os.path.join(folder, "stats_test.csv"))
        write_table(st["assumptions"], os.path.join(folder, "stats_assumptions.csv"))
        write_table(st["posthoc"], os.path.join(folder, "stats_posthoc.csv"))
        write_table(st["descriptive"], os.path.join(folder, "stats_descriptive.csv"))
        if "outliers" in st:
            write_table(st["outliers"], os.path.join(folder, "stats_outliers.csv"))
    else:
        bins, bin_labels = spec["bins"], spec["bin_labels"]
        rename = dict(zip(bins, bin_labels))
        write_table(spec["counts"].rename(columns=rename).rename_axis(spec["group"]),
                    os.path.join(folder, "contingency_table_counts.csv"), index=True)
        write_table(pd.concat([pd.DataFrame([spec["omnibus"]]), spec["pairwise"]], ignore_index=True),
                    os.path.join(folder, "stats_test.csv"))
        frac = spec["data"]
        agg = frac.groupby(spec["group"])[bins].agg(["mean", "std"]).reindex(spec["order"])
        agg.columns = [f"{rename[b]} {s}" for b, s in agg.columns]
        agg.insert(0, "n_units", frac.groupby(spec["group"]).size().reindex(spec["order"]))
        write_table(agg.rename_axis(spec["group"]), os.path.join(folder, "stats_descriptive.csv"),
                    index=True)
    for tname, table in (spec.get("extra_tables") or {}).items():
        write_table(table, os.path.join(folder, f"{tname}.csv"))
    spec["folder"] = folder
    return folder


def assemble_figure(specs: Mapping[str, dict], panel_ids: Sequence[str], ncol: int, path: str,
                    title: Optional[str] = None, dpi: int = 300) -> List[str]:
    """Assemble panels into one figure with bold panel IDs; write PDF and PNG.

    :param specs: panel id -> spec (from :func:`bar_spec` / :func:`stacked_spec`).
    :param panel_ids: ids to place, row by row.
    :param ncol: panels per row.
    :param path: output path without extension (an extension is stripped).
    :param title: figure title, or ``None``.
    :param dpi: resolution of the PNG.
    :returns: list of written file paths. Unknown ids raise ``KeyError``.
    """
    import matplotlib.pyplot as plt

    missing = [p for p in panel_ids if p not in specs]
    if missing:
        raise KeyError(f"assemble_figure: unknown panel id(s) {missing}; known: {sorted(specs)}")
    ids = list(panel_ids)
    ncol = max(1, min(int(ncol), len(ids))) if ids else 1
    nrow = int(np.ceil(len(ids) / ncol)) if ids else 1
    sizes = [_panel_size(specs[i]) for i in ids]
    col_w = [max((sizes[k][0] for k in range(c, len(ids), ncol)), default=3.0) for c in range(ncol)]
    row_h = max((s[1] for s in sizes), default=4.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(sum(col_w), nrow * (row_h + 0.6)),
                             gridspec_kw={"width_ratios": col_w}, squeeze=False)
    try:
        stacked_ax = None
        for ax, pid in zip(axes.flat, ids):
            spec = specs[pid]
            draw_panel(ax, spec, legend=False)
            if spec["kind"] == "stacked" and stacked_ax is None:
                stacked_ax = ax
            ax.text(-0.18, 1.12, pid, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom")
        for ax in list(axes.flat)[len(ids):]:
            ax.axis("off")
        if stacked_ax is not None:
            h, l = stacked_ax.get_legend_handles_labels()
            spec = next(specs[i] for i in ids if specs[i]["kind"] == "stacked")
        if title:
            fig.suptitle(title, fontsize=12, y=1.01)
        fig.tight_layout()
        if stacked_ax is not None:
            fig.legend(h, l, title=spec.get("legend_title") or None, loc="upper center",
                       ncol=min(len(l), 8), fontsize=8, title_fontsize=8, frameon=False,
                       bbox_to_anchor=(0.5, 0.0))
        base = os.path.splitext(path)[0] if path.lower().endswith((".pdf", ".png")) else path
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        written = []
        for ext in ("pdf", "png"):
            fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight")
            written.append(f"{base}.{ext}")
    finally:
        plt.close(fig)
    return written
