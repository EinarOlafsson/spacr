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

__all__ = [
    "ALPHA", "stars", "fmt_p", "condition_order", "display_label",
    "select_and_run_test", "proportion_tests", "aggregate_to_unit",
    "fractions_per_unit", "bar_spec", "stacked_spec", "draw_bar_panel",
    "draw_stacked_panel", "draw_panel", "save_panel", "assemble_figure",
    "panel_folder_name",
]

ALPHA = 0.05
WELL_KEYS: Tuple[str, ...] = ("plateID", "rowID", "columnID")
_DEFAULT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                   "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
_ANNOT_LINE_PT = 11.0


# --------------------------------------------------------------------------- helpers
def stars(p) -> str:
    """Significance stars for a p value ('ns' at or above 0.05, 'n/a' if missing)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    return ("****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2
            else "*" if p < ALPHA else "ns")


def fmt_p(p) -> str:
    """Format a p value for a panel title."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "p = n/a"
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 1e-4" if p < 1e-4 else f"p = {p:.3g}"


def _natural_key(value):
    text = str(value)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def condition_order(values: Iterable, preferred: Optional[Sequence] = None) -> List:
    """Return one x-axis order for a set of condition values.

    Values listed in ``preferred`` come first, in the order given (entries not
    present in ``values`` are skipped); the remaining values follow, sorted
    naturally (``r2`` before ``r10``). Missing values are dropped. Using the
    same ``preferred`` list for every panel keeps the conditions aligned.
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
    """Display text of a raw value: ``labels[value]`` or the value with spaces for underscores."""
    if labels:
        if value in labels:
            return str(labels[value])
        if str(value) in labels:
            return str(labels[str(value)])
    return str(value).replace("_", " ")


def panel_folder_name(panel_id: str, name: str) -> str:
    """Folder / file stem of a panel, ``<panel_id>_<name>`` with unsafe characters replaced."""
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")
    return f"{panel_id}_{safe}" if safe else str(panel_id)


# --------------------------------------------------------------------------- bar-panel tests
def _empty_test(test, n_groups, reason):
    return dict(test=test, statistic=np.nan, df1=np.nan, df2=np.nan, p_value=np.nan,
                effect_size_name="", effect_size=np.nan, all_groups_normal=np.nan,
                equal_variance=np.nan, significance="n/a", n_groups=n_groups, note=reason)


def _col(frame, *names):
    for n in names:
        if n in frame.columns:
            return frame[n]
    return pd.Series(np.nan, index=frame.index)


def select_and_run_test(df: pd.DataFrame, value: str, group: str, order: Sequence,
                        alpha: float = ALPHA) -> Dict[str, pd.DataFrame]:
    """Choose and run one omnibus test plus its post hoc test.

    :param df: one row per experimental unit (replicate or well).
    :param value: numeric column compared between groups.
    :param group: column naming the group of each row.
    :param order: groups to compare, in plotting order. Groups without data
        are reported with ``n = 0`` and left out of the test.
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
        # Welch weights are 1 / variance: undefined for a constant group.
        choice = "kruskal"
        note = "Welch ANOVA undefined (a group has no variance); Kruskal-Wallis used"
    try:
        if choice == "anova":
            import pingouin as pg
            test = "one-way ANOVA"
            a = pg.anova(data=sub, dv=value, between=group, detailed=False)
            stat, p = float(a["F"].iloc[0]), float(_col(a, "p-unc", "p_unc").iloc[0])
            df1, df2 = float(a["ddof1"].iloc[0]), float(a["ddof2"].iloc[0])
            eff_name, eff = "eta_squared", float(_col(a, "np2").iloc[0])
            ph = pg.pairwise_tukey(data=sub, dv=value, between=group)
            post = pd.DataFrame(dict(group_1=ph["A"].map(key), group_2=ph["B"].map(key),
                                     mean_1=ph["mean(A)"], mean_2=ph["mean(B)"],
                                     statistic=ph["T"], p_adj=_col(ph, "p-tukey", "p_tukey"),
                                     posthoc="Tukey HSD"))
        elif choice == "welch":
            import pingouin as pg
            test = "Welch ANOVA"
            a = pg.welch_anova(data=sub, dv=value, between=group)
            stat, p = float(a["F"].iloc[0]), float(_col(a, "p-unc", "p_unc").iloc[0])
            df1, df2 = float(a["ddof1"].iloc[0]), float(a["ddof2"].iloc[0])
            eff_name, eff = "eta_squared", float(_col(a, "np2").iloc[0])
            ph = pg.pairwise_gameshowell(data=sub, dv=value, between=group)
            post = pd.DataFrame(dict(group_1=ph["A"].map(key), group_2=ph["B"].map(key),
                                     mean_1=ph["mean(A)"], mean_2=ph["mean(B)"],
                                     statistic=ph["T"], p_adj=_col(ph, "pval", "p-val"),
                                     posthoc="Games-Howell"))
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
    except Exception as exc:  # degenerate data (identical values, n = 1 groups ...)
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


# --------------------------------------------------------------------------- proportion tests
def _cramers_v(table):
    from scipy import stats
    n = table.sum()
    k = min(table.shape) - 1
    if not n or k <= 0:
        return np.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (n * k)))


def _clean_table(table):
    table = np.asarray(table, dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    table = table[table.sum(axis=1) > 0, :]
    return table


def _nan_result(test, table):
    return dict(test=test, statistic=np.nan, dof=np.nan, p_value=np.nan, cramers_v=np.nan,
                n=int(np.nansum(table)) if np.size(table) else 0, n_resamples=0)


def _chi2_test(table, **_):
    from scipy import stats
    table = _clean_table(table)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _nan_result("chi-square test of independence", table)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return dict(test="chi-square test of independence", statistic=float(chi2), dof=int(dof),
                p_value=float(p), cramers_v=_cramers_v(table), n=int(table.sum()),
                n_resamples=0)


def _fisher_test(table, n_resamples=20000, seed=0):
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


# --------------------------------------------------------------------------- aggregation
def aggregate_to_unit(df: pd.DataFrame, value, keys: Sequence[str],
                      well_keys: Sequence[str] = WELL_KEYS,
                      replicate_column: Optional[str] = None) -> pd.DataFrame:
    """Average object values to the experimental unit.

    Objects -> well mean (per ``keys`` + ``well_keys``) -> replicate mean of the
    well means (per ``keys`` + ``replicate_column``) when ``replicate_column``
    is given; otherwise the unit is the well.

    :param value: column name or list of column names to average.
    :param keys: grouping columns kept in the output (e.g. the panel group).
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


# --------------------------------------------------------------------------- specs
def bar_spec(data: pd.DataFrame, value: str, group: str, order: Sequence, title: str = "",
             ylabel: Optional[str] = None, labels: Optional[Mapping] = None,
             annotations: Optional[Mapping] = None, ymin=None, refline=None,
             palette=None, extra_tables: Optional[Mapping[str, pd.DataFrame]] = None) -> dict:
    """Run the bar-panel test on unit data and return a panel spec for drawing/saving."""
    stats = select_and_run_test(data, value, group, order)
    return dict(kind="bar", data=data, value=value, group=group, order=list(order),
                title=title, ylabel=ylabel or value.replace("_", " "), labels=labels,
                annotations=annotations, ymin=ymin, refline=refline, palette=palette,
                stats=stats, extra_tables=dict(extra_tables or {}))


def stacked_spec(fractions: pd.DataFrame, counts: pd.DataFrame, group: str, order: Sequence,
                 bins: Sequence, bin_labels: Optional[Sequence[str]] = None,
                 method: str = "chi2", reference=None, title: str = "",
                 labels: Optional[Mapping] = None, annotations: Optional[Mapping] = None,
                 legend_title: str = "", ylabel: str = "fraction of objects (mean ± SD)",
                 n_resamples: int = 20000, seed: int = 0, colors=None,
                 extra_tables: Optional[Mapping[str, pd.DataFrame]] = None) -> dict:
    """Run the proportion test and return a stacked-panel spec for drawing/saving."""
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


# --------------------------------------------------------------------------- drawing
def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw_annotations(ax, order, annotations, fontsize=7):
    """Draw aligned text rows under the x axis; return the number of rows drawn."""
    if not annotations:
        return 0
    rows = list(annotations.items())
    for k, (row_name, mapping) in enumerate(rows):
        dy = -(k + 1) * _ANNOT_LINE_PT
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


def _set_xticks(ax, order, labels, annotations, rotation=None):
    n_rows = _draw_annotations(ax, order, annotations)
    text = [display_label(g, labels) for g in order]
    if rotation is None:
        rotation = 0 if max((len(t) for t in text), default=0) * len(order) < 40 else 35
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(text, rotation=rotation, ha="right" if rotation else "center",
                       fontsize=8)
    if n_rows:
        ax.tick_params(axis="x", pad=n_rows * _ANNOT_LINE_PT + 4)
    ax.set_xlim(-0.6, len(order) - 0.4)


def draw_bar_panel(ax, data: pd.DataFrame, value: str, group: str, order: Sequence,
                   stats: Optional[Dict[str, pd.DataFrame]] = None, title: str = "",
                   ylabel: Optional[str] = None, labels: Optional[Mapping] = None,
                   annotations: Optional[Mapping] = None, ymin=None, refline=None,
                   palette=None, show_pairs: str = "significant", max_brackets: int = 12,
                   seed: int = 0):
    """Bar of the mean of unit values, SD error bar, one dot per unit, post hoc brackets.

    :param stats: output of :func:`select_and_run_test`; the omnibus test goes
        in the title and post hoc results become brackets.
    :param show_pairs: ``'significant'`` (default), ``'all'`` or ``'none'``.
        Brackets are left out when more than ``max_brackets`` pairs qualify
        (every pair stays in ``stats_posthoc.csv``).
    :param ymin: lower y limit (``None``: 0, or the data minimum if negative).
    """
    order = list(order)
    rng = np.random.default_rng(seed)
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
    ax.bar(x, heights, bottom=base, width=0.7, alpha=0.8,
           color=[palette.get(g, "#888888") for g in order], edgecolor="black", linewidth=0.6)
    ax.errorbar(x, means.to_numpy(float), yerr=sds.to_numpy(float), fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3)
    for i, g in enumerate(order):
        y = d.loc[d[group] == g, "_v"].to_numpy(float)
        if len(y):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=12, color="black", zorder=3)
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
            pairs = post if show_pairs == "all" else post[post["p_adj"] < ALPHA]
            pairs = pairs[pairs["group_1"].isin(order) & pairs["group_2"].isin(order)]
            if 0 < len(pairs) <= max_brackets:
                for _, r in pairs.iterrows():
                    i, j = order.index(r["group_1"]), order.index(r["group_2"])
                    ax.plot([i, i, j, j], [level, level + step * 0.25, level + step * 0.25, level],
                            lw=0.9, color="black")
                    ax.text((i + j) / 2, level + step * 0.3, r["significance"], ha="center",
                            va="bottom", fontsize=8)
                    level += step * 1.05
    ax.set_ylim(ymin, level + step * 0.6)
    head = title
    if summary is not None:
        if summary['test'] == "not tested":
            line = f"not tested ({summary.get('note') or 'too few units'})"
        else:
            line = f"{summary['test']}: {fmt_p(summary['p_value'])} ({summary['significance']})"
        head = f"{title}\n{line}" if title else line
    ax.set_title(head, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or value.replace("_", " "), fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    _set_xticks(ax, order, labels, annotations)
    _despine(ax)
    return ax


def _bin_colors(n, colors=None):
    if colors is not None:
        return list(colors)
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("viridis")
    return [cmap(v) for v in np.linspace(0.05, 0.95, max(n, 1))]


def draw_stacked_panel(ax, fractions: pd.DataFrame, group: str, order: Sequence,
                       bins: Sequence, bin_labels: Optional[Sequence[str]] = None,
                       omnibus: Optional[dict] = None, pairwise: Optional[pd.DataFrame] = None,
                       method_label: str = "", title: str = "",
                       labels: Optional[Mapping] = None, annotations: Optional[Mapping] = None,
                       colors=None, ylabel: str = "fraction of objects (mean ± SD)",
                       legend: bool = True, legend_title: str = ""):
    """Stacked mean fractions over units, SD bar at the top of each segment, stars vs reference.

    :param fractions: output of :func:`fractions_per_unit` (one row per unit).
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
               width=0.7, label=bin_labels[i], edgecolor="white", linewidth=0.4)
        ax.errorbar(x, bottom + means[b].to_numpy(float), yerr=sds[b].to_numpy(float),
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
        bottom += means[b].to_numpy(float)
    if pairwise is not None and len(pairwise):
        for _, r in pairwise.iterrows():
            if r["group_2"] in order:
                ax.text(order.index(r["group_2"]), 1.075, r["significance"], ha="center",
                        va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    head = title
    if omnibus is not None:
        cv = omnibus.get("cramers_v", np.nan)
        cv_txt = f", Cramér V = {cv:.2f}" if np.isfinite(cv) else ""
        line = f"{method_label} omnibus: {fmt_p(omnibus.get('p_value'))}{cv_txt}"
        head = f"{title}\n{line}" if title else line
    ax.set_title(head, fontsize=9)
    _set_xticks(ax, order, labels, annotations)
    if legend:
        ax.legend(title=legend_title or None, fontsize=7, title_fontsize=7, frameon=False,
                  bbox_to_anchor=(1.0, 1.0), loc="upper left")
    _despine(ax)
    return ax


def draw_panel(ax, spec: dict, legend: bool = True):
    """Draw a spec made by :func:`bar_spec` or :func:`stacked_spec` on ``ax``."""
    if spec["kind"] == "bar":
        return draw_bar_panel(ax, spec["data"], spec["value"], spec["group"], spec["order"],
                              stats=spec["stats"], title=spec.get("title", ""),
                              ylabel=spec.get("ylabel"), labels=spec.get("labels"),
                              annotations=spec.get("annotations"), ymin=spec.get("ymin"),
                              refline=spec.get("refline"), palette=spec.get("palette"))
    return draw_stacked_panel(ax, spec["data"], spec["group"], spec["order"], spec["bins"],
                              bin_labels=spec.get("bin_labels"), omnibus=spec["omnibus"],
                              pairwise=spec["pairwise"], method_label=spec.get("method_label", ""),
                              title=spec.get("title", ""), labels=spec.get("labels"),
                              annotations=spec.get("annotations"), colors=spec.get("colors"),
                              ylabel=spec.get("ylabel", "fraction of objects (mean ± SD)"),
                              legend=legend, legend_title=spec.get("legend_title", ""))


def _panel_size(spec):
    n = max(len(spec["order"]), 1)
    width = 1.6 + 0.55 * n + (1.8 if spec["kind"] == "stacked" else 0)
    n_rows = len(spec.get("annotations") or {})
    return max(width, 3.0), 4.0 + 0.16 * n_rows


# --------------------------------------------------------------------------- saving
def save_panel(out_dir: str, panel_id: str, name: str, spec: dict, dpi: int = 300) -> str:
    """Write one panel folder: graph (PDF + PNG), plotted data and statistics CSVs.

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
    spec["data"].to_csv(os.path.join(folder, "data_replicates.csv"), index=False)
    if spec["kind"] == "bar":
        st = spec["stats"]
        st["test"].to_csv(os.path.join(folder, "stats_test.csv"), index=False)
        st["assumptions"].to_csv(os.path.join(folder, "stats_assumptions.csv"), index=False)
        st["posthoc"].to_csv(os.path.join(folder, "stats_posthoc.csv"), index=False)
        st["descriptive"].to_csv(os.path.join(folder, "stats_descriptive.csv"), index=False)
    else:
        bins, bin_labels = spec["bins"], spec["bin_labels"]
        rename = dict(zip(bins, bin_labels))
        spec["counts"].rename(columns=rename).rename_axis(spec["group"]).to_csv(
            os.path.join(folder, "contingency_table_counts.csv"))
        pd.concat([pd.DataFrame([spec["omnibus"]]), spec["pairwise"]], ignore_index=True).to_csv(
            os.path.join(folder, "stats_test.csv"), index=False)
        frac = spec["data"]
        agg = frac.groupby(spec["group"])[bins].agg(["mean", "std"]).reindex(spec["order"])
        agg.columns = [f"{rename[b]} {s}" for b, s in agg.columns]
        agg.insert(0, "n_units", frac.groupby(spec["group"]).size().reindex(spec["order"]))
        agg.rename_axis(spec["group"]).to_csv(os.path.join(folder, "stats_descriptive.csv"))
    for tname, table in (spec.get("extra_tables") or {}).items():
        table.to_csv(os.path.join(folder, f"{tname}.csv"), index=False)
    spec["folder"] = folder
    return folder


def assemble_figure(specs: Mapping[str, dict], panel_ids: Sequence[str], ncol: int, path: str,
                    title: Optional[str] = None, dpi: int = 300) -> List[str]:
    """Assemble panels into one figure with bold panel IDs; write PDF and PNG.

    :param specs: panel id -> spec (from :func:`bar_spec` / :func:`stacked_spec`).
    :param panel_ids: ids to place, row by row.
    :param ncol: panels per row.
    :param path: output path without extension (an extension is stripped).
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
