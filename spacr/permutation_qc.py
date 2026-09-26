"""Evaluate residual exchangeability for blocked permutation tests.

Permutation inference assumes that residuals can be exchanged within each
block. This module reports serial autocorrelation, positional gradients, and
block-level diagnostics that may invalidate that assumption. These checks
complement residual-versus-fitted and Q-Q plots: normality and constant
variance do not establish exchangeability across plate positions.
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "block_residual_report",
    "autocorrelation",
    "position_effect",
    "exchangeability_verdict",
    "plot_residual_by_position",
    "write_permutation_qc",
]


def autocorrelation(residuals: Sequence[float]) -> float:
    """Durbin-Watson on ``residuals`` in the order given.

    :param residuals: ordered residual values to test for serial dependence.
    :returns: the Durbin-Watson statistic, or ``nan`` with fewer than two
        finite residuals or a nonpositive sum of squared residuals.

    2 is no autocorrelation, 0 is perfect positive, 4 perfect negative.
    Written out rather than imported so this module does not pull
    statsmodels for one line -- and so the ORDER is explicit: it is the
    order the rows arrive in, which for a well table is plate reading order.
    """
    values = np.asarray(list(residuals), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    diff = np.diff(values)
    denominator = float(np.sum(values ** 2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(diff ** 2) / denominator)


def position_effect(residuals: Sequence[float],
                    positions: Sequence[Any]) -> Dict[str, float]:
    """How much of the residual one position column explains.

    :param residuals: residual values whose positional structure is measured.
    :param positions: position level aligned to each residual.

    :returns: a mapping containing ``eta_squared`` and ``levels``. With at
        least three finite observations it also contains ``p_value``; when
        two or more levels vary, it includes ``omega_squared``,
        ``worst_level``, and ``worst_departure``, the signed departure of
        that level's mean from the grand residual mean.
    :raises ValueError: if ``residuals`` and ``positions`` do not contain
        exactly one value for each other.

    THE NUMBER THAT MATTERS FOR EXCHANGEABILITY. If rows differ from each
    other more than cells within a row do, then shuffling across rows inside
    a plate is shuffling things that are not alike, and the permutation null
    is wider than the truth.
    """
    residual_values = list(residuals)
    position_values = list(positions)
    if len(residual_values) != len(position_values):
        raise ValueError(
            "residuals and positions must have the same length; got "
            f"{len(residual_values)} and {len(position_values)}")
    values = np.asarray(residual_values, dtype=float)
    labels = np.asarray([str(p) for p in position_values])
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]
    if values.size < 3:
        return {"eta_squared": float("nan"), "levels": 0.0}
    grand = float(values.mean())
    total = float(np.sum((values - grand) ** 2))
    levels = sorted(set(labels.tolist()))
    if total <= 0 or len(levels) < 2:
        return {"eta_squared": 0.0, "p_value": 1.0,
                "levels": float(len(levels))}
    between, worst, worst_level = 0.0, 0.0, ""
    for level in levels:
        here = values[labels == level]
        between += here.size * (float(here.mean()) - grand) ** 2
        if abs(float(here.mean()) - grand) > abs(worst):
            worst, worst_level = float(here.mean()) - grand, level

    within = total - between
    k, n = len(levels), int(values.size)
    p_value = 1.0
    if within > 0 and n > k:
        from scipy.stats import f as _f

        statistic = (between / (k - 1)) / (within / (n - k))
        p_value = float(_f.sf(statistic, k - 1, n - k))
    return {
        "eta_squared": float(between / total),
        "omega_squared": float(max(
            0.0, (between - (k - 1) * (within / (n - k))) /
            (total + (within / (n - k))))) if within > 0 and n > k else 0.0,
        "p_value": p_value,
        "levels": float(k),
        "worst_level": worst_level,
        "worst_departure": float(worst),
    }


def block_residual_report(residuals: Sequence[float],
                          blocks: Sequence[Any],
                          positions: Optional[Mapping[str, Sequence[Any]]]
                          = None) -> Dict[str, Any]:
    """Everything the permutation's assumption needs, per block and overall.

    :param residuals: the phenotype residuals that WILL BE PERMUTED -- not a
        model's residuals, which is the distinction this whole module exists
        for.
    :param blocks: the block each residual belongs to; the shuffle happens
        inside these.
    :param positions: ``{'rowID': [...], 'columnID': [...]}`` or similar.
    :returns: pooled sample and block counts, pooled and per-block
        Durbin-Watson diagnostics, per-block means and standard deviations,
        and one :func:`position_effect` result per named position column.
    :raises ValueError: if ``blocks`` or any named position column does not
        contain exactly one value for each residual.

    PER BLOCK AND NOT ONLY POOLED. The shuffle is within-block, so a pooled
    statistic can look healthy while one plate is badly structured -- and
    that one plate is where the false positives come from.
    """
    residual_values = list(residuals)
    block_values = list(blocks)
    if len(residual_values) != len(block_values):
        raise ValueError(
            "residuals and blocks must have the same length; got "
            f"{len(residual_values)} and {len(block_values)}")
    position_values = {
        str(name): list(column) for name, column in (positions or {}).items()
    }
    for name, column in position_values.items():
        if len(column) != len(residual_values):
            raise ValueError(
                f"position column {name!r} must have the same length as "
                f"residuals; got {len(column)} and {len(residual_values)}")

    values = np.asarray(residual_values, dtype=float)
    labels = np.asarray([str(b) for b in block_values])
    out: Dict[str, Any] = {
        "n": int(values.size),
        "blocks": int(len(set(labels.tolist()))),
        "durbin_watson": autocorrelation(values),
        "per_block": {},
        "position": {},
    }
    for block in sorted(set(labels.tolist())):
        here = values[labels == block]
        out["per_block"][block] = {
            "n": int(here.size),
            "durbin_watson": autocorrelation(here),
            "mean": float(here.mean()) if here.size else float("nan"),
            "sd": float(here.std(ddof=1)) if here.size > 1 else float("nan"),
        }
    for name, column in position_values.items():
        out["position"][name] = position_effect(values, column)
    return out


#: How far from 2 the Durbin-Watson may sit before it is worth acting on.
#: 0.4 puts the boundary at 1.6/2.4, which is roughly where the standard
#: tables reject at n in the hundreds.
DW_TOLERANCE = 0.4

#: How unlikely a position column's structure must be, under the null of no
#: position effect, before it is worth acting on. A p-value rather than an
#: effect size, because eta-squared's null value depends on the number of
#: levels and a fixed cut flags noise wherever there are many.
POSITION_ALPHA = 0.01


def exchangeability_verdict(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Is the within-block shuffle defensible, and if not, what to change?

    :param report: diagnostics returned by :func:`block_residual_report`.

    :returns: ``{'ok': bool, 'findings': [...], 'remedy': str}``.

    THE REMEDY IS NAMED, NOT IMPLIED. "Durbin-Watson 1.22" is a number;
    "add rowID to guide_nuisance_columns" is something the reader can do,
    and it is the same setting the run already has.
    """
    findings: List[str] = []
    dw = float(report.get("durbin_watson", float("nan")))
    if np.isfinite(dw) and abs(dw - 2.0) > DW_TOLERANCE:
        direction = "positive" if dw < 2.0 else "negative"
        findings.append(
            f"Durbin-Watson {dw:.2f} against 2 for none -- {direction} "
            f"autocorrelation in row order. Neighbouring wells are not "
            f"independent, so shuffling them treats structure as noise.")
    for block, stats in (report.get("per_block") or {}).items():
        block_dw = float(stats.get("durbin_watson", float("nan")))
        if np.isfinite(block_dw) and abs(block_dw - 2.0) > DW_TOLERANCE:
            findings.append(
                f"block {block!r}: Durbin-Watson {block_dw:.2f} over "
                f"{stats['n']} well(s). The shuffle is WITHIN blocks, so one "
                f"structured block is where false positives come from even "
                f"when the pooled figure looks healthy.")
    culprits: List[str] = []
    for name, stats in (report.get("position") or {}).items():
        p_value = float(stats.get("p_value", 1.0))
        omega = float(stats.get("omega_squared", 0.0))
        if np.isfinite(p_value) and p_value < POSITION_ALPHA:
            culprits.append(name)
            findings.append(
                f"{name} explains {omega:.1%} of the residual variance "
                f"across {int(stats.get('levels', 0))} level(s) "
                f"(p = {p_value:.2g}). Wells at different {name} values are "
                f"not alike, so they are not swappable.")
    remedy = ""
    if culprits:
        remedy = (f"Add {', '.join(sorted(culprits))} to "
                  f"guide_nuisance_columns. They are then removed before the "
                  f"residualisation and the same statistic is evaluated on "
                  f"residuals that have a better claim to exchangeability.")
    elif findings:
        remedy = ("The structure is not explained by the position columns "
                  "measured here. Check whether the block column names the "
                  "right grouping -- a batch that spans plates is not "
                  "removed by blocking on plate.")
    return {"ok": not findings, "findings": findings, "remedy": remedy}


#: The most blocks drawn as rows of the residual-by-position figure. Every
#: block is still in the JSON report; past this many the figure keeps the
#: blocks whose Durbin-Watson sits furthest from 2, because those are the
#: ones a reader has to look at.
MAX_PANEL_BLOCKS = 12


def _safe_name(text: Any) -> str:
    """A file-name-safe rendering of an outcome column name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                      for ch in str(text))
    return cleaned.strip("._") or "outcome"


def _plain(value: Any) -> Any:
    """``value`` with numpy scalars and non-finite floats made JSON-safe."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def plot_residual_by_position(residuals: Sequence[float],
                              blocks: Sequence[Any],
                              positions: Mapping[str, Sequence[Any]],
                              report: Optional[Mapping[str, Any]] = None,
                              verdict: Optional[Mapping[str, Any]] = None,
                              removed: Sequence[str] = (),
                              title: str = ""):
    """Draw the permuted residual against plate position, one row per block.

    :param residuals: the residuals that WERE PERMUTED.
    :param blocks: the block of each residual; the shuffle happens inside it.
    :param positions: ``{'rowID': [...], 'columnID': [...]}``, aligned to
        ``residuals``.
    :param report: :func:`block_residual_report` for the same residuals; when
        given, each block's Durbin-Watson and each column's pooled p-value
        are drawn on the panel they describe.
    :param verdict: :func:`exchangeability_verdict` of ``report``; its remedy
        is written under the panels, so the setting that removes a gradient
        is named where the gradient is seen.
    :param removed: nuisance columns already removed before residualisation,
        named on the figure so a run with position removed can be told from
        one without.
    :param title: the figure's title, normally the outcome column.
    :returns: a :class:`matplotlib.figure.Figure`.
    :raises ValueError: if ``positions`` is empty or a column is misaligned.

    A GRADIENT ACROSS THE PLATE IS WHAT THIS SHOWS AND NO OTHER PANEL DOES.
    Residuals-vs-fitted shows spread and Q-Q shows shape; neither shows that
    row A sits above row P inside one plate, which is exactly what makes the
    wells in that plate not swappable.
    """
    from matplotlib.figure import Figure

    from .regression_qc import _natural_key

    values = np.asarray(list(residuals), dtype=float)
    labels = np.asarray([str(b) for b in blocks])
    if labels.size != values.size:
        raise ValueError("residuals and blocks must have the same length")
    columns = {str(k): np.asarray([str(v) for v in col])
               for k, col in (positions or {}).items()}
    if not columns:
        raise ValueError("no position column to draw residuals against")
    for name, col in columns.items():
        if col.size != values.size:
            raise ValueError(
                f"position column {name!r} must have one value per residual")

    per_block = dict((report or {}).get("per_block") or {})
    position_stats = dict((report or {}).get("position") or {})
    order = sorted(set(labels.tolist()), key=_natural_key)
    if len(order) > MAX_PANEL_BLOCKS:
        def _distance(block):
            """How far a block's Durbin-Watson sits from 2, or -1 without one.

            :param block: a block label from ``labels``.
            :returns: ``|DW - 2|``, so the most autocorrelated blocks sort
                first; -1.0 when the report has no finite statistic.
            """
            dw = float(per_block.get(block, {}).get("durbin_watson",
                                                    float("nan")))
            return abs(dw - 2.0) if np.isfinite(dw) else -1.0
        kept = set(sorted(order, key=_distance,
                          reverse=True)[:MAX_PANEL_BLOCKS])
        order = [block for block in order if block in kept]

    names = list(columns)
    finite = values[np.isfinite(values)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = limit * 1.08 if limit > 0 else 1.0

    fig = Figure(figsize=(4.2 * len(names) + 0.6, 2.3 * len(order) + 1.9))
    axes = fig.subplots(len(order), len(names), squeeze=False)
    for i, block in enumerate(order):
        here = labels == block
        dw = float(per_block.get(block, {}).get("durbin_watson",
                                                float("nan")))
        y = values[here]
        for j, name in enumerate(names):
            ax = axes[i][j]
            levels = sorted(set(columns[name][here].tolist()),
                            key=_natural_key)
            index = {level: k for k, level in enumerate(levels)}
            x = np.asarray([index[v] for v in columns[name][here]],
                           dtype=float)
            means = [float(np.nanmean(y[x == k])) if np.any(x == k)
                     and np.isfinite(y[x == k]).any() else float("nan")
                     for k in range(len(levels))]
            ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
            ax.scatter(x, y, s=9, alpha=0.55, color="#4c72b0",
                       linewidths=0, zorder=2)
            ax.plot(range(len(levels)), means, color="#c44e52",
                    linewidth=1.4, marker="o", markersize=3, zorder=3)
            ax.set_ylim(-limit, limit)
            step = max(1, int(np.ceil(len(levels) / 12)))
            ax.set_xticks(range(0, len(levels), step))
            ax.set_xticklabels(levels[::step], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            head = f"{block} -- by {name}"
            if np.isfinite(dw):
                head += f"   DW {dw:.2f}"
            ax.set_title(head, fontsize=8)
            if j == 0:
                ax.set_ylabel("residual", fontsize=8)
            if i == len(order) - 1:
                label = name
                p_value = position_stats.get(name, {}).get("p_value")
                if p_value is not None and np.isfinite(float(p_value)):
                    label += f"  (pooled p = {float(p_value):.2g})"
                ax.set_xlabel(label, fontsize=8)

    heading = "Residual by position within block"
    if title:
        heading += f" -- {title}"
    fig.suptitle(heading, fontsize=10)
    lines = ["Removed before residualisation: "
             + (", ".join(removed) if removed
                else "block only (guide_nuisance_columns is empty)")]
    if verdict is not None:
        if verdict.get("ok"):
            lines.append("Nothing found: the within-block shuffle is "
                         "defensible for these residuals.")
        elif verdict.get("remedy"):
            lines.append(str(verdict["remedy"]))
    footer = "\n".join(textwrap.fill(line, 140) for line in lines)
    fig.text(0.01, 0.005, footer, fontsize=7.5, ha="left", va="bottom")
    bottom = min(0.3, (0.3 + 0.16 * (footer.count("\n") + 1))
                 / fig.get_figheight())
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    return fig


def write_permutation_qc(destination: str,
                         outcome: str,
                         residuals: Sequence[float],
                         blocks: Sequence[Any],
                         positions: Mapping[str, Sequence[Any]],
                         report: Mapping[str, Any],
                         verdict: Mapping[str, Any],
                         removed: Sequence[str] = ()) -> Dict[str, Any]:
    """Write the permutation run's QC into ``<destination>/regression_qc/``.

    The same folder a parametric run writes, so a permutation run's
    diagnostics are found where every other run's are.

    :param destination: the run's results folder.
    :param outcome: the phenotype column the residuals belong to; it names
        the files, so a multi-outcome run keeps one set per outcome.
    :param residuals: the residuals that WERE PERMUTED.
    :param blocks: the block of each residual.
    :param positions: the position columns, aligned to ``residuals``.
    :param report: :func:`block_residual_report` of the same residuals.
    :param verdict: :func:`exchangeability_verdict` of ``report``.
    :param removed: nuisance columns removed before residualisation.
    :returns: ``{'dir', 'report', 'figure', 'figure_error'}``. The JSON
        report is always written; ``figure`` is ``None`` and
        ``figure_error`` says why when there is nothing to draw against.
    """
    from .regression_qc import QC_DIRNAME

    out_dir = os.path.join(os.fspath(destination), QC_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    stem = _safe_name(outcome)
    report_path = os.path.join(out_dir, f"exchangeability_{stem}.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"outcome": str(outcome),
                   "removed_before_residualisation": list(removed),
                   "report": _plain(report),
                   "verdict": _plain(verdict)}, handle, indent=2)
    manifest: Dict[str, Any] = {"dir": out_dir, "report": report_path,
                                "figure": None, "figure_error": None}
    try:
        fig = plot_residual_by_position(
            residuals, blocks, positions, report=report, verdict=verdict,
            removed=removed, title=str(outcome))
    except ValueError as error:
        manifest["figure_error"] = str(error)
        return manifest
    figure_path = os.path.join(out_dir, f"residual_by_position_{stem}.png")
    fig.savefig(figure_path, dpi=110)
    manifest["figure"] = figure_path
    return manifest
