"""Matplotlib implementation of the apicomplexan-figures skill.

Palette and proportions sampled from Waldman et al. Cell 2020 Fig 1/3 and Giuliano et al.
Nature Microbiology 2024 Fig 1 (Lourido lab). Every helper encodes a rule from SKILL.md; the
docstrings name the rule so a reviewer can check the claim.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class C:
    """Fixed palette. Do not invent hues; do not re-map strain colours between figures."""
    GREY = "#B4B4B4"
    GREY_DARK = "#7F7F7F"
    INK = "#231F20"
    BLUE = "#2E77BC"
    BLUE_LIGHT = "#7FB3E0"
    GREEN = "#2E7D4F"
    RUST = "#C4441C"
    CORAL = "#E8A88C"
    GOLD = "#E8C33A"
    OCHRE = "#C87A28"
    PURPLE = "#8B4A82"
    NAVY = "#1F3F6E"
    SEQ = "Blues"          # single-hue ramp for p-values / scores


def use(frame="box"):
    """Apply the house rcParams. frame='box' (Nature Micro) or 'L' (Cell)."""
    box = frame == "box"
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": None,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6.2,                 # tick label size == the 0.85x tier
        "axes.labelsize": 7.0,            # the 1.0x reference tier
        "axes.titlesize": 7.0,            # descriptors only, never sentences
        "axes.titleweight": "regular", "axes.titlelocation": "center",
        "axes.edgecolor": C.INK, "axes.linewidth": 0.65, "axes.labelcolor": C.INK,
        "axes.grid": False,               # rule: no gridlines, ever
        "axes.spines.top": box, "axes.spines.right": box,
        "xtick.color": C.INK, "ytick.color": C.INK, "text.color": C.INK,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
        "legend.frameon": False, "legend.fontsize": 5.6,
        "legend.handlelength": 0.9, "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "lines.linewidth": 1.2,
    })


def panel_letter(ax, letter, dx=-0.16, dy=1.06):
    """Bold upper-case letter, top-left, outside the axes, ~2x axis-label size, no period."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13.5, fontweight="bold",
            va="bottom", ha="left", color=C.INK)


def descriptor(ax, text):
    """Optional 2-4 word descriptor above the axes. Not a sentence, not a claim."""
    ax.set_title(text, fontsize=6.6, color=C.INK)


def rotate_ticks(ax, deg=45):
    """Long categorical labels rotate 45 deg, right-aligned."""
    for t in ax.get_xticklabels():
        t.set_rotation(deg); t.set_ha("right")


def annotate_in_panel(ax, text, color=None, x=0.03, y=0.95, size=6.0):
    """In-panel annotation instead of a legend box (Giuliano Fig 1E/G idiom)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, va="top", ha="left",
            color=color or C.INK)


def text_legend(ax, labels_colors, x=0.03, y=0.95, dy=0.085, size=6.0):
    """Legend as coloured text with no markers (Waldman Fig 3B/C idiom)."""
    for i, (lab, col) in enumerate(labels_colors):
        ax.text(x, y - i * dy, lab, transform=ax.transAxes, fontsize=size, color=col,
                va="top", ha="left")


# ----------------------------------------------------------------- small-n data
def dots_with_mean(ax, groups, labels, colors=None, jitter=0.07, mean_bar=True,
                   err=None, ms=14):
    """Individual replicates as points with a mean line.

    Rule: for n <= 8 these papers never draw a bar. err in {None,'sd','sem'} adds an error bar
    only if the legend will state which it is.
    """
    colors = colors or [C.GREY_DARK] * len(groups)
    rng = np.random.default_rng(0)
    for i, (g, col) in enumerate(zip(groups, colors)):
        g = np.asarray(g, float); g = g[np.isfinite(g)]
        x = i + rng.uniform(-jitter, jitter, len(g))
        ax.scatter(x, g, s=ms, color=col, zorder=3, edgecolor="none")
        if mean_bar:
            ax.plot([i - 0.22, i + 0.22], [g.mean()] * 2, color=C.INK, lw=1.0, zorder=4)
        if err:
            e = g.std(ddof=1) if err == "sd" else g.std(ddof=1) / max(1, np.sqrt(len(g)))
            ax.errorbar(i, g.mean(), yerr=e, fmt="none", ecolor=C.INK, elinewidth=0.7,
                        capsize=2.2, zorder=4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, len(groups) - 0.45)


def superplot(ax, unit_values, unit_labels=None, palette=None, ms_small=5, ms_big=34):
    """Nested data: small points per observation coloured by unit, large points per unit mean,
    black grand mean +/- SEM over the top (Giuliano Fig 1H idiom).

    unit_values: list over categories; each element is a list over units of arrays of observations.
    """
    palette = palette or [C.BLUE, C.GREEN, C.RUST, C.GOLD, C.PURPLE, C.NAVY, C.OCHRE]
    rng = np.random.default_rng(1)
    for ci, units in enumerate(unit_values):
        means = []
        for ui, obs in enumerate(units):
            obs = np.asarray(obs, float); obs = obs[np.isfinite(obs)]
            if not len(obs):
                continue
            col = palette[ui % len(palette)]
            x = ci + rng.uniform(-0.16, 0.16, len(obs))
            ax.scatter(x, obs, s=ms_small, color=col, alpha=0.5, edgecolor="none", zorder=2)
            m = obs.mean(); means.append(m)
            ax.scatter(ci + rng.uniform(-0.10, 0.10), m, s=ms_big, color=col, alpha=0.95,
                       edgecolor="none", zorder=3)
        if means:
            means = np.array(means)
            ax.errorbar(ci, means.mean(),
                        yerr=means.std(ddof=1) / max(1, np.sqrt(len(means))),
                        fmt="o", ms=3.2, color="black", ecolor="black", elinewidth=0.9,
                        capsize=2.4, zorder=5)
    if unit_labels:
        ax.set_xticks(range(len(unit_values))); ax.set_xticklabels(unit_labels)
    ax.set_xlim(-0.55, len(unit_values) - 0.45)


# ----------------------------------------------------------------- grey + highlight
def highlight_scatter(ax, x, y, masks=None, colors=None, labels=None, s_bg=3.0, s_hi=9.0,
                      diagonal=False):
    """Everything grey except the claim. masks is a list of boolean arrays over (x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.scatter(x, y, s=s_bg, color=C.GREY, edgecolor="none", zorder=1, rasterized=True)
    if masks:
        colors = colors or [C.BLUE, C.RUST, C.GREEN, C.GOLD]
        for i, m in enumerate(masks):
            ax.scatter(x[m], y[m], s=s_hi, color=colors[i % len(colors)], edgecolor="none",
                       zorder=3)
        if labels:
            text_legend(ax, list(zip(labels, colors[:len(labels)])))
    if diagonal:
        lo = np.nanmin([np.nanmin(x), np.nanmin(y)]); hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
        ax.plot([lo, hi], [lo, hi], color=C.INK, lw=0.6, ls=":", zorder=2)


def volcano(ax, log2fc, neglog10p, fc_thr=1.0, p_thr=None, label_points=None, s=4.0):
    """Grey / GREEN up / RUST down with dotted thresholds; label only a handful of genes."""
    log2fc, neglog10p = np.asarray(log2fc, float), np.asarray(neglog10p, float)
    up = (log2fc > fc_thr) & ((neglog10p > p_thr) if p_thr else True)
    dn = (log2fc < -fc_thr) & ((neglog10p > p_thr) if p_thr else True)
    ax.scatter(log2fc, neglog10p, s=s, color=C.GREY, edgecolor="none", rasterized=True)
    ax.scatter(log2fc[up], neglog10p[up], s=s, color=C.GREEN, edgecolor="none")
    ax.scatter(log2fc[dn], neglog10p[dn], s=s, color=C.RUST, edgecolor="none")
    for v in (-fc_thr, fc_thr):
        ax.axvline(v, color=C.INK, lw=0.6, ls=":")
    if p_thr:
        ax.axhline(p_thr, color=C.INK, lw=0.6, ls=":")
    if label_points:
        for xx, yy, nm in label_points:
            ax.annotate(nm, (xx, yy), fontsize=5.4, style="italic", color=C.INK,
                        xytext=(3, 1), textcoords="offset points")
    ax.set_xlabel("log$_2$(fold change)"); ax.set_ylabel("-log$_{10}$(p-value)")


def bubble_enrichment(ax, categories, fold, count, neglog10p, sort=True, smax=60):
    """Size = count, fill = -log10 p on a blue ramp, categories ordered by effect."""
    d = list(zip(categories, fold, count, neglog10p))
    if sort:
        d.sort(key=lambda r: r[1])
    cats, f, n, p = zip(*d)
    x = np.arange(len(cats))
    sz = smax * (np.asarray(n, float) / max(np.max(n), 1)) + 4
    sc = ax.scatter(x, f, s=sz, c=p, cmap=C.SEQ, edgecolor=C.INK, linewidth=0.3, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(cats); rotate_ticks(ax)
    ax.set_ylabel("fold enrichment")
    return sc


# ----------------------------------------------------------------- time / survival
def line_with_band(ax, x, mean, err, color, label=None, alpha=0.25):
    """Line plus same-hue band at 0.25 (Waldman Fig 3B). Legend goes in as coloured text."""
    ax.plot(x, mean, color=color, lw=1.25, label=label, zorder=3)
    ax.fill_between(x, np.asarray(mean) - np.asarray(err), np.asarray(mean) + np.asarray(err),
                    color=color, alpha=alpha, lw=0, zorder=2)


def km_survival(ax, times, events, color, label=None, tmax=None):
    """Kaplan-Meier step function, no band. times: event times; events: 1=death, 0=censored."""
    times = np.asarray(times, float); events = np.asarray(events, int)
    order = np.argsort(times); times, events = times[order], events[order]
    n = len(times); surv = 1.0
    xs, ys = [0], [100.0]
    at_risk = n
    for t, e in zip(times, events):
        if e == 1:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
        xs += [t, t]; ys += [ys[-1], surv * 100]
    if tmax:
        xs.append(tmax); ys.append(ys[-1])
    ax.plot(xs, ys, color=color, lw=1.25, label=label, zorder=3)
    ax.set_ylim(0, 105); ax.set_ylabel("% survival")


def log_dotplot_with_lod(ax, groups, labels, lod, colors=None, open_mask=None, ms=18):
    """Log-scale per-animal dots, mean line, LOD as a thick grey band labelled in-plot.

    open_mask lets open vs filled circles carry a real second variable (Waldman Fig 3E).
    """
    colors = colors or [C.GREY_DARK] * len(groups)
    rng = np.random.default_rng(2)
    for i, (g, col) in enumerate(zip(groups, colors)):
        g = np.asarray(g, float)
        om = np.zeros(len(g), bool) if open_mask is None else np.asarray(open_mask[i], bool)
        x = i + rng.uniform(-0.14, 0.14, len(g))
        ax.scatter(x[~om], g[~om], s=ms, color=col, edgecolor="none", zorder=3)
        ax.scatter(x[om], g[om], s=ms, facecolor="white", edgecolor=col, linewidth=0.8, zorder=3)
        fin = np.isfinite(g) & (g > 0)
        if fin.any():
            ax.plot([i - 0.2, i + 0.2], [np.median(g[fin])] * 2, color=C.INK, lw=1.0, zorder=4)
    ax.set_yscale("log")
    ax.axhspan(lod * 0.85, lod * 1.15, color=C.GREY, alpha=0.9, lw=0, zorder=1)
    ax.text(0.98, lod * 1.25, "limit of detection", ha="right", va="bottom", fontsize=5.6,
            color=C.GREY_DARK, transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels); rotate_ticks(ax)


# ----------------------------------------------------------------- statistics
def bracket(ax, i, j, y, text, lw=0.7, tick=None, size=6.0):
    """The only statistic drawn on a panel: a bracket plus asterisks or n.s."""
    tick = tick if tick is not None else (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([i, i, j, j], [y, y + tick, y + tick, y], color=C.INK, lw=lw, clip_on=False)
    ax.text((i + j) / 2, y + tick * 1.25, text, ha="center", va="bottom", fontsize=size,
            color=C.INK)


def stars(p):
    """The convention used in these papers; n.s. is written out and shown, not omitted."""
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 \
        else "*" if p < 0.05 else "n.s."


def stat_note(test, n, convention="*p<0.05, **p<0.01, ***p<0.001, ****p<0.0001"):
    """Build the legend sentence. Every p needs its test, its n and the unit of replication."""
    return f"{test}; n = {n}. {convention}."


# ----------------------------------------------------------------- micrographs
def micrograph_row(axes, images, channel_names, channel_colors, row_label=None,
                   scalebar_ax=0, scalebar_frac=0.25, scalebar_label=None):
    """Greyscale channels in a row, channel name coloured to match, one white scale bar."""
    for k, (ax, im) in enumerate(zip(axes, images)):
        ax.imshow(im, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if channel_names and k < len(channel_names):
            ax.set_title(channel_names[k], fontsize=6.4, color=channel_colors[k])
        if k == scalebar_ax:
            w = im.shape[1]
            ax.plot([w * 0.06, w * 0.06 + w * scalebar_frac],
                    [im.shape[0] * 0.93] * 2, color="white", lw=2.2, solid_capstyle="butt")
            if scalebar_label:
                ax.text(w * 0.06, im.shape[0] * 0.90, scalebar_label, color="white",
                        fontsize=5.6, va="bottom")
    if row_label:
        axes[0].set_ylabel(row_label, fontsize=6.4, style="italic", rotation=0,
                           ha="right", va="center", labelpad=14)
