"""spaCR figure style helper (rules of the spacr-graphs skill, 2026-11-18).

Import from any figure script:

    import sys; sys.path.insert(0, "/home/olafsson/.claude/skills/spacr-graphs")
    import spacr_figstyle as fs
    fs.apply_style()
    fig = fs.figure(7.2, 9.0)                       # inches (one full page width = 7.2 in)
    ax = fs.cond_axes(fig, left=0.6, bottom=6.0, n=3, height=1.4)   # width = 3 slots, always
    fs.cond_plot(ax, [ctrl_vals, a_vals, b_vals], colors=[None, fs.BLUE, fs.TEAL], control=0)
    fs.panel_letter(fig, ax, "A")
    fs.dev_mark(fig)                                # development panels only
    problems = fs.check_figure(fig); print(problems)  # overlaps, clipping, width rules
    fig.savefig("x.png", dpi=300); fig.savefig("x.svg")

What it enforces
- Continuous-axis plots (histograms, scatter, lines, sweeps) are SQUARE: square_axes(); heatmaps have
  square cells: heatmap_axes(); only categorical-condition graphs are wide (cond_axes, slot rule).
- Schematics: free_axes(..., kind="schematic"); check_figure flags more than 10 words in one.
- Round 4 (2026-11-18): every text >= MIN_PT = 8 pt; error bars and box whiskers black; share_ylim() for
  compared panels (FDR axes FDR_LIM = 0-1 %); tight_schematic(); all four are checked by check_figure.
- Identity colours: GUIDE (teal) and DECOY (grey) everywhere; check_figure flags a legend entry named
  "guide..." or "decoy..." drawn in another colour.
- One slot per condition: SLOT_IN inches. Axis width = n * SLOT_IN, so every graph with the same
  number of conditions has the same width, and every bar, box and violin is BAR_FRAC * SLOT_IN wide
  with the same gap, in every graph of every figure.
- AUTO_N: more than 20 points in a group -> box (default) or violin instead of bar + whiskers.
- ALPHA = 0.6 for every data element (bars, boxes, violins, points, lines, fills).
- The control / reference is always CONTROL grey; pass control=<index> and its colour is forced.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# ---------------------------------------------------------------- geometry (inches)
SLOT_IN = 0.22          # width of one condition slot (fits two 8 pt digits; round 4, 2026-11-18)
BAR_FRAC = 0.60         # bar / box / violin width as a fraction of the slot (gap = 0.40 slot)
AUTO_N = 20             # > AUTO_N points per group -> box or violin
ALPHA = 0.60            # every graph element

# ---------------------------------------------------------------- palette (2026-11-18)
# Taken from the reference papers (majority: neutral grey control + one saturated blue accent;
# teal-green and magenta as second / third series). Never brown. Control always grey.
CONTROL = "#8C8C8C"     # control / reference condition (all five papers: mid grey ~#888-#989898)
BACKGROUND = "#C8C8C8"  # not-highlighted points in scatters (grey cloud)
BLUE = "#0A77B8"        # primary accent (Nat Microbiol, eLife, Nat Commun)
TEAL = "#0A9878"        # second series (eLife teal-green; spaCR teal)
NAVY = "#283868"        # third series / dark emphasis
LIGHTBLUE = "#78AED2"   # paired lighter tint of BLUE (e.g. -drug vs +drug pairs)
MAGENTA = "#C878A8"     # fourth series (EMBO J, bioRxiv accent)
AMBER = "#F0A830"       # fifth series only (never darker towards brown)
SERIES = [BLUE, TEAL, NAVY, MAGENTA, LIGHTBLUE, AMBER]
# One colour per identity across all panels and figures (maintainer, 2026-11-18). An entity that is the
# control throughout (decoys) is grey everywhere; the real entity (guides) keeps its colour everywhere.
# Do not reuse an identity colour for anything else in the same figure.
GUIDE = TEAL
GUIDE_LIGHT = "#8CCFBF"   # a lighter tint of the guide colour (e.g. one-mismatch guide reads)
DECOY = CONTROL
IDENTITY = {"guide": GUIDE, "decoy": DECOY}
REFLINE = dict(color="black", lw=0.6, ls=(0, (1.5, 1.5)), alpha=ALPHA)   # dotted reference line

MIN_PT = 8.0            # minimum size of ANY text (maintainer, 2026-11-18 round 4); size panels from it
FONT_PT = 8.0           # body text, axis labels
TICK_PT = 8.0           # tick labels
SMALL_PT = 8.0          # value labels, annotations, legends
FDR_LIM = (0, 1)        # FDR (%) axes run 0-1 % unless the data need otherwise
LETTER_PT = 10.0        # panel letters (bold, upper case)
AXIS_LW = 0.6           # axis frame and ticks
DATA_LW = 1.0           # data lines
MEDIAN_LW = 0.9
POINT_S = 4.0           # replicate points (scatter s, pt^2) n <= AUTO_N
SMALL_POINT_S = 1.2     # many-point overlays (n > AUTO_N)

_FONT_DIRS = ("/home/olafsson/.local/share/fonts", "/home/olafsson/.local/share/fonts/open_sans")


def apply_style():
    """Set rcParams: Open Sans, 7 pt, boxed axes (all four spines), outward ticks, 300 dpi."""
    for p in _FONT_DIRS:
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.lower().endswith(".ttf"):
                    font_manager.fontManager.addfont(os.path.join(p, fn))
    plt.rcParams.update({
        "font.family": ["Open Sans", "DejaVu Sans"], "font.weight": "light", "font.size": FONT_PT,
        "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "axes.titleweight": "regular",
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": AXIS_LW,
        "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "legend.frameon": False, "legend.fontsize": TICK_PT, "legend.handlelength": 1.2,
        "lines.linewidth": DATA_LW, "svg.fonttype": "none", "savefig.dpi": 300,
        "axes.titlepad": 2, "axes.labelpad": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "errorbar.capsize": 0,
    })


def figure(width_in=7.2, height_in=9.0):
    fig = plt.figure(figsize=(width_in, height_in))
    fig._fs_axes = []
    return fig


def _rect(fig, left, bottom, width, height):
    W, H = fig.get_size_inches()
    return [left / W, bottom / H, width / W, height / H]


def cond_axes(fig, left, bottom, n, height, **kw):
    """Axes for a condition graph: width = n * SLOT_IN inches, x limits = slots 0..n-1."""
    ax = fig.add_axes(_rect(fig, left, bottom, n * SLOT_IN, height), **kw)
    ax.set_xlim(-0.5, n - 0.5)
    ax._fs_n = n
    fig._fs_axes.append(ax)
    return ax


def free_axes(fig, left, bottom, width, height, kind="other", **kw):
    """Axes outside the rules (schematics, image panels, kind="other"). For data plots use
    square_axes (continuous axes), heatmap_axes (matrices) or cond_axes (categorical conditions)."""
    ax = fig.add_axes(_rect(fig, left, bottom, width, height), **kw)
    ax._fs_n = None
    ax._fs_kind = kind
    fig._fs_axes.append(ax)
    return ax


def square_axes(fig, left, bottom, size, **kw):
    """Continuous-axis plot (histogram, density, scatter, line, sweep, curve): SQUARE plot area,
    size x size inches (rule 2026-11-18: a graph takes no more space than it needs)."""
    ax = fig.add_axes(_rect(fig, left, bottom, size, size), **kw)
    ax.set_box_aspect(1)
    ax._fs_n = None
    ax._fs_kind = "continuous"
    fig._fs_axes.append(ax)
    return ax


def make_square(ax):
    """Mark an existing continuous-axis plot as square (box aspect 1) so check_figure tests it."""
    ax.set_box_aspect(1)
    ax._fs_kind = "continuous"
    return ax


def diverging_cmap(low=MAGENTA, high=BLUE, name="spacr_div"):
    """Pale second hue -> white -> blue. Use with centred_norm() so the expected value is white."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, [low, "white", high])


def centred_norm(vmin, vcenter, vmax):
    from matplotlib.colors import TwoSlopeNorm
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def mark_undefined(ax, cells, size=None):
    """Cells that cannot hold a value (e.g. true = called for a mismatch): a thin black cross, never a
    colour from the scale. Explain it in the legend."""
    for (r_, c_) in cells:
        ax.text(c_, r_, "×", ha="center", va="center", fontsize=size or FONT_PT, color="black", alpha=ALPHA)


def heatmap_axes(fig, left, bottom, ncols, nrows, cell_in=0.26, **kw):
    """Heatmap with SQUARE cells: width = ncols * cell_in, height = nrows * cell_in."""
    ax = fig.add_axes(_rect(fig, left, bottom, ncols * cell_in, nrows * cell_in), **kw)
    ax._fs_n = None
    ax._fs_kind = "heatmap"
    ax._fs_cells = (ncols, nrows)
    fig._fs_axes.append(ax)
    return ax


def color_list(n, colors=None, control=None):
    cols = list(colors) if colors is not None else [SERIES[i % len(SERIES)] for i in range(n)]
    cols = [c if c is not None else BLUE for c in cols]
    for c in ([control] if isinstance(control, int) else (control or [])):
        cols[c] = CONTROL
    return cols


def jitter_points(ax, x, vals, color, s=None, rng=None, width=None, top=None, offscale=None):
    """Jittered points inside the element width. Points above `top` become open triangles on the edge."""
    rng = rng if rng is not None else np.random.default_rng(0)
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    w = BAR_FRAC * 0.8 if width is None else width
    s = (POINT_S if len(vals) <= AUTO_N else SMALL_POINT_S) if s is None else s
    xs = x + rng.uniform(-w / 2, w / 2, len(vals))
    top = ax.get_ylim()[1] if top is None else top
    on = vals <= top
    ax.scatter(xs[on], vals[on], s=s, color=color, alpha=ALPHA, lw=0, zorder=3)
    if (~on).any():
        ax.scatter(xs[~on], np.full((~on).sum(), top), s=5, marker="^", facecolor="none", edgecolor=color,
                   lw=0.5, alpha=ALPHA, zorder=3, clip_on=False)
        if offscale is not None:
            offscale.append(int((~on).sum()))
    return int((~on).sum())


def share_ylim(axes, lo, hi, hide_inner=True):
    """Same y limits for panels that will be compared (rule 2026-11-18). Registers the group for
    check_figure; with hide_inner, only the left-most axes keeps its y tick labels."""
    axes = list(axes)
    for i, a in enumerate(axes):
        a.set_ylim(lo, hi)
        a._fs_share = id(axes[0])
        if hide_inner and i:
            a.tick_params(labelleft=False)
    return axes


def tight_schematic(ax, pad=0.02):
    """Fit a schematic's limits to its drawn content (no wasted white space)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bbs = [c.get_window_extent(r) for c in ax.patches + ax.texts + ax.lines]
    bbs = [b for b in bbs if b.width > 0 or b.height > 0]
    x0, y0 = inv.transform((min(b.x0 for b in bbs), min(b.y0 for b in bbs)))
    x1, y1 = inv.transform((max(b.x1 for b in bbs), max(b.y1 for b in bbs)))
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)
    return ax


def cond_plot(ax, data, colors=None, control=None, kind="auto", many="box", points=True, rng=None,
              bar_values=None, offscale=None, median=True):
    """One element per slot. data: list of arrays (one per condition) or scalars.

    kind='auto': > AUTO_N values -> `many` ('box' or 'violin'); else bar (mean, or bar_values[i]) + points.
    kind='bar' forces bars (single values, e.g. percentages). Returns the kind used per slot.
    Set the y limits BEFORE calling when points may run off the top (open triangles mark them).
    """
    n = len(data)
    cols = color_list(n, colors, control)
    w = BAR_FRAC
    used = []
    for i, d in enumerate(data):
        arr = np.atleast_1d(np.asarray(d, float))
        arr = arr[np.isfinite(arr)]
        k = kind if kind != "auto" else (many if len(arr) > AUTO_N else "bar")
        c = cols[i]
        if k == "bar":
            v = bar_values[i] if bar_values is not None else (arr.mean() if len(arr) else np.nan)
            ax.bar(i, v, w, color=c, alpha=ALPHA, lw=0, zorder=1)
            if points and len(arr) > 1:
                jitter_points(ax, i, arr, c, rng=rng, offscale=offscale)
        elif k == "box":
            ax.boxplot([arr], positions=[i], widths=w, patch_artist=True, showfliers=False, whis=(5, 95),
                       medianprops=dict(color="black", lw=MEDIAN_LW, alpha=ALPHA),
                       boxprops=dict(facecolor=c, edgecolor=c, lw=0.6, alpha=ALPHA),
                       whiskerprops=dict(color="black", lw=0.8, alpha=ALPHA),
                       capprops=dict(color="black", lw=0.8, alpha=ALPHA), zorder=2)
            if points:
                jitter_points(ax, i, arr, c, rng=rng, offscale=offscale, width=w * 0.9)
        elif k == "violin":
            top = ax.get_ylim()[1]
            vp = ax.violinplot([np.clip(arr, None, top)], positions=[i], widths=w, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(c)
                b.set_edgecolor("none")
                b.set_alpha(ALPHA)
            if median:
                ax.plot([i - w / 2, i + w / 2], [np.median(arr)] * 2, color="black", lw=MEDIAN_LW, alpha=ALPHA,
                        zorder=4)
        used.append(k)
    ax.set_xlim(-0.5, n - 0.5)
    return used


def ci_marker(ax, x, est, lo, hi, color="black"):
    """Pooled estimate with 95 % CI: white circle, black edge, vertical CI line (60 % alpha)."""
    ax.plot([x, x], [lo, hi], color=color, lw=0.8, alpha=ALPHA, zorder=5, solid_capstyle="butt")
    ax.scatter([x], [est], s=9, facecolor="white", edgecolor=color, lw=0.7, zorder=6)


def hist_overlay(ax, edges, series, density_pct=True):
    """Overlaid filled histograms (step-filled), each at ALPHA. series: [(counts, color, label), ...]."""
    out = []
    for counts, c, lab in series:
        counts = np.asarray(counts, float)
        y = 100 * counts / counts.sum() if density_pct else counts
        out.append(ax.stairs(y, edges, fill=True, color=c, alpha=ALPHA, lw=0, label=lab))
    return out


def bracket(ax, x1, x2, y, text, h=None):
    """Significance bracket with the exact text (p value) centred above it."""
    h = h if h is not None else 0.02 * np.diff(ax.get_ylim())[0]
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color="black", lw=0.6)
    ax.text((x1 + x2) / 2, y + h * 0.3, text, ha="center", va="bottom", fontsize=SMALL_PT)


def group_line(ax, x1, x2, label, y_axes=-0.30):
    """Group label under the x axis: thin line spanning slots x1..x2, label below (paper style)."""
    tr = ax.get_xaxis_transform()
    ax.plot([x1 - 0.35, x2 + 0.35], [y_axes, y_axes], color="black", lw=0.6, transform=tr, clip_on=False)
    ax.text((x1 + x2) / 2, y_axes - 0.03, label, ha="center", va="top", transform=tr, fontsize=TICK_PT)


def panel_letter(fig, ax, letter, dx_in=0.42, dy_in=0.10):
    """Bold upper-case panel letter, a fixed distance (inches) left of and above the axes corner."""
    W, H = fig.get_size_inches()
    bb = ax.get_position()
    fig.text(bb.x0 - dx_in / W, bb.y1 + dy_in / H, letter, fontsize=LETTER_PT, fontweight="bold",
             va="bottom", ha="left")


def dev_mark(fig, text="DEVELOPMENT PANEL – made by a script, not paper data yet"):
    fig.text(0.99, 0.995, text, ha="right", va="top", fontsize=SMALL_PT, color=CONTROL)


def check_figure(fig, pad_px=1.0):
    """Return a list of problems: overlapping texts, text outside the canvas, condition axes whose
    width breaks the n * SLOT_IN rule. Run before calling a figure done (see checklist)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    probs = []
    hidden = set()  # tick labels outside the view limits are never drawn
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update((id(tk.label1), id(tk.label2)))
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    boxes = []
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-6:
            probs.append("font %.1f pt < %g pt: %r" % (t.get_fontsize(), MIN_PT, t.get_text()[:30]))
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0:
            continue
        if bb.x0 < -pad_px or bb.y0 < -pad_px or bb.x1 > W + pad_px or bb.y1 > H + pad_px:
            probs.append("clipped text: %r at px (%d, %d)" % (t.get_text()[:40], bb.x0, bb.y0))
        boxes.append((t, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ov = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if boxes[i][0].get_rotation() % 90 and boxes[j][0].get_rotation() % 90:
                continue  # two slanted labels: their axis-aligned boxes overlap without the glyphs touching
            if ov[0] > pad_px and ov[1] > pad_px:
                probs.append("overlap: %r / %r" % (boxes[i][0].get_text()[:30], boxes[j][0].get_text()[:30]))
    # identity colours: every legend entry named after an identity must use that identity's colour
    from matplotlib.colors import to_rgb
    legends = list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend() is not None]
    for lg in legends:
        for hd, tx in zip(lg.legend_handles, lg.get_texts()):
            lab = tx.get_text().strip().lower()
            for name, col in IDENTITY.items():
                if lab == name or lab.startswith(name + " ") or lab.startswith(name + "-"):
                    try:
                        got = hd.get_facecolor() if hasattr(hd, "get_facecolor") else hd.get_color()
                    except Exception:
                        continue
                    if np.abs(np.array(to_rgb(got)) - np.array(to_rgb(col))).max() > 0.02:
                        probs.append("identity colour: %r drawn in %s, want %s" % (tx.get_text(), got, col))
    # error bars are always black
    for a in fig.axes:
        for cont in a.containers:
            if isinstance(cont, matplotlib.container.ErrorbarContainer):
                for ln in cont.lines[2]:
                    cols = ln.get_colors() if hasattr(ln, "get_colors") else [ln.get_color()]
                    if any(np.abs(np.array(to_rgb(c)[:3])).max() > 0.02 for c in cols):
                        probs.append("error bar not black")
    # shared axis limits within a comparison group
    groups = {}
    for a in fig.axes:
        if hasattr(a, "_fs_share"):
            groups.setdefault(a._fs_share, []).append(tuple(np.round(a.get_ylim(), 9)))
    for g, lims in groups.items():
        if len(set(lims)) > 1:
            probs.append("compared panels have different y limits: %s" % sorted(set(lims)))
    fw, fh = fig.get_size_inches()
    for ax in getattr(fig, "_fs_axes", []):
        n = getattr(ax, "_fs_n", None)
        pos = ax.get_position(original=False)
        w, h = pos.width * fw, pos.height * fh
        if n:
            if abs(w - n * SLOT_IN) > 1e-3:
                probs.append("axis width %.3f in for n=%d (want %.3f)" % (w, n, n * SLOT_IN))
            continue
        kind = getattr(ax, "_fs_kind", "other")
        if kind == "continuous" and abs(w / h - 1) > 0.02:
            probs.append("continuous-axis plot not square: %.2f x %.2f in" % (w, h))
        if kind == "schematic":
            import re
            words = sum(len(re.findall(r"[A-Za-z]{2,}", t.get_text())) for t in ax.texts if t.get_visible())
            if words > 10:
                probs.append("schematic has %d words (aim < 10; no sentences)" % words)
            bbs = [c.get_window_extent(r) for c in ax.patches + ax.texts + ax.lines]
            bbs = [b for b in bbs if b.width > 0 or b.height > 0]
            if bbs:
                axb = ax.get_window_extent(r)
                cw = max(b.x1 for b in bbs) - min(b.x0 for b in bbs)
                ch = max(b.y1 for b in bbs) - min(b.y0 for b in bbs)
                if cw < 0.85 * axb.width or ch < 0.85 * axb.height:
                    probs.append("schematic has wasted white space (content %.0f %% x %.0f %% of its box); "
                                 "use tight_schematic()" % (100 * cw / axb.width, 100 * ch / axb.height))
        if kind == "heatmap":
            # a heatmap may only use colours from its own colour bar: no extra filled patches
            for pt in ax.patches:
                fc = pt.get_facecolor()
                if pt.get_fill() and fc[3] > 0:
                    probs.append("heatmap has a filled patch outside its colour scale (%s)" % (fc,))
                    break
            nc, nr = ax._fs_cells
            if abs((w / nc) / (h / nr) - 1) > 0.02:
                probs.append("heatmap cells not square: %.3f x %.3f in" % (w / nc, h / nr))
    return probs
