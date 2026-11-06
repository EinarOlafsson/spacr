"""Standard assay report: A4 pages with pagination, 16-bit tiles and a figure legend.

Used by :func:`spacr.submodules.analyze_recruitment` (``report`` setting) and
written to be reused by other plate assays. Three building blocks:

**Layout engine** (:class:`ReportItem`, :class:`ReportBlock`,
:func:`paginate`, :func:`render_report`). A report is a list of blocks, each a
heading plus panel items. An item is half or full page width with a height in
mm. Items are packed into rows (two half items, or one full item), rows into
A4 portrait pages (210 x 297 mm). A block that does not fit the rest of a page
starts on a new page; a block taller than a page flows on, row by row, with
its heading repeated. Text sizes are fixed in points (never below 5 pt), so a
long report gets more pages, never smaller text. Panels whose condition axis
does not fit the page width are split into chunks of conditions
(:func:`condition_chunks`) drawn as consecutive rows.

**Tiles** (:func:`select_tiles`, :func:`extract_tile`, :func:`render_tile`,
:func:`build_tiles`). Crops of host cells around their mask centroid from the
spaCR merged arrays (image channels then mask planes). The raw crop is written
as a multichannel TIFF with the input dtype and unchanged values; the RGB
composite and the montage are written as 16-bit PNG (OpenCV, BGR order) and
16-bit TIFF (RGB). Only the figure PDF / PNG are 8-bit.

**Legend** (:func:`describe_test`, :func:`filter_sentences`,
:func:`write_legend`). Generic text for every panel: what is plotted, unit of
statistics, n per group, test and why, filters with their counts.

Statistics come from :func:`spacr.assay_panels.select_and_run_test`.
"""
from __future__ import annotations

import datetime
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .assay_panels import (ALPHA, WELL_KEYS, _DEFAULT_COLORS, _despine, _set_xticks,
                           aggregate_to_unit, bar_spec, display_label, draw_bar_panel, fmt_p,
                           panel_folder_name, select_and_run_test, stars)

__all__ = [
    "A4_MM", "MIN_FONT_PT", "ReportItem", "ReportBlock", "paginate", "render_report",
    "condition_chunks", "object_types", "well_intensity_long", "well_counts",
    "grouped_spec", "draw_grouped_panel", "stats_long", "write_panel_folder",
    "resolve_mask_dims", "select_tiles", "extract_tile", "render_tile", "write_rgb16",
    "display_ranges", "build_tiles", "describe_test", "filter_sentences", "write_legend",
    "build_assay_report",
]

A4_MM = (210.0, 297.0)
MARGIN_MM = 12.0
GAP_MM = 5.0
HEADING_MM = 7.0
MIN_FONT_PT = 5.0
MAX_BRACKETS = 4  # more significant pairs: brackets left out, all pairs in stats.csv
MM = 1 / 25.4
_CORE_OBJECTS = ("cell", "nucleus", "cytoplasm", "pathogen")
_MEAN_RE = re.compile(r"^(?P<obj>.+?)_channel_(?P<chan>\d+)_mean_intensity$")
COLOURS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "magenta": (1, 0, 1),
           "cyan": (0, 1, 1), "yellow": (1, 1, 0), "grey": (1, 1, 1), "gray": (1, 1, 1),
           "white": (1, 1, 1)}


# =========================================================================== layout
@dataclass
class ReportItem:
    """One panel (or one chunk of a panel) on a page.

    :param draw: ``draw(fig, rect)`` with ``rect = (x0, y0, w, h)`` in figure
        fractions; it adds its own axes inside the rect.
    :param width: ``'half'`` or ``'full'`` page width.
    :param height_mm: height reserved on the page, panel ID strip included.
    """
    pid: str
    label: str
    draw: Callable
    width: str = "half"
    height_mm: float = 60.0


@dataclass
class ReportBlock:
    title: str
    items: List[ReportItem] = field(default_factory=list)


def _content_width(page=A4_MM, margin=MARGIN_MM):
    return page[0] - 2 * margin


def _half_width(page=A4_MM, margin=MARGIN_MM, gap=GAP_MM):
    return (_content_width(page, margin) - gap) / 2


def _rows(items):
    """Pack items into rows: two half items side by side, full items alone."""
    rows, pending = [], None
    for it in items:
        if it.width == "full":
            if pending is not None:
                rows.append([pending])
                pending = None
            rows.append([it])
        elif pending is None:
            pending = it
        else:
            rows.append([pending, it])
            pending = None
    if pending is not None:
        rows.append([pending])
    return rows


def paginate(blocks: Sequence[ReportBlock], page=A4_MM, margin=MARGIN_MM, gap=GAP_MM,
             heading=HEADING_MM) -> List[List[dict]]:
    """Place blocks on A4 pages.

    :returns: one list per page of placements, each a dict with ``kind``
        (``'heading'`` or ``'item'``), ``text`` / ``item`` and ``rect_mm``
        ``(x0, y_top, w, h)`` measured from the top-left page corner.
    """
    usable = page[1] - 2 * margin
    full, half = _content_width(page, margin), _half_width(page, margin, gap)
    pages: List[List[dict]] = [[]]
    y = 0.0

    def new_page():
        nonlocal y
        pages.append([])
        y = 0.0

    for block in blocks:
        rows = _rows(block.items)
        if not rows:
            continue
        heights = [max(it.height_mm for it in r) for r in rows]
        total = heading + sum(heights) + gap * (len(rows) - 1)
        remaining = usable - y
        # keep a block together unless most of it fits here anyway (then flow on)
        if total > remaining and y > 0 and (remaining < heading + heights[0]
                                             or (total <= usable and remaining < 0.6 * total)):
            new_page()
        pages[-1].append(dict(kind="heading", text=block.title,
                              rect_mm=(margin, margin + y, full, heading)))
        y += heading
        on_page = 0
        for k, (row, h) in enumerate(zip(rows, heights)):
            if y + h > usable and (on_page or y > heading + 1e-9):
                new_page()
                on_page = 0
                pages[-1].append(dict(kind="heading", text=f"{block.title} (continued)",
                                      rect_mm=(margin, margin + y, full, heading)))
                y += heading
            x = margin
            for it in row:
                w = full if it.width == "full" else half
                pages[-1].append(dict(kind="item", item=it, rect_mm=(x, margin + y, w, h)))
                x += w + gap
            y += h + gap
            on_page += 1
        y += gap
    return [p for p in pages if p]


def fit_axes(fig, ax, rect, iterations=3, min_frac=0.15):
    """Shrink ``ax`` until its tight bounding box (labels, title, legend) fits ``rect``."""
    x0, y0, w, h = rect
    ax.set_position([x0, y0, w, h])
    renderer = fig.canvas.get_renderer()
    for _ in range(iterations):
        bb = ax.get_tightbbox(renderer)
        if bb is None:
            return
        bb = bb.transformed(fig.transFigure.inverted())
        pos = ax.get_position()
        left, bottom = pos.x0 - bb.x0, pos.y0 - bb.y0
        right, top = bb.x1 - pos.x1, bb.y1 - pos.y1
        nw = max(w - left - right, min_frac * w)
        nh = max(h - bottom - top, min_frac * h)
        ax.set_position([x0 + left, y0 + bottom, nw, nh])


def _new_figure(width_mm, height_mm):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(width_mm * MM, height_mm * MM))
    FigureCanvasAgg(fig)
    return fig


ID_STRIP_MM = 4.5


def _draw_item(fig, item, rect_mm, page_h_mm, page_w_mm, show_id=True):
    x, y_top, w, h = rect_mm
    fx, fw = x / page_w_mm, w / page_w_mm
    strip = ID_STRIP_MM if show_id else 0.0
    fy = (page_h_mm - y_top - h) / page_h_mm
    fh = (h - strip) / page_h_mm
    if show_id:
        fig.text(fx, (page_h_mm - y_top) / page_h_mm, item.label, fontsize=9,
                 fontweight="bold", va="top", ha="left")
    item.draw(fig, (fx, fy, fw, fh))


def render_report(pages: List[List[dict]], pdf_path: str, legend_text: str = "",
                  title: str = "", page=A4_MM, dpi: int = 200, png=True) -> dict:
    """Draw paginated placements (and the legend pages) into one PDF.

    :returns: ``{'pdf': path, 'png': [paths], 'n_pages': int, 'n_legend_pages': int}``.
        PNG files are ``<stem>.png`` for page 1 and ``<stem>_page_NN.png``
        for the next pages.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    stem = os.path.splitext(pdf_path)[0]
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    legend_pages = _legend_pages(legend_text, page) if legend_text else []
    n_total = len(pages) + len(legend_pages)
    pngs = []
    with PdfPages(pdf_path) as pdf:
        for k, placements in enumerate(pages, start=1):
            fig = _new_figure(*page)
            fig.set_facecolor("white")
            _page_header(fig, title, k, n_total, page)
            for pl in placements:
                if pl["kind"] == "heading":
                    x, y_top, w, h = pl["rect_mm"]
                    fig.text(x / page[0], (page[1] - y_top - 1.0) / page[1], pl["text"],
                             fontsize=10, fontweight="bold", va="top", ha="left")
                else:
                    _draw_item(fig, pl["item"], pl["rect_mm"], page[1], page[0])
            pdf.savefig(fig)
            if png:
                path = f"{stem}.png" if k == 1 else f"{stem}_page_{k:02d}.png"
                fig.savefig(path, dpi=dpi)
                pngs.append(path)
        for j, lines in enumerate(legend_pages, start=len(pages) + 1):
            fig = _new_figure(*page)
            _page_header(fig, title, j, n_total, page)
            fig.text(MARGIN_MM / page[0], 1 - (MARGIN_MM + 2) / page[1], "\n".join(lines),
                     fontsize=_LEGEND_PT, va="top", ha="left", family="DejaVu Sans",
                     linespacing=1.3)
            pdf.savefig(fig)
            if png:
                path = f"{stem}_page_{j:02d}.png"
                fig.savefig(path, dpi=dpi)
                pngs.append(path)
    return dict(pdf=pdf_path, png=pngs, n_pages=n_total, n_legend_pages=len(legend_pages))


def _page_header(fig, title, k, n, page):
    fig.text(MARGIN_MM / page[0], 1 - 5 / page[1], title, fontsize=7, va="top", ha="left",
             color="#555555")
    fig.text(1 - MARGIN_MM / page[0], 1 - 5 / page[1], f"page {k} of {n}", fontsize=7,
             va="top", ha="right", color="#555555")


_LEGEND_PT = 7.5


def _legend_pages(text, page=A4_MM):
    width_chars = int(_content_width(page) / (_LEGEND_PT * 0.6 / 72 * 25.4))
    line_mm = _LEGEND_PT * 1.3 / 72 * 25.4
    per_page = int((page[1] - 2 * MARGIN_MM - 4) / line_mm)
    lines: List[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width_chars) or [""])
    return [lines[i:i + per_page] for i in range(0, len(lines), per_page)]


# =========================================================================== sizing helpers
def _label_texts(order, labels):
    return [display_label(g, labels) for g in order]


def _rotation(texts):
    longest = max((len(t) for t in texts), default=0)
    return 0 if longest * len(texts) <= 24 and longest <= 10 else 35


def _xlabel_mm(texts, fontsize=8):
    """Height taken by x tick labels as drawn by assay_panels._set_xticks."""
    char_mm = fontsize * 0.55 / 72 * 25.4
    line_mm = fontsize * 1.25 / 72 * 25.4
    if _rotation(texts) == 0:
        return line_mm + 2
    longest = max((len(t) for t in texts), default=0)
    return longest * char_mm * math.sin(math.radians(35)) + line_mm + 2


def condition_chunks(order, per_category_mm, fixed_mm, max_per_row, page=A4_MM):
    """Split the condition order into chunks that fit one page width.

    :returns: (width, chunks) with width ``'half'`` or ``'full'``.
    """
    order = list(order)
    n = max(len(order), 1)
    if fixed_mm + n * per_category_mm <= _half_width(page) and n <= max(1, max_per_row // 2):
        return "half", [order]
    fit = int((_content_width(page) - fixed_mm) // per_category_mm)
    size = max(1, min(int(max_per_row), fit))
    if n <= size:
        return "full", [order]
    k = int(math.ceil(n / size))
    size = int(math.ceil(n / k))  # balance the chunks
    return "full", [order[i:i + size] for i in range(0, n, size)]


# =========================================================================== object discovery
def object_types(df: pd.DataFrame, channels: Optional[Sequence[int]] = None) -> List[str]:
    """Object prefixes with ``<obj>_channel_<c>_mean_intensity`` columns.

    cell, nucleus, cytoplasm and pathogen come first, any other prefix
    (organelles ...) after them in alphabetical order.
    """
    found = set()
    for col in df.columns:
        m = _MEAN_RE.match(str(col))
        if m and (channels is None or int(m.group("chan")) in set(map(int, channels))):
            found.add(m.group("obj"))
    core = [o for o in _CORE_OBJECTS if o in found]
    return core + sorted(found - set(core))


def well_intensity_long(df, group, channel, objects=None, well_keys=WELL_KEYS):
    """Mean intensity of ``channel`` per well for every object type (long format).

    :returns: columns ``group``, well keys, ``object_type``, ``column``,
        ``mean_intensity``, ``n_objects``.
    """
    objects = objects or object_types(df, [channel])
    frames = []
    for obj in objects:
        col = f"{obj}_channel_{channel}_mean_intensity"
        if col not in df.columns:
            continue
        unit = aggregate_to_unit(df, col, [group], well_keys=well_keys)
        unit = unit.rename(columns={col: "mean_intensity"})
        unit.insert(len(well_keys) + 1, "object_type", obj)
        unit.insert(len(well_keys) + 2, "column", col)
        frames.append(unit)
    if not frames:
        return pd.DataFrame(columns=[group, *well_keys, "object_type", "column",
                                     "mean_intensity", "n_objects"])
    return pd.concat(frames, ignore_index=True)


def count_source(df, obj, row_object="cell"):
    """How objects of type ``obj`` are counted per well (see :func:`well_counts`)."""
    if obj == row_object:
        return "rows"
    if f"{obj}_prcfo_count" in df.columns:
        return f"{obj}_prcfo_count"
    if f"{obj}_area" in df.columns:
        return f"{obj}_area"
    return None


def well_counts(df, group, obj, row_object="cell", well_keys=WELL_KEYS):
    """Objects of type ``obj`` per well in the analysed frame.

    The frame has one row per ``row_object`` (the host cell in recruitment):
    rows are counted for that type; other types sum ``<obj>_prcfo_count``
    (objects of that type merged into each row) when present, else count rows
    with a ``<obj>_area`` value. Counting from the frame, not the database
    tables, means the count follows every filter applied to the frame.
    """
    src = count_source(df, obj, row_object)
    keys = [group, *well_keys]
    if src is None:
        return pd.DataFrame(columns=keys + ["objects_per_well", "count_source"])
    if src == "rows":
        out = df.groupby(keys, observed=True).size().rename("objects_per_well")
    elif src.endswith("_prcfo_count"):
        v = pd.to_numeric(df[src], errors="coerce").fillna(0)
        out = v.groupby([df[k] for k in keys], observed=True).sum().rename("objects_per_well")
    else:
        v = pd.to_numeric(df[src], errors="coerce").notna().astype(int)
        out = v.groupby([df[k] for k in keys], observed=True).sum().rename("objects_per_well")
    out = out.reset_index()
    out["count_source"] = src
    return out


# =========================================================================== grouped panel
def grouped_spec(data, group, order, objects, value="mean_intensity", title="", ylabel=None,
                 labels=None, colors=None) -> dict:
    """Grouped bars (one colour per object type) with one test per object type."""
    stats = {}
    for obj in objects:
        sub = data[data["object_type"] == obj]
        stats[obj] = select_and_run_test(sub, value, group, order)
    colors = colors or {o: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, o in enumerate(objects)}
    return dict(kind="grouped", data=data, value=value, group=group, order=list(order),
                objects=list(objects), title=title, ylabel=ylabel or value.replace("_", " "),
                labels=labels, colors=colors, stats=stats)


def draw_grouped_panel(ax, spec, order=None, seed=0):
    """Draw a :func:`grouped_spec`: mean ± SD of unit values per object type, dots per unit."""
    order = list(order if order is not None else spec["order"])
    objects, group, value = spec["objects"], spec["group"], spec["value"]
    data = spec["data"]
    rng = np.random.default_rng(seed)
    k = max(len(objects), 1)
    width = 0.8 / k
    vals = pd.to_numeric(data[value], errors="coerce") if len(data) else pd.Series(dtype=float)
    d = data.assign(_v=vals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_v"])
    means_all = d.groupby("object_type")["_v"].mean()
    pos = means_all[means_all > 0]
    log = bool(len(pos) >= 2 and (d["_v"] > 0).all() and pos.max() / pos.min() > 30)
    x = np.arange(len(order))
    for j, obj in enumerate(objects):
        sub = d[d["object_type"] == obj]
        m = sub.groupby(group)["_v"].mean().reindex(order)
        s = sub.groupby(group)["_v"].std().reindex(order).fillna(0)
        off = (j - (k - 1) / 2) * width
        st = spec["stats"][obj]["test"].iloc[0]
        tag = "not tested" if st["test"] == "not tested" else st["significance"]
        ax.bar(x + off, m.to_numpy(float), width=width * 0.92, color=spec["colors"][obj],
               edgecolor="black", linewidth=0.4, alpha=0.85, label=f"{obj} ({tag})")
        ax.errorbar(x + off, m.to_numpy(float), yerr=s.to_numpy(float), fmt="none",
                    ecolor="black", elinewidth=0.6, capsize=1.5)
        for i, g in enumerate(order):
            y = sub.loc[sub[group] == g, "_v"].to_numpy(float)
            if len(y):
                ax.scatter(i + off + rng.uniform(-width * 0.25, width * 0.25, len(y)), y, s=3,
                           color="black", zorder=3, linewidths=0)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(spec["title"] + (" (log scale)" if log else ""), fontsize=9)
    ax.set_ylabel(spec["ylabel"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    _set_xticks(ax, order, spec.get("labels"), None)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              title="object type (test\nacross conditions)", title_fontsize=6)
    _despine(ax)
    return ax


def _draw_spec(ax, spec, order=None):
    if spec["kind"] == "grouped":
        return draw_grouped_panel(ax, spec, order)
    return draw_bar_panel(ax, spec["data"], spec["value"], spec["group"],
                          list(order if order is not None else spec["order"]),
                          stats=spec["stats"], title=spec.get("title", ""),
                          ylabel=spec.get("ylabel"), labels=spec.get("labels"),
                          ymin=spec.get("ymin"), refline=spec.get("refline"),
                          palette=spec.get("palette"), max_brackets=MAX_BRACKETS)


def stats_long(spec) -> pd.DataFrame:
    """Test, assumptions, post hoc and descriptive tables of a spec in one long table."""
    per = spec["stats"] if spec["kind"] == "grouped" else {None: spec["stats"]}
    frames = []
    for obj, st in per.items():
        for name in ("test", "assumptions", "posthoc", "descriptive"):
            t = st.get(name)
            if t is None or not len(t):
                continue
            t = t.copy()
            if spec["group"] in t.columns and name == "descriptive":
                t = t.rename(columns={spec["group"]: "group"})
            t.insert(0, "table", name)
            if obj is not None:
                t.insert(1, "object_type", obj)
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["table"])


def _panel_item_height(spec, order):
    texts = _label_texts(order, spec.get("labels"))
    title_lines = 2 if spec["kind"] == "bar" else 1
    return 40 + 4.2 * title_lines + _xlabel_mm(texts) + ID_STRIP_MM


def _per_category(spec):
    if spec["kind"] == "grouped":
        return 1.9 * len(spec["objects"]) + 2.5, 22 + 30
    return 6.0, 24


def spec_items(pid, spec, max_per_row):
    """Report items for a bar / grouped spec, split into condition chunks if needed."""
    per_cat, fixed = _per_category(spec)
    width, chunks = condition_chunks(spec["order"], per_cat, fixed, max_per_row)
    items = []
    for i, chunk in enumerate(chunks):
        def draw(fig, rect, chunk=chunk):
            ax = fig.add_axes(rect)
            _draw_spec(ax, spec, chunk)
            fit_axes(fig, ax, rect)
        label = pid if i == 0 else f"{pid} (continued)"
        items.append(ReportItem(pid=pid, label=label, draw=draw, width=width,
                                height_mm=_panel_item_height(spec, chunk)))
    return items


def write_panel_folder(out_dir, pid, name, draw, size_mm, data=None, stats=None, dpi=300):
    """One panel folder: ``<Rnn_name>.pdf/.png``, ``data.csv`` and ``stats.csv``.

    :param draw: ``draw(fig, rect)`` as for :class:`ReportItem`.
    :returns: folder path.
    """
    stem = panel_folder_name(pid, name)
    folder = os.path.join(out_dir, stem)
    os.makedirs(folder, exist_ok=True)
    w, h = size_mm
    fig = _new_figure(w, h)
    pad = 2.0
    draw(fig, (pad / w, pad / h, 1 - 2 * pad / w, 1 - 2 * pad / h))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"{stem}.{ext}"), dpi=dpi)
    if data is not None:
        data.to_csv(os.path.join(folder, "data.csv"), index=False)
    if stats is not None:
        stats.to_csv(os.path.join(folder, "stats.csv"), index=False)
    return folder


def _spec_full_size(spec):
    per_cat, fixed = _per_category(spec)
    w = max(80.0, fixed + per_cat * len(spec["order"]))
    return w, _panel_item_height(spec, spec["order"]) - ID_STRIP_MM + 4


def save_spec_panel(out_dir, pid, name, spec, dpi=300):
    def draw(fig, rect):
        ax = fig.add_axes(rect)
        _draw_spec(ax, spec)
        fit_axes(fig, ax, rect)
    return write_panel_folder(out_dir, pid, name, draw, _spec_full_size(spec),
                              data=spec["data"], stats=stats_long(spec), dpi=dpi)


# =========================================================================== tiles
def resolve_mask_dims(merged_dir, names, n_planes, overrides=None):
    """Plane index of each object mask: ``overrides``, else the plane layout
    sidecar of the merged folder, else the last ``len(names)`` planes in order."""
    from .plot import _overlay_mask_dims
    dims = _overlay_mask_dims(os.path.join(merged_dir, "x.npy"), list(names), n_planes)
    for k, v in (overrides or {}).items():
        if v is not None:
            dims[k] = int(v)
    return dims


def _label_number(value):
    m = re.search(r"(\d+)$", str(value))
    return int(m.group(1)) if m else None


def select_tiles(df, group, order, n, seed=0, tile_size=128,
                 edge_column="cell_distance_to_field_edge"):
    """Random objects per group, objects away from the field edge first.

    An object is preferred when ``edge_column`` (distance of its centre to the
    field edge) is at least half the tile size, so the crop lies inside the
    field. Candidates are shuffled with ``seed``; the caller takes the first
    ``n`` that load.

    :returns: {group value: DataFrame of candidates in pick order}.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for g in order:
        sub = df[df[group] == g]
        if not len(sub):
            out[g] = sub
            continue
        perm = sub.iloc[rng.permutation(len(sub))]
        if edge_column in perm.columns:
            inside = pd.to_numeric(perm[edge_column], errors="coerce") >= tile_size / 2
            perm = pd.concat([perm[inside.to_numpy()], perm[~inside.to_numpy()]])
        out[g] = perm
    return out


def extract_tile(arr, mask_plane, label, size):
    """Crop a ``size`` x ``size`` window centred on the centroid of ``label``.

    :returns: (raw, box, padded) - ``raw`` is the exact in-field slice
        ``arr[y0:y1, x0:x1, :]`` (input dtype, values unchanged), ``box`` is
        ``(y0, y1, x0, x1)`` and ``padded`` the slice zero padded to the full
        tile size (equal to ``raw`` away from the edge). None if the label is
        absent.
    """
    m = np.asarray(arr[..., mask_plane])
    ys, xs = np.nonzero(m == label)
    if not len(ys):
        return None
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    h, w = m.shape
    half = size // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + size, x0 + size
    cy0, cx0, cy1, cx1 = max(y0, 0), max(x0, 0), min(y1, h), min(x1, w)
    raw = np.array(arr[cy0:cy1, cx0:cx1, :])
    padded = np.zeros((size, size, arr.shape[2]), dtype=raw.dtype)
    padded[cy0 - y0:cy0 - y0 + raw.shape[0], cx0 - x0:cx0 - x0 + raw.shape[1]] = raw
    return raw, (cy0, cy1, cx0, cx1), padded


def _colour(c):
    if isinstance(c, str):
        if c.lower() not in COLOURS:
            raise ValueError(f"unknown tile colour '{c}'; use one of {sorted(COLOURS)} or an (r, g, b) tuple")
        return COLOURS[c.lower()]
    return tuple(float(v) for v in c)


def render_tile(crop, colour_map, ranges, outlines=()):
    """16-bit RGB composite of an image crop.

    :param colour_map: {channel index: colour name or (r, g, b) in 0-1}.
    :param ranges: {channel index: (low, high)} display range, the same for
        every tile.
    :param outlines: [(binary mask, (r, g, b) in 0-1)], drawn 1 px wide.
    :returns: uint16 array (h, w, 3), RGB order.
    """
    import cv2
    h, w = crop.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for chan, col in colour_map.items():
        lo, hi = ranges[int(chan)]
        x = (crop[..., int(chan)].astype(np.float64) - lo) / max(hi - lo, 1e-9)
        rgb += np.clip(x, 0, 1)[..., None] * np.asarray(_colour(col))[None, None, :]
    out = np.ascontiguousarray((np.clip(rgb, 0, 1) * 65535).round().astype(np.uint16))
    for mask, col in outlines:
        contours, _ = cv2.findContours(np.ascontiguousarray(mask.astype(np.uint8)),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, tuple(int(v * 65535) for v in col), 1)
    return out


def write_rgb16(stem, rgb):
    """Write an RGB uint16 array as ``<stem>.png`` (OpenCV, BGR order) and ``<stem>.tif``."""
    import cv2
    import tifffile
    rgb = np.ascontiguousarray(rgb.astype(np.uint16))
    cv2.imwrite(f"{stem}.png", rgb[..., ::-1])
    tifffile.imwrite(f"{stem}.tif", rgb, photometric="rgb")
    return [f"{stem}.png", f"{stem}.tif"]


def display_ranges(arrays, channels, percentiles=(0.5, 99.9), stride=4):
    """Per channel (low, high) percentiles over pixels of several fields."""
    out = {}
    for c in channels:
        px = np.concatenate([np.asarray(a[::stride, ::stride, int(c)]).ravel() for a in arrays])
        lo, hi = np.percentile(px, percentiles)
        out[int(c)] = (float(lo), float(hi if hi > lo else lo + 1))
    return out


def _safe(text):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text)).strip("_")


def build_tiles(df, group, order, merged_dir, mask_dims, colour_map, out_folder, n=5,
                size=128, seed=0, percentiles=(0.5, 99.9), max_range_fields=24,
                object_column="object_label", file_column="file_name",
                edge_column="cell_distance_to_field_edge", outline_objects=("pathogen", "cell"),
                value_column=None):
    """Pick, crop, render and write the representative tiles of every group.

    :param mask_dims: {'cell': plane, 'pathogen': plane, ...}; ``'cell'`` is
        the object the crop is centred on (``object_column`` holds its label).
    :returns: dict with ``tiles`` {group: [rgb16]}, ``table`` (one row per
        tile), ``ranges``, ``montage`` (uint16 RGB, rows = tiles, columns =
        groups), ``files`` and ``notes``.
    """
    os.makedirs(os.path.join(out_folder, "tiles"), exist_ok=True)
    candidates = select_tiles(df, group, order, n, seed, size, edge_column)
    cell_plane = mask_dims["cell"]
    cache: Dict[str, np.ndarray] = {}

    def load(name):
        if name not in cache:
            path = os.path.join(merged_dir, f"{name}.npy")
            cache[name] = np.load(path, mmap_mode="r", allow_pickle=False) if os.path.exists(path) else None
        return cache[name]

    picks, notes = {}, []
    for g in order:
        chosen = []
        for idx, row in candidates[g].iterrows():
            if len(chosen) >= n:
                break
            label = _label_number(row.get(object_column))
            arr = load(str(row.get(file_column)))
            if arr is None or label is None:
                continue
            res = extract_tile(arr, cell_plane, label, size)
            if res is None:
                continue
            chosen.append((idx, row, label, res))
        if len(chosen) < n:
            notes.append(f"{g}: {len(chosen)} of {n} tiles found")
        picks[g] = chosen
    used_files = list(dict.fromkeys(str(r[1][file_column]) for ch in picks.values() for r in ch))
    rng = np.random.default_rng(seed)
    if len(used_files) > max_range_fields:
        used_files = list(rng.choice(used_files, max_range_fields, replace=False))
    arrays = [load(f) for f in used_files]
    ranges = display_ranges(arrays, list(colour_map), percentiles) if arrays else {}

    outline_cols = {"pathogen": (1.0, 1.0, 1.0), "cell": (0.55, 0.55, 0.55),
                    "nucleus": (0.3, 0.6, 1.0)}
    rows, tiles, files = [], {}, []
    for g in order:
        tiles[g] = []
        for k, (idx, row, label, (raw, box, padded)) in enumerate(picks[g], start=1):
            cell = padded[..., cell_plane] == label
            outlines = []
            for obj in outline_objects:
                if obj not in mask_dims:
                    continue
                if obj == "cell":
                    outlines.append((cell, outline_cols["cell"]))
                    continue
                plane = padded[..., mask_dims[obj]]
                inside = np.unique(plane[cell & (plane > 0)])
                if len(inside):
                    outlines.append((np.isin(plane, inside), outline_cols.get(obj, (1, 1, 0))))
            rgb = render_tile(padded, colour_map, ranges, outlines)
            tiles[g].append(rgb)
            stem = os.path.join(out_folder, "tiles", f"{_safe(g)}_{k:02d}")
            import tifffile
            tifffile.imwrite(f"{stem}_raw.tif", np.ascontiguousarray(np.moveaxis(raw, -1, 0)),
                             photometric="minisblack", metadata={"axes": "CYX"})
            files += [f"{stem}_raw.tif"] + write_rgb16(f"{stem}_rgb16", rgb)
            rec = {group: g, "tile": k, "file_name": row.get(file_column),
                   "object_label": row.get(object_column), "mask_label": label,
                   "y0": box[0], "y1": box[1], "x0": box[2], "x1": box[3],
                   "padded": raw.shape[:2] != (size, size),
                   "raw_tiff": os.path.basename(f"{stem}_raw.tif"),
                   "rgb16_png": os.path.basename(f"{stem}_rgb16.png")}
            if value_column and value_column in row.index:
                rec[value_column] = row[value_column]
            rows.append(rec)
    montage = montage16(tiles, order, n, size)
    files += write_rgb16(os.path.join(out_folder, "montage_rgb16"), montage)
    dtype = str(arrays[0].dtype) if arrays else "unknown"
    return dict(tiles=tiles, table=pd.DataFrame(rows), ranges=ranges, montage=montage,
                files=files, notes=notes, n_range_fields=len(arrays), dtype=dtype)


def montage16(tiles, order, n, size, gap=4):
    """Rows = tile rank, columns = groups; white 16-bit gaps; missing tiles stay black."""
    h = n * size + (n - 1) * gap
    w = len(order) * size + max(len(order) - 1, 0) * gap
    out = np.full((max(h, 1), max(w, 1), 3), 65535, dtype=np.uint16)
    for j, g in enumerate(order):
        for i in range(n):
            y, x = i * (size + gap), j * (size + gap)
            ts = tiles.get(g, [])
            out[y:y + size, x:x + size] = ts[i] if i < len(ts) else 0
    return out


def tile_drawer(tiles_result, chunk, labels, n, size, colour_text, tile_mm):
    """Draw callback and height (mm) of one tile montage row of ``chunk`` conditions."""
    per_line = max(6, int(tile_mm / (7 * 0.62 / 72 * 25.4)))
    texts = ["\n".join(textwrap.wrap(t, per_line)) for t in _label_texts(chunk, labels)]
    n_lines = max((t.count("\n") + 1 for t in texts), default=1)
    rot = 0
    lab_mm = 2 + n_lines * 7 * 1.25 / 72 * 25.4
    height = n * tile_mm + lab_mm + 8 + ID_STRIP_MM
    mont = montage16(tiles_result["tiles"], chunk, n, size)

    def draw(fig, rect):
        ax = fig.add_axes(rect)
        img8 = (mont.astype(np.float64) / 257).round().astype(np.uint8)
        ax.imshow(img8, interpolation="none", aspect="auto")
        ax.set_xticks([(size + 4) * j + size / 2 for j in range(len(chunk))])
        ax.set_xticklabels(texts, fontsize=7, rotation=rot, ha="center")
        ax.xaxis.tick_top()
        ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fw, fh = fig.get_size_inches()
        key_h = 5.0 * MM / fh
        x0, y0, w, h = rect
        fit_axes(fig, ax, (x0, y0 + key_h, w, h - key_h))
        pos = ax.get_position()
        aspect = mont.shape[1] / mont.shape[0]
        w_in, h_in = pos.width * fw, pos.height * fh
        if w_in / h_in > aspect:
            ax.set_position([pos.x0, pos.y0, h_in * aspect / fw, pos.height])
        else:
            nh = w_in / aspect / fh
            ax.set_position([pos.x0, pos.y1 - nh, pos.width, nh])
        pos = ax.get_position()
        wrap = max(40, int(w * fw * 25.4 / (6.5 * 0.5 / 72 * 25.4)))
        fig.text(pos.x0, pos.y0 - 1.0 * MM / fh, "\n".join(textwrap.wrap(colour_text, wrap)),
                 fontsize=6.5, va="top", ha="left")
    return draw, height


def _tile_mm(n_cols, width_mm):
    return min(26.0, (width_mm - 2) / max(n_cols, 1) - 1)


def tile_items(pid, tiles_result, order, labels, n, size, max_per_row, colour_text):
    """Report items for the tile montage, split into condition chunks."""
    _, chunks = condition_chunks(order, 13.0, 0, max_per_row)
    items = []
    for i, chunk in enumerate(chunks):
        draw, height = tile_drawer(tiles_result, chunk, labels, n, size, colour_text,
                                   _tile_mm(len(chunk), _content_width()))
        label = pid if i == 0 else f"{pid} (continued)"
        items.append(ReportItem(pid=pid, label=label, draw=draw, width="full", height_mm=height))
    return items


# =========================================================================== legend
def describe_test(test_row: Mapping) -> str:
    """One sentence: the omnibus test, its result and why it was chosen."""
    t = test_row.get("test")
    if t == "not tested":
        return f"Not tested ({test_row.get('note') or 'too few units'})."
    why = {"one-way ANOVA": "every group passed the Shapiro-Wilk normality test and the Levene "
                            "equal-variance test; post hoc Tukey HSD",
           "Welch ANOVA": "every group passed the Shapiro-Wilk normality test but the Levene "
                          "test showed unequal variances; post hoc Games-Howell",
           "Kruskal-Wallis": "at least one group failed the Shapiro-Wilk normality test; post hoc "
                             "Dunn test with Holm correction"}.get(t, "")
    note = test_row.get("note")
    if isinstance(note, str) and note:
        why = f"{note}; post hoc Dunn test with Holm correction"
    return (f"{t} ({why}, alpha {ALPHA}): {fmt_p(test_row.get('p_value'))} "
            f"({test_row.get('significance')}).")


def _n_text(desc, group, labels):
    parts = [f"{display_label(r[group], labels)}: {int(r['n'])}" for _, r in desc.iterrows()]
    return "; ".join(parts)


def filter_sentences(summary: Optional[pd.DataFrame]) -> str:
    """Text of the filter steps in ``filter_summary`` (objects removed / remaining)."""
    if summary is None or not len(summary):
        return "No opt-in filters were set, so the filtered and unfiltered panels are identical."
    rem = [c for c in summary.columns if c.endswith("_remaining")][0]
    removed = [c for c in summary.columns if c.endswith("_removed")][0]
    first = summary.iloc[0]
    out = [f"{first['step']}: {int(first[rem])} objects."]
    for _, r in summary.iloc[1:].iterrows():
        note = f" ({r['note']})" if isinstance(r.get("note"), str) and r.get("note") else ""
        out.append(f"{r['step']}: {int(r[removed])} removed, {int(r[rem])} remaining{note}.")
    return " ".join(out)


def write_legend(path, paragraphs: Sequence[str]) -> str:
    text = "\n\n".join(p.strip() for p in paragraphs if p and p.strip())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def _spec_legend(pid, spec, what, unit_text):
    group, labels = spec["group"], spec.get("labels")
    if spec["kind"] == "grouped":
        parts = []
        n_texts = {o: _n_text(spec["stats"][o]["descriptive"], group, labels) for o in spec["objects"]}
        same = len(set(n_texts.values())) == 1
        if same:
            parts.append(f"n wells per condition (every object type): {next(iter(n_texts.values()))}.")
        for obj in spec["objects"]:
            st = spec["stats"][obj]
            parts.append(f"{obj}: " + ("" if same else f"n wells per condition {n_texts[obj]}; ")
                         + describe_test(st["test"].iloc[0].to_dict()))
        return (f"{pid}. {what} Bars: mean ± SD over {unit_text}; dots: one per unit; one colour per "
                f"object type. One test per object type across conditions. " + " ".join(parts))
    st = spec["stats"]
    return (f"{pid}. {what} Bars: mean ± SD over {unit_text}; dots: one per unit. n per condition: "
            f"{_n_text(st['descriptive'], group, labels)}. " + describe_test(st["test"].iloc[0].to_dict())
            + f" Significant post hoc pairs are drawn as brackets when there are at most {MAX_BRACKETS} "
              "(* p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001); every pair is in stats.csv.")


# =========================================================================== report builder
def build_assay_report(frames: Mapping[str, pd.DataFrame], out_dir: str, value: str, group: str,
                       order: Sequence, channels: Sequence[int], labels=None,
                       replicate_column=None, value_label=None, ratio=False,
                       filter_summary: Optional[pd.DataFrame] = None, palette=None,
                       tiles: Optional[dict] = None, max_conditions_per_row: int = 12,
                       row_object="cell", title="spaCR assay report", intro: Sequence[str] = (),
                       name="report", count_objects=None, dpi=200) -> dict:
    """Build the standard report: tiles, value panels, control panels, legend.

    :param frames: ``{'unfiltered': df, 'filtered': df}``, one row per
        ``row_object``.
    :param tiles: None to skip, else keyword arguments of :func:`build_tiles`
        (``merged_dir``, ``mask_dims``, ``colour_map``, ``n``, ``size``,
        ``seed``) - tiles come from ``frames['filtered']``.
    :returns: dict with ``pdf``, ``png``, ``legend``, ``panels`` (id -> folder)
        and ``n_pages``.
    """
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    order = list(order)
    filt, unf = frames["filtered"], frames["unfiltered"]
    blocks, legend, folders = [], list(intro), {}
    counter = [0]

    def next_id():
        counter[0] += 1
        return f"R{counter[0]:02d}"

    value_label = value_label or value.replace("_", " ")
    # ---- tiles
    if tiles:
        pid = next_id()
        folder = os.path.join(panels_dir, panel_folder_name(pid, "representative_tiles"))
        try:
            tr = build_tiles(filt, group, order, out_folder=folder, value_column=value,
                             object_column=tiles.get("object_column", "object_label"),
                             **{k: v for k, v in tiles.items() if k in
                                ("merged_dir", "mask_dims", "colour_map", "n", "size", "seed",
                                 "percentiles", "edge_column")})
        except Exception as exc:  # missing arrays must not stop the report
            tr = None
            legend.append(f"{pid}. Representative tiles could not be made: {exc}.")
        if tr is not None and len(tr["table"]):
            cmap = tiles["colour_map"]
            colour_text = ", ".join(f"{_colour_name(c)}: channel {ch}" for ch, c in cmap.items())
            n, size = tiles.get("n", 5), tiles.get("size", 128)
            items = tile_items(pid, tr, order, labels, n, size, max_conditions_per_row,
                               colour_text + "; white outline: pathogen mask, grey: host cell mask")
            blocks.append(ReportBlock("Representative cells (filtered objects)", items))
            ranges = pd.DataFrame([dict(table="display_range", channel=c, colour=_colour_name(cmap[c]),
                                        low=lo, high=hi, percentiles=str(tiles.get("percentiles", (0.5, 99.9))),
                                        n_fields=tr["n_range_fields"]) for c, (lo, hi) in tr["ranges"].items()])
            cand = pd.DataFrame([dict(table="candidates", group=g,
                                      n_filtered_objects=int((filt[group] == g).sum()),
                                      n_tiles=len(tr["tiles"].get(g, []))) for g in order])
            full_w = max(_content_width(), 24.0 * len(order) + 4)
            draw_all, h_all = tile_drawer(tr, order, labels, n, size, colour_text + "; white "
                                          "outline: pathogen mask, grey: host cell mask",
                                          _tile_mm(len(order), full_w))
            write_panel_folder(panels_dir, pid, "representative_tiles", draw_all,
                               (full_w, h_all - ID_STRIP_MM + 4),
                               data=tr["table"], stats=pd.concat([ranges, cand], ignore_index=True),
                               dpi=dpi)
            folders[pid] = folder
            pct = tiles.get("percentiles", (0.5, 99.9))
            rng_txt = "; ".join(f"channel {c}: {lo:.0f}-{hi:.0f}" for c, (lo, hi) in tr["ranges"].items())
            legend.append(
                f"{pid}. Representative host cells, {n} per condition (columns, in the condition order "
                f"of the graphs), chosen at random (seed {tiles.get('seed', 0)}) from the filtered "
                f"objects, preferring cells whose centre lies at least half a tile ({size // 2} px) from "
                f"the field edge. Each crop is {size} x {size} px centred on the centroid of the host cell "
                f"mask (zero padded at the field edge). Composite: {colour_text}. Every channel is scaled "
                f"linearly between its {pct[0]}th and {pct[1]}th percentile over the pixels of "
                f"{tr['n_range_fields']} fields ({rng_txt}); the same range is used for every tile. White "
                f"outline: pathogen mask objects inside the host cell; grey: host cell mask. Raw crops "
                f"(every image channel and mask plane, input dtype {tr.get('dtype', 'uint16')}, values "
                f"unchanged) are in tiles/*_raw.tif; 16-bit RGB composites in tiles/*_rgb16.png/.tif; the "
                f"montage at native resolution in montage_rgb16.png/.tif (16-bit). The figure PDF/PNG is "
                f"8-bit, drawn without interpolation."
                + (" " + "; ".join(tr["notes"]) + "." if tr["notes"] else ""))

    # ---- value panels
    items, sub_legend = [], []
    levels = [("unfiltered", "well"), ("unfiltered", "replicate"),
              ("filtered", "well"), ("filtered", "replicate")]
    skipped_rep = False
    for version, level in levels:
        if level == "replicate" and not replicate_column:
            skipped_rep = True
            continue
        df = frames[version]
        df = df[df[group].isin(order)]
        unit = aggregate_to_unit(df, value, [group],
                                 replicate_column=replicate_column if level == "replicate" else None)
        fin = pd.to_numeric(unit[value], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ymin = 1 if ratio and len(fin) and fin.min() >= 1 else None
        spec = bar_spec(unit, value, group, order, title=f"{value_label}: {version}, {level} level",
                        ylabel=value_label, labels=labels, ymin=ymin, refline=1 if ratio else None,
                        palette=palette)
        pid = next_id()
        folders[pid] = save_spec_panel(panels_dir, pid, f"{value}_{version}_{level}", spec, dpi=dpi)
        items += spec_items(pid, spec, max_conditions_per_row)
        unit_text = (f"replicates (column '{replicate_column}'; each the mean of its well means)"
                     if level == "replicate" else "wells (each the mean of its objects)")
        sub_legend.append(_spec_legend(pid, spec, f"{value_label} per condition, {version} objects, "
                                                  f"{level} level.", unit_text))
    blocks.append(ReportBlock(f"{value_label} per condition", items))
    if skipped_rep:
        sub_legend.append("Replicate-level panels are not drawn because replicate_column is not set.")
    legend += sub_legend

    # ---- control panels, before and after filtering
    for version in ("unfiltered", "filtered"):
        df = frames[version]
        df = df[df[group].isin(order)]
        items = []
        objs_all = object_types(df, channels)
        for chan in channels:
            objs = object_types(df, [chan])
            if not objs:
                continue
            data = well_intensity_long(df, group, chan, objs)
            spec = grouped_spec(data, group, order, objs,
                                title=f"channel {chan} in all objects ({version})",
                                ylabel=f"mean intensity, channel {chan}", labels=labels)
            pid = next_id()
            folders[pid] = save_spec_panel(panels_dir, pid, f"channel_{chan}_all_objects_{version}",
                                           spec, dpi=dpi)
            items += spec_items(pid, spec, max_conditions_per_row)
            legend.append(_spec_legend(pid, spec, f"Mean intensity of channel {chan} per well in every "
                                                  f"object type ({', '.join(objs)}), {version} objects; "
                                                  f"a log axis is used when object means differ more "
                                                  f"than 30-fold.", "wells"))
        counts = [o for o in (count_objects or objs_all) if o != "cytoplasm"]
        for obj in counts:
            data = well_counts(df, group, obj, row_object)
            if not len(data):
                continue
            spec = bar_spec(data, "objects_per_well", group, order,
                            title=f"{obj} objects per well ({version})",
                            ylabel=f"{obj} objects per well", labels=labels, palette=palette)
            pid = next_id()
            folders[pid] = save_spec_panel(panels_dir, pid, f"{obj}_objects_per_well_{version}",
                                           spec, dpi=dpi)
            items += spec_items(pid, spec, max_conditions_per_row)
            src = data["count_source"].iloc[0]
            how = ("rows of the table (one per host cell)" if src == "rows" else
                   f"the sum of {src} (objects of this type merged into each host cell)"
                   if src.endswith("_prcfo_count") else f"host cells with a {src} value")
            legend.append(_spec_legend(pid, spec, f"{obj} objects per well, {version} objects, counted as "
                                                  f"{how}.", "wells"))
        title_v = "before filtering (unfiltered)" if version == "unfiltered" else "after filtering"
        blocks.append(ReportBlock(f"Well-level controls {title_v}", items))

    legend.append("Filters. Unfiltered: spaCR output after the standard object filters. Filtered: "
                  "after the opt-in filters. " + filter_sentences(filter_summary))
    legend.append("Statistics. The experimental unit is the well (mean of its objects) or the "
                  "replicate (mean of its well means); tests never treat single cells as independent. "
                  "Per panel: Shapiro-Wilk normality per group and Levene equal variance (median "
                  "centred); all normal and equal variance: one-way ANOVA + Tukey HSD; normal with "
                  "unequal variance: Welch ANOVA + Games-Howell; otherwise Kruskal-Wallis + Dunn test "
                  "(Holm). Each panel folder holds data.csv (exactly the plotted values) and stats.csv "
                  "(test, assumptions, post hoc and descriptive tables in long format).")
    legend_path = os.path.join(out_dir, "figure_legend.txt")
    legend_text = write_legend(legend_path, [f"Figure legend: {title}"] + legend)
    pages = paginate(blocks)
    res = render_report(pages, os.path.join(out_dir, f"{name}.pdf"), legend_text, title=title, dpi=dpi)
    res.update(legend=legend_path, panels=folders)
    return res


def _colour_name(c):
    if isinstance(c, str):
        return c
    for k, v in COLOURS.items():
        if tuple(v) == tuple(float(x) for x in c):
            return k
    return str(tuple(c))
