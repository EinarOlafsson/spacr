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
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .assay_panels import (ALPHA, STYLE, WELL_KEYS, _DEFAULT_COLORS, _despine, _set_xticks, shade,
                           style_axes,
                           aggregate_to_unit, bar_spec, display_label, draw_bar_panel, fmt_p,
                           panel_folder_name, select_and_run_test, stars)

__all__ = [
    "A4_MM", "MIN_FONT_PT", "ReportItem", "ReportBlock", "paginate", "render_report",
    "condition_chunks", "object_types", "well_intensity_long", "well_counts",
    "grouped_spec", "draw_grouped_panel", "stats_long", "write_panel_folder", "Layout",
    "COMPACT", "SPACIOUS", "get_layout", "well_qc_table", "object_qc_mask", "plate_heatmap_items", "compact_style", "db_object_tables", "db_well_counts",
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
# fixed object-type colours (Okabe-Ito), so a type has the same colour in every panel
OBJECT_COLOURS = {"cell": "#0072B2", "nucleus": "#CC79A7", "cytoplasm": "#009E73",
                  "pathogen": "#E69F00"}
_EXTRA_OBJECT_COLOURS = ["#D55E00", "#56B4E9", "#F0E442", "#999999"]


def object_colours(objects):
    """Colour per object type: fixed for cell/nucleus/cytoplasm/pathogen, others in name order."""
    extra = sorted(o for o in objects if o not in OBJECT_COLOURS)
    out = {}
    for o in objects:
        out[o] = OBJECT_COLOURS.get(o) or _EXTRA_OBJECT_COLOURS[extra.index(o) % len(_EXTRA_OBJECT_COLOURS)]
    return out
_MEAN_RE = re.compile(r"^(?P<obj>.+?)_channel_(?P<chan>\d+)_mean_intensity$")
COLOURS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "magenta": (1, 0, 1),
           "cyan": (0, 1, 1), "yellow": (1, 1, 0), "grey": (1, 1, 1), "gray": (1, 1, 1),
           "white": (1, 1, 1)}


# =========================================================================== layout
@dataclass
class Layout:
    """Page geometry and type sizes of a report layout (all lengths in mm, fonts in pt).

    ``compact`` draws 5-6 pt text, slim bars and small markers and first tries
    to fit the whole figure on one page (:func:`paginate`); ``spacious`` is
    the roomy layout with 8-10 pt text that paginates freely.
    """
    name: str = "spacious"
    margin: float = 12.0
    gap: float = 5.0
    heading: float = 7.0
    heading_pt: float = 10.0
    id_strip: float = 4.5
    id_pt: float = 9.0
    compact: bool = False
    fit_one_page: bool = False
    title_pt: float = 9.0
    text_pt: float = 8.0
    plot_mm: float = 40.0
    min_plot_mm: float = 40.0
    tile_mm: float = 26.0
    min_tile_mm: float = 26.0
    tile_label_pt: float = 7.0
    widths: tuple = (0.5, 1.0)
    page: tuple = A4_MM

    @property
    def content(self):
        return self.page[0] - 2 * self.margin

    def width_mm(self, frac):
        return frac * (self.content + self.gap) - self.gap

    @property
    def usable(self):
        return self.page[1] - 2 * self.margin


SPACIOUS = Layout()
COMPACT = Layout(name="compact", margin=8.0, gap=2.5, heading=4.2, heading_pt=7.0, id_strip=3.0,
                 id_pt=6.5, compact=True, fit_one_page=True, title_pt=6.0, text_pt=5.0,
                 plot_mm=24.0, min_plot_mm=11.0, tile_mm=9.0, min_tile_mm=6.5, tile_label_pt=5.5,
                 widths=(0.2, 0.25, 1 / 3, 0.5, 1.0))
LAYOUTS = {"compact": COMPACT, "spacious": SPACIOUS}


def get_layout(layout) -> Layout:
    if isinstance(layout, Layout):
        return layout
    if layout not in LAYOUTS:
        raise ValueError(f"report_layout must be one of {sorted(LAYOUTS)}, got {layout!r}")
    return LAYOUTS[layout]


_WIDTH_NAMES = {"full": 1.0, "half": 0.5, "third": 1 / 3, "quarter": 0.25}


@dataclass
class ReportItem:
    """One panel (or one chunk of a panel) on a page.

    :param draw: ``draw(fig, rect)`` with ``rect = (x0, y0, w, h)`` in figure
        fractions; it adds its own axes inside the rect.
    :param width: fraction of the content width (or ``'half'``, ``'full'``,
        ``'third'``, ``'quarter'``).
    :param height_mm: height reserved on the page, panel ID strip included.
    :param min_height_mm: smallest height the one-page fit may shrink it to.
    :param new_row: start a new row with this item.
    """
    pid: str
    label: str
    draw: Callable
    width: object = 0.5
    height_mm: float = 60.0
    min_height_mm: Optional[float] = None
    new_row: bool = False

    @property
    def frac(self):
        return _WIDTH_NAMES.get(self.width, self.width) if isinstance(self.width, str) else float(self.width)


@dataclass
class ReportBlock:
    """A heading plus panel items. Consecutive blocks with the same ``group``
    are kept on one page when the compact layout has to paginate."""
    title: str
    items: List[ReportItem] = field(default_factory=list)
    group: Optional[str] = None


def _content_width(page=A4_MM, margin=MARGIN_MM):
    return page[0] - 2 * margin


def _half_width(page=A4_MM, margin=MARGIN_MM, gap=GAP_MM):
    return (_content_width(page, margin) - gap) / 2


def _rows(items):
    """Pack items into rows while their width fractions sum to at most 1."""
    rows, cur, used = [], [], 0.0
    for it in items:
        if cur and (it.new_row or used + it.frac > 1 + 1e-6):
            rows.append(cur)
            cur, used = [], 0.0
        cur.append(it)
        used += it.frac
    if cur:
        rows.append(cur)
    return rows


def _one_page_height(block_rows, layout):
    total = 0.0
    for rows in block_rows:
        total += layout.heading + sum(max(it.height_mm for it in r) + layout.gap for r in rows) + layout.gap
    return total - 2 * layout.gap if block_rows else 0.0


def _fit_one_page(block_rows, layout):
    """Shrink item heights (not below their minimum) so every block fits one page.

    :returns: the new block rows, or None if the figure cannot fit one page.
    """
    need = _one_page_height(block_rows, layout) - layout.usable
    if need <= 0:
        return block_rows
    slack = sum(max(it.height_mm for it in r) - max(it.min_height_mm or it.height_mm for it in r)
                for rows in block_rows for r in rows)
    if slack < need:
        return None
    s = need / slack
    out = []
    for rows in block_rows:
        new_rows = []
        for r in rows:
            h = max(it.height_mm for it in r)
            lo = max(it.min_height_mm or it.height_mm for it in r)
            nh = h - s * (h - lo)
            new_rows.append([replace(it, height_mm=nh) for it in r])
        out.append(new_rows)
    return out


def _place(block_rows, layout, flow):
    """Placements of (block, rows) pairs; ``flow`` breaks pages, else everything goes on one page."""
    margin, gap, heading, usable, full = (layout.margin, layout.gap, layout.heading,
                                          layout.usable, layout.content)
    pages: List[List[dict]] = [[]]
    y = 0.0

    def new_page():
        nonlocal y
        pages.append([])
        y = 0.0

    for block, rows in block_rows:
        heights = [max(it.height_mm for it in r) for r in rows]
        total = heading + sum(heights) + gap * (len(rows) - 1)
        remaining = usable - y
        if flow and total > remaining and y > 0 and (
                remaining < heading + heights[0] or (total <= usable and remaining < 0.6 * total)):
            new_page()
        pages[-1].append(dict(kind="heading", text=block.title,
                              rect_mm=(margin, margin + y, full, heading)))
        y += heading
        on_page = 0
        for row, h in zip(rows, heights):
            if flow and y + h > usable + 1e-6 and (on_page or y > heading + 1e-9):
                new_page()
                on_page = 0
                pages[-1].append(dict(kind="heading", text=f"{block.title} (continued)",
                                      rect_mm=(margin, margin + y, full, heading)))
                y += heading
            x = margin
            for it in row:
                w = layout.width_mm(it.frac)
                pages[-1].append(dict(kind="item", item=it, rect_mm=(x, margin + y, w, h)))
                x += w + gap
            y += h + gap
            on_page += 1
        y += gap
    return [p for p in pages if p]


def _shrunk(block_rows, layout):
    fitted = _fit_one_page([r for _, r in block_rows], layout)
    return None if fitted is None else [(b, r) for (b, _), r in zip(block_rows, fitted)]


def paginate(blocks: Sequence[ReportBlock], layout=SPACIOUS, page=None, margin=None, gap=None,
             heading=None) -> List[List[dict]]:
    """Place blocks on A4 pages.

    Spacious: blocks flow over pages; a block that does not fit the rest of a
    page starts on a new page (unless most of it fits), a block taller than
    a page flows on row by row with its heading repeated.

    Compact (``fit_one_page``): the whole figure is first fitted on one page
    by shrinking panel heights towards their minimum. If that fails, whole
    block groups (consecutive blocks sharing ``group``, e.g. the control
    blocks before and after filtering) are packed page by page, each page
    shrunk to fit; a group that cannot fit one page on its own flows row by
    row as in the spacious layout.

    :returns: one list per page of placements, each a dict with ``kind``
        (``'heading'`` or ``'item'``), ``text`` / ``item`` and ``rect_mm``
        ``(x0, y_top, w, h)`` measured from the top-left page corner.
    """
    layout = get_layout(layout)
    if any(v is not None for v in (page, margin, gap, heading)):
        layout = replace(layout, **{k: v for k, v in dict(page=page, margin=margin, gap=gap,
                                                         heading=heading).items() if v is not None})
    block_rows = [(b, _rows(b.items)) for b in blocks]
    block_rows = [(b, r) for b, r in block_rows if r]
    if not layout.fit_one_page:
        return _place(block_rows, layout, flow=True)
    whole = _shrunk(block_rows, layout)
    if whole is not None:
        return _place(whole, layout, flow=False)
    units = []
    for br in block_rows:
        g = br[0].group
        if units and g is not None and units[-1][0] == g:
            units[-1][1].append(br)
        else:
            units.append((g, [br]))
    pages, cur = [], []
    for _, unit in units:
        if _shrunk(cur + unit, layout) is not None:
            cur = cur + unit
            continue
        if cur:
            pages += _place(_shrunk(cur, layout), layout, flow=False)
            cur = []
        if _shrunk(unit, layout) is not None:
            cur = list(unit)
        else:
            pages += _place(unit, layout, flow=True)
    if cur:
        pages += _place(_shrunk(cur, layout), layout, flow=False)
    return pages


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


def _draw_item(fig, item, rect_mm, layout):
    page_w, page_h = layout.page
    x, y_top, w, h = rect_mm
    fx, fw = x / page_w, w / page_w
    fy = (page_h - y_top - h) / page_h
    fh = (h - layout.id_strip) / page_h
    fig.text(fx, (page_h - y_top) / page_h, item.label, fontsize=layout.id_pt,
             fontweight="bold", va="top", ha="left")
    item.draw(fig, (fx, fy, fw, fh))


def render_report(pages: List[List[dict]], pdf_path: str, legend_text: str = "",
                  title: str = "", layout=SPACIOUS, dpi: int = 200, png=True, page=None) -> dict:
    """Draw paginated placements, then the legend on pages of its own, into one PDF.

    :returns: ``{'pdf', 'png': [paths], 'n_pages', 'n_figure_pages', 'n_legend_pages'}``.
        PNG files are ``<stem>.png`` for page 1 and ``<stem>_page_NN.png``
        for the next pages.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    layout = get_layout(layout)
    page = layout.page if page is None else page
    stem = os.path.splitext(pdf_path)[0]
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    legend_pages = _legend_pages(legend_text, page) if legend_text else []
    n_total = len(pages) + len(legend_pages)
    pngs = []
    with PdfPages(pdf_path) as pdf:
        for k, placements in enumerate(pages, start=1):
            fig = _new_figure(*page)
            fig.set_facecolor("white")
            _page_header(fig, title, k, n_total, page, layout)
            for pl in placements:
                if pl["kind"] == "heading":
                    x, y_top, w, h = pl["rect_mm"]
                    fig.text(x / page[0], (page[1] - y_top - 0.5) / page[1], pl["text"],
                             fontsize=layout.heading_pt, fontweight="bold", va="top", ha="left")
                else:
                    _draw_item(fig, pl["item"], pl["rect_mm"], layout)
            pdf.savefig(fig)
            if png:
                path = f"{stem}.png" if k == 1 else f"{stem}_page_{k:02d}.png"
                fig.savefig(path, dpi=dpi)
                pngs.append(path)
        for j, lines in enumerate(legend_pages, start=len(pages) + 1):
            fig = _new_figure(*page)
            _page_header(fig, title, j, n_total, page, SPACIOUS)
            fig.text(MARGIN_MM / page[0], 1 - (MARGIN_MM + 2) / page[1], "\n".join(lines),
                     fontsize=_LEGEND_PT, va="top", ha="left", family="DejaVu Sans",
                     linespacing=1.3)
            pdf.savefig(fig)
            if png:
                path = f"{stem}_page_{j:02d}.png"
                fig.savefig(path, dpi=dpi)
                pngs.append(path)
    return dict(pdf=pdf_path, png=pngs, n_pages=n_total, n_figure_pages=len(pages),
                n_legend_pages=len(legend_pages))


def _page_header(fig, title, k, n, page, layout=SPACIOUS):
    y = 1 - min(5.0, layout.margin * 0.45) / page[1]
    size = max(MIN_FONT_PT, min(7.0, layout.heading_pt))
    fig.text(layout.margin / page[0], y, title, fontsize=size, va="top", ha="left", color="#555555")
    fig.text(1 - layout.margin / page[0], y, f"page {k} of {n}", fontsize=size,
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
def _pt_mm(pt):
    return pt / 72 * 25.4


def _label_texts(order, labels):
    return [display_label(g, labels) for g in order]


def _rotation(texts):
    longest = max((len(t) for t in texts), default=0)
    return 0 if longest * len(texts) <= 24 and longest <= 10 else 35


_COMPACT_WRAP = 14


def _xlabel_mm(texts, fontsize=8, compact=False):
    """Height taken by the x tick labels (spacious: assay_panels._set_xticks; compact: wrapped, 45 deg)."""
    char_mm = _pt_mm(fontsize) * 0.55
    line_mm = _pt_mm(fontsize) * 1.25
    if compact:
        wrapped = [textwrap.wrap(t, _COMPACT_WRAP, break_long_words=False) or [""] for t in texts]
        longest = max((len(line) for w in wrapped for line in w), default=0)
        n_lines = max((len(w) for w in wrapped), default=1)
        return (longest * char_mm + n_lines * line_mm) * math.sin(math.radians(45)) + 1.5
    if _rotation(texts) == 0:
        return line_mm + 2
    longest = max((len(t) for t in texts), default=0)
    return longest * char_mm * math.sin(math.radians(35)) + line_mm + 2


def condition_chunks(order, per_category_mm, fixed_mm, max_per_row, page=A4_MM, layout=None,
                     min_frac=0.0):
    """Split the condition order into chunks that fit one page width.

    :returns: (width, chunks). Without ``layout`` width is ``'half'`` or
        ``'full'`` (spacious); with a layout it is the smallest width fraction
        of ``layout.widths`` (at least ``min_frac``) that holds the panel.
    """
    order = list(order)
    n = max(len(order), 1)
    if layout is None:
        if fixed_mm + n * per_category_mm <= _half_width(page) and n <= max(1, max_per_row // 2):
            return "half", [order]
        content = _content_width(page)
    else:
        need = fixed_mm + n * per_category_mm
        if n <= max_per_row:
            for frac in layout.widths:
                if frac >= min_frac - 1e-9 and layout.width_mm(frac) >= need and \
                        (frac == 1.0 or layout.compact or n <= max(1, max_per_row // 2)):
                    return frac, [order]
        content = layout.content
    fit = int((content - fixed_mm) // per_category_mm)
    size = max(1, min(int(max_per_row), fit))
    full = "full" if layout is None else 1.0
    if n <= size:
        return full, [order]
    k = int(math.ceil(n / size))
    size = int(math.ceil(n / k))  # balance the chunks
    return full, [order[i:i + size] for i in range(0, n, size)]


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


def well_column_long(df, group, columns: Mapping[str, str], value_name, well_keys=WELL_KEYS):
    """Well means of one column per object type in long format ({object type: column})."""
    frames = []
    for obj, col in columns.items():
        if col not in df.columns:
            continue
        unit = aggregate_to_unit(df, col, [group], well_keys=well_keys).rename(columns={col: value_name})
        unit.insert(len(well_keys) + 1, "object_type", obj)
        unit.insert(len(well_keys) + 2, "column", col)
        frames.append(unit)
    if not frames:
        return pd.DataFrame(columns=[group, *well_keys, "object_type", "column", value_name, "n_objects"])
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


_NON_OBJECT_TABLES = {"png_list", "intensity_rescale", "object_relationships", "image", "run_status",
                      "settings", "settings_history", "object_counts", "pivoted_counts"}


def db_object_tables(db_path, skip=("cytoplasm",)):
    """Object tables of a measurements.db: tables with plateID, rowID and columnID.

    Bookkeeping tables (png_list, intensity_rescale, ...), versioned copies
    (``cell_1`` when ``cell`` exists) and ``skip`` (cytoplasm, one per cell)
    are left out. cell, nucleus and pathogen come first.
    """
    import sqlite3
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        names = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        out = []
        for t in names:
            base = re.sub(r"_\d+$", "", t)
            if base in _NON_OBJECT_TABLES or t.startswith("_") or t in skip or base in skip:
                continue
            if base != t and base in names:
                continue
            cols = {r[1] for r in con.execute(f'pragma table_info("{t}")')}
            if set(WELL_KEYS) <= cols:
                out.append(t)
    core = [o for o in _CORE_OBJECTS if o in out]
    return core + sorted(set(out) - set(core))


def db_well_counts(db_path, frame, group, tables=None, well_keys=WELL_KEYS):
    """Rows per well in every object table of measurements.db, for the wells of ``frame``.

    :returns: long table with ``group``, well keys, ``object_type``,
        ``objects_per_well`` and ``count_source`` (``measurements.db table <t>``).
    """
    import sqlite3
    from urllib.parse import unquote
    tables = list(tables) if tables is not None else db_object_tables(db_path)
    keys = list(well_keys)
    wells = frame[keys + [group]].drop_duplicates(keys).copy()
    wells["_plate"] = wells["plateID"].astype(str).map(lambda v: unquote(unquote(v)))
    for k in keys[1:]:
        wells[k] = wells[k].astype(str)
    frames = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        for t in tables:
            q = (f'select plateID, rowID, columnID, count(*) as n from "{t}" '
                 f'group by plateID, rowID, columnID')
            c = pd.read_sql_query(q, con)
            c["_plate"] = c["plateID"].astype(str).map(lambda v: unquote(unquote(v)))
            for k in keys[1:]:
                c[k] = c[k].astype(str)
            m = wells.merge(c.drop(columns=["plateID"]), on=["_plate"] + keys[1:], how="left")
            m["n"] = m["n"].fillna(0).astype(int)
            m = m.drop(columns=["_plate"]).rename(columns={"n": "objects_per_well"})
            m.insert(len(keys) + 1, "object_type", t)
            m["count_source"] = f"measurements.db table {t}"
            frames.append(m[[group, *keys, "object_type", "objects_per_well", "count_source"]])
    if not frames:
        return pd.DataFrame(columns=[group, *keys, "object_type", "objects_per_well", "count_source"])
    return pd.concat(frames, ignore_index=True)


# =========================================================================== QC filters
QC_MEASURES = ("count", "area", "intensity")
_MAD_K = 1.4826


def _object_count_column(df, obj, row_object):
    if obj == row_object:
        return None
    for c in (f"{obj}_prcfo_count", f"{obj}_area"):
        if c in df.columns:
            return c
    return False


def qc_object_types(df):
    """Object types with an area or a channel mean-intensity column."""
    types = object_types(df)
    extra = sorted({str(c)[:-5] for c in df.columns if str(c).endswith("_area")
                    and f"{str(c)[:-5]}_perimeter" in df.columns} - set(types))
    return types + extra


def well_qc_table(df, channels=None, measures=None, mad=3.5, row_object="cell",
                  well_keys=WELL_KEYS):
    """Well-level QC: every well against the other wells of its plate.

    Per well, for every object type: object count (rows for ``row_object``,
    else the sum of ``<obj>_prcfo_count``), median ``<obj>_area`` and median
    ``<obj>_channel_<c>_mean_intensity`` per channel. A well fails when any
    measure lies more than ``mad`` x 1.4826 x MAD from the plate median of
    the wells (measures with MAD 0 flag nothing).

    :param measures: subset of ``('count', 'area', 'intensity')``; None = all.
    :returns: one row per well: well keys, ``<measure>`` and ``<measure>_z``
        columns, ``well_qc_pass`` and ``well_qc_reason`` (measures that failed).
    """
    measures = set(measures or QC_MEASURES)
    keys = [k for k in well_keys if k in df.columns]
    groups = [df[k] for k in keys]
    cols = {}
    types = qc_object_types(df)
    for obj in types:
        if "count" in measures:
            src = _object_count_column(df, obj, row_object)
            if src is None:
                cols[f"{obj}_count"] = df.groupby(keys, observed=True).size()
            elif src is not False and src.endswith("_prcfo_count"):
                cols[f"{obj}_count"] = pd.to_numeric(df[src], errors="coerce").fillna(0).groupby(groups).sum()
        if "area" in measures and f"{obj}_area" in df.columns:
            cols[f"{obj}_area_median"] = pd.to_numeric(df[f"{obj}_area"], errors="coerce").groupby(groups).median()
        if "intensity" in measures:
            for c in (channels if channels is not None else sorted(
                    {int(m.group("chan")) for m in map(_MEAN_RE.match, map(str, df.columns))
                     if m and m.group("obj") == obj})):
                col = f"{obj}_channel_{int(c)}_mean_intensity"
                if col in df.columns:
                    cols[f"{obj}_ch{int(c)}_intensity_median"] = (
                        pd.to_numeric(df[col], errors="coerce").groupby(groups).median())
    if not cols:
        t = df[keys].drop_duplicates().reset_index(drop=True)
        t["well_qc_pass"], t["well_qc_reason"] = True, ""
        return t
    table = pd.DataFrame(cols).reset_index()
    if "plateID" in keys:
        plate = table["plateID"]
    else:
        plate = pd.Series("plate", index=table.index)
    failed = pd.Series(False, index=table.index)
    reason = pd.Series([""] * len(table), index=table.index, dtype=object)
    for m in list(cols):
        v = pd.to_numeric(table[m], errors="coerce")
        med = v.groupby(plate).transform("median")
        dev = (v - med).abs().groupby(plate).transform("median")
        z = (v - med) / (_MAD_K * dev.replace(0, np.nan))
        table[f"{m}_z"] = z
        hit = (z.abs() > mad).fillna(False)
        failed |= hit
        reason[hit] = reason[hit] + m + " "
    table["well_qc_pass"] = ~failed
    table["well_qc_reason"] = reason.str.strip()
    return table


def object_qc_mask(df, channels=None, measures=None, mad=3.5, plate_column="plateID"):
    """Object-level QC: per plate and object type, objects beyond the robust fence.

    Area and channel mean intensities are compared on log scale (values <= 0
    are never flagged).

    :returns: (keep mask, summary with ``measure``, ``column``,
        ``objects_removed`` per measure - an object can fail several).
    """
    measures = set(measures or QC_MEASURES)
    plate = df[plate_column] if plate_column in df.columns else pd.Series("plate", index=df.index)
    keep = np.ones(len(df), bool)
    rows = []
    for obj in qc_object_types(df):
        cols = []
        if "area" in measures and f"{obj}_area" in df.columns:
            cols.append(("area", f"{obj}_area"))
        if "intensity" in measures:
            chans = channels if channels is not None else sorted(
                {int(m.group("chan")) for m in map(_MEAN_RE.match, map(str, df.columns))
                 if m and m.group("obj") == obj})
            cols += [(f"ch{int(c)} intensity", f"{obj}_channel_{int(c)}_mean_intensity")
                     for c in chans if f"{obj}_channel_{int(c)}_mean_intensity" in df.columns]
        for measure, col in cols:
            v = pd.to_numeric(df[col], errors="coerce")
            lv = np.log(v.where(v > 0))
            med = lv.groupby(plate).transform("median")
            dev = (lv - med).abs().groupby(plate).transform("median")
            z = (lv - med) / (_MAD_K * dev.replace(0, np.nan))
            hit = (z.abs() > mad).fillna(False).to_numpy()
            rows.append(dict(object_type=obj, measure=measure, column=col,
                             objects_removed=int(hit.sum()), objects=int(len(df))))
            keep &= ~hit
    return keep, pd.DataFrame(rows, columns=["object_type", "measure", "column", "objects_removed",
                                             "objects"])


# =========================================================================== grouped panel
def grouped_spec(data, group, order, objects, value="mean_intensity", title="", ylabel=None,
                 labels=None, colors=None, log_ratio=30.0) -> dict:
    """Grouped bars (one colour per object type) with one test per object type.

    Outlier points are marked per object type and group with the active
    :class:`spacr.assay_panels.outlier_policy` (``outlier`` column) and left
    out of the drawing and the tests.
    """
    from .assay_panels import _OUTLIER_POLICY, mark_outliers, outlier_table, parse_outlier_method
    method = _OUTLIER_POLICY["method"]
    if parse_outlier_method(method) is not None and len(data):
        data = data.copy()
        data["outlier"] = mark_outliers(data, value, ["object_type", group], method).to_numpy()
    stats = {}
    for obj in objects:
        sub = data[data["object_type"] == obj]
        used = sub[~sub["outlier"]] if "outlier" in sub.columns else sub
        stats[obj] = select_and_run_test(used, value, group, order)
        if "outlier" in sub.columns:
            stats[obj]["outliers"] = outlier_table(sub, group, order, method)
    colors = colors or object_colours(objects)
    return dict(kind="grouped", data=data, value=value, group=group, order=list(order),
                objects=list(objects), title=title, ylabel=ylabel or value.replace("_", " "),
                labels=labels, colors=colors, stats=stats, log_ratio=log_ratio)


def draw_grouped_panel(ax, spec, order=None, seed=0, compact=False):
    """Draw a :func:`grouped_spec`: mean ± SD of unit values per object type, dots per unit."""
    order = list(order if order is not None else spec["order"])
    objects, group, value = spec["objects"], spec["group"], spec["value"]
    data = spec["data"]
    if "outlier" in data.columns:
        data = data[~data["outlier"].astype(bool)]
    rng = np.random.default_rng(seed)
    k = max(len(objects), 1)
    width = 0.8 / k
    vals = pd.to_numeric(data[value], errors="coerce") if len(data) else pd.Series(dtype=float)
    d = data.assign(_v=vals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_v"])
    means_all = d.groupby("object_type")["_v"].mean()
    pos = means_all[means_all > 0]
    log = bool(len(pos) >= 2 and (d["_v"] > 0).all()
               and pos.max() / pos.min() > spec.get("log_ratio", 30.0))
    x = np.arange(len(order))
    for j, obj in enumerate(objects):
        sub = d[d["object_type"] == obj]
        m = sub.groupby(group)["_v"].mean().reindex(order)
        s = sub.groupby(group)["_v"].std().reindex(order).fillna(0)
        off = (j - (k - 1) / 2) * width
        st = spec["stats"][obj]["test"].iloc[0]
        tag = ("n/a" if compact else "not tested") if st["test"] == "not tested" else st["significance"]
        from matplotlib.colors import to_rgba
        col = spec["colors"][obj]
        ax.bar(x + off, m.to_numpy(float), width=width * 0.92, color=to_rgba(col, STYLE["bar_alpha"]),
               edgecolor=shade(col, STYLE["edge_shade"]), linewidth=STYLE["edge_lw"] * 0.6,
               label=f"{obj} ({tag})")
        ax.errorbar(x + off, m.to_numpy(float), yerr=s.to_numpy(float), fmt="none",
                    ecolor=STYLE["err_color"], elinewidth=STYLE["err_lw"] * 0.75,
                    capsize=STYLE["capsize"] * 0.75, zorder=4)
        for i, g in enumerate(order):
            y = sub.loc[sub[group] == g, "_v"].to_numpy(float)
            if len(y):
                ax.scatter(i + off + rng.uniform(-width * 0.22, width * 0.22, len(y)), y,
                           s=STYLE["dot_size"] * 0.5, color=shade(col, STYLE["dot_shade"]),
                           alpha=STYLE["dot_alpha"], edgecolors=STYLE["dot_edge"],
                           linewidths=STYLE["dot_edge_lw"], zorder=3)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    title = spec["title"] + (" (log)" if log else "")
    ax.set_ylabel(spec["ylabel"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    _set_xticks(ax, order, spec.get("labels"), None)
    if compact:
        ncol = min(k, 2)
        leg = ax.legend(fontsize=COMPACT.text_pt, frameon=False, loc="lower left",
                        bbox_to_anchor=(0.0, 1.0), ncol=ncol, handlelength=0.8, handleheight=0.6,
                        handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1, borderpad=0.1,
                        labelspacing=0.15)
        leg.set_in_layout(False)   # a wide legend must not squeeze the bars
        rows = int(math.ceil(k / ncol))
        ax.set_title(title, fontsize=COMPACT.title_pt, pad=2 + rows * COMPACT.text_pt * 1.25)
    else:
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                  title="object type (test\nacross conditions)", title_fontsize=6)
    style_axes(ax)
    return ax


COMPACT_DOT_SIZE = 13.0   # pt^2


def compact_style(ax, layout=COMPACT, wrap=_COMPACT_WRAP):
    """Restyle a drawn panel for the compact layout: 5-6 pt text, slim bars, small dots.

    x tick labels are wrapped at word boundaries and drawn at 45 degrees.
    """
    from matplotlib.collections import PathCollection, LineCollection
    from matplotlib.patches import Rectangle
    txt = max(MIN_FONT_PT, layout.text_pt)
    ax.title.set_fontsize(max(MIN_FONT_PT, layout.title_pt))
    ax.title.set_text("\n".join("not tested" if line.startswith("not tested") else line
                                for line in ax.title.get_text().split("\n")))
    ax.xaxis.label.set_fontsize(txt)
    ax.yaxis.label.set_fontsize(txt)
    ax.yaxis.labelpad = 1.5
    ax.tick_params(labelsize=txt, length=1.5, width=0.4, pad=1)
    for t in ax.texts:
        t.set_fontsize(max(MIN_FONT_PT, min(t.get_fontsize(), layout.title_pt)))
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(txt)
    for c in ax.collections:
        if isinstance(c, PathCollection):     # jitter dots: about 3.6 pt across
            c.set_sizes(np.full(max(len(c.get_offsets()), 1), COMPACT_DOT_SIZE))
            c.set_linewidths(0.25)
        elif isinstance(c, LineCollection):
            c.set_linewidth(0.45)
    ax.yaxis.grid(True, alpha=STYLE["grid_alpha"], linewidth=0.35)
    for line in ax.lines:
        line.set_linewidth(min(line.get_linewidth(), 0.5))
        if line.get_marker() not in (None, "None", ""):
            line.set_markersize(min(line.get_markersize(), 1.8))
            line.set_markeredgewidth(0.45)
    for p in ax.patches:
        if isinstance(p, Rectangle):
            w = p.get_width()
            p.set_x(p.get_x() + w * 0.15)
            p.set_width(w * 0.7)
            p.set_linewidth(0.45)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ticks = ax.get_xticks()
    texts = [t.get_text() for t in ax.get_xticklabels()]
    if len(texts) == len(ticks) and any(texts):
        wrapped = ["\n".join(textwrap.wrap(t, wrap, break_long_words=False)) for t in texts]
        n_lines = max(t.count("\n") + 1 for t in wrapped)
        # 45 degrees while the labels clear each other, else vertical
        fig = ax.figure
        width_mm = ax.get_position().width * fig.get_size_inches()[0] * 25.4
        per_cat = width_mm / max(len(ticks), 1)
        need = n_lines * _pt_mm(txt) * 1.25 / math.sin(math.radians(45)) + 1.5
        rot = 45 if per_cat >= need else 90
        ax.set_xticks(ticks)
        ax.set_xticklabels(wrapped, rotation=rot, ha="right" if rot == 45 else "center",
                           va="top", rotation_mode="anchor" if rot == 45 else "default",
                           fontsize=txt, linespacing=1.0)


def _draw_spec(ax, spec, order=None, layout=SPACIOUS):
    if spec["kind"] == "grouped":
        draw_grouped_panel(ax, spec, order, compact=layout.compact)
    else:
        title = spec.get("short_title") if layout.compact and spec.get("short_title") else spec.get("title", "")
        draw_bar_panel(ax, spec["data"], spec["value"], spec["group"],
                       list(order if order is not None else spec["order"]),
                       stats=spec["stats"], title=title,
                       ylabel=spec.get("short_ylabel") if layout.compact and spec.get("short_ylabel") else spec.get("ylabel"),
                       labels=spec.get("labels"),
                       ymin=spec.get("ymin"), refline=spec.get("refline"),
                       palette=spec.get("palette"), max_brackets=MAX_BRACKETS)
    if layout.compact:
        compact_style(ax, layout)
    return ax


def stats_long(spec) -> pd.DataFrame:
    """Test, assumptions, post hoc and descriptive tables of a spec in one long table."""
    per = spec["stats"] if spec["kind"] == "grouped" else {None: spec["stats"]}
    frames = []
    for obj, st in per.items():
        for name in ("test", "assumptions", "posthoc", "descriptive", "outliers"):
            t = st.get(name)
            if t is None or not len(t):
                continue
            t = t.copy()
            if spec["group"] in t.columns and name in ("descriptive", "outliers"):
                t = t.rename(columns={spec["group"]: "group"})
            t.insert(0, "table", name)
            if obj is not None:
                t.insert(1, "object_type", obj)
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["table"])


def _panel_item_height(spec, order, layout=SPACIOUS, plot_mm=None):
    texts = _label_texts(order, spec.get("labels"))
    if layout.compact:
        title_lines = 2 if spec["kind"] == "bar" else 1
        legend_rows = int(math.ceil(len(spec.get("objects", [])) / 2)) if spec["kind"] == "grouped" else 0
        head = title_lines * _pt_mm(layout.title_pt) * 1.25 + legend_rows * _pt_mm(layout.text_pt) * 1.25 + 1.5
        plot = layout.plot_mm if plot_mm is None else plot_mm
        return layout.id_strip + head + plot + _xlabel_mm(texts, layout.text_pt, compact=True)
    title_lines = 2 if spec["kind"] == "bar" else 1
    return 40 + 4.2 * title_lines + _xlabel_mm(texts) + ID_STRIP_MM


def _per_category(spec, layout=SPACIOUS):
    if layout.compact:
        if spec["kind"] == "grouped":
            return 0.75 * len(spec["objects"]) + 1.4, 10
        return 4.0, 10
    if spec["kind"] == "grouped":
        return 1.9 * len(spec["objects"]) + 2.5, 22 + 30
    return 6.0, 24


def spec_items(pid, spec, max_per_row, layout=SPACIOUS, group_size=None, new_row=False):
    """Report items for a bar / grouped spec, split into condition chunks if needed.

    :param group_size: number of panels drawn side by side as one group (the
        recruitment panels, the channel panels ...); in the compact layout a
        group of up to four shares one row in equal widths.
    """
    layout = get_layout(layout)
    per_cat, fixed = _per_category(spec, layout)
    if layout.compact:
        min_frac = 1.0 / group_size if group_size and group_size <= 5 else 0.0
        width, chunks = condition_chunks(spec["order"], per_cat, fixed, max_per_row,
                                         layout=layout, min_frac=min_frac)
    else:
        width, chunks = condition_chunks(spec["order"], per_cat, fixed, max_per_row)
    items = []
    for i, chunk in enumerate(chunks):
        def draw(fig, rect, chunk=chunk):
            ax = fig.add_axes(rect)
            _draw_spec(ax, spec, chunk, layout)
            fit_axes(fig, ax, rect)
        label = pid if i == 0 else f"{pid} (continued)"
        h = _panel_item_height(spec, chunk, layout)
        hmin = _panel_item_height(spec, chunk, layout, plot_mm=layout.min_plot_mm)
        items.append(ReportItem(pid=pid, label=label, draw=draw, width=width, height_mm=h,
                                min_height_mm=hmin, new_row=new_row and i == 0))
    return items


def _save_drawing(folder, stem, draw, size_mm, dpi=300):
    """Draw into a figure of ``size_mm`` and save ``<folder>/<stem>.pdf/.png``."""
    os.makedirs(folder, exist_ok=True)
    w, h = size_mm
    fig = _new_figure(w, h)
    pad = 2.0
    draw(fig, (pad / w, pad / h, 1 - 2 * pad / w, 1 - 2 * pad / h))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"{stem}.{ext}"), dpi=dpi)


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
                 edge_column="pathogen_distance_to_field_edge"):
    """Random objects per group, objects away from the field edge first.

    An object is preferred when ``edge_column`` (distance of the centre of
    its pathogen object to the field edge) is at least half the tile size,
    so the crop can lie inside the field. Candidates are shuffled with
    ``seed``; the caller takes the first ``n`` that pass
    :func:`extract_tile`.

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


def select_median(df, group, order, value):
    """Candidates per group ordered by distance to the group median of ``value`` (typical first)."""
    out = {}
    for g in order:
        sub = df[df[group] == g]
        v = pd.to_numeric(sub[value], errors="coerce").replace([np.inf, -np.inf], np.nan)
        sub = sub[v.notna().to_numpy()]
        v = v.dropna()
        med = float(v.median()) if len(v) else np.nan
        sub = sub.iloc[np.argsort((v - med).abs().to_numpy(), kind="stable")] if len(v) else sub
        sub.attrs["median"] = med
        out[g] = sub
    return out


def extract_tile(arr, mask_plane, label, size, center_plane=None, require_inside=None):
    """Crop a ``size`` x ``size`` window around host cell ``label``.

    With ``center_plane`` (the pathogen mask plane) the window is centred on
    the centroid of the union of the pathogen objects that overlap the host
    cell, and the cell is rejected (None) when it has no pathogen object,
    when a pathogen object touches the field edge, or when the pathogen
    objects do not lie fully inside the crop. Without it the window is
    centred on the host cell mask centroid.

    :param require_inside: reject crops that reach beyond the field (default:
        True when centring on the pathogen, else False - the crop is then
        zero padded).
    :returns: (raw, box, padded) - ``raw`` is the exact in-field slice
        ``arr[y0:y1, x0:x1, :]`` (input dtype, values unchanged), ``box`` is
        ``(y0, y1, x0, x1)`` and ``padded`` the slice zero padded to the full
        tile size (equal to ``raw`` when the crop is inside the field), or
        None.
    """
    if require_inside is None:
        require_inside = center_plane is not None
    m = np.asarray(arr[..., mask_plane])
    h, w = m.shape
    ys, xs = np.nonzero(m == label)
    if not len(ys):
        return None
    if center_plane is not None:
        ya, yb = max(ys.min() - size, 0), min(ys.max() + size + 1, h)
        xa, xb = max(xs.min() - size, 0), min(xs.max() + size + 1, w)
        p = np.asarray(arr[ya:yb, xa:xb, center_plane])
        cell = m[ya:yb, xa:xb] == label
        ids = np.unique(p[cell & (p > 0)])
        if not len(ids):
            return None
        py, px = np.nonzero(np.isin(p, ids))
        py, px = py + ya, px + xa
        if py.min() == 0 or px.min() == 0 or py.max() == h - 1 or px.max() == w - 1:
            return None
        if py.max() - py.min() + 1 > size - 2 or px.max() - px.min() + 1 > size - 2:
            return None
        cy, cx = int(round(py.mean())), int(round(px.mean()))
    else:
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    half = size // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + size, x0 + size
    if require_inside and (y0 < 0 or x0 < 0 or y1 > h or x1 > w):
        return None
    if center_plane is not None and (py.min() < y0 or px.min() < x0 or py.max() >= y1 or px.max() >= x1):
        return None
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
                edge_column="pathogen_distance_to_field_edge", outline_objects=("pathogen", "cell"),
                value_column=None, center_object="pathogen", max_attempts=400,
                selection="random", range_source="fields", subdir="tiles",
                montage_name="montage_rgb16"):
    """Pick, crop, render and write the representative tiles of every group.

    :param mask_dims: {'cell': plane, 'pathogen': plane, ...}; ``object_column``
        holds the host cell label. Crops are centred on the ``center_object``
        objects of the host cell (pathogen; see :func:`extract_tile`), or on the
        host cell when that mask is absent. At most ``max_attempts`` candidates
        are tried per group.
    :param selection: ``'random'`` (seeded, objects away from the edge first)
        or ``'median'`` (objects closest to the median of ``value_column`` in
        their group first: typical objects).
    :param range_source: ``'fields'`` (display range from the pixels of the
        fields the tiles come from) or ``'crops'`` (from the pixels of the
        crops themselves, so every crop of the figure is comparable).
    :param subdir: folder for the per-crop files; ``montage_name`` the stem of
        the 16-bit montage.
    :returns: dict with ``tiles`` {group: [rgb16]}, ``table`` (one row per
        tile), ``ranges``, ``montage`` (uint16 RGB, rows = tiles, columns =
        groups), ``files`` and ``notes``.
    """
    os.makedirs(os.path.join(out_folder, subdir), exist_ok=True)
    if selection == "median":
        candidates = select_median(df, group, order, value_column)
    else:
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
        center_plane = mask_dims.get(center_object) if center_object else None
        for attempt, (idx, row) in enumerate(candidates[g].iterrows()):
            if len(chosen) >= n or attempt >= max_attempts:
                break
            label = _label_number(row.get(object_column))
            arr = load(str(row.get(file_column)))
            if arr is None or label is None:
                continue
            res = extract_tile(arr, cell_plane, label, size, center_plane=center_plane)
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
    if range_source == "crops":
        crops = [res[2] for ch in picks.values() for (_, _, _, res) in ch]
        ranges = display_ranges(crops, list(colour_map), percentiles, stride=1) if crops else {}
    else:
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
            stem = os.path.join(out_folder, subdir, f"{_safe(g)}_{k:02d}")
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
                if selection == "median":
                    rec["group_median"] = candidates[g].attrs.get("median", np.nan)
            rows.append(rec)
    montage = montage16(tiles, order, n, size)
    files += write_rgb16(os.path.join(out_folder, montage_name), montage)
    dtype = str(arrays[0].dtype) if arrays else "unknown"
    return dict(tiles=tiles, table=pd.DataFrame(rows), ranges=ranges, montage=montage,
                files=files, notes=notes, n_range_fields=len(arrays), dtype=dtype,
                centred_on=center_object if center_object in mask_dims else "cell")


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


def tile_drawer(tiles_result, chunk, labels, n, size, colour_text, layout=SPACIOUS, tile_mm=None,
                show_labels=True):
    """Draw callback, height and minimum height (mm) of one tile row of ``chunk`` conditions.

    One image column per condition (``n`` tiles stacked), its label above it,
    the colour key under the row. Columns are spaced by the wider of the tile
    and its wrapped label.
    """
    layout = get_layout(layout)
    tile_mm = layout.tile_mm if tile_mm is None else tile_mm
    pt = layout.tile_label_pt
    char_mm = _pt_mm(pt) * 0.7
    line_mm = _pt_mm(pt) * 1.25
    per_line = max(8, int(max(tile_mm, 14.0) / char_mm))
    texts = ["\n".join(textwrap.wrap(t, per_line, break_long_words=False))
             for t in _label_texts(chunk, labels)] if show_labels else [""] * len(chunk)
    n_lines = max((t.count("\n") + 1 for t in texts), default=1)
    label_w = max((len(line) for t in texts for line in t.split("\n")), default=1) * char_mm
    lab_mm = 1.0 + n_lines * line_mm if show_labels else 0.5
    key_pt = max(MIN_FONT_PT, layout.text_pt if layout.compact else 6.5)
    key_mm = 1.5 + _pt_mm(key_pt) * 1.3
    gap_px = 4

    def col_height(t_mm):
        return n * t_mm + (n - 1) * t_mm * gap_px / size

    height = layout.id_strip + lab_mm + col_height(tile_mm) + key_mm
    min_height = layout.id_strip + lab_mm + col_height(min(tile_mm, layout.min_tile_mm)) + key_mm
    cols = [montage16(tiles_result["tiles"], [g], n, size, gap=gap_px) for g in chunk]

    def draw(fig, rect):
        fw_mm, fh_mm = fig.get_size_inches() * 25.4
        x0, y0, w, h = rect
        x0m, y0m, wm, hm = x0 * fw_mm, y0 * fh_mm, w * fw_mm, h * fh_mm
        col_h = max(hm - lab_mm - key_mm, 1.0)
        t_mm = col_h / (n + (n - 1) * gap_px / size)
        col_w = t_mm * cols[0].shape[1] / size if cols else t_mm
        pitch = max(col_w + 1.0, label_w + 1.0)
        if pitch * len(chunk) > wm:          # narrower than needed: shrink the tiles
            pitch = wm / max(len(chunk), 1)
            col_w = min(col_w, pitch - 1.0)
            col_h = col_w * cols[0].shape[0] / cols[0].shape[1]
        top = y0m + hm - lab_mm
        for j, (g, img, text) in enumerate(zip(chunk, cols, texts)):
            cx = x0m + j * pitch + pitch / 2
            ax = fig.add_axes([(cx - col_w / 2) / fw_mm, (top - col_h) / fh_mm, col_w / fw_mm,
                               col_h / fh_mm])
            ax.imshow((img.astype(np.float64) / 257).round().astype(np.uint8),
                      interpolation="none", aspect="auto")
            ax.set_axis_off()
            fig.text(cx / fw_mm, (top + 0.6) / fh_mm, text, fontsize=pt, ha="center", va="bottom",
                     linespacing=1.0)
        wrap = max(40, int(wm / (_pt_mm(key_pt) * 0.55)))
        fig.text(x0m / fw_mm, (top - col_h - 0.8) / fh_mm, "\n".join(textwrap.wrap(colour_text, wrap)),
                 fontsize=key_pt, va="top", ha="left")
    return draw, height, min_height


def _tile_mm(n_cols, width_mm, layout=SPACIOUS):
    return min(layout.tile_mm, (width_mm - 2) / max(n_cols, 1) - 1)


def tile_items(pid, tiles_result, order, labels, n, size, max_per_row, colour_text, layout=SPACIOUS):
    """Report items for the tile montage, split into condition chunks."""
    layout = get_layout(layout)
    _, chunks = condition_chunks(order, 13.0 if not layout.compact else 15.0, 0, max_per_row,
                                 page=layout.page)
    items = []
    for i, chunk in enumerate(chunks):
        draw, height, min_height = tile_drawer(tiles_result, chunk, labels, n, size, colour_text,
                                               layout, _tile_mm(len(chunk), layout.content, layout))
        label = pid if i == 0 else f"{pid} (continued)"
        items.append(ReportItem(pid=pid, label=label, draw=draw, width=1.0, height_mm=height,
                                min_height_mm=min_height))
    return items


# =========================================================================== plate heatmaps
def _well_index(value):
    """'r12' / 'c3' / '12' -> 12; a plate row letter ('B') -> 2."""
    m = re.search(r"(\d+)$", str(value))
    if m:
        return int(m.group(1))
    t = str(value).strip().upper()
    return ord(t[0]) - 64 if t and t[0].isalpha() else None


def plate_shape(rows, cols):
    """Standard plate layout holding the given row / column numbers."""
    r, c = max(rows, default=1), max(cols, default=1)
    for nr, nc in ((8, 12), (16, 24), (32, 48)):
        if r <= nr and c <= nc:
            return nr, nc
    return r, c


def plate_matrices(data, value, well_keys=WELL_KEYS):
    """{plate: (matrix rows x columns, nan where no data)} of well values."""
    d = data.copy()
    d["_r"] = d[well_keys[1]].map(_well_index)
    d["_c"] = d[well_keys[2]].map(_well_index)
    d = d.dropna(subset=["_r", "_c"])
    shape = plate_shape(d["_r"].astype(int), d["_c"].astype(int))
    out = {}
    for plate, sub in d.groupby(well_keys[0], observed=True):
        m = np.full(shape, np.nan)
        v = pd.to_numeric(sub[value], errors="coerce").to_numpy(float)
        m[sub["_r"].astype(int).to_numpy() - 1, sub["_c"].astype(int).to_numpy() - 1] = v
        out[plate] = m
    return out


def heatmap_spec(data, value, title, failed=None, vrange=None, well_keys=WELL_KEYS):
    """Plate heatmap of well values; ``failed`` = {(plate, row, column)} drawn with a red x."""
    keys = list(well_keys)
    d = data[keys + [value]].copy()
    d = d.groupby(keys, observed=True)[value].mean().reset_index()
    fin = pd.to_numeric(d[value], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vrange is None:
        vrange = (float(fin.quantile(0.01)), float(fin.quantile(0.99))) if len(fin) else (0.0, 1.0)
    failed = set(failed or ())
    d["qc_failed"] = [(str(a), str(b), str(c)) in failed for a, b, c in
                      d[keys].astype(str).itertuples(index=False)]
    return dict(kind="heatmap", data=d, value=value, title=title, matrices=plate_matrices(d, value, keys),
                failed=failed, vrange=vrange, well_keys=keys)


def draw_heatmap(fig, rect, spec, plate, layout=SPACIOUS):
    """Plate heatmap with a colour bar inside ``rect``; failed QC wells marked with a red x."""
    import matplotlib
    m = spec["matrices"][plate]
    nr, nc = m.shape
    x0, y0, w, h = rect
    fw, fh = fig.get_size_inches() * 25.4
    cbar_w = (2.0 if layout.compact else 3.0) / fw
    pad = (5.5 if layout.compact else 9.0) / fw
    ax = fig.add_axes([x0, y0, w - cbar_w - pad, h])
    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    lo, hi = spec["vrange"]
    im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1,
                   interpolation="none", aspect="equal")
    keys = spec["well_keys"]
    fail = spec["data"][(spec["data"]["qc_failed"]) & (spec["data"][keys[0]].astype(str) == str(plate))]
    if len(fail):
        rr = fail[keys[1]].map(_well_index).astype(float) - 1
        cc = fail[keys[2]].map(_well_index).astype(float) - 1
        ax.scatter(cc, rr, marker="x", s=4 if layout.compact else 10, linewidths=0.5, color="#d62728")
    fs = layout.text_pt if layout.compact else 7
    step_c = 4 if nc > 12 else 2
    step_r = 4 if nr > 8 else 2
    ax.set_xticks(np.arange(0, nc, step_c))
    ax.set_xticklabels([str(i + 1) for i in range(0, nc, step_c)], fontsize=fs)
    ax.set_yticks(np.arange(0, nr, step_r))
    ax.set_yticklabels([chr(65 + i) if nr <= 26 else str(i + 1) for i in range(0, nr, step_r)],
                       fontsize=fs)
    ax.tick_params(length=1.5, width=0.4, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    multi = len(spec["matrices"]) > 1
    ax.set_title(spec["title"] + (f" ({plate})" if multi else ""),
                 fontsize=layout.title_pt if layout.compact else 8, pad=2)
    fit_axes(fig, ax, (x0, y0, w - cbar_w - pad, h))
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad * 0.25, pos.y0, cbar_w, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fs, length=1.2, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    return ax


def plate_heatmap_items(pid, spec, layout=SPACIOUS, new_row=False):
    """One report item per plate of a heatmap spec."""
    layout = get_layout(layout)
    frac = 0.25 if layout.compact else 0.5
    w = layout.width_mm(frac)
    items = []
    for i, plate in enumerate(spec["matrices"]):
        nr, nc = spec["matrices"][plate].shape
        map_w = w - (9.0 if layout.compact else 16.0)
        height = layout.id_strip + map_w * nr / nc + (7.0 if layout.compact else 11.0)

        def draw(fig, rect, plate=plate):
            draw_heatmap(fig, rect, spec, plate, layout)
        items.append(ReportItem(pid=pid, label=pid if i == 0 else f"{pid} ({plate})", draw=draw,
                                width=frac, height_mm=height, min_height_mm=height,
                                new_row=new_row and i == 0))
    return items


def save_heatmap_panel(out_dir, pid, name, spec, dpi=300):
    plates = list(spec["matrices"])
    nr, nc = spec["matrices"][plates[0]].shape if plates else (16, 24)

    def draw(fig, rect):
        n = max(len(plates), 1)
        x0, y0, w, h = rect
        for k, plate in enumerate(plates):
            draw_heatmap(fig, (x0, y0 + h * (n - 1 - k) / n, w, h / n), spec, plate, SPACIOUS)
    width = 110.0
    return write_panel_folder(out_dir, pid, name, draw,
                              (width, len(plates) * ((width - 16) * nr / nc + 14) + 4),
                              data=spec["data"], dpi=dpi)


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


def _outlier_text(spec):
    per = spec["stats"] if spec["kind"] == "grouped" else {None: spec["stats"]}
    tables = [(o, st["outliers"]) for o, st in per.items() if "outliers" in st]
    if not tables:
        return ""
    t0 = tables[0][1]
    fence = f"{t0['method'].iloc[0].upper()} fence {t0['fence'].iloc[0]:g}" if len(t0) else "robust fence"
    parts = []
    for obj, t in tables:
        rem = t[t["n_removed"] > 0]
        txt = ", ".join(f"{display_label(r[spec['group']], spec.get('labels'))}: {int(r['n_removed'])}"
                        for _, r in rem.iterrows())
        if txt:
            parts.append((f"{obj} " if obj else "") + txt)
    removed = "; ".join(parts) if parts else "none"
    return (f" Outlier points beyond the {fence} of their condition are left out of the bars and the "
            f"test (kept in data.csv with outlier = True); removed per condition: {removed}.")


def _spec_legend(pid, spec, what, unit_text):
    return _spec_legend_base(pid, spec, what, unit_text) + _outlier_text(spec)


def _spec_legend_base(pid, spec, what, unit_text):
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
                       name="report", count_objects=None, db_path=None, layout="compact",
                       well_qc: Optional[pd.DataFrame] = None, heatmaps=True, crops=None,
                       dpi=200) -> dict:
    """Build the standard report: tiles, value panels, control panels, legend.

    :param frames: ``{'unfiltered': df, 'filtered': df}``, one row per
        ``row_object``.
    :param tiles: None to skip, else keyword arguments of :func:`build_tiles`
        (``merged_dir``, ``mask_dims``, ``colour_map``, ``n``, ``size``,
        ``seed``) - tiles come from ``frames['filtered']``.
    :param db_path: measurements.db; objects per well before filtering are
        counted from its object tables (:func:`db_well_counts`), after
        filtering from the analysed frame (:func:`well_counts`).
    :param layout: ``'compact'`` (one A4 page when it fits) or ``'spacious'``.
    :param crops: None, or ``{'n': k}``: a representative crop per condition
        (closest to the condition median of ``value``) and, under every value
        graph, a grid of the ``k`` crops nearest each condition median, drawn
        from the same data version, scaled to the 1st-99th percentile of the
        crops of that figure (needs ``tiles`` for the merged arrays).
    :param well_qc: the :func:`well_qc_table`; failed wells are marked in the
        plate heatmaps (``heatmaps``: one per plotted variable, before and
        after filtering, as an appendix after the main figure).
    :returns: dict with ``pdf``, ``png``, ``legend``, ``panels`` (id -> folder),
        ``n_pages``, ``n_figure_pages`` and ``n_legend_pages``.
    """
    layout = get_layout(layout)
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    order = list(order)
    filt = frames["filtered"]
    blocks, legend, folders = [], list(intro), {}
    counter = [0]

    def next_id():
        counter[0] += 1
        return f"R{counter[0]:02d}"

    heat = {}   # key -> {'title', 'value', 'unfiltered' / 'filtered': well frame}

    def hv(key, title, version, frame, col):
        if frame is None or not len(frame) or col not in frame.columns:
            return
        heat.setdefault(key, dict(title=title, value=col))[version] = frame

    def add_items(pid, spec, size, first):
        return spec_items(pid, spec, max_conditions_per_row, layout, group_size=size, new_row=first)

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
            colour_text = (", ".join(f"{_colour_name(c)}: channel {ch}" for ch, c in cmap.items())
                           + "; white outline: pathogen mask, grey: host cell mask")
            n, size = tiles.get("n", 5), tiles.get("size", 128)
            items = tile_items(pid, tr, order, labels, n, size, max_conditions_per_row, colour_text,
                               layout)
            blocks.append(ReportBlock("Representative cells (filtered objects)", items))
            ranges = pd.DataFrame([dict(table="display_range", channel=c, colour=_colour_name(cmap[c]),
                                        low=lo, high=hi, percentiles=str(tiles.get("percentiles", (0.5, 99.9))),
                                        n_fields=tr["n_range_fields"]) for c, (lo, hi) in tr["ranges"].items()])
            cand = pd.DataFrame([dict(table="candidates", group=g,
                                      n_filtered_objects=int((filt[group] == g).sum()),
                                      n_tiles=len(tr["tiles"].get(g, []))) for g in order])
            full_w = max(_content_width(), 26.0 * len(order) + 4)
            draw_all, h_all, _ = tile_drawer(tr, order, labels, n, size, colour_text, SPACIOUS,
                                             _tile_mm(len(order), full_w))
            write_panel_folder(panels_dir, pid, "representative_tiles", draw_all,
                               (full_w, h_all - ID_STRIP_MM + 4),
                               data=tr["table"], stats=pd.concat([ranges, cand], ignore_index=True),
                               dpi=dpi)
            folders[pid] = folder
            pct = tiles.get("percentiles", (0.5, 99.9))
            rng_txt = "; ".join(f"channel {c}: {lo:.0f}-{hi:.0f}" for c, (lo, hi) in tr["ranges"].items())
            centre = ("centred on the centroid of the pathogen (vacuole) objects of the host cell; only "
                      "host cells whose pathogen objects lie fully inside the crop and away from the "
                      "field edge, with the crop inside the field, are used"
                      if tr.get("centred_on") == "pathogen" else
                      "centred on the centroid of the host cell mask (zero padded at the field edge)")
            legend.append(
                f"{pid}. Representative host cells, {n} per condition (columns, in the condition order "
                f"of the graphs), chosen at random (seed {tiles.get('seed', 0)}) from the filtered "
                f"objects, trying first those whose pathogen centre lies at least half a tile "
                f"({size // 2} px) from the field edge. Each crop is {size} x {size} px, {centre}. "
                f"Composite: {colour_text}. Every channel is scaled linearly between its {pct[0]}th and "
                f"{pct[1]}th percentile over the pixels of {tr['n_range_fields']} fields ({rng_txt}); "
                f"the same range is used for every tile. Raw crops (every image channel and mask plane, "
                f"input dtype {tr.get('dtype', 'uint16')}, values unchanged) are in tiles/*_raw.tif; "
                f"16-bit RGB composites in tiles/*_rgb16.png/.tif; the montage at native resolution in "
                f"montage_rgb16.png/.tif (16-bit). The figure PDF/PNG is 8-bit, drawn without "
                f"interpolation." + (" " + "; ".join(tr["notes"]) + "." if tr["notes"] else ""))

    crop_cfg = None
    if crops and tiles:
        crop_cfg = dict(merged_dir=tiles["merged_dir"], mask_dims=tiles["mask_dims"],
                        colour_map=tiles["colour_map"], size=tiles.get("size", 128),
                        percentiles=(1, 99), selection="median", range_source="crops",
                        subdir="crops", montage_name="montage16")
        ckey = ", ".join(f"{_colour_name(c)}: channel {ch}" for ch, c in tiles["colour_map"].items())
        pid = next_id()
        folder = os.path.join(panels_dir, panel_folder_name(pid, "representative_per_condition"))
        try:
            rp = build_tiles(filt, group, order, out_folder=folder, value_column=value, n=1, **crop_cfg)
        except Exception as exc:
            rp = None
            legend.append(f"{pid}. Representative crops could not be made: {exc}.")
        if rp is not None and len(rp["table"]):
            size = crop_cfg["size"]
            tile_mm = min(16.0 if layout.compact else 24.0, layout.content / max(len(order), 1) - 2)
            draw, h, hmin = tile_drawer(rp, order, labels, 1, size, ckey, layout, tile_mm)
            item = ReportItem(pid=pid, label=pid, draw=draw, width=1.0, height_mm=h,
                              min_height_mm=tile_drawer(rp, order, labels, 1, size, ckey, layout,
                                                        tile_mm * 0.7)[1])
            blocks.append(ReportBlock(f"Representative {value_label} per condition (filtered objects)",
                                      [item]))
            fd, fh, _ = tile_drawer(rp, order, labels, 1, size, ckey, SPACIOUS, 26.0)
            write_panel_folder(panels_dir, pid, "representative_per_condition", fd,
                               (max(_content_width(), 28.0 * len(order) + 4), fh - ID_STRIP_MM + 4),
                               data=rp["table"], dpi=dpi)
            folders[pid] = folder
            legend.append(
                f"{pid}. One representative object per condition: the filtered host cell whose "
                f"{value_label} lies closest to the median of its condition, among cells whose pathogen "
                f"objects lie fully inside the {size} x {size} px crop and away from the field edge. "
                f"Composite: {ckey}; each channel scaled linearly between its 1st and 99th percentile "
                f"over the pixels of these crops. crops/*_raw.tif hold the raw uint16 planes, "
                f"crops/*_rgb16.png/.tif and montage16.png/.tif the 16-bit composites; data.csv names "
                f"the object of each condition, its value and the condition median.")

    # ---- value panels
    items, sub_legend = [], []
    grid_items = []
    levels = [("unfiltered", "well"), ("unfiltered", "replicate"),
              ("filtered", "well"), ("filtered", "replicate")]
    if not replicate_column:
        levels = [lv for lv in levels if lv[1] == "well"]
    for k, (version, level) in enumerate(levels):
        df = frames[version]
        df = df[df[group].isin(order)]
        unit = aggregate_to_unit(df, value, [group],
                                 replicate_column=replicate_column if level == "replicate" else None)
        fin = pd.to_numeric(unit[value], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ymin = 1 if ratio and len(fin) and fin.min() >= 1 else None
        spec = bar_spec(unit, value, group, order, title=f"{value_label}: {version}, {level} level",
                        ylabel=value_label, labels=labels, ymin=ymin, refline=1 if ratio else None,
                        palette=palette)
        spec["short_title"] = f"{version}, {level} level"
        if level == "well":
            hv(value, value_label, version, unit, value)
        pid = next_id()
        folders[pid] = save_spec_panel(panels_dir, pid, f"{value}_{version}_{level}", spec, dpi=dpi)
        items += add_items(pid, spec, len(levels), k == 0)
        if crop_cfg is not None:
            gfolder = folders[pid] + "_crops"
            try:
                gr = build_tiles(df, group, order, out_folder=gfolder, value_column=value,
                                 n=int(crops.get("n", 5)), **crop_cfg)
                gr["table"].to_csv(os.path.join(gfolder, "data.csv"), index=False)
                frac = items[-1].frac
                gw = layout.width_mm(frac)
                tmm = min(layout.tile_mm * 1.2, (gw - 2) / max(len(order), 1) - 0.8)
                gdraw, gh, _ = tile_drawer(gr, order, labels, int(crops.get("n", 5)), crop_cfg["size"],
                                           "", layout, tmm, show_labels=False)
                gdraw_s, gh_s, _ = tile_drawer(gr, order, labels, int(crops.get("n", 5)), crop_cfg["size"],
                                               "", SPACIOUS, 20.0)
                _save_drawing(gfolder, os.path.basename(gfolder), gdraw_s,
                              (max(80.0, 24.0 * len(order)), gh_s), dpi=dpi)
                grid_items.append(ReportItem(pid=f"{pid} crops", label=f"{pid} crops", draw=gdraw,
                                             width=frac, height_mm=gh, min_height_mm=gh,
                                             new_row=not grid_items))
                sub_legend.append(f"{pid} crops. Under {pid}: the {int(crops.get('n', 5))} {version} objects "
                                  f"closest to the condition median of {value_label}, one column per "
                                  f"condition in the x-axis order of {pid}; scaled to the 1st-99th "
                                  f"percentile of these crops; files in {os.path.basename(gfolder)}/.")
            except Exception as exc:
                sub_legend.append(f"{pid} crops could not be made: {exc}.")
        unit_text = (f"replicates (column '{replicate_column}'; each the mean of its well means)"
                     if level == "replicate" else "wells (each the mean of its objects)")
        sub_legend.append(_spec_legend(pid, spec, f"{value_label} per condition, {version} objects, "
                                                  f"{level} level.", unit_text))
    blocks.append(ReportBlock(f"{value_label} per condition", items + grid_items))
    if not replicate_column:
        sub_legend.append("Replicate-level panels are not drawn because replicate_column is not set.")
    legend += sub_legend

    # ---- control panels, before and after filtering
    db_tables = None
    if db_path and os.path.exists(db_path):
        try:
            db_tables = db_object_tables(db_path) or None
        except Exception:  # an unreadable database falls back to counting the frame
            db_tables = None
    for version in ("unfiltered", "filtered"):
        df = frames[version]
        df = df[df[group].isin(order)]
        items = []
        objs_all = object_types(df, channels)
        chans = [c for c in channels if object_types(df, [c])]
        for k, chan in enumerate(chans):
            objs = object_types(df, [chan])
            data = well_intensity_long(df, group, chan, objs)
            for obj in objs:
                hv(f"{obj}_ch{chan}_intensity", f"{obj} ch{chan} intensity", version,
                   data[data["object_type"] == obj], "mean_intensity")
            spec = grouped_spec(data, group, order, objs,
                                title=(f"channel {chan}" if layout.compact else
                                       f"channel {chan} in all objects ({version})"),
                                ylabel=(f"mean intensity ch {chan}" if layout.compact else
                                        f"mean intensity, channel {chan}"), labels=labels)
            pid = next_id()
            folders[pid] = save_spec_panel(panels_dir, pid, f"channel_{chan}_all_objects_{version}",
                                           spec, dpi=dpi)
            items += add_items(pid, spec, len(chans), k == 0)
            legend.append(_spec_legend(pid, spec, f"Mean intensity of channel {chan} per well in every "
                                                  f"object type ({', '.join(objs)}), {version} objects; "
                                                  f"a log axis is used when object means differ more "
                                                  f"than 30-fold.", "wells"))
        if count_objects is not None:
            count_types = [o for o in count_objects if o != "cytoplasm"]
        elif db_tables:
            count_types = list(db_tables)
        else:
            count_types = [o for o in objs_all if o != "cytoplasm"]
        db_counts = None
        if version == "unfiltered" and db_tables:
            db_counts = db_well_counts(db_path, df, group, [t for t in count_types if t in db_tables])
        row_b = []   # (kind, name, spec, legend text)
        # object areas: one grouped panel, one colour per object type
        area_objs = [o for o in object_types(df) if f"{o}_area" in df.columns]
        if area_objs:
            data = well_column_long(df, group, {o: f"{o}_area" for o in area_objs}, "mean_area")
            for obj in area_objs:
                hv(f"{obj}_area", f"{obj} area", version, data[data["object_type"] == obj], "mean_area")
            spec = grouped_spec(data, group, order, area_objs, value="mean_area",
                                title="object area" if layout.compact else f"object area per well ({version})",
                                ylabel="mean area (px)", labels=labels, log_ratio=10.0)
            row_b.append((f"object_area_{version}", spec,
                          f"Mean area (px) of every object type ({', '.join(area_objs)}) per well, "
                          f"{version} objects; a log axis is used when object means differ more than "
                          f"10-fold."))
        for obj in count_types:
            if db_counts is not None and obj in set(db_counts["object_type"]):
                data = db_counts[db_counts["object_type"] == obj].drop(columns=["object_type"])
            else:
                data = well_counts(df, group, obj, row_object)
            if not len(data):
                continue
            hv(f"{obj}_count", f"{obj} per well", version, data, "objects_per_well")
            spec = bar_spec(data, "objects_per_well", group, order,
                            title=f"{obj} objects per well ({version})",
                            ylabel=f"{obj} objects per well", labels=labels, palette=palette)
            spec["short_title"] = f"{obj} per well"
            src = str(data["count_source"].iloc[0])
            how = (f"rows per well in the {src.split()[-1]} table of measurements.db (every "
                   f"segmented object, before any filter)" if src.startswith("measurements.db") else
                   "rows of the analysed table (one per host cell)" if src == "rows" else
                   f"the sum of {src} (objects of this type merged into each analysed host cell)"
                   if src.endswith("_prcfo_count") else f"analysed host cells with a {src} value")
            row_b.append((f"{obj}_objects_per_well_{version}", spec,
                          f"{obj} objects per well, {version}, counted as {how}."))
        load_col = "pathogen_prcfo_count"
        if load_col in df.columns:
            unit = aggregate_to_unit(df, load_col, [group])
            unit = unit.rename(columns={load_col: "pathogens_per_host_cell"})
            hv("pathogens_per_host_cell", "pathogens per host cell", version, unit,
               "pathogens_per_host_cell")
            spec = bar_spec(unit, "pathogens_per_host_cell", group, order,
                            title=f"pathogens per host cell ({version})",
                            ylabel="pathogens per host cell", labels=labels, palette=palette)
            spec["short_title"] = "pathogens per cell"
            row_b.append((f"pathogens_per_host_cell_{version}", spec,
                          f"Infection load: mean number of pathogen objects per analysed host cell "
                          f"(pathogen_prcfo_count) per well, {version} objects."))
        for k, (pname, spec, text) in enumerate(row_b):
            pid = next_id()
            folders[pid] = save_spec_panel(panels_dir, pid, pname, spec, dpi=dpi)
            items += add_items(pid, spec, len(row_b), k == 0)
            legend.append(_spec_legend(pid, spec, text, "wells"))
        title_v = "before filtering (unfiltered)" if version == "unfiltered" else "after filtering"
        blocks.append(ReportBlock(f"Well-level controls {title_v}", items, group="controls"))

    # ---- plate heatmaps (appendix)
    if heatmaps and heat:
        keys = list(WELL_KEYS)
        failed = set()
        if well_qc is not None and "well_qc_pass" in well_qc.columns:
            failed = {tuple(map(str, r)) for r in
                      well_qc.loc[~well_qc["well_qc_pass"].astype(bool), keys].itertuples(index=False)}
        families = [("main", f"Plate heatmaps: {value_label}"), ("count", "Plate heatmaps: objects per well"),
                    ("area", "Plate heatmaps: object area"), ("intensity", "Plate heatmaps: mean intensity"),
                    ("load", "Plate heatmaps: pathogens per host cell")]

        def family(key):
            return ("main" if key == value else "count" if key.endswith("_count") else
                    "area" if key.endswith("_area") else "load" if key == "pathogens_per_host_cell"
                    else "intensity")
        first_h, last_h = None, None
        hcount = [0]
        shared_scale = {}
        for fam, heading in families:
            items = []
            for key, hvd in heat.items():
                if family(key) != fam:
                    continue
                vals = pd.concat([pd.to_numeric(hvd[v][hvd["value"]], errors="coerce")
                                  for v in ("unfiltered", "filtered") if v in hvd])
                vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
                vrange = (float(vals.quantile(0.01)), float(vals.quantile(0.99))) if len(vals) else None
                meds = [pd.to_numeric(hvd[v][hvd["value"]], errors="coerce").median()
                        for v in ("unfiltered", "filtered") if v in hvd]
                if len(meds) == 2 and min(meds) > 0 and max(meds) / min(meds) > 2:
                    vrange = None      # different quantities before / after (e.g. counts): own scales
                shared_scale[key] = vrange is not None
                for version in ("unfiltered", "filtered"):
                    if version not in hvd:
                        continue
                    hs = heatmap_spec(hvd[version], hvd["value"], f"{hvd['title']}, {version}",
                                      failed=failed, vrange=vrange)
                    hcount[0] += 1
                    hid = f"H{hcount[0]:02d}"
                    folders[hid] = save_heatmap_panel(panels_dir, hid, f"heatmap_{key}_{version}", hs, dpi=dpi)
                    items += plate_heatmap_items(hid, hs, layout)
                    first_h = first_h or hid
                    last_h = hid
            if items:
                blocks.append(ReportBlock(heading, items, group="heatmaps"))
        if first_h:
            legend.append(f"{first_h}-{last_h}. Plate heatmaps of every plotted well-level variable, before "
                          f"(unfiltered) and after filtering, side by side; one cell per well in the plate "
                          f"layout (rows lettered, columns numbered). The colour scale spans the 1st to 99th "
                          f"percentile of the well values; it is shared by the two maps of a variable unless "
                          f"their medians differ more than 2-fold (own scales for "
                          f"{', '.join(k for k, v in shared_scale.items() if not v) or 'none'}); grey wells have no data (not annotated, or removed by the "
                          f"filters); a red x marks wells that failed well_qc"
                          f" ({len(failed)} wells). Each map is in its own panel folder (PDF, PNG, data.csv).")

    legend.append("Filters. Unfiltered: spaCR output after the standard object filters. Filtered: "
                  "after the opt-in filters. " + filter_sentences(filter_summary))
    legend.append("Statistics. The experimental unit is the well (mean of its objects) or the "
                  "replicate (mean of its well means); tests never treat single cells as independent. "
                  "Per panel: Shapiro-Wilk normality per group and Levene equal variance (median "
                  "centred); all normal and equal variance: one-way ANOVA + Tukey HSD; normal with "
                  "unequal variance: Welch ANOVA + Games-Howell; otherwise Kruskal-Wallis + Dunn test "
                  "(Holm). Each panel folder holds data.csv (exactly the plotted values; count panels "
                  "name their source in count_source) and stats.csv (test, assumptions, post hoc and "
                  "descriptive tables in long format).")
    legend_path = os.path.join(out_dir, "figure_legend.txt")
    legend_text = write_legend(legend_path, [f"Figure legend: {title}"] + legend)
    pages = paginate(blocks, layout)
    n_main = sum(1 for pg in pages if any(pl["kind"] == "item" and not str(pl["item"].pid).startswith("H")
                                          for pl in pg))
    res = render_report(pages, os.path.join(out_dir, f"{name}.pdf"), legend_text, title=title,
                        layout=layout, dpi=dpi)
    res.update(legend=legend_path, panels=folders, layout=layout.name, n_main_pages=n_main)
    return res


def _colour_name(c):
    if isinstance(c, str):
        return c
    for k, v in COLOURS.items():
        if tuple(v) == tuple(float(x) for x in c):
            return k
    return str(tuple(c))
