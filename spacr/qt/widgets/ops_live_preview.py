"""The OPS Live preview: one tab per step of the optical pooled screen (alpha).

The tabs are Align & stitch, Spots & genotyping, Objects & phenotype, Merge &
aggregation, and Clustering. While Live is on, the tab on view runs its step
on a small sample with the settings on the form, off the GUI thread, and a
changed setting runs its own tab again. Each tab reuses what the tabs before
it computed. The work itself is in :mod:`spacr.ops_live`.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QSizePolicy, QTableWidget,
                               QTabWidget, QVBoxLayout,
                               QWidget)

from ..i18n import tr
from .preview_contract import (PREVIEW_CANCEL_TEXT, PREVIEW_RUN_TEXT,
                               PREVIEW_RUNNING_MESSAGE, LivePreviewContract,
                               preview_failure_message)

LOG = logging.getLogger("spacr.qt.widgets.ops_live_preview")

_STEP_ORDER: Tuple[str, ...] = ("align", "spots", "objects", "merge", "cluster")


def _title(step: str) -> str:
    """A tab's name in the interface language."""
    return {
        "align": tr("Align & stitch"),
        "spots": tr("Spots & genotyping"),
        "objects": tr("Objects & phenotype"),
        "merge": tr("Merge & aggregation"),
        "cluster": tr("Clustering"),
    }[step]


def _blurb(step: str) -> str:
    """One sentence on what a tab runs, in the interface language."""
    return {
        "align": tr("Stitch a 2 x 2 block of sequencing tiles, place the "
                    "phenotype fields on it and align every cycle of one crop "
                    "to cycle 1."),
        "spots": tr("Segment the nuclei of one sequencing crop, detect its "
                    "reads, call their bases and give each nucleus the guide "
                    "its reads agree on."),
        "objects": tr("Segment nuclei and cells on one phenotype crop and "
                      "measure them."),
        "merge": tr("Match phenotype cells to called nuclei, then reduce the "
                    "matched cells to one profile per gene."),
        "cluster": tr("Embed the gene profiles and cluster them at the first "
                      "graph, resolution and seed the settings name."),
    }[step]


_DEBOUNCE_MS = 600

_ORPHANS: List[QThread] = []

_FIGURE_SIZE = (1080, 300)

_TABLE_ROWS = 40


def _theme_colours() -> Dict[str, str]:
    """The colours a preview picture is drawn in, from the active theme."""
    try:
        from ..theme import active_palette
        palette = active_palette()
    except Exception:
        palette = {}
    return {
        "face": palette.get("surface_alt", "#161719"),
        "fg": palette.get("fg", "#ffffff"),
        "muted": palette.get("fg_muted", "#a1a6ad"),
        "border": palette.get("border", "#707276"),
        "blue": palette.get("accent", "#4A9EFF"),
        "blue_hi": palette.get("accent_hi", "#66B2FF"),
        "navy": palette.get("accent_lo", "#2F80D9"),
        "teal": palette.get("chip_class", "#1fb6ad"),
        "warning": palette.get("warning", "#d29922"),
    }


_ON_IMAGE = {"blue": "#9fc5ff", "teal": "#5ad1c6", "text": "#ffffff"}

_CLUSTER_COLOURS = ("#4A9EFF", "#1fb6ad", "#2F80D9", "#7fd1e8", "#0a6f6a",
                    "#66B2FF", "#3b5b92", "#5ad1c6", "#9fc5ff", "#14806f")


def _scaled(image: np.ndarray, low: float = 1.0, high: float = 99.5
            ) -> np.ndarray:
    """``image`` stretched between two percentiles to 0..1, for display only."""
    image = np.asarray(image, np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def _tint(image: np.ndarray, colour: str) -> np.ndarray:
    """A 0..1 image drawn in one colour, as RGB."""
    from matplotlib.colors import to_rgb

    return np.asarray(image, np.float32)[..., None] * np.array(to_rgb(colour))


def _outline(labels: np.ndarray) -> np.ndarray:
    """The boundary pixels of a label mask."""
    from scipy import ndimage

    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(labels.shape, bool)
    grown = ndimage.grey_dilation(labels, size=3)
    shrunk = ndimage.grey_erosion(labels, size=3)
    return (labels > 0) & (grown != shrunk)


def _figure(colours: Mapping[str, str], size: Tuple[int, int]):
    """The three-panel figure every tab draws on, identical in geometry."""
    from matplotlib.figure import Figure

    dpi = 100.0
    figure = Figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    figure.patch.set_facecolor(colours["face"])
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=0.05, right=0.985, top=0.88, bottom=0.16,
                           wspace=0.28)
    for ax in axes:
        ax.set_facecolor(colours["face"])
        for spine in ax.spines.values():
            spine.set_color(colours["border"])
        ax.tick_params(colors=colours["muted"], labelsize=7)
        ax.xaxis.label.set_color(colours["fg"])
        ax.yaxis.label.set_color(colours["fg"])
        ax.title.set_color(colours["fg"])
    return figure, axes


def _picture_axes(ax) -> None:
    """Take the ticks off an image panel."""
    ax.set_xticks([])
    ax.set_yticks([])


def _rgb(figure) -> np.ndarray:
    """Render a figure to an ``(H, W, 3)`` uint8 array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _message(ax, text: str, colours: Mapping[str, str]) -> None:
    """Write a short sentence in the middle of an empty panel."""
    _picture_axes(ax)
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
            color=colours["muted"], fontsize=8, transform=ax.transAxes)


def _draw_align(result: Mapping[str, Any], colours, size):
    """Tab 1: the mosaic with the fields, and one crop before and after alignment."""
    figure, (mosaic_ax, before_ax, after_ax) = _figure(colours, size)
    step = result["step"]
    top, left = result["origin"]
    mosaic = _scaled(result["mosaic"])
    mosaic_ax.imshow(_tint(mosaic, colours["blue"]), interpolation="nearest")
    height, width = result["tile_shape"]
    from matplotlib.patches import Polygon, Rectangle

    for site, (y, x) in result["tiles"].items():
        mosaic_ax.add_patch(Rectangle(((x - left) / step, (y - top) / step),
                                      width / step, height / step, fill=False,
                                      lw=0.8, ec=_ON_IMAGE["blue"]))
        mosaic_ax.text((x - left) / step + 4, (y - top) / step + 14, str(site),
                       color=_ON_IMAGE["text"], fontsize=6)
    from spacr.ops_live import _ph_to_well

    for info in result["fields"]:
        if not info.get("files"):
            continue
        h, w = info["shape"]
        corners = _ph_to_well(info, [[0, 0], [0, w], [h, w], [h, 0]])
        xy = np.column_stack([(corners[:, 1] - left) / step,
                              (corners[:, 0] - top) / step])
        chosen = (result.get("sample_field") or {}).get("site") == info["site"]
        mosaic_ax.add_patch(Polygon(xy, closed=True, fill=False,
                                    lw=1.4 if chosen else 0.8,
                                    ec=_ON_IMAGE["teal"]))
    site = result["crop_site"]
    cy, cx = result["tiles"][site]
    row, column, crop = result["crop"]
    mosaic_ax.add_patch(Rectangle(((cx + column - left) / step,
                                   (cy + row - top) / step), crop / step,
                                  crop / step, fill=False, lw=1.0, ls="--",
                                  ec=_ON_IMAGE["text"]))
    mosaic_ax.set_title(tr("Tiles (blue), phenotype fields (teal)"), fontsize=8)
    _picture_axes(mosaic_ax)
    last = result["last_cycle"]
    for ax, (first, other), title in (
            (before_ax, result["before"],
             tr("Cycle 1 (teal) and cycle {n} (blue), before", n=last)),
            (after_ax, result["after"], tr("After alignment"))):
        overlay = np.clip(_tint(_scaled(first), colours["teal"])
                          + _tint(_scaled(other), colours["blue"]), 0, 1)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        _picture_axes(ax)
    rows = [[str(cycle), f"{dy:+.0f}", f"{dx:+.0f}", tr("kept")]
            for cycle, (dy, dx) in sorted(result["cycle_shifts"].items())]
    rows += [[str(c), "", "", tr("refused")] for c in result["refused"]]
    rows += [[str(c), "", "", tr("missing")] for c in result["missing"]]
    header = [tr("Cycle"), tr("Shift y (px)"), tr("Shift x (px)"), tr("State")]
    sample = tr("{tiles} tiles, {fields} phenotype fields; crop {size} px of tile {site}",
                tiles=len(result["tiles"]),
                fields=sum(1 for f in result["fields"] if f.get("files")),
                size=crop, site=site)
    return _rgb(figure), header, rows, sample


def _draw_spots(result: Mapping[str, Any], colours, size):
    """Tab 2: spots on the crop, read quality and the per-cell calls."""
    figure, (image_ax, quality_ax, calls_ax) = _figure(colours, size)
    dapi = _tint(_scaled(result["dapi"]), colours["navy"])
    edge = _outline(result["nuclei"])
    dapi[edge] = np.array([0.55, 0.85, 0.85])
    image_ax.imshow(np.clip(dapi, 0, 1), interpolation="nearest")
    peaks = np.asarray(result["peaks"], float).reshape(-1, 2)
    if len(peaks):
        image_ax.scatter(peaks[:, 1], peaks[:, 0], s=4, c=colours["teal"],
                         linewidths=0)
    image_ax.set_title(tr("{n} reads on {m} nuclei", n=len(peaks),
                          m=result["n_nuclei"]), fontsize=8)
    _picture_axes(image_ax)
    quality = np.asarray(result["read_quality"], float)
    quality = quality[np.isfinite(quality)]
    if quality.size:
        quality_ax.hist(quality, bins=30, color=colours["blue"], alpha=0.8)
        quality_ax.set_xlabel(tr("Read quality (worst base)"), fontsize=7)
        quality_ax.set_ylabel(tr("Reads"), fontsize=7)
    else:
        _message(quality_ax, tr("No reads above the threshold."), colours)
    quality_ax.set_title(tr("Base-call quality"), fontsize=8)
    cells = result["cells"]
    mapped = sum(1 for c in cells if c["guide"] and not c["decoy"])
    decoy = sum(1 for c in cells if c["decoy"])
    other = len(cells) - mapped - decoy
    without = max(0, int(result["n_nuclei"]) - len(cells))
    labels = [tr("Guide"), tr("Decoy"), tr("Unmapped"), tr("No call")]
    values = [mapped, decoy, other, without]
    bars = calls_ax.bar(range(4), values, color=[colours["teal"], colours["blue_hi"],
                                                 colours["navy"], colours["border"]],
                        alpha=0.8)
    for bar, value in zip(bars, values):
        calls_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      str(value), ha="center", va="bottom", fontsize=7,
                      color=colours["fg"])
    calls_ax.set_xticks(range(4), labels, fontsize=7)
    calls_ax.set_ylabel(tr("Nuclei"), fontsize=7)
    calls_ax.set_title(tr("Calls per nucleus ({rule})", rule=result["rule"]),
                       fontsize=8)
    header = [
        (tr("Nucleus"), tr("The nucleus's label in the crop's nucleus mask.")),
        (tr("Guide"), tr("The library guide the nucleus is called with; * "
                         "marks a decoy guide.")),
        (tr("Barcode (commonest read)"), tr(
            "The commonest raw read among the reads that support the guide. It "
            "may differ from the guide by up to ops_read_mismatches bases, "
            "which the match allows.")),
        (tr("Reads (all)"), tr(
            "Every read attributed to the nucleus, including reads that match "
            "no library guide.")),
        (tr("Reads for guide"), tr(
            "Reads that match the called guide, exactly or within the allowed "
            "mismatches. For a pixel-fallback call (marked px), the pixels "
            "that voted for it.")),
        (tr("Vote share"), tr(
            "The called guide's share of the nucleus's guide votes. Only reads "
            "that match a library guide vote, so reads that match none count "
            "in Reads (all) but not in this share; with ops_decode_quality_weighted "
            "each vote is weighted by its read's quality.")),
        (tr("Best read quality"), tr(
            "The highest quality among the reads for the guide; a read's "
            "quality is its worst cycle's margin between the top two bases, "
            "from 0 to 1.")),
    ]
    rows = [[str(c["object_id"]), c["guide"] + (" *" if c["decoy"] else ""),
             c["barcode"], str(c["reads"]),
             str(c["agreeing"]) + (" px" if c.get("pixels") else ""),
             f"{c['fraction']:.2f}", f"{c['quality']:.3f}"]
            for c in sorted(cells, key=lambda c: -c["reads"])]
    reads = list(result["reads"])[:8]
    if reads:
        rows += [[tr("read"), "", code, "", "", "", f"{q:.3f}"]
                 for code, q in zip(reads, result["read_quality"][:8])]
    row, column, crop = result["crop"]
    sample = tr("crop {size} px of tile {site}, {cycles} cycles, read threshold {t}",
                size=crop, site=result["site"], cycles=result["n_cycles"],
                t=result["threshold"])
    return _rgb(figure), header, rows, sample


def _first_column(table, *names: str) -> Optional[str]:
    """The first of ``names`` the table has, or its first numeric column."""
    for name in names:
        if name in table.columns:
            return name
    numeric = [c for c in table.columns if c not in ("site", "label")
               and np.issubdtype(table[c].dtype, np.number)]
    return numeric[0] if numeric else None


def _draw_objects(result: Mapping[str, Any], colours, size):
    """Tab 3: nuclei and cells on the crop and two measured features."""
    figure, (image_ax, first_ax, second_ax) = _figure(colours, size)
    stack, channels = result["stack"], result["channels"]
    dna = _scaled(stack[..., channels.index("DAPI")]) if "DAPI" in channels \
        else _scaled(stack[..., 0])
    body_name = result["cell_channel"]
    body = (_scaled(stack[..., channels.index(body_name)])
            if body_name in channels else np.zeros_like(dna))
    picture = np.clip(_tint(dna, colours["navy"]) + _tint(body, colours["teal"]) * 0.7,
                      0, 1)
    picture[_outline(result["cells"])] = np.array([0.40, 0.70, 1.00])
    picture[_outline(result["nuclei"])] = np.array([0.55, 0.95, 0.90])
    for mask in result.get("layers", {}).values():
        picture[_outline(mask)] = np.array([1.0, 1.0, 1.0])
    image_ax.imshow(picture, interpolation="nearest")
    n_cells = int(len(np.unique(result["cells"])) - 1)
    image_ax.set_title(tr("{n} cells, field {site}", n=max(n_cells, 0),
                          site=result["site"]), fontsize=8)
    _picture_axes(image_ax)
    table = result["table"]
    shown = []
    for ax, names in ((first_ax, ("cell_area",)),
                      (second_ax, ("nucleus_dapi_mean_intensity",
                                   "cell_dapi_mean_intensity"))):
        column = _first_column(table, *names) if len(table) else None
        if column is None:
            _message(ax, tr("No cells were measured."), colours)
            continue
        values = table[column].to_numpy(float)
        ax.hist(values[np.isfinite(values)], bins=25, color=colours["blue"],
                alpha=0.8)
        ax.set_xlabel(column, fontsize=7)
        ax.set_ylabel(tr("Cells"), fontsize=7)
        ax.set_title(column, fontsize=8)
        shown.append(column)
    extra = [c for c in table.columns if c not in ("site", "label", *shown)
             and np.issubdtype(table[c].dtype, np.number)][:3]
    columns = shown + extra
    header = [tr("Cell")] + columns
    rows = [[str(int(r["label"]))] + [f"{float(r[c]):.4g}" for c in columns]
            for _i, r in table.head(_TABLE_ROWS).iterrows()]
    row, column, crop = result["window"]
    sample = tr("crop {size} px of phenotype field {site}, {n} features per cell",
                size=crop, site=result["site"], n=max(0, table.shape[1] - 2))
    return _rgb(figure), header, rows, sample


def _draw_merge(result: Mapping[str, Any], colours, size):
    """Tab 4: matched cells near the sample, matches per field, cells per gene."""
    figure, (pairs_ax, field_ax, gene_ax) = _figure(colours, size)
    align = result["align"]
    cells, objects, merged = result["cells"], result["objects"], result["merged"]
    site = align["crop_site"]
    ty, tx = align["tiles"][site]
    row, column, crop = align["crop"]
    window = (ty + row - crop // 4, ty + row + crop + crop // 4,
              tx + column - crop // 4, tx + column + crop + crop // 4)

    def inside(frame, y: str, x: str):
        """The rows of ``frame`` inside the drawn window."""
        return frame[(frame[y] > window[0]) & (frame[y] < window[1])
                     & (frame[x] > window[2]) & (frame[x] < window[3])]
    if objects is not None and len(objects):
        near = inside(objects, "centroid_y", "centroid_x")
        pairs_ax.scatter(near["centroid_x"], near["centroid_y"], s=5,
                         c=colours["blue"], linewidths=0,
                         label=tr("sequencing nuclei"))
    if len(cells):
        near_cells = inside(cells, "y", "x")
        pairs_ax.scatter(near_cells["x"], near_cells["y"], s=9, facecolors="none",
                         edgecolors=colours["teal"], linewidths=0.6,
                         label=tr("phenotype cells"))
        if merged is not None and len(merged) and objects is not None:
            joined = (near_cells.merge(merged[["site", "label", "object_id"]],
                                       on=["site", "label"])
                      .merge(objects[["object_id", "centroid_y", "centroid_x"]],
                             on="object_id"))
            for one in joined.itertuples():
                pairs_ax.plot([one.x, one.centroid_x], [one.y, one.centroid_y],
                              color=colours["fg"], lw=0.5, alpha=0.7)
    pairs_ax.set_xlim(window[2], window[3])
    pairs_ax.set_ylim(window[1], window[0])
    pairs_ax.set_aspect("equal")
    pairs_ax.set_title(tr("Matched pairs near the sample"), fontsize=8)
    pairs_ax.legend(fontsize=6, loc="lower right", facecolor=colours["face"],
                    edgecolor=colours["border"], labelcolor=colours["fg"])
    _picture_axes(pairs_ax)
    fields = result["per_field"]
    if fields:
        xs = np.arange(len(fields))
        field_ax.bar(xs - 0.2, [f["cells"] for f in fields], 0.4,
                     color=colours["navy"], alpha=0.8, label=tr("cells"))
        field_ax.bar(xs + 0.2, [f["matched"] for f in fields], 0.4,
                     color=colours["teal"], alpha=0.8, label=tr("matched"))
        field_ax.set_xticks(xs, [str(f["site"]) for f in fields], fontsize=7)
        field_ax.set_xlabel(tr("Phenotype field"), fontsize=7)
        field_ax.legend(fontsize=6, facecolor=colours["face"],
                        edgecolor=colours["border"], labelcolor=colours["fg"])
    field_ax.set_title(tr("Cells matched per field"), fontsize=8)
    genes = result["genes"]
    if len(genes) and "n_cells" in genes.columns:
        counts = genes["n_cells"].to_numpy(float)
        gene_ax.hist(counts, bins=np.arange(0.5, max(counts.max(), 1) + 1.5),
                     color=colours["blue"], alpha=0.8)
        gene_ax.set_xlabel(tr("Cells per gene"), fontsize=7)
        gene_ax.set_ylabel(tr("Genes"), fontsize=7)
    else:
        _message(gene_ax, tr("The aggregate phase wrote no gene table."), colours)
    gene_ax.set_title(tr("{n} genes", n=len(genes)), fontsize=8)
    features = [c for c in genes.columns if c not in ("gene", "n_cells", "n_guides")
                and np.issubdtype(genes[c].dtype, np.number)][:3]
    header = [tr("Gene")] + [c for c in ("n_cells", "n_guides") if c in genes.columns] \
        + features
    keys = header[1:]
    ordered = genes.sort_values("n_cells", ascending=False) \
        if "n_cells" in genes.columns else genes
    rows = [[str(r["gene"])] + [f"{float(r[c]):.4g}" for c in keys]
            for _i, r in ordered.head(_TABLE_ROWS).iterrows()]
    report = result.get("report") or {}
    used = result.get("used") or {}
    inputs = (tr("the test data's stored nuclei, calls and cells")
              if used.get("kind") == "test" else
              tr("the run in {path}", path=used.get("path", "")))
    sample = tr("{merged} of {cells} cells matched; inputs: {used}",
                merged=report.get("merged", "?"), cells=report.get("cells", "?"),
                used=inputs)
    return _rgb(figure), header, rows, sample


def _draw_cluster(result: Mapping[str, Any], colours, size):
    """Tab 5: the gene embedding coloured by cluster, and the cluster sizes."""
    figure, (first_ax, second_ax, sizes_ax) = _figure(colours, size)
    components = np.asarray(result["components"], float)
    labels = result["labels"]
    if labels is None:
        colour = [colours["blue"]] * len(components)
    else:
        colour = [_CLUSTER_COLOURS[int(c) % len(_CLUSTER_COLOURS)] for c in labels]
    for ax, (a, b) in ((first_ax, (0, 1)), (second_ax, (0, 2))):
        if components.shape[1] > b:
            ax.scatter(components[:, a], components[:, b], s=10, c=colour,
                       alpha=0.85, linewidths=0)
            ax.set_xlabel(f"PC {a + 1}", fontsize=7)
            ax.set_ylabel(f"PC {b + 1}", fontsize=7)
        else:
            _message(ax, tr("Fewer than three components."), colours)
    first_ax.set_title(tr("{n} genes, {k} components", n=len(components),
                          k=result["n_components"]), fontsize=8)
    second_ax.set_title(tr("Graph {kind}, resolution {r}, seed {seed}",
                           kind=result["kind"], r=result["resolution"],
                           seed=result["seed"]), fontsize=8)
    rows: List[List[str]] = []
    if labels is None:
        _message(sizes_ax, tr("Install the clustering extra (pip install "
                              "spacr[clustering]) to see the clusters."), colours)
    else:
        ids, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        sizes_ax.bar(range(len(ids)), counts[order],
                     color=[_CLUSTER_COLOURS[int(i) % len(_CLUSTER_COLOURS)]
                            for i in ids[order]], alpha=0.8)
        sizes_ax.set_xlabel(tr("Cluster"), fontsize=7)
        sizes_ax.set_ylabel(tr("Genes"), fontsize=7)
        genes = np.asarray(result["genes"])
        for cluster in ids[order]:
            members = genes[np.asarray(labels) == cluster]
            rows.append([str(int(cluster)), str(len(members)),
                         ", ".join(members[:12]) + (" …" if len(members) > 12 else "")])
    sizes_ax.set_title(tr("Cluster sizes"), fontsize=8)
    header = [tr("Cluster"), tr("Genes"), tr("Members")]
    sample = tr("{n} of {total} genes with at least {floor} cells",
                n=len(components), total=result["total_genes"], floor=result["floor"])
    return _rgb(figure), header, rows, sample


_DRAW: Dict[str, Callable[..., Tuple[np.ndarray, List[str], List[List[str]], str]]] = {
    "align": _draw_align,
    "spots": _draw_spots,
    "objects": _draw_objects,
    "merge": _draw_merge,
    "cluster": _draw_cluster,
}


class _StepWorker(QThread):
    """Run one tab's step and draw its picture, off the GUI thread.

    Every exception is caught inside :meth:`run` and emitted as text; a
    cancelled pass emits nothing.
    """

    done = Signal(int, str, object, str)

    def __init__(self, token: int, step: str, form: Dict[str, Any], cache: dict,
                 steps, size: Tuple[int, int], cancel: threading.Event,
                 run_step: Optional[Callable[..., Any]] = None, parent=None):
        """Carry one pass onto a worker thread.

        :param token: the panel's preview generation for this pass.
        :param step: the tab's step.
        :param form: a copy of the form's settings.
        :param cache: the panel's step cache, which only this worker writes
            while it runs.
        :param steps: the setting-to-step map.
        :param size: the picture size in pixels.
        :param cancel: set when the pass is cancelled.
        :param run_step: replaces :func:`spacr.ops_live._run_step`, for tests.
        :param parent: parent object.
        """
        super().__init__(parent)
        self._token, self._step, self._form = token, step, dict(form)
        self._cache, self._steps, self._size = cache, steps, size
        self._cancel = cancel
        self._run_step = run_step
        self._colours = _theme_colours()

    def run(self) -> None:
        """Compute, draw and hand back the tab's picture and table."""
        from spacr import ops_live

        try:
            run_step = self._run_step or ops_live._run_step
            result = run_step(self._form, self._step, self._cache,
                              self._cancel.is_set, self._steps)
            if self._cancel.is_set():
                return
            picture, header, rows, sample = _DRAW[self._step](
                result, self._colours, self._size)
            self.done.emit(self._token, self._step, {
                "picture": picture, "header": header, "rows": rows,
                "sample": sample, "seconds": result.get("seconds"),
                "test_data": result.get("test_data", False)}, "")
        except ops_live._Cancelled:
            return
        except Exception as error:                            # noqa: BLE001
            LOG.info("OPS live %s failed", self._step, exc_info=True)
            try:
                self.done.emit(self._token, self._step, None,
                               f"{type(error).__name__}: {error}")
            except RuntimeError:
                pass


class _StepPage(QWidget):
    """One tab: what the step does, its picture, its table and its sample."""

    def __init__(self, blurb: str, parent=None):
        """Lay out the tab, the same geometry for every step.

        :param blurb: one sentence on what the tab runs, translated.
        :param parent: parent widget.
        """
        super().__init__(parent)
        self.setObjectName("OpsLiveStepPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.blurb = QLabel(blurb, self)
        self.blurb.setWordWrap(True)
        self.blurb.setObjectName("OpsLiveBlurb")
        layout.addWidget(self.blurb)
        self.picture = QLabel(self)
        self.picture.setObjectName("OpsLivePicture")
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setMinimumHeight(220)
        self.picture.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.picture.setText(tr("Not run yet. Switch Live on, or press Run preview."))
        layout.addWidget(self.picture, 3)
        self.table = QTableWidget(0, 0, self)
        self.table.setObjectName("OpsLiveTable")
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(150)
        from .sortable_table import install_sorting

        install_sorting(self.table)
        layout.addWidget(self.table)
        self.sample = QLabel("", self)
        self.sample.setObjectName("OpsLiveSample")
        self.sample.setWordWrap(True)
        layout.addWidget(self.sample)

    def show_result(self, payload: Mapping[str, Any]) -> None:
        """Put one pass's picture, table and sample line on the tab."""
        from ..hidpi import scaled_for
        from .live_preview import numpy_to_qpixmap
        from .sortable_table import table_item

        pixmap = numpy_to_qpixmap(payload["picture"])
        self.picture.setPixmap(scaled_for(pixmap, self.picture,
                                          self.picture.width(), self.picture.height())
                               if self.picture.width() > 50 else pixmap)
        header, rows = payload["header"], payload["rows"][:_TABLE_ROWS]
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(header))
        self.table.setRowCount(len(rows))
        labels = [h[0] if isinstance(h, tuple) else str(h) for h in header]
        self.table.setHorizontalHeaderLabels(labels)
        for column, item in enumerate(header):
            if isinstance(item, tuple):
                cell = self.table.horizontalHeaderItem(column)
                if cell is not None:
                    cell.setToolTip(item[1])
        for r, values in enumerate(rows):
            for c, value in enumerate(values[:len(header)]):
                self.table.setItem(r, c, table_item(str(value)))
        self.table.setSortingEnabled(sorting)
        seconds = payload.get("seconds")
        source = (tr("OPS test data (Funk et al. 2022, plate A, well A1)")
                  if payload.get("test_data") else tr("your data"))
        self.sample.setText(tr("{sample}. {seconds} s on {source}.",
                               sample=payload["sample"],
                               seconds=seconds if seconds is not None else "?",
                               source=source))

    def show_failure(self, text: str) -> None:
        """Say why the pass failed, in the picture's place."""
        self.picture.setPixmap(_empty_pixmap())
        self.picture.setText(text)


def _empty_pixmap():
    """An empty pixmap, so a failure replaces the last picture."""
    from PySide6.QtGui import QPixmap

    return QPixmap()


class _OpsLivePanel(LivePreviewContract, QWidget):
    """The tabbed OPS Live preview.

    :param collect: returns the form's settings.
    :param parent: parent widget.
    :param run_step: replaces :func:`spacr.ops_live._run_step`, for tests.
    """

    PREVIEW_SOURCE_HINT = "Set genotype_source, or get the OPS test data."

    def __init__(self, collect: Callable[[], Mapping[str, Any]], parent=None,
                 *, run_step: Optional[Callable[..., Any]] = None):
        """Build the tabs and the run, cancel and test data buttons."""
        super().__init__(parent)
        self.setObjectName("OpsLivePanel")
        self._collect = collect
        self._run_step = run_step
        self._cache: dict = {}
        self._steps_map: Optional[dict] = None
        self._stale = set(_STEP_ORDER)
        self._live = False
        self._worker: Optional[_StepWorker] = None
        self._retired: List[_StepWorker] = []
        self._cancel_event = threading.Event()
        self._pending: Optional[str] = None
        self.runs: List[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self._status = QLabel("", self)
        self._status.setObjectName("OpsLiveStatus")
        self._status.setWordWrap(True)
        row.addWidget(self._status, 1)
        self._data_btn = QPushButton(tr("Get OPS test data"), self)
        self._data_btn.setObjectName("OpsLiveTestDataButton")
        self._data_btn.setToolTip(tr(
            "Download the OPS end-to-end test data (about 190 MB): four "
            "sequencing tiles with all eleven cycles and the phenotype fields "
            "over them, from Funk et al. 2022. The preview uses it while "
            "genotype_source is empty."))
        self._data_btn.clicked.connect(self._get_test_data)
        row.addWidget(self._data_btn)
        self._run_btn = QPushButton(tr(PREVIEW_RUN_TEXT), self)
        self._run_btn.setObjectName("OpsLiveRunButton")
        self._run_btn.clicked.connect(self.run_preview)
        row.addWidget(self._run_btn)
        self._cancel_btn = QPushButton(tr(PREVIEW_CANCEL_TEXT), self)
        self._cancel_btn.setObjectName("OpsLiveCancelButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_preview)
        row.addWidget(self._cancel_btn)
        layout.addLayout(row)
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("OpsLiveTabs")
        self._tabs.setDocumentMode(True)
        self._pages: Dict[str, _StepPage] = {}
        for step in _STEP_ORDER:
            page = _StepPage(_blurb(step), self._tabs)
            self._pages[step] = page
            index = self._tabs.addTab(page, _title(step).replace("&", "&&"))
            self._tabs.setTabToolTip(index, _blurb(step))
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_DEBOUNCE_MS)
        self._timer.timeout.connect(self._refresh_current)
        self._refresh_data_button()

    def current_step(self) -> str:
        """The step of the tab on view."""
        return _STEP_ORDER[max(0, self._tabs.currentIndex())]

    def show_step(self, step: str) -> None:
        """Bring one step's tab to the front."""
        for index, name in enumerate(_STEP_ORDER):
            if name == step:
                self._tabs.setCurrentIndex(index)

    def page(self, step: str) -> _StepPage:
        """The tab of one step."""
        return self._pages[step]

    def stale_steps(self) -> set:
        """The steps whose shown result no longer matches the settings."""
        return set(self._stale)

    def _step_map(self) -> dict:
        """The setting-to-step map, read from the registry once."""
        if self._steps_map is None:
            from spacr.ops_live import _step_map

            self._steps_map = _step_map()
        return self._steps_map

    def _refresh_data_button(self) -> None:
        """Show the test data button only when there is nothing to preview."""
        try:
            from spacr.ops_live import _test_data_root

            form = dict(self._collect() or {})
            has_source = bool(form.get("genotype_source"))
            self._data_btn.setVisible(not has_source and _test_data_root() is None)
        except Exception:
            self._data_btn.setVisible(False)

    def _preview_blocked_reason(self) -> str:
        """Why no pass can start: no sequencing folder and no test data."""
        try:
            from spacr.ops_live import _test_data_root

            form = dict(self._collect() or {})
        except Exception:
            form = {}
        if form.get("genotype_source") or _test_data_root() is not None:
            return ""
        return tr("Set genotype_source, or get the OPS test data.")

    def set_live(self, on: bool) -> None:
        """Follow the Live switch: run the tab on view when on, stop when off."""
        self._live = bool(on)
        if not self._live:
            self._timer.stop()
            self._pending = None
            if self.preview_running():
                self.cancel_preview()
            return
        if self.current_step() in self._stale:
            self._timer.start()

    def apply_settings(self, settings: Mapping[str, Any]) -> None:
        """Take a whole new set of settings: every tab is stale.

        :param settings: the form's settings; read again from the form at
            each pass, so only the staleness is kept.
        """
        self._stale = set(_STEP_ORDER)
        self._refresh_data_button()
        if self._live:
            self._timer.start()

    def setting_committed(self, key: Optional[str]) -> None:
        """One setting changed on the form: its tab runs again.

        The setting's tab, and every tab that reads from it, are marked
        stale; the setting's own tab is brought forward and runs after the
        debounce. A setting every tab reads (an input folder, the GPU)
        leaves the tab on view in front. Nothing runs while Live is off.

        :param key: the setting, or None for a change of many at once.
        """
        from spacr.ops_live import _STEPS, _downstream

        steps = self._step_map().get(key or "", _STEPS) if key else _STEPS
        hit = set(steps)
        for step in steps:
            hit.update(_downstream(step))
        self._stale |= hit
        if key in (None, "genotype_source", "phenotype_source"):
            self._refresh_data_button()
        if not self._live:
            return
        if len(steps) == 1 and steps[0] != self.current_step():
            blocked = self._tabs.blockSignals(True)
            self.show_step(steps[0])
            self._tabs.blockSignals(blocked)
        if self.current_step() in self._stale:
            self._timer.start()

    def _on_tab_changed(self, _index: int) -> None:
        """A tab brought forward runs when its result is stale."""
        if self._live and self.current_step() in self._stale:
            self._timer.start()

    def _refresh_current(self) -> None:
        """The debounce fired: run the tab on view."""
        if self._live:
            self._start(self.current_step())

    def run_preview(self) -> None:
        """Run the tab on view now, Live or not."""
        self._timer.stop()
        self._start(self.current_step())

    def _start(self, step: str) -> None:
        """Start one pass, or queue it behind the pass in flight.

        A pass for a newer request supersedes the running one: the running
        pass is cancelled at its next step boundary and its answer dropped
        by token, and this one starts when it has stopped.
        """
        if self.preview_running():
            self._cancel_event.set()
            self._run_token = self.preview_token() + 1
            self._pending = step
            return
        if not self.begin_preview():
            return
        self._pending = None
        self._cancel_event = threading.Event()
        self.runs.append(step)
        self.set_preview_status(" · ".join((_title(step),
                                            tr(PREVIEW_RUNNING_MESSAGE))))
        page = self._pages[step]
        width = page.picture.width() if page.picture.width() > 200 else _FIGURE_SIZE[0]
        height = (page.picture.height() if page.picture.height() > 120
                  else _FIGURE_SIZE[1])
        size = (int(width), int(min(height, width / 2.6)))
        worker = _StepWorker(self.preview_token(), step, dict(self._collect() or {}),
                             self._cache, self._step_map(), size,
                             self._cancel_event, self._run_step, self)
        worker.done.connect(self._on_done)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _on_done(self, token: int, step: str, payload, error: str) -> None:
        """Put a pass's result on its tab, unless it was superseded."""
        if self.preview_stale(token):
            return
        page = self._pages[step]
        if error:
            page.show_failure(preview_failure_message(error))
            self.set_preview_status(preview_failure_message(error))
            return
        page.show_result(payload)
        self._stale.discard(step)
        self.set_preview_status(tr("{step}: done in {seconds} s.",
                                   step=_title(step), seconds=payload.get("seconds")))

    def _on_finished(self) -> None:
        """The worker's thread ended: free it, then start what was queued."""
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        if worker is not None:
            self._retired.append(worker)
        self._release_workers()
        self.set_preview_busy(False)
        if self._pending is not None:
            step, self._pending = self._pending, None
            self._start(step)

    def _release_workers(self) -> None:
        """Hand finished workers back to Python so their results are freed."""
        while self._retired:
            old = self._retired.pop()
            try:
                old.wait()
                old.setParent(None)
            except RuntimeError:
                LOG.debug("worker already deleted", exc_info=True)

    def cancel_preview(self) -> bool:
        """Drop the pass in flight and anything queued behind it."""
        self._pending = None
        self._timer.stop()
        self._cancel_event.set()
        return LivePreviewContract.cancel_preview(self)

    def _get_test_data(self) -> None:
        """Download the OPS end-to-end test data, core tier and its references."""
        from spacr.example_archives import (EXAMPLE_ARCHIVES, OPS_E2E_EXAMPLE_REPO,
                                            OPS_E2E_REFERENCE_ARCHIVES,
                                            example_set_folder)
        from ..hf_download import download_chosen_screen_data

        archives = [EXAMPLE_ARCHIVES[OPS_E2E_EXAMPLE_REPO],
                    OPS_E2E_REFERENCE_ARCHIVES["core"]]

        def done(result, error) -> None:
            """Say what happened and run the tab on view when it worked."""
            self._refresh_data_button()
            if result is None:
                self.set_preview_status(tr(
                    "The test data could not be downloaded: {detail}",
                    detail=error or tr("unknown error")))
                return
            self.apply_settings({})
            self.run_preview()

        try:
            download_chosen_screen_data(self, example_set_folder("ops_e2e"),
                                        archives, OPS_E2E_EXAMPLE_REPO, done)
        except Exception as error:                            # noqa: BLE001
            done(None, str(error))

    def shutdown(self) -> None:
        """Cancel the pass, and keep a worker still running out of the panel's hands.

        Cellpose cannot be interrupted, so a pass may outlive the panel by a
        few seconds. Its thread is unparented and kept in ``_ORPHANS`` for
        the rest of the process, rather than waited for on the GUI thread or
        destroyed while running.
        """
        self.cancel_preview()
        for worker in [self._worker, *self._retired]:
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.setParent(None)
                    _ORPHANS.append(worker)
                else:
                    worker.wait()
            except RuntimeError:
                pass
        self._worker = None
        self._retired = []

    def closeEvent(self, event):                              # noqa: N802
        """Let a running pass stop before the panel goes."""
        self.shutdown()
        super().closeEvent(event)


def _build_ops_live_card(host):
    """The OPS Live preview card and its panel, not yet in any layout.

    :param host: the OPS :class:`~spacr.qt.screens.app_screen.AppScreen`.
    :returns: ``(panel, card)``.
    """
    from .card import Card

    card = Card(title=tr("OPS live preview"))
    card.setObjectName("OpsLivePreviewCard")
    model = getattr(host, "_settings_model", None)

    def collect() -> Mapping[str, Any]:
        """The OPS form's settings now."""
        current = getattr(host, "_settings_model", None) or model
        return current.collect() if current is not None else {}
    panel = _OpsLivePanel(collect, card)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(420)
    return panel, card
