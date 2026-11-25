"""The OPS Live preview: one tab per step of the optical pooled screen (alpha).

The tabs are Align & stitch, Spots & genotyping, Objects & phenotype, Merge &
aggregation, and Clustering. While Live is on, the tab on view runs its step
on a small sample with the settings on the form, off the GUI thread, and a
changed setting runs its own tab again. Each tab reuses what the tabs before
it computed. The work itself is in :mod:`spacr.ops_live`.
"""
from __future__ import annotations

import ast
import logging
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QTableWidget,
                               QTabWidget, QVBoxLayout, QWidget)

from ..i18n import tr
from .preview_contract import (PREVIEW_CANCEL_TEXT, PREVIEW_RUN_TEXT,
                               PREVIEW_RUNNING_MESSAGE, LivePreviewContract,
                               preview_failure_message)

LOG = logging.getLogger("spacr.qt.widgets.ops_live_preview")

_STEP_ORDER: Tuple[str, ...] = ("align", "spots", "objects", "merge", "cluster",
                                "voyager3")


def _title(step: str) -> str:
    """A tab's name in the interface language."""
    return {
        "align": tr("Align & stitch"),
        "spots": tr("Spots & genotyping"),
        "objects": tr("Objects & phenotype"),
        "merge": tr("Merge & aggregation"),
        "cluster": tr("Clustering"),
        "voyager3": tr("Voyager3"),
    }[step]


def _blurb(step: str) -> str:
    """One sentence on what a tab runs, in the interface language."""
    return {
        "align": tr("Stitch the block of sequencing tiles of the chosen region, "
                    "place the phenotype fields on it and align every cycle of "
                    "one crop to cycle 1."),
        "spots": tr("Segment the nuclei of one sequencing crop, detect its "
                    "reads, call their bases and give each nucleus the guide "
                    "its reads agree on."),
        "objects": tr("Segment nuclei and cells over the whole genotyped "
                      "region, on every phenotype field placed over it, and "
                      "measure them."),
        "merge": tr("Match phenotype cells to called nuclei, then reduce the "
                    "matched cells to one profile per gene."),
        "cluster": tr("Embed the gene profiles and cluster them at the first "
                      "graph, resolution and seed the settings name."),
        "voyager3": tr("Name the biological role of the largest preview clusters "
                       "from the UniProt facts of their genes, with the AI or "
                       "the enrichment test the Voyager3 settings choose."),
    }[step]


_DEBOUNCE_MS = 600

_ORPHANS: List[QThread] = []

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


def _laptop_mode() -> bool:
    """Whether spaCR runs at the Laptop performance level.

    In laptop mode nothing in the preview refreshes by itself after switching
    Live on; the Refresh view button applies what changed. The same answer
    Mask Generation's Live view reads.
    """
    from .live_preview import _laptop_mode as laptop_mode

    return laptop_mode()


def _colour(value: str) -> np.ndarray:
    """A ``#rrggbb`` colour as three floats from 0 to 1."""
    text = str(value).lstrip("#")
    if len(text) != 6:
        return np.ones(3, np.float32)
    return np.array([int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
                    np.float32)


def _levels(image: np.ndarray, low: float = 1.0, high: float = 99.5
            ) -> Tuple[float, float]:
    """The two percentiles :func:`_scaled` stretches between."""
    image = np.asarray(image, np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    return float(lo), float(hi)


def _scaled(image: np.ndarray, low: float = 1.0, high: float = 99.5,
            levels: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """``image`` stretched between two percentiles to 0..1, for display only.

    :param levels: ``(low, high)`` values to stretch between instead, so
        two pictures of the same stain match.
    """
    image = np.asarray(image, np.float32)
    if not np.isfinite(image).any():
        return np.zeros_like(image)
    lo, hi = levels if levels is not None else _levels(image, low, high)
    return np.clip((np.nan_to_num(image) - lo) / max(hi - lo, 1e-6), 0, 1)


def _tint(image: np.ndarray, colour: str) -> np.ndarray:
    """A 0..1 image drawn in one colour, as RGB."""
    return np.asarray(image, np.float32)[..., None] * _colour(colour)


def _outline(labels: np.ndarray) -> np.ndarray:
    """The boundary pixels of a label mask."""
    from scipy import ndimage

    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(labels.shape, bool)
    grown = ndimage.grey_dilation(labels, size=3)
    shrunk = ndimage.grey_erosion(labels, size=3)
    return (labels > 0) & (grown != shrunk)


def _bytes(rgb: np.ndarray) -> np.ndarray:
    """A 0..1 RGB picture as ``(H, W, 3)`` uint8."""
    return np.ascontiguousarray(np.clip(np.asarray(rgb, np.float32), 0, 1)
                                * 255.0 + 0.5).astype(np.uint8)


def _two_tone(first: np.ndarray, second: np.ndarray,
              colours: Mapping[str, str]) -> np.ndarray:
    """Two planes over one another, the first teal and the second blue.

    Where both are bright the picture is white, so a good alignment shows
    white nuclei and a bad one teal and blue pairs.
    """
    return _bytes(_tint(_scaled(first), colours["teal"])
                  + _tint(_scaled(second), colours["blue"]))


def _ink(picture: np.ndarray, rows, cols, colour: str) -> None:
    """Colour the given pixels of an RGB picture, clipped to its size."""
    rows = np.asarray(rows, int)
    cols = np.asarray(cols, int)
    keep = ((rows >= 0) & (rows < picture.shape[0])
            & (cols >= 0) & (cols < picture.shape[1]))
    picture[rows[keep], cols[keep]] = _colour(colour)


def _polygon(picture: np.ndarray, ys: Sequence[float], xs: Sequence[float],
             colour: str) -> None:
    """Draw a closed outline through ``(ys, xs)`` onto an RGB picture."""
    from skimage.draw import polygon_perimeter

    rows, cols = polygon_perimeter(np.asarray(ys, float), np.asarray(xs, float))
    _ink(picture, rows, cols, colour)


def _box(picture: np.ndarray, top: float, left: float, height: float,
         width: float, colour: str) -> None:
    """Draw a rectangle's outline onto an RGB picture."""
    _polygon(picture, [top, top, top + height, top + height],
             [left, left + width, left + width, left], colour)


def _dots(picture: np.ndarray, points: np.ndarray, colour: str) -> None:
    """Mark each ``(row, column)`` point with a 3 x 3 dot."""
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            _ink(picture, np.round(points[:, 0]) + dy,
                 np.round(points[:, 1]) + dx, colour)


def _histogram(values, title: str, x_label: str, y_label: str,
               empty: str, bins: Any = 30) -> Dict[str, Any]:
    """A histogram graph, or a sentence when there is nothing to count."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"kind": "message", "title": title, "text": empty}
    counts, edges = np.histogram(values, bins=bins)
    return {"kind": "bars", "title": title, "x_label": x_label,
            "y_label": y_label, "x": (edges[:-1] + edges[1:]) / 2.0,
            "heights": counts.astype(float),
            "width": float(np.diff(edges).min()) if len(edges) > 1 else 1.0,
            "colours": None, "ticks": None, "labels": False}


def _draw_align(result: Mapping[str, Any], colours):
    """Tab 1: the stitched block, a seam, the placed field and the cycles.

    :returns: the tab's pictures, graphs, table and sample line.
    """
    step = result["step"]
    top, left = result["origin"]
    levels = _levels(result["mosaic"])
    mosaic = _mosaic_picture(result, colours, result["mosaic"], top, left, step,
                             levels)
    site = result["crop_site"]
    row, column, crop = result["crop"]
    images = [(tr("Stitched tiles (blue), phenotype fields (teal), crop (white)"),
               mosaic)]
    rows: List[List[str]] = []
    seam = result.get("seam")
    if seam:
        first, second = seam["sites"]
        nominal, stitched = seam["nominal"], seam["stitched"]
        images.append((tr("Seam of tiles {a} and {b} on the raster grid "
                          "({dy:+.0f}, {dx:+.0f} px)", a=first, b=second,
                          dy=nominal[0], dx=nominal[1]),
                       _two_tone(*seam["before"], colours)))
        images.append((tr("The same seam after stitching ({dy:+.0f}, {dx:+.0f} px)",
                          dy=stitched[0], dx=stitched[1]),
                       _two_tone(*seam["after"], colours)))
        rows.append([tr("Seam {a}|{b}", a=first, b=second),
                     f"{stitched[0] - nominal[0]:+.1f}",
                     f"{stitched[1] - nominal[1]:+.1f}",
                     tr("stitch shift from the raster grid")])
    placement = result.get("placement")
    after = result.get("placed_after")
    if placement and after:
        source = (tr("the placement seed") if placement.get("source") == "seed"
                  else tr("the raster fit"))
        images.append((tr("Phenotype field {site} (blue) on the sequencing crop (teal) "
                          "before tile-affine placement, at {source}",
                          site=placement["site"], source=source),
                       _two_tone(placement["sbs"], placement["phenotype"], colours)))
        images.append((tr("The same field after tile-affine placement ({affine} "
                          "affine; {dy:+.1f}, {dx:+.1f} px)",
                          affine=tr("tile group") if after["affine"] == "tile"
                          else tr("field"), dy=after["shift"][0], dx=after["shift"][1]),
                       _two_tone(after["sbs"], after["phenotype"], colours)))
        rows.append([tr("Field {site}", site=after["site"]), f"{after['shift'][0]:+.1f}",
                     f"{after['shift'][1]:+.1f}", tr("tile-affine shift from the placement before")])
    elif placement:
        images.append((tr("Phenotype field {site} (blue) on the sequencing crop "
                          "(teal): scale {scale:.3f}, rotation {rotation:+.2f} deg",
                          site=placement["site"], scale=placement["scale"],
                          rotation=placement["rotation"]),
                       _two_tone(placement["sbs"], placement["phenotype"], colours)))
        rows.append([tr("Field {site}", site=placement["site"]), "", "",
                     tr("run merge with ops_align_model 'tile_affine' to compare the "
                        "placement before and after")])
    last = result["last_cycle"]
    images.append((tr("Cycle 1 (teal) and cycle {n} (blue), before", n=last),
                   _two_tone(*result["before"], colours)))
    images.append((tr("After alignment"), _two_tone(*result["after"], colours)))
    shifts = sorted(result["cycle_shifts"].items())
    rows = [[str(cycle), f"{dy:+.0f}", f"{dx:+.0f}", tr("kept")]
            for cycle, (dy, dx) in shifts] + rows
    rows += [[str(c), "", "", tr("refused")] for c in result["refused"]]
    rows += [[str(c), "", "", tr("missing")] for c in result["missing"]]
    header = [tr("Cycle"), tr("Shift y (px)"), tr("Shift x (px)"), tr("State")]
    cycles = np.arange(len(shifts), dtype=float)
    graphs = [{"kind": "grouped", "title": tr("Cycle alignment offsets"),
               "x_label": tr("Cycle"), "y_label": tr("Shift (px)"),
               "ticks": [str(c) for c, _ in shifts], "x": cycles,
               "series": [(tr("Shift y (px)"), [float(v[0]) for _c, v in shifts],
                           colours["navy"]),
                          (tr("Shift x (px)"), [float(v[1]) for _c, v in shifts],
                           colours["teal"])]}]
    sample = tr("{tiles} tiles, {fields} phenotype fields; crop {size} px of tile {site}",
                tiles=len(result["tiles"]),
                fields=sum(1 for f in result["fields"] if f.get("files")),
                size=crop, site=site)
    lite = {key: result.get(key) for key in (
        "tiles", "tile_shape", "origin", "step", "tile_sources", "fields",
        "crop_site", "crop")}

    def render(window, fine: int) -> np.ndarray:
        """The mosaic window ``(top, left, bottom, right)`` in mosaic pixels, at ``fine``."""
        from spacr.ops_live import _mosaic_window

        well = (top + window[0] * step, left + window[1] * step,
                top + window[2] * step, left + window[3] * step)
        plane = _mosaic_window(lite, well, fine)
        return _mosaic_picture(lite, colours, plane, well[0], well[1], fine, levels)
    fields = [int(f["site"]) for f in result["fields"] if f.get("files")]
    return {"images": images, "graphs": graphs, "header": header, "rows": rows,
            "sample": sample, "mosaic": {"render": render, "step": int(step)},
            "regions": result.get("regions") or [sorted(result["tiles"])],
            "region": int(result.get("region") or 0), "region_fields": fields}


def _mosaic_picture(result: Mapping[str, Any], colours, plane: np.ndarray,
                    top: float, left: float, step: float,
                    levels: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """The stitched stain with the tiles, the placed fields and the crop drawn on it.

    One drawing for the whole-block mosaic and for the finer windows a
    zoomed view asks for.

    :param result: the Align & stitch result (or its geometry).
    :param plane: the stain, ``step`` well pixels per pixel.
    :param top: the plane top in the well frame.
    :param left: the plane left in the well frame.
    :param step: well pixels per plane pixel.
    :param levels: the stretch of the whole mosaic, so windows match it.
    :returns: RGB uint8.
    """
    from spacr.ops_live import _ph_to_well

    picture = _bytes(_tint(_scaled(plane, levels=levels), colours["blue"])).astype(
        np.float32) / 255.0
    height, width = result["tile_shape"]
    for _site, (y, x) in result["tiles"].items():
        _box(picture, (y - top) / step, (x - left) / step, height / step,
             width / step, _ON_IMAGE["blue"])
    for info in result["fields"]:
        if not info.get("files"):
            continue
        h, w = info["shape"]
        corners = _ph_to_well(info, [[0, 0], [0, w], [h, w], [h, 0]])
        _polygon(picture, (corners[:, 0] - top) / step,
                 (corners[:, 1] - left) / step, _ON_IMAGE["teal"])
    site = result["crop_site"]
    cy, cx = result["tiles"][site]
    row, column, crop = result["crop"]
    _box(picture, (cy + row - top) / step, (cx + column - left) / step,
         crop / step, crop / step, _ON_IMAGE["text"])
    return _bytes(picture)


def _draw_spots(result: Mapping[str, Any], colours):
    """Tab 2: spots on the crop, read quality and the per-cell calls.

    :returns: the tab's pictures, graphs, table and sample line.
    """
    dapi = _tint(_scaled(result["dapi"]), colours["navy"])
    dapi[_outline(result["nuclei"])] = np.array([0.55, 0.85, 0.85])
    peaks = np.asarray(result["peaks"], float).reshape(-1, 2)
    _dots(dapi, peaks, colours["teal"])
    images = [(tr("{n} reads on {m} nuclei", n=len(peaks), m=result["n_nuclei"]),
               _bytes(dapi))]
    quality = _histogram(result["read_quality"], tr("Base-call quality"),
                         tr("Read quality (worst base)"), tr("Reads"),
                         tr("No reads above the threshold."))
    cells = result["cells"]
    mapped = sum(1 for c in cells if c["guide"] and not c["decoy"])
    decoy = sum(1 for c in cells if c["decoy"])
    other = len(cells) - mapped - decoy
    without = max(0, int(result["n_nuclei"]) - len(cells))
    calls = {"kind": "bars", "title": tr("Calls per nucleus ({rule})",
                                         rule=result["rule"]),
             "x_label": "", "y_label": tr("Nuclei"), "x": np.arange(4.0),
             "heights": np.array([mapped, decoy, other, without], float),
             "width": 0.7,
             "colours": [colours["teal"], colours["blue_hi"], colours["navy"],
                         _CONTROL_GREY],
             "ticks": [tr("Guide"), tr("Decoy"), tr("Unmapped"), tr("No call")],
             "labels": True}
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
    return {"images": images, "graphs": [quality, calls], "header": header,
            "rows": rows, "sample": sample}


def _first_column(table, *names: str) -> Optional[str]:
    """The first of ``names`` the table has, or its first numeric column."""
    for name in names:
        if name in table.columns:
            return name
    numeric = [c for c in table.columns if c not in ("site", "label")
               and np.issubdtype(table[c].dtype, np.number)]
    return numeric[0] if numeric else None


def _object_picture(part: Mapping[str, Any], channels: Sequence[str],
                    body_name: str, colours) -> np.ndarray:
    """One phenotype crop with its nuclei, cells and extra objects outlined."""
    stack = part["stack"]
    dna = _scaled(stack[..., channels.index("DAPI")]) if "DAPI" in channels \
        else _scaled(stack[..., 0])
    body = (_scaled(stack[..., channels.index(body_name)])
            if body_name in channels else np.zeros_like(dna))
    picture = np.clip(_tint(dna, colours["navy"]) + _tint(body, colours["teal"]) * 0.7,
                      0, 1)
    picture[_outline(part["cells"])] = np.array([0.40, 0.70, 1.00])
    picture[_outline(part["nuclei"])] = np.array([0.55, 0.95, 0.90])
    for mask in (part.get("layers") or {}).values():
        picture[_outline(mask)] = np.array([1.0, 1.0, 1.0])
    return _bytes(picture)


def _draw_objects(result: Mapping[str, Any], colours):
    """Tab 3: nuclei and cells over the genotyped region and two measured features.

    One picture per phenotype field over the region.

    :returns: the tab's pictures, graphs, table and sample line.
    """
    channels = result["channels"]
    body_name = result["cell_channel"]
    parts = result.get("parts") or [result]
    images = []
    for part in parts:
        n_cells = int(len(np.unique(part["cells"])) - 1)
        images.append((tr("{n} cells, field {site}", n=max(n_cells, 0),
                          site=part["site"]),
                       _object_picture(part, channels, body_name, colours)))
    table = result["table"]
    shown: List[str] = []
    graphs = []
    for names in (("cell_area",), ("nucleus_dapi_mean_intensity",
                                   "cell_dapi_mean_intensity")):
        column = _first_column(table, *names) if len(table) else None
        if column is None:
            graphs.append({"kind": "message", "title": "",
                           "text": tr("No cells were measured.")})
            continue
        graphs.append(_histogram(table[column].to_numpy(float), column, column,
                                 tr("Cells"), tr("No cells were measured."),
                                 bins=25))
        shown.append(column)
    extra = [c for c in table.columns if c not in ("site", "label", *shown)
             and np.issubdtype(table[c].dtype, np.number)][:3]
    columns = shown + extra
    with_site = "site" in table.columns and len(parts) > 1
    header = ([tr("Field")] if with_site else []) + [tr("Cell")] + columns
    rows = [([str(int(r["site"]))] if with_site else []) + [str(int(r["label"]))]
            + [f"{float(r[c]):.4g}" for c in columns]
            for _i, r in table.head(_TABLE_ROWS).iterrows()]
    sites = ", ".join(str(p["site"]) for p in parts)
    sbs = result.get("sbs_crop")
    if sbs is not None:
        sample = tr("the genotyped region ({size} px of tile {tile}) on phenotype "
                    "fields {sites}, {n} cells, {features} features per cell",
                    size=sbs[2], tile=result.get("sbs_site"), sites=sites,
                    n=len(table), features=max(0, table.shape[1] - 2))
    else:
        window = result["window"]
        sample = tr("crop {size} px of phenotype field {site}, {n} features per cell",
                    size=window[2], site=result["site"],
                    n=max(0, table.shape[1] - 2))
    return {"images": images, "graphs": graphs, "header": header, "rows": rows,
            "sample": sample}


def _inside(frame, window, y: str, x: str):
    """The rows of ``frame`` whose ``(y, x)`` lie inside ``window``.

    :param window: ``(top, bottom, left, right)`` in the well frame.
    """
    return frame[(frame[y] > window[0]) & (frame[y] < window[1])
                 & (frame[x] > window[2]) & (frame[x] < window[3])]


_SHORT_FEATURES = {"cell_area": "area", "cell_eccentricity": "eccentr.",
                   "cell_dapi_mean_intensity": "DAPI", "cell_solidity": "solidity",
                   "mitotic_index": "mitotic"}


def _merge_points(result: Mapping[str, Any]):
    """The phenotype cells of a merge, each marked matched or not, and the matched pairs.

    :returns: ``(cells, joined)``: the cells with a boolean ``matched``
        column, and the matched cells with their nucleus centroids.
    """
    import pandas as pd

    cells, merged, objects = result["cells"], result["merged"], result["objects"]
    if cells is None or not len(cells):
        return pd.DataFrame(columns=["site", "label", "y", "x", "matched"]), None
    keys = set()
    if merged is not None and len(merged):
        keys = set(zip(merged["site"].astype(int), merged["label"].astype(int)))
    cells = cells.assign(matched=[(int(s), int(l)) in keys
                                  for s, l in zip(cells["site"], cells["label"])])
    joined = None
    if merged is not None and len(merged) and objects is not None and len(objects):
        joined = (cells.merge(merged[["site", "label", "object_id"]], on=["site", "label"])
                  .merge(objects[["object_id", "centroid_y", "centroid_x"]], on="object_id"))
    return cells, joined


def _group_spec(title: str, y_label: str, groups) -> Dict[str, Any]:
    """A graph of one value per member in a few groups, each labelled with its n.

    :param groups: ``(name, values, colour)``; a group of fewer than 3 is
        drawn as its points, 3 to 20 as a bar with every point, and more
        than 20 as a box.
    """
    named = []
    for name, values, colour in groups:
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        named.append((tr("{name} (n = {n})", name=name, n=len(values)), values, colour))
    return {"kind": "groups", "title": title, "x_label": "", "y_label": y_label,
            "groups": named}


def _draw_merge(result: Mapping[str, Any], colours):
    """Tab 4: how the phenotype cells matched the called nuclei, and how the genes were covered.

    Nine graphs, in this order: the share matched per
    field, the map of matched cells, the genes kept at each cells-per-gene
    floor, the matched and unmatched cells per field, the matching distance,
    the cells per gene by rank, the guides per gene, the profiles of the
    genes with most cells, and one feature for the non-targeting guides
    against the genes.

    :returns: the tab's pictures (none), graphs, table and sample line.
    """
    cells, joined = _merge_points(result)
    objects, genes, fields = result["objects"], result["genes"], result["per_field"]
    prefix = str(result.get("control_prefix") or "nontargeting").lower()
    names = genes["gene"].astype(str).str.lower() if "gene" in genes.columns else None
    control = (names.str.startswith(prefix).to_numpy() if names is not None
               else np.zeros(len(genes), bool))
    graphs: List[Dict[str, Any]] = []
    sites = [str(f["site"]) for f in fields]
    total = np.array([f["cells"] for f in fields], float)
    matched = np.array([f["matched"] for f in fields], float)
    if fields:
        share = 100.0 * matched / np.maximum(total, 1.0)
        whole = 100.0 * matched.sum() / max(total.sum(), 1.0)
        graphs.append({"kind": "bars", "title": tr("Cells matched per field"),
                       "x_label": tr("Phenotype field"), "y_label": tr("Cells matched (%)"),
                       "x": np.arange(len(sites), dtype=float), "heights": share,
                       "width": 0.6, "colours": [colours["blue"]] * len(sites),
                       "ticks": sites, "labels": False, "y_max": 100.0,
                       "hline": (whole, tr("all fields"))})
    else:
        graphs.append({"kind": "message", "title": tr("Cells matched per field"),
                       "text": tr("No phenotype field was matched.")})
    sets = []
    if len(cells):
        top, bottom = cells["y"].min() - 50, cells["y"].max() + 50
        left, right = cells["x"].min() - 50, cells["x"].max() + 50
        if objects is not None and len(objects):
            near = _inside(objects, (top, bottom, left, right), "centroid_y", "centroid_x")
            sets.append((tr("sequencing nuclei"), near["centroid_x"].to_numpy(float),
                         near["centroid_y"].to_numpy(float), "#c8c8c8", False))
        lost = cells[~cells["matched"]]
        kept = cells[cells["matched"]]
        sets.append((tr("cells, not matched"), lost["x"].to_numpy(float),
                     lost["y"].to_numpy(float), _CONTROL_GREY, False))
        sets.append((tr("cells, matched"), kept["x"].to_numpy(float),
                     kept["y"].to_numpy(float), colours["blue"], False))
    graphs.append({"kind": "scatter", "title": tr("Matched and unmatched cells"),
                   "x_label": tr("x (px)"), "y_label": tr("y (px)"), "invert_y": True,
                   "equal": True, "sets": sets, "segments": None, "size": 3})
    counts = (genes["n_cells"].to_numpy(float) if "n_cells" in genes.columns
              else np.zeros(0))
    targeting = counts[~control] if len(counts) else counts
    floors = np.arange(0, 21, dtype=float)
    kept_share = [100.0 * float(np.mean(targeting >= f)) if len(targeting) else 0.0
                  for f in floors]
    graphs.append({"kind": "line", "title": tr("Genes kept at each minimum of cells"),
                   "x_label": tr("Minimum cells per gene (ops_profile_min_cells)"),
                   "y_label": tr("Genes kept (%)"),
                   "series": [("", floors, np.asarray(kept_share), colours["blue"])],
                   "y_min": 0.0, "y_max": 100.0, "markers": True})
    if fields:
        graphs.append({"kind": "stacked", "title": tr("Cells per field"),
                       "x_label": tr("Phenotype field"), "y_label": tr("Cells"),
                       "x": np.arange(len(sites), dtype=float), "ticks": sites,
                       "layers": [(tr("matched to a nucleus"), matched, colours["blue"]),
                                  (tr("not matched"), total - matched, _CONTROL_GREY)]})
    if joined is not None and len(joined):
        distance = np.hypot(joined["y"] - joined["centroid_y"],
                            joined["x"] - joined["centroid_x"]).to_numpy(float)
        graphs.append(_histogram(distance, tr("Distance from cell to its nucleus"),
                                 tr("Cell to nucleus distance (px)"), tr("Matched cells"),
                                 tr("No cell was matched."), bins=30))
    else:
        graphs.append({"kind": "message", "title": tr("Distance from cell to its nucleus"),
                       "text": tr("No cell was matched.")})
    if len(counts):
        order = np.argsort(-counts, kind="stable")
        rank = np.arange(1, len(counts) + 1, dtype=float)
        ranked_control = control[order]
        graphs.append({"kind": "line", "title": tr("Cells per gene, genes ranked"),
                       "x_label": tr("Gene rank"), "y_label": tr("Cells per gene"),
                       "series": [(tr("all genes"), rank, counts[order], _CONTROL_GREY)],
                       "points": [(tr("non-targeting"), rank[ranked_control],
                                   counts[order][ranked_control], colours["navy"])],
                       "y_min": 0.0})
    if "n_guides" in genes.columns and (~control).any():
        guides = genes["n_guides"].to_numpy(float)[~control]
        values, numbers = np.unique(guides.astype(int), return_counts=True)
        graphs.append({"kind": "bars", "title": tr("Guides per gene"),
                       "x_label": tr("Guides per gene"), "y_label": tr("Genes"),
                       "x": np.arange(len(values), dtype=float),
                       "heights": numbers.astype(float), "width": 0.6,
                       "colours": [colours["blue"]] * len(values),
                       "ticks": [str(v) for v in values], "labels": False})
    features = [c for c in ("cell_area", "cell_eccentricity", "cell_dapi_mean_intensity",
                            "cell_solidity", "mitotic_index") if c in genes.columns]
    if features and (~control).any():
        top = genes[~control]
        if "n_cells" in top.columns:
            top = top.sort_values("n_cells", ascending=False, kind="stable")
        top = top.head(20)
        everything = genes[features].to_numpy(float)
        centre = np.nanmedian(everything, axis=0)
        spread = np.nanstd(everything, axis=0) + 1e-9
        graphs.append({"kind": "heatmap",
                       "title": tr("Profiles of the 20 genes with most cells"),
                       "x_label": tr("Feature"), "y_label": tr("Gene"),
                       "values": (top[features].to_numpy(float) - centre) / spread,
                       "columns": [_SHORT_FEATURES.get(f, f) for f in features],
                       "rows": top["gene"].astype(str).tolist(),
                       "scale": tr("z against all genes")})
    feature = next((c for c in ("cell_area", "cell_dapi_mean_intensity") if c in genes.columns),
                   None)
    if feature is not None:
        graphs.append(_group_spec(
            tr("{feature} per gene", feature=feature.replace("_", " ")),
            feature.replace("_", " "),
            [(tr("non-targeting"), genes[feature].to_numpy(float)[control], _CONTROL_GREY),
             (tr("genes"), genes[feature].to_numpy(float)[~control], colours["blue"])]))
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
    if used.get("kind") == "profiles":
        inputs = tr("the test data's stored nuclei, calls and cells; "
                    "genes from the OPS profiles archive (wells A1-A3)")
    elif used.get("kind") == "test":
        inputs = tr("the test data's stored nuclei, calls and cells")
    else:
        inputs = tr("the run in {path}", path=used.get("path", ""))
    sample = tr("{merged} of {cells} cells matched; inputs: {used}",
                merged=report.get("merged", "?"), cells=report.get("cells", "?"),
                used=inputs)
    return {"images": [], "graphs": graphs, "header": header, "rows": rows,
            "sample": sample}


def _draw_cluster(result: Mapping[str, Any], colours):
    """Tab 5: the gene embedding coloured by cluster, and the cluster sizes.

    :returns: the tab's pictures (none), graphs, table and sample line.
    """
    components = np.asarray(result["components"], float)
    labels = result["labels"]
    if labels is None:
        colour = [colours["blue"]] * len(components)
    else:
        colour = [_CLUSTER_COLOURS[int(c) % len(_CLUSTER_COLOURS)] for c in labels]
    graphs: List[Dict[str, Any]] = []
    titles = (tr("{n} genes, {k} components", n=len(components),
                 k=result["n_components"]),
              tr("Graph {kind}, resolution {r}, seed {seed}", kind=result["kind"],
                 r=result["resolution"], seed=result["seed"]))
    tile = {"genes": [str(g) for g in result["genes"]],
            "clusters": None if labels is None else [int(c) for c in labels],
            "n_cells": result.get("n_cells"), "components": components}
    for title, (a, b) in zip(titles, ((0, 1), (0, 2))):
        if components.ndim == 2 and components.shape[1] > b:
            graphs.append({"kind": "scatter", "title": title,
                           "x_label": f"PC {a + 1}", "y_label": f"PC {b + 1}",
                           "invert_y": False, "equal": False, "segments": None,
                           "keys": tile["genes"], "tile": tile,
                           "sets": [("", components[:, a], components[:, b],
                                     colour, False)]})
        else:
            graphs.append({"kind": "message", "title": title,
                           "text": tr("Fewer than three components.")})
    rows: List[List[str]] = []
    if labels is None:
        graphs.append({"kind": "message", "title": tr("Cluster sizes"),
                       "text": tr("Install the clustering extra (pip install "
                                  "spacr[clustering]) to see the clusters.")})
    else:
        ids, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        graphs.append({"kind": "bars", "title": tr("Cluster sizes"),
                       "x_label": tr("Cluster"), "y_label": tr("Genes"),
                       "x": np.arange(len(ids), dtype=float),
                       "heights": counts[order].astype(float), "width": 0.7,
                       "colours": [_CLUSTER_COLOURS[int(i) % len(_CLUSTER_COLOURS)]
                                   for i in ids[order]],
                       "ticks": [str(int(i)) for i in ids[order]], "labels": False})
        genes = np.asarray(result["genes"])
        for cluster in ids[order]:
            members = genes[np.asarray(labels) == cluster]
            rows.append([str(int(cluster)), str(len(members)),
                         ", ".join(members[:12]) + (" …" if len(members) > 12 else "")])
    string = result.get("string")
    if labels is not None and string:
        graphs[-1]["title"] = tr(
            "Cluster sizes; STRING pairs in clusters: precision {p:.2f}, recall {r:.2f}",
            p=float(string.get("precision", 0.0)), r=float(string.get("recall", 0.0)))
    header = [tr("Cluster"), tr("Genes"), tr("Members")]
    sample = tr("{n} of {total} genes with at least {floor} cells",
                n=len(components), total=result["total_genes"], floor=result["floor"])
    return {"images": [], "graphs": graphs, "header": header, "rows": rows,
            "sample": sample}


_CONFIDENCE_ORDER = ("high", "medium", "low", "none")


def _draw_voyager3(result: Mapping[str, Any], colours):
    """Tab 6: the role Voyager3 names for each cluster, how well its genes bear it out, and the candidates.

    :returns: the tab's pictures (none), graphs, table and sample line.
    """
    clusters = result["clusters"]
    real = clusters[clusters["kind"].astype(str) == "real"] if "kind" in clusters.columns \
        else clusters
    shuffled = clusters[clusters["kind"].astype(str) == "shuffled"] \
        if "kind" in clusters.columns else clusters.iloc[0:0]
    graphs: List[Dict[str, Any]] = []
    if len(real):
        words = real["confidence_rule"].astype(str).str.lower()
        counts = [float((words == w).sum()) for w in _CONFIDENCE_ORDER]
        graphs.append({"kind": "bars", "title": tr("Confidence of the named roles"),
                       "x_label": tr("Confidence (share of genes cited as support)"),
                       "y_label": tr("Clusters"), "x": np.arange(4.0),
                       "heights": np.array(counts), "width": 0.6,
                       "colours": [colours["blue"], colours["teal"], colours["navy"],
                                   _CONTROL_GREY],
                       "ticks": [tr("high"), tr("medium"), tr("low"), tr("none")],
                       "labels": True})
        support = 100.0 * real["support_fraction"].astype(float).to_numpy()
        series = [(tr("clusters"), list(support), colours["blue"])]
        if len(shuffled):
            matched = shuffled.set_index("cluster")["support_fraction"].astype(float)
            series.append((tr("shuffled genes (ref.)"),
                           [100.0 * float(matched.get(c, np.nan))
                            for c in real["cluster"]], _CONTROL_GREY))
        graphs.append({"kind": "grouped", "title": tr("Genes cited as support"),
                       "x_label": tr("Cluster"), "y_label": tr("Genes cited (%)"),
                       "ticks": [str(c) for c in real["cluster"]],
                       "x": np.arange(len(real), dtype=float), "series": series})
    else:
        graphs.append({"kind": "message", "title": tr("Confidence of the named roles"),
                       "text": tr("Voyager3 named no cluster.")})
    header = [tr("Cluster"), tr("Genes"), tr("Role"), tr("Confidence"),
              tr("Genes cited (%)"), tr("Supporting genes"), tr("Candidates")]
    rows = [[str(r["cluster"]), str(r["n_genes"]), str(r["biological_role"]),
             str(r["confidence_rule"]), f"{100.0 * float(r['support_fraction']):.1f}",
             str(r["supporting_genes"]), str(r["candidates"])]
            for _i, r in real.iterrows()]
    method = result.get("method") or "ai"
    used = (tr("the enrichment test") if method == "enrichment"
            else tr("{provider} {model}", provider=result.get("provider") or tr("the AI"),
                    model=result.get("model") or ""))
    sample = tr("{n} of {total} clusters named by {used}", n=result.get("annotated", len(real)),
                total=result.get("total_clusters", len(real)), used=used.strip())
    return {"images": [], "graphs": graphs, "header": header, "rows": rows,
            "sample": sample}


_DRAW: Dict[str, Callable[..., Dict[str, Any]]] = {
    "align": _draw_align,
    "spots": _draw_spots,
    "objects": _draw_objects,
    "merge": _draw_merge,
    "cluster": _draw_cluster,
    "voyager3": _draw_voyager3,
}


def _not_ready_text(step: str, reason: str, detail: Mapping[str, Any]) -> str:
    """What a tab is missing and what to run first, in the interface language.

    :param step: the tab whose pass found its input missing.
    :param reason: :attr:`spacr.ops_live._NotReady.reason`.
    :param detail: :attr:`spacr.ops_live._NotReady.detail`.
    """
    texts = {
        "run": tr("Merge & aggregation needs a finished OPS run in dst_root "
                  "({path}), and {missing} is not there yet. Run the OPS module "
                  "on this data first (its stitch to ph_measure phases write "
                  "measurements.db and the phenotype cell table), or clear "
                  "genotype_source to preview on the OPS test data.",
                  path=detail.get("path", ""), missing=detail.get("missing", "")),
        "rows": tr("The run in {path} has no {table} rows for this well yet. Run "
                   "its decode and objects phases first.",
                   path=detail.get("path", ""), table=detail.get("table", "")),
        "fields": tr("No phenotype field is placed on the sequencing tiles, so "
                     "there is nothing to merge yet. Set phenotype_source, then "
                     "run the Align & stitch tab."),
        "phenotype": tr("No phenotype_source is set, so there is no phenotype "
                        "field to segment. Set phenotype_source, or clear "
                        "genotype_source to preview on the OPS test data."),
        "genes": tr("The aggregate phase wrote no gene table. Run the Merge & "
                    "aggregation tab first and check what it says."),
        "clusters": tr("Voyager3 names the clusters of the Clustering tab, and it "
                       "made no cluster with 2 or more genes. Run the Clustering "
                       "tab first and check what it says."),
        "few_genes": tr("Only {n} genes have at least {floor} cells "
                        "(ops_profile_min_cells); clustering needs 5 or more. "
                        "Lower ops_profile_min_cells or preview on more data.",
                        n=detail.get("n", 0), floor=detail.get("floor", 0)),
    }
    text = texts.get(reason) or tr("This tab's input is not ready yet.")
    if step == "cluster" and reason not in ("genes", "few_genes"):
        return tr("Clustering reads the genes of Merge & aggregation, which "
                  "cannot run yet. {why}", why=text)
    return text


class _StepWorker(QThread):
    """Run one tab's step and draw its pictures and graphs, off the GUI thread.

    Every exception is caught inside :meth:`run` and emitted as text; a
    missing input is emitted as a notice; a cancelled pass emits nothing.
    """

    done = Signal(int, str, object, str)

    def __init__(self, token: int, step: str, form: Dict[str, Any], cache: dict,
                 steps, cancel: threading.Event,
                 run_step: Optional[Callable[..., Any]] = None, parent=None):
        """Carry one pass onto a worker thread.

        :param token: the panel's preview generation for this pass.
        :param step: the tab's step.
        :param form: a copy of the settings the pass runs with.
        :param cache: the panel's step cache, which only this worker writes
            while it runs.
        :param steps: the setting-to-step map.
        :param cancel: set when the pass is cancelled.
        :param run_step: replaces :func:`spacr.ops_live._run_step`, for tests.
        :param parent: parent object.
        """
        super().__init__(parent)
        self._token, self._step, self._form = token, step, dict(form)
        self._cache, self._steps = cache, steps
        self._cancel = cancel
        self._run_step = run_step
        self._colours = _theme_colours()

    def run(self) -> None:
        """Compute, draw and hand back the tab's pictures, graphs and table."""
        from spacr import ops_live

        try:
            run_step = self._run_step or ops_live._run_step
            result = run_step(self._form, self._step, self._cache,
                              self._cancel.is_set, self._steps)
            if self._cancel.is_set():
                return
            payload = dict(_DRAW[self._step](result, self._colours))
            payload.update({"seconds": result.get("seconds"),
                            "test_data": result.get("test_data", False)})
            self.done.emit(self._token, self._step, payload, "")
        except ops_live._Cancelled:
            return
        except ops_live._NotReady as missing:
            LOG.info("OPS live %s is not ready: %s", self._step, missing)
            try:
                self.done.emit(self._token, self._step,
                               {"not_ready": missing.reason,
                                "detail": dict(missing.detail)}, "")
            except RuntimeError:
                pass
        except Exception as error:                            # noqa: BLE001
            LOG.info("OPS live %s failed", self._step, exc_info=True)
            try:
                self.done.emit(self._token, self._step, None,
                               f"{type(error).__name__}: {error}")
            except RuntimeError:
                pass


def _graph_ink(widget: QWidget) -> Dict[str, str]:
    """The text, muted and rule colours a graph is drawn in, from the theme.

    Read from the theme on screen (:func:`spacr.qt.theme.active_palette`),
    as every themed widget is, so a graph reads like the page around it in
    the light and the dark theme; the plot itself has no background.

    :param widget: the graph; its palette is the fallback without a theme.
    """
    from PySide6.QtGui import QPalette

    try:
        from ..theme import active_palette

        palette = active_palette()
        return {"fg": palette["fg"], "muted": palette["fg_muted"],
                "border": palette["border"]}
    except Exception:
        qt = widget.palette()
        fg = qt.color(QPalette.WindowText).name()
        return {"fg": fg, "muted": fg, "border": fg}


_ALPHA = 0.6

_CONTROL_GREY = "#8c8c8c"


class _QtGraph(QWidget):
    """One Qt-native graph of a tab, drawn with pyqtgraph.

    It shows a graph spec from a tab's draw function: bars, grouped bars, a
    scatter with optional segments, or a sentence in the graph's place. A
    scatter whose spec names its ``keys`` reports the clicked point through
    :attr:`point_clicked`.
    """

    point_clicked = Signal(int)

    def __init__(self, parent=None):
        """Lay out the title, the plot and the sentence shown instead of it."""
        super().__init__(parent)
        self.setObjectName("OpsLiveGraph")
        from .fast_plots import HAVE_PYQTGRAPH, pg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.title = QLabel("", self)
        self.title.setObjectName("OpsLiveGraphTitle")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.key = QLabel("", self)
        self.key.setObjectName("OpsLiveGraphKey")
        self.key.setWordWrap(True)
        self.key.setTextFormat(Qt.RichText)
        self.key.setContentsMargins(0, 0, 0, 8)
        self.key.hide()
        layout.addWidget(self.key)
        self.plot = pg.PlotWidget(self) if HAVE_PYQTGRAPH else None
        if self.plot is not None:
            self.plot.setObjectName("OpsLiveGraphPlot")
            self.plot.setMinimumHeight(150)
            layout.addWidget(self.plot, 1)
        self.message = QLabel("", self)
        self.message.setObjectName("OpsLiveGraphMessage")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setMinimumHeight(60)
        self.message.hide()
        layout.addWidget(self.message, 1)
        self.spec: Optional[Mapping[str, Any]] = None
        self.scatter_items: list = []

    def show_spec(self, spec: Mapping[str, Any], colours: Mapping[str, str]) -> None:
        """Draw one graph spec.

        :param spec: from a tab's draw function; its ``kind`` is ``bars``,
            ``grouped``, ``scatter`` or ``message``.
        :param colours: from :func:`_theme_colours`.
        """
        self.spec = spec
        self.scatter_items = []
        self._key_entries: List[str] = []
        self.key.hide()
        self.title.setText(str(spec.get("title") or ""))
        if spec.get("kind") == "message" or self.plot is None:
            if self.plot is not None:
                self.plot.hide()
            self.message.setText(str(spec.get("text") or tr(
                "Install pyqtgraph to see this graph.")))
            self.message.show()
            return
        self.message.hide()
        from .fast_plots import pg

        old = self.plot
        plot = pg.PlotWidget(self)
        plot.setObjectName("OpsLiveGraphPlot")
        plot.setMinimumHeight(150)
        self.layout().replaceWidget(old, plot)
        old.hide()
        old.deleteLater()
        self.plot = plot
        ink = _graph_ink(self)
        self.ink = ink
        plot.setBackground(None)
        try:
            from ..theme import make_transparent

            make_transparent(plot, plot.viewport())
        except Exception:
            LOG.debug("the graph keeps its own background", exc_info=True)
        for name in ("bottom", "left"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(ink["border"]))
            axis.setTextPen(pg.mkPen(ink["muted"]))
            axis.setTicks(None)
        plot.setLabel("bottom", str(spec.get("x_label") or ""), color=ink["fg"])
        plot.setLabel("left", str(spec.get("y_label") or ""), color=ink["fg"])
        colours = dict(colours, fg=ink["fg"], muted=ink["muted"],
                       border=ink["border"])
        view = plot.getViewBox()
        view.invertY(bool(spec.get("invert_y")))
        view.setAspectLocked(bool(spec.get("equal")))
        kind = spec.get("kind")
        if kind == "bars":
            self._bars(spec, colours, pg)
        elif kind == "grouped":
            self._grouped(spec, colours, pg)
        elif kind == "scatter":
            self._scatter(spec, colours, pg)
        elif kind == "line":
            self._line(spec, colours, pg)
        elif kind == "stacked":
            self._stacked(spec, colours, pg)
        elif kind == "heatmap":
            self._heatmap(spec, colours, pg)
        elif kind == "groups":
            self._groups(spec, colours, pg)
        hline = spec.get("hline")
        if hline is not None:
            level, name = hline
            line = pg.InfiniteLine(pos=float(level), angle=0, pen=pg.mkPen(
                colours["fg"], width=1, style=Qt.DotLine))
            self.plot.addItem(line)
            self._key(str(name), colours["fg"], line=True)
        view.autoRange()
        if kind in ("bars", "grouped", "stacked"):
            self._floor_at_zero(spec, view)
        low, high = spec.get("y_min"), spec.get("y_max")
        if low is not None or high is not None:
            current = view.viewRange()[1]
            view.setYRange(float(low if low is not None else current[0]),
                           float(high if high is not None else current[1]), padding=0)
        if kind == "groups":
            view.setXRange(-0.7, len(spec.get("groups") or ()) - 0.3, padding=0)

    def _floor_at_zero(self, spec: Mapping[str, Any], view) -> None:
        """Start a bar graph's y axis at 0, with no room below it.

        :param spec: the bars or grouped bars drawn.
        :param view: the plot's view box.
        """
        if spec.get("kind") == "grouped":
            values = [float(v) for _name, series, _c in spec["series"] for v in series]
        elif spec.get("kind") == "stacked":
            values = list(np.sum([np.asarray(v, float) for _n, v, _c in spec["layers"]],
                                 axis=0))
        else:
            values = [float(v) for v in np.asarray(spec["heights"], float)]
        top = max([v for v in values if np.isfinite(v)] + [0.0])
        view.setYRange(0.0, top * 1.08 if top > 0 else 1.0, padding=0)

    def _ticks(self, spec: Mapping[str, Any], pg) -> None:
        """Name the bottom axis's ticks after the spec's categories."""
        ticks = spec.get("ticks")
        if ticks:
            positions = np.asarray(spec["x"], float)
            self.plot.getAxis("bottom").setTicks(
                [[(float(x), str(label)) for x, label in zip(positions, ticks)]])

    def _bars(self, spec: Mapping[str, Any], colours, pg) -> None:
        """Bars of one series, each optionally its own colour and value."""
        heights = np.asarray(spec["heights"], float)
        brushes = [pg.mkBrush(c) for c in spec["colours"]] if spec.get("colours") \
            else [pg.mkBrush(colours["blue"])] * len(heights)
        bars = pg.BarGraphItem(x=np.asarray(spec["x"], float), height=heights,
                               width=float(spec.get("width") or 0.8),
                               brushes=brushes, pen=pg.mkPen(None))
        bars.setOpacity(_ALPHA)
        self.plot.addItem(bars)
        if spec.get("labels"):
            for x, value in zip(np.asarray(spec["x"], float), heights):
                text = pg.TextItem(str(int(value)), color=colours["fg"],
                                   anchor=(0.5, 1.0))
                text.setPos(float(x), float(value))
                self.plot.addItem(text)
        self._ticks(spec, pg)

    def _grouped(self, spec: Mapping[str, Any], colours, pg) -> None:
        """Side-by-side bars of several series over the same categories."""
        x = np.asarray(spec["x"], float)
        series = list(spec["series"])
        width = 0.8 / max(1, len(series))
        legend = self.plot.addLegend(offset=(-4, 4), labelTextColor=colours["fg"])
        for index, (name, values, colour) in enumerate(series):
            shift = (index - (len(series) - 1) / 2.0) * width
            bars = pg.BarGraphItem(x=x + shift, height=np.asarray(values, float),
                                   width=width, brush=pg.mkBrush(colour),
                                   pen=pg.mkPen(None))
            bars.setOpacity(_ALPHA)
            self.plot.addItem(bars)
            legend.addItem(bars, str(name))
        self._ticks(spec, pg)

    def _scatter(self, spec: Mapping[str, Any], colours, pg) -> None:
        """Point sets, filled or hollow, and optional pair segments."""
        segments = spec.get("segments")
        if segments is not None and len(segments[0]):
            x0, y0, x1, y1, colour = segments
            xs = np.column_stack([x0, x1]).ravel()
            ys = np.column_stack([y0, y1]).ravel()
            self.plot.plot(xs, ys, connect="pairs", pen=pg.mkPen(colour, width=1))
        named = any(name for name, *_rest in spec.get("sets", ()))
        if named:
            self.plot.addLegend(offset=(-4, -4), labelTextColor=colours["fg"])
        for name, xs, ys, colour, hollow in spec.get("sets", ()):
            if isinstance(colour, (list, tuple)):
                brush = [pg.mkBrush(c) for c in colour]
            else:
                brush = pg.mkBrush(None if hollow else colour)
            pen = pg.mkPen(colour if hollow and not isinstance(colour, (list, tuple))
                           else None)
            size = spec.get("size") or (7 if hollow else 8)
            item = pg.ScatterPlotItem(x=np.asarray(xs, float), y=np.asarray(ys, float),
                                      size=size, brush=brush, pen=pen,
                                      name=str(name) if name else None)
            item.setOpacity(_ALPHA)
            if spec.get("keys") is not None:
                item.sigClicked.connect(self._on_points_clicked)
            self.plot.addItem(item)
            self.scatter_items.append(item)

    def _line(self, spec: Mapping[str, Any], colours, pg) -> None:
        """Lines over a continuous x, with optional marked points on top."""
        named = any(name for name, *_r in spec.get("series", ())) or spec.get("points")
        if named:
            self.plot.addLegend(offset=(-4, 4), labelTextColor=colours["fg"])
        for name, xs, ys, colour in spec.get("series", ()):
            item = self.plot.plot(np.asarray(xs, float), np.asarray(ys, float),
                                  pen=pg.mkPen(colour, width=2),
                                  symbol="o" if spec.get("markers") else None,
                                  symbolSize=5, symbolBrush=pg.mkBrush(colour),
                                  symbolPen=None, name=str(name) if name else None)
            item.setOpacity(_ALPHA)
        for name, xs, ys, colour in spec.get("points") or ():
            item = pg.ScatterPlotItem(x=np.asarray(xs, float), y=np.asarray(ys, float),
                                      size=8, brush=pg.mkBrush(colour), pen=None,
                                      name=str(name) if name else None)
            item.setOpacity(_ALPHA)
            self.plot.addItem(item)

    def _stacked(self, spec: Mapping[str, Any], colours, pg) -> None:
        """Bars of several layers stacked on one another, the first at the bottom."""
        x = np.asarray(spec["x"], float)
        bottom = np.zeros(len(x))
        for name, values, colour in spec["layers"]:
            values = np.asarray(values, float)
            bars = pg.BarGraphItem(x=x, y0=bottom, height=values, width=0.6,
                                   brush=pg.mkBrush(colour), pen=pg.mkPen(None))
            bars.setOpacity(_ALPHA)
            self.plot.addItem(bars)
            self._key(str(name), colour)
            bottom = bottom + values
        self._ticks(spec, pg)

    def _key(self, name: str, colour: str, line: bool = False) -> None:
        """Add one entry to the key above the plot, outside the data.

        :param name: what the mark stands for, translated.
        :param colour: its colour.
        :param line: a dotted line instead of a filled square.
        """
        import html

        mark = "\u2508\u2508" if line else "\u25a0"
        self._key_entries.append(f'<span style="color:{colour}">{mark}</span>&nbsp;'
                                 f'{html.escape(name)}')
        self.key.setText("&nbsp;&nbsp;&nbsp;".join(self._key_entries))
        self.key.show()

    def key_text(self) -> str:
        """The key above the plot as plain text, for tests and checks."""
        from PySide6.QtGui import QTextDocument

        document = QTextDocument()
        document.setHtml(self.key.text())
        return document.toPlainText() if self.key.isVisibleTo(self) else ""

    def _heatmap(self, spec: Mapping[str, Any], colours, pg) -> None:
        """A matrix of values, diverging around 0, its rows and columns named."""
        values = np.nan_to_num(np.asarray(spec["values"], float))
        limit = float(np.max(np.abs(values))) if values.size else 1.0
        limit = limit or 1.0
        image = pg.ImageItem(values.T)
        cmap = pg.ColorMap([0.0, 0.5, 1.0], [(200, 120, 168), (255, 255, 255),
                                             (10, 119, 184)])
        image.setLookupTable(cmap.getLookupTable(nPts=256))
        image.setLevels([-limit, limit])
        image.setOpacity(_ALPHA)
        self.plot.addItem(image)
        self.plot.getViewBox().invertY(True)
        self.plot.getAxis("bottom").setTicks(
            [[(i + 0.5, str(c)) for i, c in enumerate(spec.get("columns") or ())]])
        self.plot.getAxis("left").setTicks(
            [[(i + 0.5, str(r)) for i, r in enumerate(spec.get("rows") or ())]])
        try:
            bar = pg.ColorBarItem(values=(-limit, limit), colorMap=cmap,
                                  label=str(spec.get("scale") or ""), interactive=False)
            bar.setImageItem(image, insert_in=self.plot.getPlotItem())
        except Exception:
            LOG.debug("no colour bar for the heatmap", exc_info=True)

    def _groups(self, spec: Mapping[str, Any], colours, pg) -> None:
        """One slot per group: points below 3 members, a bar with points to 20, a box above."""
        rng = np.random.default_rng(0)
        ticks = []
        for slot, (name, values, colour) in enumerate(spec.get("groups") or ()):
            values = np.asarray(values, float)
            ticks.append((float(slot), str(name)))
            if not len(values):
                continue
            if 3 <= len(values) <= 20:
                bar = pg.BarGraphItem(x=[float(slot)], height=[float(np.mean(values))],
                                      width=0.6, brush=pg.mkBrush(colour), pen=pg.mkPen(None))
                bar.setOpacity(_ALPHA)
                self.plot.addItem(bar)
            if len(values) > 20:
                low, q1, median, q3, high = np.percentile(values, [5, 25, 50, 75, 95])
                box = pg.BarGraphItem(x=[float(slot)], y0=[float(q1)], height=[float(q3 - q1)],
                                      width=0.6, brush=pg.mkBrush(colour), pen=pg.mkPen(None))
                box.setOpacity(_ALPHA)
                self.plot.addItem(box)
                ink = pg.mkPen(colours["fg"], width=1)
                self.plot.plot([slot, slot], [low, q1], pen=ink)
                self.plot.plot([slot, slot], [q3, high], pen=ink)
                self.plot.plot([slot - 0.3, slot + 0.3], [median, median],
                               pen=pg.mkPen(colours["fg"], width=2))
                continue
            spread = rng.uniform(-0.15, 0.15, len(values)) if len(values) > 1 else [0.0]
            dots = pg.ScatterPlotItem(x=slot + np.asarray(spread), y=values, size=8,
                                      brush=pg.mkBrush(colour), pen=None)
            dots.setOpacity(_ALPHA)
            self.plot.addItem(dots)
        self.plot.getAxis("bottom").setTicks([ticks])

    def _on_points_clicked(self, _item, points, *_event) -> None:
        """Report the first clicked point by its position in the spec."""
        if len(points):
            self.point_clicked.emit(int(points[0].index()))

    def items(self) -> list:
        """What the plot holds, for tests and checks."""
        return [] if self.plot is None else list(self.plot.getPlotItem().items)


def _captioned(widget: QWidget, title: str) -> QWidget:
    """``widget`` under its caption, as one entry of a tab's queue."""
    holder = QWidget()
    holder.setObjectName("OpsLiveEntry")
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(2)
    caption = QLabel(str(title), holder)
    caption.setObjectName("OpsLiveImageTitle")
    caption.setWordWrap(True)
    column.addWidget(caption)
    column.addWidget(widget, 1)
    return holder


class _SquareBox(QWidget):
    """Hold one widget as the largest square that fits, centred.

    A graph in the OPS live view is square, not a wide rectangle, whatever
    shape the view around it has.

    :param child: the widget to keep square.
    :param parent: parent widget.
    """

    def __init__(self, child: QWidget, parent=None):
        """Take ``child`` and size it on every resize."""
        super().__init__(parent)
        self.setObjectName("OpsLiveSquare")
        self.child = child
        child.setParent(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 160)

    def resizeEvent(self, event):                          # noqa: N802
        """Lay the child out as the largest centred square."""
        super().resizeEvent(event)
        side = min(self.width(), self.height())
        self.child.setGeometry((self.width() - side) // 2,
                               (self.height() - side) // 2, side, side)


def _point_tile_html(tile: Mapping[str, Any], index: int) -> str:
    """The information tile of one clustered gene, as HTML.

    :param tile: the ``tile`` part of the embedding graph spec: ``genes``,
        ``clusters``, ``n_cells``, ``components`` and ``string``.
    :param index: the clicked point.
    """
    import html as html_text

    genes = list(tile.get("genes") or ())
    if not 0 <= index < len(genes):
        return ""
    gene = str(genes[index])
    clusters = tile.get("clusters")
    components = np.asarray(tile.get("components"), float)
    rows: List[Tuple[str, str]] = []
    if clusters is not None:
        cluster = int(clusters[index])
        members = [g for g, c in zip(genes, clusters) if int(c) == cluster and g != gene]
        rows.append((tr("Cluster"), tr("{k}, {n} genes", k=cluster, n=len(members) + 1)))
        rows.append((tr("Shares the cluster with"),
                     ", ".join(members[:15]) + (" …" if len(members) > 15 else "")))
    cells = tile.get("n_cells")
    if cells is not None and index < len(cells):
        rows.append((tr("Cells"), f"{int(cells[index])}"))
    if components.ndim == 2 and components.shape[0] == len(genes):
        position = ", ".join(f"{v:+.2f}" for v in components[index, :3])
        rows.append((tr("Position (PC 1-3)"), position))
        distance = np.linalg.norm(components - components[index], axis=1)
        nearest = [genes[i] for i in np.argsort(distance) if i != index][:5]
        rows.append((tr("Nearest genes in the embedding"), ", ".join(nearest)))
    quoted = html_text.escape(gene)
    links = (f'<a href="https://www.ncbi.nlm.nih.gov/gene/?term={quoted}%5Bsym%5D+AND+'
             f'human%5Borgn%5D">NCBI Gene</a> · '
             f'<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene={quoted}">'
             f'GeneCards</a> · '
             f'<a href="https://string-db.org/cgi/network?identifiers={quoted}'
             f'&species=9606">STRING</a>')
    body = "".join(f"<tr><td><b>{html_text.escape(k)}</b></td>"
                   f"<td>{html_text.escape(v)}</td></tr>" for k, v in rows)
    return (f"<h3>{quoted}</h3><table cellspacing='4'>{body}</table>"
            f"<p>{links}</p>")


class _PointGraph(QWidget):
    """A square scatter with an information tile beside it, filled by a click.

    Built for the clustering embedding: clicking a gene shows its cluster,
    the genes it shares the cluster with, its cells, its position and its
    nearest genes, with links out, as clicking a gene in the regression
    results does.

    :param spec: a scatter spec with ``keys`` and ``tile``.
    :param colours: from :func:`_theme_colours`.
    :param parent: parent widget.
    """

    def __init__(self, spec: Mapping[str, Any], colours: Mapping[str, str],
                 parent=None):
        """Lay out the square graph on the left and the tile on the right."""
        super().__init__(parent)
        self.setObjectName("OpsLivePointGraph")
        from PySide6.QtWidgets import QTextBrowser

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.graph = _QtGraph()
        self.graph.show_spec(spec, colours)
        row.addWidget(_SquareBox(self.graph, self), 3)
        self.tile = QTextBrowser(self)
        self.tile.setObjectName("OpsLivePointTile")
        self.tile.setOpenLinks(False)
        self.tile.setOpenExternalLinks(False)
        self.tile.setMinimumWidth(220)
        self.tile.anchorClicked.connect(self._open)
        self.tile.setPlainText(tr("Click a gene in the embedding to see what it is "
                                  "and which genes it clusters with."))
        row.addWidget(self.tile, 2)
        self._spec = spec
        self.shown_index: Optional[int] = None
        self.graph.point_clicked.connect(self.show_point)

    def show_point(self, index: int) -> None:
        """Fill the tile for one point of the embedding.

        :param index: the point, in the order of the spec's ``keys``.
        """
        text = _point_tile_html(self._spec.get("tile") or {}, int(index))
        if text:
            self.shown_index = int(index)
            self.tile.setHtml(text)

    @staticmethod
    def _open(url) -> None:
        """Open a link of the tile in the web browser."""
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(url)


class _MosaicWorker(QThread):
    """Draw part of the stitched block at a finer resolution, off the GUI thread."""

    done = Signal(int, object, object, int)

    def __init__(self, token: int, render: Callable[..., np.ndarray],
                 window: Tuple[float, float, float, float], step: int, parent=None):
        """Carry one render.

        :param token: the view's request number, to drop late answers.
        :param render: ``render(window, step)`` returning RGB uint8.
        :param window: ``(top, left, bottom, right)`` in mosaic pixels.
        :param step: well pixels per output pixel.
        :param parent: parent object.
        """
        super().__init__(parent)
        self._token, self._render = token, render
        self._window, self._step = window, int(step)

    def run(self) -> None:
        """Render, and hand back the picture or None on any failure."""
        try:
            picture = self._render(self._window, self._step)
        except Exception:                                    # noqa: BLE001
            LOG.info("the zoomed mosaic could not be drawn", exc_info=True)
            picture = None
        try:
            self.done.emit(self._token, picture, self._window, self._step)
        except RuntimeError:
            pass


class _MosaicView(QWidget):
    """The stitched block, drawn again from the tiles at the zoom level.

    The whole block is first drawn every ``step`` sequencing pixels. When
    the user zooms in past that, the visible part is read again from the
    tiles at the finer step the screen can show and laid over the coarse
    picture, so the mosaic stays sharp. It is a zoom view like every other
    picture in the tab; see :attr:`view`.

    :param render: ``render(window, step)``: the RGB picture of
        ``(top, left, bottom, right)`` mosaic pixels at ``step`` sequencing
        pixels per picture pixel.
    :param base_step: sequencing pixels per pixel of the coarse picture.
    :param parent: parent widget.
    """

    refined = Signal(int)

    def __init__(self, render: Callable[..., np.ndarray], base_step: int, parent=None):
        """Wrap a zoom view and watch its zoom and pan."""
        super().__init__(parent)
        self.setObjectName("OpsLiveMosaic")
        from .live_preview import _ZoomView

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = _ZoomView(self)
        self.view.setObjectName("OpsLiveImageView")
        layout.addWidget(self.view)
        self._render = render
        self.base_step = max(1, int(base_step))
        self._token = 0
        self._workers: List[_MosaicWorker] = []
        self.overlay = None
        self.overlay_step: Optional[int] = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.refine)
        self.view.zoom_changed.connect(lambda _s: self._timer.start())
        self.view.horizontalScrollBar().valueChanged.connect(lambda _v: self._timer.start())
        self.view.verticalScrollBar().valueChanged.connect(lambda _v: self._timer.start())

    def wanted_step(self) -> int:
        """The sequencing pixels per screen pixel at the present zoom, at least 1."""
        ratio = float(self.view.devicePixelRatioF() or 1.0)
        shown = abs(float(self.view.transform().m11())) * ratio
        if shown <= 0:
            return self.base_step
        return int(max(1, min(self.base_step, np.floor(self.base_step / shown))))

    def refine(self) -> None:
        """Draw the visible part again at the zoom level, or drop the finer layer at fit."""
        step = self.wanted_step()
        if step >= self.base_step or self.view._pixmap_item is None:
            self._drop_overlay()
            return
        scene = self.view.sceneRect()
        seen = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        seen = seen.intersected(scene)
        if seen.isEmpty():
            return
        while (seen.width() * self.base_step / step) * (
                seen.height() * self.base_step / step) > 3000 * 3000:
            step += 1
        window = (seen.top(), seen.left(), seen.bottom(), seen.right())
        self._token += 1
        worker = _MosaicWorker(self._token, self._render, window, step, self)
        worker.done.connect(self._on_rendered)
        worker.finished.connect(self._forget_worker)
        self._workers.append(worker)
        worker.start()

    def _forget_worker(self) -> None:
        """Let a finished worker go."""
        worker = self.sender()
        if worker in self._workers:
            self._workers.remove(worker)
            worker.deleteLater()

    def _on_rendered(self, token: int, picture, window, step: int) -> None:
        """Lay the finer picture over the coarse one where it belongs."""
        if token != self._token or picture is None:
            return
        from .live_preview import numpy_to_qpixmap

        self._drop_overlay()
        pixmap = numpy_to_qpixmap(np.asarray(picture), normalise=False)
        item = self.view.scene().addPixmap(pixmap)
        item.setPos(float(window[1]), float(window[0]))
        item.setScale(float(step) / float(self.base_step))
        item.setTransformationMode(Qt.SmoothTransformation)
        self.overlay, self.overlay_step = item, int(step)
        self.refined.emit(int(step))

    def _drop_overlay(self) -> None:
        """Remove the finer layer, if any."""
        if self.overlay is not None:
            try:
                self.view.scene().removeItem(self.overlay)
            except RuntimeError:
                pass
        self.overlay, self.overlay_step = None, None

    def wait(self) -> None:
        """Wait for every render in flight, for shutdown and tests."""
        for worker in list(self._workers):
            worker.wait()


_SAMPLES: Dict[str, Tuple[str, int, int, int, int]] = {
    "align": ("ops_live_block", 1, 4, 2, 1),
    "spots": ("ops_live_sbs_crop", 100, 1480, 400, 50),
    "objects": ("ops_live_ph_fields", 1, 9, 9, 1),
    "merge": ("ops_live_merge_fields", 0, 500, 0, 1),
    "cluster": ("ops_live_genes", 0, 20000, 0, 5),
    "voyager3": ("ops_live_voyager3_clusters", 0, 200, 3, 1),
}
"""Each tab's Sample: its preview-only setting, lowest, highest, default and
step. A lowest of 0 means every one, and the box says so."""


def _sample_words(step: str) -> Tuple[str, str]:
    """A tab's Sample unit and tooltip, in the interface language."""
    return {
        "align": (tr(" tiles per side"), tr(
            "How many sequencing tiles along each side of the region are "
            "stitched: 2 stitches a 2 x 2 block.")),
        "spots": (tr(" px crop"), tr(
            "The side of the sequencing crop that is genotyped, in pixels. "
            "Objects & phenotype segments the same area.")),
        "objects": (tr(" phenotype fields"), tr(
            "The most phenotype fields over the genotyped region that are "
            "segmented and measured, largest part first.")),
        "merge": (tr(" fields listed"), tr(
            "How many phenotype fields the cells-matched graph lists, the "
            "fields with most cells first.")),
        "cluster": (tr(" genes"), tr(
            "The most genes that are clustered, the genes with most cells "
            "first.")),
        "voyager3": (tr(" clusters"), tr(
            "How many clusters Voyager3 names, the largest first. Each one is "
            "one question to the AI when the method is ai.")),
    }[step]


def _sample_all(step: str) -> str:
    """What a tab's Sample box says at 0, which means every one."""
    return {"merge": tr("all fields"), "cluster": tr("all genes"),
            "voyager3": tr("all clusters")}.get(step, tr("all"))


def _table_widget(header, rows, parent=None) -> QTableWidget:
    """A sortable read-only table of one pass, with header tooltips."""
    from .sortable_table import install_sorting, table_item

    table = QTableWidget(0, 0, parent)
    table.setObjectName("OpsLiveTable")
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    install_sorting(table)
    rows = list(rows)[:_TABLE_ROWS]
    sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.setColumnCount(len(header))
    table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels(
        [h[0] if isinstance(h, tuple) else str(h) for h in header])
    for column, item in enumerate(header):
        if isinstance(item, tuple):
            cell = table.horizontalHeaderItem(column)
            if cell is not None:
                cell.setToolTip(item[1])
    for r, values in enumerate(rows):
        for c, value in enumerate(list(values)[:len(header)]):
            table.setItem(r, c, table_item(str(value)))
    table.setSortingEnabled(sorting)
    return table


class _StepPage(QWidget):
    """One tab: Sample, Settings and Run preview, then its figure queue.

    The tool row reads left to right: the region (Align & stitch only), the
    Sample, Settings, Run preview, Refresh (laptop mode only), the running
    indicator and the image-quality slider. Below it the tab shows one
    picture, graph or table at a time in Mask Generation's figure queue
    (:class:`~spacr.qt.widgets.figure_queue.FigureQueue`), with the
    thumbnails on the left. Pictures zoom; graphs are square pyqtgraph plots.

    :param step: the tab's step.
    :param blurb: one sentence on what the tab runs, translated.
    :param parent: parent widget.
    """

    settings_requested = Signal(str)
    refresh_requested = Signal(str)
    run_requested = Signal(str)
    quality_changed = Signal(str, int)
    sample_changed = Signal(str, str, int)
    region_changed = Signal(int)

    def __init__(self, step: str, blurb: str, parent=None):
        """Lay out the tab, the same geometry for every step."""
        super().__init__(parent)
        self.setObjectName("OpsLiveStepPage")
        self.step = step
        self._images: List[Tuple[str, np.ndarray]] = []
        self._colours = _theme_colours()
        self._quality_pending = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.blurb = QLabel(blurb, self)
        self.blurb.setWordWrap(True)
        self.blurb.setObjectName("OpsLiveBlurb")
        layout.addWidget(self.blurb)
        tools = QHBoxLayout()
        self.region = None
        if step == "align":
            from PySide6.QtWidgets import QComboBox

            self.region = QComboBox(self)
            self.region.setObjectName("OpsLiveRegion")
            self.region.setToolTip(tr(
                "The region of the well the preview stitches: a block of "
                "sequencing tiles and every phenotype field placed on it. "
                "Spots & genotyping reads the region shown here; Objects & "
                "phenotype segments the whole part of it that is genotyped."))
            self.region.addItem(tr("Region 1"), 0)
            self.region.activated.connect(
                lambda index: self.region_changed.emit(int(self.region.itemData(index) or 0)))
            tools.addWidget(self.region)
        key, low, high, default, single = _SAMPLES[step]
        unit, tip = _sample_words(step)
        self.sample_label = QLabel(tr("Sample"), self)
        self.sample_label.setObjectName("OpsLiveSampleLabel")
        tools.addWidget(self.sample_label)
        self.sample_spin = QSpinBox(self)
        self.sample_spin.setObjectName("OpsLiveSample")
        self.sample_spin.setRange(low, high)
        self.sample_spin.setSingleStep(single)
        self.sample_spin.setValue(default)
        self.sample_spin.setSuffix(unit)
        if low == 0:
            self.sample_spin.setSpecialValueText(_sample_all(step))
        self.sample_spin.setToolTip(tip)
        self.sample_spin.setKeyboardTracking(False)
        self.sample_key = key
        self.sample_spin.valueChanged.connect(
            lambda value: self.sample_changed.emit(self.step, self.sample_key, int(value)))
        tools.addWidget(self.sample_spin)
        self.settings_btn = QPushButton(tr("Settings…"), self)
        self.settings_btn.setObjectName("OpsLiveSettingsButton")
        self.settings_btn.setToolTip(tr(
            "Open every setting of this tab. A change runs the tab again on "
            "the preview data; Apply to OPS module copies them into the OPS "
            "settings the real run uses."))
        self.settings_btn.clicked.connect(
            lambda _checked=False: self.settings_requested.emit(self.step))
        tools.addWidget(self.settings_btn)
        self.run_btn = QPushButton(tr(PREVIEW_RUN_TEXT), self)
        self.run_btn.setObjectName("OpsLiveRunButton")
        self.run_btn.setToolTip(tr("Run this tab again now on the sample."))
        self.run_btn.clicked.connect(
            lambda _checked=False: self.run_requested.emit(self.step))
        tools.addWidget(self.run_btn)
        self.refresh_btn = QPushButton(tr("Refresh view"), self)
        self.refresh_btn.setObjectName("OpsLiveRefreshButton")
        self.refresh_btn.setToolTip(tr(
            "Laptop mode: apply the changed settings and image quality and "
            "redraw this tab. Nothing refreshes by itself in laptop mode."))
        self.refresh_btn.clicked.connect(
            lambda _checked=False: self.refresh_requested.emit(self.step))
        self.refresh_btn.setVisible(False)
        tools.addWidget(self.refresh_btn)
        from .activity_spinner import ActivitySpinner

        self.spinner = ActivitySpinner(self, auto=False, delay_ms=0)
        self.spinner.setObjectName("OpsLiveSpinner")
        tools.addWidget(self.spinner)
        self.state = QLabel(tr("Not run yet. Switch Live on, or press Run preview."),
                            self)
        self.state.setObjectName("OpsLiveState")
        self.state.setWordWrap(True)
        tools.addWidget(self.state, 1)
        from .live_preview import _ImageQualityControl

        self.quality_control = _ImageQualityControl(self)
        self.quality = self.quality_control.slider
        self.quality.setObjectName("OpsLiveImageQuality")
        self.quality_value = self.quality_control.value_label
        self.quality_value.setObjectName("OpsLiveImageQualityValue")
        self.quality_control.changed.connect(
            lambda percent: self.quality_changed.emit(self.step, int(percent)))
        tools.addWidget(self.quality_control)
        layout.addLayout(tools)
        self.notice = QLabel("", self)
        self.notice.setObjectName("OpsLiveNotice")
        self.notice.setWordWrap(True)
        self.notice.setAlignment(Qt.AlignCenter)
        self.notice.hide()
        layout.addWidget(self.notice)
        from .figure_queue import FigureQueue

        self.queue = FigureQueue(ram_cap=40, parent=self)
        self.queue.setObjectName("OpsLiveQueue")
        self.queue._set_figure_tools_visible(False)
        self.queue._set_opaque(True)
        layout.addWidget(self.queue, 1)
        self.views: List[Any] = []
        self.graphs: List[_QtGraph] = []
        self.mosaic: Optional[_MosaicView] = None
        self.point_graph: Optional[_PointGraph] = None
        self.table = _table_widget([], [], self)
        self.table.hide()
        self.sample = QLabel("", self)
        self.sample.setObjectName("OpsLiveSampleLine")
        self.sample.setWordWrap(True)
        layout.addWidget(self.sample)
        self._payload: Optional[Mapping[str, Any]] = None

    def sample_value(self) -> int:
        """The Sample the tab uses now."""
        return int(self.sample_spin.value())

    def set_regions(self, regions: Sequence[Sequence[int]], current: int,
                    fields: Sequence[int] = ()) -> None:
        """List the regions the Align & stitch tab can stitch.

        :param regions: each region as its sequencing tiles.
        :param current: the region stitched now.
        :param fields: the phenotype fields placed on the current region.
        """
        if self.region is None:
            return
        blocked = self.region.blockSignals(True)
        try:
            self.region.clear()
            for index, sites in enumerate(regions):
                text = tr("Region {n}: tiles {tiles}", n=index + 1,
                          tiles=", ".join(str(s) for s in sites))
                if index == current and fields:
                    text = tr("{region}; fields {fields}", region=text,
                              fields=", ".join(str(f) for f in fields))
                self.region.addItem(text, index)
            self.region.setCurrentIndex(max(0, min(int(current), len(regions) - 1)))
            self.region.setEnabled(len(regions) > 1)
        finally:
            self.region.blockSignals(blocked)

    def quality_percent(self) -> int:
        """The image quality the slider is at, in percent."""
        return int(self.quality.value())

    def _emit_quality(self) -> None:
        """Announce the slider's quality now, as when it comes to rest."""
        self.quality_control._announce()

    def set_laptop(self, on: bool) -> None:
        """Show the Refresh button in laptop mode only."""
        self.refresh_btn.setVisible(bool(on))

    def set_running(self, running: bool, text: str = "") -> None:
        """Show or clear the running indicator in the tab.

        :param running: whether the tab's pass is scheduled or running.
        :param text: the sentence beside the spinner, translated.
        """
        self.spinner.set_busy(bool(running))
        if running:
            self.state.setText(text or tr("Running {step}…", step=_title(self.step)))
            self.state.setProperty("running", True)
        else:
            self.state.setProperty("running", False)
            if text:
                self.state.setText(text)

    def is_running(self) -> bool:
        """Whether the running indicator is on."""
        return bool(self.state.property("running"))

    def mark_pending(self, text: str) -> None:
        """Say that something changed and waits for Refresh (laptop mode)."""
        self.state.setText(text)

    def mark_quality_pending(self) -> None:
        """The quality changed in laptop mode; the next Refresh applies it."""
        self._quality_pending = True
        self.mark_pending(tr("Image quality changed. Press Refresh view to redraw."))

    def quality_pending(self) -> bool:
        """Whether a quality change waits for Refresh."""
        return self._quality_pending

    def render_images(self) -> None:
        """Draw the held pictures at the slider's image quality."""
        from .live_preview import numpy_to_qpixmap

        self._quality_pending = False
        percent = self.quality_percent()
        for index, (_title_text, picture) in enumerate(self._images):
            if index >= len(self.views):
                break
            view = self.views[index]
            view.set_render_percent(percent)
            view.set_pixmap(numpy_to_qpixmap(picture, normalise=False))
            view.set_picture_name(f"ops_live_{self.step}_{index + 1}")

    def shown_size(self, index: int = 0) -> Optional[Tuple[int, int]]:
        """The ``(width, height)`` the picture in a slot is drawn at, or None."""
        if index >= len(self.views) or index >= len(self._images):
            return None
        pixmap = self.views[index].shown_pixmap()
        return None if pixmap is None else (pixmap.width(), pixmap.height())

    def entry_titles(self) -> List[str]:
        """The names of the entries in the tab's queue, in order."""
        return list(self.queue.figure_titles())

    def show_result(self, payload: Mapping[str, Any]) -> None:
        """Put one pass's pictures, graphs and table in the tab's queue."""
        from .live_preview import _ZoomView, numpy_to_qpixmap

        self.notice.hide()
        self.queue.show()
        self._payload = payload
        self._colours = _theme_colours()
        keep = max(0, self.queue._current_index())
        self._stop_mosaic()
        self.queue.clear()
        self.views, self.graphs = [], []
        self.mosaic = self.point_graph = None
        self._images = [(str(t), np.asarray(p)) for t, p in payload.get("images", ())]
        mosaic = payload.get("mosaic")
        for index, (title, picture) in enumerate(self._images):
            if index == 0 and mosaic is not None:
                self.mosaic = _MosaicView(mosaic["render"], mosaic["step"])
                view = self.mosaic.view
                picture_widget: QWidget = self.mosaic
            else:
                view = _ZoomView()
                view.setObjectName("OpsLiveImageView")
                picture_widget = view
            holder = _captioned(picture_widget, title)
            self.views.append(view)
            self.queue._add_widget(holder, title,
                                  numpy_to_qpixmap(picture, normalise=False))
        self.render_images()
        for spec in payload.get("graphs", ()):
            if spec.get("kind") == "scatter" and spec.get("tile") is not None:
                pointed = _PointGraph(spec, self._colours)
                if self.point_graph is None:
                    self.point_graph = pointed
                graph, holder = pointed.graph, pointed
            else:
                graph = _QtGraph()
                graph.show_spec(spec, self._colours)
                holder = _SquareBox(graph)
            self.graphs.append(graph)
            self.queue._add_widget(holder, str(spec.get("title") or tr("Graph")))
        header, rows = payload.get("header") or [], payload.get("rows") or []
        self.table = _table_widget(header, rows)
        if header:
            name = tr("Table: {n} rows", n=min(len(rows), _TABLE_ROWS))
            self.queue._add_widget(_captioned(self.table, name), name)
        if self.queue.count():
            self.queue.show_index(min(keep, self.queue.count() - 1))
        seconds = payload.get("seconds")
        source = (tr("OPS test data (Funk et al. 2022, plate A, well A1)")
                  if payload.get("test_data") else tr("your data"))
        self.sample.setText(tr("{sample}. {seconds} s on {source}.",
                               sample=payload.get("sample", ""),
                               seconds=seconds if seconds is not None else "?",
                               source=source))
        if self.region is not None and payload.get("regions"):
            self.set_regions(payload["regions"], int(payload.get("region") or 0),
                             payload.get("region_fields") or ())

    def _stop_mosaic(self) -> None:
        """Let a zoomed mosaic render in flight finish before its view goes."""
        if self.mosaic is not None:
            self.mosaic._timer.stop()
            self.mosaic.wait()

    def _clear_pictures(self) -> None:
        """Drop the pictures and graphs, so a notice replaces the last pass."""
        self._stop_mosaic()
        self._images = []
        self.views, self.graphs = [], []
        self.mosaic = self.point_graph = None
        self.queue.clear()
        self.queue.hide()

    def show_notice(self, text: str) -> None:
        """Say what the tab needs before it can run, in the pictures' place."""
        self._clear_pictures()
        self.notice.setText(text)
        self.notice.show()

    def show_failure(self, text: str) -> None:
        """Say why the pass failed, in the pictures' place."""
        self.show_notice(text)


def _same_value(first: Any, second: Any) -> bool:
    """Whether an editor value says the same as a setting, so leaving an editor untouched runs nothing."""
    if first == second:
        return True
    try:
        return float(first) == float(second)
    except (TypeError, ValueError):
        return str(first if first is not None else "") == str(
            second if second is not None else "")


def _editor_value(widget) -> Any:
    """The value a settings-panel editor holds.

    A line edit's text is read as a number, a list or text, the way the
    setting's default is written; empty means unset.
    """
    if isinstance(widget, QCheckBox):
        return bool(widget.isChecked())
    if isinstance(widget, QSpinBox):
        return int(widget.value())
    if isinstance(widget, QDoubleSpinBox):
        return float(widget.value())
    text = widget.text().strip()
    if not text:
        return None
    if widget.property("opsListValue"):
        return [_parse_token(part) for part in text.replace(";", ",").split(",")
                if part.strip()]
    return _parse_token(text)


def _parse_token(text: str) -> Any:
    """One typed value: a number when it reads as one, else the text."""
    text = str(text).strip()
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return value if isinstance(value, (int, float, bool)) else text


def _set_editor_value(widget, value: Any) -> None:
    """Show ``value`` in a settings-panel editor without announcing it."""
    blocked = widget.blockSignals(True)
    try:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            if value is not None:
                widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox)
                                else int(value))
        else:
            if isinstance(value, (list, tuple)):
                text = ", ".join(str(v) for v in value)
            else:
                text = "" if value is None else str(value)
            widget.setText(text)
    finally:
        widget.blockSignals(blocked)


def _editor_for(key: str, value: Any, parent=None) -> QWidget:
    """A settings-panel editor for one setting, of the kind its value is.

    :param key: the setting.
    :param value: its current value.
    :returns: a check box for a flag, a spin box for a number that is always
        set, a line edit for anything else.
    """
    try:
        from spacr.ops_settings import OPS_DEFAULTS, OPS_TYPES
        declared = OPS_TYPES.get(key)
        default = OPS_DEFAULTS.get(key, value)
    except Exception:
        declared, default = None, value
    kinds = declared if isinstance(declared, tuple) else (declared,)
    nullable = type(None) in kinds
    sample = value if value is not None else default
    if isinstance(sample, bool) or (bool in kinds and not nullable):
        widget: QWidget = QCheckBox(parent)
    elif isinstance(sample, int) and not nullable and int in kinds and float not in kinds:
        widget = QSpinBox(parent)
        widget.setRange(-1_000_000 if int(sample) < 0 else 0, 1_000_000)
    elif isinstance(sample, (int, float)) and not nullable and float in kinds:
        widget = QDoubleSpinBox(parent)
        widget.setDecimals(4)
        widget.setRange(-1e9 if float(sample) < 0 else 0.0, 1e9)
        widget.setSingleStep(max(abs(float(sample)) / 10.0, 0.01))
    else:
        widget = QLineEdit(parent)
        widget.setProperty("opsListValue", isinstance(default, (list, tuple)))
    widget.setProperty("settingKey", key)
    widget.setObjectName(f"OpsLiveSetting_{key}")
    _set_editor_value(widget, value if value is not None else
                      (default if not isinstance(widget, QLineEdit) else value))
    return widget


def _editor_signal(widget):
    """The signal an editor emits when its value changes."""
    if isinstance(widget, QCheckBox):
        return widget.toggled
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.valueChanged
    return widget.editingFinished


class _StepSettingsDialog(QDialog):
    """One tab's settings: every setting its step reads, grouped as the OPS form groups them.

    A change runs the tab again on the preview data (or waits for Refresh in
    laptop mode). Apply to OPS module copies the values tuned here into the
    OPS settings the real run uses.

    :param panel: the OPS Live panel; also the dialog's parent.
    :param step: the tab the dialog edits.
    """

    def __init__(self, panel: "_OpsLivePanel", step: str):
        """Lay out one editor per setting under its category, the extra groups and the buttons."""
        super().__init__(panel)
        self.setObjectName("OpsLiveSettingsDialog")
        self._panel, self.step = panel, step
        self.setWindowTitle(tr("{step}: live settings", step=_title(step)))
        outer = QVBoxLayout(self)
        body = QWidget(self)
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        self.editors: Dict[str, QWidget] = {}
        self.groups: Dict[str, QGroupBox] = {}
        values = panel.settings_now()
        from ..screens.settings_model import app_setting_label

        try:
            from spacr.ops_live import _registry_categories

            categories = _registry_categories()
        except Exception:
            categories = {}
        for key in panel.step_settings(step):
            title = categories.get(key) or _title(step)
            group = self.groups.get(title)
            if group is None:
                group = QGroupBox(title, body)
                group.setObjectName("OpsLiveSettingsGroup")
                QFormLayout(group)
                self.groups[title] = group
                column.addWidget(group)
            editor = _editor_for(key, values.get(key), group)
            try:
                label = app_setting_label("ops", key)
            except Exception:
                label = key
            group.layout().addRow(label or key, editor)
            self.editors[key] = editor
            _editor_signal(editor).connect(self._on_edit)
        self._borrowed: List[Tuple[QWidget, QWidget]] = []
        for title, widget in panel.step_extras(step):
            box = QGroupBox(title, body)
            box_layout = QVBoxLayout(box)
            self._borrowed.append((widget, widget.parentWidget()))
            widget.setParent(box)
            box_layout.addWidget(widget)
            widget.show()
            column.addWidget(box)
        column.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.note = QLabel("", self)
        self.note.setObjectName("OpsLiveSettingsNote")
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self.refresh_btn = QPushButton(tr("Refresh view"), self)
        self.refresh_btn.setObjectName("OpsLiveDialogRefreshButton")
        self.refresh_btn.setToolTip(tr(
            "Laptop mode: apply these settings and redraw the tab."))
        self.refresh_btn.clicked.connect(
            lambda _checked=False: panel.refresh(step))
        self.refresh_btn.setVisible(_laptop_mode())
        buttons.addButton(self.refresh_btn, QDialogButtonBox.ActionRole)
        self.apply_btn = QPushButton(tr("Apply to OPS module"), self)
        self.apply_btn.setObjectName("OpsLiveApplyButton")
        self.apply_btn.setToolTip(tr(
            "Copy the values changed here into the OPS settings, so the real "
            "OPS run uses them."))
        self.apply_btn.clicked.connect(self._on_apply)
        buttons.addButton(self.apply_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        self.close_btn = buttons.button(QDialogButtonBox.Close)
        outer.addWidget(buttons)
        try:
            from ..screens.settings_model import install_api_tooltips

            install_api_tooltips(self, "ops", {w: k for k, w in self.editors.items()})
        except Exception:
            LOG.debug("no API help for the OPS live settings", exc_info=True)
        self.resize(600, 640)

    def values(self) -> Dict[str, Any]:
        """Every setting the dialog shows, as it reads now."""
        return {key: _editor_value(widget) for key, widget in self.editors.items()}

    def load(self, values: Mapping[str, Any]) -> None:
        """Show the given values without running anything."""
        for key, widget in self.editors.items():
            if key in values:
                _set_editor_value(widget, values[key])

    def _on_edit(self, *_args) -> None:
        """One editor changed: hand its value to the panel."""
        widget = self.sender()
        key = widget.property("settingKey") if widget is not None else None
        if key in self.editors:
            value = _editor_value(self.editors[key])
            if _same_value(value, self._panel.settings_now().get(key)):
                return
            self._panel.panel_setting_changed(self.step, str(key), value)

    def _on_apply(self) -> None:
        """Copy the tuned values into the OPS settings and say how many went."""
        count = self._panel.apply_to_ops(self.step)
        self.note.setText(tr("{n} settings copied into the OPS module.", n=count)
                          if count else tr("Nothing changed here yet, so nothing "
                                           "was copied."))

    def done(self, result) -> None:
        """Hand borrowed widgets back on every exit."""
        for widget, home in self._borrowed:
            widget.hide()
            widget.setParent(home)
        self._borrowed = []
        super().done(result)


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
        self._forced = False
        self._worker: Optional[_StepWorker] = None
        self._retired: List[_StepWorker] = []
        self._cancel_event = threading.Event()
        self._pending: Optional[str] = None
        self._overrides: Dict[str, Any] = {}
        self._propagate_cb: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._dialogs: Dict[str, _StepSettingsDialog] = {}
        self._extras: Dict[str, List[Tuple[str, QWidget]]] = {}
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
            page = _StepPage(step, _blurb(step), self._tabs)
            page.settings_requested.connect(self.open_settings)
            page.refresh_requested.connect(self.refresh)
            page.run_requested.connect(self._on_run_requested)
            page.sample_changed.connect(self.panel_setting_changed)
            page.region_changed.connect(self.choose_region)
            page.quality_changed.connect(self._on_quality_changed)
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
        self._sync_laptop()

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

    def settings_now(self) -> Dict[str, Any]:
        """The settings a pass runs with: the form, then the panels' changes."""
        try:
            form = dict(self._collect() or {})
        except Exception:
            form = {}
        form.update(self._overrides)
        return form

    def step_settings(self, step: str) -> Tuple[str, ...]:
        """The settings one tab's Settings panel offers.

        :param step: the tab.
        :returns: every OPS setting the step alone reads, as the registry
            places it (:func:`spacr.ops_live._step_map`); settings every
            step reads, such as the input folders, are on the OPS form.
        """
        try:
            from spacr.ops_settings import OPS_DEFAULTS as defaults
        except Exception:
            return ()
        steps = self._step_map()
        return tuple(sorted(key for key in defaults
                            if tuple(steps.get(key, ())) == (step,)))

    def add_step_settings(self, step: str, title: str, widget: QWidget) -> None:
        """Put more inputs into one tab's Settings panel, under a heading.

        :param step: the tab.
        :param title: the group's heading, translated.
        :param widget: the inputs; lent to the panel while it is open.
        """
        self._extras.setdefault(step, []).append((title, widget))

    def step_extras(self, step: str) -> List[Tuple[str, QWidget]]:
        """The extra input groups of one tab's Settings panel."""
        return list(self._extras.get(step, ()))

    def set_propagate_callback(self, callback: Optional[Callable[[Dict[str, Any]], Any]]
                               ) -> None:
        """Say how Apply to OPS module writes values into the OPS form.

        :param callback: takes ``{setting: value}``; None leaves Apply to OPS
            module with nowhere to write.
        """
        self._propagate_cb = callback

    def open_settings(self, step: Optional[str] = None) -> _StepSettingsDialog:
        """Open, or bring forward, one tab's Settings panel.

        :param step: the tab; the tab on view when None.
        :returns: the dialog.
        """
        step = step or self.current_step()
        dialog = self._dialogs.get(step)
        if dialog is None or not dialog.isVisible():
            dialog = _StepSettingsDialog(self, step)
            dialog.finished.connect(self._on_settings_closed)
            self._dialogs[step] = dialog
            dialog.show()
        else:
            dialog.raise_()
            dialog.activateWindow()
        return dialog

    def _on_settings_closed(self, *_args) -> None:
        """Forget a closed Settings panel."""
        dialog = self.sender()
        for step, held in list(self._dialogs.items()):
            if held is dialog:
                self._dialogs.pop(step, None)

    def apply_to_ops(self, step: str) -> int:
        """Copy the values tuned on one tab into the OPS settings (Apply to OPS module).

        :param step: the tab.
        :returns: how many settings were copied.
        """
        keys = set(self.step_settings(step))
        tuned = {key: value for key, value in self._overrides.items() if key in keys}
        copied = self.propagate(tuned)
        if copied:
            for key in tuned:
                self._overrides.pop(key, None)
        return copied

    def choose_region(self, index: int) -> None:
        """Stitch another region in Align & stitch, and read it in the tabs after it.

        :param index: the region, from 0.
        """
        self.panel_setting_changed("align", "ops_live_region", int(index))

    def propagate(self, values: Mapping[str, Any]) -> int:
        """Copy settings into the OPS form.

        :param values: ``{setting: value}``.
        :returns: how many were handed to the form.
        """
        if self._propagate_cb is None or not values:
            return 0
        try:
            self._propagate_cb(dict(values))
        except Exception:                                    # noqa: BLE001
            LOG.info("OPS live could not propagate its settings", exc_info=True)
            return 0
        return len(values)

    def panel_setting_changed(self, step: str, key: str, value: Any) -> None:
        """A Settings panel changed one value: run its tab again.

        The value is laid over the form for every later pass; Apply to OPS
        module copies it into the form. Outside laptop mode the tab runs
        after the debounce, Live or not, because the panel is open for this;
        in laptop mode it waits for Refresh.

        :param step: the tab whose panel changed.
        :param key: the setting.
        :param value: its new value.
        """
        self._overrides[key] = value
        self._mark_stale(key)
        if step != self.current_step():
            self.show_step(step)
        if _laptop_mode():
            self._pages[step].mark_pending(tr("Settings changed. Press Refresh "
                                              "view to run this tab."))
            return
        self._schedule(step, forced=True)

    def _on_quality_changed(self, step: str, _percent: int) -> None:
        """A tab's slider rested: redraw at that quality, or wait in laptop mode."""
        page = self._pages[step]
        if _laptop_mode():
            page.mark_quality_pending()
            return
        page.render_images()

    def refresh(self, step: Optional[str] = None) -> None:
        """Apply what changed on one tab and redraw it (the Refresh button).

        :param step: the tab; the tab on view when None.
        """
        step = step or self.current_step()
        page = self._pages[step]
        page.render_images()
        if step != self.current_step():
            self.show_step(step)
        if step in self._stale:
            self._timer.stop()
            self._forced = False
            self._start(step)

    def _sync_laptop(self) -> bool:
        """Show the Refresh buttons when laptop mode is on.

        :returns: whether laptop mode is on.
        """
        on = _laptop_mode()
        for page in self._pages.values():
            page.set_laptop(on)
        for dialog in self._dialogs.values():
            dialog.refresh_btn.setVisible(on)
        return on

    def showEvent(self, event):                              # noqa: N802
        """Follow a performance level changed while the panel was hidden."""
        self._sync_laptop()
        super().showEvent(event)

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

    def _laptop_waits(self, step: str) -> None:
        """Say on a tab that laptop mode waits for Refresh."""
        self._pages[step].mark_pending(tr(
            "Laptop mode: press Refresh view to run this tab."))

    def set_live(self, on: bool) -> None:
        """Follow the Live switch: run the tab on view when on, stop when off.

        In laptop mode switching Live on runs the tab on view once; after
        that, changes wait for Refresh view.
        """
        self._live = bool(on)
        if not self._live:
            self._timer.stop()
            self._pending = None
            if self.preview_running():
                self.cancel_preview()
            self._clear_scheduled()
            return
        self._sync_laptop()
        if self.current_step() in self._stale:
            self._schedule(self.current_step())

    def apply_settings(self, settings: Mapping[str, Any]) -> None:
        """Take a whole new set of settings: every tab is stale.

        :param settings: the form's settings; read again from the form at
            each pass, so only the staleness is kept. The panels' own
            changes are dropped, since the form now says what to run.
        """
        self._stale = set(_STEP_ORDER)
        self._overrides.clear()
        self._reload_dialogs()
        self._refresh_data_button()
        if self._live:
            if _laptop_mode():
                self._laptop_waits(self.current_step())
            else:
                self._schedule(self.current_step())

    def _mark_stale(self, key: Optional[str]) -> Tuple[str, ...]:
        """Mark a setting's tabs, and what reads from them, stale.

        :returns: the tabs the setting belongs to.
        """
        from spacr.ops_live import _STEPS, _downstream

        steps = self._step_map().get(key or "", _STEPS) if key else _STEPS
        hit = set(steps)
        for step in steps:
            hit.update(_downstream(step))
        self._stale |= hit
        return tuple(steps)

    def _reload_dialogs(self) -> None:
        """Show the settings now in every open Settings panel."""
        values = self.settings_now()
        for dialog in self._dialogs.values():
            dialog.load(values)

    def setting_committed(self, key: Optional[str]) -> None:
        """One setting changed on the form: its tab runs again.

        The setting's tab, and every tab that reads from it, are marked
        stale; the setting's own tab is brought forward and runs after the
        debounce. A setting every tab reads (an input folder, the GPU)
        leaves the tab on view in front. Nothing runs while Live is off, or
        in laptop mode until Refresh. The form's value replaces a Settings
        panel's change of the same setting.

        :param key: the setting, or None for a change of many at once.
        """
        if key is None:
            self._overrides.clear()
        else:
            self._overrides.pop(key, None)
        self._reload_dialogs()
        steps = self._mark_stale(key)
        if key in (None, "genotype_source", "phenotype_source"):
            self._refresh_data_button()
        if not self._live:
            return
        if len(steps) == 1 and steps[0] != self.current_step():
            blocked = self._tabs.blockSignals(True)
            self.show_step(steps[0])
            self._tabs.blockSignals(blocked)
        if self.current_step() in self._stale:
            if _laptop_mode():
                self._pages[self.current_step()].mark_pending(tr(
                    "Settings changed. Press Refresh view to run this tab."))
            else:
                self._schedule(self.current_step())

    def _on_tab_changed(self, _index: int) -> None:
        """A tab brought forward runs when its result is stale."""
        if not (self._live and self.current_step() in self._stale):
            return
        if _laptop_mode():
            self._laptop_waits(self.current_step())
            return
        self._schedule(self.current_step())

    def _schedule(self, step: str, forced: bool = False) -> None:
        """Run a tab after the debounce, showing at once that it will.

        :param step: the tab, which is the tab on view.
        :param forced: run even while Live is off (a Settings panel change).
        """
        self._forced = self._forced or forced
        self._pages[step].set_running(True, tr("Starting {step}…", step=_title(step)))
        self._set_tab_running(step, True)
        self._timer.start()

    def _clear_scheduled(self) -> None:
        """Take the running indicator off every tab with no pass in flight."""
        running = self._worker.step_name if self._worker is not None else None
        for step, page in self._pages.items():
            if step != running and page.is_running():
                page.set_running(False, tr("Not run yet. Switch Live on, or press "
                                           "Run preview.") if step in self._stale
                                 else tr("Up to date."))
                self._set_tab_running(step, False)

    def _set_tab_running(self, step: str, running: bool) -> None:
        """Mark a tab's title while its pass is scheduled or running."""
        index = _STEP_ORDER.index(step)
        title = (tr("{step} (running)", step=_title(step)) if running
                 else _title(step))
        self._tabs.setTabText(index, title.replace("&", "&&"))

    def _refresh_current(self) -> None:
        """The debounce fired: run the tab on view."""
        forced, self._forced = self._forced, False
        if self._live or forced:
            self._start(self.current_step())
        else:
            self._clear_scheduled()

    def _on_run_requested(self, step: str) -> None:
        """A tab pressed its Run preview: run that tab now."""
        if step != self.current_step():
            self.show_step(step)
        self.run_preview()

    def run_preview(self) -> None:
        """Run the tab on view now, Live or not."""
        self._timer.stop()
        self._forced = False
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
            self._pages[step].set_running(True, tr("Starting {step}…",
                                                   step=_title(step)))
            self._set_tab_running(step, True)
            return
        if not self.begin_preview():
            self._pages[step].set_running(False, self.preview_status())
            self._set_tab_running(step, False)
            return
        self._pending = None
        self._cancel_event = threading.Event()
        self.runs.append(step)
        self.set_preview_status(" · ".join((_title(step),
                                            tr(PREVIEW_RUNNING_MESSAGE))))
        self._pages[step].set_running(True)
        self._set_tab_running(step, True)
        worker = _StepWorker(self.preview_token(), step, self.settings_now(),
                             self._cache, self._step_map(),
                             self._cancel_event, self._run_step, self)
        worker.step_name = step
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
            page.set_running(False, tr("{step} failed.", step=_title(step)))
            self.set_preview_status(preview_failure_message(error))
            return
        if payload is not None and "not_ready" in payload:
            text = _not_ready_text(step, payload["not_ready"], payload.get("detail") or {})
            page.show_notice(text)
            page.set_running(False, tr("{step} is waiting for its input.",
                                       step=_title(step)))
            self.set_preview_status(text)
            return
        page.show_result(payload)
        self._stale.discard(step)
        page.set_running(False, tr("Done in {seconds} s.",
                                   seconds=payload.get("seconds")))
        self.set_preview_status(tr("{step}: done in {seconds} s.",
                                   step=_title(step), seconds=payload.get("seconds")))

    def _on_finished(self) -> None:
        """The worker's thread ended: free it, then start what was queued."""
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        if worker is not None:
            self._retired.append(worker)
            step = getattr(worker, "step_name", None)
            if step in self._pages and step != self._pending:
                if self._pages[step].is_running():
                    self._pages[step].set_running(False, tr("Cancelled."))
                self._set_tab_running(step, False)
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
        self._forced = False
        self._cancel_event.set()
        cancelled = LivePreviewContract.cancel_preview(self)
        for step, page in self._pages.items():
            if page.is_running():
                page.set_running(False, tr("Cancelled."))
            self._set_tab_running(step, False)
        return cancelled

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
        for dialog in list(self._dialogs.values()):
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._dialogs.clear()
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
    apply = getattr(host, "apply_settings_dict", None)
    if callable(apply):
        panel.set_propagate_callback(apply)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(420)
    return panel, card
