"""
Extraction-preview bridge.

Turns a described dataset — either a *container file* (nd2 / czi / lif /
multi-page tiff / npz) inspected by :mod:`spacr.qt.multi_format`, or a
*folder-structured* layout recognised by :mod:`spacr.qt.folder_metadata`
— into a flat list of the individual image "planes" it would expand to,
**without reading any pixel data**.

Each plane is a plain dict row::

    {"original": <source path or series>,
     "plate":    "plate1",
     "well":     "plate1_A01",
     "field":    1,
     "channel":  1,
     "time":     1,
     "canonical": "plate1_A01_T0001F001L01C01.tif"}

The canonical names match :func:`spacr.io.convert_to_yokogawa` (the
pipeline's own container extractor) so the preview the user edits is the
layout the extraction will actually produce. The rows feed the editable
metadata table (:mod:`spacr.qt.widgets.metadata_table`) and can be written
to a ``filename_map.csv`` via :func:`rows_to_mappings` +
:func:`spacr.qt.folder_metadata.save_filename_map`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# The columns every preview row carries, in table order.
ROW_COLUMNS = ("original", "plate", "well", "field", "channel", "time", "canonical")


def _yokogawa_name(well: str, time: int, field: int, channel: int) -> str:
    """Return the Yokogawa-style filename used by ``convert_to_yokogawa``.

    e.g. ``plate1_A01_T0001F001L01C01.tif``. ``well`` already includes the
    plate prefix (``plate1_A01``) to match the pipeline converter.
    """
    return f"{well}_T{time:04d}F{field:03d}L01C{channel:02d}.tif"


def plan_container_extraction(desc: Any, plate: str = "plate1",
                              well: str = "A01") -> List[Dict[str, Any]]:
    """Enumerate the planes a container file would expand into.

    Mirrors :func:`spacr.io.convert_to_yokogawa`: a single container file
    is assigned one well, and its fields / channels / timepoints become the
    ``F`` / ``C`` / ``T`` indices of the generated TIFFs. Z-slices are
    max-projected (MIP) by the converter, so they are *not* enumerated
    here.

    :param desc: a ``DatasetDescription`` (needs ``n_fields``,
        ``n_channels``, ``n_timepoints`` and ``path``).
    :param plate: plate id for the canonical name.
    :param well: bare well id (``A01``); combined with ``plate`` into the
        Yokogawa well token ``plate1_A01``.
    :returns: one row dict per (time, field, channel) plane.
    """
    n_fields = max(1, int(getattr(desc, "n_fields", 1) or 1))
    n_channels = max(1, int(getattr(desc, "n_channels", 1) or 1))
    n_times = max(1, int(getattr(desc, "n_timepoints", 1) or 1))
    src = str(getattr(desc, "path", ""))
    well_token = f"{plate}_{well}"

    rows: List[Dict[str, Any]] = []
    for t in range(1, n_times + 1):
        for f in range(1, n_fields + 1):
            for c in range(1, n_channels + 1):
                rows.append({
                    "original": src,
                    "plate": plate,
                    "well": well_token,
                    "field": f,
                    "channel": c,
                    "time": t,
                    "canonical": _yokogawa_name(well_token, t, f, c),
                })
    return rows


def plan_folder_extraction(root: Any, plate: str = "plate1",
                           limit: Optional[int] = 200
                           ) -> List[Dict[str, Any]]:
    """Enumerate the planes a folder-structured dataset would map to.

    Uses :func:`spacr.qt.folder_metadata.detect_folder_metadata` to decide
    which fields the folder tree already provides, then
    :func:`spacr.qt.folder_metadata.assign_missing_fields` to mint the rest
    (stable, sorted order). Every image file becomes one row.

    :param root: dropped folder.
    :param plate: plate id used in the canonical names.
    :param limit: cap on the number of rows returned (the table only needs
        a representative preview). ``None`` for no cap.
    :returns: one row dict per source image, or ``[]`` if nothing matched.
    """
    from . import folder_metadata as fm

    root = Path(root)
    if not root.is_dir():
        return []

    template = fm.detect_folder_metadata(root)
    labels = tuple(getattr(template, "depth_labels", ()) or ()) if template else ()
    have_well = "well" in labels
    have_field = "field" in labels
    have_channel = ("channel" in labels
                    or bool(getattr(template, "chan_from_filename", False)))

    files = sorted(_iter_all_images(root))
    if limit is not None:
        files = files[:limit]
    if not files:
        return []

    mappings = fm.assign_missing_fields(
        files, plate=plate,
        have_well=have_well, have_field=have_field,
        have_channel=have_channel,
    )
    return [mapping_to_row(m) for m in mappings]


def _iter_all_images(root: Path) -> List[Path]:
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts]


def mapping_to_row(m: Any) -> Dict[str, Any]:
    """Convert a :class:`spacr.qt.folder_metadata.NameMapping` to a row dict."""
    return {
        "original": getattr(m, "original_path", ""),
        "plate": getattr(m, "plate", "plate1"),
        "well": getattr(m, "well", ""),
        "field": int(getattr(m, "field", 1)),
        "channel": int(getattr(m, "channel", 1)),
        "time": int(getattr(m, "time", 1)),
        "canonical": getattr(m, "canonical", ""),
    }


def rows_to_mappings(rows: Sequence[Dict[str, Any]]) -> List[Any]:
    """Convert edited table rows back into ``NameMapping`` objects ready
    for :func:`spacr.qt.folder_metadata.save_filename_map`."""
    from .folder_metadata import NameMapping
    out: List[NameMapping] = []
    for r in rows:
        out.append(NameMapping(
            original_path=str(r.get("original", "")),
            canonical=str(r.get("canonical", "")),
            plate=str(r.get("plate", "plate1")),
            well=str(r.get("well", "")),
            field=int(r.get("field", 1) or 1),
            channel=int(r.get("channel", 1) or 1),
            time=int(r.get("time", 1) or 1),
        ))
    return out


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> str:
    """One-line count summary of a preview (wells / fields / channels)."""
    if not rows:
        return "no images to extract"
    wells = {r.get("well") for r in rows}
    fields = {r.get("field") for r in rows}
    channels = {r.get("channel") for r in rows}
    times = {r.get("time") for r in rows}
    parts = [f"{len(rows)} images",
             f"{len(wells)} well(s)",
             f"{len(fields)} field(s)",
             f"{len(channels)} channel(s)"]
    if len(times) > 1:
        parts.append(f"{len(times)} timepoint(s)")
    return ", ".join(parts)
