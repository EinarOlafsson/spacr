"""
Folder-structure metadata + auto-generated field ids.

Some datasets don't carry metadata in the filename — instead the
folder tree encodes it (``plate1/A01/field_01/ch1.tif``). This
module handles both:

* :func:`detect_folder_metadata` — walk the tree, propose a folder
  template that captures plate / well / field / channel.
* :func:`assign_missing_fields` — for datasets that have no filename
  metadata AND no folder metadata, mint fresh ``wellID`` /
  ``fieldID`` / ``chanID`` values in a stable order and emit a
  ``filename_map.csv`` linking the original file paths to the
  canonical spaCR names (``<plate>_<well>_F<field>_C<channel>.tif``).

The mapping CSV format is intentionally compatible with the one
:mod:`spacr.pipeline_v2` writes so the two flows share downstream
tooling.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.qt.folder_metadata")

# 384-well plate convention: 16 rows × 24 columns (A01 … P24).
WELL_ROWS = list(ascii_uppercase[:16])   # A..P
WELL_COLS = list(range(1, 25))           # 1..24


# ---------------------------------------------------------------------------
# Folder metadata detection
# ---------------------------------------------------------------------------

@dataclass
class FolderTemplate:
    """One inferred folder metadata layout.

    :ivar depth_labels: e.g. ``("plate", "well", "field")`` — describes
        which subfolder level maps to which spaCR field.
    :ivar sample_paths: a few original files that fit the layout.
    :ivar chan_from_filename: True if the LEAF filename carries the
        channel id (e.g. ``ch1.tif``).
    """
    depth_labels:       Tuple[str, ...]
    sample_paths:       Tuple[Path, ...]
    chan_from_filename: bool


# Heuristic recognisers for common folder tokens
_WELL_RX     = re.compile(r"^[A-P](\d{1,3})$", re.I)
_FIELD_RX    = re.compile(r"^(?:F|field|fld|position)[_-]?(\d+)$", re.I)
_CHANNEL_RX  = re.compile(r"^(?:C|ch|channel)[_-]?(\d+)$", re.I)
_PLATE_RX    = re.compile(r"^plate[_-]?\d*$", re.I)


def _classify(token: str) -> Optional[str]:
    # Order matters — the FIELD / CHANNEL / PLATE recognisers use
    # explicit prefixes so they're strictly more specific than the
    # generic well pattern `[A-P]\d{1,3}` (which would otherwise
    # swallow `F01`, `C01`, `Z01`, etc.). Check them first.
    if _FIELD_RX.match(token):   return "field"
    if _CHANNEL_RX.match(token): return "channel"
    if _PLATE_RX.match(token):   return "plate"
    if _WELL_RX.match(token):    return "well"
    return None


def detect_folder_metadata(root: Path, max_probe: int = 30
                             ) -> Optional[FolderTemplate]:
    """Walk ``root``, try to recognise a folder-structured layout.

    :param root: dropped folder.
    :param max_probe: cap on files inspected — the layout should
        repeat, so no need to walk millions.
    :returns: a :class:`FolderTemplate` describing the detected
        layout, or None if we can't infer one.
    """
    root = Path(root)
    if not root.is_dir():
        return None

    matches: List[Tuple[Path, List[str]]] = []
    for p in _iter_image_files(root, cap=max_probe):
        rel = p.relative_to(root)
        parts = list(rel.parts[:-1])   # drop filename
        labels = [_classify(part) for part in parts]
        if any(labels):
            matches.append((p, [l for l in labels if l is not None]))
        if len(matches) >= max_probe:
            break

    if not matches:
        return None

    # Take the modal label sequence
    sequences: Dict[Tuple[str, ...], List[Path]] = {}
    for path, labels in matches:
        sequences.setdefault(tuple(labels), []).append(path)
    best_seq, sample = max(sequences.items(), key=lambda kv: len(kv[1]))

    # Is the leaf filename encoding the channel?
    chan_from_filename = any(
        _CHANNEL_RX.match(p.stem) or "ch" in p.stem.lower()
        for p in sample[:5]
    )
    return FolderTemplate(
        depth_labels=best_seq,
        sample_paths=tuple(sample[:5]),
        chan_from_filename=chan_from_filename,
    )


def _iter_image_files(root: Path, cap: int = 30):
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    seen = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
            seen += 1
            if seen >= cap:
                break


# ---------------------------------------------------------------------------
# Auto-generate missing well / field ids + write filename_map.csv
# ---------------------------------------------------------------------------

@dataclass
class NameMapping:
    """One row of the generated filename_map.csv."""
    original_path: str
    canonical:     str
    plate:         str
    well:          str
    field:         int
    channel:       int
    time:          int = 1


def _well_from_index(idx: int) -> str:
    """Return the 384-well-plate name for ``idx`` (0-based, row-major)."""
    row = idx // len(WELL_COLS)
    col = idx % len(WELL_COLS) + 1
    if row >= len(WELL_ROWS):
        # Fall through to a plain running counter once we exhaust
        # A01..P24 — real 384-well runs never do this.
        return f"E{idx + 1:03d}"
    return f"{WELL_ROWS[row]}{col:02d}"


def assign_missing_fields(
    filenames: Sequence[Path],
    plate: str = "plate1",
    have_well: bool = False,
    have_field: bool = False,
    have_channel: bool = True,
) -> List[NameMapping]:
    """Mint synthetic ``wellID`` / ``fieldID`` / ``chanID`` values for
    filenames that lack them.

    Filenames are grouped into "sets" of (well × field) — one entry
    per file. Ordering is stable (sorted alphabetically) so re-running
    on the same folder yields the same canonical names.

    :param filenames: absolute paths to the source images.
    :param plate: plate id used in the canonical name.
    :param have_well: True if the caller already knows well ids from
        elsewhere (folder structure); False to auto-assign A01, A02, …
    :param have_field: True if the caller knows field ids.
    :param have_channel: True when we can extract channels from the
        filename or the folder — otherwise every file is treated as
        channel 1.
    :returns: list of :class:`NameMapping`, in the same order as
        ``filenames`` after sort.
    """
    files = sorted(filenames)
    mappings: List[NameMapping] = []
    well_idx = 0
    field_idx = 1
    for i, src in enumerate(files):
        well = "A01" if have_well else _well_from_index(well_idx)
        fld = 1 if have_field else field_idx
        chan = 1 if not have_channel else 1
        canonical = f"{plate}_{well}_F{fld:03d}_C{chan:02d}.tif"
        mappings.append(NameMapping(
            original_path=str(src),
            canonical=canonical,
            plate=plate, well=well, field=fld, channel=chan,
        ))
        # Advance
        if not have_field:
            field_idx += 1
            if field_idx > 999:
                field_idx = 1
                well_idx += 1
        elif not have_well:
            well_idx += 1
    return mappings


def save_filename_map(dst: Path,
                        mappings: Sequence[NameMapping]) -> Path:
    """Write ``mappings`` to ``dst`` as a CSV Excel opens cleanly.

    Columns: ``original_path, canonical, plate, well, field, channel, time``.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cols = ["original_path", "canonical", "plate", "well",
            "field", "channel", "time"]
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in mappings:
            w.writerow([m.original_path, m.canonical, m.plate,
                        m.well, m.field, m.channel, m.time])
    LOG.info("wrote %d filename mappings → %s", len(mappings), dst)
    return dst
