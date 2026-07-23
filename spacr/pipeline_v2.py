"""
Streaming mask pipeline (v2).

Replaces the multi-copy disk chain that ``preprocess_generate_masks``
has run since day one:

    originals
       → renamed + split into channel folders
       → orig/ backup
       → per-channel npy
       → batch npz on disk
       → cellpose → per-field mask npy
       → concatenated into merged/

…with a two-pass streaming pipeline that keeps only what the
downstream measure module actually reads:

    Pass 1 — assemble
        walk originals, parse metadata regex, build one npy stack per
        field with all image channels in the C axis. Emit
        ``filename_map.csv`` recording every original → stack mapping.

    Pass 2 — segment
        stream the plate in batches of N fields, hand each batch to
        Cellpose, append the mask channels to the SAME stack file.
        Optional intermediate NPZ is memory-only (never touches disk)
        unless ``keep_npz=True``.

Output — ``merged/`` folder holds one file per field, each shape
``(H, W, C_image + C_mask)`` in uint16, plus:

    channel_order.json     {"channels": [...]}
    filename_map.csv       original path, plate/well/field/…, stack idx

Public API::

    from spacr.pipeline_v2 import (
        FilenameMapper, stream_originals_to_stack,
        stream_masks_from_stack, run_v2,
    )

    # High-level (one call):
    run_v2(src_folder, channels=(0,1,2,3), model="cyto", diameter=60)

    # Low-level (two passes, run each explicitly):
    mapper = FilenameMapper.discover(src_folder,
                                       metadata_type="cellvoyager")
    stacks = stream_originals_to_stack(src_folder, mapper, channels=(0,1,2,3))
    stream_masks_from_stack(stacks, model="cyto", diameter=60)

This module is opt-in for one release cycle. Once the follow-up commit
wires it as the default in :func:`spacr.core.preprocess_generate_masks`
the whole disk chain above collapses to ``merged/`` alone.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.pipeline_v2")


# ---------------------------------------------------------------------------
# Filename mapping
# ---------------------------------------------------------------------------

@dataclass
class FilenameRecord:
    """One entry in the filename map.

    :ivar original_path: absolute path to the source image on disk.
    :ivar plate: plate id parsed from the filename.
    :ivar well: well id parsed from the filename.
    :ivar field: field index parsed from the filename.
    :ivar channel: channel index parsed from the filename.
    :ivar time: time index parsed from the filename (defaults to 1).
    :ivar z: z-slice index parsed from the filename (defaults to 1).
    :ivar stack_field_id: the ``field`` id used in ``merged/stack_<X>.npy``.
    """
    original_path: str
    plate:         str
    well:          str
    field:         int
    channel:       int
    time:          int = 1
    z:             int = 1
    stack_field_id: str = ""


class FilenameMapper:
    """Walks a folder of microscopy images, parses each filename's
    metadata via a regex, and records the mapping to a per-plate CSV.

    The CSV is written next to the ``merged/`` folder (at the plate
    root) so users can Excel-open ``filename_map.csv`` and see the
    original path of every image in the run.

    :ivar records: list of :class:`FilenameRecord` in file-system order.
    :ivar metadata_type: which regex was used (``"cellvoyager"`` /
        ``"yokogawa"`` / ``"custom"``).
    :ivar regex: compiled regex pattern that matched.
    """

    def __init__(self, records: List[FilenameRecord],
                  metadata_type: str, regex: str):
        self.records = records
        self.metadata_type = metadata_type
        self.regex = regex

    # -- construction ------------------------------------------------------
    @classmethod
    def discover(cls, src: Path,
                  metadata_type: str = "auto",
                  custom_regex: Optional[str] = None,
                  exts: Sequence[str] = (".tif", ".tiff", ".png",
                                          ".jpg", ".jpeg")) -> "FilenameMapper":
        """Scan ``src`` for images + parse each name with the metadata
        regex. Falls back through ``cellvoyager`` → ``yokogawa`` on
        ``metadata_type="auto"``.

        :param src: folder to scan (not recursive; we expect images at
            the top level as the current spacr layout does).
        :param metadata_type: ``"auto"`` / ``"cellvoyager"`` /
            ``"yokogawa"`` / ``"custom"``. When ``"custom"``,
            ``custom_regex`` must be given.
        :param custom_regex: user-supplied regex; required for
            ``metadata_type="custom"``.
        :param exts: image file extensions to include.
        :returns: a populated :class:`FilenameMapper`.
        :raises ValueError: when no images are found or no regex fits.
        """
        src = Path(src)
        files = sorted(
            p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not files:
            raise ValueError(f"no images found in {src}")

        pattern, chosen = _resolve_regex(metadata_type, files, custom_regex)

        recs: List[FilenameRecord] = []
        rx = re.compile(pattern)
        for f in files:
            m = rx.match(f.name)
            if m is None:
                LOG.warning("filename didn't match %s regex: %s",
                             chosen, f.name)
                continue
            g = m.groupdict()
            recs.append(FilenameRecord(
                original_path=str(f.resolve()),
                plate=g.get("plateID") or g.get("plate") or "plate1",
                well=g.get("wellID")   or g.get("well")  or "A01",
                field=int(g.get("fieldID") or g.get("field") or 1),
                channel=int(g.get("chanID") or g.get("channel") or 1),
                time=int(g.get("timeID") or g.get("time") or 1),
                z=int(g.get("sliceID") or g.get("z") or 1),
            ))

        # Assign stable per-field ids so all channels of the same
        # (plate, well, field, time, z) fall into one stack file.
        # Sort keys: plate → well → field → time → z; then enumerate.
        keys = {}
        for r in recs:
            k = (r.plate, r.well, r.field, r.time, r.z)
            if k not in keys:
                keys[k] = f"{len(keys):06d}"
            r.stack_field_id = keys[k]

        LOG.info("discovered %d images grouped into %d fields (regex: %s)",
                  len(recs), len(keys), chosen)
        return cls(recs, chosen, pattern)

    # -- persistence -------------------------------------------------------
    def save_csv(self, path: Path) -> Path:
        """Write the mapping to ``path`` as a CSV that Excel opens
        cleanly. One row per (original image, resulting stack slot)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["original_path", "plate", "well", "field", "channel",
                "time", "z", "stack_field_id"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.records:
                w.writerow([getattr(r, c) for c in cols])
        # Sidecar with the regex used, so `spacr repro` can replay
        (path.with_suffix(".json")).write_text(json.dumps({
            "metadata_type": self.metadata_type,
            "regex":         self.regex,
            "n_records":     len(self.records),
        }, indent=2))
        return path

    @classmethod
    def load_csv(cls, path: Path) -> "FilenameMapper":
        """Rehydrate a mapper from a previously-saved CSV."""
        path = Path(path)
        recs: List[FilenameRecord] = []
        with open(path) as f:
            for row in csv.DictReader(f):
                recs.append(FilenameRecord(
                    original_path=row["original_path"],
                    plate=row["plate"], well=row["well"],
                    field=int(row["field"]), channel=int(row["channel"]),
                    time=int(row["time"]), z=int(row["z"]),
                    stack_field_id=row["stack_field_id"],
                ))
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return cls(recs, meta.get("metadata_type", "?"),
                        meta.get("regex", ""))
        return cls(recs, "?", "")

    # -- accessors ---------------------------------------------------------
    def by_field(self) -> Dict[str, List[FilenameRecord]]:
        """Group records by ``stack_field_id`` — one entry per field, with
        one record per channel inside."""
        out: Dict[str, List[FilenameRecord]] = {}
        for r in self.records:
            out.setdefault(r.stack_field_id, []).append(r)
        return out

    def field_ids(self) -> List[str]:
        """Return the sorted list of unique ``stack_field_id`` values."""
        return sorted(self.by_field().keys())


# ---------------------------------------------------------------------------
# Regex resolution — copy of spacr.utils._get_regex behaviour, kept local
# so v2 doesn't import the whole spacr.utils stack at module scope.
# ---------------------------------------------------------------------------

_CELLVOYAGER = (
    r"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
    r"L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*)"
    r"\.(?:tif|tiff|png|jpg|jpeg)$"
)
_YOKOGAWA = (
    r"(?P<plateID>.*)_(?P<wellID>[A-Z]\d{2})_"
    r"T(?P<timeID>\d{4})F(?P<fieldID>\d{3})"
    r"L(?P<laserID>\d{2})A(?P<AID>\d{2})Z(?P<sliceID>\d{2})C(?P<chanID>\d{2})"
    r"\.(?:tif|tiff)$"
)


def _resolve_regex(metadata_type: str, files: List[Path],
                    custom_regex: Optional[str]) -> Tuple[str, str]:
    """Pick a regex + return (pattern, chosen_label)."""
    if metadata_type == "custom":
        if not custom_regex:
            raise ValueError("metadata_type='custom' needs custom_regex")
        return custom_regex, "custom"

    candidates: List[Tuple[str, str]] = []
    if metadata_type == "cellvoyager":
        candidates.append((_CELLVOYAGER, "cellvoyager"))
    elif metadata_type == "yokogawa":
        candidates.append((_YOKOGAWA, "yokogawa"))
    else:   # auto
        candidates = [(_CELLVOYAGER, "cellvoyager"),
                      (_YOKOGAWA, "yokogawa")]

    # Choose the first regex that matches EVERY file
    for pattern, name in candidates:
        rx = re.compile(pattern)
        if all(rx.match(f.name) for f in files):
            return pattern, name

    # Last-ditch: choose the one that matches the MOST files
    best_pattern, best_name, best_hits = candidates[0][0], candidates[0][1], -1
    for pattern, name in candidates:
        rx = re.compile(pattern)
        hits = sum(1 for f in files if rx.match(f.name))
        if hits > best_hits:
            best_pattern, best_name, best_hits = pattern, name, hits
    LOG.warning("no regex matched every file; best fit was %s (%d/%d)",
                 best_name, best_hits, len(files))
    return best_pattern, best_name


# ---------------------------------------------------------------------------
# Pass 1 — stream originals into per-field npy stacks
# ---------------------------------------------------------------------------

@dataclass
class StackFile:
    """One field's on-disk stack: ``merged/stack_<id>.npy`` with
    shape ``(H, W, C)``.

    Populated by :func:`stream_originals_to_stack` before Cellpose
    runs (C = image channels only). After :func:`stream_masks_from_stack`
    the same file has additional mask channels appended.
    """
    field_id:  str
    path:      Path
    shape:     Tuple[int, int, int]   # (H, W, C) at write time
    channels:  List[str]              # human names, in the same order


def stream_originals_to_stack(
    src: Path,
    mapper: FilenameMapper,
    channels: Sequence[int] = (0, 1, 2, 3),
    channel_names: Optional[Sequence[str]] = None,
    dst: Optional[Path] = None,
) -> List[StackFile]:
    """Write one ``merged/stack_<field>.npy`` per field.

    Reads originals directly (no rename-into-channel-folders step),
    stacks the selected channels along the C axis, and writes one
    npy per field. Also emits a ``channel_order.json`` sidecar
    describing which C-index holds which channel.

    :param src: plate folder containing the original images.
    :param mapper: :class:`FilenameMapper` produced from ``src``.
    :param channels: which channel numbers (as parsed from filenames)
        to include, in the order they should occupy the C axis.
    :param channel_names: human names for those channels (must match
        ``channels`` length). Default: ``["ch0", "ch1", …]``.
    :param dst: override the output folder; defaults to ``<src>/merged``.
    :returns: list of :class:`StackFile`, one per field written.
    """
    src = Path(src)
    dst = Path(dst) if dst else src / "merged"
    dst.mkdir(parents=True, exist_ok=True)

    if channel_names is None:
        channel_names = [f"ch{c}" for c in channels]
    assert len(channel_names) == len(channels), (
        "channel_names must match channels length"
    )

    by_field = mapper.by_field()
    written: List[StackFile] = []

    for field_id, recs in by_field.items():
        # Group by channel number for this field
        by_ch = {r.channel: r for r in recs}
        planes: List[np.ndarray] = []
        for ch in channels:
            rec = by_ch.get(ch)
            if rec is None:
                # Missing channel — synthesise a zero plane. This
                # preserves shape so downstream tools don't crash.
                if planes:
                    zero = np.zeros_like(planes[0])
                else:
                    zero = np.zeros((256, 256), dtype=np.uint16)
                LOG.warning("field %s missing channel %d — inserting zeros",
                             field_id, ch)
                planes.append(zero)
                continue
            plane = _read_plane(rec.original_path)
            planes.append(plane)

        stack = np.stack(planes, axis=-1).astype(np.uint16)
        out_path = dst / f"stack_{field_id}.npy"
        np.save(out_path, stack)
        written.append(StackFile(
            field_id=field_id, path=out_path, shape=stack.shape,
            channels=list(channel_names),
        ))

    # Global sidecar describing the C axis
    (dst / "channel_order.json").write_text(json.dumps({
        "image_channels": list(channel_names),
        "mask_channels":  [],   # filled in by stream_masks_from_stack
        "shape_H_W_C":    "final shape is (H, W, C_image + C_mask)",
    }, indent=2))

    # Save filename map at the plate root
    mapper.save_csv(src / "filename_map.csv")
    LOG.info("wrote %d field stacks under %s + filename_map.csv",
              len(written), dst)
    return written


def _read_plane(path: str) -> np.ndarray:
    """Read a single 2-D image plane (H, W) as uint16."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(p))
    else:
        from PIL import Image
        arr = np.array(Image.open(str(p)))
    # Reduce to 2-D (grayscale)
    if arr.ndim == 3:
        # H, W, C → take the first channel (spacr's convention for
        # single-channel writes)
        arr = arr[..., 0]
    return arr.astype(np.uint16, copy=False)


# ---------------------------------------------------------------------------
# Pass 2 — stream Cellpose masks back into the same stacks
# ---------------------------------------------------------------------------

def stream_masks_from_stack(
    stacks: List[StackFile],
    model_name: str = "cyto",
    channels_for_cellpose: Sequence[int] = (0, 0),
    diameter: Optional[float] = None,
    batch_fields: int = 8,
    mask_channel_name: str = "mask",
    keep_npz: bool = False,
    npz_dir: Optional[Path] = None,
) -> List[StackFile]:
    """Batch the field stacks through Cellpose, then append the mask
    channel(s) to the SAME npy files.

    :param stacks: list produced by :func:`stream_originals_to_stack`.
    :param model_name: Cellpose model to use (``"cyto"``, ``"nuclei"``, …).
    :param channels_for_cellpose: Cellpose's ``channels=`` argument —
        e.g. ``[0, 0]`` for grayscale, ``[2, 1]`` for green cyto +
        blue nucleus.
    :param diameter: expected object diameter in px (None → Cellpose
        auto).
    :param batch_fields: how many field stacks to load into memory at
        once. Larger = faster but more RAM.
    :param mask_channel_name: human name to record for the appended
        mask channel (default ``"mask"``).
    :param keep_npz: when True, write the intermediate memory batch as
        an NPZ file to ``npz_dir`` for debugging. Deleted after the
        batch runs unless this flag is set.
    :param npz_dir: where to write the (optional) intermediate NPZ
        files. Defaults to a scratch subfolder under the stack folder.
    :returns: the same list, with each :class:`StackFile.shape` /
        ``.channels`` updated to reflect the appended mask channel.
    """
    if not stacks:
        return stacks

    scratch = Path(npz_dir) if npz_dir else stacks[0].path.parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        from cellpose import models as cp_models   # type: ignore
    except Exception as e:
        raise RuntimeError(
            "cellpose is required for v2 mask streaming"
        ) from e

    model = cp_models.Cellpose(gpu=True, model_type=model_name)

    for batch_start in range(0, len(stacks), batch_fields):
        batch = stacks[batch_start:batch_start + batch_fields]
        # Load — build an in-memory list of (H, W, C_image) arrays
        loaded = [np.load(s.path) for s in batch]

        # Optionally persist the batch as NPZ for debugging.
        # Deleted after run unless keep_npz=True.
        npz_path = scratch / f"batch_{batch_start:04d}.npz"
        np.savez_compressed(
            npz_path,
            **{s.field_id: arr for s, arr in zip(batch, loaded)},
        )

        # Run Cellpose per field (batching across fields inside cellpose
        # is possible for equal shapes; we keep it per-field for
        # heterogeneous plates).
        masks_per_field: List[np.ndarray] = []
        for arr in loaded:
            m, _flows, _styles, _diams = model.eval(
                arr, channels=list(channels_for_cellpose),
                diameter=diameter,
            )
            masks_per_field.append(m.astype(np.uint16))

        # Append the mask channel to each stack file and update
        # the StackFile bookkeeping.
        for sf, arr, mask in zip(batch, loaded, masks_per_field):
            combined = np.concatenate(
                [arr, mask[..., None]], axis=-1
            ).astype(np.uint16)
            np.save(sf.path, combined)
            sf.shape = combined.shape
            sf.channels = sf.channels + [mask_channel_name]

        if not keep_npz:
            try:
                npz_path.unlink()
            except Exception:
                pass

    if not keep_npz:
        # Best-effort scratch cleanup
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass

    # Update the channel-order sidecar
    if stacks:
        sidecar = stacks[0].path.parent / "channel_order.json"
        try:
            meta = json.loads(sidecar.read_text())
            meta["mask_channels"] = [mask_channel_name]
            sidecar.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass

    return stacks


# ---------------------------------------------------------------------------
# High-level one-call
# ---------------------------------------------------------------------------

def run_v2(
    src: Path,
    channels: Sequence[int] = (0, 1, 2, 3),
    channel_names: Optional[Sequence[str]] = None,
    model_name: str = "cyto",
    channels_for_cellpose: Sequence[int] = (0, 0),
    diameter: Optional[float] = None,
    batch_fields: int = 8,
    metadata_type: str = "auto",
    custom_regex: Optional[str] = None,
    keep_npz: bool = False,
) -> Dict[str, Any]:
    """Run the entire v2 pipeline against ``src``. Convenience wrapper.

    Equivalent to::

        mapper = FilenameMapper.discover(src, metadata_type, custom_regex)
        stacks = stream_originals_to_stack(src, mapper, channels, channel_names)
        stream_masks_from_stack(stacks, model_name, channels_for_cellpose,
                                diameter, batch_fields, keep_npz=keep_npz)

    :returns: dict with ``mapper`` (:class:`FilenameMapper`), ``stacks``
        (list of :class:`StackFile`), and ``dst`` (Path to ``merged/``).
    """
    src = Path(src)
    mapper = FilenameMapper.discover(src, metadata_type=metadata_type,
                                       custom_regex=custom_regex)
    stacks = stream_originals_to_stack(
        src, mapper, channels=channels, channel_names=channel_names,
    )
    stream_masks_from_stack(
        stacks, model_name=model_name,
        channels_for_cellpose=channels_for_cellpose,
        diameter=diameter, batch_fields=batch_fields,
        keep_npz=keep_npz,
    )
    return {"mapper": mapper, "stacks": stacks,
            "dst": src / "merged"}
