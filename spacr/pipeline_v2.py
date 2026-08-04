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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_util import Timer, timed

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


@timed
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

        # Determine the field's true plane shape/dtype from ANY present
        # channel first, so a zero plane synthesised for a missing channel
        # matches — even when the missing channel is the FIRST requested one
        # (otherwise np.stack raises "all input arrays must have the same
        # shape"). Cache the read so present planes aren't read twice.
        ref_shape = None
        ref_dtype = np.uint16
        read_cache: dict = {}
        for ch in channels:
            rec = by_ch.get(ch)
            if rec is not None:
                plane = _read_plane(rec.original_path)
                read_cache[ch] = plane
                ref_shape = plane.shape
                ref_dtype = plane.dtype
                break
        if ref_shape is None:
            ref_shape = (256, 256)

        planes: List[np.ndarray] = []
        for ch in channels:
            rec = by_ch.get(ch)
            if rec is None:
                # Missing channel — synthesise a zero plane matching the
                # field's real shape so downstream tools don't crash.
                LOG.warning("field %s missing channel %d — inserting zeros",
                             field_id, ch)
                planes.append(np.zeros(ref_shape, dtype=ref_dtype))
                continue
            plane = read_cache.get(ch)
            if plane is None:
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


def _record_cellpose_hash(model, model_name: str) -> None:
    """Best-effort — fingerprint the Cellpose checkpoint and record it
    on the currently-open :class:`spacr.run_journal.Run`, if any."""
    try:
        # Cellpose's model object usually exposes `pretrained_model`
        # (list of paths) or `.cp.pretrained_model`.
        ckpt_paths = []
        for attr in ("pretrained_model", "cp"):
            obj = getattr(model, attr, None)
            if obj is None:
                continue
            if isinstance(obj, (list, tuple)):
                ckpt_paths.extend(obj)
            else:
                nested = getattr(obj, "pretrained_model", None)
                if nested is not None:
                    if isinstance(nested, (list, tuple)):
                        ckpt_paths.extend(nested)
                    else:
                        ckpt_paths.append(nested)
        # Filter to real existing files
        ckpt_paths = [Path(p) for p in ckpt_paths
                       if p and Path(p).is_file()]
        if not ckpt_paths:
            return
        # Push to the OPEN run journal, if any. We do this via a
        # thread-local convenience — see spacr.run_journal for the
        # active-run registry.
        try:
            from .run_journal import current_run
            run = current_run()
            if run is None:
                return
            for ckpt in ckpt_paths:
                run.record_model(model_name, ckpt)
        except Exception as exc:
            # Provenance, not results — a journal that will not take the
            # record must not stop the segmentation. But an unrecorded model
            # is a run whose manifest cannot say which weights produced the
            # masks, and that is exactly the question asked six months later.
            LOG.warning("model %r was not recorded in the run journal (%s); "
                        "this run's manifest will not name the weights it "
                        "used.", model_name, exc)
    except Exception as exc:
        LOG.warning("could not work out which checkpoint %r is using (%s); "
                    "no model provenance was recorded for this run.",
                    model_name, exc)


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

@timed
def stream_masks_from_stack(
    stacks: List[StackFile],
    model_name: str = "cyto",
    channels_for_cellpose: Sequence[int] = (0, 0),
    diameter: Optional[float] = None,
    batch_fields: int = 8,
    mask_channel_name: str = "mask",
    keep_npz: bool = False,
    npz_dir: Optional[Path] = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size: int = 15,
    resample: bool = True,
    postprocess_settings: Optional[Dict[str, Any]] = None,
    object_type: str = "cell",
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

    # Resolve legacy names and fine-tuned checkpoint paths through the same
    # adapter as V1. The old V2 branch silently loaded stock cpsam for every
    # model_name, so a V1 run using a trained checkpoint could never match.
    import torch
    from .utils import _resolve_cellpose_pretrained

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    pretrained = _resolve_cellpose_pretrained(
        model_name, object_type=object_type)
    model = cp_models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=pretrained,
        device=device,
    )

    # Record the exact model checkpoint hash into the active run
    # journal, if one is open. Downstream reviewers can then trace
    # any mask back to the specific weights that produced it.
    _record_cellpose_hash(model, model_name)

    for batch_start in range(0, len(stacks), batch_fields):
        batch = stacks[batch_start:batch_start + batch_fields]
        with Timer(
            f"v2.batch[{batch_start}:{batch_start + len(batch)}] "
            f"load", logger="spacr.pipeline_v2",
        ):
            loaded = [np.load(s.path) for s in batch]

        # Optionally persist the batch as NPZ for debugging.
        # Deleted after run unless keep_npz=True.
        npz_path = scratch / f"batch_{batch_start:04d}.npz"
        np.savez_compressed(
            npz_path,
            **{s.field_id: arr for s, arr in zip(batch, loaded)},
        )

        # Prepare the same list-of-images batch V1 hands to Cellpose. Besides
        # being faster, keeping the call boundary identical matters for exact
        # V1/V2 reproducibility on CPSAM.
        selected_images: List[np.ndarray] = []
        for arr in loaded:
            if arr.ndim == 3:
                indices = [
                    int(channel) % arr.shape[-1]
                    for channel in channels_for_cellpose
                ]
                indices = list(dict.fromkeys(indices)) or [0]
                raw_img = arr[..., indices]
            else:
                raw_img = arr.squeeze()
            selected_images.append(raw_img)

        # V1 segments the percentile-normalised float batch under masks/*.npz,
        # not the raw uint16 planes later retained in merged/.  V2 deliberately
        # retains those raw planes, but must still present the same pixels to
        # Cellpose or small synthetic fields produce materially different
        # masks.  Reuse V1's normaliser with channel roles remapped onto this
        # compact Cellpose input (object first, optional nucleus second).
        if postprocess_settings is not None:
            from .io import _normalize_img_batch

            selected_images = [
                image[..., np.newaxis] if image.ndim == 2 else image
                for image in selected_images
            ]
            max_height = max(image.shape[0] for image in selected_images)
            max_width = max(image.shape[1] for image in selected_images)
            selected_images = [
                np.pad(
                    image,
                    (
                        (0, max_height - image.shape[0]),
                        (0, max_width - image.shape[1]),
                        (0, 0),
                    ),
                )
                for image in selected_images
            ]
            normalization_settings = dict(postprocess_settings)
            for role in ("cell", "nucleus", "pathogen", "organelle"):
                normalization_settings[f"{role}_channel"] = None
            normalization_settings[f"{object_type}_channel"] = 0
            if object_type == "cell" and selected_images[0].ndim == 3 \
                    and selected_images[0].shape[-1] > 1:
                normalization_settings["nucleus_channel"] = 1

            selected_stack = np.stack(selected_images).copy()
            normalized_stack = _normalize_img_batch(
                stack=selected_stack,
                channels=range(selected_stack.shape[-1]),
                save_dtype=np.float32,
                settings=normalization_settings,
            )
            intensity_per_field = [
                normalized_stack[index]
                for index in range(normalized_stack.shape[0])
            ]
        else:
            intensity_per_field = selected_images

        cellpose_images: List[np.ndarray] = []
        for intensity_img in intensity_per_field:
            img = np.asarray(intensity_img, dtype=np.float32)
            maximum = float(img.max()) if img.size else 0.0
            if maximum > 1:
                img = img / maximum
            cellpose_images.append(img)

        with Timer(
            f"v2.batch[{batch_start}:{batch_start + len(batch)}] "
            f"cellpose ({len(loaded)} fields)",
            logger="spacr.pipeline_v2",
        ):
            out = model.eval(
                cellpose_images,
                batch_size=len(cellpose_images),
                normalize=False,
                channel_axis=-1,
                min_size=int(min_size),
                progress=True,
                diameter=diameter,
                flow_threshold=float(flow_threshold),
                cellprob_threshold=float(cellprob_threshold),
                resample=bool(resample),
            )
            masks = out[0]
            if isinstance(masks, np.ndarray) and masks.ndim == 2:
                masks = [masks]
            masks_per_field = [
                np.asarray(mask).astype(np.uint16) for mask in masks
            ]

        if postprocess_settings is not None:
            from .object import merge_split_filter_masks
            masks_per_field = list(merge_split_filter_masks(
                masks=masks_per_field,
                intensity_images=intensity_per_field,
                settings=postprocess_settings,
                object_type=object_type,
                batch_filenames=[stack.path.name for stack in batch],
            ))

        # Append the mask channel to each stack file and update
        # the StackFile bookkeeping.
        for sf, arr, mask in zip(batch, loaded, masks_per_field):
            mask = np.asarray(mask)[:arr.shape[0], :arr.shape[1]]
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
        except Exception as exc:
            # The masks are written either way, so this does not fail the
            # stage — but channel_order.json is what every later reader uses
            # to know which plane is a mask, and a sidecar that silently did
            # not get the entry makes the stack self-describing and wrong.
            LOG.warning("channel_order.json at %s was not updated with "
                        "mask_channels=%r (%s); readers of this stack will "
                        "not know which plane holds the mask.",
                        sidecar, mask_channel_name, exc)

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
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size: int = 15,
    resample: bool = True,
    postprocess_settings: Optional[Dict[str, Any]] = None,
    object_type: str = "cell",
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
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        resample=resample,
        postprocess_settings=postprocess_settings,
        object_type=object_type,
    )
    return {"mapper": mapper, "stacks": stacks,
            "dst": src / "merged"}
