"""
Streaming mask pipeline (v2).

An alternative to the multi-copy disk chain that
``preprocess_generate_masks`` runs by default (``pipeline_style='v1'``):

    originals
       → renamed + split into channel folders
       → orig/ backup
       → per-channel npy
       → batch npz on disk
       → cellpose → per-field mask npy
       → concatenated into merged/

This module instead runs a two-pass streaming pipeline that keeps only
what the downstream measure module reads:

    Pass 1 — assemble
        walk originals, parse metadata regex, build one npy stack per
        field with all image channels in the C axis. Emit
        ``filename_map.csv`` recording every original → stack mapping.

    Pass 2 — segment
        stream the plate in batches of N fields, hand each batch to
        Cellpose, append the mask channels to the SAME stack file.
        Each batch is written to a scratch NPZ on the way through and
        deleted again unless ``keep_npz=True``.

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

This module is opt-in: :func:`spacr.core.preprocess_generate_masks` runs
it when ``pipeline_style='v2'``, and the default ``'v1'`` runs the disk
chain above.
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
_CALIBRATED_BATCH_VALUES = 64 * 1024 ** 2



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

    :param records: parsed filename records in file-system order.
    :param metadata_type: name of the metadata convention used to parse them.
    :param regex: regular-expression source used for parsing.
    :ivar records: list of :class:`FilenameRecord` in file-system order.
    :ivar metadata_type: which regex was used (``"cellvoyager"`` /
        ``"yokogawa"`` / ``"custom"``).
    :ivar regex: compiled regex pattern that matched.
    """

    def __init__(self, records: List[FilenameRecord],
                  metadata_type: str, regex: str):
        """Store parsed filename records and the metadata rule that made them."""
        self.records = records
        self.metadata_type = metadata_type
        self.regex = regex

    @classmethod
    def discover(cls, src: Path,
                  metadata_type: str = "auto",
                  custom_regex: Optional[str] = None,
                  exts: Sequence[str] = (".tif", ".tiff", ".png",
                                          ".jpg", ".jpeg")) -> "FilenameMapper":
        """Scan ``src`` for images + parse each name with the metadata
        regex. Falls back through ``cellvoyager`` → ``yokogawa`` on
        ``metadata_type="auto"``.

        :param src: folder to scan. Not recursive: images are expected at
            the top level, as in the current spacr layout.
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
            and not p.name.startswith(".")
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

        keys = {}
        for r in recs:
            k = (r.plate, r.well, r.field, r.time, r.z)
            if k not in keys:
                keys[k] = f"{len(keys):06d}"
            r.stack_field_id = keys[k]

        LOG.info("discovered %d images grouped into %d fields (regex: %s)",
                  len(recs), len(keys), chosen)
        return cls(recs, chosen, pattern)

    def save_csv(self, path: Path) -> Path:
        """Write the mapping to ``path`` as a CSV that Excel opens
        cleanly. One row per (original image, resulting stack slot).

        :param path: destination CSV path; its parent directory is created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["original_path", "plate", "well", "field", "channel",
                "time", "z", "stack_field_id"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.records:
                w.writerow([getattr(r, c) for c in cols])
        (path.with_suffix(".json")).write_text(json.dumps({
            "metadata_type": self.metadata_type,
            "regex":         self.regex,
            "n_records":     len(self.records),
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_csv(cls, path: Path) -> "FilenameMapper":
        """Rehydrate a mapper from a previously-saved CSV.

        :param path: mapping CSV previously written by :meth:`save_csv`.
        """
        path = Path(path)
        recs: List[FilenameRecord] = []
        with open(path, encoding="utf-8") as f:
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
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return cls(recs, meta.get("metadata_type", "?"),
                        meta.get("regex", ""))
        return cls(recs, "?", "")

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
    else:
        candidates = [(_CELLVOYAGER, "cellvoyager"),
                      (_YOKOGAWA, "yokogawa")]

    for pattern, name in candidates:
        rx = re.compile(pattern)
        if all(rx.match(f.name) for f in files):
            return pattern, name

    best_pattern, best_name, best_hits = candidates[0][0], candidates[0][1], -1
    for pattern, name in candidates:
        rx = re.compile(pattern)
        hits = sum(1 for f in files if rx.match(f.name))
        if hits > best_hits:
            best_pattern, best_name, best_hits = pattern, name, hits
    LOG.warning("no regex matched every file; best fit was %s (%d/%d)",
                 best_name, best_hits, len(files))
    return best_pattern, best_name



@dataclass
class StackFile:
    """One field's on-disk stack: ``merged/stack_<id>.npy`` with
    shape ``(H, W, C)``.

    Populated by :func:`stream_originals_to_stack` before Cellpose
    runs (C = image channels only). After :func:`stream_masks_from_stack`
    the same file has additional mask channels appended.

    :ivar field_id: stable field identifier used in the stack filename.
    :ivar path: path to the on-disk NumPy stack.
    :ivar shape: ``(height, width, channels)`` shape at write time.
    :ivar channels: human-readable channel names in array order.
    """
    field_id:  str
    path:      Path
    shape:     Tuple[int, int, int]
    channels:  List[str]


@timed
def stream_originals_to_stack(
    src: Path,
    mapper: FilenameMapper,
    channels: Sequence[int] = (0, 1, 2, 3),
    channel_names: Optional[Sequence[str]] = None,
    dst: Optional[Path] = None,
    *, metadata_root: Optional[Path] = None,
    calibrated: bool = False,
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
    :param metadata_root: mapping CSV destination, defaulting to the source folder.
    :param calibrated: preserve numeric planes and refuse missing, duplicate,
        non-2D or oversized channels for the separate calibrated route.
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
        by_ch = {r.channel: r for r in recs}
        if calibrated:
            from .qt.i18n import tr

            if len(by_ch) != len(recs) or any(ch not in by_ch for ch in channels):
                raise ValueError(tr("Calibrated segmentation requires every selected physical channel exactly once in each field."))

        ref_shape = None
        ref_dtype = np.uint16
        read_cache: dict = {}
        for ch in channels:
            rec = by_ch.get(ch)
            if rec is not None:
                plane = (_read_plane(rec.original_path, calibrated=True,
                                     max_pixels=(16 * 1024 ** 2) // len(channels))
                         if calibrated else _read_plane(rec.original_path))
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
                LOG.warning("field %s missing channel %d — inserting zeros",
                             field_id, ch)
                planes.append(np.zeros(ref_shape, dtype=ref_dtype))
                continue
            plane = read_cache.get(ch)
            if plane is None:
                plane = (_read_plane(rec.original_path, calibrated=True,
                                     max_pixels=(16 * 1024 ** 2) // len(channels))
                         if calibrated else _read_plane(rec.original_path))
            if calibrated and plane.shape != ref_shape:
                from .qt.i18n import tr

                raise ValueError(tr("Calibrated segmentation channels must share the same acquired YX shape."))
            planes.append(plane)

        stack = np.stack(planes, axis=-1)
        if not calibrated:
            stack = stack.astype(np.uint16)
        out_path = dst / f"stack_{field_id}.npy"
        np.save(out_path, stack)
        written.append(StackFile(
            field_id=field_id, path=out_path, shape=stack.shape,
            channels=list(channel_names),
        ))

    metadata = {
        "image_channels": list(channel_names),
        "mask_channels":  [],
        "shape_H_W_C":    "final shape is (H, W, C_image + C_mask)",
    }
    if calibrated:
        metadata.update(coordinate_system="calibrated_reference_canvas",
                        quantitative=False, image_dtype="float32",
                        mask_storage="uint16 label values stored exactly in float32",
                        source_channels=[int(channel) for channel in channels],
                        calibration_record="../psf/segmentation_application.json")
    (dst / "channel_order.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mapper.save_csv((Path(metadata_root) if metadata_root is not None else src)
                    / "filename_map.csv")
    LOG.info("wrote %d field stacks under %s + filename_map.csv",
              len(written), dst)
    return written


def _record_cellpose_hash(model, model_name: str) -> None:
    """Best-effort — fingerprint the Cellpose checkpoint and record it
    on the currently-open :class:`spacr.run_journal.Run`, if any."""
    try:
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
        ckpt_paths = [Path(p) for p in ckpt_paths
                       if p and Path(p).is_file()]
        if not ckpt_paths:
            return
        try:
            from .run_journal import current_run
            run = current_run()
            if run is None:
                return
            for ckpt in ckpt_paths:
                run.record_model(model_name, ckpt)
        except Exception as exc:
            LOG.warning("model %r was not recorded in the run journal (%s); "
                        "this run's manifest will not name the weights it "
                        "used.", model_name, exc)
    except Exception as exc:
        LOG.warning("could not work out which checkpoint %r is using (%s); "
                    "no model provenance was recorded for this run.",
                    model_name, exc)


def _read_plane(path: str, *, calibrated=False, max_pixels=None) -> np.ndarray:
    """Read a source plane, preserving the calibrated route's acquired values.

    :param path: source image path.
    :param calibrated: require a genuine single-channel YX image without casting.
    :param max_pixels: calibrated per-plane pixel limit checked before decoding.
    :returns: the acquired numeric plane, or the legacy uint16 view.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        if calibrated:
            from .qt.i18n import tr

            with tifffile.TiffFile(p) as image:
                shape = image.series[0].shape
                if len(image.series) != 1 or len(shape) != 2 or not all(shape):
                    raise ValueError(tr("Calibrated batch segmentation requires acquired single-channel 2D images."))
                if max_pixels is not None and int(np.prod(shape)) > max_pixels:
                    raise ValueError(tr("Calibrated segmentation exceeds the supported source-pixel budget."))
        arr = tifffile.imread(str(p))
    else:
        from PIL import Image
        with Image.open(str(p)) as image:
            if calibrated:
                from .qt.i18n import tr

                if len(image.getbands()) != 1 or getattr(image, "n_frames", 1) != 1:
                    raise ValueError(tr("Calibrated batch segmentation requires acquired single-channel 2D images."))
                if max_pixels is not None and image.width * image.height > max_pixels:
                    raise ValueError(tr("Calibrated segmentation exceeds the supported source-pixel budget."))
            arr = np.array(image)
    if calibrated:
        return arr
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint16, copy=False)




def _as_hwc(arr: np.ndarray) -> np.ndarray:
    """A loaded field as (H, W, C), whatever singleton axes it was saved with.

    A stack written as a bare plane is (H, W) on disk, and both ends of the
    mask pass need a channel axis: Cellpose is called with
    ``channel_axis=-1`` and refuses a 2-D image outright, and the write-back
    concatenates the mask onto this same array, which numpy refuses when the
    two disagree on rank. Promoting once, at load, keeps one shape convention
    for the whole pass instead of a squeeze that puts the mismatch off until
    the batch is already segmented.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr
    squeezed = arr.squeeze()
    if squeezed.ndim == 2:
        return squeezed[..., np.newaxis]
    return squeezed


def _cellpose_channel_indices(
        channels_for_cellpose: Sequence[int], n_channels: int
        ) -> Tuple[int, ...]:
    """Resolve requested channels into persisted C-axis positions once."""
    if n_channels <= 0:
        raise ValueError("a V2 field has no persisted intensity channels")
    indices = [
        int(channel) % n_channels for channel in channels_for_cellpose
    ]
    return tuple(dict.fromkeys(indices)) or (0,)


def _backend_mask_settings(settings: Optional[Dict[str, Any]],
                           object_type: str, *, diameter: Optional[float],
                           flow_threshold: float,
                           cellprob_threshold: float) -> Dict[str, Any]:
    """The settings a prefixed model's masks function reads, as V1 has them.

    V1 hands the run's settings to :func:`spacr.object._cellpose3_masks`
    and the object's magnification diameter as its default. The same run's
    settings arrive here as ``postprocess_settings``; the thresholds and
    diameter this pass was called with win over them, and the magnification
    default is kept under ``_default_diameter``.

    :param settings: the run's settings, or None.
    :param object_type: the object being segmented.
    :param diameter: the diameter this pass was called with; None keeps the
        object's own setting.
    :param flow_threshold: this pass's flow threshold.
    :param cellprob_threshold: this pass's cell probability threshold.
    :returns: a new settings dict.
    """
    merged = dict(settings or {})
    merged[f"{object_type}_flow_threshold"] = float(flow_threshold)
    merged[f"{object_type}_cellprob_threshold"] = float(cellprob_threshold)
    if diameter is not None:
        merged[f"{object_type}_diameter"] = diameter
    default = 30.0
    if merged.get("magnification") is not None:
        from .utils import _get_diam
        default = _get_diam(merged["magnification"], obj=object_type)
    elif diameter:
        default = float(diameter)
    merged["_default_diameter"] = default
    return merged


def _calibrated_label_array(mask):
    """Validate label precision before the calibrated float32 output is assembled.

    :param mask: predicted or postprocessed label array.
    :returns: uint16 labels exactly representing the validated input values.
    """
    from .qt.i18n import tr

    values = np.asarray(mask)
    if (values.ndim != 2 or np.iscomplexobj(values) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > np.iinfo(np.uint16).max)
            or np.any(values != np.floor(values))):
        raise ValueError(tr("Calibrated segmentation labels must be finite nonnegative integers representable exactly as uint16."))
    return values.astype(np.uint16)


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
    illumination_session: Optional[Any] = None,
    psf_session: Optional[Any] = None,
) -> List[StackFile]:
    """Batch the field stacks through Cellpose, then append the mask
    channel(s) to the SAME npy files.

    :param stacks: list produced by :func:`stream_originals_to_stack`.
    :param model_name: Cellpose model to use (``"cyto"``, ``"nuclei"``, …).
    :param channels_for_cellpose: C-axis indices selected out of each
        assembled stack before it is handed to Cellpose — taken modulo the
        channel count and de-duplicated in order. It is NOT forwarded as
        Cellpose's ``channels=`` argument; ``[0, 0]`` therefore yields a
        single plane, not a grayscale pair. Calibrated sessions instead require
        explicit in-range positions. The first selected channel is the object's
        own channel for absolute mean-intensity filtering.
    :param diameter: expected object diameter in px (None → Cellpose
        auto).
    :param batch_fields: how many field stacks to load into memory at
        once. Larger = faster but more RAM. Calibrated batches refuse above
        ``64 * 1024**2`` packed input or padded model-input values.
    :param mask_channel_name: human name to record for the appended
        mask channel (default ``"mask"``).
    :param keep_npz: the intermediate batch is compressed to an NPZ under
        ``npz_dir`` on every batch regardless; this flag only decides
        whether that file and the scratch folder survive the run.
    :param npz_dir: where to write the (optional) intermediate NPZ
        files. Defaults to a scratch subfolder under the stack folder.
    :param illumination_session: optional
        :class:`spacr.illumination.SegmentationIlluminationSession`. Its
        corrector sees private selected-channel copies immediately before
        normalisation/Cellpose; persisted intensity planes and scratch NPZs
        remain raw, and completion is recorded only after the combined stack
        has been atomically replaced.
    :param psf_session: optional captured PSF session. Unmixes each whole
        field first when unmixing is on, then processes selected intensities
        after illumination and before normalization, padding or Cellpose. Stored image channels stay raw; only the appended labels
        depend on PSF processing. A calibrated session instead supplies a
        signed valid crop for a nonnegative segmentation view, restores labels
        to the reference canvas and stores separate nonquantitative float32 display
        channels in its separate output tree, with signed crops under ``signed/``.
        These display channels are not Measure inputs; calibrated outputs cannot
        be resumed and completion requires both durable display and signed files.
    :returns: the same list, with each :class:`StackFile.shape` /
        ``.channels`` updated to reflect the appended mask channel.
    """
    if not stacks:
        return stacks
    calibrated = bool(getattr(psf_session, "calibrated", False))
    if calibrated:
        from .qt.i18n import tr
        from .psf_pipeline import _validate_calibrated_batch_settings

        _validate_calibrated_batch_settings(postprocess_settings or {})
        if object_type != "cell" or illumination_session is not None:
            raise ValueError(tr("Calibrated V2 segmentation requires cell objects without illumination correction."))
        expected_parent = (psf_session.root / "merged").resolve()
        if (expected_parent != psf_session.root.resolve() / "merged"
                or any(stack.path.resolve().parent != expected_parent for stack in stacks)):
            raise ValueError(tr("Calibrated segmentation can write only inside its separate output merged directory."))
        field_values = []
        for stack in stacks:
            field = np.load(stack.path, mmap_mode="r", allow_pickle=False)
            if (field.ndim != 3 or not all(field.shape)
                    or field.size > 16 * 1024 ** 2 or np.iscomplexobj(field)
                    or not np.isfinite(field).all()):
                raise ValueError(tr("Calibrated segmentation requires finite 2D channel-last fields within the supported source-pixel budget."))
            if (not channels_for_cellpose
                    or any(isinstance(channel, (bool, np.bool_))
                           or not isinstance(channel, (int, np.integer))
                           or not 0 <= channel < field.shape[-1]
                           for channel in channels_for_cellpose)):
                raise ValueError(tr("Calibrated V2 segmentation requires cell objects and explicit unique physical channels with valid model positions."))
            psf_session.validate_calibrated_shape(field.shape)
            field_values.append(field.size)
            del field
        for start in range(0, len(stacks), batch_fields):
            if sum(field_values[start:start + batch_fields]) > _CALIBRATED_BATCH_VALUES:
                raise ValueError(tr("Calibrated segmentation exceeds the supported aggregate batch-pixel budget; reduce the field batch size."))

    from .utils import _resolve_cellpose_pretrained, _validated_intensity_bounds

    intensity_settings = postprocess_settings or {}
    minimum, maximum = _validated_intensity_bounds(
        intensity_settings.get(f"{object_type}_min_intensity", 0),
        intensity_settings.get(f"{object_type}_max_intensity", 0),
    )
    from .qt.mask_engine import filters_need_intensity, settings_filters
    filter_by_raw_intensity = (minimum > 0 or maximum > 0 or filters_need_intensity(
        settings_filters(intensity_settings, object_type)))

    scratch = Path(npz_dir) if npz_dir else stacks[0].path.parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        from .object import _prefixed_model_route
    except ImportError as e:
        raise RuntimeError(
            "cellpose is required for v2 mask streaming: spacr.object, "
            f"which routes a model setting to its backend, could not be "
            f"imported ({e})"
        ) from e
    route = _prefixed_model_route(model_name, postprocess_settings)
    if route is not None:
        from . import _segmentation_backends
        backend, backend_masks = route
        model = _segmentation_backends._load_backend(
            backend, model_name=model_name, object_type=object_type)
        backend_settings = _backend_mask_settings(
            postprocess_settings, object_type, diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold)
    else:
        try:
            from cellpose import models as cp_models   # type: ignore
        except Exception as e:
            raise RuntimeError(
                "cellpose is required for v2 mask streaming"
            ) from e

        from .accelerator import cellpose_kwargs
        pretrained = _resolve_cellpose_pretrained(
            model_name, object_type=object_type)
        model = cp_models.CellposeModel(
            pretrained_model=pretrained,
            **cellpose_kwargs(),
        )

        _record_cellpose_hash(model, model_name)

    for batch_start in range(0, len(stacks), batch_fields):
        batch = stacks[batch_start:batch_start + batch_fields]
        with Timer(
            f"v2.batch[{batch_start}:{batch_start + len(batch)}] "
            f"load", logger="spacr.pipeline_v2",
        ):
            loaded = [(np.array(np.load(s.path, mmap_mode="r", allow_pickle=False), copy=True)
                       if calibrated else _as_hwc(np.load(s.path))) for s in batch]

        npz_path = scratch / f"batch_{batch_start:04d}.npz"
        np.savez_compressed(
            npz_path,
            **{s.field_id: arr for s, arr in zip(batch, loaded)},
        )

        selected_images: List[np.ndarray] = []
        calibrated_results = []
        raw_intensity_per_field: List[np.ndarray] = []
        for sf, arr in zip(batch, loaded):
            indices = _cellpose_channel_indices(
                channels_for_cellpose, arr.shape[-1])
            if filter_by_raw_intensity:
                raw_intensity_per_field.append(arr[..., indices[0]])
            if calibrated:
                result = psf_session.prepare_calibrated(arr, sf.field_id)
                calibrated_results.append(result)
                source = np.maximum(result.image, 0)
            else:
                source = (psf_session.unmix(arr) if psf_session is not None
                          else arr)
            selected = source[..., list(indices)]
            if illumination_session is not None:
                from .measure_hooks import PreprocessingContext
                context = PreprocessingContext(
                    file_name=sf.path.name,
                    channels=indices,
                    settings=postprocess_settings or {},
                )
                selected = illumination_session.correct(
                    sf.field_id, selected, context)
            if psf_session is not None and not calibrated:
                selected = psf_session.correct(selected)
            selected_images.append(selected)

        if postprocess_settings is not None:
            from .io import _normalize_img_batch

            selected_images = [
                image[..., np.newaxis] if image.ndim == 2 else image
                for image in selected_images
            ]
            max_height = max(image.shape[0] for image in selected_images)
            max_width = max(image.shape[1] for image in selected_images)
            if (calibrated and len(selected_images) * max_height * max_width
                    * selected_images[0].shape[-1] > _CALIBRATED_BATCH_VALUES):
                from .qt.i18n import tr

                raise ValueError(tr("Calibrated segmentation exceeds the supported padded batch-pixel budget; reduce the field batch size."))
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
            from .object_roles import SEGMENTED_ROLES
            for role in SEGMENTED_ROLES:
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
            if route is not None:
                masks, _flows = backend_masks(
                    model, cellpose_images, backend_settings, object_type,
                    min_size=int(min_size),
                    default_diameter=backend_settings['_default_diameter'],
                    batch_size=max(8, len(cellpose_images)))
                out = (masks,)
            else:
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
                _calibrated_label_array(mask) if calibrated
                else np.asarray(mask).astype(np.uint16) for mask in masks
            ]
            if calibrated and len(masks_per_field) != len(batch):
                from .qt.i18n import tr

                raise ValueError(tr("Calibrated segmentation must return one label image for every field."))

        if postprocess_settings is not None:
            from .object import merge_split_filter_masks
            filter_images = intensity_per_field
            if filter_by_raw_intensity:
                filter_images = [
                    np.pad(raw, (
                        (0, image.shape[0] - raw.shape[0]),
                        (0, image.shape[1] - raw.shape[1]),
                    ))
                    for raw, image in zip(raw_intensity_per_field,
                                          cellpose_images)
                ]
            masks_per_field = list(merge_split_filter_masks(
                masks=masks_per_field,
                intensity_images=filter_images,
                settings=postprocess_settings,
                object_type=object_type,
                batch_filenames=[stack.path.name for stack in batch],
            ))

        for index, (sf, arr, mask) in enumerate(zip(batch, loaded, masks_per_field)):
            if calibrated:
                result = calibrated_results[index]
                shape = result.image.shape[:2]
                labels = _calibrated_label_array(mask)
                if labels.shape != cellpose_images[index].shape[:2]:
                    from .qt.i18n import tr

                    raise ValueError(tr("Calibrated segmentation labels must match the image passed to the model."))
                labels = labels[:shape[0], :shape[1]]
                mask = psf_session.restore_calibrated(labels, result, sf.field_id)
                mask = _calibrated_label_array(mask)
                display = psf_session.calibrated_display(arr, result)
                combined = np.concatenate([display, mask[..., None]], axis=-1).astype(np.float32)
            else:
                mask = np.asarray(mask)[:arr.shape[0], :arr.shape[1]]
                combined = np.concatenate(
                    [arr, mask[..., None]], axis=-1
                ).astype(np.uint16)
            from .io import _save_array_atomic
            _save_array_atomic(str(sf.path), combined)
            sf.shape = combined.shape
            sf.channels = sf.channels + [mask_channel_name]
            if illumination_session is not None:
                illumination_session.mark_completed(sf.field_id)
            if psf_session is not None:
                if calibrated:
                    signed_path = psf_session.root / "signed" / sf.path.name
                    signed_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_array_atomic(str(signed_path), calibrated_results[index].image)
                    psf_session.mark_completed(sf.field_id, output_path=sf.path,
                                               signed_path=signed_path,
                                               mask_role='cell' if object_type == 'cell' else None)
                else:
                    psf_session.mark_completed(sf.field_id)

        if not keep_npz:
            try:
                npz_path.unlink()
            except Exception:
                pass

    if not keep_npz:
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass

    sidecar = stacks[0].path.parent / "channel_order.json"
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["mask_channels"] = [mask_channel_name]
        sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:
        LOG.warning("channel_order.json at %s was not updated with "
                    "mask_channels=%r (%s); readers of this stack will "
                    "not know which plane holds the mask.",
                    sidecar, mask_channel_name, exc)

    if illumination_session is not None:
        illumination_session.finish(sf.field_id for sf in stacks)
    if psf_session is not None:
        psf_session.finish(sf.field_id for sf in stacks)

    return stacks



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
    illumination_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the entire v2 pipeline against ``src``. Convenience wrapper.

    Equivalent to::

        mapper = FilenameMapper.discover(src, metadata_type, custom_regex)
        stacks = stream_originals_to_stack(src, mapper, channels, channel_names)
        stream_masks_from_stack(stacks, model_name, channels_for_cellpose,
                                diameter, batch_fields, keep_npz=keep_npz)

    :param src: plate folder holding the originals, scanned non-recursively.
        Normally the mapping CSV lands here and output goes to ``<src>/merged``.
        Calibrated runs use a fresh ``calibrated_segmentation`` output directory
        instead, preserving source metadata and existing merged arrays. Occupied
        names receive a numeric suffix; calibrated results are never resumed.
    :param channels: channel numbers as parsed from the filename
        (``C01`` gives 1), not C-axis positions. The default
        ``(0, 1, 2, 3)`` therefore misfits stock CellVoyager/Yokogawa
        names: channel 0 never exists, so plane 0 is all zeros, and
        ``C04`` is dropped. Pass ``(1, 2, 3, 4)`` for those layouts.
    :param channel_names: names recorded in ``channel_order.json`` and on
        each :class:`StackFile`. A length mismatch with ``channels`` trips
        a bare ``assert``, so it goes unchecked under ``python -O``.
    :param model_name: resolved by
        :func:`spacr.utils._resolve_cellpose_pretrained`, not passed as
        Cellpose's ``model_type``. On Cellpose 4 every legacy name
        (``"cyto"``, ``"nuclei"``, …) collapses to ``cpsam``, so only a
        fine-tuned checkpoint path actually changes the weights, and a
        path with no file behind it raises instead of falling back.
    :param channels_for_cellpose: despite the name this never reaches
        Cellpose's ``channels=`` argument; it selects C-axis indices from
        the assembled stack, taken modulo the channel count (7 with C=4
        becomes 3) and de-duplicated in order. The default ``(0, 0)``
        collapses to one plane, and an empty sequence falls back to plane 0.
        Calibrated runs require explicit in-range positions instead.
    :param diameter: forwarded to ``model.eval``; ``None`` leaves Cellpose
        to size objects itself. Unlike ``model_name`` it is still honoured
        on Cellpose 4, which rescales the image by ``30 / diameter``.
    :param batch_fields: fields loaded and segmented per batch, the
        memory-versus-speed dial. ``0`` raises ``ValueError``; a negative
        value segments nothing at all, yet ``channel_order.json`` is still
        rewritten to claim a mask channel that no stack received. Calibrated
        batches are additionally bounded to ``64 * 1024**2`` packed input or padded
        model-input values; lower this field count when a batch refuses.
    :param metadata_type: ``"cellvoyager"``, ``"yokogawa"``, ``"custom"``
        or ``"auto"``. Only ``"custom"`` is special-cased; every other
        unrecognised string quietly behaves as ``"auto"`` instead of
        raising.
    :param custom_regex: required when ``metadata_type="custom"``, else
        ``ValueError``. It must supply the named groups (``plateID``,
        ``wellID``, ``fieldID``, ``chanID``, ``timeID``, ``sliceID``); any
        group it omits silently defaults, so a regex without ``chanID``
        makes every image channel 1 and therefore its own single-plane
        field.
    :param keep_npz: the per-batch NPZ is compressed into
        ``merged/_scratch`` either way — this only decides whether it and
        the scratch folder survive, so ``False`` does not save the write.
    :param cellprob_threshold: forwarded to ``model.eval`` through
        ``float()``.
    :param flow_threshold: forwarded to ``model.eval`` through ``float()``.
    :param min_size: forwarded to ``model.eval`` through ``int()``;
        ``None`` raises ``TypeError`` rather than meaning "no minimum".
    :param resample: coerced with ``bool()``, so any non-empty string —
        ``"false"`` included — is True, while ``None`` is False.
    :param postprocess_settings: ``None`` hands the raw selected planes to
        Cellpose and skips post-processing entirely. Any dict, ``{}``
        included, switches on both :func:`spacr.io._normalize_img_batch`
        and :func:`spacr.object.merge_split_filter_masks`. The four
        ``*_channel`` role keys are rewritten on a copy (the caller's dict
        is left alone): all cleared to None, then ``object_type`` set to 0,
        plus ``nucleus_channel=1`` when ``object_type`` is ``"cell"`` and
        two or more planes were selected. An explicit ``unmix_calibration`` with
        ``unmix`` selects the organized 2D calibrated cell route: signed valid-ROI
        crops are retained separately, segmentation uses a nonnegative view and
        labels are restored to the original reference canvas with incomplete
        boundary objects removed. Unsupported preprocessing, custom/intensity/
        perimeter filters and quality exclusion refuse before inference.
    :param object_type: picks the weights during ``model_name``
        resolution, names the role given channel 0 in normalisation, and
        is passed to the mask post-processor. An unrecognised value does
        not raise — it adds a dead ``<value>_channel`` key and leaves every
        real role unset.
    :param illumination_settings: full Mask settings mapping. When it enables
        ``illumination_correction``, one model is prepared from the raw
        persisted stacks after pass 1 and its session corrects only the
        private Cellpose inputs in pass 2; omitted/off preserves the previous
        byte-level output contract.
    :returns: dict with ``mapper`` (:class:`FilenameMapper`), ``stacks``
        (list of :class:`StackFile`), and ``dst`` (Path to ``merged/``).
        Calibrated output also names its separate ``output_root``; its float32
        display intensities are aligned and nonquantitative, never Measure inputs.
        Signed valid-ROI float32 diagnostics live under ``output_root/signed``;
        its application record binds both artifacts and remains incomplete on error.
    :raises ValueError: from :meth:`FilenameMapper.discover` when ``src``
        holds no images, when ``metadata_type="custom"`` has no
        ``custom_regex``, or from ``batch_fields=0``.
    """
    src = Path(src)
    calibrated = bool((postprocess_settings or {}).get("unmix_calibration"))
    output_root = src
    if calibrated:
        from .qt.i18n import tr
        from .psf_pipeline import (
            _initialize_calibrated_output, _validate_calibrated_batch_settings)

        _validate_calibrated_batch_settings(postprocess_settings)
        if (illumination_settings or {}).get("illumination_correction"):
            raise ValueError(tr("Calibrated V2 segmentation requires cell objects without illumination correction."))
        if (object_type != "cell" or not channels or len(channels) > 16
                or len(set(channels)) != len(channels)
                or any(isinstance(ch, (bool, np.bool_)) or not isinstance(ch, (int, np.integer))
                       or ch < 0 for ch in channels)
                or any(int(ch) != ch or not 0 <= int(ch) < len(channels)
                       for ch in channels_for_cellpose)):
            raise ValueError(tr("Calibrated V2 segmentation requires cell objects and explicit unique physical channels with valid model positions."))
        for index in range(10000):
            candidate = src / ("calibrated_segmentation" if index == 0
                               else f"calibrated_segmentation_{index:04d}")
            try:
                candidate.mkdir()
                output_root = candidate
                break
            except FileExistsError:
                continue
        else:
            raise ValueError(tr("No unused calibrated segmentation output directory is available."))
        _initialize_calibrated_output(output_root, postprocess_settings)
    mapper = FilenameMapper.discover(src, metadata_type=metadata_type,
                                       custom_regex=custom_regex)
    if calibrated:
        from .qt.i18n import tr

        acquired_axes = {}
        for record in mapper.records:
            key = (record.plate, record.well, record.field)
            acquired_axes.setdefault(key, set()).add((record.time, record.z))
        if any(len(values) != 1 for values in acquired_axes.values()):
            raise ValueError(tr("Calibrated batch segmentation does not support time series or Z stacks."))
        stacks = stream_originals_to_stack(
            src, mapper, channels=channels, channel_names=channel_names,
            dst=output_root / "merged", metadata_root=output_root,
            calibrated=True)
    else:
        stacks = stream_originals_to_stack(
            src, mapper, channels=channels, channel_names=channel_names,
        )
    from .image_quality import screen_fields
    rejected_quality = set(screen_fields(output_root, postprocess_settings or {},
                                         [stack.path for stack in stacks], channels))
    stacks = [stack for stack in stacks if stack.path.name not in rejected_quality]
    if not stacks:
        result = {'mapper': mapper, 'stacks': [], 'dst': output_root / 'merged'}
        if calibrated:
            result['output_root'] = output_root
        return result
    illumination_session = None
    if (stacks and illumination_settings and
            illumination_settings.get('illumination_correction', False)):
        from .illumination import prepare_segmentation_illumination
        persisted_positions = _cellpose_channel_indices(
            channels_for_cellpose, len(stacks[0].channels))
        illumination_session = prepare_segmentation_illumination(
            illumination_settings,
            src=stacks[0].path.parent,
            channels=persisted_positions,
            pipeline_style='v2',
        )
    from .psf_pipeline import _prepare_segmentation_psf
    if calibrated:
        from functools import partial

        field_identities = {}
        for field_id, records in mapper.by_field().items():
            record = records[0]
            identity = {
                "plate": record.plate, "well": record.well, "field": record.field,
                "time": record.time, "z": record.z,
                "source_dataset": str(src.resolve())}
            field_identities[field_id] = identity
            field_identities[f"stack_{field_id}.npy"] = identity
        psf_session = _prepare_segmentation_psf(
            postprocess_settings, output_root,
            _cellpose_channel_indices(channels_for_cellpose, len(stacks[0].channels)),
            pipeline_style="v2", stack_dir=stacks[0].path.parent,
            load=partial(np.load, mmap_mode="r", allow_pickle=False),
            source_channels=tuple(int(channel) for channel in channels),
            field_identities=field_identities)
    else:
        psf_session = _prepare_segmentation_psf(
            postprocess_settings or {}, src,
            _cellpose_channel_indices(channels_for_cellpose, len(stacks[0].channels)),
            pipeline_style="v2", stack_dir=stacks[0].path.parent,
            load=lambda path: _as_hwc(np.load(path)))
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
        illumination_session=illumination_session,
        psf_session=psf_session,
    )
    result = {"mapper": mapper, "stacks": stacks,
              "dst": output_root / "merged"}
    if calibrated:
        result["output_root"] = output_root
    return result
