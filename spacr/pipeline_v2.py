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
        if getattr(mapper, '_partitioned', False):
            from .cancellation import checkpoint

            checkpoint()
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
    partitioned = calibrated and bool(psf_session.configuration.get("plate_partition"))
    if partitioned:
        from .cancellation import checkpoint

        checkpoint()
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

    if partitioned:
        checkpoint()
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
        if partitioned:
            checkpoint()
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
            if partitioned:
                checkpoint()
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
            if partitioned:
                checkpoint()
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
        if partitioned:
            checkpoint()
        psf_session.finish(sf.field_id for sf in stacks)

    return stacks



def _partition_plane_header(path):
    """Read reliable source bounds without decoding an unrelated target plane.

    :param path: original single-channel image path.
    :returns: acquired shape, numeric dtype and native file identity.
    """
    from .psf_pipeline import _partition_input_error

    path = Path(path)
    before = path.stat()
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile

        with tifffile.TiffFile(path) as image:
            if len(image.series) != 1:
                raise _partition_input_error()
            shape, dtype = image.series[0].shape, np.dtype(image.series[0].dtype)
    else:
        from PIL import Image, ImageMode

        with Image.open(path) as image:
            if len(image.getbands()) != 1 or getattr(image, 'n_frames', 1) != 1:
                raise _partition_input_error()
            shape = (image.height, image.width)
            dtype = np.dtype(ImageMode.getmode(image.mode).typestr)
    after = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if (identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(shape) != 2 or not all(shape) or dtype.kind not in 'uif' or dtype.itemsize > 8):
        raise _partition_input_error()
    return dict(shape=list(shape), dtype=dtype.str, identity=list(identity))


def _partition_packed_fingerprint(array, channels):
    """Identify canonical packed values, explicitly not serialized NPY file bytes.

    :param array: acquired channel-last control values.
    :param channels: physical channel order represented by the last axis.
    :returns: dtype, shape, channel order and C-order value hash.
    """
    import hashlib

    contiguous = np.ascontiguousarray(array)
    return dict(stage='canonical_channel_packed_control_values', shape=list(contiguous.shape),
                dtype=contiguous.dtype.str, channels=list(channels),
                values_sha256=hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest())


def _partition_read_control(records, channels, headers):
    """Read one actual control field after its header bounds have been admitted.

    :param records: original records for this exact field.
    :param channels: selected physical channels in persisted order.
    :param headers: immutable preflight source bounds and native identities.
    :returns: numeric finite acquired HWC values without changing the originals.
    """
    from .cancellation import checkpoint
    from .psf_pipeline import _partition_input_error

    by_channel = {record.channel: record for record in records}
    planes = []
    for channel in channels:
        checkpoint()
        record = by_channel[channel]
        expected = headers[record.original_path]
        if _partition_plane_header(record.original_path) != expected:
            raise _partition_input_error()
        plane = _read_plane(record.original_path, calibrated=True,
                            max_pixels=(16 * 1024 ** 2) // len(channels))
        if (list(plane.shape) != expected['shape'] or plane.dtype.str != expected['dtype']
                or not np.isfinite(plane).all()
                or _partition_plane_header(record.original_path) != expected):
            raise _partition_input_error()
        planes.append(plane)
    return np.stack(planes, axis=-1)


def _preflight_plate_partitions(src, mapper, settings, options):
    """Admit every plate's headers, optics and sampled controls before any output.

    Target pixel values not read here are still validated by the ordinary child
    route before its model is constructed. Control arrays are released per plate.

    :param src: original source folder, never replaced by a synthetic child source.
    :param mapper: one discovery containing the original global field identities.
    :param settings: explicit partition, controls and optical-map settings.
    :param options: V2 channel, model-position and batching options.
    :returns: bounded independently captured plate plans.
    """
    import io
    from .cancellation import checkpoint
    from .psf_pipeline import (
        _partition_input_error, _capture_plate_calibrations, _plate_unmix_controls,
        _estimate_bleed_through, _UNMIX_MAX_FIELDS, _UNMIX_CALIBRATED_CONTROL_VALUES,
        _validate_calibrated_batch_settings, _cell_record_hash,
    )
    from .schema import parse_well

    channels = tuple(options['channels'])
    model_positions = tuple(options['channels_for_cellpose'])
    batch_fields = options['batch_fields']
    if (not channels or len(channels) > 16 or len(set(channels)) != len(channels)
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                   or value < 0 for value in channels)
            or not model_positions or any(isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not 0 <= value < len(channels)
                for value in model_positions)
            or type(batch_fields) is not int or batch_fields <= 0
            or options['object_type'] != 'cell'
            or (options['channel_names'] is not None and len(options['channel_names']) != len(channels))
            or (options['illumination_settings'] or {}).get('illumination_correction')):
        raise _partition_input_error()
    channels = tuple(int(value) for value in channels)
    options['channels'] = channels
    percentile = float(settings.get('unmix_background_percentile', 5.0))
    if not 0 <= percentile < 100:
        raise _partition_input_error()
    from .image_quality import quality_policy

    quality_policy(settings)
    _validate_calibrated_batch_settings(settings)
    grouped = mapper.by_field()
    if not 1 <= len(grouped) <= 8192:
        raise _partition_input_error()
    plate_records, contexts = {}, {}
    for record in mapper.records:
        plate_records.setdefault(record.plate, []).append(record)
        contexts.setdefault(record.plate, set()).add((record.time, record.z))
    if any(len(values) != 1 for values in contexts.values()):
        raise _partition_input_error()
    contexts = {plate: next(iter(values)) for plate, values in contexts.items()}
    optics = _capture_plate_calibrations(settings, src, contexts, channels)
    plans = []
    retained_bytes = sum(len(value['raw']) for value in optics.values())
    header_bytes = retained_bytes
    for number, plate in enumerate(sorted(plate_records), 1):
        checkpoint()
        child_mapper = FilenameMapper(plate_records[plate], mapper.metadata_type, mapper.regex)
        child_mapper._partitioned = True
        identities = {}
        headers, field_shapes, regions = {}, {}, {}
        calibration = optics[plate]['calibration']
        for field_id, records in child_mapper.by_field().items():
            checkpoint()
            record = records[0]
            identity = dict(plate=record.plate, well=record.well, field=record.field,
                            time=record.time, z=record.z, source_dataset=str(src.resolve()))
            if any((row.plate, row.well, row.field, row.time, row.z) !=
                   (record.plate, record.well, record.field, record.time, record.z) for row in records):
                raise _partition_input_error()
            identities[field_id] = identity
            identities[f'stack_{field_id}.npy'] = identity
            by_channel = {row.channel: row for row in records}
            if len(by_channel) != len(records) or any(channel not in by_channel for channel in channels):
                raise _partition_input_error()
            selected_headers = []
            for channel in channels:
                path = by_channel[channel].original_path
                headers[path] = _partition_plane_header(path)
                selected_headers.append(headers[path])
            shape = selected_headers[0]['shape']
            if (any(item['shape'] != shape for item in selected_headers)
                    or int(np.prod(shape, dtype=object)) * len(channels) > 16 * 1024 ** 2):
                raise _partition_input_error()
            field_shapes[field_id] = tuple(shape) + (len(channels),)
            regions[field_id] = calibration.valid_region(field_shapes[field_id])
            header_bytes += len(json.dumps(dict(
                identities={field_id: identity, f'stack_{field_id}.npy': identity},
                headers={by_channel[channel].original_path: headers[by_channel[channel].original_path]
                         for channel in channels}), allow_nan=False).encode())
            if header_bytes > 16 * 1024 ** 2:
                raise _partition_input_error()
        controls, selection = _plate_unmix_controls(settings, channels, identities)
        ordered = list(child_mapper.by_field())
        selected_count = len(_cellpose_channel_indices(model_positions, len(channels)))
        for start in range(0, len(ordered), batch_fields):
            batch = ordered[start:start + batch_fields]
            if sum(int(np.prod(field_shapes[field_id], dtype=object)) for field_id in batch) > _CALIBRATED_BATCH_VALUES:
                raise _partition_input_error()
            height = max(regions[field_id][0].stop - regions[field_id][0].start for field_id in batch)
            width = max(regions[field_id][1].stop - regions[field_id][1].start for field_id in batch)
            if len(batch) * height * width * selected_count > _CALIBRATED_BATCH_VALUES:
                raise _partition_input_error()
        captured = optics[plate]
        metadata = dict(child_name=f'plate_{number:06d}', plate_id=plate,
            time_id=contexts[plate][0], z_id=contexts[plate][1],
            calibration_source_sha256=captured['sha256'],
            calibration_identity_sha256=_cell_record_hash({key: value for key, value
                in calibration.provenance().items() if key != 'source'}),
            pixel_size_um=list(captured['entry']['pixel_size_um']),
            control_map_sha256=selection['controls_map_sha256'],
            acquisition_compatibility_verified=False)
        plans.append(dict(mapper=child_mapper, identities=identities, headers=headers,
                          metadata=metadata, optics=captured, field_shapes=field_shapes,
                          controls=controls))
        summary = [dict(metadata=plan['metadata'], headers=plan['headers'], identities=plan['identities'])
                   for plan in plans]
        if retained_bytes + len(json.dumps(summary, allow_nan=False).encode()) > 16 * 1024 ** 2:
            raise _partition_input_error()
    for plan in plans:
        checkpoint()
        child_mapper, identities, headers = plan['mapper'], plan['identities'], plan['headers']
        field_shapes, controls = plan['field_shapes'], plan['controls']
        calibration = plan['optics']['calibration']
        samples, selected_fields, fields_by_dye, values, input_bytes = {}, {}, {}, 0, 0
        for dye, wells in controls.items():
            wanted = {parse_well(well, strict=True) for well in wells}
            chosen = [field_id for field_id in sorted(child_mapper.by_field())
                      if parse_well(identities[field_id]['well'], strict=True) in wanted][:_UNMIX_MAX_FIELDS]
            if not chosen:
                raise _partition_input_error()
            selected_fields[str(dye)] = ['stack_' + field_id for field_id in chosen]
            fields_by_dye[dye] = []
            for field_id in chosen:
                checkpoint()
                shape = field_shapes[field_id]
                values += int(np.prod(shape, dtype=object))
                if values > _UNMIX_CALIBRATED_CONTROL_VALUES:
                    raise _partition_input_error()
                field = _partition_read_control(child_mapper.by_field()[field_id], channels, headers)
                header = io.BytesIO()
                np.lib.format.write_array_header_1_0(header, {
                    'descr': np.lib.format.dtype_to_descr(field.dtype), 'fortran_order': False,
                    'shape': field.shape})
                size = field.nbytes + len(header.getvalue())
                input_bytes += size
                if size > 128 * 1024 ** 2 + 16384 or input_bytes > 512 * 1024 ** 2:
                    raise _partition_input_error()
                samples[field_id] = _partition_packed_fingerprint(field, channels)
                matched, unused_region = calibration.apply(field)
                fields_by_dye[dye].append(matched)
                del field
        matrix = _estimate_bleed_through(fields_by_dye, channels)
        del fields_by_dye
        plan['metadata'].update(control_values=samples, control_fields=selected_fields,
                                matrix=matrix.tolist())
        summary = [dict(metadata=item['metadata'], headers=item['headers'], identities=item['identities'])
                   for item in plans]
        if retained_bytes + len(json.dumps(summary, allow_nan=False).encode()) > 16 * 1024 ** 2:
            raise _partition_input_error()
    return plans


def _record_partition_qc(result, child, status, error=None):
    """Atomically record child or aggregate QC without changing segmentation facts.

    :param result: trusted plural result returned by this partition producer.
    :param child: one returned child entry, or None for aggregate QC state.
    :param status: pending, running, complete, off, skipped, failed or cancelled.
    :param error: optional original failure whose type is recorded without its message.
    """
    from .psf_pipeline import _write_json, _partition_input_error

    if status not in {'pending', 'running', 'complete', 'off', 'skipped', 'failed', 'cancelled'}:
        raise _partition_input_error()
    path = Path(result['manifest_path'])
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('artifact') != 'spacr_calibrated_plate_batch' or record.get('version') != 1:
        raise _partition_input_error()
    if child is not None:
        matches = [entry for entry in record['children'] if entry['child_name'] == child['child_name']]
        if len(matches) != 1 or path.parent / matches[0]['path'] != Path(child['output_root']):
            raise _partition_input_error()
        matches[0]['qc_status'] = status
        if error is not None:
            matches[0]['qc_error_type'] = type(error).__name__
    else:
        record['qc_status'] = status
        complete = (status in {'complete', 'off'} and record['segmentation_complete']
                    and all(entry['qc_status'] in {'complete', 'off'} for entry in record['children']))
        record['aggregate_complete'] = record['complete'] = complete
        record['status'] = 'complete' if complete else ('partial' if status in {'complete', 'off', 'skipped'} else status)
        if error is not None:
            record['qc_error_type'] = type(error).__name__
    _write_json(path, record)


def _run_partitioned_v2(src, settings, options):
    """Execute preflighted independent child datasets under a collection-only root.

    :param src: original source folder whose image files and merged data remain untouched.
    :param settings: enabled partition settings with explicit per-plate optical assignments.
    :param options: unchanged V2 processing arguments used separately for each child.
    :returns: plural child results and a durable collection manifest path.
    """
    import copy
    import hashlib
    from functools import partial
    from .qt.i18n import tr
    from .cancellation import checkpoint, PipelineCancelled
    from .psf_pipeline import (
        _partition_input_error, _write_json, _initialize_calibrated_output,
        _prepare_segmentation_psf, _cell_record_hash,
    )

    mapper = FilenameMapper.discover(src, metadata_type=options['metadata_type'],
                                     custom_regex=options['custom_regex'])
    plans = _preflight_plate_partitions(src, mapper, settings, options)
    checkpoint()
    for index in range(10000):
        root = src / ('calibrated_plate_batch' if index == 0 else f'calibrated_plate_batch_{index:04d}')
        try:
            root.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise _partition_input_error()
    manifest_path = root / 'plate_batch.json'
    manifest = dict(artifact='spacr_calibrated_plate_batch', version=1,
        stage='calibrated_segmentation', source_dataset=str(src.resolve()),
        physical_channel_order=[int(value) for value in options['channels']],
        complete=False, segmentation_complete=False, aggregate_complete=False,
        status='running', qc_status='pending', children=[])
    for plan in plans:
        entry = dict(plan['metadata'], path=plan['metadata']['child_name'],
                     status='pending', qc_status='pending', record_sha256=None)
        manifest['children'].append(entry)
    _write_json(root / 'psf/segmentation_application.json', dict(
        configuration=dict(version=1, pipeline_style='v2',
            coordinate_system='calibrated_reference_canvas', collection_only=True,
            measurement_intensity_source='collection only; select an individual calibrated child dataset'),
        complete=False, fields=[], archives={}))
    _write_json(manifest_path, manifest)
    print(tr("Calibrated plate collection created at {path}; each child is an independent signed-cell dataset, and the collection itself is not a Measure input.", path=str(root)))
    children = []
    active = None
    try:
        for plan, entry in zip(plans, manifest['children']):
            checkpoint()
            active = entry
            entry['status'] = 'running'
            _write_json(manifest_path, manifest)
            child_root = root / entry['path']
            child_settings = copy.deepcopy(settings)
            child_settings.update(unmix_partition_plates=False,
                unmix_calibration=str(child_root / 'psf/input_calibration.json'),
                unmix_pixel_size_um=list(entry['pixel_size_um']), src=str(child_root))
            _initialize_calibrated_output(child_root, child_settings)
            optics_path = child_root / 'psf/input_calibration.json'
            with optics_path.open('xb') as stream:
                stream.write(plan['optics']['raw'])
                stream.flush()
                import os
                os.fsync(stream.fileno())
            checkpoint()
            stacks = stream_originals_to_stack(src, plan['mapper'],
                channels=options['channels'], channel_names=options['channel_names'],
                dst=child_root / 'merged', metadata_root=child_root, calibrated=True)
            for field_id, expected in entry['control_values'].items():
                checkpoint()
                field = np.load(child_root / 'merged' / f'stack_{field_id}.npy',
                                mmap_mode='r', allow_pickle=False)
                try:
                    if _partition_packed_fingerprint(field, options['channels']) != expected:
                        raise _partition_input_error()
                finally:
                    field._mmap.close()
            from .image_quality import screen_fields

            rejected = screen_fields(child_root, child_settings,
                                     [stack.path for stack in stacks], options['channels'])
            if rejected:
                raise _partition_input_error()
            session = _prepare_segmentation_psf(child_settings, child_root,
                _cellpose_channel_indices(options['channels_for_cellpose'], len(options['channels'])),
                pipeline_style='v2', stack_dir=child_root / 'merged',
                load=partial(np.load, mmap_mode='r', allow_pickle=False),
                source_channels=tuple(int(value) for value in options['channels']),
                field_identities=plan['identities'])
            actual_fields = {str(dye): list(stems) for dye, stems in session.unmixing.fields.items()}
            actual_samples = json.loads(session.unmixing.control_selection)['sampled_fields']
            if (actual_fields != entry['control_fields']
                    or set(actual_samples) != {'stack_' + field_id for field_id in entry['control_values']}
                    or not np.array_equal(np.asarray(session.unmixing.matrix), np.asarray(entry['matrix']))):
                raise _partition_input_error()
            actual_optics = session.unmixing.calibration.provenance()
            if _cell_record_hash({key: value for key, value in actual_optics.items() if key != 'source'}) != entry['calibration_identity_sha256']:
                raise _partition_input_error()
            session.configuration['plate_partition'] = dict(
                collection_manifest='../plate_batch.json', child_name=entry['child_name'],
                acquisition_compatibility=plan['optics']['entry']['acquisition_compatibility'],
                acquisition_compatibility_verified=False,
                control_value_preflight=entry['control_values'],
                calibration_source_sha256=entry['calibration_source_sha256'])
            session._write(False)
            checkpoint()
            stream_masks_from_stack(stacks, model_name=options['model_name'],
                channels_for_cellpose=options['channels_for_cellpose'],
                diameter=options['diameter'], batch_fields=options['batch_fields'],
                keep_npz=options['keep_npz'], cellprob_threshold=options['cellprob_threshold'],
                flow_threshold=options['flow_threshold'], min_size=options['min_size'],
                resample=options['resample'], postprocess_settings=child_settings,
                object_type='cell', psf_session=session)
            record_path = child_root / 'psf/segmentation_application.json'
            with record_path.open('rb') as stream:
                raw_record = stream.read(16 * 1024 ** 2 + 1)
            if len(raw_record) > 16 * 1024 ** 2:
                raise _partition_input_error()
            entry['record_sha256'] = hashlib.sha256(raw_record).hexdigest()
            entry['status'] = 'complete'
            children.append(dict(child_name=entry['child_name'], plate_id=entry['plate_id'],
                time_id=entry['time_id'], z_id=entry['z_id'], output_root=child_root,
                mapper=plan['mapper'], stacks=stacks, dst=child_root / 'merged',
                settings=copy.deepcopy(child_settings)))
            manifest['segmentation_complete'] = all(row['status'] == 'complete' for row in manifest['children'])
            _write_json(manifest_path, manifest)
            print(tr("Calibrated child outputs saved to {path}; display planes are nonquantitative, and quantitative measurements use the signed-cell API.", path=str(child_root)))
        checkpoint()
        manifest['segmentation_complete'] = True
        manifest['status'] = 'segmented'
        _write_json(manifest_path, manifest)
    except BaseException as error:
        status = 'cancelled' if isinstance(error, PipelineCancelled) else 'failed'
        if active is not None and active['status'] != 'complete':
            active['status'] = status
            active['error_type'] = type(error).__name__
        manifest['status'] = status
        manifest['error_type'] = type(error).__name__
        try:
            _write_json(manifest_path, manifest)
        except BaseException as status_error:
            note = getattr(error, 'add_note', None)
            if callable(note):
                try:
                    note(tr("Collection status could not be saved: {error_type}", error_type=type(status_error).__name__))
                except BaseException:
                    pass
        raise
    return dict(partitioned=True, plate_results=children,
                output_roots=[child['output_root'] for child in children], manifest_path=manifest_path)


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
        Optional ``unmix_per_plate`` selects complete physical-channel controls
        from a JSON master map for one exact source plate/time/plane; mixed
        sources and missing assignments refuse before output creation.
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
    A nonempty ``unmix_matrix_file`` in ``postprocess_settings`` supplies a
    version-1 JSON matrix instead of estimating one. This requires calibrated
    organized 2D cell V2 processing, no control wells, and ``unmix_per_plate=False``.
    Its physical channels must match the packed order exactly; no transpose,
    channel reordering or diagonal renormalization is inferred. ``matrix[i][j]``
    is the camera reading in row-channel ``i`` per unit of the dye measured in
    its own column-channel ``j``. The diagonal is exactly one; outputs retain
    camera-linear units and signed residuals, not calibrated concentrations.
    For physical order ``[3, 1]``, the file has exactly these keys::

        {
          "artifact": "spacr_spectral_matrix",
          "version": 1,
          "source_context": {"plate": "plate1", "time": "", "z": ""},
          "channels": [3, 1],
          "orientation": "rows_readings_columns_dyes",
          "matrix": [[1, 0.4], [0.7, 1]],
          "background_percentile": 0,
          "background_operator": "per_field_percentile_subtract_solve_restore",
          "calibration_source_sha256": "<SHA256 of the exact calibration-file bytes>",
          "acquisition_compatibility": "user_asserts_same_channel_gains_exposures_and_linear_response"
        }

    Replace the example calibration hash with 64 lowercase hexadecimal digits.
    Background percentile must equal ``unmix_background_percentile``. Plate,
    time and Z must equal every mapped field's exact acquisition IDs, including
    their string or integer types; an absent time or Z uses the empty string.
    This supplied-file route requires POSIX descriptor-relative access without
    symbolic links; unsupported platforms refuse before creating outputs.
    The acquisition-compatibility token is an unverified user assertion: unknown
    gains, exposures and response remain unknown. The file is captured before
    outputs and copied byte-for-byte into ``psf/spectral_matrix_input.json``.
    Signed-cell readers validate that copied bundle and never depend on the
    original external path. An empty path preserves control-estimation behavior.

    With ``unmix_partition_plates=True`` and ``unmix_per_plate=True``, one
    mixed-plate source produces independent calibrated child datasets. Supply
    ``unmix_plate_calibrations`` as a JSON string mapping every exact plate ID
    to ``calibration_file``, positive YX ``pixel_size_um``, ``source_context``
    containing exact ``plate``, ``time`` and ``z`` IDs, and
    ``acquisition_compatibility`` set to
    ``user_asserts_same_channel_gains_exposures_and_linear_response``.
    Relative optical paths resolve against the original source folder. Each
    plate has one time/Z context, complete physical-channel controls in the
    master map, and its own optical assignment and pixel spacing. The assertion
    concerns controls and targets within that plate, remains unverified, and
    does not assert equal acquisition settings between plates. Global optical
    settings and supplied matrix files cannot be combined with partitioning.

    Before creating outputs, partitioning captures optical bytes, admits all
    source headers and batch bounds, and estimates each plate's sampled-control
    matrix sequentially. Unread target values are validated later by the child
    route before model construction. Canonical control-value fingerprints are
    compared after assembly; actual NPY file hashes remain the child record's
    authority. Limits are 8192 fields, 16 physical channels and 16 MiB retained
    metadata, with the ordinary per-field, control and batch budgets unchanged.
    Supported source formats are unchanged; QC supports report or off only.
    Directory-relative, no-follow and nonblocking descriptor access is required;
    unsupported hosts refuse before creating outputs.

    A fresh ``calibrated_plate_batch`` collection directory contains a
    ``plate_batch.json`` manifest and numbered ``plate_000001`` child datasets.
    Existing names receive a suffix. The collection is never a measurable
    dataset; each child retains the ordinary version-1 calibrated signed-cell
    contract and original global field/source identities. Original files and
    existing merged data remain untouched. Cancellation or failure preserves
    completed children and rethrows the original error. Segmentation completion
    is distinct from aggregate completion, which also requires completed or
    disabled QC. The opt-in return has ``partitioned=True``, ``plate_results``
    with independent resolved ``settings`` and ``output_root`` per child,
    ``output_roots``, and ``manifest_path``; it has no singular output fallback.
    With partitioning disabled, the existing return and processing are unchanged.

    :raises ValueError: from :meth:`FilenameMapper.discover` when ``src``
        holds no images, when ``metadata_type="custom"`` has no
        ``custom_regex``, or from ``batch_fields=0``.
    """
    src = Path(src)
    from .psf_pipeline import _validate_unmix_partition_scope

    if _validate_unmix_partition_scope(postprocess_settings or {}, allow_partition=True):
        return _run_partitioned_v2(src, postprocess_settings, dict(
            channels=channels,
            channel_names=channel_names,
            model_name=model_name,
            channels_for_cellpose=channels_for_cellpose,
            diameter=diameter,
            batch_fields=batch_fields,
            metadata_type=metadata_type,
            custom_regex=custom_regex,
            keep_npz=keep_npz,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            resample=resample,
            object_type=object_type,
            illumination_settings=illumination_settings))
    calibrated = bool((postprocess_settings or {}).get("unmix_calibration"))
    from .psf_pipeline import _validate_unmix_plate_scope

    per_plate = _validate_unmix_plate_scope(
        postprocess_settings or {}, allow_per_plate=calibrated)
    from .psf_pipeline import _validate_unmix_matrix_scope

    supplied_matrix = _validate_unmix_matrix_scope(
        postprocess_settings or {}, allow_matrix=calibrated)
    prepared_unmix = None
    prepared_mapper = None
    if per_plate or supplied_matrix:
        from .psf_pipeline import _plate_unmix_controls

        prepared_mapper = FilenameMapper.discover(
            src, metadata_type=metadata_type, custom_regex=custom_regex)
        identities = {record.original_path: dict(
            plate=record.plate, well=record.well, field=record.field,
            time=record.time, z=record.z, source_dataset=str(src.resolve()))
            for record in prepared_mapper.records}
        if per_plate:
            _plate_unmix_controls(postprocess_settings, channels, identities)
        if supplied_matrix:
            from .psf_pipeline import _prepare_supplied_matrix

            prepared_unmix = _prepare_supplied_matrix(postprocess_settings, channels, identities)
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
    mapper = (prepared_mapper if prepared_mapper is not None else
              FilenameMapper.discover(src, metadata_type=metadata_type,
                                      custom_regex=custom_regex))
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
            field_identities=field_identities, prepared_unmix=prepared_unmix)
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
